"""Unified factory for all baselines1 models.

All models return logits in canonical [B, N, num_classes] layout via OutputNormalizer.
"""
import torch
import torch.nn as nn

from .pointnet2 import PointNet2Seg
from .pointnext import PointNeXtSeg
from .kpconv import AnisotropicKPConvSeg, KPConvSeg
from .dgcnn import DGCNNSeg
from .proposed import build_proposed_model


class OutputNormalizer(nn.Module):
    """Wrap a model to guarantee [B, N, C] output regardless of internal layout."""

    def __init__(self, model: nn.Module, num_classes: int):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, pos: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        logits = self.model(pos, feats)
        if logits.dim() != 3:
            return logits
        N = pos.shape[1]
        # Already [B, N, C]
        if logits.shape[1] == N and logits.shape[2] == self.num_classes:
            return logits
        # [B, C, N] -> [B, N, C]
        if logits.shape[1] == self.num_classes and logits.shape[2] == N:
            return logits.permute(0, 2, 1).contiguous()
        # Fallback: assume class is the smaller axis
        if logits.shape[1] < logits.shape[2]:
            return logits.permute(0, 2, 1).contiguous()
        return logits


def build_baseline_model(name: str, num_classes: int = 4,
                         in_feat: int = 24, pos_dim: int = 2) -> nn.Module:
    name = name.lower()
    if name in ("pointnet2", "pointnet++", "pointnet_pp"):
        return OutputNormalizer(
            PointNet2Seg(num_classes=num_classes, in_feat=in_feat, pos_dim=pos_dim), num_classes)
    if name in ("pointnext", "pointnet_next"):
        return OutputNormalizer(
            PointNeXtSeg(num_classes=num_classes, in_feat=in_feat, pos_dim=pos_dim), num_classes)
    if name in ("kpconv",):
        return OutputNormalizer(
            KPConvSeg(num_classes=num_classes, in_feat=in_feat, pos_dim=pos_dim), num_classes)
    if name in ("anisotropic_kpconv", "anisokpconv", "anisotropic-kpconv"):
        return OutputNormalizer(
            AnisotropicKPConvSeg(num_classes=num_classes, in_feat=in_feat,
                                 pos_dim=pos_dim), num_classes)
    if name in ("dgcnn",):
        return OutputNormalizer(
            DGCNNSeg(num_classes=num_classes, in_feat=in_feat, pos_dim=pos_dim), num_classes)
    raise ValueError(f"Unknown baseline model: {name}")


def build_proposed_wrapped(source_root: str, variant: str = "full",
                            num_classes: int = 4, dropout: float = 0.3) -> nn.Module:
    """Load proposed model and wrap to guarantee [B, N, C] output."""
    model = build_proposed_model(source_root, variant=variant,
                                 num_classes=num_classes, dropout=dropout)
    return OutputNormalizer(model, num_classes)
