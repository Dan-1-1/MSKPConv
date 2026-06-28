"""Loader for the proposed model (PhysicsKPConvNet from /root/autodl-tmp/KPConv_new_GPU/model_1.py)
plus the ablation variants used in Table 2.

baselines1 keeps the proposed-model source untouched and applies *runtime*
substitutions (swap modules, wrap blocks) to obtain each ablated variant.
All variants are trained with standard cross-entropy (see ../losses.py).

Variants
--------
- ``full``                    : full PhysicsKPConvNet
- ``without_physical_stats``  : zero out the 24-D statistical features at
                                 input time, position only
- ``isotropic_kpconv``        : replace every FixedStripKPConv with an
                                 isotropic-circle KPConv (same parameter count)
- ``single_scale_strip``      : use only the first scale of every
                                 MultiScaleFixedStripKPBlock
- ``without_track_transformer``: bypass the distance-biased PointTrackTransformer
- ``without_gated_decoder``   : replace decoder ResidualChannelGate modules
                                 with identity mappings
"""
import importlib.util
import math
import os
import sys
from contextlib import contextmanager

import torch
import torch.nn as nn


@contextmanager
def _source_path(path):
    inserted = False
    if path not in sys.path:
        sys.path.insert(0, path)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


# ---------------------------------------------------------------------------
# Local utility KPConv-on-circle (used by the ``isotropic_kpconv`` ablation)
# ---------------------------------------------------------------------------
class _IsotropicCircleKPConv(nn.Module):
    """Isotropic kernel-point conv used to replace FixedStripKPConv.

    Kernel points uniformly distributed on a circle (origin point + ring),
    matching the operator signature of FixedStripKPConv.
    """

    def __init__(self, in_channels, out_channels, num_kernel_points=16,
                 radius=3.0, sigma=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_k = num_kernel_points
        self.radius = float(radius)
        self.sigma = float(sigma) if sigma is not None else self.radius * 0.35
        angles = torch.linspace(0, 2 * math.pi, num_kernel_points + 1)[:-1]
        kp = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * self.radius
        kp[0] = 0.0  # one centre point
        self.register_buffer("kernel_pts", kp)
        self.weights = nn.Parameter(torch.empty(num_kernel_points, in_channels, out_channels))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, q_pts, s_pts, s_feats, neighbor_idx):
        # Mirrors the API of FixedStripKPConv: feats are channel-first [B,C,N]
        B, Nq, K = neighbor_idx.shape
        C = s_feats.shape[1]
        s_pts_flat = s_pts.reshape(-1, 2)
        s_feats_flat = s_feats.permute(0, 2, 1).reshape(-1, C)
        offsets = torch.arange(B, device=s_pts.device).view(B, 1, 1) * s_pts.shape[1]
        flat_idx = (neighbor_idx + offsets).reshape(-1)
        n_pts = s_pts_flat[flat_idx].reshape(B, Nq, K, 2)
        n_feats = s_feats_flat[flat_idx].reshape(B * Nq, K, C)
        diff = n_pts - q_pts.unsqueeze(2)
        dist = torch.cdist(
            diff.reshape(B * Nq, K, 2),
            self.kernel_pts.view(1, self.num_k, 2).expand(B * Nq, -1, -1),
        )
        influence = torch.clamp(1.0 - dist / self.sigma, min=0.0)
        w = self.weights.permute(1, 0, 2).reshape(C, -1)
        weighted = torch.matmul(n_feats, w).view(B * Nq, K, self.num_k, -1)
        out = (weighted * influence.unsqueeze(-1)).sum(dim=(1, 2))
        return out.reshape(B, Nq, -1).permute(0, 2, 1).contiguous()


class _IsotropicKPAdapter(nn.Module):
    def __init__(self, old_strip_kpconv: nn.Module):
        super().__init__()
        self.iso = _IsotropicCircleKPConv(
            old_strip_kpconv.in_channels,
            old_strip_kpconv.out_channels,
            num_kernel_points=getattr(old_strip_kpconv, "num_k", 16),
            radius=float(getattr(old_strip_kpconv, "radius", 3.0)),
            sigma=float(getattr(old_strip_kpconv, "sigma", 1.0)),
        )

    def forward(self, q_pts, s_pts, s_feats, neighbor_idx):
        return self.iso(q_pts, s_pts, s_feats, neighbor_idx)


def _replace_strip_with_isotropic(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if child.__class__.__name__ == "FixedStripKPConv":
            setattr(module, name, _IsotropicKPAdapter(child))
        else:
            _replace_strip_with_isotropic(child)


# ---------------------------------------------------------------------------
# Single-scale strip wrapper (uses only the first KPConv branch of each block)
# ---------------------------------------------------------------------------
def _gather_features_chw(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    idx_exp = idx.unsqueeze(1).expand(-1, x.shape[1], -1)
    return torch.gather(x, 2, idx_exp)


class _SingleScaleStripBlock(nn.Module):
    """Replace MultiScaleFixedStripKPBlock with its smallest scale only."""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.downsample = block.downsample
        self.unary1 = block.unary1
        self.kpconv = block.kpconv_r1
        self.proj = nn.Sequential(
            nn.Conv1d(block.kpconv_r1.out_channels, block.fusion[0].out_channels, 1, bias=False),
            nn.BatchNorm1d(block.fusion[0].out_channels),
        )
        self.shortcut = block.shortcut
        self.act = block.act

    def forward(self, x, q_pos, s_pos, neighbor_idx_list, q_idx=None):
        if self.downsample:
            if q_idx is not None:
                identity_src = _gather_features_chw(x, q_idx)
            else:
                identity_src = nn.functional.interpolate(x, size=q_pos.shape[1], mode="nearest")
            identity = self.shortcut(identity_src)
        else:
            identity = self.shortcut(x)
        mid = self.unary1(x)
        idx = neighbor_idx_list[0]
        out = self.proj(self.kpconv(q_pos, s_pos, mid, idx))
        return self.act(out + identity)


def _replace_multiscale_with_single_scale(model: nn.Module) -> None:
    for name in ("enc1", "enc2", "enc3", "enc4"):
        if hasattr(model, name):
            setattr(model, name, _SingleScaleStripBlock(getattr(model, name)))


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------
class _IdentityTrackTransformer(nn.Module):
    def forward(self, local_features, pos_raw):
        return local_features


def _replace_decoder_gates_with_identity(model: nn.Module) -> None:
    for name in ("skip_gate3", "skip_gate2", "skip_gate1"):
        if hasattr(model, name):
            setattr(model, name, nn.Identity())


class _PhysStatsMaskedModel(nn.Module):
    """Wrap a model so the 24-D statistical features are zeroed out at runtime."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pos, features):
        return self.model(pos, torch.zeros_like(features))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_proposed_model(source_root: str,
                         variant: str = "full",
                         num_classes: int = 4,
                         dropout: float = 0.3) -> nn.Module:
    """Load PhysicsKPConvNet from source and apply the requested ablation.

    Parameters
    ----------
    source_root : path containing ``model_1.py`` (the user's main repository).
    variant     : one of ``full``, ``without_physical_stats``,
                  ``isotropic_kpconv``, ``single_scale_strip``,
                  ``without_track_transformer``,
                  ``without_gated_decoder``.
    """
    model_path = os.path.join(source_root, "model_1.py")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model_1.py not found at {model_path}")
    with _source_path(source_root):
        spec = importlib.util.spec_from_file_location("baselines1_proposed_model_1", model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load model_1.py from {model_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.PhysicsKPConvNet(num_classes=num_classes, dropout=dropout)

    if variant == "full":
        return model
    if variant == "without_physical_stats":
        return _PhysStatsMaskedModel(model)
    if variant == "isotropic_kpconv":
        _replace_strip_with_isotropic(model)
        return model
    if variant == "single_scale_strip":
        _replace_multiscale_with_single_scale(model)
        return model
    if variant == "without_track_transformer":
        if hasattr(model, "track_transformer"):
            model.track_transformer = _IdentityTrackTransformer()
        return model
    if variant == "without_gated_decoder":
        _replace_decoder_gates_with_identity(model)
        return model
    raise ValueError(f"Unknown variant: {variant}")
