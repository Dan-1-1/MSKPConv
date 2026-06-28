"""Loss functions. baselines1 uses ONLY standard cross-entropy across all experiments."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardCELoss(nn.Module):
    """Plain cross-entropy with optional class weights.

    All baselines1 experiments (E1 comparison + E2 ablation + main proposed model)
    use this loss to ensure fair comparison.
    """

    def __init__(self, num_classes: int = 4, weight: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits shape: [B, N, C]; labels shape: [B, N]
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
        return F.cross_entropy(
            logits,
            labels,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
        )

    def step_lovasz_weight(self, *args, **kwargs):
        # Compatibility shim for any caller that mirrors physics-loss API.
        return None
