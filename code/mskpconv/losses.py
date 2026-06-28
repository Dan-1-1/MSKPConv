import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """Compute Lovasz gradient from sorted binary ground truth."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes="present"):
    """Compute multiclass Lovasz-Softmax on flattened probabilities and labels."""
    if probas.numel() == 0:
        return probas * 0.0

    num_classes = probas.size(1)
    losses = []
    class_to_sum = list(range(num_classes)) if classes in ["all", "present"] else classes

    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue

        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))

    return sum(losses) / len(losses)


class DynamicWeightedCELoss(nn.Module):
    """Compute the same plain cross-entropy loss used in ablation runs."""

    def __init__(self, noise_penalty=1.0, ce_weight=0.5, dice_weight=0.5):
        """Keep signature for backward compatibility."""
        super().__init__()

    def forward(self, logits, labels, depth):
        """Return plain cross-entropy and zero placeholders for compatibility."""
        _ = depth
        loss = F.cross_entropy(logits, labels)
        zero = logits.new_tensor(0.0)
        return loss, loss, zero


class PhysicsLovaszLoss(nn.Module):
    """Focal-style CE with dynamic class weights plus staged Lovasz-Softmax term."""

    def __init__(self, max_depth_penalty=5, gamma=4.0, lovasz_weight=1, deep_expert_weight=2.0, deep_threshold=5):
        """Initialize focal gamma and staged Lovasz weighting controls."""
        super().__init__()
        self.max_penalty = max_depth_penalty
        self.gamma = gamma
        self.max_lovasz_weight = lovasz_weight
        self.current_lovasz_weight = 0.0

    def step_lovasz_weight(self, current_epoch, total_epochs):
        """Update current Lovasz weight according to training epoch schedule."""
        start_epoch = int(total_epochs * 0.1)
        end_epoch = int(total_epochs * 0.6)

        if current_epoch < start_epoch:
            self.current_lovasz_weight = 0.0
        elif current_epoch > end_epoch:
            self.current_lovasz_weight = self.max_lovasz_weight
        else:
            progress = (current_epoch - start_epoch) / (end_epoch - start_epoch)
            self.current_lovasz_weight = progress * self.max_lovasz_weight

    def forward(self, logits, labels, raw_depth):
        """Compute total loss and return total, focal-depth, and Lovasz parts."""
        _ = raw_depth
        _, num_classes, _ = logits.shape
        probs = F.softmax(logits, dim=1)

        dynamic_weights = []
        total_elements = labels.numel()
        for i in range(num_classes):
            num_class_i = (labels == i).sum().float()
            weight_i = 1.0 / (num_class_i / (total_elements + 1e-6) + 0.1)
            dynamic_weights.append(weight_i)

        dynamic_weights = torch.tensor(dynamic_weights, dtype=torch.float32, device=logits.device)
        dynamic_weights = dynamic_weights / dynamic_weights.sum() * num_classes

        ce_loss = F.cross_entropy(
            logits,
            labels,
            weight=dynamic_weights,
            reduction="none",
            label_smoothing=0.1,
        )

        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        pointwise_loss = ce_loss * focal_weight
        loss_focal_depth = pointwise_loss.mean()

        if self.current_lovasz_weight > 0:
            probs_flat = probs.transpose(1, 2).reshape(-1, num_classes)
            labels_flat = labels.reshape(-1)
            loss_lovasz = lovasz_softmax_flat(probs_flat, labels_flat)
        else:
            loss_lovasz = torch.tensor(0.0, device=logits.device)

        total_loss = loss_focal_depth * 10 + self.current_lovasz_weight * loss_lovasz
        return total_loss, loss_focal_depth, 0.5*loss_lovasz
