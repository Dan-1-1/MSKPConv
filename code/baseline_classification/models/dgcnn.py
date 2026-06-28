"""DGCNN (Wang et al., ACM TOG 2019) — strict re-implementation for ICESat-2.

Reference: "Dynamic Graph CNN for Learning on Point Clouds"
(https://arxiv.org/abs/1801.07829).

Core operator: EdgeConv
-----------------------
For each point x_i and its k-nearest neighbours {x_j} (recomputed in feature
space at every layer):
    e_ij = MLP(concat(x_i, x_j - x_i))
    x_i' = max_j e_ij

The segmentation network stacks 3-4 EdgeConv layers, concatenates the outputs,
appends a global feature, and uses point-wise MLPs to produce per-point logits.

Adaptations for ICESat-2: 2D positions are concatenated with 24 statistical
features as initial point representations.
"""
import torch
import torch.nn as nn


def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """k-NN in feature space. x: [B, C, N] -> idx: [B, N, k]."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(dim=1, keepdim=True)
    pairwise = -xx - inner - xx.transpose(2, 1)
    idx = pairwise.topk(k=k, dim=-1)[1]
    return idx


def _get_graph_feature(x: torch.Tensor, k: int = 20, idx: torch.Tensor = None) -> torch.Tensor:
    """Construct the EdgeConv input tensor [B, 2C, N, k]."""
    B, C, N = x.shape
    if idx is None:
        idx = _knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)
    feat = x.transpose(2, 1).contiguous().view(B * N, C)[idx, :]
    feat = feat.view(B, N, k, C)
    x_rep = x.transpose(2, 1).contiguous().view(B, N, 1, C).expand(-1, -1, k, -1)
    out = torch.cat([feat - x_rep, x_rep], dim=3).permute(0, 3, 1, 2).contiguous()
    return out


class EdgeConv(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, k: int = 20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel * 2, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        feat = _get_graph_feature(x, k=self.k)
        feat = self.conv(feat)
        return feat.max(dim=-1)[0]


class DGCNNSeg(nn.Module):
    """DGCNN segmentation network for ICESat-2 photon classification."""

    def __init__(self, num_classes: int = 4, in_feat: int = 24, pos_dim: int = 2,
                 k: int = 20, emb_dim: int = 1024):
        super().__init__()
        in_channel = in_feat + pos_dim  # concatenate position and features
        self.conv1 = EdgeConv(in_channel, 64, k=k)
        self.conv2 = EdgeConv(64, 64, k=k)
        self.conv3 = EdgeConv(64, 64, k=k)
        self.conv4 = EdgeConv(64, 128, k=k)

        self.embed = nn.Sequential(
            nn.Conv1d(64 + 64 + 64 + 128, emb_dim, 1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv1d(emb_dim + 64 + 64 + 64 + 128, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256, num_classes, 1),
        )

    def forward(self, pos: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # [B, N, C] -> [B, C, N]
        x = torch.cat([pos, feats], dim=-1).permute(0, 2, 1).contiguous()
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        cat_local = torch.cat([x1, x2, x3, x4], dim=1)  # [B, 320, N]
        emb = self.embed(cat_local)  # [B, emb_dim, N]
        # Global feature: max pool across points, broadcast back
        global_feat = emb.max(dim=-1, keepdim=True)[0].expand(-1, -1, x.size(-1))
        x = torch.cat([global_feat, cat_local], dim=1)
        x = self.head(x)
        return x.permute(0, 2, 1).contiguous()
