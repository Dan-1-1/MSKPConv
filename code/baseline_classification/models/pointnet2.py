"""PointNet++ (Qi et al., NeurIPS 2017) — strict re-implementation for ICESat-2.

Reference: "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a
Metric Space" (https://arxiv.org/abs/1706.02413).

Adaptations for ICESat-2 photon point clouds:
- Input coordinates are 2D (along-track x, height z) instead of 3D.
- 24-D statistical features attached as point features.
- Set-Abstraction (SA) blocks use ball-query in 2D metric space.
- Feature-Propagation (FP) blocks use 3-NN inverse-distance interpolation.
- Output is per-point 4-class logits (noise / surface / land / seabed).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._point_ops import (
    SharedMLP,
    SharedMLP1d,
    farthest_point_sample,
    index_points,
    query_ball_point,
    square_distance,
)


class SetAbstraction(nn.Module):
    """SA layer: FPS -> ball-query group -> shared MLP -> max-pool."""

    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels, pos_dim=2):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pos_dim = pos_dim
        dims = [in_channel + pos_dim] + list(mlp_channels)
        self.mlp = SharedMLP(dims)

    def forward(self, xyz, points):
        # xyz: [B,N,pos_dim], points: [B,N,C] or None
        B, N, _ = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)
        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz
        # [B, S, K, C] -> [B, C, S, K]
        new_points = new_points.permute(0, 3, 1, 2).contiguous()
        new_points = self.mlp(new_points)
        new_points = new_points.max(dim=-1)[0]  # max over K -> [B, C', S]
        new_points = new_points.permute(0, 2, 1).contiguous()  # [B, S, C']
        return new_xyz, new_points


class FeaturePropagation(nn.Module):
    """FP layer: 3-NN inverse-distance weighted interpolation + skip + MLP."""

    def __init__(self, in_channel, mlp_channels):
        super().__init__()
        dims = [in_channel] + list(mlp_channels)
        self.mlp = SharedMLP1d(dims)

    def forward(self, xyz1, xyz2, points1, points2):
        # xyz1: dense (target) [B, N, pos_dim]; xyz2: sparse (source) [B, S, pos_dim]
        # points1: skip features [B, N, C1] or None; points2: source features [B, S, C2]
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interp = points2.expand(-1, N, -1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists.sqrt() + 1e-8)
            norm = dist_recip.sum(dim=-1, keepdim=True)
            weight = (dist_recip / norm).unsqueeze(-1)  # [B,N,3,1]
            grouped = index_points(points2, idx)  # [B,N,3,C2]
            interp = (grouped * weight).sum(dim=2)
        if points1 is not None:
            new_points = torch.cat([points1, interp], dim=-1)
        else:
            new_points = interp
        # [B, N, C] -> [B, C, N]
        new_points = new_points.permute(0, 2, 1).contiguous()
        new_points = self.mlp(new_points)
        return new_points.permute(0, 2, 1).contiguous()


class PointNet2Seg(nn.Module):
    """PointNet++ for per-point segmentation, ICESat-2 adapted.

    Input
    -----
    pos:   [B, N, 2]  (x_norm, z_norm)
    feats: [B, N, 24] (statistical features)
    """

    def __init__(self, num_classes: int = 4, in_feat: int = 24, pos_dim: int = 2):
        super().__init__()
        self.pos_dim = pos_dim
        # Encoder
        self.sa1 = SetAbstraction(npoint=1024, radius=0.1, nsample=32,
                                  in_channel=in_feat, mlp_channels=[64, 64, 128], pos_dim=pos_dim)
        self.sa2 = SetAbstraction(npoint=256, radius=0.2, nsample=32,
                                  in_channel=128, mlp_channels=[128, 128, 256], pos_dim=pos_dim)
        self.sa3 = SetAbstraction(npoint=64, radius=0.4, nsample=32,
                                  in_channel=256, mlp_channels=[256, 256, 512], pos_dim=pos_dim)
        self.sa4 = SetAbstraction(npoint=16, radius=0.8, nsample=32,
                                  in_channel=512, mlp_channels=[512, 512, 1024], pos_dim=pos_dim)
        # Decoder
        self.fp4 = FeaturePropagation(1024 + 512, [512, 512])
        self.fp3 = FeaturePropagation(512 + 256, [512, 256])
        self.fp2 = FeaturePropagation(256 + 128, [256, 128])
        self.fp1 = FeaturePropagation(128 + in_feat, [128, 128, 128])
        # Head
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, pos: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # Encoder
        l0_xyz = pos
        l0_points = feats
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # Decoder
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        # Head: [B, N, C] -> [B, C, N] -> [B, num_classes, N] -> [B, N, num_classes]
        x = l0_points.permute(0, 2, 1).contiguous()
        x = self.head(x)
        return x.permute(0, 2, 1).contiguous()
