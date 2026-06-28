"""PointNeXt (Qian et al., NeurIPS 2022) — strict re-implementation for ICESat-2.

Reference: "PointNeXt: Revisiting PointNet++ with Improved Training and Scaling
Strategies" (https://arxiv.org/abs/2206.04670).

Key building blocks reproduced:
- SetAbstraction (SA) with ball-query and shared MLP, kept identical to PointNet++.
- InvResMLP block: pointwise channel reduction -> grouper + reduction (max) ->
  channel restore -> residual addition. This is the "Inverted Residual MLP" that
  is the central scaling primitive of PointNeXt.
- Decoder identical to PointNet++ FP layers (3-NN interpolation + MLP).

Adaptations for ICESat-2:
- Input is (B, N, 2) along-track + height; 24-D statistical features.
- Output is per-point 4-class logits.
- We follow the PointNeXt-S configuration (radius scaling 0.5x base, 1 InvResMLP
  block per stage), matching the PointNeXt official scaling rule for small
  point clouds.
"""
import torch
import torch.nn as nn

from ._point_ops import (
    SharedMLP,
    SharedMLP1d,
    farthest_point_sample,
    index_points,
    query_ball_point,
    square_distance,
)


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp_channels, pos_dim=2):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        dims = [in_channel + pos_dim] + list(mlp_channels)
        self.mlp = SharedMLP(dims)

    def forward(self, xyz, points):
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)
        if points is not None:
            grouped = index_points(points, idx)
            new_points = torch.cat([grouped_xyz, grouped], dim=-1)
        else:
            new_points = grouped_xyz
        new_points = new_points.permute(0, 3, 1, 2).contiguous()
        new_points = self.mlp(new_points)
        new_points = new_points.max(dim=-1)[0]
        return new_xyz, new_points.permute(0, 2, 1).contiguous()


class InvResMLPBlock(nn.Module):
    """Inverted Residual MLP block from PointNeXt.

    Structure (channels=C, expansion=E):
        x -> Conv1d(C->C*E) -> BN -> ReLU
          -> Group(ball_query) + reduction (max with relative pos)
          -> Conv1d(C*E -> C) -> BN
          -> Residual + ReLU
    """

    def __init__(self, channel, radius, nsample, expansion=2, pos_dim=2):
        super().__init__()
        self.radius = radius
        self.nsample = nsample
        hidden = channel * expansion
        self.pre = nn.Sequential(
            nn.Conv1d(channel, hidden, 1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.local = SharedMLP([hidden + pos_dim, hidden, hidden])
        self.post = nn.Sequential(
            nn.Conv1d(hidden, channel, 1, bias=False),
            nn.BatchNorm1d(channel),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz, points):
        # points: [B, N, C]
        identity = points
        B, N, C = points.shape
        x = points.permute(0, 2, 1).contiguous()
        x = self.pre(x)  # [B, hidden, N]
        x_perm = x.permute(0, 2, 1).contiguous()
        idx = query_ball_point(self.radius, self.nsample, xyz, xyz)
        grouped_xyz = index_points(xyz, idx) - xyz.unsqueeze(2)
        grouped = index_points(x_perm, idx)
        local_in = torch.cat([grouped_xyz, grouped], dim=-1)
        local_in = local_in.permute(0, 3, 1, 2).contiguous()
        local_out = self.local(local_in).max(dim=-1)[0]  # [B, hidden, N]
        out = self.post(local_out)
        out = self.act(out + identity.permute(0, 2, 1).contiguous())
        return xyz, out.permute(0, 2, 1).contiguous()


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp_channels):
        super().__init__()
        dims = [in_channel] + list(mlp_channels)
        self.mlp = SharedMLP1d(dims)

    def forward(self, xyz1, xyz2, points1, points2):
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
            weight = (dist_recip / norm).unsqueeze(-1)
            grouped = index_points(points2, idx)
            interp = (grouped * weight).sum(dim=2)
        if points1 is not None:
            new_points = torch.cat([points1, interp], dim=-1)
        else:
            new_points = interp
        x = new_points.permute(0, 2, 1).contiguous()
        x = self.mlp(x)
        return x.permute(0, 2, 1).contiguous()


class PointNeXtSeg(nn.Module):
    """PointNeXt-S for ICESat-2 photon segmentation."""

    def __init__(self, num_classes: int = 4, in_feat: int = 24, pos_dim: int = 2,
                 base_channels: int = 32, expansion: int = 4):
        super().__init__()
        c1, c2, c3, c4 = (base_channels, base_channels * 2,
                          base_channels * 4, base_channels * 8)
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_feat, c1, 1, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )
        # Encoder: SA (downsample) followed by InvResMLP
        self.sa1 = SetAbstraction(1024, 0.1, 32, c1, [c2, c2], pos_dim)
        self.ir1 = InvResMLPBlock(c2, 0.1, 32, expansion=expansion, pos_dim=pos_dim)

        self.sa2 = SetAbstraction(256, 0.2, 32, c2, [c3, c3], pos_dim)
        self.ir2 = InvResMLPBlock(c3, 0.2, 32, expansion=expansion, pos_dim=pos_dim)

        self.sa3 = SetAbstraction(64, 0.4, 32, c3, [c4, c4], pos_dim)
        self.ir3 = InvResMLPBlock(c4, 0.4, 32, expansion=expansion, pos_dim=pos_dim)

        self.sa4 = SetAbstraction(16, 0.8, 16, c4, [c4 * 2, c4 * 2], pos_dim)

        # Decoder (PointNet++ FP)
        self.fp4 = FeaturePropagation(c4 * 2 + c4, [c4, c4])
        self.fp3 = FeaturePropagation(c4 + c3, [c3, c3])
        self.fp2 = FeaturePropagation(c3 + c2, [c2, c2])
        self.fp1 = FeaturePropagation(c2 + c1, [c1, c1])

        self.head = nn.Sequential(
            nn.Conv1d(c1, c1, 1, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(c1, num_classes, 1),
        )

    def forward(self, pos: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        x = feats.permute(0, 2, 1).contiguous()
        x = self.stem(x)
        l0_xyz = pos
        l0_points = x.permute(0, 2, 1).contiguous()

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_xyz, l1_points = self.ir1(l1_xyz, l1_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_xyz, l2_points = self.ir2(l2_xyz, l2_points)

        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_xyz, l3_points = self.ir3(l3_xyz, l3_points)

        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        out = l0_points.permute(0, 2, 1).contiguous()
        out = self.head(out)
        return out.permute(0, 2, 1).contiguous()
