"""KPConv (Thomas et al., ICCV 2019) — strict re-implementation for ICESat-2.

Reference: "KPConv: Flexible and Deformable Convolution for Point Clouds"
(https://arxiv.org/abs/1904.08889).

Core operator
-------------
KPConv computes, for each query point q with neighbours {n_i}:
    f'(q) = sum_i  sum_k  h(n_i - q, x_k) * f(n_i) * W_k

where {x_k} are K kernel-points distributed in the local domain and
    h(y, x_k) = max(0, 1 - ||y - x_k|| / sigma).

Adaptations for ICESat-2 photon point clouds
--------------------------------------------
- Kernel points are placed on a regular 2D grid inside a circle (matching the
  isotropic-circle layout used by the original code in 2D), since the input
  domain is (along-track, height).
- ``AnisotropicKPConvSeg`` keeps the same encoder-decoder skeleton but applies
  an elliptical metric in the KPConv influence distance, making the local
  support elongated along track.
- Two encoder stages with stride-2 grid-down-sampling (FPS used here for
  framework simplicity) and matching upsampling via nearest-neighbour.
- Per-point classifier head.
"""
import math
import torch
import torch.nn as nn

from ._point_ops import (
    SharedMLP1d,
    farthest_point_sample,
    index_points,
    knn_point,
)


def _make_kernel_points(num_kp: int, radius: float, dim: int = 2) -> torch.Tensor:
    """Place kernel points: 1 at the centre, the rest on concentric rings."""
    pts = [torch.zeros(dim)]
    remaining = num_kp - 1
    if remaining > 0:
        # Two rings at r/2 and r, distributing points proportionally
        rings = [(0.5 * radius, max(1, remaining // 3)),
                 (radius, remaining - max(1, remaining // 3))]
        for r, n in rings:
            for i in range(n):
                theta = 2 * math.pi * i / n
                pts.append(torch.tensor([r * math.cos(theta), r * math.sin(theta)][:dim]))
    return torch.stack(pts, dim=0)[:num_kp]


class KPConvLayer(nn.Module):
    """Rigid KPConv operator (no kernel deformation)."""

    def __init__(self, in_channel: int, out_channel: int, num_kp: int = 15,
                 radius: float = 0.1, sigma: float = None, k_neighbors: int = 16,
                 pos_dim: int = 2, metric_scale=None):
        super().__init__()
        self.num_kp = num_kp
        self.radius = radius
        self.sigma = sigma if sigma is not None else radius / 2.5
        self.k_neighbors = k_neighbors
        self.pos_dim = pos_dim
        if metric_scale is None:
            metric_scale = (1.0,) * pos_dim
        if len(metric_scale) != pos_dim:
            raise ValueError(f"metric_scale must have {pos_dim} entries")
        kp = _make_kernel_points(num_kp, radius, pos_dim)
        self.register_buffer("kernel_points", kp.float())  # [K, pos_dim]
        self.register_buffer("metric_scale", torch.tensor(metric_scale, dtype=torch.float32))
        self.weights = nn.Parameter(torch.empty(num_kp, in_channel, out_channel))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, q_xyz, s_xyz, s_feats):
        """
        q_xyz: [B, M, pos_dim] query points
        s_xyz: [B, N, pos_dim] support points (could equal q_xyz)
        s_feats: [B, N, C_in]
        Returns: [B, M, C_out]
        """
        B, M, _ = q_xyz.shape
        idx = knn_point(self.k_neighbors, s_xyz, q_xyz)  # [B, M, K]
        nn_xyz = index_points(s_xyz, idx)  # [B, M, K, pos_dim]
        nn_feats = index_points(s_feats, idx)  # [B, M, K, C_in]
        # Relative coordinates
        rel = nn_xyz - q_xyz.unsqueeze(2)  # [B, M, K, pos_dim]
        # Distance to each kernel point: [B, M, K, num_kp]
        diff = rel.unsqueeze(3) - self.kernel_points.view(1, 1, 1, self.num_kp, -1)
        diff = diff * self.metric_scale.view(1, 1, 1, 1, -1)
        dist = diff.norm(dim=-1)
        h = torch.clamp(1.0 - dist / self.sigma, min=0.0)  # [B, M, K, num_kp]
        # Weighted feature aggregation: combine kernel weights and per-neighbour features
        # f_k = sum_n h(n, k) * f_n  -> [B, M, num_kp, C_in]
        weighted_feats = torch.einsum("bmkp,bmkc->bmpc", h, nn_feats)
        # Apply per-kernel linear and sum: [num_kp, C_in, C_out]
        out = torch.einsum("bmpc,pco->bmo", weighted_feats, self.weights)
        return out


class AnisotropicKPConvLayer(KPConvLayer):
    """KPConv layer with an elliptical distance metric.

    Compressing the x coordinate in the distance calculation makes influence
    decay more slowly along track than along height.
    """

    def __init__(self, in_channel: int, out_channel: int, num_kp: int = 15,
                 radius: float = 0.1, sigma: float = None, k_neighbors: int = 16,
                 pos_dim: int = 2, x_scale: float = 0.25, z_scale: float = 1.0):
        if pos_dim != 2:
            raise ValueError("AnisotropicKPConvLayer expects 2D (x, z) positions")
        super().__init__(
            in_channel,
            out_channel,
            num_kp=num_kp,
            radius=radius,
            sigma=sigma,
            k_neighbors=k_neighbors,
            pos_dim=pos_dim,
            metric_scale=(x_scale, z_scale),
        )


class KPResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_kp=15, radius=0.1,
                 k_neighbors=16, kpconv_cls=KPConvLayer, **kpconv_kwargs):
        super().__init__()
        mid = max(out_channel // 2, 16)
        self.pre = nn.Sequential(
            nn.Conv1d(in_channel, mid, 1, bias=False),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
        )
        self.kpconv = kpconv_cls(mid, mid, num_kp, radius,
                                 k_neighbors=k_neighbors, **kpconv_kwargs)
        self.bn1 = nn.BatchNorm1d(mid)
        self.act1 = nn.ReLU(inplace=True)
        self.post = nn.Sequential(
            nn.Conv1d(mid, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
        )
        self.shortcut = (
            nn.Identity() if in_channel == out_channel
            else nn.Sequential(
                nn.Conv1d(in_channel, out_channel, 1, bias=False),
                nn.BatchNorm1d(out_channel),
            )
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz, feats):
        # feats: [B, N, C_in]
        x = feats.permute(0, 2, 1).contiguous()
        identity = self.shortcut(x)
        x = self.pre(x)
        x = self.kpconv(xyz, xyz, x.permute(0, 2, 1).contiguous())  # [B,N,mid]
        x = x.permute(0, 2, 1).contiguous()
        x = self.act1(self.bn1(x))
        x = self.post(x)
        x = self.act(x + identity)
        return x.permute(0, 2, 1).contiguous()


class KPConvSeg(nn.Module):
    """Encoder-decoder KPConv segmentation network for ICESat-2."""

    def __init__(self, num_classes: int = 4, in_feat: int = 24, pos_dim: int = 2,
                 kpconv_cls=KPConvLayer, **kpconv_kwargs):
        super().__init__()
        # Stem: lift point features to base channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_feat, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        # Encoder
        self.block1 = KPResidualBlock(64, 64, num_kp=15, radius=0.05,
                                      k_neighbors=16, kpconv_cls=kpconv_cls,
                                      **kpconv_kwargs)
        self.block2 = KPResidualBlock(64, 128, num_kp=15, radius=0.1,
                                      k_neighbors=16, kpconv_cls=kpconv_cls,
                                      **kpconv_kwargs)
        self.block3 = KPResidualBlock(128, 256, num_kp=15, radius=0.2,
                                      k_neighbors=16, kpconv_cls=kpconv_cls,
                                      **kpconv_kwargs)
        self.block4 = KPResidualBlock(256, 512, num_kp=15, radius=0.4,
                                      k_neighbors=16, kpconv_cls=kpconv_cls,
                                      **kpconv_kwargs)
        # Downsampling counts
        self.npoints = [1024, 256, 64]

        # Decoder: shared MLPs after concatenation with skip features
        self.up3 = SharedMLP1d([512 + 256, 256])
        self.up2 = SharedMLP1d([256 + 128, 128])
        self.up1 = SharedMLP1d([128 + 64, 128, 128])

        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1),
        )

    def _down(self, xyz, feats, npoint):
        npoint = min(npoint, xyz.shape[1])
        idx = farthest_point_sample(xyz, npoint)
        return index_points(xyz, idx), index_points(feats, idx)

    def _up(self, dense_xyz, sparse_xyz, sparse_feats):
        idx = knn_point(3, sparse_xyz, dense_xyz)
        grouped = index_points(sparse_feats, idx)  # [B, N, 3, C]
        return grouped.mean(dim=2)

    def forward(self, pos, feats):
        # Stem
        x = feats.permute(0, 2, 1).contiguous()
        x0 = self.stem(x).permute(0, 2, 1).contiguous()  # [B, N, 64]

        # Encoder stage 1 (full resolution)
        f1 = self.block1(pos, x0)  # [B, N, 64]
        # Down to 1024
        xyz2, f2_in = self._down(pos, f1, self.npoints[0])
        f2 = self.block2(xyz2, f2_in)  # [B, 1024, 128]
        xyz3, f3_in = self._down(xyz2, f2, self.npoints[1])
        f3 = self.block3(xyz3, f3_in)  # [B, 256, 256]
        xyz4, f4_in = self._down(xyz3, f3, self.npoints[2])
        f4 = self.block4(xyz4, f4_in)  # [B, 64, 512]

        # Decoder
        u3 = self._up(xyz3, xyz4, f4)  # [B, 256, 512]
        u3 = torch.cat([u3, f3], dim=-1).permute(0, 2, 1).contiguous()
        u3 = self.up3(u3).permute(0, 2, 1).contiguous()  # [B, 256, 256]

        u2 = self._up(xyz2, xyz3, u3)
        u2 = torch.cat([u2, f2], dim=-1).permute(0, 2, 1).contiguous()
        u2 = self.up2(u2).permute(0, 2, 1).contiguous()  # [B, 1024, 128]

        u1 = self._up(pos, xyz2, u2)
        u1 = torch.cat([u1, f1], dim=-1).permute(0, 2, 1).contiguous()
        u1 = self.up1(u1)  # [B, 128, N]

        out = self.head(u1)
        return out.permute(0, 2, 1).contiguous()


class AnisotropicKPConvSeg(KPConvSeg):
    """Elliptical-kernel KPConv baseline for the E2 ablation table."""

    def __init__(self, num_classes: int = 4, in_feat: int = 24, pos_dim: int = 2,
                 x_scale: float = 0.25, z_scale: float = 1.0):
        super().__init__(
            num_classes=num_classes,
            in_feat=in_feat,
            pos_dim=pos_dim,
            kpconv_cls=AnisotropicKPConvLayer,
            x_scale=x_scale,
            z_scale=z_scale,
        )
