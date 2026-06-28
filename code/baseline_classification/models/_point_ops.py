"""Common point operations used by PointNet++/PointNeXt/KPConv (CUDA-friendly).

These are pure-PyTorch implementations of the canonical operators from the
official papers. Adapted to the 2D photon point-cloud (along-track x, height z).
"""
import torch
import torch.nn as nn


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distances. src: [B,N,C], dst: [B,M,C] -> [B,N,M]."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += (src ** 2).sum(-1, keepdim=True)
    dist += (dst ** 2).sum(-1, keepdim=True).transpose(1, 2)
    return dist.clamp(min=0)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather operator. points: [B,N,C], idx: [B,...] -> [B,...,C]."""
    device = points.device
    B = points.shape[0]
    view_shape = [B] + [1] * (idx.dim() - 1)
    rep_shape = [1] + list(idx.shape[1:])
    batch_idx = torch.arange(B, dtype=torch.long, device=device).view(view_shape).expand(*([B] + list(idx.shape[1:])))
    return points[batch_idx, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """FPS as in PointNet++. xyz: [B,N,C] -> idx: [B,npoint]."""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].unsqueeze(1)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = distance.argmax(dim=-1)
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """Ball-query: indices of up to nsample points within radius of each query."""
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).expand(B, S, N).clone()
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # Fallback when no neighbour falls inside the radius for some queries:
    # use the kNN index of the closest support point so we never emit `N` (which
    # would index out-of-bounds in the subsequent gather).
    knn_first = sqrdists.argmin(dim=-1, keepdim=True)  # [B, S, 1]
    group_first = knn_first.expand(-1, -1, nsample).clone()
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    # Final safety clamp in case both branches degenerated.
    group_idx = group_idx.clamp(0, N - 1)
    return group_idx


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """k-NN by squared distance. Returns [B, S, k] indices into xyz."""
    sqrdist = square_distance(new_xyz, xyz)
    _, idx = sqrdist.topk(k, dim=-1, largest=False)
    return idx


def sample_and_group(npoint: int, radius: float, nsample: int,
                     xyz: torch.Tensor, points: torch.Tensor):
    """Returns (new_xyz, new_points): grouped relative coords concatenated with features."""
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    return new_xyz, new_points, fps_idx, idx


class SharedMLP(nn.Module):
    """Shared MLP (1x1 conv on the channel axis) used throughout PointNet variants."""

    def __init__(self, dims, act=True, bn=True, last_act=True):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv2d(dims[i], dims[i + 1], 1, bias=not bn))
            is_last = (i == len(dims) - 2)
            if bn:
                layers.append(nn.BatchNorm2d(dims[i + 1]))
            if act and (not is_last or last_act):
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SharedMLP1d(nn.Module):
    def __init__(self, dims, act=True, bn=True, last_act=True):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Conv1d(dims[i], dims[i + 1], 1, bias=not bn))
            is_last = (i == len(dims) - 2)
            if bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            if act and (not is_last or last_act):
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
