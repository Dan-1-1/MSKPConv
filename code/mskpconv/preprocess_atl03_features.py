import os

import numpy as np
import pandas as pd
import torch

from config import Config


def memory_safe_knn_gpu(points_gpu, k, max_mem_elements=50_000_000):
    """Run chunked KNN on GPU and return fixed-size neighbors excluding each point itself."""
    num_points = points_gpu.shape[0]
    if num_points == 0:
        return torch.zeros((0, 0), dtype=torch.long, device=points_gpu.device)
    if num_points == 1:
        return torch.zeros((1, k), dtype=torch.long, device=points_gpu.device)

    actual_k = min(k, num_points - 1)
    chunk_size = max(1, max_mem_elements // num_points)
    indices = torch.zeros((num_points, k), dtype=torch.long, device=points_gpu.device)

    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        chunk = points_gpu[start:end]
        dist_sq = torch.cdist(chunk, points_gpu, p=2) ** 2
        local_rows = torch.arange(end - start, device=points_gpu.device)
        global_cols = torch.arange(start, end, device=points_gpu.device)
        dist_sq[local_rows, global_cols] = float("inf")
        _, chunk_indices = torch.topk(dist_sq, actual_k, largest=False, sorted=True)
        if actual_k < k:
            pad = chunk_indices[:, -1:].expand(-1, k - actual_k)
            chunk_indices = torch.cat([chunk_indices, pad], dim=1)
        indices[start:end] = chunk_indices

    return indices


class PreprocessCache:
    """Cache scaled coordinate spaces and KNN indices for repeated feature extraction."""

    def __init__(self, x_norm, z_norm):
        self.x_norm = x_norm
        self.z_norm = z_norm

        depth = torch.clamp(-z_norm, min=0.0)
        self.pull_factor = torch.clamp(1.0 + depth / 2.0, max=20.0)
        self.pf_sq = (self.pull_factor**2).unsqueeze(1)
        self.max_pf = torch.max(self.pull_factor).item()
        self.cache = {}

    def get_space_and_knn(self, sx, sz, required_k):
        key = (sx, sz)
        if key not in self.cache:
            p_scaled = torch.stack([self.x_norm * sx, self.z_norm * sz], dim=1)
            max_k = min(150, max(1, p_scaled.shape[0] - 1))
            indices = memory_safe_knn_gpu(p_scaled, max_k)
            self.cache[key] = (p_scaled, indices)

        p_scaled, all_indices = self.cache[key]
        actual_k = min(required_k, all_indices.shape[1])
        indices = all_indices[:, :actual_k]
        if actual_k < required_k:
            pad = indices[:, -1:].expand(-1, required_k - actual_k)
            indices = torch.cat([indices, pad], dim=1)
        return p_scaled, indices


def compute_multi_aspect_density_gpu(x_norm, z_norm, configs, cache: PreprocessCache):
    """Compute multi-scale density features in anisotropically scaled spaces."""
    density_features = []
    pf_sq = cache.pf_sq
    max_pf = cache.max_pf

    for radius, sx, sz in configs:
        num_neighbors = min(int(radius * 15 * max_pf), 150)
        num_neighbors = max(1, num_neighbors)
        p_scaled, indices = cache.get_space_and_knn(sx, sz, num_neighbors)

        neighbors = p_scaled[indices]
        curr_p = p_scaled.unsqueeze(1)
        raw_dist_sq = torch.sum((neighbors - curr_p) ** 2, dim=2)
        pulled_dist_sq = raw_dist_sq / pf_sq
        mask = pulled_dist_sq <= radius**2

        gaussian_weight = torch.exp(-pulled_dist_sq / 2.0)
        rho_i = torch.sum(gaussian_weight * mask.float(), dim=1)
        density_features.append(rho_i)

    return density_features


def compute_directional_asymmetry_gpu(x_norm, z_norm, configs, cache: PreprocessCache):
    """Compute multi-scale vertical and horizontal asymmetry from density-style neighborhoods."""
    vertical_asymmetry_features = []
    horizontal_asymmetry_features = []
    pf_sq = cache.pf_sq
    max_pf = cache.max_pf
    vertical_margins = (0.05, 0.10, 0.15, 0.20)
    horizontal_margins = (0.5, 1.0, 1.5, 2.0)

    for scale_idx, (radius, sx, sz) in enumerate(configs):
        num_neighbors = min(int(radius * 15 * max_pf), 150)
        num_neighbors = max(1, num_neighbors)
        p_scaled, indices = cache.get_space_and_knn(sx, sz, num_neighbors)

        neighbors = p_scaled[indices]
        curr_p = p_scaled.unsqueeze(1)
        raw_dist_sq = torch.sum((neighbors - curr_p) ** 2, dim=2)
        pulled_dist_sq = raw_dist_sq / pf_sq
        mask = pulled_dist_sq <= radius**2

        weights = torch.exp(-pulled_dist_sq / 2.0) * mask.float()
        weights_norm = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-6)

        neighbors_x = x_norm[indices]
        neighbors_z = z_norm[indices]
        dx = neighbors_x - x_norm.unsqueeze(1)
        dz = neighbors_z - z_norm.unsqueeze(1)

        vertical_margin = vertical_margins[min(scale_idx, len(vertical_margins) - 1)]
        horizontal_margin = horizontal_margins[min(scale_idx, len(horizontal_margins) - 1)]

        upper_density = torch.sum(weights_norm * (dz > vertical_margin).float(), dim=1)
        lower_density = torch.sum(weights_norm * (dz < -vertical_margin).float(), dim=1)
        vertical_asymmetry = (upper_density - lower_density) / (upper_density + lower_density + 1e-6)
        vertical_asymmetry_features.append(vertical_asymmetry)

        right_density = torch.sum(weights_norm * (dx > horizontal_margin).float(), dim=1)
        left_density = torch.sum(weights_norm * (dx < -horizontal_margin).float(), dim=1)
        horizontal_asymmetry = (right_density - left_density) / (right_density + left_density + 1e-6)
        horizontal_asymmetry_features.append(horizontal_asymmetry)

    return vertical_asymmetry_features, horizontal_asymmetry_features


def extract_knn_statistical_features_gpu(x_norm, z_norm, configs, cache: PreprocessCache):
    """Extract multi-scale elevation, outlier, and vertical spread features."""
    z_diff_features = []
    lofe_features = []
    z_std_features = []

    pull_factor = cache.pull_factor
    pf_sq = cache.pf_sq

    for k, sx, sz in configs:
        p_scaled, indices = cache.get_space_and_knn(sx, sz, k)

        neighbors_scaled = p_scaled[indices]
        curr_p_scaled = p_scaled.unsqueeze(1)
        raw_dist_sq = torch.sum((neighbors_scaled - curr_p_scaled) ** 2, dim=2)
        pulled_dist_sq = raw_dist_sq / pf_sq

        weights = torch.exp(-pulled_dist_sq / 2.0)
        weights_norm = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-6)

        neighbors_z = z_norm[indices]
        weighted_mean_z = torch.sum(neighbors_z * weights_norm, dim=1)
        z_diff_k = z_norm - weighted_mean_z
        z_diff_features.append(z_diff_k / torch.sqrt(pull_factor))

        z_centered = neighbors_z - weighted_mean_z.unsqueeze(1)
        weighted_var_z = torch.sum((z_centered**2) * weights_norm, dim=1)
        z_std_features.append(torch.sqrt(weighted_var_z + 1e-8) / pull_factor)

        pulled_dists = torch.sqrt(pulled_dist_sq + 1e-8)
        weighted_mean_dist = torch.sum(pulled_dists * weights_norm, dim=1)
        lofe_features.append(-weighted_mean_dist)

    return z_diff_features, lofe_features, z_std_features


def _group_norm_inplace(tensor, indices):
    """Apply file-level standardization to one feature group."""
    sub_group = tensor[:, indices]
    mean = sub_group.mean()
    std = sub_group.std() + 1e-6
    tensor[:, indices] = (sub_group - mean) / std


def load_and_preprocess_single_csv(filepath: str, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load one CSV and compute stable multi-scale local statistical features."""
    df = pd.read_csv(filepath)

    actual_z_col = next((col for col in Config.COL_Z if col in df.columns), None)
    if actual_z_col is None:
        raise KeyError(f"Missing Z coordinate column. Available columns: {list(df.columns)}")

    df = df.dropna(subset=[Config.COL_X, actual_z_col])
    if len(df) == 0:
        raise ValueError(f"File {os.path.basename(filepath)} is empty after dropping NaN coordinates.")

    x_raw = df[Config.COL_X].values.astype(np.float64)
    z_raw = df[actual_z_col].values.astype(np.float64)

    if Config.COL_LABEL in df.columns:
        labels_raw = df[Config.COL_LABEL].values.astype(np.int64)
        labels = np.array([Config.LABEL_MAP.get(label, 0) for label in labels_raw], dtype=np.int64)
    else:
        labels = np.zeros(len(x_raw), dtype=np.int64)

    x_raw_gpu = torch.tensor(x_raw, dtype=torch.float32, device=device)
    z_raw_gpu = torch.tensor(z_raw, dtype=torch.float32, device=device)

    try:
        with torch.no_grad():
            x_norm_gpu = x_raw_gpu - torch.mean(x_raw_gpu)
            z_norm_gpu = z_raw_gpu

            shared_cache = PreprocessCache(x_norm_gpu, z_norm_gpu)

            knn_configs = [
                (5, 1 / 5, 1.0),
                (10, 1 / 10, 2.0),
                (15, 1 / 15, 2.0),
                (25, 1 / 20, 4.0),
            ]
            z_diff_list, lofe_list, z_std_list = extract_knn_statistical_features_gpu(
                x_norm_gpu,
                z_norm_gpu,
                knn_configs,
                shared_cache,
            )

            density_configs = [
                (1.0, 1 / 5, 1.0),
                (2.0, 1 / 10, 2.0),
                (4.0, 1 / 15, 2.0),
                (5.0, 1 / 20, 4.0),
            ]
            density_list = compute_multi_aspect_density_gpu(
                x_norm_gpu,
                z_norm_gpu,
                density_configs,
                shared_cache,
            )

            vertical_asymmetry_list, horizontal_asymmetry_list = compute_directional_asymmetry_gpu(
                x_norm_gpu,
                z_norm_gpu,
                density_configs,
                shared_cache,
            )

            features_raw = torch.stack(
                [
                    z_diff_list[0],
                    z_diff_list[1],
                    z_diff_list[2],
                    z_diff_list[3],
                    lofe_list[0],
                    lofe_list[1],
                    lofe_list[2],
                    lofe_list[3],
                    density_list[0],
                    density_list[1],
                    density_list[2],
                    density_list[3],
                    z_std_list[0],
                    z_std_list[1],
                    z_std_list[2],
                    z_std_list[3],
                    vertical_asymmetry_list[0],
                    vertical_asymmetry_list[1],
                    vertical_asymmetry_list[2],
                    vertical_asymmetry_list[3],
                    horizontal_asymmetry_list[0],
                    horizontal_asymmetry_list[1],
                    horizontal_asymmetry_list[2],
                    horizontal_asymmetry_list[3],
                ],
                dim=-1,
            )
            features_gpu = torch.nan_to_num(features_raw, nan=0.0, posinf=0.0, neginf=0.0)

            _group_norm_inplace(features_gpu, [0, 1, 2, 3])
            _group_norm_inplace(features_gpu, [4, 5, 6, 7])
            features_gpu[:, 8:12] = torch.log1p(torch.clamp(features_gpu[:, 8:12], min=0.0))
            _group_norm_inplace(features_gpu, [8, 9, 10, 11])
            _group_norm_inplace(features_gpu, [12, 13, 14, 15])
            _group_norm_inplace(features_gpu, [16, 17, 18, 19])
            _group_norm_inplace(features_gpu, [20, 21, 22, 23])
            features_gpu = torch.nan_to_num(features_gpu, nan=0.0)

        result = {
            "x_norm": x_norm_gpu.cpu().numpy(),
            "z_norm": z_norm_gpu.cpu().numpy(),
            "x_raw": x_raw.astype(np.float32),
            "z_raw": z_raw.astype(np.float32),
            "features": features_gpu.cpu().numpy(),
            "labels": labels,
            "filename": os.path.basename(filepath),
            "num_points": len(x_raw),
        }
    finally:
        if "x_raw_gpu" in locals():
            del x_raw_gpu
        if "z_raw_gpu" in locals():
            del z_raw_gpu
        if "features_gpu" in locals():
            del features_gpu
        if "x_norm_gpu" in locals():
            del x_norm_gpu
        if "z_norm_gpu" in locals():
            del z_norm_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result
