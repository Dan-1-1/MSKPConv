import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class FixedStripKPConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_kernel_points=20,
        radius=0.1,
        sigma=0.1,
        metric_scale=(1.0, 1.0),
        rect_rows=3,
        rect_width=0.3,
    ):
        """Initialize a fixed horizontal strip KPConv."""
        super(FixedStripKPConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_k = num_kernel_points
        self.radius = radius
        self.sigma = sigma
        self.metric_scale = tuple(float(v) for v in metric_scale)

        rows = max(1, min(int(rect_rows), num_kernel_points))
        cols = int(math.ceil(num_kernel_points / rows))
        main_axis = torch.linspace(-1, 1, cols)
        orth_axis = torch.linspace(-float(rect_width), float(rect_width), rows)
        grid_orth, grid_main = torch.meshgrid(orth_axis, main_axis, indexing="ij")
        initial_points = torch.stack([grid_main.reshape(-1), grid_orth.reshape(-1)], dim=1)
        initial_points = initial_points[:num_kernel_points]
        self.register_buffer("rigid_kernel_scaled", initial_points * radius)

        self.weights = nn.Parameter(torch.Tensor(num_kernel_points, in_channels, out_channels))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, q_pts, s_pts, s_feats, neighbor_idx):
        """Apply fixed strip KPConv from support points to query points."""
        B, N_q, K_n = neighbor_idx.shape
        Num_K = self.num_k
        In_C = self.in_channels
        device = q_pts.device

        kernel_pts = self.rigid_kernel_scaled.view(1, 1, Num_K, 2)

        s_pts_flat = s_pts.reshape(-1, 2)
        s_feats_flat = s_feats.permute(0, 2, 1).reshape(-1, In_C)

        batch_offsets = torch.arange(B, device=device).view(-1, 1, 1) * s_pts.shape[1]
        neighbor_idx_flat = (neighbor_idx + batch_offsets).reshape(-1)

        neighbors_pts = s_pts_flat[neighbor_idx_flat].reshape(B, N_q, K_n, 2)
        neighbors_diff = neighbors_pts - q_pts.unsqueeze(2)
        neighbors_feats = s_feats_flat[neighbor_idx_flat].reshape(B * N_q, K_n, In_C)

        metric_scale = neighbors_diff.new_tensor(self.metric_scale).view(1, 1, 1, 2)
        neighbors_diff = neighbors_diff * metric_scale
        kernel_adj = kernel_pts * metric_scale

        dist = torch.cdist(
            neighbors_diff.reshape(B * N_q, K_n, 2),
            kernel_adj.expand(B, N_q, -1, -1).reshape(B * N_q, Num_K, 2),
        )
        weights_influence = torch.clamp(1.0 - dist / self.sigma, min=0.0)

        kernel_weights = self.weights.reshape(Num_K, In_C, -1).permute(1, 0, 2).reshape(In_C, -1)
        weighted_feats = torch.matmul(neighbors_feats, kernel_weights).view(
            B * N_q, K_n, Num_K, self.out_channels
        )

        output = (weighted_feats * weights_influence.unsqueeze(-1)).sum(dim=(1, 2))
        output = output.reshape(B, N_q, self.out_channels).permute(0, 2, 1)
        return output


def _gather_points(x, idx):
    """Gather [B, N, C] points with [B, M] indices to [B, M, C]."""
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    return torch.gather(x, 1, idx_exp)


def _gather_features_chw(x, idx):
    """Gather [B, C, N] features with [B, M] indices to [B, C, M]."""
    idx_exp = idx.unsqueeze(1).expand(-1, x.shape[1], -1)
    return torch.gather(x, 2, idx_exp)


def _farthest_point_sampling(pos, num_samples):
    """Batched FPS on [B, N, 2], returning [B, num_samples] indices."""
    B, N, _ = pos.shape
    num_samples = max(1, min(num_samples, N))
    device = pos.device

    centroids = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)

    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = pos[batch_idx, farthest, :].unsqueeze(1)
        dist = torch.sum((pos - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=1)[1]
    return centroids


def _density_aware_sampling(pos, num_samples, grid_size=32, alpha=0.7):
    """Lightweight density-aware sampling on [B, N, 2], returning [B, num_samples]."""
    B, N, _ = pos.shape
    num_samples = max(1, min(num_samples, N))
    device = pos.device
    alpha = float(max(0.0, min(1.0, alpha)))
    num_density = int(round(num_samples * alpha))
    num_density = max(0, min(num_samples, num_density))
    num_fps = num_samples - num_density

    pmin = pos.min(dim=1, keepdim=True)[0]
    pmax = pos.max(dim=1, keepdim=True)[0]
    pnorm = (pos - pmin) / (pmax - pmin + 1e-6)
    cell = (pnorm * (grid_size - 1)).long().clamp(0, grid_size - 1)
    cell_id = cell[..., 0] * grid_size + cell[..., 1]

    fps_idx = _farthest_point_sampling(pos, num_samples)
    out = torch.empty((B, num_samples), dtype=torch.long, device=device)

    for b in range(B):
        cell_b = cell_id[b]
        counts = torch.bincount(cell_b, minlength=grid_size * grid_size).float()
        inv_density = 1.0 / (counts[cell_b] + 1e-6)
        prob = inv_density / (inv_density.sum() + 1e-12)
        density_pick = torch.multinomial(prob, num_samples=num_samples, replacement=False)

        used = torch.zeros(N, dtype=torch.bool, device=device)
        chosen = []

        def _append_unique(src, limit):
            if limit <= 0:
                return 0
            valid = src[~used[src]]
            take = min(limit, valid.shape[0])
            if take > 0:
                picked = valid[:take]
                used[picked] = True
                chosen.append(picked)
            return take

        taken = _append_unique(fps_idx[b], num_fps)
        _ = taken
        _append_unique(density_pick, num_density)
        filled = sum(t.shape[0] for t in chosen)

        if filled < num_samples:
            _append_unique(fps_idx[b], num_samples - filled)
            filled = sum(t.shape[0] for t in chosen)
        if filled < num_samples:
            _append_unique(torch.randperm(N, device=device), num_samples - filled)

        out[b] = torch.cat(chosen, dim=0)[:num_samples]

    return out


class MultiScaleFixedStripKPBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_points=(8, 12, 16, 20),
        radius=0.1,
        downsample=False,
    ):
        """Initialize unified multi-scale horizontal strip KPConv branches."""
        super().__init__()
        self.downsample = downsample
        k1, k2, k3, k4 = kernel_points

        scale_channels = out_channels // 4
        mid_channels = out_channels // 2

        self.unary1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(0.2),
        )

        self.kpconv_r1 = FixedStripKPConv(
            mid_channels,
            scale_channels,
            num_kernel_points=k1,
            radius=radius,
            sigma=radius * 0.2,
            metric_scale=(1 / 5, 1.0),
            rect_rows=2,
            rect_width=0.05,
        )
        self.kpconv_r2 = FixedStripKPConv(
            mid_channels,
            scale_channels,
            num_kernel_points=k2,
            radius=radius * 2.0,
            sigma=radius * 0.4,
            metric_scale=(1 / 10, 2.0),
            rect_rows=2,
            rect_width=0.06,
        )
        self.kpconv_r3 = FixedStripKPConv(
            mid_channels,
            scale_channels,
            num_kernel_points=k3,
            radius=radius * 3.0,
            sigma=radius * 0.6,
            metric_scale=(1 / 15, 2.0),
            rect_rows=3,
            rect_width=0.07,
        )
        self.kpconv_r4 = FixedStripKPConv(
            mid_channels,
            scale_channels,
            num_kernel_points=k4,
            radius=radius * 4.0,
            sigma=radius * 0.8,
            metric_scale=(1 / 20, 4.0),
            rect_rows=3,
            rect_width=0.08,
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(scale_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, q_pos, s_pos, neighbor_idx_list, q_idx=None):
        """Run multi-scale KP branches and fuse with residual connection."""
        idx_r1, idx_r2, idx_r3, idx_r4 = neighbor_idx_list

        if self.downsample:
            if q_idx is not None:
                identity_src = _gather_features_chw(x, q_idx)
            else:
                identity_src = F.interpolate(x, size=q_pos.shape[1], mode="nearest")
            identity = self.shortcut(identity_src)
        else:
            identity = self.shortcut(x)

        out_mid = self.unary1(x)
        scale1 = self.kpconv_r1(q_pos, s_pos, out_mid, idx_r1)
        scale2 = self.kpconv_r2(q_pos, s_pos, out_mid, idx_r2)
        scale3 = self.kpconv_r3(q_pos, s_pos, out_mid, idx_r3)
        scale4 = self.kpconv_r4(q_pos, s_pos, out_mid, idx_r4)

        concat_feat = torch.cat([scale1, scale2, scale3, scale4], dim=1)
        out = self.fusion(concat_feat)
        out = self.act(out + identity)
        return out


class PointInterpolateBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        """Initialize interpolation block for decoder feature upsampling."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels + skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x_s, pos_s, x_d, pos_d):
        """Interpolate sparse features to dense points with 3-NN distance weighting."""
        dist = torch.cdist(pos_d, pos_s)
        dists, idx = dist.topk(3, dim=-1, largest=False)
        dist_recip = 1.0 / (dists + 1e-8)
        weight = dist_recip / torch.sum(dist_recip, dim=2, keepdim=True)

        B, C, _ = x_s.shape
        idx_expanded = idx.unsqueeze(1).expand(B, C, -1, -1)
        x_s_exp = x_s.unsqueeze(2).expand(-1, -1, pos_d.shape[1], -1)
        interpolated = torch.gather(x_s_exp, 3, idx_expanded)
        interpolated = torch.sum(interpolated * weight.unsqueeze(1), dim=-1)

        return self.conv(torch.cat([interpolated, x_d], dim=1))
        
class PointTrackTransformer(nn.Module):
    class _BiasedAttnBlock(nn.Module):
        def __init__(self, feature_dim, num_heads, dim_feedforward, dropout, distance_tau):
            super().__init__()
            self.norm1 = nn.LayerNorm(feature_dim)
            self.attn = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(feature_dim)
            self.ffn = nn.Sequential(
                nn.Linear(feature_dim, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, feature_dim),
                nn.Dropout(dropout),
            )
            self.distance_tau = float(distance_tau)

        def forward(self, x, sorted_pos):
            x_norm = self.norm1(x)
            x_coords = sorted_pos[..., 0]
            x_coords = x_coords - x_coords.mean(dim=1, keepdim=True)
            x_scale = x_coords.std(dim=1, keepdim=True).clamp_min(1e-6)
            x_coords = x_coords / x_scale
            dist = torch.abs(x_coords.unsqueeze(2) - x_coords.unsqueeze(1))
            attn_bias = -dist / max(self.distance_tau, 1e-6)
            B, N, _ = x.shape
            attn_mask = attn_bias.unsqueeze(1).expand(-1, self.attn.num_heads, -1, -1).reshape(
                B * self.attn.num_heads, N, N
            )
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=False)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x

    def __init__(self, feature_dim=512, num_heads=8, num_layers=2, dim_feedforward=1024, distance_tau=0.5):
        """Initialize distance-biased transformer block for ordered point-track aggregation."""
        super().__init__()

        self.pos_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, feature_dim),
        )

        self.blocks = nn.ModuleList(
            [
                self._BiasedAttnBlock(
                    feature_dim=feature_dim,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=0.1,
                    distance_tau=distance_tau,
                )
                for _ in range(num_layers)
            ]
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, local_features, pos_raw):
        """Sort by x, encode positions, apply transformer, then restore original order."""
        B, C, N = local_features.shape

        x_coords = pos_raw[:, 0, :]
        sorted_x, sorted_indices = torch.sort(x_coords, dim=1)
        _ = sorted_x
        batch_indices = torch.arange(B).unsqueeze(1).expand(B, N).to(local_features.device)

        local_features_t = local_features.transpose(1, 2)
        sorted_features = local_features_t[batch_indices, sorted_indices, :]

        sorted_pos = pos_raw.transpose(1, 2)[batch_indices, sorted_indices, :]
        pos_enc = self.pos_mlp(sorted_pos.reshape(B * N, 2)).reshape(B, N, C)
        seq_input = sorted_features + pos_enc

        attended_features = seq_input
        for block in self.blocks:
            attended_features = block(attended_features, sorted_pos)

        _, inverse_indices = torch.sort(sorted_indices, dim=1)
        restored_global_features = attended_features[batch_indices, inverse_indices, :]

        concat_features = torch.cat([local_features_t, restored_global_features], dim=-1)
        final_features = self.fusion_mlp(concat_features.reshape(B * N, 2 * C)).reshape(B, N, C)

        return final_features.transpose(1, 2)


class ResidualChannelGate(nn.Module):
    def __init__(self, channels, reduction=4, init_alpha=0.1):
        """Weak residual channel gate for decoder skip features."""
        super().__init__()
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        gate = self.channel_mlp(x)
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        return x * (1.0 + alpha * gate)


class PhysicsKPConvNet(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES, dropout=Config.DROPOUT):
        """Initialize encoder-decoder KPConv network for point-wise classification."""
        super().__init__()
        self.K_neighbors = Config.K_NEIGHBORS
        self.downsample_strategy = getattr(Config, "DOWNSAMPLE_STRATEGY", "hybrid").lower()
        self.density_grid_size = getattr(Config, "DENSITY_GRID_SIZE", 32)
        self.density_aware_alpha = getattr(Config, "DENSITY_AWARE_ALPHA", 0.7)
        self.register_buffer("dist_scale", torch.tensor([Config.DIST_SCALE_X, Config.DIST_SCALE_Z]))

        self.start_conv = nn.Sequential(
            nn.Conv1d(Config.MODEL_INPUT_DIMS, 64, 1, bias=True),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.InstanceNorm1d(64),
            nn.LeakyReLU(0.2),
        )

        self.enc1 = MultiScaleFixedStripKPBlock(64, 64, radius=Config.INITIAL_RADIUS)
        self.enc2 = MultiScaleFixedStripKPBlock(
            64, 128, radius=Config.INITIAL_RADIUS * 2, downsample=True
        )
        self.enc3 = MultiScaleFixedStripKPBlock(
            128, 256, radius=Config.INITIAL_RADIUS * 4, downsample=True
        )
        self.enc4 = MultiScaleFixedStripKPBlock(
            256, 512, radius=Config.INITIAL_RADIUS * 8, downsample=True
        )

        self.track_transformer = PointTrackTransformer(feature_dim=512, num_heads=8, num_layers=2)

        self.skip_gate3 = ResidualChannelGate(256)
        self.skip_gate2 = ResidualChannelGate(128)
        self.skip_gate1 = ResidualChannelGate(64)

        self.up4 = PointInterpolateBlock(512, 256, 256)
        self.up3 = PointInterpolateBlock(256, 128, 128)
        self.up2 = PointInterpolateBlock(128, 64, 64)
        self.up1 = PointInterpolateBlock(64, 64, 64)

        self.cls_head = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, 1),
        )

    def get_multiscale_neighbors(self, q, s):
        """Compute four dilated KNN groups from one raw-coordinate neighbor list."""
        dilations = (1, 2, 3, 4)
        k_total = min(self.K_neighbors * max(dilations), s.shape[1])
        dist = torch.cdist(q, s)
        _, topk_idx = dist.topk(k_total, dim=-1, largest=False)

        target_total = self.K_neighbors * max(dilations)
        if k_total < target_total:
            pad = topk_idx[:, :, -1:].expand(-1, -1, target_total - k_total)
            topk_idx = torch.cat([topk_idx, pad], dim=-1)

        idx_groups = []
        for dilation in dilations:
            idx = topk_idx[:, :, ::dilation][:, :, : self.K_neighbors]
            if idx.shape[-1] < self.K_neighbors:
                pad = idx[:, :, -1:].expand(-1, -1, self.K_neighbors - idx.shape[-1])
                idx = torch.cat([idx, pad], dim=-1)
            idx_groups.append(idx)
        return tuple(idx_groups)

    def _select_downsample_idx(self, pos, stage):
        """Select downsample indices by configured strategy."""
        target = (pos.shape[1] + 1) // 2
        if self.downsample_strategy == "density_fps_stride":
            if stage == 2:
                return _density_aware_sampling(
                    pos,
                    target,
                    grid_size=self.density_grid_size,
                    alpha=self.density_aware_alpha,
                )
            if stage == 3:
                return _farthest_point_sampling(pos, target)
            return torch.arange(0, pos.shape[1], 2, device=pos.device, dtype=torch.long).unsqueeze(0).expand(
                pos.shape[0], -1
            )
        if self.downsample_strategy == "fps":
            return _farthest_point_sampling(pos, target)
        if self.downsample_strategy == "stride":
            return torch.arange(0, pos.shape[1], 2, device=pos.device, dtype=torch.long).unsqueeze(0).expand(
                pos.shape[0], -1
            )
        if self.downsample_strategy == "hybrid":
            if stage == 2:
                return _farthest_point_sampling(pos, target)
            return torch.arange(0, pos.shape[1], 2, device=pos.device, dtype=torch.long).unsqueeze(0).expand(
                pos.shape[0], -1
            )

    def forward(self, pos, features_local):
        """Run end-to-end encoding, bottleneck aggregation, and decoding."""
        B, N, _ = pos.shape
        pos_eff = pos

        x_raw = pos[..., 0:1]
        z_raw = pos[..., 1:2]
        x_std = x_raw.std(dim=1, keepdim=True) + 1e-6
        z_std = z_raw.std(dim=1, keepdim=True) + 1e-6

        pos_norm = torch.cat([x_raw / x_std, z_raw / z_std], dim=-1)
        local_feats = features_local[..., : Config.LOCAL_FEATURE_DIMS]
        feat = torch.cat([pos_norm, local_feats], dim=-1).transpose(1, 2)

        x_in = self.start_conv(feat)

        n1_list = self.get_multiscale_neighbors(pos_eff, pos_eff)
        x1 = self.enc1(x_in, pos_eff, pos_eff, n1_list)

        idx2 = self._select_downsample_idx(pos_eff, stage=2)
        pos2 = _gather_points(pos_eff, idx2)
        n2_list = self.get_multiscale_neighbors(pos2, pos_eff)
        x2 = self.enc2(x1, pos2, pos_eff, n2_list, q_idx=idx2)

        idx3 = self._select_downsample_idx(pos2, stage=3)
        pos3 = _gather_points(pos2, idx3)
        n3_list = self.get_multiscale_neighbors(pos3, pos2)
        x3 = self.enc3(x2, pos3, pos2, n3_list, q_idx=idx3)

        idx4 = self._select_downsample_idx(pos3, stage=4)
        pos4 = _gather_points(pos3, idx4)
        n4_list = self.get_multiscale_neighbors(pos4, pos3)
        x4 = self.enc4(x3, pos4, pos3, n4_list, q_idx=idx4)

        raw_pos4 = pos4.transpose(1, 2)
        x4_global = self.track_transformer(x4, raw_pos4)

        x3_filtered = self.skip_gate3(x3)
        x2_filtered = self.skip_gate2(x2)
        x1_filtered = self.skip_gate1(x1)

        d3 = self.up4(x4_global, pos4, x3_filtered, pos3)
        d2 = self.up3(d3, pos3, x2_filtered, pos2)
        d1 = self.up2(d2, pos2, x1_filtered, pos_eff)
        d0 = self.up1(d1, pos_eff, x_in, pos_eff)

        logits = self.cls_head(d0)
        return logits


# Backward-compatible names for older training scripts that import these classes.
PCAOrientedDeformableStripKPConv = FixedStripKPConv
MultiScalePCADeformableKPBlock = MultiScaleFixedStripKPBlock
