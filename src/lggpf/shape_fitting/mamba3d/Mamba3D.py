"""Standalone Mamba3D point cloud classification model.

Self-contained implementation featuring bidirectional State Space Models (BiMamba),
local geometry aggregation (LGA), and pure PyTorch fallbacks for FPS and KNN operations.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    try:
        from timm.layers import DropPath, trunc_normal_
    except ImportError:
        def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
            with torch.no_grad():
                return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

        class DropPath(nn.Module):
            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = drop_prob

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
                random_tensor.floor_()
                return x.div(keep_prob) * random_tensor

from .bimamba_ssm.modules.mamba_simple import Mamba
from .rope import VisionRotaryEmbeddingFast
from .z_order import get_z_values


# ---------------------------------------------------------------------------
# Sampling and Neighbor Querying (with pure PyTorch fallback)
# ---------------------------------------------------------------------------

try:
    from pointnet2_ops import pointnet2_utils
    _HAS_POINTNET2 = True
except ImportError:
    pointnet2_utils = None
    _HAS_POINTNET2 = False

try:
    from knn_cuda import KNN as _CudaKNN
    _HAS_CUDA_KNN = True
except ImportError:
    _CudaKNN = None
    _HAS_CUDA_KNN = False


def _furthest_point_sample_pytorch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Furthest point sampling (FPS) in pure PyTorch."""
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def fps(data: torch.Tensor, number: int) -> torch.Tensor:
    """Furthest Point Sampling selecting `number` representative points."""
    if _HAS_POINTNET2 and pointnet2_utils is not None and data.is_cuda:
        try:
            fps_idx = pointnet2_utils.furthest_point_sample(data.contiguous(), number)
            fps_data = pointnet2_utils.gather_operation(
                data.transpose(1, 2).contiguous(), fps_idx
            ).transpose(1, 2).contiguous()
            return fps_data
        except Exception:
            pass

    fps_idx = _furthest_point_sample_pytorch(data, number)
    B = data.shape[0]
    batch_indices = torch.arange(B, device=data.device)[:, None]
    return data[batch_indices, fps_idx]


class KNN(nn.Module):
    """K-Nearest Neighbors query module with optional CUDA extension and PyTorch fallback."""

    def __init__(self, k: int, transpose_mode: bool = True):
        super().__init__()
        self.k = k
        self.transpose_mode = transpose_mode
        self._cuda_knn = _CudaKNN(k=k, transpose_mode=transpose_mode) if _HAS_CUDA_KNN else None

    def forward(self, ref: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cuda_knn is not None and ref.is_cuda:
            try:
                return self._cuda_knn(ref.contiguous(), query.contiguous())
            except Exception:
                pass

        r = ref if self.transpose_mode else ref.transpose(1, 2)
        q = query if self.transpose_mode else query.transpose(1, 2)
        dist = torch.cdist(q, r)
        dists, indices = torch.topk(dist, self.k, dim=-1, largest=False, sorted=True)
        return dists, indices


# ---------------------------------------------------------------------------
# Feature Extraction and Geometry Modules
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Mini-PointNet embedding module for point patch groups."""

    def __init__(self, encoder_channel: int):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    """FPS + KNN patch grouping module."""

    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group)
        _, idx = self.knn(xyz, center)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class GroupFeature(nn.Module):
    """KNN grouping for patch features."""

    def __init__(self, group_size: int):
        super().__init__()
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_points, _ = xyz.shape
        c = feat.shape[-1]
        center = xyz
        _, idx = self.knn(xyz, xyz)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous()
        neighborhood_feat = feat.contiguous().view(-1, c)[idx, :]
        neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size, c).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, neighborhood_feat


class K_Norm(nn.Module):
    """Local Geometry Aggregation (LGA) normalization module."""

    def __init__(self, out_dim: int, k_group_size: int, alpha: float, beta: float):
        super().__init__()
        self.group_feat = GroupFeature(k_group_size)
        self.affine_alpha_feat = nn.Parameter(torch.ones([1, 1, 1, out_dim]))
        self.affine_beta_feat = nn.Parameter(torch.zeros([1, 1, 1, out_dim]))

    def forward(self, lc_xyz: torch.Tensor, lc_x: torch.Tensor) -> torch.Tensor:
        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x)

        mean_x = lc_x.unsqueeze(dim=-2)
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)

        b, g, k, _ = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(b, g, 1, -1).repeat(1, 1, k, 1)], dim=-1)
        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat
        knn_x_w = knn_x.permute(0, 3, 1, 2)
        return knn_x_w


class K_Pool(nn.Module):
    """Exponential-weighted pooling over local neighbors."""

    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w: torch.Tensor) -> torch.Tensor:
        e_x = torch.exp(knn_x_w)
        up = (knn_x_w * e_x).mean(-1)
        down = e_x.mean(-1)
        return torch.div(up, down)


class Post_ShareMLP(nn.Module):
    """Shared 1D convolution MLP."""

    def __init__(self, in_dim: int, out_dim: int, permute: bool = True):
        super().__init__()
        self.share_mlp = nn.Conv1d(in_dim, out_dim, 1)
        self.permute = permute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.permute:
            return self.share_mlp(x).permute(0, 2, 1)
        return self.share_mlp(x)


class LNPBlock(nn.Module):
    """LGA -> Pool -> MLP -> Act local geometric feature block."""

    def __init__(
        self,
        lga_out_dim: int,
        k_group_size: int,
        alpha: float,
        beta: float,
        mlp_in_dim: int,
        mlp_out_dim: int,
        num_group: int = 128,
        act_layer: type[nn.Module] = nn.SiLU,
        drop_path: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.num_group = num_group
        self.lga_out_dim = lga_out_dim
        self.lga = K_Norm(self.lga_out_dim, k_group_size, alpha, beta)
        self.kpool = K_Pool()
        self.mlp = Post_ShareMLP(mlp_in_dim, mlp_out_dim)
        self.pre_norm_ft = norm_layer(self.lga_out_dim)
        self.act = act_layer()

    def forward(self, center: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        b, g, c = feat.shape
        cls_token = feat[:, 0, :].view(b, 1, c)
        feat = feat[:, 1:, :]

        lc_x_w = self.lga(center, feat)
        lc_x_w = self.kpool(lc_x_w)
        lc_x_w = self.pre_norm_ft(lc_x_w.permute(0, 2, 1))
        lc_x = self.mlp(lc_x_w.permute(0, 2, 1))
        lc_x = self.act(lc_x)
        return torch.cat((cls_token, lc_x), dim=1)


class Mamba3DBlock(nn.Module):
    """Mamba3D block combining local geometric aggregation and bidirectional SSM."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.SiLU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        k_group_size: int = 8,
        alpha: float = 100.0,
        beta: float = 1000.0,
        num_group: int = 128,
        num_heads: int = 6,
        bimamba_type: str = "v4",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.num_heads = num_heads

        self.lfa = LNPBlock(
            lga_out_dim=dim * 2,
            k_group_size=self.k_group_size,
            alpha=alpha,
            beta=beta,
            mlp_in_dim=dim * 2,
            mlp_out_dim=dim,
            num_group=self.num_group,
            act_layer=act_layer,
            drop_path=drop_path,
            norm_layer=norm_layer,
        )
        self.mixer = Mamba(dim, bimamba_type=bimamba_type)

    def forward(self, center: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.lfa(center, self.norm1(x)))
        x = x + self.drop_path(self.mixer(self.norm2(x)))
        return x


class Mamba3DEncoder(nn.Module):
    """Stack of Mamba3D blocks."""

    def __init__(
        self,
        k_group_size: int = 8,
        embed_dim: int = 768,
        depth: int = 4,
        drop_path_rate: float | list[float] = 0.0,
        num_group: int = 128,
        num_heads: int = 6,
        bimamba_type: str = "v4",
    ):
        super().__init__()
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.num_heads = num_heads

        dpr = drop_path_rate if isinstance(drop_path_rate, list) else [drop_path_rate] * depth
        self.blocks = nn.ModuleList([
            Mamba3DBlock(
                dim=embed_dim,
                k_group_size=self.k_group_size,
                drop_path=dpr[i],
                num_group=self.num_group,
                num_heads=self.num_heads,
                bimamba_type=bimamba_type,
            )
            for i in range(depth)
        ])

    def forward(self, center: torch.Tensor, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(center, x + pos)
        return x


# ---------------------------------------------------------------------------
# Full Mamba3D Classification Architecture
# ---------------------------------------------------------------------------

class Mamba3D(nn.Module):
    """Complete Mamba3D point cloud classification model."""

    def __init__(self, config: Any = None, **kwargs):
        super().__init__()
        cfg = {}
        if config is not None:
            if isinstance(config, Mapping):
                cfg.update(config)
            elif hasattr(config, "__dict__"):
                cfg.update(vars(config))
        cfg.update(kwargs)

        self.trans_dim = int(cfg.get("trans_dim", 384))
        self.depth = int(cfg.get("depth", 12))
        self.drop_path_rate = float(cfg.get("drop_path_rate", 0.2))
        self.cls_dim = int(cfg.get("cls_dim", 3))
        self.num_heads = int(cfg.get("num_heads", 6))
        self.group_size = int(cfg.get("group_size", 32))
        self.num_group = int(cfg.get("num_group", 128))
        self.encoder_dims = int(cfg.get("encoder_dims", 384))
        self.bimamba_type = str(cfg.get("bimamba_type", "v4"))
        self.k_group_size = int(cfg.get("center_local_k", 4))
        self.ordering = bool(cfg.get("ordering", False))
        self.label_smooth = float(cfg.get("label_smooth", 0.0))

        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = Mamba3DEncoder(
            embed_dim=self.trans_dim,
            k_group_size=self.k_group_size,
            depth=self.depth,
            drop_path_rate=dpr,
            num_group=self.num_group,
            num_heads=self.num_heads,
            bimamba_type=self.bimamba_type,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.cls_dim),
        )

        self.loss_ce = nn.CrossEntropyLoss(label_smoothing=self.label_smooth)

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.cls_pos, std=0.02)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """Forward pass. pts: FloatTensor of shape (B, N, 3)."""
        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        x = self.blocks(center, x, pos)
        x = self.norm(x)

        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0] + x[:, 1:].mean(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret
