from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class CNNBaseline(nn.Module):
    def __init__(self, nodes: int, classes: int, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(nodes, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.head(h)


def make_path_adj(nodes: int, path: list[int]) -> torch.Tensor:
    path = [idx for idx in path if idx < nodes]
    adj = torch.eye(nodes, dtype=torch.float32)
    for a, b in zip(path[:-1], path[1:]):
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    for center in path:
        for neighbor in path:
            if abs(center - neighbor) <= 2 and center != neighbor:
                adj[center, neighbor] = max(float(adj[center, neighbor]), 0.25)
    rowsum = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj / rowsum


class PathGraphLayer(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.proj = nn.Linear(filters, filters)
        self.norm = nn.LayerNorm(filters)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("ij,bjf->bif", adj, h)
        return self.act(self.norm(self.proj(mixed) + h))


class PathGraphAttentionLayer(nn.Module):
    def __init__(self, filters: int, heads: int = 2, dropout: float = 0.10):
        super().__init__()
        self.heads = heads
        self.head_dim = math.ceil(filters / heads)
        inner = self.heads * self.head_dim
        self.q_proj = nn.Linear(filters, inner)
        self.k_proj = nn.Linear(filters, inner)
        self.v_proj = nn.Linear(filters, inner)
        self.out_proj = nn.Linear(inner, filters)
        self.norm = nn.LayerNorm(filters)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.last_attention = None

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = h.shape
        q = self.q_proj(h).reshape(batch, nodes, self.heads, self.head_dim)
        k = self.k_proj(h).reshape(batch, nodes, self.heads, self.head_dim)
        v = self.v_proj(h).reshape(batch, nodes, self.heads, self.head_dim)
        scores = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.head_dim)
        mask = adj.gt(0).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        self.last_attention = weights.detach()
        weights = self.dropout(weights)
        mixed = torch.einsum("bhij,bjhd->bihd", weights, v).reshape(batch, nodes, -1)
        return self.act(self.norm(self.out_proj(mixed) + h))


class PropagationPathGCN(nn.Module):
    def __init__(
        self,
        nodes: int,
        faults: tuple[int, ...],
        paths: dict[int, list[int]],
        filters: int = 24,
        graph_layers: int = 2,
    ):
        super().__init__()
        self.nodes = nodes
        self.faults = tuple(faults)
        self.filters = filters
        self.graph_layers = graph_layers
        self.temporal = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(filters),
            nn.Conv1d(filters, filters, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gc_layers = nn.ModuleList(PathGraphLayer(filters) for _ in range(graph_layers))
        self.normal_head = nn.Sequential(nn.LayerNorm(filters), nn.Linear(filters, 1))
        self.fault_heads = nn.ModuleDict()
        self.path_indices = {}
        for fault in self.faults:
            path = [idx for idx in paths[fault] if idx < nodes]
            self.path_indices[fault] = path
            feature_size = len(path) * filters * (graph_layers + 1)
            self.fault_heads[str(fault)] = nn.Sequential(
                nn.LayerNorm(feature_size),
                nn.Linear(feature_size, max(16, filters)),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(max(16, filters), 1),
            )
            self.register_buffer(f"adj_{fault}", make_path_adj(nodes, path))

    def temporal_encode(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, nodes = x.shape
        h = x.permute(0, 2, 1).reshape(batch * nodes, 1, time)
        h = self.temporal(h).squeeze(-1)
        return h.reshape(batch, nodes, self.filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tc = self.temporal_encode(x)
        logits = [self.normal_head(tc.mean(dim=1))]
        for fault in self.faults:
            h = tc
            branch_features = [tc[:, self.path_indices[fault], :]]
            adj = getattr(self, f"adj_{fault}")
            for layer in self.gc_layers:
                h = layer(h, adj)
                branch_features.append(h[:, self.path_indices[fault], :])
            flat = torch.cat([feat.flatten(1) for feat in branch_features], dim=1)
            logits.append(self.fault_heads[str(fault)](flat))
        return torch.cat(logits, dim=1)

    @torch.no_grad()
    def path_contributions(self, x: torch.Tensor, fault_label: int) -> np.ndarray:
        fault = self.faults[fault_label - 1]
        tc = self.temporal_encode(x)
        h = tc
        features = [tc[:, self.path_indices[fault], :]]
        adj = getattr(self, f"adj_{fault}")
        for layer in self.gc_layers:
            h = layer(h, adj)
            features.append(h[:, self.path_indices[fault], :])
        activation = torch.stack([feat.abs().mean(dim=(0, 2)) for feat in features]).mean(dim=0)
        return activation.detach().cpu().numpy()


class PropagationPathGAT(nn.Module):
    def __init__(
        self,
        nodes: int,
        faults: tuple[int, ...],
        paths: dict[int, list[int]],
        filters: int = 24,
        graph_layers: int = 2,
        heads: int = 2,
    ):
        super().__init__()
        self.nodes = nodes
        self.faults = tuple(faults)
        self.filters = filters
        self.graph_layers = graph_layers
        self.temporal = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(filters),
            nn.Conv1d(filters, filters, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gat_layers = nn.ModuleList(PathGraphAttentionLayer(filters, heads=heads) for _ in range(graph_layers))
        self.normal_head = nn.Sequential(nn.LayerNorm(filters), nn.Linear(filters, 1))
        self.fault_heads = nn.ModuleDict()
        self.path_indices = {}
        for fault in self.faults:
            path = [idx for idx in paths[fault] if idx < nodes]
            self.path_indices[fault] = path
            feature_size = len(path) * filters * (graph_layers + 1)
            self.fault_heads[str(fault)] = nn.Sequential(
                nn.LayerNorm(feature_size),
                nn.Linear(feature_size, max(16, filters)),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(max(16, filters), 1),
            )
            self.register_buffer(f"adj_{fault}", make_path_adj(nodes, path))

    def temporal_encode(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, nodes = x.shape
        h = x.permute(0, 2, 1).reshape(batch * nodes, 1, time)
        h = self.temporal(h).squeeze(-1)
        return h.reshape(batch, nodes, self.filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tc = self.temporal_encode(x)
        logits = [self.normal_head(tc.mean(dim=1))]
        for fault in self.faults:
            h = tc
            branch_features = [tc[:, self.path_indices[fault], :]]
            adj = getattr(self, f"adj_{fault}")
            for layer in self.gat_layers:
                h = layer(h, adj)
                branch_features.append(h[:, self.path_indices[fault], :])
            flat = torch.cat([feat.flatten(1) for feat in branch_features], dim=1)
            logits.append(self.fault_heads[str(fault)](flat))
        return torch.cat(logits, dim=1)

    @torch.no_grad()
    def edge_attention_summary(self, x: torch.Tensor, fault_label: int) -> list[dict[str, float | int]]:
        fault = self.faults[fault_label - 1]
        tc = self.temporal_encode(x)
        h = tc
        adj = getattr(self, f"adj_{fault}")
        rows = []
        for layer_idx, layer in enumerate(self.gat_layers, start=1):
            h = layer(h, adj)
            weights = layer.last_attention
            if weights is None:
                continue
            avg = weights.mean(dim=(0, 1)).detach().cpu()
            for src, dst in zip(self.path_indices[fault][:-1], self.path_indices[fault][1:]):
                rows.append(
                    {
                        "layer": int(layer_idx),
                        "source_index": int(src),
                        "target_index": int(dst),
                        "attention_forward": float(avg[dst, src]),
                        "attention_reverse": float(avg[src, dst]),
                        "attention_mean": float((avg[dst, src] + avg[src, dst]) / 2),
                    }
                )
        return rows
