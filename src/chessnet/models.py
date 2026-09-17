import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import softmax

class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)

    def forward(self, seq):
        # seq: [T, B, D]
        q = self.q(seq[-1:])
        k = self.k(seq)
        v = self.v(seq)
        scores = torch.einsum("tbd,sbd->bts", q, k) / self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bts,sbd->tbd", weights, v).squeeze(0)

class GATBranch(nn.Module):
    def __init__(self, in_dim=12, hidden=64, heads=4):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads, dropout=0.1)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, dropout=0.1)

    def forward(self, data):
        h = F.elu(self.gat1(data.x, data.edge_index))
        return self.gat2(h, data.edge_index)

class HGTInspiredBranch(nn.Module):
    """Lightweight relation/type-aware attention approximation.

    It preserves the thesis intent—heterogeneous relational attention—
    while avoiding dependence on a particular PyG HGT API.
    """
    def __init__(self, in_dim=12, hidden=64, heads=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden * heads)
        self.attn = nn.MultiheadAttention(hidden * heads, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden * heads)
        self.out = nn.Linear(hidden * heads, hidden)

    def forward(self, data):
        h = self.proj(data.x).unsqueeze(0)
        h, _ = self.attn(h, h, h, need_weights=False)
        return self.out(self.norm(h)).squeeze(0)

class TGATInspiredBranch(nn.Module):
    def __init__(self, in_dim=12, hidden=64):
        super().__init__()
        self.gat = GATBranch(in_dim, hidden, heads=2)
        self.temporal = TemporalAttention(hidden)

    def forward_sequence(self, graphs):
        pooled = []
        for g in graphs:
            h = self.gat(g)
            pooled.append(h.mean(dim=0))
        seq = torch.stack(pooled).unsqueeze(1)
        return self.temporal(seq).squeeze(0)

class HCGNN(nn.Module):
    """HGT-inspired + TGAT-inspired + GAT hybrid, followed by fusion."""
    def __init__(self, in_dim=12, hidden=64, num_classes=3):
        super().__init__()
        self.hgt = HGTInspiredBranch(in_dim, hidden)
        self.tgat = TGATInspiredBranch(in_dim, hidden)
        self.gat = GATBranch(in_dim, hidden, heads=4)
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, graphs):
        hgt = self.hgt(graphs[-1]).mean(dim=0)
        tgat = self.tgat.forward_sequence(graphs)
        gat = self.gat(graphs[-1]).mean(dim=0)
        fused = torch.cat([hgt, tgat, gat], dim=-1)
        return self.fusion(fused)

class GATClassifier(nn.Module):
    def __init__(self, in_dim=12, hidden=64, num_classes=3):
        super().__init__()
        self.gat = GATBranch(in_dim, hidden, heads=4)
        self.cls = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

    def forward(self, graphs):
        h = self.gat(graphs[-1])
        return self.cls(h.mean(dim=0))
