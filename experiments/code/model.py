"""Identical-slot response heads with explicit information controls."""
import torch
from torch import nn


class PairResponseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(1152), nn.Linear(1152, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(444, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 5))

    def forward(self, wt, edited, kept, i, j, gij, gji, arm="C"):
        if arm not in "SABCD" or len(arm) != 1:
            raise ValueError("unknown control arm")
        if len(kept) == 0 or edited.shape != (len(kept), 1152) or wt.shape[1] != 1152:
            raise ValueError("invalid residue embedding shape")
        if gij.shape != (len(i), 60) or gji.shape != gij.shape:
            raise ValueError("structural features must have width 60")
        zero = wt.new_zeros((len(kept), 64))
        w = self.project(wt)[kept] if arm in "ABC" else zero
        m = self.project(edited) if arm in "BCD" else zero
        response = m - w
        pool = m if arm == "D" else response if arm in "BC" else zero
        local = m if arm == "D" else response if arm == "C" else zero
        mean = pool.mean(0).expand(len(i), -1)
        std = pool.std(0, unbiased=False).expand(len(i), -1)
        def oriented(a, b, g):
            return self.head(torch.cat((w[a], w[b], mean, std, local[a], local[b], g), -1))
        return .5 * (oriented(i, j, gij) + oriented(j, i, gji))
