"""WT-only 60-column relationship inputs; pair indices refer to edited positions."""
import numpy as np

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def pair_features(wt_sequence, wt_coords, kept, i, j):
    coords = np.asarray(wt_coords, dtype=float)
    kept = np.asarray(kept, dtype=int)
    i, j = np.asarray(i, dtype=int), np.asarray(j, dtype=int)
    L, K = len(wt_sequence), len(kept)
    if coords.shape != (L, 3) or K == 0 or np.any(np.diff(kept) <= 0) or np.any(kept < 0) or np.any(kept >= L):
        raise ValueError("invalid WT coordinates or retained map")
    if i.shape != j.shape or np.any(i < 0) or np.any(j < 0) or np.any(i >= K) or np.any(j >= K):
        raise ValueError("pair indices must index retained arrays")
    deleted = np.setdiff1d(np.arange(L), kept)
    d = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]) >= 3
    contacts = (d < 8) & sep
    E = (d < 15) & sep
    degree = contacts.sum(1)
    lost = contacts[:, deleted].sum(1) / np.maximum(degree, 1)
    junction = np.flatnonzero(np.diff(kept) > 1)
    junction_nodes = np.unique(np.r_[junction, junction + 1])
    def nearest(positions, targets):
        return np.minimum(np.abs(positions[:, None] - targets).min(1), 32) / 32 if len(targets) else np.ones(len(positions))
    nd = nearest(kept, deleted)
    nj = nearest(np.arange(K), junction_nodes)
    spans = np.split(deleted, np.flatnonzero(np.diff(deleted) != 1) + 1) if len(deleted) else []
    coverage = E[np.ix_(kept, kept)].sum() / E.sum() if E.sum() else 0.
    onehot = np.eye(20)[[AA_ORDER.index(a) for a in wt_sequence]]
    def oriented(a, b):
        x, y = kept[a], kept[b]
        columns = [d[x, y] / 15, np.abs(x-y)/L, np.abs(a-b)/K,
                   x/max(L-1, 1), y/max(L-1, 1), a/max(K-1, 1), b/max(K-1, 1),
                   nd[a], nd[b], nj[a], nj[b], degree[x]/64, degree[y]/64, lost[x], lost[y]]
        columns += [np.full(len(a), v) for v in (len(deleted)/L, len(spans)/8,
                    max(map(len, spans), default=0)/L, coverage, L/1022)]
        out = np.column_stack(columns + [onehot[x], onehot[y]]).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("nonfinite reference feature")
        return out
    return oriented(i, j), oriented(j, i)
