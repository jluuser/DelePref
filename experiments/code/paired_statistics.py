"""Paired, hierarchical inference for fixed-denominator endpoint bounds.

Rows contain wt_id, group_id, method, lower, upper. Average repeated variants,
seeds, etc. inside each WT, then average WTs inside each homology group. Supply
the full expected WT roster, including failed WTs with explicit endpoint bounds.
Undefined G is rejected: callers must explicitly choose valid bounds or report
the endpoint as undefined, never silently exclude a failed WT.
"""
import math
import random


def _bounds(lower, upper):
    if lower is None or upper is None or not all(math.isfinite(x) for x in (lower, upper)) or lower > upper:
        raise ValueError("undefined or invalid endpoint bounds (including undefined G)")
    return float(lower), float(upper)


def regret_interval(candidate_bounds, selected):
    """Protocol conservative bounds; these need not be sharp (even for one candidate)."""
    bounds = [_bounds(*b) for b in candidate_bounds]
    if not bounds or type(selected) is not int or not 0 <= selected < len(bounds):
        raise ValueError("selected candidate must be present")
    lo, hi = bounds[selected]
    return max(0., max(b[0] for b in bounds) - hi), max(0., max(b[1] for b in bounds) - lo)


def paired_group_bounds(rows, method_x, method_y, expected_wts):
    """Return paired X-Y bounds per group; expected_wts maps WT to group."""
    if not expected_wts or method_x == method_y:
        raise ValueError("nonempty WT roster and distinct methods required")
    values = {(wt, m): [] for wt in expected_wts for m in (method_x, method_y)}
    for r in rows:
        if r["method"] not in (method_x, method_y):
            continue
        wt = r["wt_id"]
        if wt not in expected_wts or r["group_id"] != expected_wts[wt]:
            raise ValueError("WT/group differs from frozen roster")
        values[(wt, r["method"])].append(_bounds(r["lower"], r["upper"]))
    grouped = {}
    for wt in sorted(expected_wts):
        means = []
        for method in (method_x, method_y):
            v = sorted(values[(wt, method)])
            if not v:
                raise ValueError(f"missing paired WT {wt}: {method}; provide failure bounds")
            means.append(tuple(math.fsum(b[i] for b in v) / len(v) for i in (0, 1)))
        x, y = means
        grouped.setdefault(expected_wts[wt], []).append((x[0] - y[1], x[1] - y[0]))
    return {g: tuple(math.fsum(b[i] for b in grouped[g]) / len(grouped[g]) for i in (0, 1))
            for g in sorted(grouped)}


def _quantile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    a = int(index)
    b = min(a + 1, len(values) - 1)
    return values[a] + (values[b] - values[a]) * (index - a)


def paired_bootstrap(rows, method_x, method_y, expected_wts, *, replicates=10000,
                     seed=20261013, confidence=.95):
    if type(replicates) is not int or replicates < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    groups = paired_group_bounds(rows, method_x, method_y, expected_wts)
    bounds = list(groups.values())
    n = len(bounds)
    rng = random.Random(seed)
    lowers, uppers = [], []
    for _ in range(replicates):
        sample = [bounds[rng.randrange(n)] for _ in range(n)]
        lowers.append(math.fsum(b[0] for b in sample) / n)
        uppers.append(math.fsum(b[1] for b in sample) / n)
    alpha = (1 - confidence) / 2
    return {"difference_bounds": tuple(math.fsum(b[i] for b in bounds) / n for i in (0, 1)),
            "confidence_interval": (_quantile(lowers, alpha), _quantile(uppers, 1 - alpha)),
            "n_wts": len(expected_wts), "n_groups": n, "replicates": replicates,
            "seed": seed, "confidence": confidence, "group_bounds": groups}
