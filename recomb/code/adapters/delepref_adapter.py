"""Pure deletion identity; intervals are closed, zero-based, maximal spans."""
import numpy as np


def checked_mask(mask, length):
    array = np.asarray(mask)
    if array.shape != (length,) or not np.all(np.isin(array, [0, 1])):
        raise ValueError("mask must have WT length and contain only zero/one")
    return array.astype(bool)


def retained_mapping(wt, edited, mask):
    mask = checked_mask(mask, len(wt))
    kept = np.flatnonzero(~mask)
    if "".join(wt[i] for i in kept) != edited:
        raise ValueError("mask does not reconstruct edited sequence")
    return kept


def intervals_from_deletion_mask(mask):
    mask = checked_mask(mask, len(mask))
    padded = np.r_[False, mask, False].astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def deletion_mask_from_intervals(length, intervals):
    mask = np.zeros(length, dtype=bool)
    previous = -2
    for start, end in intervals:
        if not (0 <= start <= end < length) or start <= previous + 1:
            raise ValueError("intervals must be ordered, separated maximal closed spans")
        mask[start:end + 1] = True
        previous = end
    return mask


def strict_delete_count(length, budget):
    if length < 2 or not 0 < budget < 1:
        raise ValueError("strict budget requires length>=2 and 0<budget<1")
    return max(1, min(length - 1, round(budget * length)))


def morphology(mask):
    spans = intervals_from_deletion_mask(mask)
    lengths = [b - a + 1 for a, b in spans]
    return {"intervals": spans, "num_spans": len(spans),
            "longest_span": max(lengths, default=0),
            "span_lengths": sorted(lengths),
            "terminal": bool(mask[0] or mask[-1]), "deleted_count": int(sum(mask))}
