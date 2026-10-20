"""Stable evaluation of the unchanged TM missing-coordinate interval formula."""
import math


def stable_interval(tm):
    result=dict(tm)
    lo=float(tm['lower']);unknown=int(tm['unknown_retained_residues']);length=int(tm['full_wt_length'])
    upper=lo+unknown/length
    if not math.isfinite(lo) or unknown<0 or length<1 or abs(upper-float(tm['upper']))>1e-12:
        raise ValueError('TM interval mismatch exceeds numerical roundoff; do not silently clamp')
    result.update(upper=upper,original_floating_upper=tm['upper'],
                  interval_roundoff_correction=upper-float(tm['upper']),
                  interval_computation='lower + integer_unknown_count / full_WT_length; algebraically unchanged, avoids add-large-then-subtract cancellation')
    return result
