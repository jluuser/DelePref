"""Fixed per-position gap guidance, distinct from natural event co-occurrence."""
import math
import numpy as np
import delepref_unlimited_planning as planning


def frequencies(profile, length):
    counts = np.zeros(length, dtype=np.int64)
    denominators = np.zeros(length, dtype=np.int64)
    events = {h['target']: h['events'] for h in planning.natural_events(profile, length)}
    for representative in profile['representatives']:
        projection = representative['projected']
        if len(projection) != length:
            raise ValueError('projection length differs from reviewed WT')
        gap = np.zeros(length, dtype=bool)
        for a, b in events.get(representative['target'], ()):
            gap[a:b+1] = True
        observed = np.array([c in planning.AA for c in projection])
        counts += gap
        denominators += observed | gap
    rates = np.divide(counts, denominators, out=np.zeros(length), where=denominators > 0)
    return rates, counts, denominators


def proposals(logits, beta, profile):
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    rates, counts, denominators = frequencies(profile, len(values))
    positive = np.flatnonzero(rates > 0)
    ordered = positive[np.lexsort((positive, -values[positive], -rates[positive]))]
    d = planning.delete_count(len(values), beta)
    seen, result, attempts = set(), [], []
    for numerator in (1, 2, 3, 4):
        n = min(len(ordered), math.ceil(d*numerator/4))
        if not n:
            continue
        forced = np.zeros(len(values), dtype=bool); forced[ordered[:n]] = True
        spans, _ = planning.complete(values, beta, must_delete=forced)
        duplicate = spans in seen
        attempts.append(dict(quarter=numerator, forced_positions=int(n), status='duplicate' if duplicate else 'accepted'))
        if not duplicate:
            seen.add(spans)
            result.append(dict(spans=spans, seed_spans=planning.intervals(forced),
                kind='gap_frequency_guided', policy_energy=float(values[planning.mask_for(len(values), spans)].mean())))
    return result, dict(intervention='per-position reliable gap frequency; fixed quarter-budget prefixes plus Policy completion',
        positive_positions=len(ordered), attempts=attempts, source_representatives=len(profile['representatives']),
        rates=rates.tolist(), gap_counts=counts.tolist(), observed_denominators=denominators.tolist(),
        same_event_cooccurrence_used=False, not_observed_natural_event_bundles=True)
