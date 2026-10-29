"""Validate manuscript evidence from frozen per-target records.

The calculations here start from previously evaluated predictions. They do
not generate candidates, run folding models, or independently verify coordinates.
"""
from collections import defaultdict
import json
import math
import statistics
import zipfile

import numpy as np


def _read_json(archive, name):
    return json.loads(archive.read(name).decode('utf-8-sig'))


def _paired_effect(values, components, seed, replicates):
    """Reproduce the recorded component-paired bootstrap, including ties."""
    ids = sorted(values)
    labels = [components[pid] for pid in ids]
    unique = sorted(set(labels))
    data = np.asarray([values[pid] for pid in ids], dtype=float)
    if not np.isfinite(data).all():
        raise ValueError('Non-finite paired difference')
    sums = np.array([data[np.array(labels) == label].sum() for label in unique])
    counts = np.array([labels.count(label) for label in unique])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(unique), size=(replicates, len(unique)))
    boot = sums[indices].sum(1) / counts[indices].sum(1)
    return dict(N=len(ids), components=len(unique), mean_difference=float(data.mean()),
                joint_CI97_5=np.quantile(boot, [.0125, .9875]).tolist(),
                improved=int((data > 1e-8).sum()), degraded=int((data < -1e-8).sum()),
                unchanged=int((np.abs(data) <= 1e-8).sum()))


def _check_close(actual, expected, context):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('{} disagrees with frozen record: {} vs {}'.format(context, actual, expected))


def validate(path):
    """Recalculate means and paired intervals from frozen per-WT records."""
    with zipfile.ZipFile(path) as archive:
        protocol = _read_json(archive, 'completion/statistical_protocol.json')
        paired = _read_json(archive, 'completion/paired_per_WT_and_repeat.json')
        recorded = _read_json(archive, 'completion/paired_comparisons.json')
        raygun = _read_json(archive, 'raygun/strict_recalculation.json')

    grouped = defaultdict(dict)
    for row in paired:
        key = (row['endpoint'], row['protein_id'])
        rep = row['replicate']
        if rep in grouped[key]:
            raise ValueError('Duplicate WT/endpoint/control draw')
        grouped[key][rep] = row
    if len(paired) != 500 or any(set(rows) != set(range(5)) for rows in grouped.values()):
        raise ValueError('Expected five draws for each WT and endpoint')
    by_endpoint = defaultdict(dict)
    for (endpoint, pid), rows in grouped.items():
        by_endpoint[endpoint][pid] = {
            'lower': statistics.fmean(rows[i]['difference_lower'] for i in range(5)),
            'upper': statistics.fmean(rows[i]['difference_upper'] for i in range(5)),
        }
    if any(len(rows) != 20 for rows in by_endpoint.values()):
        raise ValueError('Expected 20 WTs for each endpoint')
    expected = {(r['analysis'], r['endpoint']): r for r in recorded}
    completion = []
    for endpoint, rows in by_endpoint.items():
        item = dict(endpoint=endpoint, post_outcome_mechanism_supplement=True)
        for bound in ('lower', 'upper'):
            actual = _paired_effect({pid: value[bound] for pid, value in rows.items()},
                                    protocol['component_by_protein_id'],
                                    protocol['bootstrap_seed'], protocol['bootstrap_replicates'])
            original = expected[('within_WT_average_of_five', endpoint)][bound + '_effect']
            _check_close(actual['mean_difference'], original['mean_difference'], endpoint + ' mean')
            for got, want in zip(actual['joint_CI97_5'], original['joint_Bonferroni_CI97_5']):
                _check_close(got, want, endpoint + ' interval')
            item[bound + '_effect'] = actual
        completion.append(item)

    strict = []
    groups = (('Enzyme80', 'complete_catalytic_AA_identity'),
              ('Enzyme80', 'identity_aware_catalytic_neighbor_contact_recovery'),
              ('Binder60', 'interface_contact_recovery'), ('Binder60', 'interface_F1'))
    summaries = {r['cohort']: r['strict_canonical'] for r in raygun['summaries']}
    for cohort, endpoint in groups:
        rows = [r for r in raygun['per_output'] if r['cohort'] == cohort]
        if len(rows) != {'Enzyme80': 80, 'Binder60': 60}[cohort]:
            raise ValueError('Unexpected Raygun cohort size: ' + cohort)
        values = [r['strict_canonical'][endpoint] for r in rows]
        mean = statistics.fmean(values)
        _check_close(mean, summaries[cohort][endpoint], cohort + ' ' + endpoint)
        strict.append(dict(cohort=cohort, N=len(rows), endpoint=endpoint,
                           strict_canonical=mean, mapping='canonical_global_NW_exact_identity'))
    return dict(completion=completion, raygun_strict=strict,
                paired_records=paired, raygun_records=raygun['per_output'],
                completion_bootstrap_seed=protocol['bootstrap_seed'],
                completion_bootstrap_replicates=protocol['bootstrap_replicates'])
