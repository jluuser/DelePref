"""Freeze a reviewed subset of existing application retrieval evidence; no search."""
import argparse
from pathlib import Path
from recomb_confirm_v2_diagnostics import read, sha, write
from recomb_benign_profiles_20261017 import filtered_profile


def main(root, cohort):
    base = root / ('retrieval/recomb_confirm_v2_' + cohort)
    review = root / ('reports/recomb_confirm_v2_' + cohort + '_evidence_review')
    panel = root / ('reports/recomb_confirm_v2_' + cohort + '_constructs/' + cohort + '_panel.json')
    parent = base / 'profiles_unreviewed_evidence.json'
    target = base / 'profiles.json'
    freeze = root / ('configs/recomb_confirm_v2_' + cohort + '_profiles_freeze.json')
    if target.exists() or freeze.exists():
        raise FileExistsError('Reviewed profiles already frozen; inspect completion before resuming')
    inventory, summary = read(review / 'inventory.json'), read(review / 'completion.json')
    if inventory['profiles_sha256'] != sha(parent) or summary['audit_sha256'] != sha(review / 'target_metadata_audit.json'):
        raise ValueError('Reviewed profile or metadata identity changed')
    if summary['errors'] or summary['queries'] != int(cohort.removeprefix('Binder').removeprefix('Enzyme')):
        raise ValueError('Metadata requests incomplete or unexpected query roster')
    profiles, audit, queries = read(parent), read(review / 'target_metadata_audit.json'), read(panel)
    if set(profiles) != {q['protein_id'] for q in queries}:
        raise ValueError('Application query roster changed')
    result = {q['protein_id']: filtered_profile(profiles[q['protein_id']], q, audit) for q in queries}
    counts = {pid: len(p['representatives']) for pid, p in result.items()}
    paths = [parent, panel, review / 'inventory.json', review / 'completion.json',
             review / 'target_metadata_audit.json', review / 'policy.json',
             root / 'analysis/recomb_confirm_v2_application_profiles.py',
             root / 'analysis/recomb_benign_profiles_20261017.py']
    write(target, result)
    write(freeze, dict(status='APPLICATION_REVIEWED_RETRIEVAL_PROFILE_FROZEN', cohort=cohort,
        files={str(p.relative_to(root)): sha(p) for p in paths}, profiles_sha256=sha(target),
        WT=len(result), representative_order_and_projection_unchanged=True,
        no_backfill=True, no_new_search=True, no_new_model_calls=True, empty_profiles_retained=True,
        original_weighting_semantics='quality ordered representatives; no numeric weights to renormalize',
        metadata_review_not_variant_quality=True, generation_requires_real_WT_cache_and_separate_freeze=True))
    done = dict(status='APPLICATION_REVIEWED_RETRIEVAL_PROFILE_FROZEN', cohort=cohort,
        WT=len(result), representative_rows=sum(counts.values()),
        unique_representatives=len({r['target'] for p in result.values() for r in p['representatives']}),
        with_evidence=sum(v > 0 for v in counts.values()), without_evidence=sum(v == 0 for v in counts.values()),
        per_query_counts=counts, profiles_sha256=sha(target), freeze_sha256=sha(freeze),
        new_model_calls=0, new_structure_predictions=0,
        some_evidence_not_necessarily_budget_sufficient=True)
    write(base / 'reviewed_profiles_completion.json', done)
    print({k: v for k, v in done.items() if k != 'per_query_counts'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cohort', choices=['Binder60', 'Enzyme80'], required=True)
    args = parser.parse_args()
    main(args.root, args.cohort)
