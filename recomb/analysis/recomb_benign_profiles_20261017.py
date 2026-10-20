"""Filter existing homolog evidence by reviewed identity, preserving its ordering."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from recomb_scope_screen_20261017 import read, sha, write


def filtered_profile(profile, query, reviewed):
    if profile['query']['sequence_sha256'] != query['sequence_sha256']:
        raise ValueError('query identity mismatch')
    if hashlib.sha256(profile['query']['sequence'].encode()).hexdigest() != query['sequence_sha256']:
        raise ValueError('query sequence changed')
    kept = []
    for rep in profile['representatives']:
        entry = reviewed.get(rep['target'], {})
        if entry.get('status') != 'RETAIN':
            continue
        if entry.get('representative_sequence_sha256') != rep['target_sequence_sha256']:
            raise ValueError('approved representative identity mismatch')
        if len(rep['projected']) != len(profile['query']['sequence']):
            raise ValueError('projection length mismatch')
        kept.append(copy.deepcopy(rep))
    result = copy.deepcopy(profile)
    result['representatives'] = kept
    result['parent_audit'] = result.pop('audit', {})
    result['audit'] = dict(parent_representatives=len(profile['representatives']), selected=len(kept),
                          removed_by_scope=len(profile['representatives'])-len(kept),
                          source_order_and_within_homolog_projection_preserved=True)
    result['scope_filter'] = 'positive_source_and_exact_representative_identity_v1'
    return result


def main(root):
    parent = root / 'retrieval/delepref_v8_main/profiles.json'
    scope = root / 'reports/recomb_benign_scope_20261017/general_panel.json'
    metadata = root / 'reports/recomb_benign_retrieval_scope_20261017/target_metadata_audit.json'
    inventory = root / 'reports/recomb_benign_retrieval_scope_20261017/inventory.json'
    if sha(parent) != read(inventory)['parent_profiles_sha256']:
        raise ValueError('parent profile changed after metadata audit')
    queries = read(scope)
    profiles, reviewed = read(parent), read(metadata)
    selected = {q['protein_id']: filtered_profile(profiles[q['protein_id']], q, reviewed) for q in queries}
    out = root / 'retrieval/recomb_benign_general_20261017'
    out.mkdir(exist_ok=False)
    write(out / 'profiles.json', selected)
    counts = {pid: len(p['representatives']) for pid, p in selected.items()}
    files = [parent, scope, metadata, inventory, Path(__file__), root / 'analysis/delepref_unlimited_planning.py']
    freeze = dict(status='SCREENED_EVIDENCE_SUBSET_FROZEN_NO_GENERATION',
        files={str(p.relative_to(root)): sha(p) for p in files}, profiles_sha256=sha(out / 'profiles.json'),
        WT=len(selected), original_weighting_semantics='quality ordered representatives, no numeric weights or renormalization in the original planner',
        original_representative_order_preserved=True, raw_alignment_and_projection_unchanged=True,
        no_new_search=True, no_backfill_from_unprocessed_hits=True, model_results_used=False,
        empty_profiles_retained=True, new_model_calls=0, new_structure_predictions=0)
    write(root / 'configs/recomb_benign_general_profiles_freeze_20261017.json', freeze)
    completion = dict(status=freeze['status'], WT=len(selected), representative_rows=sum(counts.values()),
        with_evidence=sum(n>0 for n in counts.values()), without_evidence=sum(n==0 for n in counts.values()),
        per_query_counts=counts, profiles_sha256=freeze['profiles_sha256'],
        freeze_sha256=sha(root / 'configs/recomb_benign_general_profiles_freeze_20261017.json'),
        new_model_calls=0, new_structure_predictions=0)
    write(out / 'completion.json', completion)
    print(json.dumps({k:v for k,v in completion.items() if k!='per_query_counts'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    main(p.parse_args().root)
