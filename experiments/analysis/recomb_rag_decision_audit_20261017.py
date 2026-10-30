"""Separate retrieval provenance, candidate changes and actual editing decisions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import delepref_unlimited_planning as planning
from recomb_scope_screen_20261017 import read, sha, write

BASE = Path('reports/recomb_rag_decision_audit_20261017')


def category(kind, retrieved_unique, mask_changed, sequence_changed):
    if not retrieved_unique:
        if mask_changed or sequence_changed:
            raise ValueError('fallback changed the original decision')
        return 'no_usable_proposal_fallback'
    if kind == 'retrieval_direct':
        return 'direct_guidance_selected'
    if kind in ('retrieval_completed', 'retrieval_refined'):
        return 'partial_guidance_selected'
    if kind == 'original_policy_unlimited':
        return 'retrieval_proposed_policy_selected'
    raise ValueError('unknown generation provenance: ' + str(kind))


def validate_row(row, sequence, profile):
    spans = row['deleted_intervals']
    if isinstance(spans, str):
        spans = json.loads(spans)
    mask = planning.validate_plan(len(sequence), float(row['budget']), spans)
    edited = ''.join(a for a, deleted in zip(sequence, mask) if not deleted)
    if row['wildtype_sequence'] != sequence:
        raise ValueError('WT identity changed')
    if row.get('edited_sequence', edited) != edited:
        raise ValueError('edited sequence reconstruction failed')
    if row.get('edited_sequence_sha256', hashlib.sha256(edited.encode()).hexdigest()) != hashlib.sha256(edited.encode()).hexdigest():
        raise ValueError('edited sequence SHA mismatch')
    target = row.get('retrieval_source')
    if target:
        events = {h['target']: set(h['events']) for h in planning.natural_events(profile, len(sequence))}
        seeds = row['retrieval_seed_spans']
        if isinstance(seeds, str):
            seeds = json.loads(seeds)
        if not seeds or target not in events:
            raise ValueError('missing reviewed retrieval provenance')
        for a, b in seeds:
            if (a, b) not in events[target] or not mask[a:b+1].all() or mask[a-1] or mask[b+1]:
                raise ValueError('retrieved event or preserved anchor mismatch')
    return edited


def audit_cohort(root, name, panel_path, profile_path, folders, sources):
    panel = {p['protein_id']: p for p in read(root/panel_path)}
    profiles = read(root/profile_path)
    if set(panel) != set(profiles):
        raise ValueError('reviewed profile/query mismatch')
    selections, pools, diagnostics = {}, defaultdict(list), {}
    def key(row):
        return row['protein_id'], float(row['budget']), int(row['generation_seed'])
    for folder in folders:
        for file in ('all_selections.json', 'candidate_scores.json', 'generation_diagnostics.json'):
            path = root/folder/file
            sources[str(path.relative_to(root))] = sha(path)
        for row in read(root/folder/'all_selections.json'):
            if row['method'] in ('delepref_unlimited', 'rag_unlimited'):
                k = key(row) + (row['method'],)
                if k in selections:
                    raise ValueError('duplicate selected condition')
                selections[k] = row
        for row in read(root/folder/'candidate_scores.json'):
            if row['method'] in ('delepref_unlimited', 'rag_unlimited'):
                pools[key(row) + (row['method'],)].append(row)
        for d in read(root/folder/'generation_diagnostics.json'):
            k = d['protein_id'], float(d['budget']), int(d['seed'])
            if k in diagnostics:
                raise ValueError('duplicate condition diagnostics')
            diagnostics[k] = d
    rows = []
    for k, d in sorted(diagnostics.items()):
        pid, beta, seed = k
        if pid not in panel:
            raise ValueError('unreviewed query')
        p = panel[pid]; sequence = p['sequence']; profile = profiles[pid]
        a = selections[k + ('delepref_unlimited',)]
        b = selections[k + ('rag_unlimited',)]
        masks = []
        for method in ('delepref_unlimited', 'rag_unlimited'):
            pool = pools[k + (method,)]
            mask_set = {r['deletion_mask_sha256'] for r in pool}
            if len(pool) != 8 or len(mask_set) != 8 or sorted(int(r['scorer_rank']) for r in pool) != list(range(1, 9)):
                raise ValueError('unique K8/ranking mismatch')
            for r in pool:
                validate_row(r, sequence, profile)
            winner = selections[k + (method,)]
            if min(pool, key=lambda r: int(r['scorer_rank']))['deletion_mask_sha256'] != winner['deletion_mask_sha256']:
                raise ValueError('wrong same-Scorer selection')
            validate_row(winner, sequence, profile)
            masks.append(mask_set)
        changed = a['deletion_mask_sha256'] != b['deletion_mask_sha256']
        seq_changed = a['edited_sequence_sha256'] != b['edited_sequence_sha256']
        unique = int(d['retrieved_unique'])
        if bool(d['fallback']) != (unique == 0):
            raise ValueError('fallback diagnostic mismatch')
        if d['fallback'] and masks[0] != masks[1]:
            raise ValueError('fallback candidate pool changed')
        events = planning.natural_events(profile, len(sequence))
        covered = {i for h in events for a0, b0 in h['events'] for i in range(a0, b0+1)}
        seeds = planning.seed_sets(profile, len(sequence), beta)
        reason = None
        if not unique:
            reason = ('no_reviewed_representatives' if not profile['representatives'] else
                      'no_reliable_internal_event' if not events else
                      'no_intact_event_subset_within_budget' if not seeds else
                      'no_feasible_unique_anchored_proposal')
        rows.append(dict(cohort=name, protein_id=pid, budget=beta, generation_seed=seed,
            stratum=p.get('stratum', p.get('dataset')), participation=category(b['generation_kind'], unique, changed, seq_changed),
            fallback_reason=reason, selected_generation_kind=b['generation_kind'],
            mask_changed=changed, edited_sequence_changed=seq_changed, candidate_set_changed=masks[0] != masks[1],
            new_masks_in_RAG_pool=len(masks[1]-masks[0]), selected_mask_absent_from_original_K8=b['deletion_mask_sha256'] not in masks[0],
            reliable_representatives=len(profile['representatives']), reliable_internal_event_homologs=len(events),
            reliable_gap_coverage=len(covered)/len(sequence),
            maximal_single_homolog_budget_coverage=max((s['deleted'] for s in seeds), default=0)/planning.delete_count(len(sequence), beta),
            retrieved_proposals=unique, structure_effect_evaluated=False))
    sources[panel_path] = sha(root/panel_path); sources[profile_path] = sha(root/profile_path)
    return rows


def run(root):
    out = root/BASE
    if out.exists():
        raise FileExistsError('preserve the completed decision audit; do not overwrite')
    sources = {}; rows = []
    rows += audit_cohort(root, 'general', 'reports/recomb_benign_scope_20261017/general_panel.json',
        'retrieval/recomb_benign_general_20261017/profiles.json',
        [Path('reports/recomb_benign_plans_20261017'), Path('reports/recomb_benign_heads_extension_20261017')], sources)
    rows += audit_cohort(root, 'application', 'reports/recomb_benign_app_model_20261017/panel.json',
        'retrieval/recomb_benign_app_profiles_20261017/profiles.json',
        [Path('reports/recomb_benign_app_model_20261017')], sources)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['cohort'], row['budget'], row['generation_seed'])].append(row)
    summaries = []
    for (cohort, beta, seed), items in sorted(groups.items()):
        summaries.append(dict(cohort=cohort, budget=beta, generation_seed=seed, N=len(items),
            participation=dict(Counter(r['participation'] for r in items)),
            fallback_reasons=dict(Counter(r['fallback_reason'] for r in items if r['fallback_reason'])),
            mask_changes=sum(r['mask_changed'] for r in items),
            edited_sequence_changes=sum(r['edited_sequence_changed'] for r in items),
            candidate_set_changes=sum(r['candidate_set_changed'] for r in items)))
    expected = {('general', .2, 13): 44, ('general', .2, 29): 44,
                ('general', .1, 13): 20, ('general', .3, 13): 20, ('general', .4, 13): 20,
                ('application', .2, 13): 23}
    if {k: len(v) for k,v in groups.items()} != expected:
        raise ValueError('final prespecified cohort sizes differ')
    sources['analysis/recomb_rag_decision_audit_20261017.py'] = sha(Path(__file__))
    sources['analysis/delepref_unlimited_planning.py'] = sha(root/'analysis/delepref_unlimited_planning.py')
    out.mkdir()
    write(out/'per_condition.json', rows); write(out/'summary.json', summaries)
    write(out/'completion.json', dict(status='PASS_DECISION_AND_PROVENANCE_AUDIT_NOT_STRUCTURAL_EFFECT',
        conditions=len(rows), original_and_RAG_K=8, unchanged_original_Scorer=True,
        every_candidate_budget_source_anchor_and_selection_checked=True,
        source_sha256=sources, structure_effect_evaluated=False, new_model_forwards=0, new_structure_predictions=0,
        outputs={n: sha(out/n) for n in ('per_condition.json','summary.json')}))
    print(json.dumps(summaries, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path.cwd())
    run(p.parse_args().root)
