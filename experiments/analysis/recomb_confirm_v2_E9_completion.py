"""Post-outcome, same-seed learned versus score-free legal completion control.

Only the original complete() objective uses learned residue logits. Candidate
admission is paired and score-blind so the two final pools have equal quotas.
Original General100 files, weights, caches and scientific winners are read-only.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import delepref_unlimited_planning as planning
from recomb_confirm_v2_diagnostics import read, sha, write

BASE = Path('reports/recomb_confirm_v2_E9_completion')
FREEZE = Path('configs/recomb_confirm_v2_E9_completion_freeze.json')
WINNERS = Path('configs/recomb_confirm_v2_E9_selected_winners_freeze.json')
PARENT = Path('reports/recomb_confirm_v2_General100_plans')
PANEL = Path('reports/recomb_confirm_v2_general_native_v2/General100_panel.json')
PROFILES = Path('retrieval/recomb_confirm_v2_General100/profiles.json')
ARMS = ('policy_completion', 'simple_legal_completion')
CHECKPOINT = '1bc7694be77e24e4b671773be39331daed241dcfe101fd6d2d26f8e23fe7ff4d'
GENERAL_FREEZE_SHA = '7c42d417fea99f5aee5b90b2094c5a15b43532e78de2ce743555e72f55f6a04d'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def spans(row):
    value = row['deleted_intervals']
    return tuple(map(tuple, json.loads(value) if isinstance(value, str) else value))


def constraints(length, seed, beta=.2):
    forced = planning.mask_for(length, seed['spans'])
    kept = np.zeros(length, dtype=bool)
    for a, b in seed['spans']:
        if a == 0 or b == length-1:
            raise ValueError('A natural seed must be internal')
        kept[a-1] = True
        kept[b+1] = True
    remaining = planning.delete_count(length, beta)-int(forced.sum())
    if (forced & kept).any() or remaining <= 0 or remaining > int((~(forced | kept)).sum()):
        raise ValueError('Seed is not a feasible partial plan')
    return forced, kept, remaining


def input_roster(panel, profiles, count=20):
    eligible = []
    for row in panel:
        pid, sequence = row['protein_id'], row['sequence']
        # This native-stage snapshot intentionally precedes generation release.
        # The exact final General confirmation below supplies that later release.
        reviewed = {'PASS_SOURCE_AND_FUNCTION_METADATA', 'BOTH_PPI_PARTNERS_METADATA_SCOPE_PASS',
                    'PASS_SOURCE_FUNCTION_AND_CANONICAL_IDENTITY'}
        if not row['WT_encoding_allowed'] or row.get('source_scope_review_status') not in reviewed:
            raise ValueError('A parent General input is not positively released')
        if hashlib.sha256(sequence.encode()).hexdigest() != row['WT_sha256']:
            raise ValueError('Reviewed WT sequence identity differs')
        partial = []
        for seed in planning.seed_sets(profiles[pid], len(sequence), .2):
            try:
                constraints(len(sequence), seed)
            except ValueError:
                continue
            partial.append(seed)
        if partial:
            order = hashlib.sha256(('E9_matched_partial_seed_v1|'+row['WT_sha256']).encode()).hexdigest()
            eligible.append(dict(protein_id=pid, WT_sha256=row['WT_sha256'], input_order_sha256=order,
                sequence_length=len(sequence), partial_seed_count=len(partial), seed=partial[0]))
    eligible.sort(key=lambda r: (r['input_order_sha256'], r['protein_id']))
    if len(eligible) < count:
        raise ValueError('Not enough input-eligible WT; do not replace the selection criterion')
    return eligible, eligible[:count]


def policy_completion(logits, seed, condition_seed, attempt, beta=.2):
    forced, kept, _ = constraints(len(logits), seed, beta)
    values = np.asarray(logits, dtype=np.float64)
    if attempt:
        natural_spans = tuple(map(tuple, seed['spans']))
        salt = int(hashlib.sha256((seed['target']+str(natural_spans)).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng((int(condition_seed)+salt+(attempt-1)*1_000_003) % (2**32))
        values = values+rng.standard_normal(len(values))
    result, _ = planning.complete(values, beta, forced, kept)
    return result


def legal_completion(length, seed, condition_seed, attempt, beta=.2):
    """Uniform free-position sampling; signature deliberately excludes logits."""
    forced, kept, remaining = constraints(length, seed, beta)
    salt = int(digest(dict(namespace='E9_simple_legal_v1', target=seed['target'], spans=seed['spans']))[:16], 16)
    rng = np.random.default_rng((int(condition_seed)+salt+attempt*1_000_003) % (2**64))
    free = np.flatnonzero(~(forced | kept))
    mask = forced.copy()
    mask[rng.choice(free, size=remaining, replace=False)] = True
    return planning.intervals(mask)


def paired_candidates(logits, seed, condition_seed, original8, max_attempts=128, proposer=None):
    n = len(logits)
    forced, kept, remaining = constraints(n, seed)
    old = {spans(r) for r in original8}
    if len(original8) != 8 or len(old) != 8:
        raise ValueError('Frozen original8 must contain eight unique masks')
    results = {arm: [] for arm in ARMS}
    seen = {arm: set() for arm in ARMS}
    history = []
    for attempt in range(max_attempts):
        outputs = (proposer(attempt) if proposer is not None else {
            ARMS[0]: policy_completion(logits, seed, condition_seed, attempt),
            ARMS[1]: legal_completion(n, seed, condition_seed, attempt)})
        trial = dict(attempt=attempt, target=seed['target'], seed_spans=seed['spans'], arms={})
        for arm in ARMS:
            value = tuple(map(tuple, outputs[arm]))
            mask = planning.validate_plan(n, .2, value)
            if not mask[forced].all() or mask[kept].any():
                raise ValueError('A completion lost its seed or anchor')
            trial['arms'][arm] = dict(legal=True, original_duplicate=value in old,
                prior_paired_duplicate=value in seen[arm], spans=value)
        accept = all(not t['original_duplicate'] and not t['prior_paired_duplicate'] for t in trial['arms'].values())
        trial['pair_accepted'] = accept
        history.append(trial)
        if accept:
            for arm in ARMS:
                value = trial['arms'][arm]['spans']
                seen[arm].add(value)
                results[arm].append(dict(spans=value, source_target=seed['target'], seed_spans=seed['spans'],
                    all_reliable_events_used=seed['all_reliable_events_used'],
                    kind='retrieval_completed' if arm == ARMS[0] else 'retrieval_simple_legal_completed',
                    paired_attempt_index=attempt))
        if len(results[ARMS[0]]) == 4:
            break
    paired_count = len(results[ARMS[0]])
    if len(results[ARMS[1]]) != paired_count:
        raise ValueError('Paired admission count differs')
    return results, dict(attempts=len(history), max_attempts=max_attempts, paired_retrieval_slots=paired_count,
        common_original_policy_slots=8-paired_count, requested_retrieval_slots=4,
        seed_deleted=int(forced.sum()), completion_deleted=remaining,
        shortfall=4-paired_count, attempt_log=history)


def ranked(rows):
    order = sorted(range(len(rows)), key=lambda i: (-float(rows[i]['scorer_score']), spans(rows[i])))
    result = copy.deepcopy(rows)
    for rank, i in enumerate(order, 1):
        result[i]['scorer_rank'] = rank
    return result


def claim(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)


def prepare(root):
    if (root/FREEZE).exists() or (root/BASE/'panel.json').exists():
        raise FileExistsError('E9 already frozen; inspect its claim/job before continuing')
    if sha(root/'configs/recomb_confirm_v2_General100_confirmation_freeze.json') != GENERAL_FREEZE_SHA:
        raise ValueError('Original scientific parent differs')
    release = read(root/'configs/recomb_confirm_v2_General100_confirmation_freeze.json')
    if not release['fold_allowed'] or not release['all_selected_outputs_frozen']:
        raise ValueError('Original reviewed General cohort was not finally released')
    parent = read(root/'configs/recomb_confirm_v2_General100_plans_freeze.json')
    for name, expected in parent['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Original generation input/source differs: '+name)
    done = read(root/PARENT/'completion.json')
    for name, expected in done['outputs'].items():
        if sha(root/PARENT/name) != expected:
            raise ValueError('Original completed plans differ: '+name)
    eligible, panel = input_roster(read(root/PANEL), read(root/PROFILES))
    protocol = copy.deepcopy(read(root/'reports/recomb_confirm_v2_General100_catalog/statistical_protocol.json'))
    protocol.update(status='POST_OUTCOME_E9_MATCHED_PARTIAL_SEED_CONTROL', WT=20,
        methods=list(ARMS), requested_comparisons=[[ARMS[0], ARMS[1]]],
        primary_endpoints=['structure_mapped_native_contact_recovery', 'structure_mapped_CA_lddt'],
        CI95_and_joint_CI975=True, no_quality_selection=True)
    for name, value in [('eligible_inputs.json', eligible), ('panel.json', panel), ('statistical_protocol.json', protocol)]:
        write(root/BASE/name, value)
    names = ['analysis/recomb_confirm_v2_E9_completion.py', 'analysis/tests/test_recomb_confirm_v2_E9_completion.py',
        'slurm/recomb_confirm_v2_E9_completion.sbatch', 'analysis/delepref_unlimited_planning.py',
        'analysis/delepref_unlimited_scorer.py', 'analysis/delepref_v8_runtime.py', 'analysis/delepref_v8_main.py',
        'analysis/delepref_v8_rag_unlimited.py', 'analysis/recomb_confirm_v2_diagnostics.py',
        'configs/delepref_v8_baseline_freeze.json', 'configs/recomb_confirm_v2_General100_plans_freeze.json',
        'configs/recomb_confirm_v2_General100_confirmation_freeze.json',
        'reports/recomb_confirm_v2_General100_model/encoding_completion.json',
        'reports/recomb_confirm_v2_general_native/encoder_inputs.json', str(PANEL), str(PROFILES)]
    names += [str(PARENT/n) for n in ['completion.json', 'candidate_scores.json', 'all_selections.json']]
    names += [str(BASE/n) for n in ['eligible_inputs.json', 'panel.json', 'statistical_protocol.json']]
    write(root/FREEZE, dict(status='E9_INPUT_ONLY_PANEL_AND_MATCHED_COMPLETION_CONTRACT_FROZEN',
        files={n: sha(root/n) for n in names}, eligible_WT=len(eligible), fixed_WT=20,
        original_checkpoint_sha256=CHECKPOINT, K=8, original_quota=4, retrieval_quota=4,
        max_paired_attempts=128, budget=.2, generation_seed=13, common_seed_per_WT=True,
        same_admitted_attempt_slots=True, admission_uses_quality_or_scorer=False,
        fixed_original_backfill_on_paired_shortfall=True, post_outcome_supplement=True,
        comparison_scope='Matched first partial seed with paired score-blind admission; not original RAG winner replay',
        new_training=False, new_WT_encoder_forwards=0, structure_prediction_allowed=False))
    print(sha(root/FREEZE), flush=True)


def run(root, checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('E9 heads require CPU Slurm')
    if sha(root/FREEZE) != checksum:
        raise ValueError('E9 source freeze differs')
    frozen = read(root/FREEZE)
    for name, expected in frozen['files'].items():
        if sha(root/name) != expected:
            raise ValueError('E9 frozen source/input differs: '+name)
    for name, expected in {**read(root/'configs/delepref_v8_baseline_freeze.json')['inputs'],
                           **read(root/'configs/delepref_v8_baseline_freeze.json')['code']}.items():
        if sha(root/name) != expected:
            raise ValueError('Original formal model dependency differs: '+name)
    claim(root/BASE/'execution_claim.json', dict(job_id=os.environ['SLURM_JOB_ID'], freeze_sha256=checksum))
    import torch
    import delepref_v8_runtime as rt
    import delepref_v8_main as parent
    import delepref_unlimited_scorer as scoring
    torch.set_num_threads(4)
    cache_path = root/'reports/recomb_confirm_v2_General100_model/wt_cache.pt'
    if sha(cache_path) != read(root/'reports/recomb_confirm_v2_General100_model/encoding_completion.json')['cache_sha256']:
        raise ValueError('Original genuine WT cache differs')
    start = time.monotonic()
    runtime, api = rt.load(root, device='cpu', cached_heads_only=True)
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    if not cache['real_encoder'] or cache['unique_WT'] != 100 or cache['checkpoint_sha256'] != CHECKPOINT:
        raise ValueError('Not the original real100 WT representation')
    records = read(root/'reports/recomb_confirm_v2_general_native/encoder_inputs.json')
    by_record = {r['protein_id']: r for r in records}
    chunk = parent.cached_chunk(api, cache, records)
    old = defaultdict(list)
    for row in read(root/PARENT/'candidate_scores.json'):
        if row['budget'] == .2 and row['generation_seed'] == 13:
            old[row['protein_id'], row['method']].append(row)
    pools, winners, behavior = [], [], []
    scorer_batches = scorer_candidates = reused_scores = 0
    with torch.inference_mode():
        for item in read(root/BASE/'panel.json'):
            pid = item['protein_id']; record = by_record[pid]; sequence = record['wildtype_sequence']
            original = sorted(old[pid, 'delepref_unlimited'], key=lambda r: r['raw_candidate_rank'])
            if len(sequence) != item['sequence_length'] or hashlib.sha256(sequence.encode()).hexdigest() != item['WT_sha256']:
                raise ValueError('Frozen E9 input identity differs')
            embedding = chunk.encoded.embeddings[chunk.encoded.key_for(pid, sequence)]
            logits = runtime.model.propose_from_embeddings([embedding], [.2])[0].residue_logits.detach().cpu().double().numpy()
            if not np.isfinite(logits).all() or np.std(logits) == 0:
                raise ValueError('Invalid formal Policy logits')
            condition_seed = parent.controls.condition_seed(13, pid, sequence, .2)
            if int(original[0]['condition_seed']) != condition_seed:
                raise ValueError('Original scientific condition seed differs')
            additions, diagnostic = paired_candidates(logits, item['seed'], condition_seed, original)
            known = {}
            for method in ['delepref_unlimited', 'rag_unlimited', 'gap_frequency', 'terminal']:
                for row in sorted(old[pid, method], key=lambda r: r['raw_candidate_rank']):
                    known.setdefault(spans(row), row)
            arm_rows, pending = {}, []
            for arm in ARMS:
                new = parent.make_rows(api, original[0], additions[arm], len(sequence), logits)
                for row, proposal in zip(new, additions[arm]):
                    row['paired_attempt_index'] = proposal['paired_attempt_index']
                    row['method'] = arm
                    saved = known.get(spans(row))
                    if saved is None:
                        pending.append((arm, len(arm_rows.get(arm, [])), row))
                    else:
                        row.update(scorer_score=saved['scorer_score'],
                            scorer_span_extrapolation=saved['scorer_span_extrapolation'],
                            scorer_training_span_scale=saved['scorer_training_span_scale'],
                            score_reused_from_frozen_method=saved['method'])
                        reused_scores += 1
                    arm_rows.setdefault(arm, []).append(row)
            if pending:
                values = scoring.score_candidates(runtime, api, chunk, [r for arm, i, r in pending])
                scorer_batches += 1;scorer_candidates += len(values)
                for (arm, i, oldrow), row in zip(pending, values):
                    arm_rows[arm][i] = row
            n_retrieved = diagnostic['paired_retrieval_slots']
            for arm in ARMS:
                rows = copy.deepcopy(original[:4])+arm_rows[arm]+copy.deepcopy(original[4:8-n_retrieved])
                if len(rows) != 8 or len({spans(r) for r in rows}) != 8:
                    raise ValueError('E9 final pool is not unique K8')
                for i, row in enumerate(rows):
                    row.update(method=arm, raw_candidate_rank=i+1)
                    planning.validate_plan(len(sequence), .2, spans(row))
                rows = ranked(rows)
                winner = copy.deepcopy(min(rows, key=lambda r: r['scorer_rank']))
                mask = planning.mask_for(len(sequence), spans(winner))
                edited = ''.join(c for c, deleted in zip(sequence, mask) if not deleted)
                winner.update(edited_sequence=edited, edited_sequence_sha256=hashlib.sha256(edited.encode()).hexdigest(),
                    evaluation_role='post_outcome_E9_same_seed_completion', pool_sha256=digest(rows),
                    common_seed_sha256=digest(item['seed']), paired_retrieval_slots=n_retrieved,
                    original_policy_slots=8-n_retrieved)
                winners.append(winner)
                pools.append(dict(protein_id=pid, method=arm, rows=rows, pool_sha256=digest(rows)))
                diagnostic[arm] = dict(final_adopted_seed_completion=winner['generation_kind'].startswith('retrieval'),
                    selected_mask_sha256=winner['deletion_mask_sha256'], selected_sequence_sha256=winner['edited_sequence_sha256'],
                    admitted_attempt_indices=[r['paired_attempt_index'] for r in additions[arm]])
            behavior.append(dict(protein_id=pid, WT_sha256=item['WT_sha256'], seed=item['seed'], **diagnostic))
            print('E9_MATCHED_COMPLETION_DONE '+pid, flush=True)
    if len(winners) != 40 or len(pools) != 40 or len(behavior) != 20:
        raise ValueError('E9 all20 paired coverage differs')
    for name, value in [('candidate_pools.json', pools), ('selected_structure_outputs.json', winners), ('behavior.json', behavior)]:
        write(root/BASE/name, value)
    names = ['candidate_pools.json', 'selected_structure_outputs.json', 'behavior.json']
    done = dict(status='E9_ALL20_PAIRED_K8_WINNERS_FROZEN_NOT_YET_STRUCTURE_EVALUATED',
        freeze_sha256=checksum, fixed_WT=20, logical_final_winners=40, genuine_Policy_WT_forwards=20,
        newly_scored_candidates=scorer_candidates, new_Scorer_batches=scorer_batches,
        historical_completion_candidate_scores_reused=reused_scores, historical_original_policy_scores_reused=2*sum(r['common_original_policy_slots'] for r in behavior),
        new_WT_encoder_forwards=0, new_structure_predictions=0, new_training=False,
        paired_shortfall_WT=sum(r['shortfall'] > 0 for r in behavior),
        post_outcome_supplement=True, job_id=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-start,
        outputs={n: sha(root/BASE/n) for n in names})
    write(root/BASE/'completion.json', done)
    frozen_names = [str(FREEZE)]+[str(BASE/n) for n in names+['completion.json', 'panel.json', 'statistical_protocol.json']]
    write(root/WINNERS, dict(status='E9_MATCHED_FINAL_WINNERS_FROZEN_BEFORE_NEW_STRUCTURE_QUALITY',
        files={n: sha(root/n) for n in frozen_names}, logical_winners=40, WT=20, original_model_unchanged=True))
    print(json.dumps(done), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    prepare(args.root) if args.prepare else run(args.root, args.freeze_sha256)
