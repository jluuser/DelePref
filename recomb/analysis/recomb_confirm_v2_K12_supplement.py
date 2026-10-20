"""Post-outcome K12 candidate supplement; original K8 pools are immutable inputs."""
import argparse
import copy
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

BASE = Path('reports/recomb_confirm_v2_K12_supplement')
FREEZE = Path('configs/recomb_confirm_v2_K12_supplement_freeze.json')
PARENT = Path('reports/recomb_confirm_v2_General100_plans')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def spans(row):
    value = row.get('deleted_intervals', row.get('spans'))
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(tuple(int(i) for i in pair) for pair in value)


def continue_unique(prefix, propose, start_attempt, total_cap=192, target=12):
    """Return additions only; do not regenerate, score, or mutate the old prefix."""
    if len(prefix) != 8 or len({spans(r) for r in prefix}) != 8:
        raise ValueError('Expected the exact eight unique frozen Policy masks')
    if not 8 <= start_attempt <= total_cap:
        raise ValueError('Invalid recorded original attempt count')
    seen = {spans(r) for r in prefix}
    additions, attempts, duplicates = [], [], 0
    for attempt in range(start_attempt, total_cap):
        item = propose(attempt)
        key = spans(item)
        attempts.append(attempt)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        additions.append(dict(item, K12_attempt_index=attempt))
        if len(prefix)+len(additions) == target:
            break
    return additions, dict(original_attempts=start_attempt, total_attempt_cap=total_cap,
                           new_attempt_indices=attempts, new_attempts=len(attempts),
                           new_duplicate_decodes=duplicates,
                           total_attempts=start_attempt+len(attempts),
                           returned=len(prefix)+len(additions), requested=target)


def rag_union(prefix, retrieval_rows, policy_additions):
    if len(retrieval_rows) > 4:
        raise ValueError('Frozen RAG K8 supplied more than four retrieval rows')
    result = copy.deepcopy(prefix)
    seen = {spans(r) for r in result}
    for row in list(retrieval_rows)+list(policy_additions):
        if spans(row) in seen:
            continue
        result.append(copy.deepcopy(row))
        seen.add(spans(row))
        if len(result) == 12:
            break
    return result


def selection_order(rows):
    return sorted(range(len(rows)), key=lambda i: (-float(rows[i]['scorer_score']), spans(rows[i])))


def edited_sequence(sequence, row):
    deleted = {i for a, b in spans(row) for i in range(a, b+1)}
    return ''.join(c for i, c in enumerate(sequence) if i not in deleted)


def selected(sequence, rows, method):
    order = selection_order(rows)
    row = copy.deepcopy(rows[order[0]])
    edited = edited_sequence(sequence, row)
    row.update(source_method=row['method'], source_scorer_rank=row['scorer_rank'],
               method=method, scorer_rank=1, selector='unchanged_original_Scorer',
               edited_sequence=edited, edited_sequence_sha256=hashlib.sha256(edited.encode()).hexdigest(),
               K12_requested=12, K12_realized=len(rows),
               evaluation_role='post_outcome_exploratory_K12_winner',
               pool_membership_and_scores_sha256=digest([(spans(r), r['scorer_score']) for r in rows]))
    return row


def prepare(root):
    if (root/FREEZE).exists() or (root/BASE/'execution_claim.json').exists():
        raise FileExistsError('K12 already frozen/claimed; inspect its existing state')
    state = read(root/'reports/recomb_confirm_v2_execution/execution.json')
    if not state.get('application_main_readout', {}).get('status', '').startswith('COMPLETE_'):
        raise ValueError('Application main readout must precede K12')
    parent = read(root/'configs/recomb_confirm_v2_General100_plans_freeze.json')
    for name, expected in parent['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Frozen parent source/input changed: '+name)
    done = read(root/PARENT/'completion.json')
    for name, expected in done['outputs'].items():
        if sha(root/PARENT/name) != expected:
            raise ValueError('Frozen parent generated output changed: '+name)
    records = read(root/'reports/recomb_confirm_v2_general_native/encoder_inputs.json')
    subset = read(root/'reports/recomb_confirm_v2_General100_subsets/subsets.json')['budget40']
    if (len(records) != 100 or len(subset) != 40 or len({r['protein_id'] for r in subset}) != 40
            or not {r['protein_id'] for r in subset} <= {r['protein_id'] for r in records}):
        raise ValueError('Fixed input-hash rosters differ')
    names = ['analysis/recomb_confirm_v2_K12_supplement.py',
             'analysis/tests/test_recomb_confirm_v2_K12_supplement.py',
             'slurm/recomb_confirm_v2_K12_supplement.sbatch',
             'configs/recomb_confirm_v2_RAG_small_amendment_protocol.json',
             'configs/recomb_confirm_v2_General100_plans_freeze.json',
             'configs/delepref_v8_baseline_freeze.json',
             'configs/delepref_original_restored.yaml',
             'provenance/original_delepref_assets_20261016.json',
             'reports/recomb_confirm_v2_General100_model/encoding_completion.json',
             'reports/recomb_confirm_v2_General100_model/wt_cache.pt',
             'reports/recomb_confirm_v2_general_native/encoder_inputs.json',
             'reports/recomb_confirm_v2_General100_subsets/subsets.json',
             'reports/recomb_confirm_v2_General100_catalog/statistical_protocol.json',
             'configs/recomb_confirm_v2_General100_confirmation_freeze.json']
    names += [str(PARENT/name) for name in ['candidate_scores.json', 'all_selections.json',
                                          'generation_diagnostics.json', 'completion.json']]
    names += [name for name in parent['files'] if name.startswith('analysis/')]
    files = {name: sha(root/name) for name in sorted(set(names))}
    encoded = read(root/'reports/recomb_confirm_v2_General100_model/encoding_completion.json')
    if files['reports/recomb_confirm_v2_General100_model/wt_cache.pt'] != encoded['cache_sha256']:
        raise ValueError('Original real WT cache differs')
    protocol = read(root/'configs/recomb_confirm_v2_RAG_small_amendment_protocol.json')
    if protocol['K12']['total_attempt_cap'] != 192:
        raise ValueError('Authorized K12 attempt cap changed')
    write(root/FREEZE, dict(status='K12_CPU_SOURCE_INPUT_ROSTERS_AND_STATISTICS_FROZEN', files=files,
        original_checkpoint_sha256='1bc7694be77e24e4b671773be39331daed241dcfe101fd6d2d26f8e23fe7ff4d',
        CPU_WT=100, structure_WT=40, structure_roster=subset, generation_seed=13, budget=.2,
        K=12, original_prefix_K=8, total_attempt_cap=192,
        new_PLM_forwards=0, new_training=False, folding_allowed=False,
        maximum_logical_selected_winner_predictions_before_reuse=80,
        post_outcome_exploratory_not_new_blind_confirmation=True,
        statistics=dict(component_bootstrap_replicates=10000, bootstrap_seed=20261016,
            continuous_paired_differences=True, no_new_success_threshold=True,
            original_common_and_fixed_edit_endpoints_preserved=True,
            all_fixed40_cases_failures_unknowns_and_pool_shortfalls_retained=True),
        ranking='original Scorer score descending, integer interval tuples lexicographic',
        old_pool_rows_and_scores_immutable=True,
        row_scorer_rank_scope='source K8 or newly scored batch; K12 selection_order stored separately'))
    print(sha(root/FREEZE), flush=True)


def run(root, expected_freeze):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('K12 heads only through CPU Slurm')
    if sha(root/FREEZE) != expected_freeze:
        raise ValueError('K12 source/input freeze changed')
    frozen = read(root/FREEZE)
    for name, expected in frozen['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Frozen K12 source/input changed: '+name)
    baseline = read(root/'configs/delepref_v8_baseline_freeze.json')
    for name, expected in {**baseline['inputs'], **baseline['code']}.items():
        if sha(root/name) != expected:
            raise ValueError('Original immutable model dependency changed: '+name)
    out = root/BASE
    out.mkdir(parents=True, exist_ok=True)
    fd = os.open(out/'execution_claim.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, 'w') as f:
        json.dump(dict(job_id=os.environ['SLURM_JOB_ID'], freeze_sha256=expected_freeze), f)
    import torch
    import numpy as np
    import delepref_v8_runtime as rt
    import delepref_v8_main as parent
    import delepref_unlimited_planning as planning
    import delepref_unlimited_scorer as scoring
    if rt.CHECKPOINT_SHA != frozen['original_checkpoint_sha256']:
        raise ValueError('Original formal4000 checkpoint differs')
    torch.set_num_threads(4)
    start = time.monotonic()
    runtime, api = rt.load(root, device='cpu', cached_heads_only=True)
    cache = torch.load(root/'reports/recomb_confirm_v2_General100_model/wt_cache.pt', map_location='cpu', weights_only=True)
    if not cache['real_encoder'] or cache['unique_WT'] != 100 or cache['checkpoint_sha256'] != rt.CHECKPOINT_SHA:
        raise ValueError('Not the original real100 WT encoder cache')
    records = read(root/'reports/recomb_confirm_v2_general_native/encoder_inputs.json')
    chunk = parent.cached_chunk(api, cache, records)
    groups = defaultdict(list)
    for row in read(root/PARENT/'candidate_scores.json'):
        if row['budget'] == .2 and row['generation_seed'] == 13:
            groups[row['protein_id'], row['method']].append(row)
    old_diag = {r['protein_id']: r for r in read(root/PARENT/'generation_diagnostics.json')
                if r['budget'] == .2 and r['seed'] == 13}
    old_selected = {(r['protein_id'], r['method']): r for r in read(root/PARENT/'all_selections.json')
                    if r['budget'] == .2 and r['generation_seed'] == 13}
    if len(old_diag) != 100:
        raise ValueError('Original main100 condition roster differs')
    cfg = runtime.config['decoder']
    if 12*int(cfg.get('max_attempts_multiplier', 16)) != 192:
        raise ValueError('Frozen original decoder cap multiplier differs')
    scale = float(cfg.get('perturbation_scale', 1.))
    pools, winners, diagnostics = [], [], []
    policy_calls = scorer_batches = scorer_candidates = 0
    with torch.inference_mode():
        for record in records:
            before = time.monotonic()
            pid = record['protein_id'];sequence = record['wildtype_sequence']
            prefix = sorted(groups[pid, 'delepref_unlimited'], key=lambda r: r['raw_candidate_rank'])
            old_rag = sorted(groups[pid, 'rag_unlimited'], key=lambda r: r['raw_candidate_rank'])
            prefix_saved = copy.deepcopy(prefix)
            if (len(prefix) != 8 or [r['raw_candidate_rank'] for r in prefix] != list(range(1, 9))
                    or any(r['wildtype_sequence'] != sequence for r in prefix)):
                raise ValueError('Frozen original prefix/input differs')
            for row in prefix:
                planning.validate_plan(len(sequence), .2, spans(row))
            if selected(sequence, prefix, 'check')['deletion_mask_sha256'] != old_selected[pid, 'delepref_unlimited']['deletion_mask_sha256']:
                raise ValueError('Unchanged Scorer tie rule did not recover frozen original winner')
            embedding = chunk.encoded.embeddings[chunk.encoded.key_for(pid, sequence)]
            logits = runtime.model.propose_from_embeddings([embedding], [.2])[0].residue_logits.detach().cpu().double().numpy()
            policy_calls += 1
            if not np.isfinite(logits).all() or np.std(logits) == 0:
                raise ValueError('Invalid original trained Policy logits')
            base = torch.as_tensor(np.asarray(logits, dtype=np.float64).copy(), dtype=torch.float64)
            condition_seed = int(prefix[0]['condition_seed'])
            if condition_seed != parent.controls.condition_seed(13, pid, sequence, .2):
                raise ValueError('Frozen original condition seed differs')

            def propose(attempt):
                generator = torch.Generator(device='cpu').manual_seed(condition_seed+attempt*1_000_003)
                values = base if scale == 0 else base+scale*torch.randn(base.shape, generator=generator, dtype=base.dtype)
                intervals, _ = planning.complete(values.numpy(), .2)
                return dict(spans=intervals, kind='original_policy_unlimited')

            additions, cost = continue_unique(prefix, propose, old_diag[pid]['original_generation']['attempts'])
            new_rows = parent.make_rows(api, prefix[0], additions, len(sequence), logits,
                dict(attempts=cost['total_attempts'], duplicates=old_diag[pid]['original_generation']['duplicates']+cost['new_duplicate_decodes']))
            score_cache = {}
            for method in ['delepref_unlimited', 'rag_unlimited', 'gap_frequency', 'terminal']:
                for row in sorted(groups[pid, method], key=lambda r: r['raw_candidate_rank']):
                    score_cache.setdefault(spans(row), row)
            need_scoring = []
            reused = 0
            for i, row in enumerate(new_rows):
                row['raw_candidate_rank'] = 9+i
                row['K12_attempt_index'] = additions[i]['K12_attempt_index']
                row['rank_metadata_scope'] = 'new_policy_continuation_batch'
                known = score_cache.get(spans(row))
                if known is not None:
                    row.update(scorer_score=known['scorer_score'], scorer_span_extrapolation=known['scorer_span_extrapolation'],
                               scorer_training_span_scale=known['scorer_training_span_scale'],
                               score_reused_from_frozen_method=known['method'])
                    reused += 1
                else:
                    need_scoring.append(i)
            if need_scoring:
                scored = scoring.score_candidates(runtime, api, chunk, [new_rows[i] for i in need_scoring])
                scorer_batches += 1;scorer_candidates += len(scored)
                for i, row in zip(need_scoring, scored):
                    new_rows[i] = row
            policy = copy.deepcopy(prefix)+new_rows
            retrieved = [r for r in old_rag if r['generation_kind'].startswith('retrieval')]
            rag = rag_union(prefix, retrieved, new_rows)
            if prefix != prefix_saved or policy[:8] != prefix_saved or rag[:8] != prefix_saved:
                raise ValueError('Frozen prefix8 rows or scores were changed')
            if not retrieved and rag != policy:
                raise ValueError('No-proposal fallback must be exact PolicyK12')
            if not {spans(r) for r in old_rag} <= {spans(r) for r in rag}:
                raise ValueError('RAG K12 lost a frozen RAG K8 candidate')
            for method, rows in [('policy_K12', policy), ('rag_K12', rag)]:
                if len({spans(r) for r in rows}) != len(rows) or not 8 <= len(rows) <= 12:
                    raise ValueError('K12 mask count or uniqueness differs')
                for row in rows:
                    planning.validate_plan(len(sequence), .2, spans(row))
                winner = selected(sequence, rows, method)
                if winner['scorer_score'] < old_selected[pid, 'delepref_unlimited']['scorer_score']:
                    raise ValueError('Original incumbent was lost from a nested pool')
                winners.append(winner)
                pools.append(dict(protein_id=pid, method=method, requested_K=12, realized_K=len(rows),
                    prefix8_exactly_frozen=True, rows=rows, selection_order_raw_positions=[i+1 for i in selection_order(rows)],
                    unique_edited_sequences=len({edited_sequence(sequence, row) for row in rows})))
            diagnostics.append(dict(protein_id=pid, **cost, retrieval_rows_reused=len(retrieved),
                original_retrieved_unique=old_diag[pid]['retrieved_unique'], exact_PolicyK12_fallback=not retrieved,
                new_Scorer_candidate_evaluations=len(need_scoring), old_Scorer_values_reused_for_additions=reused,
                original8_Scorer_values_reused=True, new_Policy_WT_forward_calls=1, new_PLM_forwards=0,
                original_condition_seed=condition_seed, perturbation_scale=scale,
                seconds=time.monotonic()-before))
            print('K12_REAL_CPU_POLICY_AND_NEW_SCORER_DONE '+pid, flush=True)
    subset = {r['protein_id'] for r in frozen['structure_roster']}
    structural = [r for r in winners if r['protein_id'] in subset]
    if len(winners) != 200 or len(structural) != 80 or policy_calls != 100:
        raise ValueError('Frozen K12100/40 selected-output denominator differs')
    outputs = dict(candidate_pools=pools, all_selections=winners, selected_structure_outputs=structural,
                   generation_diagnostics=diagnostics,
                   integrity_audit=dict(all100_prefix8_rows_and_scores_exactly_preserved=True,
                     all_old_RAG_K8_masks_retained=True, no_new_retrieval_or_PLM=True,
                     pool_shortfalls_retained=sum(p['realized_K'] != 12 for p in pools),
                     original_Scorer_and_tie_rule_preserved=True,
                     source_rank_metadata_is_not_K12_rank=True))
    for name, value in outputs.items():
        write(out/(name+'.json'), value)
    report = dict(status='K12_CPU_CANDIDATES_AND_SELECTED_WINNERS_COMPLETE_NOT_STRUCTURAL_QUALITY',
        input_freeze_sha256=expected_freeze, job_id=os.environ['SLURM_JOB_ID'], CPU_WT=100,
        selected_outputs=200, fixed_structure_WT=40, logical_structural_winners_before_cache_reuse=80,
        new_Policy_WT_forwards=policy_calls, new_Scorer_batches=scorer_batches,
        new_Scorer_candidate_evaluations=scorer_candidates, new_PLM_forwards=0, new_training=False,
        new_structure_predictions=0, post_outcome_exploratory=True,
        seconds=time.monotonic()-start,
        outputs={name+'.json': sha(out/(name+'.json')) for name in outputs})
    write(out/'completion.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    prepare(args.root) if args.prepare else run(args.root, args.freeze_sha256)
