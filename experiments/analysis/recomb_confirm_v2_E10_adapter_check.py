"""CPU equivalence proof for standalone request adapters, before timing GPUs."""
import argparse
import json
import os
from pathlib import Path
import time

from recomb_confirm_v2_diagnostics import read, sha, write
from recomb_confirm_v2_E9_completion import claim
from recomb_confirm_v2_E10_requests import HEAD_METHODS, head_request

BASE = Path('reports/recomb_confirm_v2_E10_adapter_check')
FREEZE = Path('configs/recomb_confirm_v2_E10_adapter_check_freeze.json')


def prepare(root):
    if (root/FREEZE).exists():
        raise FileExistsError('E10 adapter verification already frozen')
    files = ['analysis/recomb_confirm_v2_E10_adapter_check.py', 'analysis/recomb_confirm_v2_E10_requests.py',
        'analysis/recomb_confirm_v2_E9_completion.py', 'analysis/recomb_confirm_v2_diagnostics.py',
        'slurm/recomb_confirm_v2_E10_adapter_check.sbatch', 'configs/delepref_v8_baseline_freeze.json',
        'reports/recomb_confirm_v2_General100_model/encoding_completion.json',
        'reports/recomb_confirm_v2_General100_plans/completion.json',
        'reports/recomb_confirm_v2_General100_plans/candidate_scores.json',
        'reports/recomb_confirm_v2_General100_plans/all_selections.json',
        'reports/recomb_confirm_v2_General100_costs/cases.json',
        'reports/recomb_confirm_v2_general_native/encoder_inputs.json',
        'retrieval/recomb_confirm_v2_General100/profiles.json']
    parent = read(root/'configs/recomb_confirm_v2_General100_plans_freeze.json')
    for name, h in parent['files'].items():
        if sha(root/name) != h:
            raise ValueError('Original generation source/input changed: '+name)
    files += list(parent['files'])
    write(root/FREEZE, dict(status='E10_STANDALONE_ADAPTER_EQUIVALENCE_CHECK_FROZEN',
        files={n: sha(root/n) for n in sorted(set(files))}, WT=20, methods=list(HEAD_METHODS),
        main_budget=.2, additional_cached_request_budget=.3,
        original_frozen_pool_order_and_winner_mask_must_match=True,
        score_numeric_deltas_recorded_without_retuning=True,
        new_WT_encoder_calls=0, new_structure_predictions=0, new_training=False))
    (root/BASE).mkdir(parents=True, exist_ok=True)
    print(sha(root/FREEZE), flush=True)


def run(root, checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Adapter check requires CPU Slurm')
    if sha(root/FREEZE) != checksum:
        raise ValueError('Adapter check freeze differs')
    frozen = read(root/FREEZE)
    for p, h in frozen['files'].items():
        if sha(root/p) != h:
            raise ValueError('Adapter source/input changed: '+p)
    out = root/BASE
    claim(out/'execution_claim.json', dict(job_id=os.environ['SLURM_JOB_ID'], freeze_sha256=checksum))
    import torch
    import delepref_v8_runtime as rt
    import delepref_v8_main as parent
    torch.set_num_threads(4)
    start = time.monotonic()
    runtime, api = rt.load(root, device='cpu', cached_heads_only=True)
    path = root/'reports/recomb_confirm_v2_General100_model/wt_cache.pt'
    if sha(path) != read(root/'reports/recomb_confirm_v2_General100_model/encoding_completion.json')['cache_sha256']:
        raise ValueError('Original genuine WT embedding cache differs')
    cache = torch.load(path, map_location='cpu', weights_only=True)
    records = read(root/'reports/recomb_confirm_v2_general_native/encoder_inputs.json')
    by_record = {r['protein_id']: r for r in records}
    chunk = parent.cached_chunk(api, cache, records)
    profiles = read(root/'retrieval/recomb_confirm_v2_General100/profiles.json')
    old_pools = {}
    for row in read(root/'reports/recomb_confirm_v2_General100_plans/candidate_scores.json'):
        if row['generation_seed'] == 13:
            old_pools.setdefault((row['protein_id'], row['budget'], row['method']), []).append(row)
    old_selected = {(r['protein_id'], r['budget'], r['method']): r
        for r in read(root/'reports/recomb_confirm_v2_General100_plans/all_selections.json') if r['generation_seed'] == 13}
    audit, outputs = [], []
    with torch.inference_mode():
        for case in read(root/'reports/recomb_confirm_v2_General100_costs/cases.json'):
            pid = case['protein_id'];record = by_record[pid]
            for beta in [.2, .3]:
                for method in HEAD_METHODS:
                    key = pid, beta, method
                    if key not in old_pools:
                        continue
                    value = head_request(runtime, api, chunk, record, profiles[pid], method, beta)
                    old = sorted(old_pools[key], key=lambda r: r['raw_candidate_rank'])
                    new = value['candidate_pool']
                    old_masks = [r['deletion_mask_sha256'] for r in old]
                    new_masks = [r['deletion_mask_sha256'] for r in new]
                    winner_match = value['selected']['deletion_mask_sha256'] == old_selected[key]['deletion_mask_sha256']
                    sequence_match = value['selected']['edited_sequence_sha256'] == old_selected[key]['edited_sequence_sha256']
                    item = dict(protein_id=pid, budget=beta, method=method, exact_ordered_pool_masks=old_masks == new_masks,
                        exact_selected_mask=winner_match, exact_selected_sequence=sequence_match,
                        max_Scorer_abs_delta=max(abs(a['scorer_score']-b['scorer_score']) for a, b in zip(old, new)),
                        exact_score_values=all(a['scorer_score'] == b['scorer_score'] for a, b in zip(old, new)))
                    audit.append(item)
                    outputs.append(dict(protein_id=pid, budget=beta, **value))
                    if not item['exact_ordered_pool_masks'] or not winner_match or not sequence_match:
                        write(out/'failed_equivalence_audit.json', audit)
                        raise ValueError('Standalone scientific output differs; repair adapter before measuring: '+str(key))
            print('E10_CACHED_ADAPTER_EQUIVALENCE_DONE '+pid, flush=True)
    main = [r for r in audit if r['budget'] == .2]
    if len(main) != 80 or len({r['protein_id'] for r in main}) != 20:
        raise ValueError('Fixed cost20 four-method equivalence coverage differs')
    write(out/'head_adapter_equivalence.json', audit)
    write(out/'complete_adapter_outputs.json', outputs)
    done = dict(status='E10_ALL20_FOUR_METHOD_POOL_MASK_AND_SELECTED_SEQUENCE_EQUIVALENCE_PASS',
        WT=20, main_method_conditions=80, existing_budget30_conditions=sum(r['budget'] == .3 for r in audit),
        exact_score_conditions=sum(r['exact_score_values'] for r in audit),
        maximum_score_abs_delta=max(r['max_Scorer_abs_delta'] for r in audit),
        actual_CPU_Policy_WT_calls=len(audit), new_WT_encoder_calls=0, new_structure_predictions=0,
        freeze_sha256=checksum, job_id=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-start,
        outputs={n: sha(out/n) for n in ['head_adapter_equivalence.json', 'complete_adapter_outputs.json']})
    write(out/'completion.json', done)
    print(json.dumps(done), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    prepare(args.root) if args.prepare else run(args.root, args.freeze_sha256)
