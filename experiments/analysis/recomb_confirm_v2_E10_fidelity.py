"""Isolate encoding layout effects before full request latency measurement.

Original weights, dtype, profile and planning/scoring functions remain fixed.
The padded single-query path is a diagnostic control, never a new main protocol.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from recomb_confirm_v2_diagnostics import read, sha, write
from recomb_confirm_v2_E9_completion import claim, digest

BASE = Path('reports/recomb_confirm_v2_E10_fidelity')
FREEZE = Path('configs/recomb_confirm_v2_E10_fidelity_freeze.json')
METHODS = ('delepref_unlimited', 'rag_unlimited', 'gap_frequency', 'terminal')


def microbatch_layout(records, size=32):
    unique = {}
    for row in records:
        sequence = row['wildtype_sequence']
        key = row['protein_id'], hashlib.sha256(sequence.encode()).hexdigest()
        unique[key] = sequence
    ordered = sorted(unique.items())
    return [dict(protein_ids=[key[0] for key, sequence in ordered[i:i+size]],
                 residue_lengths=[len(sequence) for key, sequence in ordered[i:i+size]],
                 padded_token_width=max(len(sequence)+2 for key, sequence in ordered[i:i+size]))
            for i in range(0, len(ordered), size)]


def padding_count(current, target):
    if current < 2 or target < current:
        raise ValueError('Diagnostic padding cannot truncate or shorten a query')
    return target-current


def compare_pools(reference, changed):
    def masks(pool):
        return {r['deletion_mask_sha256']: r for r in pool}
    a, b = masks(reference), masks(changed)
    choose = lambda pool: min(pool, key=lambda r: (-float(r['scorer_score']),
        tuple(map(tuple, json.loads(r['deleted_intervals'])))))['deletion_mask_sha256']
    common = sorted(set(a) & set(b))
    return dict(reference_K=len(reference), changed_K=len(changed),
        common_masks=len(common), introduced_masks=sorted(set(b)-set(a)),
        lost_masks=sorted(set(a)-set(b)), winner_mask_changed=choose(reference) != choose(changed),
        common_mask_score_max_abs_delta=max([abs(float(a[k]['scorer_score'])-float(b[k]['scorer_score'])) for k in common], default=None))


def tensor_identity(value):
    import torch
    value = value.detach().cpu().contiguous()
    return dict(dtype=str(value.dtype), shape=list(value.shape),
        bytes_sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def tensor_delta(reference, changed):
    import torch
    if reference.shape != changed.shape or reference.dtype != changed.dtype:
        raise ValueError('Encoding comparison shape/dtype differs')
    a, b = reference.float(), changed.float()
    diff = b-a
    return dict(exact_equal=bool(torch.equal(reference, changed)),
        differing_elements=int((reference != changed).sum()), max_abs=float(diff.abs().max()),
        mean_abs=float(diff.abs().mean()), relative_L2=float(diff.norm()/a.norm().clamp_min(1e-30)))


def replay(root, runtime, api, record, embedding, profile):
    import delepref_v8_main as parent
    import delepref_unlimited_scorer as scoring
    import delepref_unlimited_planning as planning
    from delepref_v8_controls import terminal_proposals
    from recomb_gap_frequency_control_20261017 import proposals as gap_proposals
    from recomb_general100_mechanisms_20261017 import select_rows
    pid, sequence = record['protein_id'], record['wildtype_sequence']
    cache = dict(embeddings={pid: embedding}, sequences={pid: sequence})
    chunk = parent.cached_chunk(api, cache, [record])
    rows, winners, diagnostic = parent.score_condition(runtime, api, chunk, record, profile, 13)
    original = sorted([r for r in rows if r['method'] == METHODS[0]], key=lambda r: r['raw_candidate_rank'])
    native = [dict(spans=tuple(map(tuple, json.loads(r['deleted_intervals']))), kind=r['generation_kind'], row=r) for r in original]
    logits = runtime.model.propose_from_embeddings([embedding], [.2])[0].residue_logits.detach().cpu().double().numpy()
    gaps, gap_log = gap_proposals(logits, .2, profile)
    mixed, fallback = planning.mix_candidates(native, gaps)
    gap = [dict(r) for r in original] if fallback else scoring.score_candidates(runtime, api, chunk,
        parent.make_rows(api, original[0], mixed, len(sequence), logits))
    terminal = scoring.score_candidates(runtime, api, chunk,
        parent.make_rows(api, original[0], terminal_proposals(len(sequence), .2), len(sequence), logits))
    for method, pool in [('gap_frequency', gap), ('terminal', terminal)]:
        rows.extend(dict(r, method=method) for r in pool)
        winners.append(select_rows(api, sequence, pool, method))
    return dict(pools={method: [r for r in rows if r['method'] == method] for method in METHODS},
        winners={r['method']: r for r in winners}, logits=logits.tolist(),
        diagnostic=diagnostic, gap_diagnostic=gap_log,
        real_CPU_Policy_WT_calls=2)


def prepare(root):
    if (root/FREEZE).exists():
        raise FileExistsError('E10 fidelity already frozen; inspect its exact job')
    cases = read(root/'reports/recomb_confirm_v2_General100_costs/cases.json')
    if len(cases) != 20 or len({c['protein_id'] for c in cases}) != 20:
        raise ValueError('Existing fixed cost20 roster differs')
    names = ['analysis/recomb_confirm_v2_E10_fidelity.py', 'analysis/tests/test_recomb_confirm_v2_E10_fidelity.py',
        'slurm/recomb_confirm_v2_E10_fidelity.sbatch', 'analysis/recomb_confirm_v2_E9_completion.py',
        'analysis/recomb_confirm_v2_diagnostics.py', 'analysis/delepref_v8_main.py',
        'analysis/delepref_v8_runtime.py', 'analysis/delepref_v8_rag_unlimited.py',
        'analysis/delepref_unlimited_planning.py', 'analysis/delepref_unlimited_scorer.py',
        'analysis/recomb_gap_frequency_control_20261017.py', 'analysis/recomb_general100_mechanisms_20261017.py',
        'configs/delepref_v8_baseline_freeze.json', 'configs/delepref_original_restored.yaml',
        'configs/recomb_confirm_v2_General100_GPU_costs_freeze.json',
        'reports/recomb_confirm_v2_General100_costs/cases.json',
        'reports/recomb_confirm_v2_General100_model/encoding_completion.json',
        'reports/recomb_confirm_v2_general_native/encoder_inputs.json',
        'reports/recomb_confirm_v2_General100_plans/candidate_scores.json',
        'reports/recomb_confirm_v2_General100_GPU_costs/esmc/per_WT_timings.json',
        'retrieval/recomb_confirm_v2_General100/profiles.json']
    original = read(root/'configs/delepref_v8_baseline_freeze.json')
    old_cost = read(root/'configs/recomb_confirm_v2_General100_GPU_costs_freeze.json')
    for p, h in old_cost['files'].items():
        if sha(root/p) != h:
            raise ValueError('Original fixed cost20 source/input changed: '+p)
    esmc_assets = {p: h for p, h in old_cost['assets_sha256'].items() if '/esmc_600m/snapshot/' in p}
    if not esmc_assets:
        raise ValueError('Original ESMC assets are not bound')
    for p, h in {**original['inputs'], **original['code'], **esmc_assets}.items():
        if sha(p if Path(p).is_absolute() else root/p) != h:
            raise ValueError('Original dependency changed: '+p)
    write(root/FREEZE, dict(status='E10_FIXED20_BATCH_LAYOUT_FIDELITY_DIAGNOSIS_FROZEN',
        files={n: sha(root/n) for n in names}, original_dependencies={**original['inputs'], **original['code']},
        esmc_assets=esmc_assets, cases=cases, fixed_WT=20, hardware='A100_80GB',
        original_microbatch_size=32, original_dtype='bfloat16', original_generation_functions_unchanged=True,
        compare=['old_scientific_cache', 'fresh_original100_batch', 'fresh_single_WT', 'fresh_single_original_padding_width'],
        repeated_single_query_first_fixed_case=1, no_numeric_threshold_or_positive_gate=True,
        no_scientific_cache_or_winner_replacement=True, new_structure_predictions=0, new_training=False))
    (root/BASE).mkdir(parents=True, exist_ok=True)
    print(sha(root/FREEZE), flush=True)


def run(root, checksum):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('E10 diagnostic requires a Slurm GPU allocation')
    if sha(root/FREEZE) != checksum:
        raise ValueError('E10 fidelity freeze differs')
    frozen = read(root/FREEZE)
    for p, h in {**frozen['files'], **frozen['original_dependencies'], **frozen['esmc_assets']}.items():
        if sha(p if Path(p).is_absolute() else root/p) != h:
            raise ValueError('E10 frozen dependency changed: '+p)
    out = root/BASE
    claim(out/'execution_claim.json', dict(job_id=os.environ['SLURM_JOB_ID'], freeze_sha256=checksum))
    import torch
    import delepref_v8_runtime as rt
    import delepref_v8_main as parent
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('E10 comparison retains the original A100 protocol')
    torch.set_num_threads(4)
    start = time.monotonic()
    gpu, api = rt.load(root)
    from delepref.preference_core import residue_embeddings_from_backend
    cpu, cpuapi = rt.load(root, device='cpu', cached_heads_only=True)
    cache_file = root/'reports/recomb_confirm_v2_General100_model/wt_cache.pt'
    if sha(cache_file) != read(root/'reports/recomb_confirm_v2_General100_model/encoding_completion.json')['cache_sha256']:
        raise ValueError('Original scientific embedding identity differs')
    old = torch.load(cache_file, map_location='cpu', weights_only=True)
    records = read(root/'reports/recomb_confirm_v2_general_native/encoder_inputs.json')
    by_record = {r['protein_id']: r for r in records}
    profiles = read(root/'retrieval/recomb_confirm_v2_General100/profiles.json')
    layout = microbatch_layout(records, gpu.encoder_microbatch_size)
    if len(records) != 100 or gpu.encoder_microbatch_size != 32:
        raise ValueError('Original scientific batch recipe differs')
    widths = {pid: b['padded_token_width'] for b in layout for pid in b['protein_ids']}
    counter = {'batches': 0}
    def count_forward(module, inputs, output):
        counter['batches'] += 1
    hook = gpu.model.encoder.backend.register_forward_hook(count_forward)
    original_pools = {}
    for r in read(root/'reports/recomb_confirm_v2_General100_plans/candidate_scores.json'):
        if r['budget'] == .2 and r['generation_seed'] == 13:
            original_pools.setdefault((r['protein_id'], r['method']), []).append(r)
    results, replays, saved = [], [], {}
    with torch.inference_mode():
        batch = api.encode_chunk(gpu, records)
        torch.cuda.synchronize()
        batch_embeddings = {k.protein_id: v.detach().cpu() for k, v in batch.encoded.embeddings.items()}
        del batch
        for i, case in enumerate(frozen['cases']):
            pid = case['protein_id']; record = by_record[pid]; seq = record['wildtype_sequence']
            if hashlib.sha256(seq.encode()).hexdigest() != case['WT_sha256']:
                raise ValueError('Fixed20 WT identity differs')
            single_chunk = api.encode_chunk(gpu, [record]);torch.cuda.synchronize()
            single = next(iter(single_chunk.encoded.embeddings.values())).detach().cpu()
            del single_chunk
            tokenizer = gpu.model.encoder.tokenizer
            class PaddedTokenizer:
                def __call__(self, *args, **kwargs):
                    encoded = tokenizer(*args, **kwargs)
                    current = int(encoded['input_ids'].shape[1])
                    extra = padding_count(current, widths[pid])
                    result = {}
                    for key, value in encoded.items():
                        fill = int(tokenizer.pad_token_id) if key == 'input_ids' else (1 if key == 'special_tokens_mask' else 0)
                        result[key] = torch.nn.functional.pad(value, (0, extra), value=fill) if value.ndim == 2 else value
                    return result
            padded = residue_embeddings_from_backend(gpu.model.encoder.backend, [seq], gpu.device,
                tokenizer=PaddedTokenizer(), max_length=1022, freeze=True)[0].detach().cpu()
            torch.cuda.synchronize()
            variants = dict(old_scientific_cache=old['embeddings'][pid],
                fresh_original100_batch=batch_embeddings[pid], fresh_single_WT=single,
                fresh_single_original_padding_width=padded)
            detail = dict(protein_id=pid, WT_sha256=case['WT_sha256'], length=len(seq),
                original_batch_token_width=widths[pid], single_token_width=len(seq)+2,
                tensor_identities={name: tensor_identity(v) for name, v in variants.items()},
                embedding_comparisons={name: tensor_delta(variants['old_scientific_cache'], v) for name, v in variants.items()},
                method_comparisons={})
            if i == 0:
                repeated = api.encode_chunk(gpu, [record]);torch.cuda.synchronize()
                repeat_embedding = next(iter(repeated.encoded.embeddings.values())).detach().cpu()
                detail['single_repeat_comparison'] = tensor_delta(single, repeat_embedding)
                del repeated
            for name, embedding in variants.items():
                replayed = replay(root, cpu, cpuapi, record, embedding, profiles[pid])
                detail['method_comparisons'][name] = {
                    method: compare_pools(original_pools[pid, method], replayed['pools'][method]) for method in METHODS}
                replays.append(dict(protein_id=pid, embedding_condition=name, **replayed))
            saved[pid] = variants
            results.append(detail)
            write(out/'progress.json', dict(completed_WT=len(results), fixed_WT=20, actual_encoder_batches=counter['batches']))
            print('E10_BATCH_FIDELITY_DONE '+pid, flush=True)
    hook.remove()
    expected_calls = len(layout)+2*20+1
    if len(results) != 20 or counter['batches'] != expected_calls:
        raise ValueError('Measured E10 fixed20 diagnostic forward coverage differs')
    torch.save(saved, out/'isolated_embedding_controls.pt')
    write(out/'batch_layout.json', layout)
    write(out/'per_WT_fidelity.json', results)
    write(out/'full_generation_replays.json', replays)
    summary = {name: {method: dict(WT=20,
        winner_mask_matches_original=sum(not r['method_comparisons'][name][method]['winner_mask_changed'] for r in results),
        complete_pool_mask_matches_original=sum(r['method_comparisons'][name][method]['common_masks'] == 8 for r in results))
        for method in METHODS} for name in results[0]['method_comparisons']}
    write(out/'fidelity_summary.json', summary)
    names = ['isolated_embedding_controls.pt', 'batch_layout.json', 'per_WT_fidelity.json', 'full_generation_replays.json', 'fidelity_summary.json']
    done = dict(status='E10_ALL20_ISOLATED_BATCH_AND_PADDING_FIDELITY_MEASUREMENTS_COMPLETE',
        fixed_WT=20, actual_encoder_batches=counter['batches'], encoder_sequence_instances=100+20+20+1,
        CPU_Policy_WT_calls=20*4*2, no_new_structure_predictions=True, new_training=False,
        no_scientific_output_replacement=True, dtype=str(next(gpu.model.encoder.backend.parameters()).dtype),
        actual_GPU=torch.cuda.get_device_name(), freeze_sha256=checksum,
        seconds=time.monotonic()-start, outputs={n: sha(out/n) for n in names}, job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json', done)
    print(json.dumps(done), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    prepare(args.root) if args.prepare else run(args.root, args.freeze_sha256)
