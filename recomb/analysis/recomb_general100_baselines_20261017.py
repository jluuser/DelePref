"""Reuse validated native SCISOR/Raygun implementations on frozen missing inputs."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from recomb_general100_common_20261017 import ROOT, BASE, verify, write, sha, sequence_sha, require_gpu, validate_sequence


def run_raygun(root, frozen, shard, shards):
    source = '/public/nfs-data/keyuan.zhou/evocompress/results/formal_v9/external_raygun/tools/raygun_worker.py'
    spec = importlib.util.spec_from_file_location('reused_native_raygun_worker', source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    tasks = json.loads((root / BASE / 'raygun_tasks.json').read_text())
    out = root / BASE / 'raygun'
    out.mkdir(exist_ok=True)
    claim = out / ('claim_%d.json' % shard)
    if claim.exists():
        raise FileExistsError('Raygun shard already claimed; inspect existing job instead of resubmitting')
    write(claim, dict(job_id=os.environ['SLURM_JOB_ID'], shard=shard, shards=shards))
    started = time.monotonic()
    torch, esm, ray, encoder, alphabet, incompatible = native.load_raygun_models('cuda')
    contract = json.loads((root / BASE / 'raygun_checkpoint_load_contract.json').read_text())
    if sorted(incompatible.missing_keys) != sorted(contract['missing_keys']) or list(incompatible.unexpected_keys) != contract['unexpected_keys']:
        raise ValueError('Raygun checkpoint incompatibility differs from completed official-native run')
    print(json.dumps(dict(event='GENERAL100_RAYGUN_MODEL_LOADED', gpu=torch.cuda.get_device_name(0), shard=shard)), flush=True)
    completed = 0
    last_sequence, embedding = None, None
    for index, task in enumerate(tasks):
        if index % shards != shard:
            continue
        path = out / ('condition_%03d.json' % index)
        if path.exists():
            raise FileExistsError('condition already exists')
        sequence = task['sequence']
        seed = native.stable_seed(20260830, 'official-native', task['dataset'], task['protein_id'], task['budget'])
        random.seed(seed); np.random.seed(seed & 0xffffffff)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        before = time.monotonic()
        print(json.dumps(dict(event='GENERAL100_RAYGUN_FORWARD_START', index=index, protein_id=task['protein_id'], budget=task['budget'])), flush=True)
        if last_sequence != sequence:
            embedding, _ = native._esm_embeddings(torch, encoder, alphabet, [sequence], 'cuda')
            embedding = embedding[:1]
            last_sequence = sequence
        generated, per_runtime = native._generate_sequences(torch, encoder, alphabet, ray, embedding,
                     [task['target_length']]*500, [1.]*500, 'N/A', 'cuda', batch_size=8)
        if len(generated) != 500:
            raise ValueError('native500 count mismatch')
        for candidate in generated:
            validate_sequence(candidate, task['target_length'])
        generation_seconds = time.monotonic()-before
        scores, pll_seconds = native._pll_scores(torch, encoder, alphabet, generated, 'cuda', batch_size=8)
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite Raygun PLL')
        order = sorted(range(500), key=lambda i: (-scores[i], i))
        raw = out / ('raw_%03d.jsonl.gz' % index)
        with gzip.open(raw, 'xt', encoding='utf-8') as f:
            for i, candidate in enumerate(generated):
                f.write(json.dumps(dict(generation_index=i, sequence=candidate, adjusted_PLL=scores[i]))+'\n')
        candidates = [dict(generation_index=i, sequence=generated[i], sequence_sha256=sequence_sha(generated[i]),
                           pll_rank=rank+1, adjusted_PLL=scores[i]) for rank, i in enumerate(order[:50])]
        best = candidates[0]
        insertions, deletions, substitutions = native._edit_counts(sequence, best['sequence'])
        result = dict(task, method='Raygun_500_native', edited_sequence=best['sequence'],
                      edited_sequence_sha256=best['sequence_sha256'], generation_seed=seed,
                      raw_candidate_count=500, PLL_ranked_count=50, selection='native_adjusted_PLL_top1',
                      native_top50=candidates, generation_seconds=generation_seconds, pll_seconds=pll_seconds,
                      seconds=time.monotonic()-before, raw_sha256=sha(raw),
                      num_insertions=insertions, num_deletions=deletions, num_substitutions=substitutions,
                      job_id=os.environ['SLURM_JOB_ID'], actual_GPU=torch.cuda.get_device_name(0),
                      new_structure_predictions=0)
        write(path, result)
        completed += 1
        print(json.dumps(dict(event='GENERAL100_RAYGUN_CONDITION_COMPLETE', index=index, completed=completed,
                              seconds=result['seconds'])), flush=True)
    write(out / ('completion_%d.json' % shard), dict(status='RAYGUN_SHARD_COMPLETE', shard=shard, shards=shards,
          conditions=completed, seconds=time.monotonic()-started, job_id=os.environ['SLURM_JOB_ID']))


def run_scisor(root, frozen):
    import torch
    import delepref_v8_controls as controls
    import delepref_unlimited_planning as planning
    from delepref_v8_scisor_inference_import_v2 import load_native_class
    from delepref_v8_scisor_compat import patch_tokenizer
    from delepref_v8_scisor_mapping import mapped_plan
    tasks = json.loads((root / BASE / 'scisor_tasks.json').read_text())
    original = {r['protein_id']: r for r in json.loads((root / 'reports/delepref_v8_scisor_native_v3/native_plans.json').read_text())}
    out = root / BASE / 'scisor'
    out.mkdir(exist_ok=True)
    if (out / 'claim.json').exists():
        raise FileExistsError('SCISOR already claimed; inspect existing job')
    write(out / 'claim.json', dict(job_id=os.environ['SLURM_JOB_ID']))
    sys.path.insert(0, '/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
    cls = load_native_class()
    model = cls.load_from_checkpoint(frozen['scisor_assets']['checkpoint']['path'], map_location='cpu').to('cuda').eval()
    patch_tokenizer(model.tokenizer)
    model.p0 = torch.load(frozen['scisor_assets']['p0']['path'], map_location='cuda', weights_only=True)
    rate = 1/1.1
    model.alpha = lambda t: (1-t)**rate
    model.beta = lambda t: rate/(1-t)
    model.log_alpha = lambda t: -torch.log(model.alpha(t))
    print(json.dumps(dict(event='GENERAL100_SCISOR_MODEL_LOADED', GPU=torch.cuda.get_device_name(0))), flush=True)
    rows = []
    with torch.inference_mode():
        for index, task in enumerate(tasks):
            sequence, pid, beta = task['sequence'], task['protein_id'], task['budget']
            count = planning.delete_count(len(sequence), beta)
            pool = []
            for candidate_index in range(task['candidate_count']):
                if task['mode'] == 'same_scorer_K8' and candidate_index == 0:
                    old = original[pid]
                    if old['wildtype_sequence'] != sequence or old['budget'] != beta:
                        raise ValueError('SCISOR native cache not exact')
                    pool.append(dict(old, candidate_index=0, reused_native=True))
                    continue
                seed = controls.condition_seed(13, pid, sequence, beta) + candidate_index*1_000_003
                random.seed(seed); np.random.seed(seed % (2**32-1)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                ids = model.tokenizer(sequence).input_ids
                if len(ids) != len(sequence)+2:
                    raise ValueError('SCISOR token count mismatch')
                x = torch.tensor(ids, device='cuda').unsqueeze(0)
                before = time.monotonic()
                print(json.dumps(dict(event='GENERAL100_SCISOR_FORWARD_START', condition=index, candidate_index=candidate_index)), flush=True)
                output, preserved = model.shrink_sequence(x, torch.tensor([count], device='cuda'), temperature=1.)
                result = mapped_plan(sequence, beta, preserved[0], output[0].detach().cpu().tolist(), model.tokenizer)
                validate_sequence(result['edited_sequence'], task['target_length'])
                mask = planning.mask_for(len(sequence), result['deleted_intervals'])
                pool.append(dict(result, candidate_index=candidate_index, condition_seed=seed,
                                 generation_seed=13, seconds=time.monotonic()-before,
                                 edited_sequence_sha256=sequence_sha(result['edited_sequence']),
                                 deletion_mask_sha256=hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()))
            row = dict(task, candidates=pool, method='SCISOR_U90_S_native' if task['mode']=='native' else 'SCISOR_K8_same_scorer',
                       job_id=os.environ['SLURM_JOB_ID'], actual_GPU=torch.cuda.get_device_name(0))
            if task['mode'] == 'native':
                row.update(pool[0])
            write(out / ('condition_%03d.json' % index), row)
            rows.append(row)
            print(json.dumps(dict(event='GENERAL100_SCISOR_CONDITION_COMPLETE', completed=len(rows), K=len(pool))), flush=True)
    write(out / 'all_conditions.json', rows)
    write(out / 'completion.json', dict(status='SCISOR_MISSING_GENERATION_COMPLETE', conditions=len(rows),
          actual_new_sampler_calls=200, native_conditions=60, K8_conditions=20, new_structure_predictions=0,
          all_exact_budgets_and_explicit_coordinates=True, job_id=os.environ['SLURM_JOB_ID']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['raygun', 'scisor'], required=True)
    p.add_argument('--freeze-sha256', required=True)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    a = p.parse_args()
    require_gpu()
    f = verify(a.root, a.freeze_sha256)
    if a.method == 'raygun':
        run_raygun(a.root, f, a.shard_index, a.shards)
    else:
        run_scisor(a.root, f)
