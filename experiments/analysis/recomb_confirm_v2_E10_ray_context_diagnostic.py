"""First-native-batch context audit for an observed timed Raygun mismatch.

This is not a latency trial or a scientific winner search. It uses the frozen
first WT, original seed, weights, noise1 and exact upstream first batch of8.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
BASE = Path('reports/recomb_confirm_v2_E10_ray_context_diagnostic')
FREEZE = Path('configs/recomb_confirm_v2_E10_ray_context_diagnostic_freeze.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def prepare(root):
    import gzip
    f = read(root/'configs/recomb_confirm_v2_E10_timing_padding_v2_freeze.json')
    case = f['cases'][0]
    tasks = read(root/'reports/recomb_confirm_v2_General100_raygun_v2/raygun_tasks.json')
    index = next(i for i, row in enumerate(tasks) if row['protein_id'] == case['protein_id'] and row['budget'] == .2)
    old_dir = root/'reports/recomb_confirm_v2_General100_raygun_v2/raygun'
    raw = old_dir/('raw_%03d.jsonl.gz' % index)
    original = [json.loads(line) for line in gzip.open(raw, 'rt')]
    new_dir = root/'reports/recomb_confirm_v2_E10_timing_padding_v2/requests'/('%02d_%s' % (0, case['WT_sha256'][:12]))/'raygun_native'
    timed = read(new_dir/'cold/candidate_pool.json')
    previous = read(old_dir/('condition_%03d.json' % index))
    actual = read(new_dir/'cold/selected_output.json')
    if hashlib.sha256(tasks[index]['sequence'].encode()).hexdigest() != case['WT_sha256'] or previous['generation_seed'] != actual['seed']:
        raise ValueError('Observed mismatch is not a matched scientific input/seed')
    files = ['analysis/recomb_confirm_v2_E10_ray_context_diagnostic.py', 'slurm/recomb_confirm_v2_E10_ray_context_diagnostic.sbatch',
        'analysis/recomb_confirm_v2_E10_requests.py', 'analysis/recomb_confirm_v2_E10_fidelity.py',
        'analysis/recomb_general100_baselines_20261017.py',
        'configs/recomb_confirm_v2_E10_timing_padding_v2_freeze.json',
        str(raw), str(new_dir/'cold/candidate_pool.json'), str(new_dir/'cold/selected_output.json'),
        str(new_dir/'actual_cold_native_WT_cache.pt')]
    source = read(root/'configs/recomb_confirm_v2_General100_raygun_freeze.json')
    assets = {v['path']: v['sha256'] for v in source['assets'].values() if isinstance(v, dict) and 'path' in v and 'sha256' in v}
    frozen = dict(status='OBSERVED_RAYGUN_FIRST_BATCH_CONTEXT_DIAGNOSTIC_FROZEN',
        files={p: sha(root/p) for p in files}, native_external_code=source['external_code'], assets=assets,
        case=case, budget=.2, seed=actual['seed'], target_length=tasks[index]['target_length'],
        original_first8=[hashlib.sha256(x['sequence'].encode()).hexdigest() for x in original[:8]],
        timing_first8=[x['edited_sequence_sha256'] for x in timed[:8]],
        modes=['original_helper_no_outer_context', 'timing_outer_inference_mode'],
        native_batch=8, native_noise=1., native_temperature='N/A', structure_predictions=0,
        no_score_or_selector_call=True, no_scientific_or_timing_output_replacement=True,
        Raygun_python=f['Raygun_python'])
    write(root/FREEZE, frozen)
    (root/BASE).mkdir(exist_ok=True)
    print(sha(root/FREEZE), flush=True)


def child(root, checksum, mode):
    import random
    import numpy as np
    import torch
    from recomb_confirm_v2_E10_requests import load_native
    from recomb_confirm_v2_E10_fidelity import tensor_identity
    if sha(root/FREEZE) != checksum:
        raise ValueError('Diagnostic source freeze differs')
    f = read(root/FREEZE)
    if mode not in f['modes']:
        raise ValueError('Unknown execution context')
    if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name():
        raise RuntimeError('Diagnostic retains the existing A100 protocol')
    torch.set_num_threads(4)
    state = load_native(root, 'raygun_native')
    native, ray, encoder, alphabet = (state[k] for k in ['native', 'model', 'encoder', 'alphabet'])
    seed = int(f['seed'])
    random.seed(seed);np.random.seed(seed & 0xffffffff);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    def rng():
        return dict(CPU=sha_bytes(torch.get_rng_state()), CUDA=sha_bytes(torch.cuda.get_rng_state()))
    def sha_bytes(value):
        return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
    calls = []
    def hook(module, inputs):
        calls.append(dict(inference_mode=torch.is_inference_mode_enabled(), grad_enabled=torch.is_grad_enabled(),
                          training=module.training))
    handle = encoder.register_forward_pre_hook(hook)
    context = torch.inference_mode() if mode == 'timing_outer_inference_mode' else contextlib.nullcontext()
    with context:
        before = rng()
        embedding, _ = native._esm_embeddings(torch, encoder, alphabet, [f['case']['wildtype_sequence']], 'cuda')
        torch.cuda.synchronize()
        after_embedding = rng()
        identity = tensor_identity(embedding)
        generated, _ = native._generate_sequences(torch, encoder, alphabet, ray, embedding[:1],
            [f['target_length']]*8, [1.]*8, 'N/A', 'cuda', batch_size=8)
        torch.cuda.synchronize()
        final_rng = rng()
    handle.remove()
    if len(generated) != 8 or any(len(s) != f['target_length'] for s in generated):
        raise ValueError('First native diagnostic batch differs')
    identities = [hashlib.sha256(s.encode()).hexdigest() for s in generated]
    write(root/BASE/(mode+'.json'), dict(mode=mode, protein_id=f['case']['protein_id'], seed=seed,
        generated=generated, sequence_sha256=identities, WT_embedding_identity=identity,
        input_preparation_rng_before=before, input_preparation_rng_after=after_embedding, after_generation_rng=final_rng,
        ESM_forward_contexts=calls, ray_training=ray.training, encoder_training=encoder.training,
        first8_match_original=sum(a == b for a, b in zip(identities, f['original_first8'])),
        first8_match_timed=sum(a == b for a, b in zip(identities, f['timing_first8'])),
        actual_native_generated_sequences=8, no_scoring_or_structure_calls=True,
        actual_GPU=torch.cuda.get_device_name(), freeze_sha256=checksum))


def execute(root, checksum):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU diagnostics only via Slurm')
    if sha(root/FREEZE) != checksum:
        raise ValueError('Diagnostic freeze differs')
    f = read(root/FREEZE)
    for p, expected in {**f['files'], **f['native_external_code'], **f['assets']}.items():
        if sha(root/p) != expected:
            raise ValueError('Frozen native diagnostic source/asset changed: '+p)
    write(root/BASE/'execution_claim.json', dict(job_id=os.environ['SLURM_JOB_ID'], freeze_sha256=checksum))
    env = dict(os.environ, PYTHONPATH='/public/nfs-data/keyuan.zhou/evocompress/results/formal_v9/external_raygun/tools/fair_esm_site:/public/nfs-data/keyuan.zhou/evocompress/external/raygun/repo:'+os.environ.get('PYTHONPATH', ''), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
    for mode in f['modes']:
        with (root/BASE/(mode+'.log')).open('x') as stream:
            subprocess.run([f['Raygun_python'], '-u', str(root/'analysis/recomb_confirm_v2_E10_ray_context_diagnostic.py'),
                '--root', str(root), '--freeze-sha256', checksum, '--child-mode', mode], env=env,
                stdout=stream, stderr=subprocess.STDOUT, check=True)
    rows = [read(root/BASE/(mode+'.json')) for mode in f['modes']]
    names = [mode+'.json' for mode in f['modes']]
    write(root/BASE/'completion.json', dict(status='OBSERVED_RAYGUN_CONTEXT_PREFIX8_DIAGNOSTIC_COMPLETE',
        freeze_sha256=checksum, actual_native_generated_sequences=16, scored_sequences=0, new_structure_predictions=0,
        modes=[{k: v for k, v in row.items() if k not in ['generated']} for row in rows],
        outputs={n: sha(root/BASE/n) for n in names}, job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(read(root/BASE/'completion.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--prepare', action='store_true');parser.add_argument('--freeze-sha256');parser.add_argument('--child-mode')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root)
    elif args.child_mode:
        child(args.root, args.freeze_sha256, args.child_mode)
    else:
        execute(args.root, args.freeze_sha256)
