"""CUDA RNG hardware signature for Raygun's observed native noise shape.

No protein model is loaded or called. No sequence is generated or scored.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
BASE = Path('reports/recomb_confirm_v2_E10_noise_signature')
FREEZE = Path('configs/recomb_confirm_v2_E10_noise_signature_freeze.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(root):
    context = json.loads((root/'configs/recomb_confirm_v2_E10_ray_context_diagnostic_freeze.json').read_text())
    done = json.loads((root/'reports/recomb_confirm_v2_E10_ray_context_diagnostic/completion.json').read_text())
    if done['modes'][0]['sequence_sha256'] != done['modes'][1]['sequence_sha256']:
        raise ValueError('The current context control did not establish identical first-batch outputs')
    names = ['analysis/recomb_confirm_v2_E10_noise_signature.py', 'slurm/recomb_confirm_v2_E10_noise_signature.sbatch',
        'configs/recomb_confirm_v2_E10_ray_context_diagnostic_freeze.json',
        'reports/recomb_confirm_v2_E10_ray_context_diagnostic/completion.json']
    with (root/FREEZE).open('x') as stream:
        json.dump(dict(status='RAYGUN_LATENT_NOISE_SHAPE_HARDWARE_SIGNATURE_FROZEN',
            files={p: sha(root/p) for p in names}, seed=context['seed'], shape=[8, 50, 1280], dtype='float32',
            exact_native_operation='torch.randn_like(redstd), contiguous batch8/reduction50/dim1280',
            devices=['A40', 'A100'], protein_model_calls=0, generated_sequences=0, structure_predictions=0,
            no_actual_generator_noise_or_weights_changed=True), stream, indent=2)
    (root/BASE).mkdir(exist_ok=True)
    print(sha(root/FREEZE), flush=True)


def run(root, checksum, expected_device):
    import numpy as np
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Only a normal Slurm GPU allocation may run this diagnostic')
    if sha(root/FREEZE) != checksum:
        raise ValueError('Noise probe source freeze changed')
    f = json.loads((root/FREEZE).read_text())
    for p, h in f['files'].items():
        if sha(root/p) != h:
            raise ValueError('Frozen software diagnostic source changed: '+p)
    if expected_device not in f['devices'] or expected_device not in torch.cuda.get_device_name():
        raise ValueError('Noise signature is not on the requested permitted GPU class')
    out = root/BASE/expected_device
    out.mkdir(exist_ok=False)
    seed = int(f['seed'])
    template = torch.empty(tuple(f['shape']), device='cuda', dtype=torch.float32)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    before = hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    with torch.no_grad():
        value = torch.randn_like(template)
    torch.cuda.synchronize()
    after = hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    array = value.cpu().numpy()
    np.savez_compressed(out/'noise.npz', noise=array)
    properties = torch.cuda.get_device_properties(0)
    header = Path(torch.__file__).resolve().parent/'include/ATen/native/cuda/DistributionTemplates.h'
    result = dict(status='CUDA_NATIVE_NOISE_SIGNATURE_MEASURED_NO_MODEL_CALL', expected_device=expected_device,
        actual_GPU=torch.cuda.get_device_name(), GPU_multi_processor_count=properties.multi_processor_count,
        GPU_compute_capability=[properties.major, properties.minor], GPU_total_memory=properties.total_memory,
        torch_version=torch.__version__, CUDA_build=torch.version.cuda, seed=seed, shape=f['shape'],
        RNG_state_before_sha256=before, RNG_state_after_sha256=after,
        noise_tensor_sha256=hashlib.sha256(array.tobytes()).hexdigest(), artifact_sha256=sha(out/'noise.npz'),
        distribution_header_path=str(header), distribution_header_sha256=sha(header) if header.exists() else None,
        source_freeze_sha256=checksum, job_id=os.environ['SLURM_JOB_ID'], protein_model_calls=0,
        generated_sequences=0, scored_sequences=0, new_structure_predictions=0)
    with (out/'completion.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--prepare', action='store_true');parser.add_argument('--freeze-sha256')
    parser.add_argument('--device', choices=['A40', 'A100'])
    args = parser.parse_args()
    prepare(args.root) if args.prepare else run(args.root, args.freeze_sha256, args.device)
