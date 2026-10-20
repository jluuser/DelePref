"""Shared identity and atomic output helpers for the approved General100 run."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
BASE = Path('reports/recomb_general100_execution_20261017')
FREEZE = Path('configs/recomb_general100_baselines_freeze_20261017.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sequence_sha(sequence):
    return hashlib.sha256(sequence.encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def verify(root, checksum):
    if sha(root / FREEZE) != checksum:
        raise ValueError('General100 baseline freeze changed')
    frozen = json.loads((root / FREEZE).read_text())
    for path, digest in frozen['files'].items():
        target = Path(path) if Path(path).is_absolute() else root / path
        if sha(target) != digest:
            raise ValueError('frozen source/input changed: ' + path)
    return frozen


def validate_sequence(sequence, target_length):
    if len(sequence) != target_length or not sequence:
        raise ValueError('wrong generated target length')
    if set(sequence) - set('ACDEFGHIKLMNPQRSTVWYX'):
        raise ValueError('unsupported generated sequence characters')


def require_gpu():
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Slurm GPU allocation required')
    name = torch.cuda.get_device_name(0)
    if not any(kind in name for kind in ('A40', 'A100')):
        raise RuntimeError('authorized A40/A100 required: ' + name)
    return name
