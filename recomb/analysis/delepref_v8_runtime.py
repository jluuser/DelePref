"""Explicit legacy-checkpoint compatibility; original trained heads, unchanged bytes."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

import torch
import yaml

CHECKPOINT_SHA = '1bc7694be77e24e4b671773be39331daed241dcfe101fd6d2d26f8e23fe7ff4d'
CONFIG_SHA = '433652e8e13986cd5d265f6b4f12d534d9346e4aa2995b09945e3105d1c2b365'
LEGACY_SCHEMA = 'student_v3_evaluation_checkpoint_v1'
LEGACY_VERSION = 'strict_budget_student_v3_final_readiness_v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def load(root, device='cuda', cached_heads_only=False):
    registry = json.loads((root/'provenance/original_delepref_assets_20261016.json').read_text())
    checkpoint = Path(registry['checkpoint']['path'])
    config_path = root/'configs/delepref_original_restored.yaml'
    if sha(checkpoint) != CHECKPOINT_SHA or sha(config_path) != CONFIG_SHA:
        raise ValueError('original checkpoint/config identity changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (payload['schema_version'], payload['student_version'], payload['selected_step']) != (LEGACY_SCHEMA, LEGACY_VERSION, 4000):
        raise ValueError('unrecognized legacy checkpoint')
    config = yaml.safe_load(config_path.read_text())
    canonical = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if canonical != payload['config_sha256']:
        raise ValueError('original canonical configuration mismatch')
    public = Path(registry['public_code'])
    vendor = Path(registry['source_root'])/config['model']['esm_runtime_vendor']
    for p in (public/'src', public/'scripts', vendor):
        sys.path.insert(0, str(p))
    api = importlib.import_module('evaluation_runtime')
    from delepref.student_model import FrozenWTEncoder, DeletionPreferenceStudent
    if cached_heads_only:
        class CachedOnlyEncoder(torch.nn.Module):
            def forward(self, *args, **kwargs):
                raise RuntimeError('cache-only head runner cannot encode or fabricate sequences')
        mc, pc = config['model'], config['policy']
        encoder = FrozenWTEncoder(CachedOnlyEncoder(), None, hidden_size=1152, max_length=1022)
        model = DeletionPreferenceStudent(encoder, hidden_size=1152,
            scorer_hidden_dim=mc['scorer_hidden_dim'], span_feature_dim=mc['span_feature_dim'],
            budget_hidden_dim=pc['budget_hidden_dim'], policy_hidden_dim=pc['policy_hidden_dim'],
            policy_span_hidden_dim=pc['span_hidden_dim'], qmax=pc['qmax'],
            min_span_len=pc['min_span_len'], max_span_len_fraction=pc['max_span_len_fraction'], dropout=mc['dropout'])
    else:
        if device != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('real original ESMC requires CUDA; no encoder substitution')
        from train_student_pilot import build_delepref
        model = build_delepref(config, torch.device(device))
    missing, extra = model.load_state_dict(payload['student_head_state_dict'], strict=False)
    if extra or [k for k in missing if not k.startswith('encoder.')]:
        raise ValueError('trained Policy/Scorer parameter mismatch')
    if not all(torch.equal(model.state_dict()[k].detach().cpu(), v) for k,v in payload['student_head_state_dict'].items()):
        raise ValueError('trained-head values changed during load')
    model.eval()
    runtime = api.EvaluationRuntime(model=model, config=config, checkpoint=payload,
        device=torch.device(device), source_checkpoint_sha256=payload['source_checkpoint_sha256'],
        selected_step=4000, encoder_microbatch_size=int(config['batching']['encoder_microbatch_size']), k=8)
    return runtime, api
