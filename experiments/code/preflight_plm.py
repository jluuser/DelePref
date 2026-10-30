"""CPU-only token equivalence and meta-device checkpoint compatibility checks."""
import argparse
import json
from pathlib import Path
import time
import torch
from tokenizers import Tokenizer
from safetensors import safe_open
from common import AUTHORIZED_ROOT, read_jsonl, sha_file, identity, write_json
from cache_plm import candidates
from adapters.esmc_runtime import tokenizer_profile
from adapters.esmc_loader import ESMCSequenceTokenizer, normalize_esmc_state_key


def run(protocol_path, manifest_path, output):
    start=time.perf_counter();torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise RuntimeError('CPU preflight must not initialize CUDA')
    protocol=json.loads(Path(protocol_path).read_text())
    seal=json.loads((AUTHORIZED_ROOT/'configs/execution_freeze.json').read_text())
    if sha_file(manifest_path)!=seal['input_hashes']['manifests/development.jsonl']:
        raise ValueError('development manifest identity changed')
    snapshot=Path(protocol['weights']['esmc']);tokenizer=ESMCSequenceTokenizer(max_length=1022)
    profile=tokenizer_profile(tokenizer);reference=Tokenizer.from_file(str(snapshot/'tokenizer.json'))
    if tokenizer._token_to_id!=reference.get_vocab():raise ValueError('snapshot vocabulary differs')
    rows=candidates(read_jsonl(manifest_path));token_hashes=[]
    for row in rows:
        actual=tokenizer(row['sequence'])['input_ids'][0].tolist()
        expected=reference.encode(row['sequence']).ids
        if actual!=expected:raise ValueError('token mismatch: '+row['sequence_sha256'])
        if len(actual)!=row['length']+2:raise ValueError('special-token/length mismatch')
        token_hashes.append(identity([row['sequence_sha256'],actual]))
    from esm.models.esmc import ESMC
    import inspect
    kwargs=dict(d_model=1152,n_heads=18,n_layers=36,tokenizer=tokenizer)
    if 'use_flash_attn' in inspect.signature(ESMC).parameters:kwargs['use_flash_attn']=False
    with torch.device('meta'):
        model=ESMC(**kwargs).eval()
    expected_state=model.state_dict()
    if not all(v.is_meta for v in expected_state.values()):raise ValueError('model construction escaped meta device')
    checkpoint=snapshot/'model.safetensors';actual_shapes={};skipped=[]
    with safe_open(str(checkpoint),framework='pt',device='cpu') as handle:
        for key in handle.keys():
            target=normalize_esmc_state_key(key)
            if target is None:skipped.append(key);continue
            if target in actual_shapes:raise ValueError('duplicate mapped checkpoint key')
            actual_shapes[target]=tuple(handle.get_slice(key).get_shape())
    missing=sorted(set(expected_state)-set(actual_shapes));unexpected=sorted(set(actual_shapes)-set(expected_state))
    mismatched=[k for k in set(expected_state)&set(actual_shapes) if tuple(expected_state[k].shape)!=actual_shapes[k]]
    if missing or unexpected or mismatched:raise ValueError(f'checkpoint compatibility failed: {missing[:5]}, {unexpected[:5]}, {mismatched[:5]}')
    if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
    report={'status':'PLM_CPU_PREFLIGHT_PASSED','sequences_verified':len(rows),'token_ids_digest':identity(token_hashes),
            'manifest_sha256':sha_file(manifest_path),'protocol_sha256':sha_file(protocol_path),
            'tokenizer_snapshot_sha256':sha_file(snapshot/'tokenizer.json'),'runtime_profile':profile,
            'checkpoint_tensors_verified':len(actual_shapes),'skipped_checkpoint_keys':skipped,
            'checkpoint_bytes':checkpoint.stat().st_size,'model_device':'meta','cuda_initialized':False,
            'model_weights_loaded':False,'model_forward_calls':0,'seconds':time.perf_counter()-start}
    write_json(output,report);print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('protocol','manifest','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();run(args.protocol,args.manifest,args.output)
