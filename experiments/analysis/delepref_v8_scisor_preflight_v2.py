"""Offline CPU loading/tokenizer check of the verified historical SCISOR model."""
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch
import delepref_v8_runtime as rt
from delepref_v8_baseline import ROOT,write
from delepref_v8_scisor_compat import patch_tokenizer


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm allocation required')
    inventory=json.loads((ROOT/'reports/delepref_v8_scisor_inventory.json').read_text())
    for p,h in inventory['code_sha256'].items():
        if rt.sha(Path(p))!=h:raise ValueError('SCISOR source changed')
    for asset in inventory['assets'].values():
        if rt.sha(Path(asset['path']))!=asset['sha256']:raise ValueError('SCISOR weights changed')
    start=time.time();torch.set_num_threads(2)
    result=dict(job_id=os.environ['SLURM_JOB_ID'],CPU_only=True,model_loaded=False,sequence_forward_run=False,
        GPU_calls=0,new_structure_predictions=0,new_training=False,shared_environment_changed=False,paper_changes=False)
    try:
        source=Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
        sys.path.insert(0,str(source))
        cls=importlib.import_module('SCISOR.shortening_scud').ShorteningSCUD
        checkpoint=inventory['assets']['checkpoint']['path']
        # Known user-owned checkpoint, SHA identical to the original evaluation protocol.
        # Opcode inventory has already documented its tokenizer pickle globals.
        model=cls.load_from_checkpoint(checkpoint,map_location='cpu').eval()
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=payload['state_dict'];parameters=dict(model.named_parameters())
        missing=[key for key in parameters if key not in state]
        changed=[key for key,value in parameters.items() if key in state and not torch.equal(value.detach().cpu(),state[key])]
        nonfinite=[key for key,value in parameters.items() if not torch.isfinite(value).all()]
        if missing or changed or nonfinite:raise ValueError('trained SCISOR parameters differ: '+str(dict(missing=missing,changed=changed,nonfinite=nonfinite)))
        p0=torch.load(inventory['assets']['p0']['path'],map_location='cpu',weights_only=True)
        if p0.ndim!=1 or not torch.isfinite(p0).all() or (p0<0).any():raise ValueError('invalid SCISOR p0')
        result['checkpoint_parameters_verified']=True
        patch_tokenizer(model.tokenizer)
        sequence='ACDEFGHIKLMNPQRSTVWY'
        ids=model.tokenizer(sequence).input_ids
        if len(ids)!=22:raise ValueError('SCISOR tokenizer must produce BOS+20 residues+EOS')
        tokens=model.tokenizer.convert_ids_to_tokens(ids[1:-1])
        if ''.join(tokens)!=sequence:raise ValueError('SCISOR tokenizer roundtrip mismatch')
        result.update(status='SCISOR_OFFLINE_CPU_MODEL_AND_TOKENIZER_PASS',model_loaded=True,
            parameter_tensors=len(parameters),parameters=sum(x.numel() for x in parameters.values()),
            parameters_equal_checkpoint=True,tokenizer_roundtrip_passed=True,
            checkpoint_sha256=inventory['assets']['checkpoint']['sha256'],
            model_runtime='server-local SCISOR source and existing venv-v7; no model sequence forward',
            p0_size=int(p0.numel()),public_upstream_revision_verified=False,
            tokenizer_compatibility='original V3-25 missing-special-token attribute patch',
            native_generation_still_requires_Slurm_GPU=True)
    except Exception as error:
        result.update(status='SCISOR_CPU_PREFLIGHT_FAILED_NO_GPU_RUN',error_type=type(error).__name__,
            error=str(error),traceback=traceback.format_exc())
    result['seconds']=time.time()-start
    write(ROOT/'reports/delepref_v8_scisor_preflight_v2.json',result)
    print(json.dumps(result),flush=True)
    if not result['model_loaded']:raise SystemExit(1)


if __name__=='__main__':main()
