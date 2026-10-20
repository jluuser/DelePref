"""Verify official SCISOR-L on CPU with the already validated inference adapter."""
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from recomb_confirm_v2_diagnostics import read,sha,write
from delepref_v8_scisor_compat import patch_tokenizer
from delepref_v8_scisor_inference_import_v2 import load_native_class

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
EXPECTED='6925f185f82b9916698d03cc7b32b42166e5272d460f599eb9c64e8902442fe7'


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    out=ROOT/'reports/recomb_confirm_v2_scisor_L'
    if (out/'preflight.json').exists():raise FileExistsError('Inspect existing L preflight before retrying')
    ckpt=ROOT/'models/scisor_u90_l/SCISOR_U90_L.ckpt'
    if ckpt.stat().st_size!=8070717202 or sha(ckpt)!=EXPECTED:raise ValueError('Official L checkpoint identity differs')
    f=read(ROOT/'configs/recomb_confirm_v2_scisor_L_preflight_freeze.json')
    for p,h in f['files'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen preflight source differs')
    inventory=read(ROOT/'reports/delepref_v8_scisor_inventory.json')
    for p,h in inventory['code_sha256'].items():
        if sha(p)!=h:raise ValueError('Read-only SCISOR source changed')
    torch.set_num_threads(4);start=time.monotonic()
    result=dict(job_id=os.environ['SLURM_JOB_ID'],CPU_only=True,new_GPU_calls=0,new_structure_calls=0,
        sequence_forward_run=False,new_training=False,checkpoint_sha256=EXPECTED)
    try:
        sys.path.insert(0,'/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
        cls=load_native_class();model=cls.load_from_checkpoint(str(ckpt),map_location='cpu').eval()
        patch_tokenizer(model.tokenizer)
        sequence='ACDEFGHIKLMNPQRSTVWY';tokens=model.tokenizer(sequence).input_ids
        if len(tokens)!=22 or ''.join(model.tokenizer.convert_ids_to_tokens(tokens[1:-1]))!=sequence:
            raise ValueError('Tokenizer identity/roundtrip differs')
        payload=torch.load(ckpt,map_location='cpu',weights_only=False);state=payload['state_dict']
        params=dict(model.named_parameters());missing=[];changed=[];nonfinite=[]
        for k,v in params.items():
            if k not in state:missing.append(k)
            elif not torch.equal(v.detach(),state[k]):changed.append(k)
            if not torch.isfinite(v).all():nonfinite.append(k)
        if missing or changed or nonfinite:raise ValueError(str(dict(missing=missing,changed=changed,nonfinite=nonfinite)))
        result.update(status='OFFICIAL_SCISOR_L_CPU_MODEL_TOKENIZER_AND_ALL_PARAMETERS_PASS',
            parameter_tensors=len(params),parameters=sum(p.numel() for p in params.values()),
            tokenizer_roundtrip=True,all_parameter_values_equal_checkpoint=True,model_loaded=True,
            shared_source_and_environment_unchanged=True)
    except Exception as e:
        result.update(status='SCISOR_L_CPU_PREFLIGHT_FAILED_NO_SEQUENCE_FORWARD',error=repr(e),traceback=traceback.format_exc())
    result['seconds']=time.monotonic()-start;write(out/'preflight.json',result);print(result,flush=True)
    if not result.get('model_loaded'):raise SystemExit(1)


if __name__=='__main__':main()
