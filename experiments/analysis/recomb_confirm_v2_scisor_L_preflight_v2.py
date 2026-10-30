"""CPU identity check after resolving the offline ESM configuration dependency."""
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_confirm_v2_scisor_L_loader import load

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')


def main():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    out=ROOT/'reports/recomb_confirm_v2_scisor_L';path=out/'preflight_v2.json'
    if path.exists():raise FileExistsError('Inspect preflight v2 before any retry')
    f=read(ROOT/'configs/recomb_confirm_v2_scisor_L_preflight_freeze_v2.json')
    for p,h in f['files'].items():
        if sha(ROOT/p)!=h:raise ValueError('Frozen L loader/source/config differs')
    inventory=read(ROOT/'reports/delepref_v8_scisor_inventory.json')
    for p,h in inventory['code_sha256'].items():
        if sha(p)!=h:raise ValueError('Read-only SCISOR source changed')
    torch.set_num_threads(4);start=time.monotonic()
    result=dict(job_id=os.environ['SLURM_JOB_ID'],CPU_only=True,new_GPU_calls=0,new_structure_calls=0,
        sequence_forward_run=False,new_training=False,old_failure_job=1118697)
    try:
        sys.path.insert(0,'/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit')
        model,payload,provenance=load(ROOT);result.update(provenance)
        seq='ACDEFGHIKLMNPQRSTVWY';ids=model.tokenizer(seq).input_ids
        if len(ids)!=22 or ''.join(model.tokenizer.convert_ids_to_tokens(ids[1:-1]))!=seq:raise ValueError('Tokenizer roundtrip differs')
        state=model.state_dict();original=payload['state_dict']
        if set(state)!=set(original):raise ValueError('Checkpoint state keys differ')
        for k,v in state.items():
            if not torch.equal(v,original[k]) or not torch.isfinite(v).all():raise ValueError('Loaded state differs: '+k)
        result.update(status='OFFICIAL_SCISOR_L_OFFLINE_ALL_STATE_TOKENIZER_PASS',
            model_loaded=True,checkpoint_state_tensors=len(state),parameter_tensors=len(list(model.parameters())),
            parameters=sum(p.numel() for p in model.parameters()),all_state_tensor_values_equal=True,tokenizer_roundtrip=True)
    except Exception as e:result.update(status='L_CPU_PREFLIGHT_V2_FAILED_NO_FORWARD',error=repr(e),traceback=traceback.format_exc())
    result['seconds']=time.monotonic()-start;write(path,result);print(result,flush=True)
    if not result.get('model_loaded'):raise SystemExit(1)


if __name__=='__main__':main()
