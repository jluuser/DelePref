"""One genuine frozen-ESMC encoding per qualified General100 WT; no plans/folds."""
import argparse
import os
from pathlib import Path
import time
import torch
import delepref_v8_runtime as rt
from recomb_confirm_v2_diagnostics import read,sha,write


def main(root):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('CUDA via Slurm required')
    hardware=torch.cuda.get_device_name()
    if not any(t in hardware for t in ('A40','A100')):raise RuntimeError('Unapproved GPU type')
    fz=root/'configs/recomb_confirm_v2_General100_encoder_freeze.json';f=read(fz)
    for name,h in f['files'].items():
        if sha(root/name)!=h:raise ValueError('Qualified encoder input/source changed: '+name)
    panel=read(root/'reports/recomb_confirm_v2_general_native/General100_panel.json')
    if len(panel)!=100 or not all(r['WT_encoding_allowed'] and not r['structure_prediction_allowed'] for r in panel):
        raise ValueError('Wrong model operation or unqualified WT roster')
    out=root/'reports/recomb_confirm_v2_General100_model'
    if (out/'encoding_completion.json').exists():raise FileExistsError('General100 already encoded')
    records=read(root/'reports/recomb_confirm_v2_general_native/encoder_inputs.json')
    torch.set_num_threads(4);start=time.monotonic();runtime,api=rt.load(root)
    print('CONFIRM_V2_ORIGINAL_MODEL_LOADED '+hardware,flush=True)
    print('CONFIRM_V2_GENERAL100_REAL_WT_ENCODING_START',flush=True)
    with torch.inference_mode():chunk=api.encode_chunk(runtime,records)
    if chunk.encoded.unique_count!=100:raise ValueError('Missing or duplicate real WT encoding')
    cache=dict(embeddings={k.protein_id:v.detach().cpu() for k,v in chunk.encoded.embeddings.items()},
        sequences={k.protein_id:v for k,v in chunk.encoded.sequences.items()},checkpoint_sha256=rt.CHECKPOINT_SHA,
        real_encoder=True,unique_WT=100,encoder_profile=dict(dtype='bfloat16',hidden_size=1152,
        snapshot=runtime.config['model']['esmc_snapshot'],actual_hardware=hardware,max_length=1022))
    out.mkdir(parents=True,exist_ok=True);torch.save(cache,out/'wt_cache.pt')
    report=dict(status='GENERAL100_REAL_ORIGINAL_WT_ENCODING_COMPLETE',WT=100,mutant_encodings=0,
        seconds=time.monotonic()-start,cache_sha256=sha(out/'wt_cache.pt'),actual_GPU=hardware,
        original_checkpoint_sha256=rt.CHECKPOINT_SHA,encoder_freeze_sha256=sha(fz),new_training=False,
        new_structure_predictions=0,job_id=os.environ['SLURM_JOB_ID'])
    write(out/'encoding_completion.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
