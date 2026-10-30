"""Original DelePref/RAG on the frozen Binder20 and Enzyme20; no retraining."""
import argparse
import json
import os
from pathlib import Path
import time
import torch

import delepref_v8_runtime as rt
from delepref_v8_main import score_condition
from delepref_v8_rag_unlimited import cached_chunk
from recomb_general100_common_20261017 import ROOT,sha,write,require_gpu
from recomb_general100_app_prepare_20261017 import BASE as APP
from recomb_general100_app_retrieval_20261017 import BASE as RETRIEVAL


def encode(root):
    gpu=require_gpu();out=root/APP
    if (out/'encoding_completion.json').exists():raise FileExistsError('application encoding already completed')
    panel=json.loads((out/'panel.json').read_text());started=time.monotonic()
    frozen=json.loads((root/'configs/recomb_general100_app_inputs_freeze_20261017.json').read_text())
    if sha(out/'panel.json')!=frozen['panel_sha256']:raise ValueError('application roster changed')
    write(root/'configs/recomb_general100_app_model_freeze_20261017.json',dict(
        status='APPLICATION_ORIGINAL_MODEL_FROZEN',files={p:sha(root/p) for p in [
          'analysis/recomb_general100_app_model_20261017.py','analysis/delepref_v8_main.py',
          'analysis/delepref_v8_runtime.py','analysis/delepref_unlimited_planning.py','analysis/delepref_unlimited_scorer.py',
          APP+'/panel.json','configs/delepref_original_restored.yaml']},
        checkpoint_sha256=rt.CHECKPOINT_SHA,K=8,qmax=None,budget=.2,seed=13,new_training=False,
        allowed_GPU_types=['A40','A100'],structure_prediction_limit=None,structural_prediction_allowed=False))
    torch.set_num_threads(4);runtime,api=rt.load(root)
    print(json.dumps(dict(event='APPLICATION_ORIGINAL_MODEL_LOADED',WT=40,actual_GPU=gpu)),flush=True)
    records=[dict(protein_id=p['protein_id'],wildtype_sequence=p['sequence'],budget=.2) for p in panel]
    print('APPLICATION_REAL_WT_ENCODING_START',flush=True)
    with torch.inference_mode():chunk=api.encode_chunk(runtime,records)
    if chunk.encoded.unique_count!=40:raise ValueError('expected40 real original WT encodings')
    cache=dict(embeddings={key.protein_id:value.detach().cpu() for key,value in chunk.encoded.embeddings.items()},
        sequences={key.protein_id:value for key,value in chunk.encoded.sequences.items()},
        checkpoint_sha256=rt.CHECKPOINT_SHA,real_encoder=True,unique_WT=40,
        encoder_profile=dict(dtype='bfloat16',snapshot=runtime.config['model']['esmc_snapshot'],max_length=1022,actual_hardware=gpu))
    torch.save(cache,out/'wt_cache.pt')
    write(out/'encoding_completion.json',dict(status='APPLICATION_REAL_WT_ENCODING_COMPLETE',WT=40,
         seconds=time.monotonic()-started,cache_sha256=sha(out/'wt_cache.pt'),actual_GPU=gpu,
         job_id=os.environ['SLURM_JOB_ID'],mutant_encodings=0,new_training=False,new_structure_predictions=0))


def plans(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU Slurm required')
    out=root/APP
    if (out/'planning_completion.json').exists():raise FileExistsError('application plans already completed')
    encoded=json.loads((out/'encoding_completion.json').read_text())
    if sha(out/'wt_cache.pt')!=encoded['cache_sha256']:raise ValueError('application WT cache changed')
    frozen=json.loads((root/'configs/recomb_general100_app_model_freeze_20261017.json').read_text())
    for p,h in frozen['files'].items():
        if sha(root/p)!=h:raise ValueError('application model input/source changed')
    retrieval=json.loads((root/RETRIEVAL/'completion.json').read_text())
    if sha(root/RETRIEVAL/'profiles.json')!=retrieval['profiles_sha256']:raise ValueError('application profiles changed')
    profiles=json.loads((root/RETRIEVAL/'profiles.json').read_text())
    panel=json.loads((out/'panel.json').read_text());records=[dict(protein_id=p['protein_id'],wildtype_sequence=p['sequence'],budget=.2) for p in panel]
    torch.set_num_threads(4);started=time.monotonic();runtime,api=rt.load(root,device='cpu',cached_heads_only=True)
    cache=torch.load(out/'wt_cache.pt',map_location='cpu',weights_only=True);chunk=cached_chunk(api,cache,records)
    scores=[];selections=[];diagnostics=[]
    with torch.inference_mode():
        for r in records:
            pool,selected,diagnostic=score_condition(runtime,api,chunk,r,profiles[r['protein_id']],13)
            scores.extend(pool);selections.extend(selected);diagnostics.append(diagnostic)
            print(json.dumps(dict(event='APPLICATION_ORIGINAL_RAG_SELECTION_COMPLETE',protein_id=r['protein_id'])),flush=True)
    if len(selections)!=80 or len(scores)!=640:raise ValueError('application output roster incomplete')
    write(out/'all_selections.json',selections);write(out/'candidate_scores.json',scores);write(out/'generation_diagnostics.json',diagnostics)
    write(out/'planning_completion.json',dict(status='APPLICATION_ORIGINAL_POLICY_SCORER_AND_RAG_COMPLETE',WT=40,
        selected_outputs=80,candidates=640,seconds=time.monotonic()-started,original_checkpoint_sha256=rt.CHECKPOINT_SHA,
        outputs={n:sha(out/n) for n in ('all_selections.json','candidate_scores.json','generation_diagnostics.json')},
        real_WT_cache_sha256=encoded['cache_sha256'],retrieval_profile_sha256=retrieval['profiles_sha256'],
        new_structure_predictions=0,new_training=False,job_id=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--stage',choices=['encode','plans'],required=True)
    a=p.parse_args();(encode if a.stage=='encode' else plans)(a.root)
