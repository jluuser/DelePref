"""Frozen original Scorer evaluation on fixed measured variants; no design generation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import delepref_v8_runtime as rt
from delepref_v8_baseline import ROOT,write
from delepref_v8_external_inputs import prepare_mappings,mapping_predictions

BASE='external/delepref_v8_scorer'
DATA='external/rag_v6/preparation/primary_rows.jsonl'
PROTOCOL='external/rag_v6/preparation/protocol.json'
ENTRY_CONTRACT='external/rag_v6/preparation/evaluation_entry_contract.json'
FREEZE='configs/delepref_v8_external_scorer_freeze.json'


def prepare(root):
    rows=[json.loads(x) for x in (root/DATA).read_text().splitlines() if x.strip()]
    protocol=json.loads((root/PROTOCOL).read_text())
    if rt.sha(root/DATA)!=protocol['primary_rows_sha256']:raise ValueError('external cohort changed')
    wt,edits,links=prepare_mappings(rows)
    if len(wt)!=5 or len(rows)!=1317:raise ValueError('unexpected fixed external cohort')
    out=root/BASE;out.mkdir(exist_ok=False)
    write(out/'inputs.json',dict(wildtypes=wt,edits=edits,links=links))
    sources=['analysis/delepref_v8_external_inputs.py','analysis/delepref_v8_external_scorer.py',
        'analysis/tests/test_delepref_v8_external_inputs.py','analysis/tests/test_delepref_v8_external_scorer_interface.py','slurm/delepref_v8_external_scorer.sbatch',
        'slurm/delepref_v8_external_metrics.sbatch']
    inputs=[DATA,PROTOCOL,ENTRY_CONTRACT,'configs/delepref_v8_baseline_freeze.json',BASE+'/inputs.json']
    freeze=dict(status='ORIGINAL_SCORER_EXTERNAL_PROTOCOL_FROZEN',checkpoint_sha256=rt.CHECKPOINT_SHA,
        sources={p:rt.sha(root/p) for p in sources},inputs={p:rt.sha(root/p) for p in inputs},
        WT=len(wt),physical_observations=len(rows),unique_edit_mappings=len(edits),mapping_predictions=len(links),
        measured_data_used_for_fitting=False,score_direction='original_preference_score_higher_is_preferred',
        ambiguous_maps='all_exact_maps_equal_mean_via_frozen_evaluator',
        original_budget_profile='single_deletion_actual_fraction_outside_10_20_percent_training',
        RAG_rescoring=False,new_training=False,new_structure_predictions=0,paper_changes=False)
    write(root/FREEZE,freeze)
    return dict(freeze_sha256=rt.sha(root/FREEZE),WT=len(wt),unique_edit_mappings=len(edits),mapping_predictions=len(links))


def verified(root,checksum):
    if rt.sha(root/FREEZE)!=checksum:raise ValueError('external freeze changed')
    f=json.loads((root/FREEZE).read_text())
    for p,h in {**f['sources'],**f['inputs']}.items():
        if rt.sha(root/p)!=h:raise ValueError('external input/source changed: '+p)
    original=json.loads((root/'configs/delepref_v8_baseline_freeze.json').read_text())
    for p,h in {**original['inputs'],**original['code']}.items():
        if rt.sha(root/p)!=h:raise ValueError('original source/input changed: '+p)
    return f


def execute(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU allocation required')
    freeze=verified(root,checksum);out=root/BASE
    write(out/'scorer_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=checksum))
    inputs=json.loads((out/'inputs.json').read_text());start=time.time();torch.set_num_threads(4)
    runtime,api=rt.load(root)
    from delepref.preference_core import DeletionCandidate
    print(json.dumps(dict(event='ORIGINAL_EXTERNAL_SCORER_LOADED',step=4000)),flush=True)
    with torch.inference_mode():
        encoded=runtime.model.encode_unique_wildtypes(sorted(inputs['wildtypes'].items()),microbatch_size=5)
        cache=dict(embeddings={k.protein_id:v.detach().cpu() for k,v in encoded.embeddings.items()},
            sequences={k.protein_id:v for k,v in encoded.sequences.items()},checkpoint_sha256=rt.CHECKPOINT_SHA,
            real_encoder=True,role='fixed_measured_deletion_evaluation_only')
        torch.save(cache,out/'original_external_wt_cache.pt')
        scores={}
        for start_index in range(0,len(inputs['edits']),128):
            chunk=inputs['edits'][start_index:start_index+128];candidates=[]
            for item in chunk:
                pos=item['deleted_position'];beta=item['budget']
                candidates.append(DeletionCandidate(protein_id=item['target'],perturbation_id=item['edit_key'],
                    plan_id=item['edit_key'],wildtype_sequence=item['wt_sequence'],deleted_intervals=((pos,pos),),
                    deleted_length=1,deletion_fraction=beta,target_delete_fraction=beta,
                    budget_bin='retrospective_single_deletion',generation_policy='observed_measured_variant'))
            values=runtime.model.score_plans_from_embeddings(candidates,encoded)['preference_score'].detach().float().cpu()
            if not torch.isfinite(values).all():raise ValueError('nonfinite external prediction')
            scores.update({item['edit_key']:float(value) for item,value in zip(chunk,values)})
    predictions=mapping_predictions(inputs['links'],scores)
    path=out/'mapping_scores.jsonl'
    with path.open('x') as f:
        for row in predictions:f.write(json.dumps(row,sort_keys=True)+'\n')
    report=dict(status='ORIGINAL_SCORER_EXTERNAL_PREDICTIONS_COMPLETE',job_id=os.environ['SLURM_JOB_ID'],
        seconds=time.time()-start,checkpoint_sha256=rt.CHECKPOINT_SHA,freeze_sha256=checksum,
        WT_encodings=encoded.unique_count,encoder_batches=runtime.model.encoder.forward_batches,
        mutant_encodings=0,unique_edit_mappings=len(scores),mapping_predictions=len(predictions),
        scores_sha256=rt.sha(path),cache_sha256=rt.sha(out/'original_external_wt_cache.pt'),
        structural_quality_measured=False,model_fits=0,variant_recommendations_emitted=False,
        score_direction_changed_from_results=False,paper_changes=False)
    write(out/'scorer_completion.json',report);print(json.dumps(report),flush=True)


def metrics(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    verified(root,checksum);out=root/BASE
    complete=json.loads((out/'scorer_completion.json').read_text())
    if rt.sha(out/'mapping_scores.jsonl')!=complete['scores_sha256']:raise ValueError('external score file changed')
    command=[str(Path(os.sys.executable)),str(root/'analysis/rag_v6_external_evaluate.py'),
        '--dataset',str(root/DATA),'--protocol',str(root/PROTOCOL),'--entry-contract',str(root/ENTRY_CONTRACT),
        '--scores',str(out/'mapping_scores.jsonl'),'--level','mapping','--method','original_DelePref_step4000_Scorer',
        '--output',str(out/'aggregate_metrics.json')]
    subprocess.run(command,check=True)
    report=dict(status='ORIGINAL_SCORER_EXTERNAL_AGGREGATION_COMPLETE',job_id=os.environ['SLURM_JOB_ID'],
        metrics_sha256=rt.sha(out/'aggregate_metrics.json'),checkpoint_sha256=rt.CHECKPOINT_SHA,
        phenotype_ranked_variant_outputs=False,WT=5,physical_observations=1317,
        interpretation='Retrospective single-deletion Scorer transfer; not large multi-span or RAG generator validation.')
    write(out/'metrics_completion.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true')
    p.add_argument('--metrics',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(a.root)),flush=True)
    elif a.metrics:metrics(a.root,a.freeze_sha256)
    else:execute(a.root,a.freeze_sha256)
