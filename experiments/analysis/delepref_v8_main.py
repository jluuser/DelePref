"""100-WT frozen original DelePref/RAG: one GPU encoding, CPU planning/scoring."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import delepref_v8_runtime as rt
import delepref_unlimited_planning as planning
import delepref_unlimited_scorer as scoring
import delepref_v8_controls as controls
from delepref_v8_baseline import ROOT,write
from delepref_v8_rag_unlimited import cached_chunk,make_rows
import delepref_v8_engineering_acceptance as acceptance

BASE='reports/delepref_v8_main'
FREEZE='configs/delepref_v8_main_freeze.json'
PREP='reports/delepref_v8_evaluation_preparation'


def prepare(root):
    accepted=acceptance.accept(root,rt.CHECKPOINT_SHA)
    out=root/BASE;out.mkdir(exist_ok=False)
    write(out/'engineering_acceptance.json',accepted)
    prepared=json.loads((root/PREP/'completion.json').read_text())
    for name,digest in prepared['outputs'].items():
        if rt.sha(root/PREP/name)!=digest:raise ValueError('prepared evaluation input changed: '+name)
    sources=['analysis/delepref_v8_main.py','analysis/delepref_v8_engineering_acceptance.py',
        'analysis/delepref_v8_runtime.py','analysis/delepref_v8_rag_unlimited.py',
        'analysis/delepref_v8_baseline.py','analysis/delepref_unlimited_planning.py',
        'analysis/delepref_unlimited_scorer.py','analysis/delepref_v8_controls.py',
        'analysis/tests/test_delepref_v8_main.py','slurm/delepref_v8_main_encode.sbatch','slurm/delepref_v8_main_plans.sbatch']
    inputs=['configs/delepref_v8_baseline_freeze.json','configs/delepref_v8_rag_unlimited_freeze.json',
        'configs/delepref_v8_evaluation_preparation_freeze.json','configs/delepref_v8_main_retrieval_freeze.json',
        'retrieval/delepref_v8_main/profiles.json','reports/delepref_v8_native_panel/main_panel.json',
        BASE+'/engineering_acceptance.json',PREP+'/completion.json']+[PREP+'/'+name for name in prepared['outputs']]
    write(root/FREEZE,dict(status='V8_MAIN_MODEL_AND_GENERATION_FROZEN',WT=100,K=8,qmax=None,
        primary_budget=.2,primary_seed=13,stability_seed=29,secondary_budget=.1,
        original_checkpoint_sha256=rt.CHECKPOINT_SHA,source_hashes={p:rt.sha(root/p) for p in sources},
        inputs={p:rt.sha(root/p) for p in inputs},trained_heads_unchanged=True,
        original_encoder_WT_only=True,CPU_heads_after_cache=True,methods=['delepref_unlimited','rag_unlimited','msa_conservation','terminal'],
        new_training=False,structural_prediction_allowed=False,paper_changes=False))
    return dict(freeze_sha256=rt.sha(root/FREEZE),engineering=accepted)


def verify(root,checksum):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm allocation required')
    if rt.sha(root/FREEZE)!=checksum:raise ValueError('main freeze mismatch')
    frozen=json.loads((root/FREEZE).read_text())
    for p,h in {**frozen['source_hashes'],**frozen['inputs']}.items():
        if rt.sha(root/p)!=h:raise ValueError('main source/input changed: '+p)
    baseline=json.loads((root/'configs/delepref_v8_baseline_freeze.json').read_text())
    for p,h in {**baseline['inputs'],**baseline['code']}.items():
        if rt.sha(root/p)!=h:raise ValueError('original dependency changed: '+p)
    accepted=json.loads((root/BASE/'engineering_acceptance.json').read_text())
    for p,h in accepted['artifacts'].items():
        if rt.sha(Path(p))!=h:raise ValueError('accepted engineering artifact changed: '+p)
    return frozen


def encode(root,checksum):
    verify(root,checksum)
    if not torch.cuda.is_available():raise RuntimeError('real CUDA original ESMC required')
    out=root/BASE;write(out/'encode_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=checksum))
    start=time.time();torch.set_num_threads(4);runtime,api=rt.load(root)
    print(json.dumps(dict(event='MAIN_ORIGINAL_MODEL_LOADED',WT=100,step=4000)),flush=True)
    records=json.loads((root/PREP/'main_encoder_inputs.json').read_text())
    if len(records)!=100 or len({r['protein_id'] for r in records})!=100:raise ValueError('expected100 unique WTs')
    with torch.inference_mode():chunk=api.encode_chunk(runtime,records)
    if chunk.encoded.unique_count!=100:raise RuntimeError('unexpected encoding count')
    cache=dict(embeddings={key.protein_id:value.detach().cpu() for key,value in chunk.encoded.embeddings.items()},
        sequences={key.protein_id:value for key,value in chunk.encoded.sequences.items()},
        checkpoint_sha256=rt.CHECKPOINT_SHA,config_sha256=runtime.checkpoint['config_sha256'],
        encoder_profile=dict(dtype='bfloat16',snapshot=runtime.config['model']['esmc_snapshot'],max_length=1022,
            actual_hardware=torch.cuda.get_device_name(0)),real_encoder=True,unique_WT=100)
    torch.save(cache,out/'wt_cache.pt')
    report=dict(status='V8_MAIN_REAL_WT_ENCODING_COMPLETE',job_id=os.environ['SLURM_JOB_ID'],WT=100,
        encoder_batches=runtime.model.encoder.forward_batches,seconds=time.time()-start,
        checkpoint_sha256=rt.CHECKPOINT_SHA,freeze_sha256=checksum,cache_sha256=rt.sha(out/'wt_cache.pt'),
        mutant_encodings=0,new_training=False,new_structure_predictions=0,paper_changes=False)
    write(out/'encoding_completion.json',report);print(json.dumps(report),flush=True)


def score_condition(runtime,api,chunk,record,profile,seed,control_rows=None,ablation=False):
    pid=record['protein_id'];sequence=record['wildtype_sequence'];budget=float(record['budget'])
    template=dict(record,sequence_length=len(sequence),sequence_sha256=api.sequence_sha256(sequence),
        generation_seed=seed,condition_seed=controls.condition_seed(seed,pid,sequence,budget),
        student_version=runtime.checkpoint['student_version'],source_checkpoint_sha256=runtime.source_checkpoint_sha256,
        selected_checkpoint_step=4000)
    embedding=chunk.encoded.embeddings[chunk.encoded.key_for(pid,sequence)]
    logits=runtime.model.propose_from_embeddings([embedding],[budget])[0].residue_logits.detach().cpu().double().numpy()
    if not np.isfinite(logits).all() or np.std(logits)==0:raise RuntimeError('invalid trained Policy logits')
    cfg=runtime.config['decoder']
    items,cost=planning.policy_proposals(logits,budget,template['condition_seed'],
        perturbation_scale=float(cfg.get('perturbation_scale',1.)),max_attempts=8*int(cfg.get('max_attempts_multiplier',16)))
    if len(items)!=8:raise RuntimeError('incomplete original K: '+pid)
    original=scoring.score_candidates(runtime,api,chunk,make_rows(api,template,items,len(sequence),logits,cost))
    native=[dict(item,row=row) for item,row in zip(items,original)]
    retrieved,attempts=planning.retrieved_proposals(logits,budget,profile,template['condition_seed'])
    mixed,fallback=planning.mix_candidates(native,retrieved)
    rag=([dict(x) for x in original] if fallback else scoring.score_candidates(runtime,api,chunk,
        make_rows(api,template,mixed,len(sequence),logits,cost)))
    pools={'delepref_unlimited':original,'rag_unlimited':rag}
    none,flag=planning.mix_candidates(native,[])
    if not flag or [x['row'] for x in none]!=original:raise RuntimeError('counterfactual fallback mismatch')
    if control_rows is not None:
        for method in ('msa_conservation','terminal'):
            prepared=control_rows[method]
            if not prepared:
                if method!='msa_conservation':raise RuntimeError('missing terminal controls')
                pools[method]=[dict(x) for x in original];continue
            control_items=[dict(spans=tuple(map(tuple,x['deleted_intervals'])),kind=method) for x in prepared]
            rows=make_rows(api,template,control_items,len(sequence),logits)
            if [x['deletion_mask_sha256'] for x in rows]!=[x['deletion_mask_sha256'] for x in prepared]:
                raise ValueError('prepared control masks differ')
            for row,p in zip(rows,prepared):
                row.update(generation_attempts=p['generation_cost']['attempts'],duplicate_decodes=p['generation_cost']['duplicates'],
                    decoder='fixed_terminal_splits' if method=='terminal' else planning.CONTRACT['decoder'])
            pools[method]=scoring.score_candidates(runtime,api,chunk,rows)
    if ablation:
        direct=[dict(spans=x['spans'],source_target=x['target'],seed_spans=x['spans'],kind='retrieval_direct')
            for x in planning.seed_sets(profile,len(sequence),budget) if x['complete_natural_plan']][:4]
        direct_mix,direct_fallback=planning.mix_candidates(native,direct)
        pools['rag_direct_only']=([dict(x) for x in original] if direct_fallback else
            scoring.score_candidates(runtime,api,chunk,make_rows(api,template,direct_mix,len(sequence),logits,cost)))
    selected=[]
    for method,pool in pools.items():
        if len(pool)!=8 or len({x['deletion_mask_sha256'] for x in pool})!=8:raise RuntimeError('invalid unique K: '+method)
        for row in pool:row['method']=method
        best=dict(min(pool,key=lambda x:x['scorer_rank']))
        mask=planning.validate_plan(len(sequence),budget,json.loads(best['deleted_intervals']))
        edited=''.join(aa for aa,deleted in zip(sequence,mask) if not deleted)
        best.update(edited_sequence=edited,edited_sequence_sha256=controls.sequence_sha(edited))
        selected.append(best)
    diagnostic=dict(protein_id=pid,budget=budget,seed=seed,fallback=fallback,
        counterfactual_fallback_exact=True,original_generation=cost,retrieval_attempts=attempts,
        retrieved_unique=len(retrieved),policy_logit_std=float(np.std(logits)),
        original_selected_mask=selected[0]['deletion_mask_sha256'],rag_selected_mask=selected[1]['deletion_mask_sha256'])
    return [row for pool in pools.values() for row in pool],selected,diagnostic


def plans(root,checksum):
    verify(root,checksum)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU heads only')
    out=root/BASE;complete=json.loads((out/'encoding_completion.json').read_text())
    if complete['freeze_sha256']!=checksum or complete['checkpoint_sha256']!=rt.CHECKPOINT_SHA or rt.sha(out/'wt_cache.pt')!=complete['cache_sha256']:
        raise ValueError('main encoding identity mismatch')
    write(out/'planning_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=checksum))
    start=time.time();torch.set_num_threads(4);runtime,api=rt.load(root,device='cpu',cached_heads_only=True)
    cache=torch.load(out/'wt_cache.pt',map_location='cpu',weights_only=True)
    if not cache['real_encoder'] or cache['unique_WT']!=100 or cache['checkpoint_sha256']!=rt.CHECKPOINT_SHA:
        raise ValueError('not100 original real WT encodings')
    records=json.loads((root/PREP/'main_encoder_inputs.json').read_text());chunk=cached_chunk(api,cache,records)
    profiles=json.loads((root/'retrieval/delepref_v8_main/profiles.json').read_text())
    subsets=json.loads((root/PREP/'subsets.json').read_text())
    grouped=defaultdict(list)
    for row in json.loads((root/PREP/'control_candidates.json').read_text()):
        grouped[(row['protein_id'],float(row['budget']),int(row['generation_seed']),row['method'])].append(row)
    all_scores=[];selections=[];diagnostics=[]
    with torch.inference_mode():
        for record in records:
            pid=record['protein_id'];conditions=[(.2,13),(.2,29)]
            if pid in subsets['secondary_10pct']:conditions.append((.1,13))
            for budget,seed in conditions:
                control={method:sorted(grouped[(pid,budget,seed,method)],key=lambda x:x['raw_candidate_rank'])
                    for method in ('msa_conservation','terminal')} if budget==.2 else None
                scores,selected,diag=score_condition(runtime,api,chunk,dict(record,budget=budget),profiles[pid],seed,
                    control_rows=control,ablation=budget==.2 and seed==13 and pid in subsets['completion_ablation'])
                for row in selected:
                    row['evaluation_role']=('CPU_stability_only' if seed==29 else
                        ('completion_ablation' if row['method']=='rag_direct_only' else ('primary_20pct' if budget==.2 else 'secondary_10pct')))
                all_scores.extend(scores);selections.extend(selected);diagnostics.append(diag)
                print(json.dumps(dict(event='MAIN_CPU_CONDITION_COMPLETE',protein_id=pid,budget=budget,seed=seed,
                    methods=len(selected),fallback=diag['fallback'])),flush=True)
    write(out/'candidate_scores.json',all_scores);write(out/'all_selections.json',selections)
    write(out/'generation_diagnostics.json',diagnostics)
    final=[x for x in selections if x['evaluation_role']!='CPU_stability_only']
    write(out/'primary_secondary_ablation_selections.json',final)
    second=[dict(x,evaluation_role='second_predictor') for x in final if x['evaluation_role']=='primary_20pct'
        and x['protein_id'] in subsets['second_predictor'] and x['method'] in ('delepref_unlimited','rag_unlimited','terminal')]
    second.extend(dict(protein_id=r['protein_id'],method='WT',evaluation_role='second_predictor',edited_sequence=r['wildtype_sequence'],
        edited_sequence_sha256=controls.sequence_sha(r['wildtype_sequence'])) for r in records if r['protein_id'] in subsets['second_predictor'])
    if len(final)!=500 or len(second)!=32:raise RuntimeError('evaluation selection roster incomplete')
    write(out/'second_predictor_selections.json',second)
    report=dict(status='V8_MAIN_TRAINED_HEAD_SELECTIONS_COMPLETE_NOT_STRUCTURE_EVALUATED',job_id=os.environ['SLURM_JOB_ID'],
        seconds=time.time()-start,WT=100,conditions=len(diagnostics),scored_candidates=len(all_scores),
        selected_outputs=len(selections),primary_secondary_ablation_outputs=len(final),second_predictor_outputs=len(second),
        primary_RAG_changed_WT=sum(d['original_selected_mask']!=d['rag_selected_mask'] for d in diagnostics if d['budget']==.2 and d['seed']==13),
        freeze_sha256=checksum,checkpoint_sha256=rt.CHECKPOINT_SHA,cache_sha256=complete['cache_sha256'],
        outputs={p.name:rt.sha(p) for p in out.glob('*.json') if p.name not in ('encode_claim.json','planning_claim.json')},
        all_no_evidence_fallback_tests_exact=True,extra_PLM_forwards=0,mutant_encodings=0,new_training=False,
        new_structure_predictions=0,structural_prediction_allowed=False,paper_changes=False)
    write(out/'planning_completion.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true')
    p.add_argument('--stage',choices=['encode','plans']);p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(a.root)),flush=True)
    elif a.stage=='encode':encode(a.root,a.freeze_sha256)
    elif a.stage=='plans':plans(a.root,a.freeze_sha256)
    else:p.error('prepare or stage required')
