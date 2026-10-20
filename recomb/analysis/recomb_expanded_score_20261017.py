"""Original frozen Scorer/U and published single-deletion SCISOR on expanded fixed data."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

from recomb_expanded_external_prepare_20261017 import ROOT,BASE,sha,write,seqsha
FREEZE='configs/recomb_expanded_scores_20261017.json'


def read(path):return json.loads((ROOT/path).read_text())


def prepare():
    if (ROOT/FREEZE).exists():raise FileExistsError('expanded model scoring already frozen')
    cohort=read(BASE+'/cohort_verified_manifest.json')
    for n,h in cohort['files'].items():assert sha(ROOT/BASE/n)==h
    public=read(BASE+'/coverage_report.json');assert sha(ROOT/BASE/'public_score_matrix.jsonl')==public['matrix_sha256']
    old=read('configs/delepref_v8_external_baselines_freeze.json')
    paths=[BASE+'/'+n for n in ['inputs.json','rows.jsonl','protocol.json','cohort_verified_manifest.json',
           'training_exposure.json','homology_completion.json','public_score_matrix.jsonl','coverage_report.json']]
    paths+=['analysis/recomb_expanded_score_20261017.py','analysis/recomb_expanded_metrics_20261017.py',
            'analysis/delepref_unlimited_scorer.py','analysis/delepref_v8_runtime.py',
            'analysis/delepref_v8_external_baseline_core.py',
            'configs/delepref_v8_external_baselines_freeze.json',
            'external/delepref_v8_scorer/inputs.json','external/delepref_v8_scorer/mapping_scores.jsonl',
            'external/delepref_v8_scorer/original_external_wt_cache.pt',
            'external/delepref_v8_scorer/scorer_completion.json',
            'external/delepref_v8_baselines/scoring_completion.json','external/delepref_v8_baselines/original_U_mapping_scores.jsonl',
            BASE+'/scorer_interface_preflight.json']
    write(FREEZE,dict(status='EXPANDED_FIXED_VARIANT_SCORE_PROTOCOL_FROZEN',seed=20261017,
        files={p:sha(ROOT/p) for p in paths},original_baseline_freeze_sha256=sha(ROOT/'configs/delepref_v8_external_baselines_freeze.json'),
        old_baseline=old,models=['original_Scorer','original_U','global_similarity','retained_similarity','junction_plausibility',
                                'SCISOR_U90_S_M1_single_deletion_only','Policy_additive_reference','terminal_distance','deletion_length_only'],
        public_models=public['columns'],public_scores_already_direction_adjusted_by_official_merge=True,
        scorer='original4000-step heads; unchanged unlimited-span adapter; >3 spans extrapolation disclosed',
        scisor='Original published M=1 event probability only for D=1; multi-deletion rows explicitly missing, no invented official fitness',
        teacher='Originalbf16/BOS-EOS exclusion/threeviews/6decimals/.35-.45-.20; identical old map scores reused',
        new_training=False,new_structure_predictions=0,paper_changes=False))
    print(json.dumps(dict(freeze_sha256=sha(ROOT/FREEZE))))


def verify(checksum):
    assert sha(ROOT/FREEZE)==checksum
    freeze=read(FREEZE)
    for p,h in freeze['files'].items():assert sha(ROOT/p)==h,p
    from delepref_v8_external_baselines import verified
    verified(freeze['original_baseline_freeze_sha256'])
    return freeze


def old_map_scores():
    old=read('external/delepref_v8_scorer/inputs.json');lookup={}
    scorer={(r['row_id'],r['deleted_position']):r['score'] for r in
            map(json.loads,(ROOT/'external/delepref_v8_scorer/mapping_scores.jsonl').read_text().splitlines())}
    utility={(r['row_id'],r['deleted_position']):r for r in
            map(json.loads,(ROOT/'external/delepref_v8_baselines/original_U_mapping_scores.jsonl').read_text().splitlines())}
    edits={x['edit_key']:x for x in old['edits']}
    for link in old['links']:
        edit=edits[link['edit_key']];key=(edit['wt_sequence'],(edit['deleted_position'],));k=(link['row_id'],link['deleted_position'])
        u=utility[k]
        lookup[key]=dict(original_Scorer=scorer[k],original_U=u['score'],
            **{v:u[v] for v in ['global_similarity','retained_similarity','junction_plausibility']})
    return lookup


def preflight():
    import torch
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    import delepref_v8_runtime as rt
    from delepref_v8_rag_unlimited import cached_chunk
    from delepref_unlimited_scorer import preference_scores
    runtime,api=rt.load(ROOT,device='cpu',cached_heads_only=True);torch.set_num_threads(4)
    cache=torch.load(ROOT/'external/delepref_v8_scorer/original_external_wt_cache.pt',map_location='cpu',weights_only=True)
    records=[dict(protein_id=k,wildtype_sequence=s,budget=.1) for k,s in cache['sequences'].items()]
    chunk=cached_chunk(api,cache,records)
    from delepref.preference_core import DeletionCandidate
    edits=read('external/delepref_v8_scorer/inputs.json')['edits'][:16];candidates=[]
    for x in edits:
        p=x['deleted_position'];b=x['budget']
        candidates.append(DeletionCandidate(protein_id=x['target'],perturbation_id=x['edit_key'],plan_id=x['edit_key'],
            wildtype_sequence=x['wt_sequence'],deleted_intervals=((p,p),),deleted_length=1,
            deletion_fraction=b,target_delete_fraction=b,budget_bin='retrospective',generation_policy='observed_variant'))
    with torch.inference_mode():scores=preference_scores(runtime.model.scorer,candidates,chunk.encoded).detach().float().cpu().tolist()
    old=old_map_scores();diff=max(abs(v-old[(x['wt_sequence'],(x['deleted_position'],))]['original_Scorer']) for x,v in zip(edits,scores))
    assert diff<1e-4
    write(BASE+'/scorer_interface_preflight.json',dict(status='REAL_ORIGINAL_HEADS_AND_REAL_LEGACY_CACHE_INTERFACE_VERIFIED',
        job_id=os.environ['SLURM_JOB_ID'],legacy_mappings_checked=len(edits),maximum_CPU_GPU_score_difference=diff,
        synthetic_embeddings=False,new_encoder_forwards=0,new_target_effect_labels_read=False,
        source_sha256=sha(__file__)))
    print(json.dumps(read(BASE+'/scorer_interface_preflight.json')),flush=True)


def score(checksum):
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
    freeze=verify(checksum);torch.set_num_threads(4);torch.manual_seed(freeze['seed']);np.random.seed(freeze['seed'])
    torch.backends.cuda.matmul.allow_tf32=True
    inputs=read(BASE+'/inputs.json');WT=inputs['wildtypes'];edits=inputs['edits'];old=old_map_scores()
    scores={x['edit_key']:dict(old.get((x['wt_sequence'],tuple(x['deleted_positions'])),{})) for x in edits}
    reused=sum(bool(v) for v in scores.values());started=time.monotonic()
    import delepref_v8_runtime as rt
    import delepref_unlimited_scorer as scoring
    from delepref_v8_rag_unlimited import cached_chunk
    runtime,api=rt.load(ROOT)
    print('EXPANDED_ORIGINAL_MODEL_LOADED',flush=True)
    oldcache=torch.load(ROOT/'external/delepref_v8_scorer/original_external_wt_cache.pt',map_location='cpu',weights_only=True)
    assert oldcache['real_encoder'] and oldcache['checkpoint_sha256']==rt.CHECKPOINT_SHA
    old_by_seq={oldcache['sequences'][k]:v for k,v in oldcache['embeddings'].items()}
    embeddings={k:old_by_seq[s] for k,s in WT.items() if s in old_by_seq}
    remaining=sorted([(k,s) for k,s in WT.items() if k not in embeddings],key=lambda x:(len(x[1]),x[0]))
    with torch.inference_mode():
        for i in range(0,len(remaining),4):
            enc=runtime.model.encode_unique_wildtypes(remaining[i:i+4],microbatch_size=4)
            embeddings.update({k.protein_id:v.detach().cpu() for k,v in enc.embeddings.items()})
    cache=dict(embeddings=embeddings,sequences=WT,real_encoder=True,checkpoint_sha256=rt.CHECKPOINT_SHA,
               source='original profile; exact old Scorer WT cache reused by sequence; remaining WT encoded once')
    torch.save(cache,ROOT/BASE/'original_wt_cache.pt')
    del runtime,enc;gc.collect();torch.cuda.empty_cache()
    runtime,api=rt.load(ROOT,device='cpu',cached_heads_only=True)
    # Cache packaging requires a valid nominal condition; candidate-specific beta below remains exact D/L.
    records=[dict(protein_id=k,wildtype_sequence=v,budget=.1) for k,v in WT.items()];chunk=cached_chunk(api,cache,records)
    from delepref.preference_core import DeletionCandidate
    policy={}
    with torch.inference_mode():
        for i in range(0,len(edits),64):
            batch=edits[i:i+64];candidates=[]
            for x in batch:
                candidates.append(DeletionCandidate(protein_id=x['target'],perturbation_id=x['edit_key'],plan_id=x['edit_key'],
                    wildtype_sequence=x['wt_sequence'],deleted_intervals=tuple(map(tuple,x['deleted_intervals'])),
                    deleted_length=x['deleted_length'],deletion_fraction=x['budget'],target_delete_fraction=x['budget'],
                    budget_bin='retrospective_fixed_measured_deletion',generation_policy='observed_variant'))
            values=scoring.preference_scores(runtime.model.scorer,candidates,chunk.encoded).detach().float().cpu().tolist()
            for x,value in zip(batch,values):
                scores[x['edit_key']].setdefault('original_Scorer',value)
                key=(x['target'],x['deleted_length'])
                if key not in policy:
                    embedding=embeddings[x['target']]
                    policy[key]=runtime.model.propose_from_embeddings([embedding],[x['budget']])[0].residue_logits.detach().float().cpu().numpy()
                ids=x['deleted_positions'];L=len(x['wt_sequence'])
                scores[x['edit_key']].update(Policy_additive_reference=float(policy[key][ids].sum()),
                    terminal_distance=-float(np.mean([min(p,L-1-p) for p in ids]))/L,
                    deletion_length_only=-float(len(ids)))
    del runtime,chunk;gc.collect()
    from delepref_v8_scisor_compat import patch_tokenizer
    from delepref_v8_scisor_inference_import_v2 import load_native_class
    from delepref_v8_external_baseline_core import residue_deletion_probabilities,load_original_teacher,original_utility
    SOURCE=Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit');sys.path.insert(0,str(SOURCE))
    oldf=freeze['old_baseline'];model=load_native_class().load_from_checkpoint(oldf['SCISOR_checkpoint'],map_location='cpu').cuda().eval()
    patch_tokenizer(model.tokenizer);by_target={}
    print('EXPANDED_SCISOR_SINGLE_DELETION_FORWARD_START',flush=True)
    with torch.inference_mode():
        for k,s in sorted(WT.items()):
            ids=model.tokenizer([s],return_tensors='pt',padding=True)['input_ids'].cuda()
            assert ''.join(model.tokenizer.convert_ids_to_tokens(ids[0,1:-1].tolist()))==s
            logits=model.model_predict(ids,torch.zeros(1,device='cuda'),None,torch.ones(1,device='cuda'))
            by_target[k]=residue_deletion_probabilities(logits.detach().float().cpu().numpy(),len(s))
        for x in edits:
            if x['deleted_length']==1:scores[x['edit_key']]['SCISOR_U90_S_M1_single_deletion_only']=float(by_target[x['target']][x['deleted_positions'][0]])
    del model,logits,ids;gc.collect();torch.cuda.empty_cache()
    sys.path.insert(0,oldf['teacher_vendor']);teacher=load_original_teacher(ROOT)
    args=SimpleNamespace(teacher_model_name_or_path=oldf['teacher_snapshot'],teacher_backend=teacher.ESMC_OFFICIAL_SAFETENSORS_BACKEND,
        teacher_device='cuda',teacher_cache_dir=None,teacher_allow_download=False,teacher_model='frozen_esmc_600m_delepref',
        teacher_group_by_protein=True,teacher_batch_size=8,deletion_perturbations_csv=str(ROOT/BASE/'rows.jsonl'))
    backend=teacher.load_real_plm_teacher_backend(args);needed=defaultdict(list)
    for x in edits:
        if 'original_U' not in scores[x['edit_key']]:needed[x['mutant_sequence']].append(x)
    targets={x['target'] for group in needed.values() for x in group};wt_embeddings={}
    print('EXPANDED_ORIGINAL_TEACHER_FORWARD_START',flush=True)
    with torch.inference_mode():
        for k in sorted(targets):wt_embeddings[k]=teacher.residue_embeddings_for_sequences(backend,[WT[k]],batch_size=1)[0].detach().cpu()
        seqs=sorted(needed,key=lambda s:(len(s),seqsha(s)))
        for i in range(0,len(seqs),8):
            batch=seqs[i:i+8];encoded=teacher.residue_embeddings_for_sequences(backend,batch,batch_size=8)
            for s,e in zip(batch,encoded):
                for x in needed[s]:
                    row=dict(protein_id=x['target'],perturbation_id=x['edit_key'],plan_id=x['edit_key'],wildtype_sequence=x['wt_sequence'],
                        wildtype_length=len(x['wt_sequence']),mutant_sequence=s,coordinate_system='0_based_closed_inclusive',
                        deleted_intervals=json.dumps(x['deleted_intervals']),deleted_length=x['deleted_length'],
                        num_intervals=len(x['deleted_intervals']),deletion_fraction=x['budget'])
                    impact=teacher.real_plm_teacher_impact_from_embeddings(row,args,backend,wt_embeddings[x['target']].cuda(),e.cuda())
                    scores[x['edit_key']].update(original_U=original_utility(impact),
                        **{v:float(impact[v]) for v in ['global_similarity','retained_similarity','junction_plausibility']})
            print('EXPANDED_TEACHER_VARIANTS '+str(min(i+8,len(seqs)))+'/'+str(len(seqs)),flush=True)
    assert len(scores)==len(edits) and all('original_U' in x and 'original_Scorer' in x for x in scores.values())
    write(BASE+'/model_mapping_scores.json',scores)
    write(BASE+'/scoring_completion.json',dict(status='EXPANDED_ORIGINAL_MODELS_FIXED_VARIANT_SCORES_COMPLETE',
        job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=checksum,seconds=time.monotonic()-started,WT=len(WT),
        new_Scorer_WT_encodings=len(remaining),reused_Scorer_WT_encodings=len(WT)-len(remaining),
        reused_old_mapping_scores=reused,unique_edit_mappings=len(edits),SCISOR_WT_forwards=len(WT),
        new_teacher_WT_encodings=len(targets),new_teacher_variant_encodings=len(seqs),
        GPU_name=torch.cuda.get_device_name(),scores_sha256=sha(ROOT/BASE/'model_mapping_scores.json'),
        new_training=False,new_structure_predictions=0,no_phenotype_edit_recommendations=True))
    print(json.dumps(read(BASE+'/scoring_completion.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.prepare:prepare()
    elif a.preflight:preflight()
    else:score(a.freeze_sha256)
