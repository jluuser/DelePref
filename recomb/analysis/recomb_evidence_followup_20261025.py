"""Five frozen morphology-matched control redraws; no training or WT re-encoding."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

# Archive adapter: the original study's local paths are supplied explicitly.
ROOT = Path(os.environ.get('DELEPREF_RESEARCH_ROOT', '.')).resolve()
OUT = Path(os.environ.get('DELEPREF_FOLLOWUP_OUTPUT', ROOT / 'reports/recomb_evidence_followup_20261025')).resolve()
sys.path[:0] = [str(ROOT/'analysis'), str(ROOT/'code')]

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')
def spans(r): return tuple(map(tuple,json.loads(r['deleted_intervals']) if isinstance(r['deleted_intervals'],str) else r['deleted_intervals']))
def cpu():
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Use a CPU Slurm allocation')

def prepare():
    cpu()
    if (OUT/'protocol.json').exists(): raise FileExistsError('Experiment already prepared')
    import torch
    import numpy as np
    import recomb_E9_shape_20261021 as original
    from recomb_E9_shape_sampler_20261021 import bits_for,intervals,signature,sample
    import delepref_v8_runtime as rt
    import delepref_v8_main as parent
    import delepref_unlimited_scorer as scoring
    from recomb_confirm_v2_E9_catalog import explicit_correspondence,physical_request,compatible_identity
    original.check(ROOT,sha(ROOT/original.FZ))
    source=ROOT/'reports/recomb_confirm_v2_E9_completion'
    panel=read(source/'panel.json')
    pools={r['protein_id']:r for r in read(source/'candidate_pools.json') if r['method']=='policy_completion'}
    winners=[r for r in read(source/'selected_structure_outputs.json') if r['method']=='policy_completion']
    original_rows=read(ROOT/'reports/recomb_confirm_v2_General100_plans/candidate_scores.json')
    assert len(panel)==len(pools)==len(winners)==20
    rules=dict(status='FROZEN_BEFORE_NEW_CONTROL_SAMPLING',WT=20,repetitions=5,seed=20261025,
        namespace='DelePref_completion_seed_robustness_20261025_v1',K=8,common_policy_slots=4,
        matched_control_slots=4,max_attempts_per_slot=131072,new_training=False,new_WT_encoding=False,
        primary_analysis='Average five control effects within each WT, then paired component bootstrap over20 WTs; do not treat100 outcomes as independent.',
        selection='same original frozen scorer; all replicates retained, no best-seed selection',
        original_freeze_sha256=sha(ROOT/original.FZ),source_script_sha256=sha(__file__),
        original_checkpoint_sha256=original.CHECKPOINT,post_outcome_sensitivity_analysis=True,
        source_files={str(source/n):sha(source/n) for n in ['panel.json','candidate_pools.json','selected_structure_outputs.json','statistical_protocol.json']})
    write(OUT/'protocol.json',rules)
    torch.set_num_threads(4)
    runtime,api=rt.load(ROOT,device='cpu',cached_heads_only=True)
    cache=torch.load(ROOT/'reports/recomb_confirm_v2_General100_model/wt_cache.pt',map_location='cpu',weights_only=True)
    assert cache['real_encoder'] and cache['unique_WT']==100 and cache['checkpoint_sha256']==original.CHECKPOINT
    chunk=parent.cached_chunk(api,cache,read(ROOT/'reports/recomb_confirm_v2_general_native/encoder_inputs.json'))
    all_pools=[];audits=[];compat=[]
    for wt in panel:
        pid=wt['protein_id'];pool=pools[pid];n=wt['sequence_length'];parents=pool['rows'][4:]
        forced=bits_for(wt['seed']['spans']);anchors=0
        for a,b in wt['seed']['spans']: anchors|=1<<(a-1);anchors|=1<<(b+1)
        old8=[r for r in original_rows if r['protein_id']==pid and r['method']=='delepref_unlimited' and r['budget']==.2 and r['generation_seed']==13]
        assert len(old8)==8 and len(parents)==4
        checked=scoring.score_candidates(runtime,api,chunk,copy.deepcopy(pool['rows'][:4]))
        error=max(abs(a['scorer_score']-b['scorer_score']) for a,b in zip(checked,pool['rows'][:4]))
        if error>1e-5: raise ValueError('Original scorer mismatch '+pid)
        compat.append(dict(protein_id=pid,max_score_error=error))
        excluded={bits_for(spans(r))&~forced for r in old8+parents}
        for rep in range(5):
            prior=set();new=[]
            for slot,p in enumerate(parents):
                learned=bits_for(spans(p));shape=signature(learned&~forced,n)
                stream=int(digest(dict(namespace=rules['namespace'],seed=rules['seed'],replicate=rep,WT=wt['WT_sha256'],slot=slot,shape=shape))[:16],16)
                val,diag=sample(n,shape,forced|anchors,excluded|prior,stream,rules['max_attempts_per_slot'])
                audits.append(dict(protein_id=pid,replicate=rep,slot=slot,stream=stream,**diag))
                if val is None: raise RuntimeError('Matching shortfall; do not relax constraints')
                full=forced|val
                assert signature(full,n)==signature(learned,n) and full&anchors==0 and full.bit_count()==round(.2*n)
                prior.add(val);ss=tuple(map(tuple,intervals(full,n)));row=copy.deepcopy(p)
                row.update(method='shape_repeat_'+str(rep),deleted_intervals=api._intervals_json(ss),
                    deleted_length=sum(b-a+1 for a,b in ss),deletion_mask_sha256=api._mask_sha256(n,ss),
                    span_count=len(ss),policy_energy=0.,policy_native_rank=0,scorer_rank=0,scorer_score='',
                    generation_kind='retrieval_shape_repeat_completed',parent_completion_slot=slot,
                    replicate=rep,sampler_seed=stream)
                new.append(row)
            scored=scoring.score_candidates(runtime,api,chunk,new)
            combined=copy.deepcopy(pool['rows'][:4])+scored
            order=sorted(range(8),key=lambda i:(-combined[i]['scorer_score'],spans(combined[i])))
            for rank,i in enumerate(order,1):combined[i]['scorer_rank']=rank
            for i,r in enumerate(combined):r.update(method='shape_repeat_'+str(rep),raw_candidate_rank=i+1,replicate=rep)
            chosen=copy.deepcopy(combined[order[0]])
            edited,mapping=explicit_correspondence(chosen['wildtype_sequence'],spans(chosen))
            chosen.update(edited_sequence=edited,edited_sequence_sha256=hashlib.sha256(edited.encode()).hexdigest(),pool_sha256=digest(combined))
            winners.append(chosen);all_pools.append(dict(protein_id=pid,replicate=rep,rows=combined))
        print('REPEAT_POOLS_COMPLETE '+pid,flush=True)
    assert len(winners)==120 and len(all_pools)==100 and len(audits)==400
    for name,value in [('selected_outputs.json',winners),('candidate_pools.json',all_pools),('sampling_audit.json',audits),('scorer_compatibility.json',compat)]:write(OUT/name,value)
    native={r['protein_id']:r for r in read(ROOT/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    profile=read(ROOT/'reports/recomb_confirm_v2_General100_catalog/predictor_profiles.json')['esmfold']
    preflight=read(ROOT/'reports/delepref_v8_predictors/esmfold_preflight.json')
    weight=next(r['sha256'] for r in preflight['files'] if r['path'].endswith('pytorch_model.bin'))
    physical={};logical=[]
    for r in winners:
        wt=native[r['protein_id']];edited,mapping=explicit_correspondence(wt['sequence'],spans(r))
        assert edited==r['edited_sequence']
        req=physical_request(edited,profile,weight);physical.setdefault(req['task_id'],req)
        row=dict(protein_id=r['protein_id'],cohort='general',WT_sha256=wt['WT_sha256'],wildtype_sequence=wt['sequence'],
            edited_sequence=edited,edited_sequence_sha256=r['edited_sequence_sha256'],method=r['method'],budget=.2,
            evaluation_role='completion_random_seed_sensitivity',predictor='esmfold',profile_name='esmfold',profile_family='esmfold',
            seed=req['seed'],sample=0,physical_id=req['task_id'],correspondence=mapping,
            native_to_prediction_chains=[dict(native_chain_id=wt['reference_chain'],prediction_chain_id='A',role='design')])
        row['request_id']=digest({k:row[k] for k in ['protein_id','method','physical_id']});logical.append(row)
    folders=['recomb_E9_shape_structures_20261021','recomb_confirm_v2_E9_structures','recomb_confirm_v2_General100_structures',
             'recomb_benign_structures_20261017','delepref_v8_structures','recomb_confirm_v2_span_structures']
    reuse={};journals={}
    for req in physical.values():
        for folder in folders:
            d=ROOT/'reports'/folder
            journal=next((d/n for n in ['prediction_journal.jsonl','budget_ledger.jsonl'] if (d/n).exists()),None)
            if journal is None:continue
            if str(journal) not in journals:
                raw=journal.read_bytes()
                if raw and not raw.endswith(b'\n'):raise ValueError('Incomplete historical journal')
                journals[str(journal)]={x['task_id']:x for x in map(json.loads,raw.splitlines())}
            for name in [req['task_id'],'esmfold_'+req['sequence_sha256']]:
                record=d/'records'/(name+'.json')
                if not record.exists():continue
                saved=read(record)
                if not compatible_identity(saved,req):continue
                event=journals[str(journal)].get(saved.get('task_id',name),{})
                if saved.get('status')!='SUCCEEDED' or event.get('state') not in ['succeeded','cache_hit'] or event.get('output_sha')!=saved['output_sha256']:raise ValueError('Invalid matching cache')
                for a,b in [('sequence_sha','sequence_sha256'),('predictor_weights_sha','weights_sha256'),('profile_sha','profile_sha256'),('seed','seed'),('sample','sample')]:
                    assert event.get(a)==req[b]
                artifact=Path(saved['output_path']);assert sha(artifact)==saved['output_sha256']
                with np.load(artifact,allow_pickle=False) as z:
                    assert str(z['sequence'])==req['sequence'] and z['coordinates'].shape==(len(req['sequence']),3)
                reuse[req['task_id']]=dict(record_path=str(record),record_sha256=sha(record),artifact_path=str(artifact),artifact_sha256=sha(artifact),journal_path=str(journal),journal_sha256=sha(journal))
                break
            if req['task_id'] in reuse:break
    assert all(r['physical_id'] in reuse for r in logical if r['method']=='policy_completion')
    missing=[r for k,r in sorted(physical.items()) if k not in reuse]
    assert len(missing)<=100
    stats=copy.deepcopy(read(source/'statistical_protocol.json'))
    stats.update(WT=20,repetitions=5,bootstrap_seed=20261025,methods=['policy_completion']+['shape_repeat_'+str(i) for i in range(5)],
        status='ADDITIONAL_CONTROL_DRAWS_FIXED20',within_WT_first=True)
    stats['compute']=dict(structure_call_limit=max(1,len(missing)*2),structure_categories=dict(general_esmfold=len(missing),general_second_predictor=0,retry=len(missing)),structure_retries_max=1)
    for name,v in [('logical_requests.json',logical),('physical_requests.json',sorted(physical.values(),key=lambda x:x['task_id'])),('compatible_cache.json',reuse),('predictor_profiles.json',{'esmfold':profile}),('statistical_protocol.json',stats),('missing_requests.json',missing)]:write(OUT/'catalog'/name,v)
    files={str(p):sha(p) for p in [OUT/'protocol.json',Path(__file__),OUT/'selected_outputs.json',OUT/'candidate_pools.json',OUT/'sampling_audit.json']+list((OUT/'catalog').glob('*.json'))}
    for f in read(ROOT/'configs/recomb_confirm_v2_General100_confirmation_freeze.json')['files']:
        assert sha(f['path'])==f['sha256'];files[f['path']]=f['sha256']
    for rel in ['analysis/recomb_E9_shape_predict_20261021.py','analysis/recomb_confirm_v2_general_evaluate.py','analysis/recomb_confirm_v2_E9_evaluate.py','code/prediction_journal.py','code/budget_ledger.py']:
        files[str(ROOT/rel)]=sha(ROOT/rel)
    write(OUT/'confirmation.json',dict(fold_allowed=True,all_selected_outputs_frozen=True,files=[dict(path=p,sha256=h) for p,h in sorted(files.items())],WT=20,repetitions=5,logical=120,physical=len(physical),cache=len(reuse),missing=len(missing)))
    done=dict(status='PREPARED',WT=20,repetitions=5,logical=120,physical=len(physical),cache=len(reuse),new_predictions_required=len(missing),confirmation_sha256=sha(OUT/'confirmation.json'),job_id=os.environ['SLURM_JOB_ID'])
    write(OUT/'preparation_completion.json',done);print(json.dumps(done),flush=True)

def fold():
    import recomb_E9_shape_predict_20261021 as worker
    worker.BASE=OUT/'structures';worker.CAT=OUT/'catalog';worker.FREEZE=OUT/'confirmation.json'
    sys.argv=['frozen_esmfold_worker','--predictor','esmfold','--freeze-sha256',sha(OUT/'confirmation.json'),'--phase','all']
    worker.main()

def evaluate():
    cpu()
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    from recomb_confirm_v2_general_evaluate import metric_pair,effect
    from recomb_confirm_v2_E9_evaluate import bounds,ENDPOINTS
    from recomb_confirm_v2_application_serialization import json_native
    for p in read(OUT/'confirmation.json')['files']:
        if sha(p['path'])!=p['sha256']:raise ValueError('Frozen input changed '+p['path'])
    logical=read(OUT/'catalog/logical_requests.json');physical={r['task_id']:r for r in read(OUT/'catalog/physical_requests.json')};reuse=read(OUT/'catalog/compatible_cache.json')
    native={r['protein_id']:r for r in read(ROOT/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    artifacts={}
    for task,req in physical.items():
        if task in reuse:p=Path(reuse[task]['artifact_path']);h=reuse[task]['artifact_sha256']
        else:
            record=read(OUT/'structures/records'/(task+'.json'))
            if record['status']!='SUCCEEDED':raise RuntimeError('Unsuccessful prediction retained; complete result unavailable')
            for k in ['sequence_sha256','weights_sha256','profile_sha256','seed','sample']:assert record[k]==req[k]
            p=Path(record['output_path']);h=record['output_sha256']
        assert sha(p)==h;artifacts[task]=str(p)
    def key(r):return (r['protein_id'],r['physical_id'],json.dumps(r['correspondence'],sort_keys=True))
    old={}
    for rel in ['recomb_confirm_v2_General100_evaluation','recomb_confirm_v2_E9_evaluation','recomb_E9_shape_evaluation_20261021']:
        for r in read(ROOT/'reports'/rel/'per_output_metrics.json'):
            if r['predictor']=='esmfold':old[key(r)]={k:r[k] for k in ['common_structure','fixed_identity_geometry']}
    metrics={};jobs={}
    for r in logical:
        k=key(r)
        if k in old:metrics[k]=old[k]
        else:jobs.setdefault(k,(str(ROOT),k,native[r['protein_id']]['wt_native_json'],artifacts[r['physical_id']],len(r['edited_sequence']),r['correspondence']))
    if jobs:
        with ProcessPoolExecutor(max_workers=4) as pool:metrics.update(pool.map(metric_pair,jobs.values()))
    rows=[dict(r,prediction_status='SUCCEEDED',**metrics[key(r)]) for r in logical]
    by={(r['protein_id'],r['method']):r for r in rows};ids=sorted({r['protein_id'] for r in rows})
    assert len(rows)==len(by)==120 and len(ids)==20
    protocol=read(OUT/'catalog/statistical_protocol.json');comparisons=[];paired=[]
    for endpoint in ENDPOINTS:
        rep_pairs={i:([],[]) for i in range(5)};avglo=[];avghi=[]
        for pid in ids:
            p=by[pid,'policy_completion'];a,b=bounds(p,endpoint);lows=[];highs=[]
            for rep in range(5):
                q=by[pid,'shape_repeat_'+str(rep)];c,d=bounds(q,endpoint)
                same=key(p)==key(q);lo=0. if same else a-d;hi=0. if same else b-c
                lows.append(lo);highs.append(hi);rep_pairs[rep][0].append((pid,lo));rep_pairs[rep][1].append((pid,hi))
                paired.append(dict(protein_id=pid,replicate=rep,endpoint=endpoint,difference_lower=lo,difference_upper=hi,same_output=same))
            avglo.append((pid,float(np.mean(lows))));avghi.append((pid,float(np.mean(highs))))
        for name,lo,hi in [('within_WT_average_of_five',avglo,avghi)]+[('replicate_'+str(i),*rep_pairs[i]) for i in range(5)]:
            comparisons.append(dict(analysis=name,endpoint=endpoint,lower_effect=effect(lo,protocol['component_by_protein_id'],20261025),upper_effect=effect(hi,protocol['component_by_protein_id'],20261025),WT=20))
    for name,v in [('per_output_metrics.json',rows),('paired_per_WT_and_repeat.json',paired),('paired_comparisons.json',comparisons)]:write(OUT/name,json_native(v))
    done=dict(status='COMPLETE',WT=20,repetitions=5,new_predictions=len(physical)-len(reuse),logical=120,source_sha256=sha(__file__),confirmation_sha256=sha(OUT/'confirmation.json'),job_id=os.environ['SLURM_JOB_ID'])
    write(OUT/'completion.json',done);print(json.dumps(done),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','fold','evaluate']);args=parser.parse_args()
    {'prepare':prepare,'fold':fold,'evaluate':evaluate}[args.phase]()
