"""Verify the four-method application matrix, including complete partner identities."""
import gzip
import hashlib
import json
import math
from pathlib import Path
from recomb_scope_screen_20261017 import read,sha,write
from recomb_edit_correspondence_20261017 import consensus


def build(root):
    baseline=root/'reports/recomb_benign_app_baselines_20261017'
    if read(baseline/'raygun/completion_0.json')['conditions']!=12 or read(baseline/'scisor/completion.json')['conditions']!=2:raise ValueError('incomplete baselines')
    panel_path=root/'reports/recomb_benign_app_model_20261017/panel.json'
    expected_sha=read(root/'configs/recomb_benign_app_inputs_freeze_20261017.json')['panel_sha256']
    if sha(panel_path)!=expected_sha:raise ValueError('reviewed application input changed')
    panel={r['protein_id']:r for r in read(panel_path)}
    approved={r['protein_id'] for r in read(root/'reports/recomb_benign_application_scope_20261017/audit.json') if r['status']=='RETAIN'}
    if set(panel)!=approved or len(panel)!=23:raise ValueError('scope changed')
    result=[];sources={}
    def add(row,method,path):
        p=panel[row['protein_id']];wt=p['sequence'];edited=row['edited_sequence']
        if row.get('wildtype_sequence',row.get('sequence',wt))!=wt:raise ValueError('WT mismatch')
        if len(edited)!=p['target_length'] or hashlib.sha256(edited.encode()).hexdigest()!=row['edited_sequence_sha256']:raise ValueError('edited identity/budget mismatch')
        if method=='raygun_native':mapping=consensus(wt,edited)
        else:
            spans=row['deleted_intervals'];spans=json.loads(spans) if isinstance(spans,str) else spans
            deleted=set()
            for a,b in spans:
                if not 0<=a<=b<len(wt) or deleted.intersection(range(a,b+1)):raise ValueError('invalid deletion coordinates')
                deleted.update(range(a,b+1))
            if ''.join(c for i,c in enumerate(wt) if i not in deleted)!=edited:raise ValueError('explicit reconstruction mismatch')
            mp=[];j=0
            for i in range(len(wt)):
                mp.append(None if i in deleted else j)
                if i not in deleted:j+=1
            mapping=dict(method='explicit_native_deletion_plan',wt_to_edit=mp,
                position_states=['deleted' if i in deleted else 'stable' for i in range(len(wt))],
                stable_positions=len(edited),certain_deleted_positions=len(deleted),ambiguous_positions=0)
        chains=[dict(native_chain_id=p['design_chain'],role='design',sequence=edited)]
        chains.extend(dict(native_chain_id=c['chain_id'],role='partner',sequence=c['sequence']) for c in p['partners'])
        signature=hashlib.sha256(json.dumps(chains,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        relative=str(path.relative_to(root))
        if relative not in sources:sources[relative]=sha(path)
        result.append(dict(protein_id=p['protein_id'],dataset=p['dataset'],method=method,budget=.2,
            WT_sha256=p['sequence_sha256'],wildtype_sequence=wt,edited_sequence=edited,
            edited_sequence_sha256=row['edited_sequence_sha256'],correspondence=mapping,
            chains=chains,full_chain_construct_sha256=signature,source_path=relative,
            legacy_complex_construct_unchanged=p.get('legacy_construct_identity_unchanged'),
            old_structure_cache_not_automatically_accepted=True))
    path=root/'reports/recomb_benign_app_model_20261017/all_selections.json'
    completion=read(path.parent/'planning_completion.json')
    if sha(path)!=completion['outputs'][path.name]:raise ValueError('model selections changed')
    for row in read(path):add(row,row['method'],path)
    path=baseline/'reused_generation_outputs.json'
    if sha(path)!=read(root/'configs/recomb_benign_app_baselines_freeze_20261017.json')['files'][str(path.relative_to(root))]:raise ValueError('reused outputs changed')
    for row in read(path):add(row,'raygun_native' if row['method']=='Raygun_500_native' else 'scisor_native',path)
    for i,task in enumerate(read(baseline/'raygun_tasks.json')):
        path=baseline/'raygun'/('condition_%03d.json'%i);row=read(path)
        for key in ['protein_id','sequence','target_length','budget']:
            if row[key]!=task[key]:raise ValueError('Raygun task changed')
        raw=path.with_name('raw_%03d.jsonl.gz'%i)
        if sha(raw)!=row['raw_sha256']:raise ValueError('raw pool changed')
        with gzip.open(raw,'rt') as f:pool=[json.loads(line) for line in f]
        if len(pool)!=500 or len(row['native_top50'])!=50:raise ValueError('native candidate count mismatch')
        if any(len(c['sequence'])!=task['target_length'] or not math.isfinite(c['adjusted_PLL']) for c in pool):raise ValueError('invalid raw candidate')
        order=sorted(range(500),key=lambda j:(-pool[j]['adjusted_PLL'],j))[:50]
        for rank,(j,c) in enumerate(zip(order,row['native_top50']),1):
            if c['generation_index']!=j or c['pll_rank']!=rank or c['sequence']!=pool[j]['sequence']:raise ValueError('native ranking changed')
        if row['edited_sequence']!=pool[order[0]]['sequence']:raise ValueError('Raygun winner mismatch')
        add(row,'raygun_native',path)
    path=baseline/'scisor/all_conditions.json'
    for row in read(path):add(row,'scisor_native',path)
    keys=[(r['protein_id'],r['method']) for r in result]
    expected={(pid,m) for pid in panel for m in ['delepref_unlimited','rag_unlimited','scisor_native','raygun_native']}
    if len(keys)!=92 or set(keys)!=expected:raise ValueError('incomplete/duplicate application matrix')
    out=root/'reports/recomb_benign_application_outputs_20261017';out.mkdir(exist_ok=False)
    write(out/'all_selected_outputs.json',result)
    report=dict(status='REVIEWED_APPLICATION_FOUR_METHOD_MATRIX_VERIFIED',WT=23,Binder_cases=12,Enzyme_cases=11,
        methods=4,outputs=92,new_Raygun_conditions=12,new_Raygun_raw_candidates_audited=6000,
        reused_generation_outputs=32,full_partner_construct_identity_included=True,
        source_sha256=sources,panel_sha256=sha(panel_path),output_sha256=sha(out/'all_selected_outputs.json'),
        structure_prediction_allowed=False,new_structure_predictions=0)
    write(out/'completion.json',report);print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))


if __name__=='__main__':build(Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
