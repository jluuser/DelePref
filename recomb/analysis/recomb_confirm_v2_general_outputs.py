"""Assemble the full frozen General100 matrix, preserving edit and shape distinctions."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write
from recomb_edit_correspondence_20261017 import consensus

BASE=Path('reports/recomb_confirm_v2_General100_outputs')
FZ=Path('configs/recomb_confirm_v2_General100_output_assembly_freeze.json')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('General full output assembly already frozen')
    names=['analysis/recomb_confirm_v2_general_outputs.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_edit_correspondence_20261017.py','slurm/recomb_confirm_v2_general_outputs.sbatch',
        'reports/recomb_confirm_v2_general_native_v2/General100_panel.json',
        'reports/recomb_confirm_v2_General100_subsets/subsets.json',
        'reports/recomb_confirm_v2_General100_plans/completion.json','reports/recomb_confirm_v2_General100_plans/all_selections.json',
        'reports/recomb_confirm_v2_General100_Raygun_native_audit/completion.json',
        'reports/recomb_confirm_v2_General100_scisor_L/completion.json','reports/recomb_confirm_v2_General100_scisor_L/all_conditions.json',
        'reports/recomb_confirm_v2_General100_SCISOR_L_K8_score/completion.json',
        'reports/recomb_confirm_v2_General100_SCISOR_L_K8_score/all_selections.json',
        'reports/recomb_confirm_v2_General100_mechanisms/completion.json']
    mech=root/'reports/recomb_confirm_v2_General100_mechanisms'
    names += [str((mech/n).relative_to(root)) for n in ['all_selections.json','full_pool_candidates.json','quota_control_selections.json','diagnostics.json','subsets.json']]
    for folder in ['recomb_confirm_v2_General100_plans','recomb_confirm_v2_General100_SCISOR_L_K8_score','recomb_confirm_v2_General100_mechanisms']:
        done=read(root/'reports'/folder/'completion.json')
        for p,h in done['outputs'].items():
            if sha(root/'reports'/folder/p)!=h:raise ValueError('Completed generator output changed')
    audit=read(root/'reports/recomb_confirm_v2_General100_Raygun_native_audit/completion.json')
    if audit['conditions']!=220 or audit['raw_candidates_audited']!=110000:raise ValueError('Native Raygun pool audit incomplete')
    native={r['wt_native_json']:r['wt_native_json_sha256'] for r in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    write(root/FZ,dict(status='GENERAL100_ALL_MAIN_BUDGET_MECHANISM_AND_POOL_OUTPUT_ASSEMBLY_FROZEN',
        files={p:sha(root/p) for p in names},Raygun_source_sha256=audit['source_sha256'],native_reference_sha256=native,
        methods=['delepref_unlimited','rag_unlimited','gap_frequency','terminal','scisor_L_native','raygun_native'],
        main_WT=100,core_nonWT=1320,mechanism_controls=220,full_pool_logical_candidates=192,
        quota_controls=12,SCISOR_K8_controls=20,WT_references=100,
        Raygun_correspondence='all optimal unit-edit stable correspondence, preserve substitutions/insertions/ambiguity',
        biological_identity_not_inferred_from_structural_alignment=True,structure_prediction_allowed=False))
    (root/BASE).mkdir(parents=True,exist_ok=True);print(sha(root/FZ),flush=True)


def ray_mapping(item):
    row=item
    mapping=consensus(row['sequence'],row['edited_sequence'])
    mapping['stable_substitution_positions']=[i for i,j in enumerate(mapping['wt_to_edit']) if j is not None and row['sequence'][i]!=row['edited_sequence'][j]]
    return row,mapping


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen output assembly input changed: '+p)
    for p,h in {**f['Raygun_source_sha256'],**f['native_reference_sha256']}.items():
        path=Path(p) if Path(p).is_absolute() else root/p
        if sha(path)!=h:raise ValueError('Native pool or reference identity changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Output assembly claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    panel={r['protein_id']:r for r in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    result=[]
    def add(row,method,role,source,mapping=None):
        pid=row['protein_id'];wt=panel[pid]['sequence'];beta=float(row.get('budget',0.))
        if row.get('wildtype_sequence',row.get('sequence',wt))!=wt:raise ValueError('Output WT differs from ordinary frozen roster')
        if mapping is None:
            raw=row.get('deleted_intervals',[]);spans=json.loads(raw) if isinstance(raw,str) else raw
            deleted=set()
            for a,b in spans:
                if not 0<=a<=b<len(wt) or deleted.intersection(range(a,b+1)):raise ValueError('Invalid explicit deletion spans')
                deleted.update(range(a,b+1))
            edited=''.join(c for i,c in enumerate(wt) if i not in deleted)
            mp=[];j=0
            for i in range(len(wt)):
                mp.append(None if i in deleted else j)
                if i not in deleted:j+=1
            mapping=dict(method='explicit_native_deletion_plan' if beta else 'identity_WT',wt_to_edit=mp,
                position_states=['deleted' if i in deleted else 'stable' for i in range(len(wt))],
                stable_positions=len(edited),certain_deleted_positions=len(deleted),ambiguous_positions=0)
            if row.get('edited_sequence',edited)!=edited:raise ValueError('Explicit edit reconstruction differs')
        else:edited=row['edited_sequence']
        expected=len(wt) if beta==0 else len(wt)-max(1,min(len(wt)-1,round(len(wt)*beta)))
        digest=hashlib.sha256(edited.encode()).hexdigest()
        if len(edited)!=expected or row.get('edited_sequence_sha256',digest)!=digest:raise ValueError('Selected output budget/SHA differs')
        result.append(dict(protein_id=pid,method=method,budget=beta,evaluation_role=role,
            wildtype_sequence=wt,WT_sha256=panel[pid]['WT_sha256'],edited_sequence=edited,
            edited_sequence_sha256=digest,correspondence=mapping,source_path=source,
            selection_metadata={k:row.get(k) for k in ['generation_seed','candidate_index','raw_candidate_rank',
                'scorer_score','scorer_rank','policy_energy','retrieval_source','retrieval_seed_spans','generation_kind',
                'raw_candidate_count','PLL_ranked_count','actual_unique_masks','num_insertions','num_deletions','num_substitutions']}))
    gp='reports/recomb_confirm_v2_General100_plans/all_selections.json'
    for row in read(root/gp):
        if row['generation_seed']==13:add(row,row['method'],'primary20pct' if row['budget']==.2 else 'budget_curve',gp)
    sp='reports/recomb_confirm_v2_General100_scisor_L/all_conditions.json'
    for row in read(root/sp):
        if row['mode']=='native':add(row,'scisor_L_native','primary20pct' if row['budget']==.2 else 'budget_curve',sp)
    rays=[];paths=[]
    for index in range(220):
        p='reports/recomb_confirm_v2_General100_raygun_v2/raygun/condition_%03d.json'%index
        rays.append(read(root/p));paths.append(p)
    with ProcessPoolExecutor(max_workers=8) as executor:
        for (row,mapping),source in zip(executor.map(ray_mapping,rays),paths):
            add(row,'raygun_native','primary20pct' if row['budget']==.2 else 'budget_curve',source,mapping)
    if len(result)!=1320:raise ValueError('Incomplete six-method core')
    core=list(result)
    mp='reports/recomb_confirm_v2_General100_mechanisms/'
    for row in read(root/(mp+'all_selections.json')):add(row,row['method'],'mechanism_control',mp+'all_selections.json')
    for row in read(root/(mp+'quota_control_selections.json')):add(row,row['method'],'quota_control',mp+'quota_control_selections.json')
    for row in read(root/(mp+'full_pool_candidates.json')):
        add(row,'pool_'+row['full_pool_source']+'_'+str(row['raw_candidate_rank']),'full_pool',mp+'full_pool_candidates.json')
    kp='reports/recomb_confirm_v2_General100_SCISOR_L_K8_score/all_selections.json'
    for row in read(root/kp):add(row,row['method'],'SCISOR_K8_control',kp)
    for pid,p in panel.items():add(dict(protein_id=pid,budget=0.,edited_sequence=p['sequence']),'WT','WT_reference','General100_panel.json')
    keys=[(r['protein_id'],r['budget'],r['method'],r['evaluation_role']) for r in result]
    if len(result)!=1864 or len(keys)!=len(set(keys)):raise ValueError('General complete matrix lost or duplicated logical records')
    write(out/'core_selected_outputs.json',core);write(out/'all_logical_outputs.json',result)
    done=dict(status='GENERAL100_FULL_SIX_METHOD_AND_CONTROL_OUTPUTS_VERIFIED_NOT_STRUCTURE_EVALUATED',
        logical_outputs=len(result),core_nonWT=len(core),role_counts=dict(Counter(r['evaluation_role'] for r in result)),
        primary20pct_methods=dict(Counter(r['method'] for r in core if r['budget']==.2)),
        Raygun_X_query_outputs=sum('X' in r['edited_sequence'] for r in core if r['method']=='raygun_native'),
        Raygun_ambiguous_WT_positions=sum(r['correspondence']['ambiguous_positions'] for r in core if r['method']=='raygun_native'),
        outputs={p:sha(out/p) for p in ['core_selected_outputs.json','all_logical_outputs.json']},
        job_id=os.environ['SLURM_JOB_ID'],new_model_calls=0,new_structure_predictions=0,fold_allowed=False)
    write(out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
