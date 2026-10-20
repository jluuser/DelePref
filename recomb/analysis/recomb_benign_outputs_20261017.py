"""Verify completed screened generators and build a common output roster; no folding."""
import gzip
import hashlib
import json
import math
from pathlib import Path
from recomb_scope_screen_20261017 import read,sha,write
from recomb_edit_correspondence_20261017 import consensus

BASE=Path('reports/recomb_benign_outputs_20261017')


def build(root):
    contract=read(root/'configs/recomb_benign_output_assembly_contract_20261017.json')
    for path,digest in contract['source_sha256'].items():
        if sha(root/path)!=digest:raise ValueError('assembly source changed')
    b=root/'reports/recomb_benign_baselines_20261017'
    scope='reports/recomb_benign_scope_20261017/general_panel.json'
    if sha(root/scope)!=read(root/'configs/recomb_benign_baselines_freeze_20261017.json')['files'][scope]:raise ValueError('reviewed scope changed')
    for shard in [0,1]:
        if read(b/'raygun'/('completion_%d.json'%shard))['conditions']!=29:raise ValueError('Raygun shard incomplete')
    panel=read(root/'reports/recomb_benign_scope_20261017/general_panel.json');allowed={p['protein_id']:p for p in panel}
    subsets=read(root/'reports/recomb_benign_scope_20261017/retained_subsets.json');stress=set(subsets['completion_ablation'])
    expected={(pid,.2) for pid in allowed}|{(pid,beta) for pid in stress for beta in [.1,.3,.4]}
    result=[];source_hashes={}
    def add(row,method,source,role='comparison'):
        pid=row['protein_id'];wt=allowed[pid]['sequence'];edited=row['edited_sequence'];beta=float(row['budget'])
        if row.get('wildtype_sequence',row.get('sequence',wt))!=wt:raise ValueError('WT identity mismatch')
        if len(edited)!=len(wt)-max(1,min(len(wt)-1,round(len(wt)*beta))):raise ValueError('edited budget mismatch')
        digest=hashlib.sha256(edited.encode()).hexdigest()
        if digest!=row['edited_sequence_sha256']:raise ValueError('edited SHA mismatch')
        if method=='raygun_native':mapping=consensus(wt,edited)
        else:
            spans=row['deleted_intervals'];spans=json.loads(spans) if isinstance(spans,str) else spans
            deleted=set()
            for a,z in spans:
                if not 0<=a<=z<len(wt) or deleted.intersection(range(a,z+1)):raise ValueError('invalid deletion intervals')
                deleted.update(range(a,z+1))
            if ''.join(c for i,c in enumerate(wt) if i not in deleted)!=edited:raise ValueError('explicit mapping reconstruction mismatch')
            mp=[];j=0
            for i in range(len(wt)):
                mp.append(None if i in deleted else j)
                if i not in deleted:j+=1
            mapping=dict(method='explicit_native_deletion_plan',wt_to_edit=mp,
                position_states=['deleted' if i in deleted else 'stable' for i in range(len(wt))],
                stable_positions=len(edited),certain_deleted_positions=len(deleted),ambiguous_positions=0)
        relative=str(source.relative_to(root))
        if relative not in source_hashes:source_hashes[relative]=sha(source)
        result.append(dict(protein_id=pid,method=method,budget=beta,evaluation_role=role,
            wildtype_sequence=wt,WT_sha256=allowed[pid]['sequence_sha256'],edited_sequence=edited,
            edited_sequence_sha256=digest,correspondence=mapping,source_path=relative,
            selection_metadata={k:row.get(k) for k in ['generation_seed','candidate_index','scorer_score','scorer_rank',
                'raw_candidate_count','PLL_ranked_count','retrieval_source','retrieval_seed_spans','generation_kind',
                'selection','num_insertions','num_deletions','num_substitutions']}))
    # Only reviewed query files are opened. Parent mixed outputs are not resumed.
    raygun=[]
    for index,task in enumerate(read(b/'raygun_tasks.json')):
        path=b/'raygun'/('condition_%03d.json'%index);raygun.append((task,path,read(path)))
    for item in read(b/'raygun_reused_conditions.json'):
        path=root/item['result_path']
        if sha(path)!=item['result_sha256']:raise ValueError('reused result changed')
        row=read(path);raygun.append((row,path,row))
    seen=set()
    for task,path,row in raygun:
        key=(row['protein_id'],float(row['budget']))
        if key not in expected or key in seen:raise ValueError('Raygun condition mismatch')
        seen.add(key)
        for k in ['protein_id','sequence','target_length','budget']:
            if task[k]!=row[k]:raise ValueError('Raygun task identity mismatch')
        if row['sequence']!=allowed[key[0]]['sequence']:raise ValueError('unreviewed WT')
        raw=path.with_name(path.name.replace('condition_','raw_').replace('.json','.jsonl.gz'))
        if sha(raw)!=row['raw_sha256']:raise ValueError('raw candidates changed')
        with gzip.open(raw,'rt') as f:pool=[json.loads(line) for line in f]
        if len(pool)!=500 or len(row['native_top50'])!=50 or row['raw_candidate_count']!=500 or row['PLL_ranked_count']!=50:raise ValueError('native500/50 count mismatch')
        for i,c in enumerate(pool):
            if c['generation_index']!=i or len(c['sequence'])!=row['target_length'] or set(c['sequence'])-set('ACDEFGHIKLMNPQRSTVWYX') or not math.isfinite(c['adjusted_PLL']):raise ValueError('invalid native candidate')
        order=sorted(range(500),key=lambda i:(-pool[i]['adjusted_PLL'],i))[:50]
        for rank,(index,c) in enumerate(zip(order,row['native_top50']),1):
            if c['generation_index']!=index or c['sequence']!=pool[index]['sequence'] or c['pll_rank']!=rank:raise ValueError('native PLL ranking mismatch')
            if c['adjusted_PLL']!=pool[index]['adjusted_PLL'] or c['sequence_sha256']!=hashlib.sha256(c['sequence'].encode()).hexdigest():raise ValueError('native top50 identity mismatch')
        if row['edited_sequence']!=pool[order[0]]['sequence']:raise ValueError('Raygun winner changed')
        add(row,'raygun_native',path)
    if seen!=expected:raise ValueError('incomplete Raygun matrix')
    path=root/'reports/recomb_benign_plans_20261017/all_selections.json'
    for row in read(path):
        if row['generation_seed']!=13:continue
        add(row,row['method'],path,'completion_ablation' if row['method']=='rag_direct_only' else 'comparison')
    path=root/'reports/recomb_task_budget_20261017/all_selections.json'
    for row in read(path):
        if row['protein_id'] in allowed and (row['protein_id'],float(row['budget'])) in expected and row['generation_seed']==13 and row['method']=='terminal':add(row,'terminal',path)
    path=root/'reports/delepref_v8_scisor_native_v3/native_plans.json'
    for row in read(path):
        if row['protein_id'] in allowed:add(row,'scisor_native',path)
    path=b/'scisor/all_conditions.json'
    for row in read(path):
        if row['mode']=='native':add(row,'scisor_native',path)
    path=root/'reports/recomb_benign_scisor_score_20261017/all_selections.json'
    for row in read(path):add(row,'scisor_K8_same_scorer',path,'shared_scorer_control')
    keys=[(r['protein_id'],r['budget'],r['method']) for r in result]
    if len(keys)!=len(set(keys)):raise ValueError('duplicate output condition')
    for method in ['delepref_unlimited','rag_unlimited','terminal','scisor_native','raygun_native']:
        if {(r['protein_id'],r['budget']) for r in result if r['method']==method}!=expected:raise ValueError('incomplete common method roster')
    if len(result)!=339:raise ValueError('unexpected non-WT output count')
    out=root/BASE;out.mkdir(exist_ok=False)
    write(out/'all_selected_outputs.json',result)
    summary=dict(status='SCREENED_COMMON_OUTPUT_MATRIX_VERIFIED_NOT_STRUCTURE_EVALUATED',WT=44,
        main_methods=5,primary_20pct_outputs=220,stress_7WT_3budget_outputs=105,
        completion_ablation_outputs=7,shared_scorer_control_outputs=7,total_nonWT_outputs=len(result),
        raygun_conditions=65,raygun_raw_candidates_audited=32500,raygun_native_PLL_winners_verified=True,
        raygun_ambiguous_WT_positions=sum(r['correspondence']['ambiguous_positions'] for r in result if r['method']=='raygun_native'),
        source_sha256=source_hashes,output_sha256=sha(out/'all_selected_outputs.json'),
        assembly_contract_sha256=sha(root/'configs/recomb_benign_output_assembly_contract_20261017.json'),
        correspondence_contract='all_optimal_unit_edit_consensus_for_Raygun_explicit_provenance_for_deletion_methods',
        structure_prediction_allowed=False,new_structure_predictions=0,
        next='Finalize scoped roster/subsets, missing mechanisms and both predictor/statistical packages before structure evaluation')
    write(out/'completion.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='source_sha256'}))


if __name__=='__main__':build(Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
