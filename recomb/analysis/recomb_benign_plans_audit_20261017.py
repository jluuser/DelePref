"""Independent scope, reconstruction, baseline identity and fallback checks."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from recomb_scope_screen_20261017 import read, sha, write


def audit(root):
    out=root/'reports/recomb_benign_plans_20261017'
    completion=read(out/'planning_completion.json')
    for name,digest in completion['outputs'].items():
        if sha(out/name)!=digest:raise ValueError('output SHA mismatch')
    allowed={r['protein_id']:r for r in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    profiles=read(root/'retrieval/recomb_benign_general_20261017/profiles.json')
    pools=defaultdict(list)
    def key(row):return row['protein_id'],float(row['budget']),int(row['generation_seed']),row['method']
    for row in read(out/'candidate_scores.json'):
        if row['protein_id'] not in allowed:raise ValueError('unreviewed candidate')
        pools[key(row)].append(row)
    old={}
    for path in ['reports/delepref_v8_main/all_selections.json','reports/recomb_task_budget_20261017/all_selections.json']:
        for row in read(root/path):
            if row['protein_id'] in allowed and row['method']=='delepref_unlimited':old[key(row)]=row
    selections=read(out/'all_selections.json')
    selected={key(row):row for row in selections}
    original_matches=0
    for row in selections:
        pid=row['protein_id'];wt=allowed[pid]['sequence'];pool=pools[key(row)]
        if len(pool)!=8 or len({p['deletion_mask_sha256'] for p in pool})!=8:raise ValueError('K mismatch')
        winner=min(pool,key=lambda p:int(p['scorer_rank']))
        if winner['deletion_mask_sha256']!=row['deletion_mask_sha256']:raise ValueError('winner mismatch')
        intervals=json.loads(row['deleted_intervals']) if isinstance(row['deleted_intervals'],str) else row['deleted_intervals']
        deleted=set()
        for a,b in intervals:
            if not 0<=a<=b<len(wt) or deleted.intersection(range(a,b+1)):raise ValueError('invalid coordinates')
            deleted.update(range(a,b+1))
        if len(deleted)!=max(1,min(len(wt)-1,round(len(wt)*float(row['budget'])))):raise ValueError('budget mismatch')
        edited=''.join(c for i,c in enumerate(wt) if i not in deleted)
        if edited!=row['edited_sequence'] or hashlib.sha256(edited.encode()).hexdigest()!=row['edited_sequence_sha256']:
            raise ValueError('reconstruction mismatch')
        source=row.get('retrieval_source')
        if source and source not in {r['target'] for r in profiles[pid]['representatives']}:raise ValueError('unreviewed evidence source')
        if row['method']=='delepref_unlimited':
            if old[key(row)]['deletion_mask_sha256']!=row['deletion_mask_sha256']:raise ValueError('original baseline changed')
            original_matches+=1
    fallbacks=0
    for d in read(out/'generation_diagnostics.json'):
        if d['fallback']:
            k=(d['protein_id'],float(d['budget']),int(d['seed']))
            a=selected[k+('delepref_unlimited',)];b=selected[k+('rag_unlimited',)]
            if a['edited_sequence_sha256']!=b['edited_sequence_sha256'] or a['deletion_mask_sha256']!=b['deletion_mask_sha256']:
                raise ValueError('fallback not exact')
            fallbacks+=1
    report=dict(status='PASS_INDEPENDENT_SCREENED_SELECTION_AUDIT',WT=len(allowed),
        selections=len(selections),original_baseline_exact_matches=original_matches,
        exact_fallback_conditions=fallbacks,all_K8=True,all_exact_budgets=True,
        all_sources_reviewed=True,original_outputs_changed=False,new_structure_predictions=0,
        planning_completion_sha256=sha(out/'planning_completion.json'))
    write(out/'semantic_audit.json',report)
    print(json.dumps(report))


if __name__=='__main__':audit(Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
