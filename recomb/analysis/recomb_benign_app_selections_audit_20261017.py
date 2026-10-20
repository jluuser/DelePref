"""Check reviewed application selections independently of the model runner."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from recomb_scope_screen_20261017 import read,sha,write


def audit(root):
    out=root/'reports/recomb_benign_app_model_20261017'
    completion=read(out/'planning_completion.json')
    for name,digest in completion['outputs'].items():
        if sha(out/name)!=digest:raise ValueError('output identity changed')
    panel={r['protein_id']:r for r in read(out/'panel.json')}
    approved={r['protein_id'] for r in read(root/'reports/recomb_benign_application_scope_20261017/audit.json') if r['status']=='RETAIN'}
    if set(panel)!=approved or len(panel)!=23:raise ValueError('application scope changed')
    profiles=read(root/'retrieval/recomb_benign_app_profiles_20261017/profiles.json')
    pools=defaultdict(list)
    for row in read(out/'candidate_scores.json'):pools[(row['protein_id'],row['method'])].append(row)
    selected=read(out/'all_selections.json');lookup={}
    for row in selected:
        pid=row['protein_id'];p=panel[pid];wt=p['sequence'];pool=pools[(pid,row['method'])]
        if row['wildtype_sequence']!=wt or len(pool)!=8 or len({r['deletion_mask_sha256'] for r in pool})!=8:raise ValueError('WT/K mismatch')
        if min(pool,key=lambda r:r['scorer_rank'])['deletion_mask_sha256']!=row['deletion_mask_sha256']:raise ValueError('wrong selection')
        spans=json.loads(row['deleted_intervals'])
        deleted=set()
        for a,b in spans:
            if not 0<=a<=b<len(wt) or deleted.intersection(range(a,b+1)):raise ValueError('invalid coordinates')
            deleted.update(range(a,b+1))
        if len(deleted)!=len(wt)-p['target_length']:raise ValueError('wrong exact budget')
        edited=''.join(c for i,c in enumerate(wt) if i not in deleted)
        if edited!=row['edited_sequence'] or hashlib.sha256(edited.encode()).hexdigest()!=row['edited_sequence_sha256']:raise ValueError('reconstruction mismatch')
        if row.get('retrieval_source') and row['retrieval_source'] not in {r['target'] for r in profiles[pid]['representatives']}:raise ValueError('unreviewed evidence')
        lookup[(pid,row['method'])]=row
    if len(lookup)!=46:raise ValueError('incomplete or duplicate application selection')
    fallback=0
    for d in read(out/'generation_diagnostics.json'):
        if d['fallback']:
            pid=d['protein_id']
            if lookup[(pid,'rag_unlimited')]['deletion_mask_sha256']!=lookup[(pid,'delepref_unlimited')]['deletion_mask_sha256']:raise ValueError('inexact fallback')
            fallback+=1
    report=dict(status='PASS_REVIEWED_APPLICATION_SELECTION_AUDIT',WT=23,selected_outputs=46,
        all_K8=True,all_exact_budgets=True,all_reconstructions_match=True,all_retrieval_sources_reviewed=True,
        exact_fallback_conditions=fallback,new_structure_predictions=0,planning_completion_sha256=sha(out/'planning_completion.json'))
    write(out/'semantic_audit.json',report);print(json.dumps(report))


if __name__=='__main__':audit(Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
