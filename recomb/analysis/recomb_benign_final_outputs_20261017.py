"""Audit only new native pools, reuse verified rows, and complete the common matrix."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

from recomb_scope_screen_20261017 import read, sha, write
from recomb_edit_correspondence_20261017 import consensus

BASE = Path('reports/recomb_benign_final_outputs_20261017')


def explicit_correspondence(wt, edited, spans):
    spans = json.loads(spans) if isinstance(spans,str) else spans
    deleted = set()
    for a,b in spans:
        if not 0 <= a <= b < len(wt) or deleted.intersection(range(a,b+1)):
            raise ValueError('invalid explicit coordinates')
        deleted.update(range(a,b+1))
    if ''.join(c for i,c in enumerate(wt) if i not in deleted) != edited:
        raise ValueError('explicit sequence reconstruction mismatch')
    mp = []; j = 0
    for i in range(len(wt)):
        mp.append(None if i in deleted else j)
        if i not in deleted:
            j += 1
    return dict(method='explicit_native_deletion_plan',wt_to_edit=mp,
        position_states=['deleted' if i in deleted else 'stable' for i in range(len(wt))],
        stable_positions=len(edited),certain_deleted_positions=len(deleted),ambiguous_positions=0)


def build(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU Slurm required')
    out = root/BASE
    if out.exists():
        raise FileExistsError('final assembly already exists; inspect before repeating')
    sources = {}
    def load(path):
        sources[path] = sha(root/path)
        return read(root/path)
    panel_path = 'reports/recomb_benign_scope_20261017/general_panel.json'
    allowed = {r['protein_id']: r for r in load(panel_path)}
    if sha(root/panel_path) != read(root/'configs/recomb_benign_baselines_freeze_20261017.json')['files'][panel_path]:
        raise ValueError('reviewed cohort changed')
    sub = load('reports/recomb_benign_final_subsets_20261017/subsets.json')
    stress = set(sub['completion_ablation'])
    if len(allowed) != 44 or len(stress) != 20 or not stress.issubset(allowed):
        raise ValueError('final cohort/subset mismatch')
    old_base = 'reports/recomb_benign_outputs_20261017/'
    old_done = load(old_base+'completion.json')
    if sha(root/(old_base+'all_selected_outputs.json')) != old_done['output_sha256']:
        raise ValueError('prior 339-output audit changed')
    for p,h in old_done['source_sha256'].items():
        if sha(root/p) != h:
            raise ValueError('previously verified source changed')
    core = load(old_base+'all_selected_outputs.json')
    if len(core) != 339:
        raise ValueError('wrong prior core size')
    controls_base = 'reports/recomb_benign_final_controls_20261017/'
    controls_done = load(controls_base+'completion.json')
    for n,h in controls_done['outputs'].items():
        if sha(root/(controls_base+n)) != h:
            raise ValueError('final control output changed')
    mechanism_rows = load(controls_base+'all_selections.json')
    pools = defaultdict(list)
    for row in load(controls_base+'candidate_scores.json'):
        pools[(row['protein_id'],row['method'])].append(row)
    original_pools = defaultdict(list)
    for row in read(root/'reports/recomb_benign_plans_20261017/candidate_scores.json'):
        if row['protein_id'] in stress and row['budget'] == .2 and row['generation_seed'] == 13:
            original_pools[(row['protein_id'],row['method'])].append(row)
    previous = {(r['protein_id'],r['method']):r for r in read(root/'reports/recomb_benign_mechanisms_20261017/all_selections.json')}
    reused = 0
    for row in mechanism_rows:
        k = row['protein_id'],row['method']
        if k in previous:
            if row != previous[k]:
                raise ValueError('reused mechanism selection changed')
            reused += 1
        method = row['method']
        pool = (original_pools[(row['protein_id'],'rag_unlimited')] if method == 'rag_no_scorer' else
                original_pools[(row['protein_id'],'delepref_unlimited')] if method == 'rag_evidence_zero' else pools[k])
        if len(pool) != 8 or len({r['deletion_mask_sha256'] for r in pool}) != 8:
            raise ValueError('mechanism pool is not unique K8')
        if method == 'rag_no_scorer':
            winner = min(pool,key=lambda r:(-r['policy_energy'],tuple(map(tuple,json.loads(r['deleted_intervals'])))))
        else:
            winner = min(pool,key=lambda r:r['scorer_rank'])
        if winner['deletion_mask_sha256'] != row['deletion_mask_sha256']:
            raise ValueError('mechanism selector mismatch')
    if reused != 25:
        raise ValueError('incorrect mechanism reuse count')
    mechanisms = []
    def add(row,method,path,destination,role='comparison'):
        pid = row['protein_id']
        if pid not in allowed:
            raise ValueError('unreviewed output query')
        wt = allowed[pid]['sequence']; edited = row['edited_sequence']; beta = float(row['budget'])
        if row.get('wildtype_sequence',row.get('sequence',wt)) != wt:
            raise ValueError('WT changed')
        if len(edited) != len(wt)-max(1,min(len(wt)-1,round(len(wt)*beta))):
            raise ValueError('deletion budget mismatch')
        digest = hashlib.sha256(edited.encode()).hexdigest()
        if digest != row['edited_sequence_sha256']:
            raise ValueError('edited identity mismatch')
        mapping = consensus(wt,edited) if method == 'raygun_native' else explicit_correspondence(wt,edited,row['deleted_intervals'])
        sources.setdefault(path,sha(root/path))
        destination.append(dict(protein_id=pid,method=method,budget=beta,evaluation_role=role,
            wildtype_sequence=wt,WT_sha256=allowed[pid]['sequence_sha256'],edited_sequence=edited,
            edited_sequence_sha256=digest,correspondence=mapping,source_path=path,
            selection_metadata={k:row.get(k) for k in ['generation_seed','candidate_index','scorer_score','scorer_rank',
                'raw_candidate_count','PLL_ranked_count','retrieval_source','retrieval_seed_spans','generation_kind',
                'selection','selector','evidence_is_counterfactual','evidence_is_independently_derived',
                'num_insertions','num_deletions','num_substitutions']}))
    extension = 'reports/recomb_benign_baseline_extension_20261017/'
    tasks = load(extension+'raygun_tasks.json')
    if len(tasks) != 39:
        raise ValueError('wrong Raygun delta size')
    for shard,n in [(0,20),(1,19)]:
        if load(extension+'raygun/completion_%d.json'%shard)['conditions'] != n:
            raise ValueError('Raygun delta shard incomplete')
    for index,task in enumerate(tasks):
        path = extension+'raygun/condition_%03d.json'%index; row = load(path)
        if row['protein_id'] not in allowed or row['sequence'] != allowed[row['protein_id']]['sequence']:
            raise ValueError('unreviewed Raygun condition')
        if any(task[k] != row[k] for k in ['protein_id','sequence','budget','target_length']):
            raise ValueError('Raygun delta task changed')
        raw = extension+'raygun/raw_%03d.jsonl.gz'%index
        if sha(root/raw) != row['raw_sha256']:
            raise ValueError('Raygun raw pool SHA mismatch')
        sources[raw] = row['raw_sha256']
        with gzip.open(root/raw,'rt') as f:
            pool = [json.loads(line) for line in f]
        if len(pool) != 500 or len(row['native_top50']) != 50 or row['raw_candidate_count'] != 500 or row['PLL_ranked_count'] != 50:
            raise ValueError('Raygun native 500/50 count mismatch')
        for i,c in enumerate(pool):
            if c['generation_index'] != i or len(c['sequence']) != task['target_length'] or set(c['sequence'])-set('ACDEFGHIKLMNPQRSTVWYX') or not math.isfinite(c['adjusted_PLL']):
                raise ValueError('invalid Raygun raw candidate')
        order = sorted(range(500),key=lambda j:(-pool[j]['adjusted_PLL'],j))[:50]
        for rank,(j,c) in enumerate(zip(order,row['native_top50']),1):
            if c['generation_index'] != j or c['sequence'] != pool[j]['sequence'] or c['pll_rank'] != rank or c['adjusted_PLL'] != pool[j]['adjusted_PLL'] or c['sequence_sha256'] != hashlib.sha256(c['sequence'].encode()).hexdigest():
                raise ValueError('Raygun native PLL ranking mismatch')
        if row['edited_sequence'] != pool[order[0]]['sequence']:
            raise ValueError('Raygun native winner mismatch')
        add(row,'raygun_native',path,core)
    path = 'reports/recomb_benign_heads_extension_20261017/all_selections.json'
    hd = load('reports/recomb_benign_heads_extension_20261017/completion.json')
    if sha(root/path) != hd['outputs']['all_selections.json']:
        raise ValueError('heads delta changed')
    for row in load(path):
        add(row,row['method'],path,core,'completion_ablation' if row['method']=='rag_direct_only' else 'comparison')
    added = set(t['protein_id'] for t in tasks)
    path = 'reports/recomb_task_budget_20261017/all_selections.json'
    for row in read(root/path):
        if row['protein_id'] in added and row['budget'] in [.1,.3,.4] and row['generation_seed'] == 13 and row['method'] == 'terminal':
            add(row,'terminal',path,core)
    path = extension+'scisor/all_conditions.json'
    for row in load(path):
        if row['mode'] == 'native':
            add(row,'scisor_native',path,core)
    path = controls_base+'scisor_delta_selections.json'
    scisor_scores = defaultdict(list)
    for row in load(controls_base+'scisor_delta_candidate_scores.json'):
        scisor_scores[row['protein_id']].append(row)
    for row in load(path):
        pool = scisor_scores[row['protein_id']]
        if len(pool) != 8 or len({r['deletion_mask_sha256'] for r in pool}) != 8 or min(pool,key=lambda r:r['scorer_rank'])['deletion_mask_sha256'] != row['deletion_mask_sha256']:
            raise ValueError('SCISOR delta common-Scorer selection mismatch')
        add(row,'scisor_K8_same_scorer',path,core,'shared_scorer_control')
    for row in mechanism_rows:
        add(row,row['method'],controls_base+'all_selections.json',mechanisms,'mechanism_or_scarcity')
    expected = {(pid,.2) for pid in allowed} | {(pid,b) for pid in stress for b in [.1,.3,.4]}
    keys = [(r['protein_id'],r['budget'],r['method']) for r in core]
    if len(keys) != 560 or len(set(keys)) != 560:
        raise ValueError('final core duplicate or incomplete')
    for m in ['delepref_unlimited','rag_unlimited','terminal','scisor_native','raygun_native']:
        if {(r['protein_id'],r['budget']) for r in core if r['method']==m} != expected:
            raise ValueError('method does not cover the identical complete cohort')
    for m in ['rag_direct_only','scisor_K8_same_scorer']:
        if {(r['protein_id'],r['budget']) for r in core if r['method']==m} != {(p,.2) for p in stress}:
            raise ValueError('fixed20 control roster mismatch')
    wanted = {(p,m) for p in stress for m in ['rag_marginal_gap','rag_cooccurrence_null','rag_no_scorer','rag_position_gap_frequency']}
    wanted |= {(p,m) for p in sub['scarcity'] for m in ['rag_evidence_min8','rag_evidence_zero']}
    if len(mechanisms) != 100 or {(r['protein_id'],r['method']) for r in mechanisms} != wanted:
        raise ValueError('mechanism matrix incomplete')
    out.mkdir()
    write(out/'core_selected_outputs.json',core); write(out/'mechanism_selected_outputs.json',mechanisms)
    sources['analysis/recomb_benign_final_outputs_20261017.py'] = sha(Path(__file__))
    sources['analysis/recomb_edit_correspondence_20261017.py'] = sha(root/'analysis/recomb_edit_correspondence_20261017.py')
    report = dict(status='FINAL_REVIEWED_COMMON_AND_MECHANISM_OUTPUTS_VERIFIED',main_WT=44,budget_and_mechanism_WT=20,
        core_nonWT_outputs=560,primary_outputs=220,additional_budget_outputs=300,direct_only_outputs=20,SCISOR_K8_outputs=20,
        mechanism_nonWT_outputs=100,scarcity_WT=10,reused_core_outputs=339,reused_mechanism_selections=25,
        additional_Raygun_conditions_audited=39,additional_Raygun_raw_candidates_audited=19500,
        full_Raygun_conditions=104,full_Raygun_raw_candidates_audited=52000,
        Raygun_all_native_PLL_winners_verified=True,
        Raygun_ambiguous_WT_positions=sum(r['correspondence']['ambiguous_positions'] for r in core if r['method']=='raygun_native'),
        effective_randomized_event_WT=controls_done['effective_null_WT'],effective_min8_WT=controls_done['effective_min8_WT'],
        source_sha256=sources,outputs={n:sha(out/n) for n in ['core_selected_outputs.json','mechanism_selected_outputs.json']},
        new_structure_predictions=0,structure_prediction_allowed=False,job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json',report); print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    build(p.parse_args().root)
