"""Restore planned subset sizes within the already reviewed cohort, using input hashes only."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from recomb_scope_screen_20261017 import read,sha,write


def finalize(root):
    source=root/'reports/recomb_benign_scope_20261017/general_panel.json'
    accepted=read(root/'configs/recomb_benign_baselines_freeze_20261017.json')
    if sha(source)!=accepted['files'][str(source.relative_to(root))]:raise ValueError('reviewed cohort changed')
    panel=read(source);groups=defaultdict(list)
    for p in panel:
        key=hashlib.sha256(('recomb_v8_eval_subsets_20261016\0'+p['sequence_sha256']).encode()).hexdigest()
        groups[p['stratum']].append((key,p))
    ordered={s:[p for _,p in sorted(rows,key=lambda x:(x[0],x[1]['protein_id']))] for s,rows in groups.items()}
    if min(map(len,ordered.values()))<10:raise ValueError('insufficient reviewed source strata')
    large=[p for s in sorted(ordered) for p in ordered[s][:10]]
    second=[p['protein_id'] for s in sorted(ordered) for p in ordered[s][:4]]
    scarcity=[]
    for source_name in sorted(ordered):
        eligible=[p for p in large if p['stratum']==source_name]
        eligible.sort(key=lambda p:hashlib.sha256(('recomb-scarcity-20261017|'+p['sequence_sha256']).encode()).hexdigest())
        scarcity.extend(p['protein_id'] for p in eligible[:5])
    old=read(root/'reports/recomb_benign_scope_20261017/retained_subsets.json')
    new=[p['protein_id'] for p in large]
    if not set(old['completion_ablation']).issubset(new) or not set(old['second_predictor']).issubset(second):raise ValueError('original retained subsets not nested')
    out=root/'reports/recomb_benign_final_subsets_20261017';out.mkdir(exist_ok=False)
    subsets=dict(budget_and_mechanism=new,completion_ablation=new,second_predictor=second,scarcity=scarcity)
    write(out/'subsets.json',subsets)
    added=[p for p in large if p['protein_id'] not in old['completion_ablation']]
    templates={p['protein_id']:p for p in read(root/'reports/recomb_benign_scope_20261017/raygun_eligible_tasks.json') if p['budget']==.2}
    raygun=[];scisor=[]
    for p in added:
        for beta in [.1,.3,.4]:
            t=dict(templates[p['protein_id']],budget=beta,target_length=len(p['sequence'])-round(len(p['sequence'])*beta))
            raygun.append(t);scisor.append(dict(t,candidate_count=1,mode='native',reuse_original_native_as_candidate0=False))
        scisor.append(dict(templates[p['protein_id']],candidate_count=8,mode='same_scorer_K8',reuse_original_native_as_candidate0=True))
    write(out/'raygun_delta_tasks.json',raygun);write(out/'scisor_delta_tasks.json',scisor)
    report=dict(status='INPUT_ONLY_SUBSET_REPLENISHMENT_FROZEN',main_WT=44,budget_and_mechanism_WT=20,second_predictor_WT=8,scarcity_WT=10,
        source_balance_20=dict(cath_domain=10,pdb_chain=10),source_balance_8=dict(cath_domain=4,pdb_chain=4),
        prior_retained_budget_WT=7,added_budget_WT=len(added),additional_Raygun_conditions=len(raygun),
        additional_SCISOR_conditions=len(scisor),additional_SCISOR_sampler_calls=sum(t['candidate_count']-(t['mode']=='same_scorer_K8') for t in scisor),
        new_target_proteins=0,new_models=0,new_hyperparameters=0,new_structure_results_read=False,
        selection_rule='unchanged original salted sequence-hash order within approved source strata; restore planned counts, not select by support or outcomes',
        input_panel_sha256=sha(source),subsets_sha256=sha(out/'subsets.json'),
        files={p.name:sha(p) for p in out.glob('*.json')},structure_prediction_allowed=False)
    write(out/'completion.json',report);print(json.dumps(report))


if __name__=='__main__':finalize(Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
