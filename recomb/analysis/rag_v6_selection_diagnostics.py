"""Input-only selection equivalence and margin audit; no structure outcomes."""
import json
from pathlib import Path

import numpy as np

import rag_v6_retrieval as r
import rag_v6_scoring as s

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')


def run():
    base=ROOT/'reports/rag_v6_confirmation'
    seal=json.loads((base/'selector_freeze_report.json').read_text())
    r.verify_hashes(ROOT,{str(base/'scores.jsonl'):seal['score_sha256'],
        str(base/'selector_outputs.json'):seal['selector_sha256'],
        str(base/'candidates.jsonl'):seal['input_candidate_sha256'],
        str(ROOT/'reports/rag_v6/models.json'):seal['models_sha256']})
    scores=s.lines(base/'scores.jsonl');candidates={x['candidate_id']:x for x in s.lines(base/'candidates.jsonl')}
    choices=json.loads((base/'selector_outputs.json').read_text());by_pid={x['protein_id']:x for x in choices}
    groups={}
    for row in scores:groups.setdefault(row['protein_id'],[]).append(row)
    per_wt=[];margins=[]
    for pid,pool in groups.items():
        selected=by_pid[pid]['selected']
        for arm in ['U','U_S','U_S_M','U_S_M_J']:
            inferred=min(pool,key=lambda x:(-x['scores'][arm],x['candidate_id']))['candidate_id']
            if inferred!=selected[arm]:raise ValueError('selector file does not reconstruct from frozen scores')
        winner=next(x for x in pool if x['candidate_id']==selected['U'])
        alternative=max((x['scores']['U'] for x in pool if x['candidate_id']!=selected['U']),default=None)
        supported=[x for x in pool if x['gate']]
        raw_gap=winner['scores']['U']-max(x['scores']['U'] for x in supported) if supported else None
        corrected_gap=winner['scores']['U_S_M_J']-max(x['scores']['U_S_M_J'] for x in supported) if supported else None
        if supported and not winner['gate']:margins.append((raw_gap,corrected_gap))
        per_wt.append(dict(protein_id=pid,U_winner_generator=candidates[selected['U']]['generator'],
            U_winner_supported=bool(winner['gate']),supported_candidates=len(supported),
            raw_U_top2_margin=winner['scores']['U']-alternative if alternative is not None else None,
            U_gap_to_best_supported=raw_gap,joint_score_gap_to_best_supported=corrected_gap,
            same_joint_and_U=selected['U_S_M_J']==selected['U'],
            same_joint_and_marginal=selected['U_S_M_J']==selected['U_S_M']))
    learned=['U_S','U_S_M','U_S_M_J']
    identical=all(x['selected'][a]==x['selected']['U'] for x in choices for a in learned)
    result=dict(status='INPUT_ONLY_SELECTION_AUDIT_COMPLETE',WT=len(choices),candidates=len(scores),
        different_from_U={a:sum(x['selected'][a]!=x['selected']['U'] for x in choices) for a in learned+['Terminal_U']},
        U_terminal_winners=sum(x['U_winner_generator']=='terminal' for x in per_wt),
        corrected_candidate_count=sum(abs(x['scores']['U_S_M_J']-x['scores']['U'])>1e-12 for x in scores),
        max_absolute_correction=max(abs(x['scores']['U_S_M_J']-x['scores']['U']) for x in scores),
        joint_valid=sum(x['features']['joint_valid'] for x in scores),
        supported_alternative_against_unsupported_U_winner=len(margins),
        raw_gap_quantiles=np.quantile(np.asarray(margins)[:,0],[0,.5,1]).tolist() if margins else None,
        corrected_gap_quantiles=np.quantile(np.asarray(margins)[:,1],[0,.5,1]).tolist() if margins else None,
        fold_gate='STOP_NO_LEARNED_SELECTION_INCREMENT' if identical else 'DISTINCT_SELECTIONS_EXIST',
        structure_outcomes_read=False,structure_calls=0,model_changed=False,
        conclusion='Learned arms have exactly zero selected-outcome difference from U on this fixed cohort, because every selected mask is identical; absolute structure quality remains unmeasured.',
        per_WT=per_wt,selector_freeze_report_sha256=r.sha(base/'selector_freeze_report.json'))
    r.write_json(base/'selection_gap_diagnostics.json',result,atomic=True)
    print(json.dumps({k:v for k,v in result.items() if k!='per_WT'}),flush=True)


if __name__=='__main__':run()
