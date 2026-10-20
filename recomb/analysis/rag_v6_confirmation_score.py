"""Freeze confirmation selector outputs; reads no confirmation structure outcomes."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

import rag_v6_features as f
import rag_v6_retrieval as r
import rag_v6_scoring as s

BASE="reports/rag_v6_confirmation"
FREEZE="configs/rag_v6_confirmation_scoring_freeze.json"


def prepare(root):
    pins={p:r.sha(root/p) for p in ['reports/rag_v6/models.json','reports/rag_v6/profiles.json',
          BASE+'/candidates.jsonl','provenance/utility_baseline_sources.json','analysis/prepare_utility_baseline.py',
          'reports/utility_baseline/report.json',
          'configs/protocol_v2.json','configs/rag_v6_scoring_freeze.json','configs/rag_v6_embedding_freeze.json']}
    s.verified(root,pins['configs/rag_v6_scoring_freeze.json'])
    seal=dict(status='CONFIRMATION_SCORING_FROZEN',input_hashes=pins,
              source_hashes={p:r.sha(root/p) for p in ['analysis/rag_v6_confirmation_score.py','slurm/rag_v6_confirmation_score_cpu.sbatch']},
              selectors=['U','U_S','U_S_M','U_S_M_J','Terminal_U'],second_predictor_WT=8,
              second_predictor_selectors=['U_S_M_J','U_S_M','U','Terminal_U'],
              tie_rule='exact_score_then_candidate_id',reference_coordinates_read=False,
              variant_structure_outcomes_read=False,hyperparameter_selection=False,paper_changes=False)
    r.write_json(root/FREEZE,seal,atomic=True)
    return dict(freeze_sha256=r.sha(root/FREEZE))


def execute(root,checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('CPU Slurm required')
    r.verify_hashes(root,{FREEZE:checksum});seal=json.loads((root/FREEZE).read_text())
    for key in ['input_hashes','source_hashes']:r.verify_hashes(root,seal[key])
    model=json.loads((root/'reports/rag_v6/models.json').read_text())
    profiles=json.loads((root/'reports/rag_v6/profiles.json').read_text())
    rows=s.lines(root/BASE/'candidates.jsonl');start=time.time()
    out=root/BASE
    r.write_json(out/'scoring_execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=checksum,started=start))
    indexes=[root/f'caches/embeddings/rag_v6_confirmation_shard{i}_index.jsonl' for i in (0,1)]
    for i in (0,1):
        report=json.loads((root/f'caches/embeddings/rag_v6_confirmation_shard{i}_report.json').read_text())
        if report['failures']:raise ValueError('embedding shard has failures')
    dispatch=json.loads((root/'reports/dispatch_after_pilot.json').read_text())
    code=Path(dispatch['code_snapshot']);r.verify_hashes(code,dispatch['code_hashes'])
    sys.path.insert(0,str(code))
    import torch
    from train import Inputs
    from build_pairs import reference_pairs  # imported by inherited helper, never called below
    import importlib.util
    spec=importlib.util.spec_from_file_location('rag_v6_original_utility',root/'analysis/prepare_utility_baseline.py')
    utility=importlib.util.module_from_spec(spec);spec.loader.exec_module(utility)
    sources=json.loads((root/'provenance/utility_baseline_sources.json').read_text())
    for record in sources.values():r.verify_hashes(root,{record['snapshot']:record['sha256']})
    teacher=utility.load_module('rag_v6_frozen_teacher',sources['teacher_scoring']['snapshot'])
    teacher.mean=statistics.fmean
    selection=utility.load_module('rag_v6_frozen_teacher_weights',sources['proposal_validation']['snapshot'])
    inputs=Inputs(indexes,r.sha(root/'configs/protocol_v2.json'));torch.set_num_threads(4)
    values=[]
    for row in rows:
        pure=f.pure_candidate(row)
        score=utility.utility_scores(pure,inputs.embedding(pure['wt_sequence']),inputs.embedding(pure['sequence']),teacher,selection)
        values.append(score['utility'])
    old_utility=json.loads((root/'reports/utility_baseline/report.json').read_text())
    if sorted(inputs.profiles)!=old_utility['embedding_profile_hashes']:
        raise ValueError('confirmation ESMC profile differs from training utility profile')
    fs=[f.plan_features(row['wt_sequence'],row['mask'],profiles[row['protein_id']]) for row in rows]
    x=np.asarray([a['values'] for a in fs]);e=np.asarray([a['evidence'] for a in fs],dtype=bool)
    valid=np.repeat(e[:,None],len(f.NAMES),axis=1);valid[:,-1]&=np.asarray([a['joint_valid'] for a in fs])
    z=s.standardize(x,valid,np.asarray(model['mean']),np.asarray(model['scale']))
    gate=s.final_gate(e,rows,model['calibration']['allow_mixed_residual'])
    scores=s.predict(model['models'],z,gate,np.asarray(values),model['s_U'])
    selections=[];second=[]
    by_wt=s.groups(rows)
    chosen_WT=sorted({p for p,b in by_wt},key=lambda p:f.digest('20261015:'+p))[:8]
    for (pid,budget),ids in by_wt.items():
        selected={}
        for arm,array in scores.items():
            best=min(ids,key=lambda i:(-array[i],rows[i]['candidate_id']))
            selected[arm]=rows[best]['candidate_id']
        terminal=[i for i in ids if rows[i]['generator']=='terminal']
        selected['Terminal_U']=rows[min(terminal,key=lambda i:(-values[i],rows[i]['candidate_id']))]['candidate_id']
        selections.append(dict(protein_id=pid,budget=budget,selected=selected))
        if pid in chosen_WT:second.append(dict(protein_id=pid,WT_control=True,candidate_ids=sorted({selected[a] for a in seal['second_predictor_selectors']})))
    score_rows=[dict(candidate_id=row['candidate_id'],protein_id=row['protein_id'],budget=row['budget'],
                     raw_evidence=bool(e[i]),gate=float(gate[i]),features=fs[i],
                     scores={a:float(v[i]) for a,v in scores.items()}) for i,row in enumerate(rows)]
    with (out/'scores.jsonl').open('x',encoding='utf-8') as handle:
        for row in score_rows:handle.write(json.dumps(row,sort_keys=True)+'\n')
    r.write_json(out/'selector_outputs.json',selections,atomic=True)
    r.write_json(out/'second_predictor_selection.json',second,atomic=True)
    report=dict(status='CONFIRMATION_SELECTOR_OUTPUTS_FROZEN',seconds=time.time()-start,freeze_sha256=checksum,
                models_sha256=seal['input_hashes']['reports/rag_v6/models.json'],
                input_candidate_sha256=seal['input_hashes'][BASE+'/candidates.jsonl'],
                score_sha256=r.sha(out/'scores.jsonl'),selector_sha256=r.sha(out/'selector_outputs.json'),
                second_selection_sha256=r.sha(out/'second_predictor_selection.json'),
                cache_index_sha256={str(p):r.sha(p) for p in indexes},WT=len(selections),candidates=len(rows),
                supported=int(gate.sum()),joint_valid=int(valid[:,-1].sum()),
                joint_vs_U_changed=sum(a['selected']['U_S_M_J']!=a['selected']['U'] for a in selections),
                joint_vs_marginal_changed=sum(a['selected']['U_S_M_J']!=a['selected']['U_S_M'] for a in selections),
                all_structure_outcomes_unread=True,paper_changes=False,structure_calls=0)
    r.write_json(out/'selector_freeze_report.json',report,atomic=True)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    p.add_argument('--freeze-sha256');a=p.parse_args()
    if a.prepare:print(json.dumps(prepare(a.root)),flush=True)
    else:execute(a.root,a.freeze_sha256)
