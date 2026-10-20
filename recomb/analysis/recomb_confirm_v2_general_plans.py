"""Original trained DelePref, reviewed RAG, gap and terminal on real General100 caches."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_General100_plans')
PROFILE=Path('retrieval/recomb_confirm_v2_General100/profiles.json')
FZ=Path('configs/recomb_confirm_v2_General100_plans_freeze.json')
METHODS=('delepref_unlimited','rag_unlimited','gap_frequency','terminal')


def prepare(root):
    if (root/FZ).exists() or (root/PROFILE).exists():raise FileExistsError('General100 reviewed plans already released')
    from recomb_benign_profiles_20261017 import filtered_profile
    panel=read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')
    review=root/'reports/recomb_confirm_v2_General100_evidence_review'
    inventory=read(review/'inventory.json');done=read(review/'completion.json')
    if done['errors']:raise ValueError('Unresolved metadata request failures')
    raw=root/'retrieval/recomb_confirm_v2_General100/profiles_unreviewed_evidence.json'
    if sha(raw)!=inventory['profiles_sha256'] or sha(review/'target_metadata_audit.json')!=done['audit_sha256']:
        raise ValueError('Evidence profile/audit identity changed')
    profiles=read(raw);audits=read(review/'target_metadata_audit.json');selected={}
    if len(panel)!=100 or set(profiles)!={r['protein_id'] for r in panel}:raise ValueError('Reviewed query roster differs')
    for r in panel:
        p=filtered_profile(profiles[r['protein_id']],dict(sequence_sha256=r['WT_sha256']),audits)
        p['evidence_status']='SOURCE_FUNCTION_AND_FULL_SEQUENCE_IDENTITY_REVIEWED'
        p['scope_filter']='confirm_v2_reference_source_groups_and_function_review'
        selected[r['protein_id']]=p
    write(root/PROFILE,selected)
    write(root/BASE/'profile_completion.json',dict(WT=100,representative_rows=sum(len(p['representatives']) for p in selected.values()),
        queries_with_evidence=sum(bool(p['representatives']) for p in selected.values()),
        original_order_projection_and_quality_rules_unchanged=True,no_backfill=True,
        profile_sha256=sha(root/PROFILE),new_model_calls=0))
    names=['analysis/recomb_confirm_v2_general_plans.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_benign_profiles_20261017.py','analysis/recomb_gap_frequency_control_20261017.py',
        'analysis/recomb_general100_mechanisms_20261017.py','analysis/delepref_v8_main.py',
        'analysis/delepref_v8_runtime.py','analysis/delepref_v8_rag_unlimited.py','analysis/delepref_v8_controls.py',
        'analysis/delepref_unlimited_planning.py','analysis/delepref_unlimited_scorer.py',
        'configs/delepref_v8_baseline_freeze.json','configs/recomb_confirm_v2_General100_encoder_freeze.json',
        'reports/recomb_confirm_v2_General100_model/encoding_completion.json',
        'reports/recomb_confirm_v2_general_native_v2/General100_panel.json',
        'reports/recomb_confirm_v2_general_native/encoder_inputs.json',
        'reports/recomb_confirm_v2_General100_subsets/subsets.json',
        'reports/recomb_confirm_v2_General100_evidence_review/policy.json',
        'reports/recomb_confirm_v2_General100_evidence_review/target_metadata_audit.json',
        'reports/recomb_confirm_v2_General100_evidence_review/completion.json',
        str(PROFILE),'slurm/recomb_confirm_v2_general_plans.sbatch']
    write(root/FZ,dict(status='QUALIFIED_GENERAL100_HEADS_AND_REVIEWED_EVIDENCE_FROZEN',files={p:sha(root/p) for p in names},
        WT=100,methods=METHODS,K=8,qmax=None,main_budget=.2,budget_subset_N=40,other_budgets=[.1,.3,.4],
        seed=13,CPU_stability_seed=29,original_checkpoint_step=4000,main_RAG='unchanged4+4',
        gap='same frozen reliable per-position gap frequency, quarter-budget prefixes and original Policy completion',
        terminal='fixed eight N/C splits; same original Scorer',heads_allowed=True,
        new_encoder_forwards=0,new_training=False,structure_prediction_allowed=False))
    print(sha(root/FZ),flush=True)


def semantic_audit(root,scores,selections,diagnostics,profiles):
    import delepref_unlimited_planning as planning
    groups=defaultdict(list)
    for r in scores:groups[(r['protein_id'],r['budget'],r['generation_seed'],r['method'])].append(r)
    if len(groups)!=1280 or len(selections)!=1280 or len(diagnostics)!=320:raise ValueError('Incomplete frozen matrix')
    for r in selections:
        seq=r['wildtype_sequence'];g=groups[(r['protein_id'],r['budget'],r['generation_seed'],r['method'])]
        if len(g)!=8 or len({c['deletion_mask_sha256'] for c in g})!=8 or sorted(c['scorer_rank'] for c in g)!=list(range(1,9)):
            raise ValueError('Not unique ranked K8')
        if r['deletion_mask_sha256']!=min(g,key=lambda c:c['scorer_rank'])['deletion_mask_sha256']:raise ValueError('Wrong winner')
        for c in g:
            mask=planning.validate_plan(len(seq),c['budget'],json.loads(c['deleted_intervals']))
            if int(mask.sum())!=planning.delete_count(len(seq),c['budget']):raise ValueError('Budget differs')
            if c.get('retrieval_source') and c['generation_kind'].startswith('retrieval'):
                bytarget={h['target']:set(h['events']) for h in planning.natural_events(profiles[c['protein_id']],len(seq))}
                if c['retrieval_source'] not in bytarget:raise ValueError('Unreviewed evidence source used')
                for a,b in c.get('retrieval_seed_spans') or []:
                    if (a,b) not in bytarget[c['retrieval_source']] or not mask[a:b+1].all():raise ValueError('Natural seed not preserved')
                    if a and mask[a-1] or b<len(seq)-1 and mask[b+1]:raise ValueError('Retrieval anchor removed')
        mask=planning.validate_plan(len(seq),r['budget'],json.loads(r['deleted_intervals']))
        edited=''.join(c for c,d in zip(seq,mask) if not d)
        import hashlib
        if edited!=r['edited_sequence'] or hashlib.sha256(edited.encode()).hexdigest()!=r['edited_sequence_sha256']:
            raise ValueError('Edited sequence reconstruction differs')
    for d in diagnostics:
        if d['fallback']:
            k=(d['protein_id'],d['budget'],d['seed'])
            a=groups[k+('delepref_unlimited',)];b=groups[k+('rag_unlimited',)]
            if [(r['deletion_mask_sha256'],r['scorer_score'],r['scorer_rank']) for r in a]!=[(r['deletion_mask_sha256'],r['scorer_score'],r['scorer_rank']) for r in b]:
                raise ValueError('No-evidence fallback not exact')
    return dict(status='ALL_GENERAL100_K8_BUDGET_SOURCE_ANCHOR_RANK_RECONSTRUCTION_AND_FALLBACK_PASS',
        conditions=320,method_pools=1280,candidates=len(scores),new_structure_calls=0)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen plan source/input differs: '+p)
    baseline=read(root/'configs/delepref_v8_baseline_freeze.json')
    for p,h in {**baseline['inputs'],**baseline['code']}.items():
        if sha(root/p)!=h:raise ValueError('Original immutable model dependency changed')
    out=root/BASE
    if (out/'execution_claim.json').exists():raise FileExistsError('Plans already claimed; inspect existing job')
    write(out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID']))
    import torch
    import delepref_v8_runtime as rt
    import delepref_v8_main as parent
    import delepref_unlimited_planning as planning
    import delepref_unlimited_scorer as scoring
    from delepref_v8_controls import terminal_proposals
    from recomb_gap_frequency_control_20261017 import proposals as gap_proposals
    from recomb_general100_mechanisms_20261017 import select_rows
    cachepath=root/'reports/recomb_confirm_v2_General100_model/wt_cache.pt'
    encoded=read(root/'reports/recomb_confirm_v2_General100_model/encoding_completion.json')
    if sha(cachepath)!=encoded['cache_sha256']:raise ValueError('Real WT cache changed')
    cache=torch.load(cachepath,map_location='cpu',weights_only=True)
    if not cache['real_encoder'] or cache['checkpoint_sha256']!=rt.CHECKPOINT_SHA or cache['unique_WT']!=100:
        raise ValueError('Not the real original100WT cache')
    torch.set_num_threads(4);start=time.monotonic();runtime,api=rt.load(root,device='cpu',cached_heads_only=True)
    records=read(root/'reports/recomb_confirm_v2_general_native/encoder_inputs.json')
    chunk=parent.cached_chunk(api,cache,records);profiles=read(root/PROFILE)
    subs=read(root/'reports/recomb_confirm_v2_General100_subsets/subsets.json')
    stress={r['protein_id'] for r in subs['budget40']};scores=[];selections=[];diagnostics=[]
    with torch.inference_mode():
        for record in records:
            pid=record['protein_id'];seq=record['wildtype_sequence'];conditions=[(.2,13),(.2,29)]
            if pid in stress:conditions += [(b,13) for b in [.1,.3,.4]]
            for beta,seed in conditions:
                s,w,d=parent.score_condition(runtime,api,chunk,dict(record,budget=beta),profiles[pid],seed)
                original=sorted([c for c in s if c['method']=='delepref_unlimited'],key=lambda r:r['raw_candidate_rank'])
                native=[dict(spans=tuple(map(tuple,json.loads(c['deleted_intervals']))),kind=c['generation_kind'],row=c) for c in original]
                emb=chunk.encoded.embeddings[chunk.encoded.key_for(pid,seq)]
                logits=runtime.model.propose_from_embeddings([emb],[beta])[0].residue_logits.detach().cpu().double().numpy()
                gaps,gdiag=gap_proposals(logits,beta,profiles[pid]);mixed,gfallback=planning.mix_candidates(native,gaps)
                gap=([dict(c) for c in original] if gfallback else scoring.score_candidates(runtime,api,chunk,
                    parent.make_rows(api,original[0],mixed,len(seq),logits)))
                terminal=scoring.score_candidates(runtime,api,chunk,parent.make_rows(api,original[0],terminal_proposals(len(seq),beta),len(seq),logits))
                for method,pool in [('gap_frequency',gap),('terminal',terminal)]:
                    s.extend(dict(c,method=method) for c in pool);w.append(select_rows(api,seq,pool,method))
                for c in w:c['evaluation_role']='CPU_stability_only' if seed==29 else ('primary20pct' if beta==.2 else 'budget_curve')
                d.update(gap_fallback=gfallback,gap_guidance=gdiag)
                scores.extend(s);selections.extend(w);diagnostics.append(d)
            print('GENERAL100_REAL_HEADS_DONE '+pid,flush=True)
    audit=semantic_audit(root,scores,selections,diagnostics,profiles)
    write(out/'candidate_scores.json',scores);write(out/'all_selections.json',selections)
    write(out/'generation_diagnostics.json',diagnostics);write(out/'semantic_audit.json',audit)
    report=dict(status='GENERAL100_ORIGINAL_RAG_GAP_TERMINAL_FROZEN_OUTPUTS_COMPLETE_NOT_QUALITY_EVALUATED',
        WT=100,conditions=320,method_pools=1280,candidates=len(scores),selected_outputs=len(selections),
        main_20pct_outputs=400,budget_curve_outputs=480,CPU_stability_outputs=400,
        primary_RAG_changed_WT=sum(d['original_selected_mask']!=d['rag_selected_mask'] for d in diagnostics if d['budget']==.2 and d['seed']==13),
        primary_exact_RAG_fallback=sum(d['fallback'] for d in diagnostics if d['budget']==.2 and d['seed']==13),
        seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],freeze_sha256=sha(root/FZ),
        new_encoder_forwards=0,new_training=False,new_structure_predictions=0,structure_prediction_allowed=False,
        outputs={n:sha(out/n) for n in ['candidate_scores.json','all_selections.json','generation_diagnostics.json','semantic_audit.json']})
    write(out/'completion.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root) if a.prepare else run(a.root)
