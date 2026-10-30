"""Enzyme80 heads, including both predeclared 10% and 30% secondary budgets."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from recomb_confirm_v2_diagnostics import read, sha, write

METHODS=('delepref_unlimited','rag_unlimited','gap_frequency','terminal')
SPECS={'Binder60':(60,20),'Enzyme80':(80,30)}


def paths(cohort):
    return Path('reports/recomb_confirm_v2_'+cohort+'_plans'),Path('configs/recomb_confirm_v2_'+cohort+'_plans_freeze.json')


def prepare(root,cohort):
    out,fz=paths(cohort);n,ns=SPECS[cohort]
    if (root/fz).exists():raise FileExistsError('Application heads already frozen; inspect existing state')
    cp='reports/recomb_confirm_v2_'+cohort+'_constructs/'
    subs='reports/recomb_confirm_v2_'+cohort+'_subsets/subsets.json'
    panel=read(root/(cp+cohort+'_panel.json'))
    if len(panel)!=n or any(not r['WT_encoding_allowed'] for r in panel):raise ValueError('Ordinary application protein roster not released')
    sub=read(root/subs)['budget10pct_and_three_structure_seeds']
    if len(sub)!=ns or not {r['protein_id'] for r in sub}<={r['protein_id'] for r in panel}:
        raise ValueError('Fixed application budget/repetition subset differs')
    pf='configs/recomb_confirm_v2_'+cohort+'_profiles_freeze.json'
    profile='retrieval/recomb_confirm_v2_'+cohort+'/profiles.json'
    if sha(root/profile)!=read(root/pf)['profiles_sha256']:raise ValueError('Reviewed homolog profile differs')
    names=['analysis/recomb_confirm_v2_enzyme_plans.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/recomb_gap_frequency_control_20261017.py','analysis/recomb_general100_mechanisms_20261017.py',
        'analysis/delepref_v8_main.py','analysis/delepref_v8_runtime.py','analysis/delepref_v8_rag_unlimited.py',
        'analysis/delepref_v8_controls.py','analysis/delepref_unlimited_planning.py','analysis/delepref_unlimited_scorer.py',
        'configs/delepref_v8_baseline_freeze.json','configs/recomb_confirm_v2_'+cohort+'_encoder_freeze.json',
        cp+cohort+'_panel.json',cp+'completion.json',cp+'encoder_inputs.json',subs,pf,profile,
        'slurm/recomb_confirm_v2_enzyme_plans.sbatch']
    write(root/fz,dict(status='APPLICATION_ORIGINAL_HEADS_AND_REVIEWED_RAG_FROZEN',cohort=cohort,
        files={p:sha(root/p) for p in names},WT=n,methods=METHODS,K=8,qmax=None,
        main_budget=.2,secondary_budgets=[.1,.3],secondary_budget_N=ns,main_seed=13,CPU_stability_seed=29,
        expected_conditions=2*n+2*ns,expected_selected_outputs=4*(2*n+2*ns),
        RAG='unchanged4+4',same_original_scorer_for_all_four_methods=True,
        no_functional_site_or_interface_protection=True,new_encoder_forwards=0,new_training=False,
        generation_stage_release=True,real_WT_encoder_completion_required_at_runtime=True,
        chemical_predictor_context_and_folding_require_separate_release=True,structure_prediction_allowed=False))
    (root/out).mkdir(parents=True,exist_ok=True);print(sha(root/fz),flush=True)


def audit(scores,selections,diagnostics,profiles,expected_conditions):
    import delepref_unlimited_planning as planning
    groups=defaultdict(list)
    for r in scores:groups[(r['protein_id'],r['budget'],r['generation_seed'],r['method'])].append(r)
    if len(groups)!=4*expected_conditions or len(selections)!=len(groups) or len(diagnostics)!=expected_conditions:
        raise ValueError('Application matrix incomplete')
    for s in selections:
        seq=s['wildtype_sequence'];key=(s['protein_id'],s['budget'],s['generation_seed'],s['method']);pool=groups[key]
        if len(pool)!=8 or len({p['deletion_mask_sha256'] for p in pool})!=8 or sorted(p['scorer_rank'] for p in pool)!=list(range(1,9)):
            raise ValueError('Application K8 unique masks/ranks differ')
        if min(pool,key=lambda p:p['scorer_rank'])['deletion_mask_sha256']!=s['deletion_mask_sha256']:
            raise ValueError('Application selected sequence is not the original Scorer winner')
        for p in pool:
            mask=planning.validate_plan(len(seq),p['budget'],json.loads(p['deleted_intervals']))
            if int(mask.sum())!=planning.delete_count(len(seq),p['budget']):raise ValueError('Application exact deletion count differs')
            if p.get('retrieval_source') and p['generation_kind'].startswith('retrieval'):
                bytarget={e['target']:set(e['events']) for e in planning.natural_events(profiles[p['protein_id']],len(seq))}
                if p['retrieval_source'] not in bytarget:raise ValueError('Unreviewed application donor used')
                for a,b in p.get('retrieval_seed_spans') or []:
                    if (a,b) not in bytarget[p['retrieval_source']] or not mask[a:b+1].all():raise ValueError('Natural seed projection changed')
                    if (a and mask[a-1]) or (b<len(seq)-1 and mask[b+1]):raise ValueError('Natural seed anchor removed')
        mask=planning.validate_plan(len(seq),s['budget'],json.loads(s['deleted_intervals']))
        edited=''.join(aa for aa,deleted in zip(seq,mask) if not deleted)
        if edited!=s['edited_sequence'] or hashlib.sha256(edited.encode()).hexdigest()!=s['edited_sequence_sha256']:
            raise ValueError('Application edit reconstruction differs')
    for d in diagnostics:
        if not d['fallback']:continue
        key=(d['protein_id'],d['budget'],d['seed'])
        a=groups[key+('delepref_unlimited',)];b=groups[key+('rag_unlimited',)]
        if [(p['deletion_mask_sha256'],p['scorer_score'],p['scorer_rank']) for p in a]!=[(p['deletion_mask_sha256'],p['scorer_score'],p['scorer_rank']) for p in b]:
            raise ValueError('Application fallback differs from original model')
    return dict(status='APPLICATION_K8_BUDGET_RANK_RECONSTRUCTION_SOURCE_ANCHOR_AND_EXACT_FALLBACK_PASS',
        conditions=expected_conditions,selected_outputs=len(selections),candidates=len(scores),new_model_encodings=0)


def run(root,cohort):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    out,fz=paths(cohort);f=read(root/fz);n,ns=SPECS[cohort]
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Frozen application heads source/input changed: '+p)
    baseline=read(root/'configs/delepref_v8_baseline_freeze.json')
    for p,h in {**baseline['inputs'],**baseline['code']}.items():
        if sha(root/p)!=h:raise ValueError('Original model dependency changed')
    if (root/out/'execution_claim.json').exists():raise FileExistsError('Application heads claimed; inspect existing job')
    enc=root/('reports/recomb_confirm_v2_'+cohort+'_model');encoded=read(enc/'encoding_completion.json')
    encoder_freeze='configs/recomb_confirm_v2_'+cohort+'_encoder_freeze.json'
    if encoded['freeze_sha256']!=f['files'][encoder_freeze] or sha(enc/'wt_cache.pt')!=encoded['cache_sha256']:
        raise ValueError('Application real WT cache/encoder contract differs')
    write(root/out/'execution_claim.json',dict(job_id=os.environ['SLURM_JOB_ID'],
        encoder_completion_sha256=sha(enc/'encoding_completion.json'),WT_cache_sha256=encoded['cache_sha256']))
    import torch
    import delepref_v8_runtime as rt
    import delepref_v8_main as parent
    import delepref_unlimited_planning as planning
    import delepref_unlimited_scorer as scoring
    from delepref_v8_controls import terminal_proposals
    from recomb_gap_frequency_control_20261017 import proposals as gap_proposals
    from recomb_general100_mechanisms_20261017 import select_rows
    cache=torch.load(enc/'wt_cache.pt',map_location='cpu',weights_only=True)
    if not cache['real_encoder'] or cache['unique_WT']!=n or cache['checkpoint_sha256']!=rt.CHECKPOINT_SHA:
        raise ValueError('Not original real application WT embeddings')
    torch.set_num_threads(4);start=time.monotonic();runtime,api=rt.load(root,device='cpu',cached_heads_only=True)
    records=read(root/('reports/recomb_confirm_v2_'+cohort+'_constructs/encoder_inputs.json'))
    chunk=parent.cached_chunk(api,cache,records);profiles=read(root/('retrieval/recomb_confirm_v2_'+cohort+'/profiles.json'))
    subs=read(root/('reports/recomb_confirm_v2_'+cohort+'_subsets/subsets.json'))
    secondary={r['protein_id'] for r in subs['budget10pct_and_three_structure_seeds']}
    scores=[];selected=[];diagnostics=[]
    with torch.inference_mode():
        for record in records:
            pid=record['protein_id'];seq=record['wildtype_sequence'];conditions=[(.2,13),(.2,29)]
            if pid in secondary:conditions += [(b,13) for b in (.1,.3)]
            for beta,seed in conditions:
                s,w,d=parent.score_condition(runtime,api,chunk,dict(record,budget=beta),profiles[pid],seed)
                original=sorted([p for p in s if p['method']=='delepref_unlimited'],key=lambda p:p['raw_candidate_rank'])
                native=[dict(spans=tuple(map(tuple,json.loads(p['deleted_intervals']))),kind=p['generation_kind'],row=p) for p in original]
                emb=chunk.encoded.embeddings[chunk.encoded.key_for(pid,seq)]
                logits=runtime.model.propose_from_embeddings([emb],[beta])[0].residue_logits.detach().cpu().double().numpy()
                gaps,gdiag=gap_proposals(logits,beta,profiles[pid]);mixed,fallback=planning.mix_candidates(native,gaps)
                gap=([dict(p) for p in original] if fallback else scoring.score_candidates(runtime,api,chunk,
                    parent.make_rows(api,original[0],mixed,len(seq),logits)))
                terminal=scoring.score_candidates(runtime,api,chunk,
                    parent.make_rows(api,original[0],terminal_proposals(len(seq),beta),len(seq),logits))
                for method,pool in [('gap_frequency',gap),('terminal',terminal)]:
                    s.extend(dict(p,method=method) for p in pool);w.append(select_rows(api,seq,pool,method))
                for p in w:
                    p['evaluation_role']='CPU_stability_only' if seed==29 else ('primary20pct' if beta==.2 else 'secondary_budget')
                    p['cohort']=cohort
                d.update(gap_fallback=fallback,gap_guidance=gdiag)
                scores.extend(s);selected.extend(w);diagnostics.append(d)
            print('APPLICATION_REAL_HEADS_DONE '+cohort+' '+pid,flush=True)
    result=audit(scores,selected,diagnostics,profiles,f['expected_conditions'])
    write(root/out/'candidate_scores.json',scores);write(root/out/'all_selections.json',selected)
    write(root/out/'generation_diagnostics.json',diagnostics);write(root/out/'semantic_audit.json',result)
    done=dict(status='APPLICATION_ORIGINAL_RAG_GAP_TERMINAL_OUTPUTS_COMPLETE_NOT_QUALITY_EVALUATED',
        cohort=cohort,WT=n,conditions=len(diagnostics),selected_outputs=len(selected),candidates=len(scores),
        primary_RAG_changed_WT=sum(d['original_selected_mask']!=d['rag_selected_mask'] for d in diagnostics if d['budget']==.2 and d['seed']==13),
        primary_exact_RAG_fallback=sum(d['fallback'] for d in diagnostics if d['budget']==.2 and d['seed']==13),
        new_encoder_forwards=0,new_training=False,new_structure_predictions=0,structure_prediction_allowed=False,
        freeze_sha256=sha(root/fz),job_id=os.environ['SLURM_JOB_ID'],seconds=time.monotonic()-start,
        outputs={p:sha(root/out/p) for p in ['candidate_scores.json','all_selections.json','generation_diagnostics.json','semantic_audit.json']})
    write(root/out/'completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cohort',choices=['Enzyme80'],required=True);p.add_argument('--prepare',action='store_true')
    a=p.parse_args();prepare(a.root,a.cohort) if a.prepare else run(a.root,a.cohort)
