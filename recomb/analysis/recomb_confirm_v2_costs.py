"""E8 costs on the fixed20WT; separate real native timings and cached CPU branch replay."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from recomb_confirm_v2_diagnostics import read,sha,write

BASE=Path('reports/recomb_confirm_v2_General100_costs')
FZ=Path('configs/recomb_confirm_v2_General100_costs_freeze.json')
SUBSET=Path('reports/recomb_confirm_v2_General100_subsets/subsets.json')


def prepare(root):
    if (root/FZ).exists():raise FileExistsError('E8 observed/cached cost protocol already frozen')
    cases=read(root/SUBSET)['cost_and_SCISOR_K8_20'];panel={r['protein_id']:r for r in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    if len(cases)!=20 or len({r['protein_id'] for r in cases})!=20:raise ValueError('Fixed E8 cost20 roster differs')
    for r in cases:
        if r['WT_sha256']!=panel[r['protein_id']]['WT_sha256']:raise ValueError('Cost/ordinary WT identity mismatch')
    write(root/BASE/'cases.json',cases)
    names=['analysis/recomb_confirm_v2_costs.py','analysis/recomb_confirm_v2_diagnostics.py',
        'analysis/delepref_v8_runtime.py','analysis/delepref_v8_main.py','analysis/delepref_v8_rag_unlimited.py',
        'analysis/delepref_unlimited_planning.py','analysis/delepref_unlimited_scorer.py','analysis/delepref_v8_controls.py',
        'analysis/recomb_gap_frequency_control_20261017.py','analysis/recomb_general100_mechanisms_20261017.py',
        'slurm/recomb_confirm_v2_costs.sbatch',str(SUBSET),str(BASE/'cases.json'),
        'reports/recomb_confirm_v2_general_native_v2/General100_panel.json',
        'reports/recomb_confirm_v2_General100_model/encoding_completion.json',
        'reports/recomb_confirm_v2_General100_plans/completion.json','reports/recomb_confirm_v2_General100_plans/all_selections.json',
        'reports/recomb_confirm_v2_General100_SCISOR_L_K8_score/completion.json',
        'reports/recomb_confirm_v2_General100_SCISOR_L_K8_score/candidate_scores.json',
        'reports/recomb_confirm_v2_General100_SCISOR_L_K8_score/all_selections.json',
        'reports/recomb_confirm_v2_General100_scisor_L/completion.json','reports/recomb_confirm_v2_General100_scisor_L/all_conditions.json',
        'reports/recomb_confirm_v2_General100_Raygun_native_audit/completion.json',
        'reports/recomb_confirm_v2_General100_Raygun_native_audit/per_condition_audit.json',
        'reports/recomb_confirm_v2_General100_ESM_supplement/completion.json',
        'reports/recomb_confirm_v2_General100_ESM_supplement/per_output_metrics.json',
        'retrieval/recomb_confirm_v2_General100/profiles.json','retrieval/recomb_confirm_v2_General100/completion.json',
        'configs/recomb_confirm_v2_General100_plans_freeze.json','configs/delepref_v8_baseline_freeze.json']
    # Timings are descriptive resource measurements. They never choose new candidate outputs.
    write(root/FZ,dict(status='E8_FIXED20_NATIVE_OBSERVED_AND_CACHED_CPU_COST_PROTOCOL',files={n:sha(root/n) for n in names},
        WT=20,budget=.2,generation_seed=13,cached_CPU_trials=3,CPU_threads=4,
        one_input_hash_first_case_warmup=True,all_candidate_outputs_must_match_frozen_selections=True,
        no_structural_result_or_selection_changes=True,
        baseline_shared_work='one Policy logit call + original K8 planning/scoring; RAG/gap/terminal branch increments recorded separately',
        process_setup_separate=True,PLM_encode_and_uncached_retrieval_not_in_cached_CPU_latency=True,
        existing_native_GPU_hardware_reported_separately=True,no_cross_hardware_speedup_claim=True,
        missing_true_uncached_WT_and_cold_process_latency='not inferred from batched100WT total; a later same-device E8 GPU measurement is required',
        new_structure_predictions=0,new_training=False))
    print(sha(root/FZ),flush=True)


def verify(root):
    f=read(root/FZ)
    for p,h in f['files'].items():
        if sha(root/p)!=h:raise ValueError('Fixed E8 source/input changed: '+p)
    return f


def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(N=len(a),mean=float(a.mean()),median=float(np.median(a)),p25=float(np.quantile(a,.25)),p75=float(np.quantile(a,.75)))


def observed(root):
    verify(root);out=root/BASE
    if (out/'observed_completion.json').exists():raise FileExistsError('Observed E8 cost extraction already completed')
    ids={r['protein_id'] for r in read(out/'cases.json')};results=[];quality=[]
    audit=read(root/'reports/recomb_confirm_v2_General100_Raygun_native_audit/per_condition_audit.json')
    for row in audit:
        if row['protein_id'] in ids and row['budget']==.2:
            results.append(dict(protein_id=row['protein_id'],method='raygun_native',hardware=row['actual_GPU'],
                generation_seconds=row['generation_seconds'],PLL_seconds=row['PLL_seconds'],
                resident_generation_and_selection_seconds=row['generation_seconds']+row['PLL_seconds'],
                generated_candidates=500,PLL_candidates=50,source='original native500/top50 audited records',
                cold_model_load_included=False,structure_evaluation_included=False))
    native=read(root/'reports/recomb_confirm_v2_General100_scisor_L/all_conditions.json')
    for row in native:
        if row['protein_id'] not in ids or row['budget']!=.2:continue
        if row['mode']=='native':
            results.append(dict(protein_id=row['protein_id'],method='scisor_L_native',hardware=row['actual_GPU'],
                resident_generation_and_selection_seconds=row['seconds'],sampler_calls=1,
                source='actual official L sampler with mapped-plan CPU synchronization',cold_model_load_included=False,structure_evaluation_included=False))
        else:
            if len(row['candidates'])!=8:raise ValueError('Existing L K8 cost pool is incomplete')
            results.append(dict(protein_id=row['protein_id'],method='SCISOR_U90_L_K8_same_scorer',hardware=row['actual_GPU'],
                standalone_K8_sampling_seconds=sum(c['seconds'] for c in row['candidates']),
                actual_incremental_7new_sampling_seconds=sum(c['seconds'] for c in row['candidates'] if not c['reused_native']),
                sampler_calls_standalone=8,new_calls_in_record=7,reused_native_first_candidate=True,
                Scorer_and_WT_encoding_not_in_sampling_seconds=True,cold_model_load_included=False,structure_evaluation_included=False))
    if len(results)!=60:raise ValueError('Observed native cost20 three-row roster incomplete')
    qs=read(root/'reports/recomb_confirm_v2_General100_ESM_supplement/per_output_metrics.json')
    for row in qs:
        if row['protein_id'] in ids and (row['evaluation_role']=='primary20pct' or row['evaluation_role']=='SCISOR_K8_control'):
            quality.append(dict(protein_id=row['protein_id'],method=row['method'],budget=row['budget'],
                **{k:row['common_structure'].get(k) for k in ['structure_mapped_CA_lddt','structure_mapped_native_contact_recovery','full_WT_TM']}))
    if len(quality)!=140:raise ValueError('Fixed cost20 seven-method quality comparison incomplete')
    summaries=[]
    for method in sorted({r['method'] for r in results}):
        group=[r for r in results if r['method']==method]
        key='standalone_K8_sampling_seconds' if method=='SCISOR_U90_L_K8_same_scorer' else 'resident_generation_and_selection_seconds'
        summaries.append(dict(method=method,measured_component=key,seconds=stats([r[key] for r in group]),
            hardware_counts=dict(__import__('collections').Counter(r['hardware'] for r in group)),not_a_same_hardware_end_to_end_comparison=True))
    setup=dict(original100WT_model_load_encode_and_cache_write_total=read(root/'reports/recomb_confirm_v2_General100_model/encoding_completion.json'),
        original320condition_four_method_CPU_total=read(root/'reports/recomb_confirm_v2_General100_plans/completion.json')['seconds'],
        original100WT_natural_search_and_projection_total=read(root/'retrieval/recomb_confirm_v2_General100/completion.json'),
        these_batch_totals_are_not_per_WT_cold_or_hot_latency=True,
        external_database_construction_and_source_metadata_review_not_in_per_query_timing=True)
    for name,value in [('observed_native_per_WT.json',results),('observed_native_summary.json',summaries),('fixed20_quality.json',quality),('existing_batch_setup_totals.json',setup)]:write(out/name,value)
    done=dict(status='E8_FIXED20_OBSERVED_NATIVE_COSTS_AND_FIXED_OUTPUT_QUALITY_EXTRACTED',WT=20,
        native_timing_records=60,quality_records=140,new_model_calls=0,new_structure_predictions=0,
        no_claim_of_equal_compute_or_equal_device=True,
        outputs={p:sha(out/p) for p in ['observed_native_per_WT.json','observed_native_summary.json','fixed20_quality.json','existing_batch_setup_totals.json']})
    write(out/'observed_completion.json',done);print(done,flush=True)


def cached_trial(runtime,api,chunk,record,profile):
    import torch
    import delepref_v8_main as parent
    import delepref_unlimited_planning as planning
    import delepref_unlimited_scorer as scoring
    from delepref_v8_controls import condition_seed,terminal_proposals,sequence_sha
    from recomb_gap_frequency_control_20261017 import proposals as gap_proposals
    from recomb_general100_mechanisms_20261017 import select_rows
    pid=record['protein_id'];seq=record['wildtype_sequence'];beta=.2
    template=dict(record,budget=beta,sequence_length=len(seq),sequence_sha256=api.sequence_sha256(seq),
        generation_seed=13,condition_seed=condition_seed(13,pid,seq,beta),student_version=runtime.checkpoint['student_version'],
        source_checkpoint_sha256=runtime.source_checkpoint_sha256,selected_checkpoint_step=4000)
    before=time.perf_counter();emb=chunk.encoded.embeddings[chunk.encoded.key_for(pid,seq)]
    logits=runtime.model.propose_from_embeddings([emb],[beta])[0].residue_logits.detach().cpu().double().numpy()
    policy_logit_seconds=time.perf_counter()-before
    start=time.perf_counter();cfg=runtime.config['decoder']
    items,cost=planning.policy_proposals(logits,beta,template['condition_seed'],perturbation_scale=float(cfg.get('perturbation_scale',1.)),max_attempts=8*int(cfg.get('max_attempts_multiplier',16)))
    original=scoring.score_candidates(runtime,api,chunk,parent.make_rows(api,template,items,len(seq),logits,cost))
    original_plan_and_score_seconds=time.perf_counter()-start
    native=[dict(item,row=row) for item,row in zip(items,original)];pools={'delepref_unlimited':original};increments={};score_calls=1
    start=time.perf_counter();retrieved,_=planning.retrieved_proposals(logits,beta,profile,template['condition_seed'])
    mixed,fallback=planning.mix_candidates(native,retrieved)
    pools['rag_unlimited']=([dict(r) for r in original] if fallback else scoring.score_candidates(runtime,api,chunk,parent.make_rows(api,template,mixed,len(seq),logits,cost)))
    increments['rag_unlimited']=time.perf_counter()-start;score_calls+=int(not fallback)
    start=time.perf_counter();gaps,_=gap_proposals(logits,beta,profile);mixed,gfallback=planning.mix_candidates(native,gaps)
    pools['gap_frequency']=([dict(r) for r in original] if gfallback else scoring.score_candidates(runtime,api,chunk,parent.make_rows(api,template,mixed,len(seq),logits)))
    increments['gap_frequency']=time.perf_counter()-start;score_calls+=int(not gfallback)
    start=time.perf_counter();pools['terminal']=scoring.score_candidates(runtime,api,chunk,parent.make_rows(api,template,terminal_proposals(len(seq),beta),len(seq),logits))
    increments['terminal']=time.perf_counter()-start;score_calls+=1
    selected={}
    for method,pool in pools.items():
        selected[method]=select_rows(api,seq,pool,method)
    return dict(protein_id=pid,policy_logits_seconds=policy_logit_seconds,
        original_K8_plan_and_score_seconds=original_plan_and_score_seconds,branch_increment_seconds=increments,
        CPU_policy_invocations=1,CPU_scorer_invocations=score_calls,RAG_fallback=fallback,gap_fallback=gfallback,
        selected_output_SHA={m:r['edited_sequence_sha256'] for m,r in selected.items()},
        selected_mask_SHA={m:r['deletion_mask_sha256'] for m,r in selected.items()})


def cached(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm only')
    f=verify(root);out=root/BASE
    if (out/'cached_completion.json').exists():raise FileExistsError('Cached CPU E8 benchmark already complete')
    import torch
    import delepref_v8_runtime as rt
    from delepref_v8_main import cached_chunk
    import delepref_unlimited_scorer as scoring
    cases=read(out/'cases.json');ids={r['protein_id'] for r in cases}
    panel={r['protein_id']:r for r in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    records=[dict(protein_id=c['protein_id'],wildtype_sequence=panel[c['protein_id']]['sequence'],budget=.2) for c in cases]
    frozen=read(root/'reports/recomb_confirm_v2_General100_plans/all_selections.json')
    expected={(r['protein_id'],r['method']):r for r in frozen if r['protein_id'] in ids and r['generation_seed']==13 and r['budget']==.2}
    torch.set_num_threads(4);start=time.perf_counter()
    path=root/'reports/recomb_confirm_v2_General100_model/wt_cache.pt'
    if sha(path)!=read(root/'reports/recomb_confirm_v2_General100_model/encoding_completion.json')['cache_sha256']:raise ValueError('Real original WT cache changed')
    cache=torch.load(path,map_location='cpu',weights_only=True)
    if not cache['real_encoder'] or cache['checkpoint_sha256']!=rt.CHECKPOINT_SHA:raise ValueError('Not formal original-model embeddings')
    cache_load_seconds=time.perf_counter()-start;start=time.perf_counter()
    runtime,api=rt.load(root,device='cpu',cached_heads_only=True);model_setup_seconds=time.perf_counter()-start
    chunk=cached_chunk(api,cache,records);profiles=read(root/'retrieval/recomb_confirm_v2_General100/profiles.json')
    warmup=None;trials=[];K8timings=[]
    K8rows=read(root/'reports/recomb_confirm_v2_General100_SCISOR_L_K8_score/candidate_scores.json');groups=defaultdict(list)
    for r in K8rows:groups[r['protein_id']].append(r)
    frozenK8={r['protein_id']:r for r in read(root/'reports/recomb_confirm_v2_General100_SCISOR_L_K8_score/all_selections.json')}
    with torch.inference_mode():
        warmup=cached_trial(runtime,api,chunk,records[0],profiles[records[0]['protein_id']])
        for trial in range(f['cached_CPU_trials']):
            for record in records[trial:]+records[:trial]:
                pid=record['protein_id'];r=cached_trial(runtime,api,chunk,record,profiles[pid]);r['trial']=trial
                for method in r['selected_output_SHA']:
                    saved=expected[pid,method]
                    if saved['edited_sequence_sha256']!=r['selected_output_SHA'][method] or saved['deletion_mask_sha256']!=r['selected_mask_SHA'][method]:
                        raise ValueError('Cost replay differs from frozen scientific output, no replacement allowed')
                trials.append(r)
                before=time.perf_counter();scored=scoring.score_candidates(runtime,api,chunk,groups[pid]);seconds=time.perf_counter()-before
                winner=min(scored,key=lambda c:c['scorer_rank'])
                if len(scored)!=8 or winner['deletion_mask_sha256']!=frozenK8[pid]['deletion_mask_sha256']:raise ValueError('Existing L-K8 Scorer replay differs')
                K8timings.append(dict(protein_id=pid,trial=trial,CPU_Scorer_seconds=seconds,scored_existing_candidates=8,additional_encoder_calls=0))
    write(out/'cached_CPU_trials.json',trials);write(out/'L_K8_existing_pool_CPU_Scorer_trials.json',K8timings)
    done=dict(status='FIXED20_CACHED_CPU_BRANCH_TIMINGS_MATCH_FROZEN_OUTPUTS',WT=20,trials_per_WT=3,measured_conditions=len(trials),
        first_case_warmup=warmup,cache_load_seconds=cache_load_seconds,heads_model_setup_seconds=model_setup_seconds,
        CPU_hostname=platform.node(),CPU_processor=platform.processor(),CPU_threads=4,
        CPU_policy_invocations_measured=sum(r['CPU_policy_invocations'] for r in trials),
        CPU_scorer_invocations_measured=sum(r['CPU_scorer_invocations'] for r in trials)+len(K8timings),
        warmup_extra_policy_and_scorer_invocations=1+warmup['CPU_scorer_invocations'],
        new_PLM_encoder_calls=0,new_structure_predictions=0,new_training=False,
        not_complete_cold_process_or_uncached_WT_latency=True,all20_three_trial_four_method_outputs_match=True,
        outputs={n:sha(out/n) for n in ['cached_CPU_trials.json','L_K8_existing_pool_CPU_Scorer_trials.json']},job_id=os.environ['SLURM_JOB_ID'])
    write(out/'cached_completion.json',done);print(done,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--stage',choices=['observed','cached'])
    a=p.parse_args()
    if a.prepare:prepare(a.root)
    elif a.stage=='observed':observed(a.root)
    elif a.stage=='cached':cached(a.root)
    else:p.error('prepare or stage required')
