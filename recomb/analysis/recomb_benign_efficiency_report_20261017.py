"""Measured computation ledger; native search costs and cached phases stay explicit."""
import argparse
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from recomb_scope_screen_20261017 import read,sha,write


def describe(values):
    if not values:return dict(N=0)
    x=np.asarray(values,dtype=float)
    return dict(N=len(values),sum=float(x.sum()),median=float(np.median(x)),mean=float(x.mean()),
        minimum=float(x.min()),maximum=float(x.max()))


def build(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU Slurm required')
    sys.path.insert(0,str(root/'code'))
    from prediction_journal import PredictionJournal
    out=root/'reports/recomb_benign_efficiency_20261017'
    if out.exists():raise FileExistsError('efficiency report exists')
    evaluation=root/'reports/recomb_benign_structure_evaluation_20261017/completion.json'
    # The full fixed evaluator may require a versioned technical repair. The
    # caller can supply its documented completion without altering predictions.
    integrity=root/'reports/recomb_benign_prediction_integrity_20261017/completion.json'
    if not integrity.exists():raise ValueError('full prediction integrity check must complete first')
    cat=root/'reports/recomb_benign_prediction_catalog_20261017';struct=root/'reports/recomb_benign_structures_20261017'
    physical={r['task_id']:r for r in read(cat/'physical_requests.json')};reuse=read(cat/'compatible_cache.json')
    journal=PredictionJournal(cat/'statistical_protocol.json',struct/'prediction_journal.jsonl').snapshot()
    call_times=defaultdict(list);peak=defaultdict(list);states=Counter()
    for r in journal['attempts']:
        if r['state'] not in ('succeeded','failed'):raise ValueError('unfinished prediction accounting')
        p=physical[r['task_id']];states[r['state']]+=1
        if r['seconds'] is not None:call_times[p['profile_name']].append(r['seconds'])
        if r['max_memory'] is not None:peak[p['profile_name']].append(r['max_memory'])
    core_path=root/'reports/recomb_benign_final_outputs_20261017/core_selected_outputs.json'
    core=read(core_path);raygun={}
    for r in core:
        if r['method']!='raygun_native':continue
        p=root/r['source_path'];row=read(p)
        if row['protein_id']!=r['protein_id'] or row['edited_sequence_sha256']!=r['edited_sequence_sha256']:raise ValueError('Raygun timing/source identity differs')
        raygun[(r['protein_id'],r['budget'])]=row
    if len(raygun)!=104:raise ValueError('Raygun timing scope differs')
    scisor=[]
    for folder in ['recomb_benign_baselines_20261017','recomb_benign_baseline_extension_20261017']:
        for r in read(root/'reports'/folder/'scisor/all_conditions.json'):
            if r['mode']=='native':scisor.append(r)
    if len(scisor)!=60:raise ValueError('SCISOR new native stress scope differs')
    matched_raygun=[raygun[(r['protein_id'],r['budget'])] for r in scisor]
    wall=subprocess.run(['/opt/gridview/slurm/bin/sacct','-j','1117977','--format=JobID,ElapsedRaw,State','-P'],capture_output=True,text=True,check=True).stdout
    allocations=[]
    for line in wall.splitlines()[1:]:
        parts=line.split('|')
        if len(parts)>=3 and parts[0].startswith('1117977_') and '.' not in parts[0] and '[' not in parts[0]:
            allocations.append(dict(job_id=parts[0],elapsed_seconds=int(parts[1]),state=parts[2],allocated_GPUs=1))
    if len(allocations)!=8:raise ValueError('not all eight array allocations present')
    cpu=[]
    for job,purpose in [('1117865','reviewed original/RAG heads batch'),('1117886','reviewed application original/RAG heads batch'),
                        ('1117927','13WT missing-budget/direct-only heads'),('1117949','final mechanism+SCISOR common-Scorer batch')]:
        text=subprocess.run(['/opt/gridview/slurm/bin/sacct','-j',job,'--format=JobID,ElapsedRaw,State','-P'],capture_output=True,text=True,check=True).stdout
        row=next(line.split('|') for line in text.splitlines()[1:] if line.startswith(job+'|'))
        cpu.append(dict(job_id=job,purpose=purpose,allocated_cpus=4,elapsed_seconds=int(row[1]),state=row[2],
            includes_head_load_and_mixed_methods=True,not_per_method_inference_latency=True))
    report=dict(status='MEASURED_FULL_SCREENED_EXPERIMENT_COMPUTATION_ACCOUNTED',new_training=False,
        original_Policy_Scorer='fixed4000-step checkpoint',main_WT=44,budget_mechanism_WT=20,binding_cases=12,enzyme_cases=11,
        native_inference_contract=dict(DelePref='K8 proposal + original Scorer; cached single WT representation',
            RAG='sameK8 and Scorer; up to4 original +4 retrieved/completed, exact fallbackK8',
            SCISOR='nativeK1; separate20-caseK8+sameScorer control',Raygun='500 raw native candidates;50 PLL-ranked;top1'),
        Raygun_all104=dict(raw_candidates=52000,generation_seconds=describe([r['generation_seconds'] for r in raygun.values()]),
            PLL_seconds=describe([r['pll_seconds'] for r in raygun.values()]),reported_total_seconds=describe([r['seconds'] for r in raygun.values()])),
        matched60_stress_conditions=dict(SCISOR_native_K1_seconds=describe([r['seconds'] for r in scisor]),
            Raygun_native500_50_top1_seconds=describe([r['seconds'] for r in matched_raygun]),
            not_equal_K_or_compute=True),
        SCISOR_generation_counts=dict(all_native_conditions=104,K8_control_pools=20,K8_candidates=160,
            reused_primary_native_samples=44,new_scope_sampler_calls=200,new_control_only_sampler_calls=140),
        CPU_heads_batches=cpu,per_method_original_vs_RAG_head_latency_not_separately_instrumented=True,
        original_WT_encoder_cache='general44 reuses pre-existing100WT batch; do not fabricate a separate44WT encoding benchmark',
        application_WT_encoder_completion=read(root/'reports/recomb_benign_app_model_20261017/encoding_completion.json'),
        cold_retrieval_and_cache='general/app profiles already frozen and reused; search/metadata preparation are separate setup stages, not warm head latency',
        structure=dict(logical_requests=867,unique_identities=651,reused_verified_cache=len(reuse),planned_missing=419,
            actual_new_calls=journal['consumed'],states=dict(states),historical_calls=758,
            total_actual_historical_and_current_calls=758+journal['consumed'],retry_calls=sum(r['attempt']>0 for r in journal['attempts']),
            timed_call_wall_seconds_by_profile={k:describe(v) for k,v in call_times.items()},
            peak_GPU_allocated_memory_bytes_by_profile={k:describe(v) for k,v in peak.items()},
            Slurm_allocations=allocations,Slurm_reserved_GPU_hours=sum(r['elapsed_seconds'] for r in allocations)/3600,
            timed_calls_are_not_GPU_FLOPs_or_kernel_time=True,structure_limit=None),
        input_sha256=dict(core_selected_outputs=sha(core_path),prediction_journal=sha(struct/'prediction_journal.jsonl'),
            prediction_integrity=sha(integrity)),source_sha256=sha(Path(__file__)),job_id=os.environ['SLURM_JOB_ID'])
    if evaluation.exists():report['input_sha256']['full_metrics_completion']=sha(evaluation)
    out.mkdir();write(out/'computation_accounting.json',report)
    s=report['structure'];m=report['matched60_stress_conditions']
    lines=['# 计算量与复用记录','',
        '各方法沿用原生搜索协议，不能把K8、K1和Raygun500→50→1称为等计算量。20例SCISOR-K8共同Scorer控制单列。下面的秒数为实际记录，不补造原模型和RAG分别的冷/热延迟。',
        '', '| 项目 | 实测记录 |','|---|---:|',
        f"| 结构逻辑记录 / 去重身份 | 867 / 651 |",f"| 已验证缓存 / 预定缺失 | {len(reuse)} / 419 |",
        f"| 当前实际新调用 / 技术重试 | {s['actual_new_calls']} / {s['retry_calls']} |",
        f"| 当前成功 / 失败 | {states['succeeded']} / {states['failed']} |",
        f"| 八分片Slurm预留GPU小时（每任务1GPU） | {s['Slurm_reserved_GPU_hours']:.3f} |",
        f"| 同60个额外预算条件SCISOR K1中位秒数 | {m['SCISOR_native_K1_seconds']['median']:.3f} |",
        f"| 同60条件Raygun原生500/50/top1中位秒数 | {m['Raygun_native500_50_top1_seconds']['median']:.3f} |",'',
        'Slurm GPU小时包含资产核验、模型加载、输入/输出和空隙，不等于GPU内核计算时长；逐预测seconds也包括调用内部开销。原/RAG混合CPU批处理含头部加载与多种方法，没有分别计时，不能从该记录宣称RAG加速倍数。通用44的WT缓存来自历史100WT批次，未强行按44分摊旧批次耗时。',
        '', '应用23WT的独立编码完成记录、四个CPU heads批次、各预测profile调用时间和GPU显存峰值均在JSON中保留。完整模型权重、输入和已完成结构复用；没有新增训练，也没有为凑数量增加预测。论文未修改。']
    (out/'computation_accounting_zh.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(out/'completion.json',dict(status=report['status'],outputs={n:sha(out/n) for n in ['computation_accounting.json','computation_accounting_zh.md']},new_model_calls=0,new_structure_predictions=0))
    print(json.dumps(dict(status=report['status'],actual_new_structure_calls=journal['consumed'],Slurm_GPU_hours=s['Slurm_reserved_GPU_hours'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    build(p.parse_args().root)
