"""Readable, component-specific E8 report and versioned candidate-count correction."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent.parent;base=root/'reports/recomb_confirm_v2_General100_costs'
def read(n):return json.loads((base/n).read_text(encoding='utf-8'))
for completion in ['observed_completion.json','cached_completion.json']:
    for n,h in read(completion)['outputs'].items():
        if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h:raise ValueError('Measured cost output changed: '+n)
scope=read('raygun_scoring_scope.json')
if not scope['all_actual_code_matches_frozen']:raise ValueError('Raygun source identity changed')
corrected=read('observed_native_per_WT.json')
for r in corrected:
    if r['method']=='raygun_native':
        r.pop('PLL_candidates',None)
        r.update(ESM_candidate_sequences_scored=500,saved_top_ranked_sequences=50,
            score_definition='upstream-named PLL, unmasked ESM token log-score with frozen adjustments',
            generation_scores_and_selected_sequences_unchanged=True)
(base/'observed_native_per_WT_v2.json').write_text(json.dumps(corrected,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
trials=read('cached_CPU_trials.json');groups=defaultdict(list)
for r in trials:groups[r['protein_id']].append(r)
if len(groups)!=20 or any(len(rs)!=3 for rs in groups.values()):raise ValueError('Repeated CPU timing cohort differs')
def ms_cell(values):
    x=np.asarray(values,float)*1000
    return f'{x.mean():.3f} | {np.median(x):.3f} | {np.quantile(x,.25):.3f}–{np.quantile(x,.75):.3f}'
lines=['# General100 固定20例的效率实验阶段结果','',
    '已完成现有原生生成计时的核验和真实WT表示缓存命中后的CPU组件测量。20例各测3次，四种DelePref分支的最终序列与删除mask均与冻结实验完全一致；没有重新选择结构结果。完整同设备冷启动和未缓存WT端到端耗时仍待GPU测量，因此暂不给总体加速比。','',
    '## 缓存命中后的CPU组件','',
    '使用原4000步Policy/Scorer及真实ESMC WT缓存，4个CPU线程。先对每个WT的三次计时求均值，再汇总20个WT；第一例额外预热不进入下表。共享原候选池先生成和评分，RAG、gap及Terminal分支增量另列，不把增量当完整方法耗时。所有项均不含PLM编码、检索缓存缺失、模型加载和最终序列输出格式化。','',
    '| 测量组件 | WT数 | 均值 ms | 中位数 ms | 四分位范围 ms |','|---|---:|---:|---:|---|']
for field,title in [('policy_logits_seconds','一次Policy logits'),('original_K8_plan_and_score_seconds','原K8规划与Scorer')]:
    lines.append('| '+title+' | 20 | '+ms_cell([np.mean([r[field] for r in rs]) for rs in groups.values()])+' |')
for method,title in [('rag_unlimited','RAG分支增量 全体'),('gap_frequency','gap分支增量 全体'),('terminal','Terminal分支增量')]:
    values=[np.mean([r['branch_increment_seconds'][method] for r in rs]) for rs in groups.values()]
    lines.append('| '+title+' | 20 | '+ms_cell(values)+' |')
for flag,title in [(False,'RAG 有可用提案'),(True,'RAG 精确回退')]:
    values=[np.mean([r['branch_increment_seconds']['rag_unlimited'] for r in rs]) for rs in groups.values() if rs[0]['RAG_fallback']==flag]
    lines.append('| '+title+' | '+str(len(values))+' | '+ms_cell(values)+' |')
k8=defaultdict(list)
for r in read('L_K8_existing_pool_CPU_Scorer_trials.json'):k8[r['protein_id']].append(r['CPU_Scorer_seconds'])
lines.append('| SCISOR已生成K8的同Scorer评分 | 20 | '+ms_cell([np.mean(v) for v in k8.values()])+' |')
done=read('cached_completion.json')
lines += ['',
    f'缓存载入实测{done["cache_load_seconds"]:.3f}秒，CPU头模型建立{done["heads_model_setup_seconds"]:.3f}秒，均单独记录。表中RAG总体中位数很低，与13/20例精确回退有关；真正有提案的7例应单独看，不能把回退耗时包装成所有检索引导的成本。','',
    '## 既有GPU原生生成组件','',
    '以下来自原始作业，不是新同设备基准。原生模型已驻留显存，模型启动时间不在这些分量中；Scorer、WT编码等未测部分按行标明。各方法的采样量不同，不称等计算量。','',
    '| 原生流程 | WT数 | 硬件 | 均值 s | 中位数 s | 四分位范围 s |','|---|---:|---|---:|---:|---|']
titles={'scisor_L_native':'SCISOR-L 原生K1','SCISOR_U90_L_K8_same_scorer':'SCISOR-L K8采样分量 不含Scorer/WT编码','raygun_native':'Raygun 500生成及500评分 保存50选1'}
for r in read('observed_native_summary.json'):
    s=r['seconds'];hardware=' / '.join(name+'×'+str(n) for name,n in r['hardware_counts'].items())
    lines.append('| '+titles[r['method']]+' | 20 | '+hardware+f' | {s["mean"]:.3f} | {s["median"]:.3f} | {s["p25"]:.3f}–{s["p75"]:.3f} |')
lines += ['',
    'Raygun原成本字段把50误标成评分候选数，现已在 observed_native_per_WT_v2.json 更正：500个生成序列全部评分，50是保存的前50，最终选1。旧v1文件留存，分数、计时及科学输出均未改变。其PLL名称沿用[官方实现](https://github.com/rohitsinghlab/raygun/blob/main/raygun/pll.py)，实际为未遮盖ESM token对数评分，不是逐位置遮盖的leave-one-out PLL。','',
    'SCISOR-K8实际只新采7次并复用既有原生第1次。上表按独立生成K8的8次原始采样分量求和；只新增7次的实际增量另存于逐WT记录，避免重复算账。CPU同Scorer耗时见上一表，尚未包含未缓存WT编码。','',
    '## 同一20例的冻结质量结果','',
    '全部采用同一ESMFold及既定共同结构对应。此表只复用最终输出，没有用性能计时重新选候选。','',
    '| 方法 | N | 对齐局部Cα几何 | 原生Cα接触恢复 | Full-WT TM |','|---|---:|---:|---:|---:|']
quality=defaultdict(list)
for r in read('fixed20_quality.json'):quality[r['method']].append(r)
names={'delepref_unlimited':'DelePref','rag_unlimited':'DelePref-RAG','gap_frequency':'简单gap引导','terminal':'Terminal参考','scisor_L_native':'SCISOR-L 原生K1','raygun_native':'Raygun 固定500/500评分/前50/选1','SCISOR_U90_L_K8_same_scorer':'SCISOR-L K8加同Scorer'}
for method in names:
    rs=quality[method]
    if len(rs)!=20:raise ValueError('Cost-quality method cohort differs')
    vals=[float(np.mean([r[k] for r in rs])) for k in ['structure_mapped_CA_lddt','structure_mapped_native_contact_recovery','full_WT_TM']]
    lines.append('| '+names[method]+' | 20 | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
setup=read('existing_batch_setup_totals.json')
lines += ['',
    '## 仍需补齐的效率测量','',
    f'原100WT批次的模型加载、编码及缓存写入总时长为{setup["original100WT_model_load_encode_and_cache_write_total"]["seconds"]:.3f}秒，100WT自然检索/投影批次为{setup["original100WT_natural_search_and_projection_total"]["seconds"]:.3f}秒。它们保留为批次背景，不能除以100便称单WT冷启动耗时，也不加入只有某些方法计费的结构评价成本。','',
    '同A100下的模型载入、未缓存WT逐条编码、端到端冷/热路径及峰值显存尚未测齐；GPU空出后按固定20例补测，所有计时候选只用于计时，不进入科学候选池或替换冻结输出。当前两张GPU继续原实验，不启动第三张卡。','',
    '本节只完成E8的现有计时核验与CPU缓存组件，不将E8或整套实验标为完成。论文未修改。','']
report=base/'cost_report_zh.md';report.write_text('\n'.join(lines),encoding='utf-8')
payload=dict(status='E8_PARTIAL_COST_REPORT_WITH_CORRECTED_SCORING_SCOPE_NOT_FULL_END_TO_END_BENCHMARK',
    fixed_WT=20,cached_CPU_trials_per_WT=3,all_frozen_four_method_outputs_match=True,
    RAG_active_cases=7,RAG_exact_fallback_cases=13,
    Raygun_generated=500,Raygun_candidate_sequences_scored=500,Raygun_saved=50,Raygun_selected=1,
    raw_v1_preserved=True,no_scientific_scores_or_outputs_changed=True,
    files={n:hashlib.sha256((base/n).read_bytes()).hexdigest() for n in ['observed_native_per_WT_v2.json','raygun_scoring_scope.json','cost_report_zh.md']},
    full_same_device_latency_and_memory_measurements_pending=True,new_model_calls_from_presentation=0,new_structure_predictions=0)
(base/'presentation_v2_completion.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(payload)
