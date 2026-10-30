"""Fixed20, selected-winner-only E9 evaluation with unchanged General metric functions."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
from pathlib import Path

from recomb_confirm_v2_E9_completion import read, sha, write
from recomb_confirm_v2_application_serialization import json_native
from recomb_confirm_v2_general_evaluate import metric_pair, effect

CAT = Path('reports/recomb_confirm_v2_E9_catalog')
STRUCT = Path('reports/recomb_confirm_v2_E9_structures')
OUT = Path('reports/recomb_confirm_v2_E9_evaluation')
CONFIRM = Path('configs/recomb_confirm_v2_E9_confirmation_freeze.json')
ENDPOINTS = ['structure_mapped_CA_lddt', 'structure_mapped_native_contact_recovery', 'full_WT_TM',
             'fixed_retained_CA_lddt', 'fixed_native_contact_recovery']


def bounds(row, endpoint):
    if endpoint in ENDPOINTS[:3]:
        a = b = row['common_structure'].get(endpoint)
    else:
        key = 'retained_ca_lddt' if endpoint == 'fixed_retained_CA_lddt' else 'contact_recovery'
        a = row['fixed_identity_geometry'].get(key+'_lower')
        b = row['fixed_identity_geometry'].get(key+'_upper')
    return (a, b) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b) else (None, None)


def summary(rows, endpoint):
    values = [bounds(r, endpoint) for r in rows]
    n = sum(a is not None and b is not None for a, b in values)
    return dict(N_all=len(rows), N_defined=n,
                mean_lower=sum(a for a, b in values)/len(rows) if n == len(rows) and rows else None,
                mean_upper=sum(b for a, b in values)/len(rows) if n == len(rows) and rows else None,
                no_silent_exclusion=True)


def evaluate(root, checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('E9 metrics require CPU Slurm')
    if sha(root/CONFIRM) != checksum:
        raise ValueError('E9 structural confirmation changed')
    for item in read(root/CONFIRM)['files']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen E9 input/source changed: '+item['path'])
    out = root/OUT;out.mkdir(parents=True, exist_ok=True)
    if (out/'completion.json').exists():
        done = read(out/'completion.json')
        if done['confirmation_sha256'] != checksum:
            raise ValueError('Existing E9 readout has a different parent')
        for n, h in done['outputs'].items():
            if sha(out/n) != h:
                raise ValueError('Completed E9 readout changed')
        print(json.dumps(done), flush=True)
        return
    logical = read(root/CAT/'logical_requests.json')
    physical = {r['task_id']: r for r in read(root/CAT/'physical_requests.json')}
    cache = read(root/CAT/'compatible_cache.json')
    protocol = read(root/CAT/'statistical_protocol.json')
    panel = {r['protein_id']: r for r in read(root/'reports/recomb_confirm_v2_general_native_v2/General100_panel.json')}
    artifacts, statuses, audit = {}, {}, []
    for task, request in physical.items():
        if task in cache:
            info = cache[task]
            if sha(info['record_path']) != info['record_sha256'] or sha(info['journal_path']) != info['journal_sha256']:
                raise ValueError('Compatible cache provenance changed')
            path = Path(info['artifact_path']);expected = info['artifact_sha256'];status = 'VERIFIED_CACHE'
        else:
            p = root/STRUCT/'records'/(task+'.json')
            saved = read(p) if p.exists() else {}
            status = saved.get('status', 'NOT_PRODUCED')
            if status not in ['SUCCEEDED', 'FAILED']:
                raise ValueError('A frozen E9 physical identity is not terminal')
            path = Path(saved['output_path']) if status == 'SUCCEEDED' else None
            expected = saved.get('output_sha256')
            if any(saved.get(k) != request[k] for k in ['sequence_sha256', 'weights_sha256', 'profile_sha256', 'seed', 'sample']):
                raise ValueError('E9 prediction identity differs')
        if path:
            if sha(path) != expected:
                raise ValueError('E9 prediction artifact changed')
            import numpy as np
            with np.load(path, allow_pickle=False) as z:
                if str(z['sequence']) != request['sequence'] or z['coordinates'].shape != (len(request['sequence']), 3):
                    raise ValueError('E9 predicted sequence/coordinates differ')
        artifacts[task] = str(path) if path else None;statuses[task] = status
        audit.append(dict(physical_id=task, status=status, artifact_path=artifacts[task], artifact_sha256=expected))
    previous_dir = root/'reports/recomb_confirm_v2_General100_evaluation'
    previous_done = read(previous_dir/'completion.json')
    if sha(previous_dir/'per_output_metrics.json') != previous_done['outputs']['per_output_metrics.json']:
        raise ValueError('Completed General reference readout changed')
    previous = read(previous_dir/'per_output_metrics.json')
    key_for = lambda r: (r['protein_id'], r['physical_id'], json.dumps(r['correspondence'], sort_keys=True))
    old_metrics = {key_for(r): dict(common_structure=r['common_structure'], fixed_identity_geometry=r['fixed_identity_geometry'])
                   for r in previous if r['predictor'] == 'esmfold'}
    jobs, metrics, reused = {}, {}, 0
    for row in logical:
        key = key_for(row)
        if key in old_metrics:
            metrics[key] = old_metrics[key];reused += 1
        else:
            jobs.setdefault(key, (str(root), key, panel[row['protein_id']]['wt_native_json'],
                                  artifacts[row['physical_id']], len(row['edited_sequence']), row['correspondence']))
    if jobs:
        with ProcessPoolExecutor(max_workers=8) as executor:
            metrics.update(executor.map(metric_pair, jobs.values()))
    result = [dict(row, prediction_status=statuses[row['physical_id']], **metrics[key_for(row)]) for row in logical]
    ids = sorted({r['protein_id'] for r in logical})
    baseline = [r for r in previous if r['predictor'] == 'esmfold' and r['evaluation_role'] == 'primary20pct'
                and r['protein_id'] in ids and r['method'] in ['delepref_unlimited', 'rag_unlimited']]
    if len(result) != 40 or len(ids) != 20 or len(baseline) != 40:
        raise ValueError('E9 fixed20 comparison denominator differs')
    combined = baseline+result
    by = {(r['protein_id'], r['method']): r for r in combined}
    if len(by) != 80:
        raise ValueError('Duplicated or missing E9/old-reference logical row')
    table = [dict(method=method, endpoint=endpoint, **summary([r for r in combined if r['method'] == method], endpoint))
             for method in protocol['methods'] for endpoint in ENDPOINTS]
    comparisons = []
    for method, reference in protocol['requested_comparisons']:
        for endpoint in ENDPOINTS:
            lower, upper = [], []
            for pid in ids:
                a, b = bounds(by[pid, method], endpoint);c, d = bounds(by[pid, reference], endpoint)
                lower.append((pid, a-d if a is not None and d is not None else None))
                upper.append((pid, b-c if b is not None and c is not None else None))
            lo = effect(lower, protocol['component_by_protein_id'], protocol['bootstrap_seed'], protocol['bootstrap_replicates'])
            hi = lo if lower == upper else effect(upper, protocol['component_by_protein_id'], protocol['bootstrap_seed'], protocol['bootstrap_replicates'])
            comparisons.append(dict(method=method, versus=reference, endpoint=endpoint,
                lower_effect=lo, upper_effect=hi,
                CI95_conservative_envelope=[lo['CI95'][0], hi['CI95'][1]] if 'CI95' in lo and 'CI95' in hi else None,
                joint_Bonferroni_CI97_5_conservative_envelope=[lo['joint_Bonferroni_CI97_5'][0], hi['joint_Bonferroni_CI97_5'][1]] if 'joint_Bonferroni_CI97_5' in lo and 'joint_Bonferroni_CI97_5' in hi else None,
                post_outcome_exploratory=True))
    outputs = dict(per_output_metrics=result, reused_original_K8_reference_rows=baseline, mean_table=table,
                   paired_comparisons=comparisons, prediction_identity_audit=audit)
    for name, value in outputs.items():
        write(out/(name+'.json'), json_native(value))
    with (out/'complete_mean_table.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'endpoint', 'N_all', 'N_defined', 'mean_lower', 'mean_upper', 'no_silent_exclusion'])
        w.writeheader();w.writerows(table)
    def fmt(x):
        return '未定义' if x is None else f'{x:.6f}'
    lines = ['# 相同同源种子下的补全对照：固定20 WT', '',
        '两臂使用相同部分种子、锚点、配对尝试槽、原Policy4候选、K8和正式4000步Scorer。只改变剩余删除位置的补全规则。',
        '本实验是原结果可见后冻结的机制补充，N=20个WT/组件；原General100主结果不改。', '',
        '|方法|端点|全体N|定义N|均值下界|均值上界|', '|---|---|---:|---:|---:|---:|']
    for r in table:
        lines.append('|'+r['method']+'|'+r['endpoint']+'|20|'+str(r['N_defined'])+'|'+fmt(r['mean_lower'])+'|'+fmt(r['mean_upper'])+'|')
    lines += ['', '## Policy补全减简单合法补全', '']
    for r in comparisons:
        lines.append('- '+r['endpoint']+'：差值界['+fmt(r['lower_effect'].get('mean_difference'))+', '+fmt(r['upper_effect'].get('mean_difference'))+
                     ']；95%区间'+str(r['CI95_conservative_envelope'])+'；共同主要97.5%区间'+str(r['joint_Bonferroni_CI97_5_conservative_envelope'])+'。')
    behavior = read(root/'reports/recomb_confirm_v2_E9_completion/behavior.json')
    lines += ['', '## 行为、复用与解释', '',
        '样本只按输入是否存在可合法补全的部分天然种子及固定哈希顺序选取，不依据历史赢家或结构成绩。',
        '第一候选沿用原Policy补全；随后沿用单位高斯扰动约定，配对尝试上限128。仅当两臂均产生未重复合法mask时共同入池，使种子、尝试槽与候选配额相同；这是匹配机制实验，不是原全profile RAG赢家重放。',
        '每WT首先固定4个原Policy候选，再加入4个配对补全候选；不足时两臂同序回填原Policy并保留病例。实际不足病例数：'+str(sum(r['shortfall']>0 for r in behavior))+'。',
        '实际配对尝试数合计：'+str(sum(r['attempts'] for r in behavior))+'。',
        '同源种子和新增补全的删除量、逐尝试来源、最终采用率及完整候选均保存在生成包behavior.json与candidate_pools.json。',
        '原DelePref与原RAG仅作为同20例背景参照；新主要比较是Policy补全与不读学习得分的均匀合法补全。主要端点为共同原生接触及共同局部Cα几何，普通95%与两主要端点97.5%区间并列。',
        'Scorer原三槽及段数除以3尺度不变，超过三段继续标注外推；没有新训练、没有按结果换模型、选择器或名单。身份对应与结构对应两层并列，未知不填零；本实验不等于活性或结合验证。',
        '逻辑赢家40，实际输入'+str(len(physical))+'；复用'+str(len(cache))+'，缺失预测'+str(len(physical)-len(cache))+'；失败'+str(sum(v=='FAILED' for v in statuses.values()))+'。',
        '既有精确指标关系复用'+str(reused)+'条；新计算关系'+str(len(jobs))+'个。']
    (out/'E9_result_zh.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    files = [name+'.json' for name in outputs]+['complete_mean_table.csv', 'E9_result_zh.md']
    done = dict(status='E9_FIXED20_SELECTED_WINNER_EXPLORATORY_READOUT_COMPLETE_NOT_FULL_POOL_QUALITY',
        confirmation_sha256=checksum, fixed_WT=20, new_logical_outputs=40, original_reference_logical_rows=40,
        unique_physical_selected_sequences=len(physical), compatible_cache_physical=len(cache),
        missing_or_failed=sum(v == 'FAILED' for v in statuses.values()),
        exact_old_metric_logical_relationships_reused=reused, new_CPU_metric_relationships=len(jobs),
        no_case_or_method_or_selector_reselection=True, not_functional_validation=True,
        original_General100_confirmation_unchanged=True,
        source_old_General_completion_sha256=sha(previous_dir/'completion.json'),
        outputs={name: sha(out/name) for name in files}, job_id=os.environ['SLURM_JOB_ID'])
    write(out/'completion.json', done);print(json.dumps(done), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    args = parser.parse_args()
    evaluate(args.root, args.freeze_sha256)
