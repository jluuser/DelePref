"""Package completed E9 quality and behavioral readouts; no new metrics or models."""
import argparse
import csv
import json
from pathlib import Path

from recomb_confirm_v2_diagnostics import read, sha, write
from recomb_confirm_v2_E9_completion import ARMS, claim

OUT = Path('reports/recomb_confirm_v2_E9_result')
METRICS = Path('reports/recomb_confirm_v2_E9_evaluation')
BEHAVIOR = Path('reports/recomb_confirm_v2_E9_behavior')
ENDPOINTS = ['structure_mapped_CA_lddt', 'structure_mapped_native_contact_recovery',
             'full_WT_TM', 'fixed_retained_CA_lddt', 'fixed_native_contact_recovery']


def run(root):
    out = root/OUT
    if (out/'completion.json').exists():
        raise FileExistsError('E9 result already packaged')
    done = read(root/METRICS/'completion.json')
    behavior_done = read(root/BEHAVIOR/'completion.json')
    if done['fixed_WT'] != 20 or done['new_logical_outputs'] != 40 or behavior_done['rows'] != 40:
        raise ValueError('Frozen E9 fixed20 paired scope differs')
    sources = {}
    for base, completion in [(METRICS, done), (BEHAVIOR, behavior_done)]:
        sources[(base/'completion.json').as_posix()] = sha(root/base/'completion.json')
        for name, expected in completion['outputs'].items():
            path = root/base/name
            if sha(path) != expected:
                raise ValueError('A completed experimental output changed: '+str(path))
            sources[(base/name).as_posix()] = expected
    claim(out/'execution_claim.json', dict(source_metrics_completion_sha256=sha(root/METRICS/'completion.json'),
        source_behavior_completion_sha256=sha(root/BEHAVIOR/'completion.json')))
    new = read(root/METRICS/'per_output_metrics.json')
    reference = read(root/METRICS/'reused_original_K8_reference_rows.json')
    comparisons = read(root/METRICS/'paired_comparisons.json')
    summary = read(root/BEHAVIOR/'behavior_summary.json')
    if len(new) != 40 or len(reference) != 40 or len({r['protein_id'] for r in new}) != 20:
        raise ValueError('Final quality/reference relationship coverage differs')
    quality = []
    for row in new+reference:
        item = dict(protein_id=row['protein_id'], method=row['method'],
            scope='matched_completion_control' if row['method'] in ARMS else 'original_same20_reference',
            physical_id=row['physical_id'], edited_sequence_sha256=row['edited_sequence_sha256'],
            prediction_status=row.get('prediction_status', 'ORIGINAL_VERIFIED_REFERENCE'))
        for endpoint in ENDPOINTS:
            if endpoint in ENDPOINTS[:3]:
                lower = upper = row['common_structure'].get(endpoint)
            else:
                key = 'retained_ca_lddt' if endpoint == 'fixed_retained_CA_lddt' else 'contact_recovery'
                lower = row['fixed_identity_geometry'].get(key+'_lower')
                upper = row['fixed_identity_geometry'].get(key+'_upper')
            item[endpoint+'_lower'] = lower
            item[endpoint+'_upper'] = upper
        quality.append(item)
    if len({(r['protein_id'], r['method']) for r in quality}) != 80:
        raise ValueError('Final per-WT quality table has duplicates/missing methods')
    paired = []
    for value in comparisons:
        if (value['method'], value['versus']) != ARMS:
            raise ValueError('Unexpected quality contrast')
        lo, hi = value['lower_effect'], value['upper_effect']
        paired.append(dict(method=value['method'], versus=value['versus'], endpoint=value['endpoint'],
            primary=value['endpoint'] in ENDPOINTS[:2], N=lo.get('N'), components=lo.get('components'),
            N_defined=lo.get('N_defined', lo.get('N')), status=lo['status'],
            mean_difference_lower=lo.get('mean_difference'), mean_difference_upper=hi.get('mean_difference'),
            CI95=json.dumps(value['CI95_conservative_envelope']),
            joint_CI97_5=json.dumps(value['joint_Bonferroni_CI97_5_conservative_envelope']),
            improved_lower=lo.get('improved'), degraded_lower=lo.get('degraded'), unchanged_lower=lo.get('unchanged'),
            improved_upper=hi.get('improved'), degraded_upper=hi.get('degraded'), unchanged_upper=hi.get('unchanged'),
            bootstrap_seed=lo.get('bootstrap_seed'), bootstrap_replicates=lo.get('replicates')))
    if len(paired) != 5:
        raise ValueError('Five pre-existing endpoint contrasts required')
    for name, rows in [('per_WT_quality.csv', quality), ('paired_effect_table.csv', paired)]:
        with (out/name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    write(out/'per_WT_quality.json', quality)
    write(out/'paired_effect_table.json', paired)
    def number(value):
        return '未定义' if value is None else f'{value:.6f}'
    text = ['# 相同同源种子下的补全规则对照', '',
        '固定20个普通General WT，20%删除预算。两臂使用相同的部分天然缺失种子、锚点、配对尝试槽、4个原Policy候选和4个补全候选，由同一正式4000步Scorer选出最终序列。',
        'Policy臂使用原学习式补全目标；简单合法臂从允许自由位置均匀无放回采样，不读取学习得分。', '',
        '## 已完成的质量比较', '',
        '差值为Policy补全减简单合法补全。完整20 WT/组件保留；N不乘方法数，区间直接复用已冻结的原指标和组件bootstrap结果。', '',
        '|端点|主要端点|差值下界|差值上界|95%区间|97.5%区间|', '|---|---|---:|---:|---|---|']
    for row in paired:
        text.append('|'+row['endpoint']+'|'+('是' if row['primary'] else '否')+'|'+number(row['mean_difference_lower'])+'|'+number(row['mean_difference_upper'])+'|'+row['CI95']+'|'+row['joint_CI97_5']+'|')
    text += ['', '共同Cα局部几何及原生接触恢复为两项主要端点；Full-WT TM与固定原始编辑对应两层结果保留。RMSD不用于这组比较。',
        '', '## 完成与采用行为', '', '|补全规则|尝试数|合法率|最终采用种子补全|完整4+4|', '|---|---:|---:|---:|---:|']
    for row in summary:
        text.append('|'+row['method']+'|'+str(row['attempted_completions'])+'|'+number(row['pooled_legal_rate'])+'|'+str(row['selected_seed_completion'])+'/20|'+str(row['fixed_original4_and_completion4_WT'])+'/20|')
    text += ['', '行为中的合法性与采用率分别说明约束满足和Scorer选择，不替代实际结构质量。种子与新增删除贡献按实际来源日志统计，不以偶然gap重合追认来源。',
        '', '## 预测与复用', '',
        '40条最终赢家关系对应'+str(done['unique_physical_selected_sequences'])+'个实际输入，兼容缓存'+str(done['compatible_cache_physical'])+'个；其余缺失输入才进行新预测。失败/未知数为'+str(done['missing_or_failed'])+'，分母保持20。',
        '本实验仅预测冻结最终赢家，没有完整候选池结构，不报告候选池oracle、regret或Success@K。原DelePref/RAG同20例只作背景参照，完整数值见per_WT_quality.csv及原评价均值表。',
        '', '## 解释范围', '',
        '这是原General100结果可见后冻结的机制补充。两臂固定第一组可行部分种子，并采用配对、无评分的候选入池规则；它回答相同种子下补全规则的差异，不是原全profile RAG赢家的重放。',
        '同一Policy与Scorer权重、三槽与段数除以3尺度不变；大于三段的Scorer外推继续披露。没有新训练、替代选择器或基于成绩的换样。',
        '结果说明本固定队列和计算结构端点下的表现，不等同酶活性或结合验证。原v2的所有主结果和正负比较均保持。',
        '', '完整逐WT、种子/补全长度、每次尝试及采用信息见同级recomb_confirm_v2_E9_behavior；原统计结果见recomb_confirm_v2_E9_evaluation。']
    (out/'E9_result_zh.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    sources['analysis/recomb_confirm_v2_E9_result.py'] = sha(root/'analysis/recomb_confirm_v2_E9_result.py')
    names = ['per_WT_quality.csv', 'paired_effect_table.csv', 'per_WT_quality.json', 'paired_effect_table.json', 'E9_result_zh.md']
    write(out/'completion.json', dict(status='E9_FIXED20_BEHAVIOR_AND_FINAL_WINNER_QUALITY_RESULT_PACKAGE_COMPLETE',
        WT=20, arms=2, per_WT_quality_rows=80, quality_contrasts=5, primary_endpoints=2,
        new_metric_or_bootstrap_calculations=0, new_model_calls=0, new_structure_predictions=0,
        sources_sha256=sources, outputs={n: sha(out/n) for n in names}))
    print(json.dumps(paired), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
