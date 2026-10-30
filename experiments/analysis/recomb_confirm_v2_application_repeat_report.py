"""Read-only completion of predeclared application seed means and readable tables.

Consumes completed frozen metrics; never predicts, changes selections, or drops an
undefined WT. The existing component bootstrap and endpoint adapter are reused.
"""
import argparse
from collections import Counter, defaultdict
import csv
import math
import os
from pathlib import Path

import numpy as np
from recomb_confirm_v2_diagnostics import read, sha, write

CAT = Path('reports/recomb_confirm_v2_application_catalog_v2')
EVALUATION = Path('reports/recomb_confirm_v2_application_evaluation/all')
OUT = Path('reports/recomb_confirm_v2_application_final_report')
FZ = Path('configs/recomb_confirm_v2_application_repeat_report_freeze.json')
CONFIRM = Path('configs/recomb_confirm_v2_application_confirmation_freeze.json')
METHOD_LABELS = dict(delepref_unlimited='DelePref', rag_unlimited='DelePref-RAG',
                     gap_frequency='Gap-frequency', terminal='Terminal',
                     scisor_L_native='SCISOR-L native K1', raygun_native='Raygun native', WT='WT')
MAIN_N = dict(Binder60=60, Enzyme80=80)
REPEAT_N = dict(Binder60=20, Enzyme80=30)


def validate_bound(a, b):
    if a is None and b is None:
        return
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b) or a > b:
        raise ValueError('Malformed endpoint uncertainty interval')


def validate_repeat_rows(rows, protocol):
    by = {}
    expected = {(cohort, pid, method)
                for cohort, ids in protocol['repeat_subset_by_cohort'].items()
                for pid in ids for method in protocol['methods'] + ['WT']}
    for row in rows:
        key = row['cohort'], row['protein_id'], row['method']
        if key in by or key not in expected or row['seed_count'] != 3:
            raise ValueError('Lost, duplicate, or unexpected fixed repeat member')
        if not row['endpoints']:
            raise ValueError('No endpoint means')
        for value in row['endpoints'].values():
            validate_bound(value['mean_lower'], value['mean_upper'])
        by[key] = row
    if set(by) != expected:
        raise ValueError('Repeat roster incomplete; do not silently exclude WT/methods')
    for cohort, ids in protocol['repeat_subset_by_cohort'].items():
        names = [set(by[cohort, pid, method]['endpoints'])
                 for pid in ids for method in protocol['methods'] + ['WT']]
        if any(n != names[0] for n in names):
            raise ValueError('Endpoint roster differs within a cohort')
        if any(pid not in protocol['component_by_protein_id'] for pid in ids):
            raise ValueError('Missing input-frozen component membership')
    return by


def descriptive_bounds(values):
    for a, b in values:
        validate_bound(a, b)
    defined = [(a, b) for a, b in values if a is not None]
    complete = len(defined) == len(values) and bool(values)
    return dict(N_all=len(values), N_defined=len(defined),
                N_exact=sum(a == b for a, b in defined),
                mean_lower=float(np.mean([a for a, b in values])) if complete else None,
                mean_upper=float(np.mean([b for a, b in values])) if complete else None,
                no_silent_case_exclusion=True)


def bounded_comparison(pairs, components, endpoint, bootstrap=None):
    if bootstrap is None:
        from recomb_confirm_v2_general_evaluate import effect
        bootstrap = effect
    lower, upper = [], []
    count = Counter(improved=0, degraded=0, unchanged=0, uncertain=0, undefined=0)
    higher = 'RMSD' not in endpoint
    for pid, a, b, c, d in pairs:
        validate_bound(a, b); validate_bound(c, d)
        lo = a - d if a is not None and d is not None else None
        hi = b - c if b is not None and c is not None else None
        lower.append((pid, lo)); upper.append((pid, hi))
        if lo is None or hi is None:
            count['undefined'] += 1
            continue
        favorable_lo, favorable_hi = (lo, hi) if higher else (-hi, -lo)
        if favorable_lo > 1e-8:
            count['improved'] += 1
        elif favorable_hi < -1e-8:
            count['degraded'] += 1
        elif abs(favorable_lo) <= 1e-8 and abs(favorable_hi) <= 1e-8:
            count['unchanged'] += 1
        else:
            count['uncertain'] += 1
    lo = bootstrap(lower, components, seed=20261016, replicates=10000)
    hi = bootstrap(upper, components, seed=20261016, replicates=10000)
    # Underlying effect() labels higher numeric deltas 'improved'. Rename those
    # bound-only counts; actual favorable/uncertain classification is above.
    for result in [lo, hi]:
        for old, new in [('improved', 'numeric_increase'), ('degraded', 'numeric_decrease'),
                         ('unchanged', 'numeric_unchanged')]:
            if old in result:
                result[new] = result.pop(old)
    return dict(N_all=len(pairs), components=len({components[p[0]] for p in pairs}),
                endpoint=endpoint, difference='method minus original DelePref in original endpoint units',
                favorable_direction='higher' if higher else 'lower',
                lower_effect=lo, upper_effect=hi,
                mean_difference_bounds=[lo.get('mean_difference'), hi.get('mean_difference')],
                CI95_conservative_envelope=[lo['CI95'][0], hi['CI95'][1]]
                if 'CI95' in lo and 'CI95' in hi else None,
                favorable_WT_counts=dict(count), numeric_tolerance=1e-8,
                undefined_WT_not_excluded=True,
                secondary_repeat_analysis_not_full_cohort_primary=True)


def repeat_tables(rows, protocol, bootstrap=None):
    by = validate_repeat_rows(rows, protocol)
    tables, comparisons = [], []
    for cohort, ids in sorted(protocol['repeat_subset_by_cohort'].items()):
        ids = sorted(ids)
        methods = protocol['methods'] + ['WT']
        endpoints = sorted(by[cohort, ids[0], methods[0]]['endpoints'])
        for method in methods:
            fields = {}
            for endpoint in endpoints:
                vals = [by[cohort, pid, method]['endpoints'][endpoint] for pid in ids]
                fields[endpoint] = descriptive_bounds([(v['mean_lower'], v['mean_upper']) for v in vals])
                for field in ['point_seed_SD', 'point_worst_seed']:
                    v = [p[field] for p in vals]
                    fields[endpoint][field + '_WT_mean'] = float(np.mean(v)) if all(x is not None for x in v) else None
                    fields[endpoint][field + '_defined_WT'] = sum(x is not None for x in v)
            tables.append(dict(cohort=cohort, method=method, budget=.2 if method != 'WT' else 0,
                               N_WT=len(ids), seeds_per_WT=3, endpoints=fields))
            if method in ['delepref_unlimited', 'WT']:
                continue
            for endpoint in endpoints:
                pairs = []
                for pid in ids:
                    a = by[cohort, pid, method]['endpoints'][endpoint]
                    b = by[cohort, pid, 'delepref_unlimited']['endpoints'][endpoint]
                    pairs.append((pid, a['mean_lower'], a['mean_upper'], b['mean_lower'], b['mean_upper']))
                comparisons.append(dict(cohort=cohort, method=method, versus='delepref_unlimited',
                    seeds_averaged_before_inference=True,
                    **bounded_comparison(pairs, protocol['component_by_protein_id'], endpoint, bootstrap)))
    return tables, comparisons


def validate_seed_provenance(per_output, summaries, protocol):
    from recomb_confirm_v2_application_evaluate import endpoint_bounds, seed_mean
    groups = defaultdict(list)
    for row in per_output:
        if (row['protein_id'] in protocol['repeat_subset_by_cohort'][row['cohort']]
                and (row['budget'] == .2 or row['method'] == 'WT')):
            groups[row['cohort'], row['protein_id'], row['method']].append(row)
    for row in summaries:
        values = groups[row['cohort'], row['protein_id'], row['method']]
        if sorted(v['seed'] for v in values) != protocol['structure_seeds']:
            raise ValueError('Actual seed roster is missing or duplicated')
        for endpoint, saved in row['endpoints'].items():
            expected = seed_mean([endpoint_bounds(v, endpoint) for v in values])
            for a, b in zip(expected, [saved['mean_lower'], saved['mean_upper']]):
                if (a is None) != (b is None) or (a is not None and abs(a-b) > 1e-12):
                    raise ValueError('Saved per-WT mean differs from frozen per-seed metrics')


def coverage_tables(rows, protocol):
    groups = defaultdict(list)
    for r in rows:
        if r['seed'] == 20261017:
            groups[r['cohort'], r['method'], r['budget'], 'main_seed_all_roster'].append(r)
        if (r['protein_id'] in protocol['repeat_subset_by_cohort'][r['cohort']]
                and (r['budget'] == .2 or r['method'] == 'WT')):
            groups[r['cohort'], r['method'], r['budget'], 'fixed_three_seed_subset'].append(r)
    return [dict(cohort=c, method=m, budget=b, analysis=role, logical=len(g),
                 WT=len({r['protein_id'] for r in g}), physical=len({r['physical_id'] for r in g}),
                 statuses=dict(Counter(r['prediction_status'] for r in g)),
                 failed_or_missing_logical=sum(r['prediction_status'] != 'SUCCEEDED' for r in g))
            for (c, m, b, role), g in sorted(groups.items())]


def annotation_tables(root, rows):
    from recomb_confirm_v2_application_evaluate import endpoint_bounds
    panel = {r['protein_id']: r for r in read(root/'reports/recomb_confirm_v2_Enzyme80_constructs/Enzyme80_panel.json')}
    groups = defaultdict(list)
    for r in rows:
        if r['cohort'] == 'Enzyme80' and r['seed'] == 20261017:
            groups[panel[r['protein_id']]['annotation_tier'], r['method'], r['budget']].append(r)
    return [dict(annotation_tier=t, method=m, budget=b,
                 endpoint='complete_catalytic_AA_identity',
                 **descriptive_bounds([endpoint_bounds(r, 'complete_catalytic_AA_identity') for r in g]))
            for (t, m, b), g in sorted(groups.items())]


def fmt(value):
    if value is None:
        return '未定义'
    if isinstance(value, (tuple, list)):
        if any(v is None for v in value):
            return '未定义'
        return f'{value[0]:.4f}' if value[0] == value[1] else f'[{value[0]:.4f}, {value[1]:.4f}]'
    return f'{value:.4f}'


def readable_report(tables, comparisons, repeats, repeat_comparisons, coverage, tiers, completion):
    lines = ['# RECOMB v2 应用完整结果', '',
             '以下为已冻结全部输入与方法的计算结果。结构、界面和催化身份均不是实测结合或酶活性。', '',
             f"全应用逻辑输出{completion['logical']}；失败或缺失实际身份{completion['missing_or_failed']}。全部预定分母保留。", '',
             '## 全队列与预算表', '',
             '主预算20%，WT单列；10%/30%为固定嵌套子集，不能与主表直接比较样本均值推断预算效应。以下是固定主seed结果。', '']
    fields = dict(Binder60=['interface_contact_recovery', 'interface_F1', 'structure_mapped_CA_lddt',
                           'structure_mapped_native_contact_recovery', 'full_WT_TM'],
                  Enzyme80=['complete_catalytic_AA_identity', 'catalytic_neighbor_contact_recovery',
                            'identity_aware_catalytic_neighbor_contact_recovery', 'structure_mapped_CA_lddt',
                            'structure_mapped_native_contact_recovery', 'full_WT_TM'])
    for cohort, endpoints in fields.items():
        lines += [f'### {cohort}', '', '| 方法 | 预算 | N | ' + ' | '.join(endpoints) + ' |',
                  '|---|---:|---:|' + '---:|' * len(endpoints)]
        for row in tables:
            if row['cohort'] != cohort:
                continue
            values = [fmt([row['endpoints'][e]['mean_lower'], row['endpoints'][e]['mean_upper']])
                      + f" ({row['endpoints'][e]['N_defined']}/{row['N_all']})" for e in endpoints]
            lines.append(f"| {METHOD_LABELS[row['method']]} | {row['budget']:.0%} | {row['N_all']} | " + ' | '.join(values) + ' |')
        lines += ['']
    lines += ['## 预定主比较', '',
              'RAG减原DelePref；界面接触恢复与完整催化AA身份分别是两应用的主要终点。区间端点对应最保守的差值界及组件bootstrap置信包络。', '']
    for row in comparisons:
        if row.get('primary_comparison'):
            lo, hi = row['lower_effect'], row['upper_effect']
            lines += [f"- {row['cohort']} {row['endpoint']}：N={lo['N']}；均值差{fmt([lo.get('mean_difference'), hi.get('mean_difference')])}；95%CI包络{fmt(row['CI95_conservative_envelope'])}。"]
    lines += ['', '## 固定子集的三seed重复', '',
              'Binder20与Enzyme30，每个WT/方法先平均全部三个固定seed，再按原组件bootstrap10000次、seed20261016。它们是重复稳定性次要分析，不替代全队列主seed终点。N不乘3；未知结果不删病例，不取最好seed。', '',
              '| 队列 | 方法 | WT数 | 终点 | 三seed均值范围 | 95%CI包络 | 改善/退化/不变/不确定/未定义 |',
              '|---|---|---:|---|---|---|---|']
    primary = dict(Binder60='interface_contact_recovery', Enzyme80='complete_catalytic_AA_identity')
    for row in repeat_comparisons:
        if row['endpoint'] != primary[row['cohort']]:
            continue
        counts = row['favorable_WT_counts']
        count_text = '/'.join(str(counts[k]) for k in ['improved', 'degraded', 'unchanged', 'uncertain', 'undefined'])
        lines.append(f"| {row['cohort']} | {METHOD_LABELS[row['method']]}减DelePref | {row['N_all']} | {row['endpoint']} | {fmt(row['mean_difference_bounds'])} | {fmt(row['CI95_conservative_envelope'])} | {count_text} |")
    lines += ['', '完整三seed方法表、SD/最差seed的可识别分母及各终点配对差值见JSON；SD与最差seed只在三个结果都精确可识别时汇总。RMSD越低越好，差值仍为方法减原模型，不能将数值上升称改善。', '',
              '## 覆盖与注释边界', '',
              '每组完整逻辑数、实际身份数、成功/失败/缺失状态见coverage.json；不把重复或去重缓存当独立蛋白。', '',
              '酶79例M-CSA完整催化网络与1例reviewed UniProt Active site分层表见annotation_tiers.json；EC7单例不构成类别充分验证。保留身份不等于活性，形状对齐不证明身份。', '',
              'Binder主终点为H-抗原固定原生重原子<5Å接触恢复；H-轻链单列于原始指标。自定义主链RMSD分量不是官方DockQ总分。Raygun对应歧义、缺失原子与化学表示限制保留，所有正负结果与WT能力均须共同解释。', '',
              '本报告仅只读完成既定统计与展示，未重选输入、方法、候选、Scorer、seed或终点；论文保持暂停。', '']
    return '\n'.join(lines)


def prepare(root):
    if (root/FZ).exists():
        raise FileExistsError('Repeat/report software protocol already frozen')
    names = ['analysis/recomb_confirm_v2_application_repeat_report.py',
             'analysis/tests/test_recomb_confirm_v2_application_repeat_report.py',
             'analysis/recomb_confirm_v2_general_evaluate.py',
             'analysis/recomb_confirm_v2_application_evaluate.py',
             'analysis/recomb_confirm_v2_diagnostics.py',
             'slurm/recomb_confirm_v2_application_repeat_report.sbatch',
             'slurm/recomb_confirm_v2_application_repeat_QA.sbatch']
    for cohort in MAIN_N:
        names += [f'reports/recomb_confirm_v2_{cohort}_constructs/{cohort}_panel.json',
                  f'reports/recomb_confirm_v2_{cohort}_subsets/subsets.json']
    write(root/FZ, dict(status='APPLICATION_READ_ONLY_REPEAT_STATISTICS_AND_REPORT_SOFTWARE_FROZEN',
        files={p: sha(root/p) for p in names}, main_WT=MAIN_N, repeat_WT=REPEAT_N,
        protocol='all3fixed seeds mean within WT/method, then component paired bootstrap10000 seed20261016',
        uncertainty='method lower minus baseline upper; method upper minus baseline lower; no case deletion',
        direction='higher favorable except RMSD; raw differences retain original endpoint units',
        no_model_calls=True, no_new_structure_predictions=True,
        consumes_only_completed_all_phase_metrics=True,
        runtime_full_confirmation_and_output_SHA_validation=True,
        added_before_application_prediction_outcomes=True,
        existing_primary_statistics_unchanged=True))
    (root/OUT).mkdir(parents=True, exist_ok=True)
    print(sha(root/FZ), flush=True)


def run(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Read-only full report requires CPU Slurm')
    out = root/OUT
    if (out/'completion.json').exists():
        raise FileExistsError('Application final report is already complete')
    f = read(root/FZ)
    for p, h in f['files'].items():
        if sha(root/p) != h:
            raise ValueError('Repeat/report frozen source or input changed')
    confirm = read(root/CONFIRM)
    for p in confirm['files']:
        if sha(p['path']) != p['sha256']:
            raise ValueError('Full application confirmation source/input changed')
    done = read(root/EVALUATION/'completion.json')
    if done['phase'] != 'all' or done['logical'] != 2160:
        raise ValueError('Only complete fixed application all-phase metrics may be summarized')
    for p, h in done['outputs'].items():
        if sha(root/EVALUATION/p) != h:
            raise ValueError('Frozen completed application metrics changed')
    protocol = read(root/CAT/'statistical_protocol.json')
    if (protocol['structure_seeds'] != [20261017, 20261018, 20261019]
            or protocol['bootstrap_seed'] != 20261016 or protocol['bootstrap_replicates'] != 10000
            or protocol['main_roster'] != MAIN_N
            or {c: len(ids) for c, ids in protocol['repeat_subset_by_cohort'].items()} != REPEAT_N):
        raise ValueError('Predeclared protocol or denominators differ')
    rows = read(root/EVALUATION/'per_output_metrics.json')
    summaries = read(root/EVALUATION/'within_WT_three_seed_summaries.json')
    tables = read(root/EVALUATION/'mean_tables.json')
    comparisons = read(root/EVALUATION/'paired_comparisons.json')
    if len(rows) != 2160:
        raise ValueError('Lost logical outcomes')
    repeat, paired = repeat_tables(summaries, protocol)
    validate_seed_provenance(rows, summaries, protocol)
    coverage = coverage_tables(rows, protocol); tiers = annotation_tables(root, rows)
    outputs = dict(repeat_method_tables=repeat, repeat_component_comparisons=paired,
                   coverage=coverage, annotation_tiers=tiers,
                   full_main_and_budget_tables=tables, full_main_and_budget_comparisons=comparisons)
    for name, value in outputs.items():
        write(out/(name+'.json'), value)
    report = readable_report(tables, comparisons, repeat, paired, coverage, tiers, done)
    (out/'application_result_zh.md').write_text(report, encoding='utf-8')
    with (out/'endpoint_tables.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['analysis','cohort','method','budget','N_all','endpoint','N_defined','N_exact','mean_lower','mean_upper']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for kind, source in [('main_seed', tables), ('three_seed_WT_mean', repeat)]:
            for row in source:
                for endpoint, value in row['endpoints'].items():
                    writer.writerow(dict(analysis=kind, cohort=row['cohort'], method=row['method'],
                        budget=row['budget'], N_all=value['N_all'], endpoint=endpoint,
                        **{k:value.get(k) for k in ['N_defined','N_exact','mean_lower','mean_upper']}))
    output_paths = [name+'.json' for name in outputs] + ['application_result_zh.md', 'endpoint_tables.csv']
    write(out/'completion.json', dict(status='APPLICATION_ALL_FROZEN_METRICS_REPEAT_COMPONENT_INFERENCE_AND_REPORT_COMPLETE',
        all_phase_metrics_sha256=sha(root/EVALUATION/'completion.json'), confirmation_sha256=sha(root/CONFIRM),
        protocol_sha256=sha(root/CAT/'statistical_protocol.json'), software_freeze_sha256=sha(root/FZ),
        logical=2160, repeat_WT=REPEAT_N, no_model_calls=True, no_new_structures=True,
        repeated_seeds_not_independent_WT=True, no_case_or_endpoint_reselection=True,
        not_functional_assay=True, job_id=os.environ['SLURM_JOB_ID'],
        outputs={p:sha(out/p) for p in output_paths}))
    print(read(out/'completion.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare(args.root) if args.prepare else run(args.root)
