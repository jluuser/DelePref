"""Verify and summarize all240 measured requests without new model execution."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from recomb_confirm_v2_diagnostics import read, sha, write
from recomb_confirm_v2_E9_completion import claim, digest
from recomb_confirm_v2_E10_fidelity import tensor_identity
import delepref_unlimited_planning as planning

BASE = Path('reports/recomb_confirm_v2_E10_timing_padding_v2')
TIMING_FREEZE = Path('configs/recomb_confirm_v2_E10_timing_padding_v2_freeze.json')
FREEZE = Path('configs/recomb_confirm_v2_E10_result_freeze.json')
OUT = Path('reports/recomb_confirm_v2_E10_result')
HEADS = ('delepref_unlimited', 'rag_unlimited', 'gap_frequency', 'terminal')


def prepare(root):
    if (root/FREEZE).exists():
        raise FileExistsError('E10 result verification is already frozen')
    names = ['analysis/recomb_confirm_v2_E10_result.py', 'slurm/recomb_confirm_v2_E10_result.sbatch',
        'analysis/recomb_confirm_v2_diagnostics.py', 'analysis/recomb_confirm_v2_E9_completion.py',
        'analysis/recomb_confirm_v2_E10_fidelity.py', 'analysis/delepref_unlimited_planning.py', str(TIMING_FREEZE)]
    write(root/FREEZE, dict(status='E10_ALL240_REQUEST_INTEGRITY_AND_DESCRIPTIVE_SUMMARY_SOURCE_FROZEN',
        files={p: sha(root/p) for p in names}, WT=20, methods=6, scenarios=2, expected_requests=240,
        no_best_run_or_success_case_exclusion=True, no_new_model_forwards=True, no_new_structure_predictions=True))
    (root/OUT).mkdir(parents=True, exist_ok=True)
    print(sha(root/FREEZE), flush=True)


def stats(values):
    a = np.asarray(values, dtype=float)
    if len(a) != 20 or not np.isfinite(a).all():
        raise ValueError('Timing summary must contain all20 finite measurements')
    return dict(N=20, mean=float(a.mean()), median=float(np.median(a)),
                p25=float(np.quantile(a, .25)), p75=float(np.quantile(a, .75)),
                minimum=float(a.min()), maximum=float(a.max()))


def run(root, checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Final integrity analysis requires CPU Slurm')
    if sha(root/FREEZE) != checksum:
        raise ValueError('Result verification source freeze differs')
    for p, h in read(root/FREEZE)['files'].items():
        if sha(root/p) != h:
            raise ValueError('Result source/input changed: '+p)
    done = read(root/BASE/'measurement_completion.json')
    if done['fixed_WT'] != 20 or done['methods'] != 6 or done['scenarios'] != 2:
        raise ValueError('Complete measurement scope differs')
    for p, h in done['outputs'].items():
        if sha(root/BASE/p) != h:
            raise ValueError('Completed timing table changed: '+p)
    frozen = read(root/TIMING_FREEZE)
    if done['freeze_sha256'] != sha(root/TIMING_FREEZE):
        raise ValueError('Measurement source release differs')
    records = read(root/BASE/'all_request_measurements.json')
    by = {(r['protein_id'], r['method'], r['scenario']): r for r in records}
    if len(records) != 240 or len(by) != 240:
        raise ValueError('All240 unique actual requests are required')
    out = root/OUT
    claim(out/'execution_claim.json', dict(job_id=os.environ['SLURM_JOB_ID'], source_freeze_sha256=checksum,
        measurement_completion_sha256=sha(root/BASE/'measurement_completion.json')))
    import torch
    rows, identities = [], []
    devices, nodes = set(), set()
    search_count = 0
    actual_outputs_match = 0
    for index, case in enumerate(frozen['cases']):
        pid = case['protein_id'];sequence = case['wildtype_sequence']
        for method in frozen['methods']:
            directory = root/BASE/'requests'/('%02d_%s' % (index, case['WT_sha256'][:12]))/method
            observed = read(directory/'parent_observed_requests.json')
            worker = read(directory/'worker_completion.json')
            if worker['freeze_sha256'] != sha(root/TIMING_FREEZE) or [r['scenario'] for r in observed] != ['cold', 'cached']:
                raise ValueError('A two-request method process has different provenance')
            if method in HEADS:
                cache = torch.load(directory/'actual_cold_WT_cache.pt', map_location='cpu', weights_only=True)
                if cache['sequences'][pid] != sequence or tensor_identity(cache['embeddings'][pid]) != case['scientific_embedding_identity']:
                    raise ValueError('Actual timed WT tensor does not match scientific identity')
            for scenario, beta in [('cold', .2), ('cached', .3)]:
                value = by[pid, method, scenario]
                if value not in observed or value['beta'] != beta:
                    raise ValueError('Parent timing observation differs from completed table')
                internal = {k: v for k, v in value.items() if k != 'parent_observed_request_seconds'}
                if internal not in worker['measurements'] or internal != read(directory/scenario/'measurement.json'):
                    raise ValueError('Parent/worker measured-output event differs')
                for name, h in value['output_files_sha256'].items():
                    if sha(directory/scenario/name) != h:
                        raise ValueError('Actual timed output SHA differs')
                ident = read(directory/scenario/'request_identity.json')
                selected = read(directory/scenario/'selected_output.json')
                pool = read(directory/scenario/'candidate_pool.json')
                if ident['WT_sha256'] != case['WT_sha256'] or ident['method'] != method or ident['beta'] != beta:
                    raise ValueError('Actual request input/method identity differs')
                actual_sha = hashlib.sha256(selected['edited_sequence'].encode()).hexdigest()
                if actual_sha != value['selected_sequence_sha256'] or actual_sha != selected['edited_sequence_sha256']:
                    raise ValueError('Actual final sequence identity differs')
                expected = case['scientific_final_sequences'][method][str(beta)]['edited_sequence_sha256']
                match = actual_sha == expected
                if match != value['selected_sequence_matches_scientific_output']:
                    raise ValueError('Scientific sequence comparison flag differs')
                actual_outputs_match += int(match)
                if len(selected['edited_sequence']) != len(sequence)-planning.delete_count(len(sequence), beta):
                    raise ValueError('Actual shrinking budget differs')
                if method in HEADS or method == 'scisor_L_native':
                    raw = selected['deleted_intervals']
                    intervals = json.loads(raw) if isinstance(raw, str) else raw
                    mask = planning.validate_plan(len(sequence), beta, intervals)
                    if ''.join(c for c, d in zip(sequence, mask) if not d) != selected['edited_sequence']:
                        raise ValueError('Actual deletion mapping does not reconstruct final sequence')
                if method in HEADS:
                    if len(pool) != 8 or len({r['deletion_mask_sha256'] for r in pool}) != 8:
                        raise ValueError('Head-family final candidate pool is not unique K8')
                    choose = min(pool, key=lambda r: (-r['scorer_score'], tuple(map(tuple, json.loads(r['deleted_intervals'])))))
                    if choose['deletion_mask_sha256'] != selected['deletion_mask_sha256']:
                        raise ValueError('Actual scientific Scorer selector differs')
                    if sha(directory/'actual_cold_WT_cache.pt') != value['stages']['actual_WT_cache_sha256']:
                        raise ValueError('Timed WT cache bytes changed')
                elif method == 'scisor_L_native':
                    if len(pool) != 1 or value['components']['native_K'] != 1:
                        raise ValueError('Native L must remain K1')
                else:
                    if len(pool) != 500 or value['components']['candidates_generated'] != 500 or value['components']['candidates_scored'] != 500:
                        raise ValueError('Native Raygun all500 generation/scoring coverage differs')
                    if not all(math.isfinite(r['native_score']) for r in pool):
                        raise ValueError('A native score is nonfinite')
                    order = sorted(range(500), key=lambda i: (-pool[i]['native_score'], i))
                    if value['components']['retained_top50_indices'] != order[:50] or selected['candidate_index'] != order[0]:
                        raise ValueError('Native top50/top1 selection differs')
                    if any(hashlib.sha256(r['edited_sequence'].encode()).hexdigest() != r['edited_sequence_sha256'] for r in pool):
                        raise ValueError('A raw native audit-pool sequence changed')
                if method not in HEADS and sha(directory/'actual_cold_native_WT_cache.pt') != value['stages']['native_input_cache_sha256']:
                    raise ValueError('Native WT input cache changed')
                if scenario == 'cold' and method in ['rag_unlimited', 'gap_frequency']:
                    search = value['stages']['uncached_retrieval']
                    if len(search['commands']) != 3 or sha(directory/'cold_uncached_search/hits.tsv') != search['hits_sha256']:
                        raise ValueError('Actual uncached search/alignment evidence differs')
                    profile = read(directory/'cold_uncached_search/reviewed_profile.json')
                    if digest(profile) != ident['actual_reviewed_profile_sha256']:
                        raise ValueError('Actual reviewed homology profile differs')
                    search_count += 1
                devices.add(ident['CUDA_VISIBLE_DEVICES']);nodes.add(ident['SLURMD_NODENAME'])
                seconds = value['parent_observed_request_seconds']
                if not math.isfinite(seconds) or seconds <= 0 or seconds < value['internal_request_seconds']:
                    raise ValueError('Real request latency boundary is invalid')
                components, stages = value['components'], value['stages']
                rows.append(dict(protein_id=pid, method=method, scenario=scenario, budget=beta,
                    request_seconds=seconds, internal_seconds=value['internal_request_seconds'],
                    model_load_seconds=stages.get('model_setup_seconds', 0.),
                    retrieval_seconds=stages.get('uncached_retrieval', {}).get('measured_uncached_query_total_seconds', 0.),
                    input_encoding_or_tokenization_seconds=stages.get('WT_encoding_transfer_and_cache_seconds',
                        components.get('WT_encoding_seconds', components.get('input_tokenization_seconds', 0.))),
                    policy_logits_seconds=components.get('policy_logits_seconds', 0.),
                    generation_seconds=components['generation_seconds'], scoring_seconds=components['scoring_seconds'],
                    final_output_serialization_seconds=stages['final_output_serialization_seconds'],
                    peak_GPU_allocated_GiB=value['peak_GPU_allocated_bytes']/2**30,
                    peak_GPU_reserved_GiB=value['peak_GPU_reserved_bytes']/2**30,
                    final_sequence_sha256=actual_sha, sequence_matches_scientific_output=match))
                identities.append(dict(protein_id=pid, method=method, scenario=scenario,
                    output_identity=ident, output_files_sha256=value['output_files_sha256']))
            if read(directory/'cold/request_identity.json')['actual_WT_cache_sha256'] != read(directory/'cached/request_identity.json')['actual_WT_cache_sha256']:
                raise ValueError('The further request did not reuse the actual first WT representation')
    if len(rows) != 240 or search_count != 40 or len(devices) != 1 or len(nodes) != 1:
        raise ValueError('Full request/scenario/search/same-card measurement coverage differs')
    summaries = []
    fields = [k for k in rows[0] if k.endswith('_seconds') or k.endswith('_GiB')]
    for method in frozen['methods']:
        for scenario in ['cold', 'cached']:
            group = [r for r in rows if r['method'] == method and r['scenario'] == scenario]
            for field in fields:
                summaries.append(dict(method=method, scenario=scenario, measure=field, **stats([r[field] for r in group])))
    for name, data in [('complete_per_request.csv', rows), ('descriptive_summary.csv', summaries)]:
        with (out/name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write(out/'complete_per_request.json', rows)
    write(out/'descriptive_summary.json', summaries)
    write(out/'all_request_identity_audit.json', identities)
    lookup = {(r['method'], r['scenario'], r['measure']): r for r in summaries}
    text = ['# 固定20 WT六方法的真实端到端请求计时', '',
        '全部120个独立方法进程及240个实际请求收齐。首次请求预算20%；同一WT在同一已加载进程内的进一步请求预算30%。每个方法/场景均为20 WT。',
        '', '|方法|首次请求均值/中位数 秒|进一步请求均值/中位数 秒|首次Torch峰值显存均值 GiB|', '|---|---:|---:|---:|']
    for method in frozen['methods']:
        cold = lookup[method, 'cold', 'request_seconds'];warm = lookup[method, 'cached', 'request_seconds']
        memory = lookup[method, 'cold', 'peak_GPU_allocated_GiB']
        text.append('|'+method+'|'+f"{cold['mean']:.4f} / {cold['median']:.4f}"+'|'+f"{warm['mean']:.4f} / {warm['median']:.4f}"+'|'+f"{memory['mean']:.3f}"+'|')
    text += ['', '## 请求与计时边界', '',
        '首次请求从真实子进程启动计到最终输出完成事件，包括imports、模型载入、必要的未缓存查询createdb/search/convertalis、编码、规划/生成、评分和候选/最终输出审计写入。父进程同机monotonic直接测量，不由E8组件相加。进一步请求从父进程发出请求信号计到实际输出，两个请求间等待不计入。',
        '模型、目标数据库和既有来源审核元数据为就绪资产；目标数据库离线构建不重复，OS文件缓存可能已热。完整部署校验另列，Slurm排队不计入方法服务时间。全部方法使用同一分配A100、同一节点；Torch显存指标不包含驱动上下文。',
        '首次与进一步请求预算不同，二者描述真实使用场景，差值或比值不解释为仅由缓存导致的因果加速。L保留原生K1，Raygun每请求生成并评分500、原生top50/top1选择；完整500池作为内部审计记录，审计输出写入计入本次延迟。',
        'DelePref/RAG/gap/Terminal保留原科学输入导出的填充宽度；每个实际新WT表征逐位核对原科学tensor。原encoder/Policy/Scorer与原生外部方法未改变。L缓存模型和输入tokens，Raygun缓存实际WT表征；不虚构原生接口不支持的latent缓存。',
        '', '## 一致性与完整覆盖', '',
        '240个实际最终序列中，与各自冻结科学产物完全相同的数量：'+str(actual_outputs_match)+'。全部匹配与不匹配均保留在逐请求CSV中，未按速度、赢家或结果挑选运行。',
        '40次实际未缓存同源查询及其alignment/profile身份均核验。全部输入、模型/来源/候选池、最终序列和实际cache身份绑定到原始输出；本实验没有新增结构预测或训练。',
        '完整组件、四分位数、范围、显存和逐WT数据见descriptive_summary.csv与complete_per_request.csv；源码/源freeze、全量输出SHA及核验结果见completion.json。']
    (out/'E10_result_zh.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    names = ['complete_per_request.csv', 'descriptive_summary.csv', 'complete_per_request.json',
        'descriptive_summary.json', 'all_request_identity_audit.json', 'E10_result_zh.md']
    write(out/'completion.json', dict(status='E10_ALL240_REAL_REQUEST_TIMING_AND_IDENTITY_READOUT_COMPLETE',
        WT=20, methods=6, scenarios=2, actual_method_processes=120, actual_requests=240,
        actual_uncached_homology_queries=40, actual_sequences_matching_scientific_reference=actual_outputs_match,
        all_outputs_included=True, same_node=list(nodes), same_CUDA_visible_devices=list(devices),
        new_model_forwards=0, new_structure_predictions=0, analysis_source_freeze_sha256=checksum,
        timing_freeze_sha256=sha(root/TIMING_FREEZE), measurement_completion_sha256=sha(root/BASE/'measurement_completion.json'),
        outputs={name: sha(out/name) for name in names}, job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(read(out/'completion.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true');parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    prepare(args.root) if args.prepare else run(args.root, args.freeze_sha256)
