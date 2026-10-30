"""Native Raygun hardware provenance and completed RNG/context evidence.

Read-only analysis of existing records; no protein model calls or resampling.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from recomb_confirm_v2_diagnostics import read, sha, write

OUT = Path('reports/recomb_confirm_v2_E10_native_fidelity')


def run(root):
    out = root/OUT
    out.mkdir(exist_ok=False)
    timing = read(root/'configs/recomb_confirm_v2_E10_timing_padding_v2_freeze.json')
    context_base = root/'reports/recomb_confirm_v2_E10_ray_context_diagnostic'
    context = read(context_base/'completion.json')
    for name, h in context['outputs'].items():
        if sha(context_base/name) != h:
            raise ValueError('Completed context experiment changed')
    modes = context['modes']
    if len(modes) != 2 or modes[0]['sequence_sha256'] != modes[1]['sequence_sha256']:
        raise ValueError('First-batch context equality is not established')
    if modes[0]['WT_embedding_identity'] != modes[1]['WT_embedding_identity'] or modes[0]['after_generation_rng'] != modes[1]['after_generation_rng']:
        raise ValueError('Context changes an audited input or RNG state')
    noise_base = root/'reports/recomb_confirm_v2_E10_noise_signature'
    arrays, noise_done = {}, {}
    for device in ['A40', 'A100']:
        done = read(noise_base/device/'completion.json')
        if sha(noise_base/device/'noise.npz') != done['artifact_sha256']:
            raise ValueError('Noise signature changed')
        with np.load(noise_base/device/'noise.npz', allow_pickle=False) as z:
            arrays[device] = z['noise']
        if hashlib.sha256(arrays[device].tobytes()).hexdigest() != done['noise_tensor_sha256']:
            raise ValueError('Actual random tensor identity differs')
        noise_done[device] = done
    a, b = (noise_done[d] for d in ['A40', 'A100'])
    for key in ['seed', 'shape', 'torch_version', 'CUDA_build', 'distribution_header_sha256',
                'RNG_state_before_sha256', 'RNG_state_after_sha256']:
        if a[key] != b[key]:
            raise ValueError('Controlled noise comparison differs beyond hardware: '+key)
    if a['RNG_state_before_sha256'] != modes[0]['input_preparation_rng_after']['CUDA'] or b['RNG_state_after_sha256'] != modes[0]['after_generation_rng']['CUDA']:
        raise ValueError('Noise probe does not match native first-batch RNG consumption')
    comparison = dict(seed=a['seed'], shape=a['shape'], torch_version=a['torch_version'], CUDA_build=a['CUDA_build'],
        same_noise_tensor=bool(np.array_equal(arrays['A40'], arrays['A100'])),
        equal_element_fraction=float((arrays['A40'] == arrays['A100']).mean()),
        exactly_equal_rows=[bool(np.array_equal(arrays['A40'][i], arrays['A100'][i])) for i in range(8)],
        same_before_and_after_RNG_states=True, same_distribution_header_sha256=a['distribution_header_sha256'],
        A40_SM=a['GPU_multi_processor_count'], A100_SM=b['GPU_multi_processor_count'])
    tasks_path = root/'reports/recomb_confirm_v2_General100_raygun_v2/raygun_tasks.json'
    tasks = read(tasks_path)
    conditions = root/'reports/recomb_confirm_v2_General100_raygun_v2/raygun'
    rows, sources = [], {tasks_path.relative_to(root).as_posix(): sha(tasks_path)}
    first_case = timing['cases'][0]
    first_comparison = []
    for case in timing['cases']:
        for beta in [.2, .3]:
            index = next(i for i, task in enumerate(tasks) if task['protein_id'] == case['protein_id'] and task['budget'] == beta)
            p = conditions/('condition_%03d.json' % index)
            old = read(p)
            if hashlib.sha256(tasks[index]['sequence'].encode()).hexdigest() != case['WT_sha256']:
                raise ValueError('Historical native WT differs from the fixed input')
            if old['edited_sequence_sha256'] != case['scientific_final_sequences']['raygun_native'][str(beta)]['edited_sequence_sha256']:
                raise ValueError('Historical native condition is not the frozen scientific reference')
            sources[p.relative_to(root).as_posix()] = sha(p)
            rows.append(dict(protein_id=case['protein_id'], budget=beta, original_condition_index=index,
                original_actual_GPU=old['actual_GPU'], original_job_id=old['job_id'], original_generation_seed=old['generation_seed'],
                source_condition_path=p.relative_to(root).as_posix(), source_condition_sha256=sha(p),
                original_raw_pool_sha256=old['raw_sha256'], original_selected_sequence_sha256=old['edited_sequence_sha256']))
            if case['protein_id'] == first_case['protein_id']:
                scenario = 'cold' if beta == .2 else 'cached'
                raw = conditions/('raw_%03d.jsonl.gz' % index)
                if sha(raw) != old['raw_sha256']:
                    raise ValueError('Historical first-case raw pool changed')
                old_pool = [json.loads(line) for line in gzip.open(raw, 'rt')]
                directory = root/'reports/recomb_confirm_v2_E10_timing_padding_v2/requests'/('%02d_%s' % (0, case['WT_sha256'][:12]))/'raygun_native'/scenario
                actual_pool = read(directory/'candidate_pool.json')
                actual = read(directory/'selected_output.json')
                if len(old_pool) != 500 or len(actual_pool) != 500 or actual['budget'] != beta or actual['method'] != 'raygun_native':
                    raise ValueError('Observed comparison is not the two complete native500 pools')
                if actual['seed'] != old['generation_seed']:
                    raise ValueError('Observed native mismatch has a changed seed')
                first_comparison.append(dict(scenario=scenario, beta=beta, original_GPU=old['actual_GPU'], timing_GPU='NVIDIA A100 80GB PCIe',
                    same_raw_candidate_at_index=sum(x['sequence'] == y['edited_sequence'] for x, y in zip(old_pool, actual_pool)),
                    raw_candidates=500, same_selected_sequence=actual['edited_sequence_sha256'] == old['edited_sequence_sha256'],
                    same_input_and_seed=True, original_raw_sha256=sha(raw), actual_raw_sha256=sha(directory/'candidate_pool.json')))
    if len(rows) != 40 or len(first_comparison) != 2:
        raise ValueError('Full historical cost20/both-budget native provenance is missing')
    write(out/'CUDA_noise_comparison.json', comparison)
    write(out/'first_case_raw_pool_comparison.json', first_comparison)
    write(out/'all40_historical_Raygun_generation_hardware.json', rows)
    with (out/'all40_historical_Raygun_generation_hardware.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    text = ['# Raygun计时产物的硬件与调用上下文核对', '',
        '首个固定WT、相同输入/种子：历史20%预算参考来自A40，当前统一A100计时的原始500池有114条逐项相同、赢家不同；历史30%预算参考来自A100，该进一步请求的500池和赢家均完全复现。', '',
        '## 受控核对', '',
        '在同一A100上，原生helper调用方式与计时外层inference_mode各只生成首批8条。两臂WT表征、CUDA RNG进入/退出状态及8条输出完全相同，均复现当前A100计时的首8条。这一批没有发现调用上下文导致的变化。该技术核对共生成16条、不评分、不预测结构。',
        '另在A40和A100上，仅执行原生噪声规格torch.randn_like：float32、8×50×1280、相同seed。两卡使用相同PyTorch '+a['torch_version']+' / CUDA '+a['CUDA_build']+'、同一分布kernel头文件，进入和退出RNG状态也相同，但噪声张量仅'+f"{100*comparison['equal_element_fraction']:.1f}"+'%元素逐位一致；只有前两行完整相同。两个探针均未载入或调用蛋白模型。',
        '', '## 对计时和质量引用的处理', '',
        '相同seed与相同RNG状态不能保证本次原生CUDA采样在A40/A100间逐位一致。历史参考的生成GPU属于产物出处；当前端到端计时统一在同一A100测量，保留其实际产物和对应身份，不更改噪声、打分或选择器来强制匹配。',
        '全部20个WT、两个预算的历史Raygun生成硬件、job和条件SHA见all40_historical_Raygun_generation_hardware.csv。最终逐请求表继续报告所有原科学序列的匹配/不匹配，未丢弃任何案例。不同序列的历史结构质量不直接绑定到新计时产物形成逐序列质量-成本曲线。',
        '原E8、原科学变体和所有主结果保持。DelePref系列的填充宽度问题另有完整20例诊断和实际tensor一致性门槛；本次原生采样核对不改变该门槛。']
    (out/'native_fidelity_zh.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    sources['analysis/recomb_confirm_v2_E10_native_fidelity.py'] = sha(root/'analysis/recomb_confirm_v2_E10_native_fidelity.py')
    for p in [context_base/'completion.json', noise_base/'A40/completion.json', noise_base/'A100/completion.json',
              root/'configs/recomb_confirm_v2_E10_timing_padding_v2_freeze.json']:
        sources[p.relative_to(root).as_posix()] = sha(p)
    names = ['CUDA_noise_comparison.json', 'first_case_raw_pool_comparison.json',
        'all40_historical_Raygun_generation_hardware.json', 'all40_historical_Raygun_generation_hardware.csv', 'native_fidelity_zh.md']
    write(out/'completion.json', dict(status='NATIVE_RAYGUN_HARDWARE_AND_CONTEXT_IDENTITY_AUDIT_COMPLETE_NOT_FULL_TIMING_COMPLETION',
        fixed_WT=20, historical_native_conditions=40, context_diagnostic_generated_sequences=16,
        protein_model_calls_in_this_analysis=0, new_structure_predictions=0, full_timing_unchanged=True,
        sources_sha256=sources, outputs={p: sha(out/p) for p in names}))
    print(json.dumps(dict(noise=comparison, first_case=first_comparison)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
