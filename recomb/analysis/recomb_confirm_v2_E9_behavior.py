"""Complete E9 behavioral endpoints from frozen provenance, without new models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import delepref_unlimited_planning as planning
from recomb_confirm_v2_diagnostics import read, sha, write
from recomb_confirm_v2_E9_completion import ARMS, BASE as INPUT, WINNERS, claim, digest, spans

OUT = Path('reports/recomb_confirm_v2_E9_behavior')


def run(root):
    source = root/'analysis/recomb_confirm_v2_E9_behavior.py'
    frozen = read(root/WINNERS)
    for p, h in frozen['files'].items():
        if sha(root/p) != h:
            raise ValueError('E9 frozen winner/provenance differs: '+p)
    out = root/OUT
    if (out/'completion.json').exists():
        raise FileExistsError('E9 behavioral analysis already completed')
    claim(out/'execution_claim.json', dict(source_sha256=sha(source), winner_freeze_sha256=sha(root/WINNERS)))
    behavior = read(root/INPUT/'behavior.json')
    winners = {(r['protein_id'], r['method']): r for r in read(root/INPUT/'selected_structure_outputs.json')}
    pools = {(r['protein_id'], r['method']): r['rows'] for r in read(root/INPUT/'candidate_pools.json')}
    if len(behavior) != 20 or len(winners) != 40 or len(pools) != 40:
        raise ValueError('Behavioral coverage must remain all20 WT and both arms')
    rows, trial_rows = [], []
    for record in behavior:
        pid = record['protein_id']; seed = record['seed']
        admitted = {arm: record[arm]['admitted_attempt_indices'] for arm in ARMS}
        if admitted[ARMS[0]] != admitted[ARMS[1]]:
            raise ValueError('Paired accepted attempts differ')
        for arm in ARMS:
            winner = winners[pid, arm]
            sequence = winner['wildtype_sequence']; n = len(sequence)
            budget_count = planning.delete_count(n, .2)
            if record['seed_deleted']+record['completion_deleted'] != budget_count:
                raise ValueError('Seed/completion deletion decomposition differs')
            proposed = [r for r in pools[pid, arm] if r['generation_kind'].startswith('retrieval')]
            if len(proposed) != record['paired_retrieval_slots'] or len(pools[pid, arm]) != 8:
                raise ValueError('Realized matched candidate quotas differ')
            for candidate in proposed:
                if digest(candidate['retrieval_seed_spans']) != digest(seed['spans']):
                    raise ValueError('Candidate does not use the frozen common seed')
                mask = planning.validate_plan(n, .2, spans(candidate))
                for a, b in candidate['retrieval_seed_spans']:
                    if not mask[a:b+1].all() or mask[a-1] or mask[b+1]:
                        raise ValueError('Frozen proposal loses a natural seed or flank')
            trials = record['attempt_log']
            if len(trials) != record['attempts']:
                raise ValueError('Attempt denominator differs')
            for trial in trials:
                value = trial['arms'][arm]
                mask = planning.validate_plan(n, .2, value['spans'])
                actual_legal = all(mask[a:b+1].all() and not mask[a-1] and not mask[b+1] for a, b in seed['spans'])
                if bool(value['legal']) != actual_legal:
                    raise ValueError('Attempt legality differs from exact mask/anchor reconstruction')
                trial_rows.append(dict(protein_id=pid, method=arm, attempt_index=trial['attempt'],
                    legal=actual_legal, original8_duplicate=value['original_duplicate'],
                    admitted_duplicate=value['prior_paired_duplicate'], pair_accepted=trial['pair_accepted']))
            own = [r for r in trial_rows if r['protein_id'] == pid and r['method'] == arm]
            adopted = winner['generation_kind'].startswith('retrieval')
            mask = planning.validate_plan(n, .2, spans(winner))
            edited = ''.join(c for c, deleted in zip(sequence, mask) if not deleted)
            if edited != winner['edited_sequence'] or hashlib.sha256(edited.encode()).hexdigest() != winner['edited_sequence_sha256']:
                raise ValueError('Final output identity differs')
            if adopted and digest(winner['retrieval_seed_spans']) != digest(seed['spans']):
                raise ValueError('Selected seed provenance differs')
            rows.append(dict(protein_id=pid, method=arm, WT_length=n, target_delete_count=budget_count,
                frozen_seed_deleted=record['seed_deleted'], required_completion_deleted=record['completion_deleted'],
                seed_fraction_of_target=record['seed_deleted']/budget_count,
                completion_fraction_of_target=record['completion_deleted']/budget_count,
                paired_attempts=record['attempts'], max_attempts=record['max_attempts'],
                legal_attempts=sum(r['legal'] for r in own), legal_rate=sum(r['legal'] for r in own)/len(own),
                original8_duplicate_attempts=sum(r['original8_duplicate'] for r in own),
                admitted_duplicate_attempts=sum(r['admitted_duplicate'] for r in own),
                paired_completion_slots=record['paired_retrieval_slots'], original_policy_slots=record['common_original_policy_slots'],
                shortfall=record['shortfall'], final_adopted_seed_completion=adopted,
                final_generation_kind=winner['generation_kind'], final_length=len(edited),
                final_seed_provenance_deleted=record['seed_deleted'] if adopted else 0,
                final_completion_provenance_deleted=record['completion_deleted'] if adopted else 0,
                final_mask_sha256=winner['deletion_mask_sha256'], final_sequence_sha256=winner['edited_sequence_sha256'],
                final_span_count=winner['span_count'], final_Scorer_extrapolation=winner['scorer_span_extrapolation']))
    summary = []
    for arm in ARMS:
        group = [r for r in rows if r['method'] == arm]
        attempts = sum(r['paired_attempts'] for r in group)
        legal = sum(r['legal_attempts'] for r in group)
        summary.append(dict(method=arm, WT=20, attempted_completions=attempts, legal_completions=legal,
            pooled_legal_rate=legal/attempts, paired_attempts_mean=float(np.mean([r['paired_attempts'] for r in group])),
            paired_attempts_min=min(r['paired_attempts'] for r in group), paired_attempts_max=max(r['paired_attempts'] for r in group),
            completion_deleted_mean=float(np.mean([r['required_completion_deleted'] for r in group])),
            seed_deleted_mean=float(np.mean([r['frozen_seed_deleted'] for r in group])),
            selected_seed_completion=sum(r['final_adopted_seed_completion'] for r in group),
            selected_seed_completion_fraction=sum(r['final_adopted_seed_completion'] for r in group)/20,
            paired_shortfall_WT=sum(r['shortfall'] > 0 for r in group),
            full_K8_WT=20, fixed_original4_and_completion4_WT=sum(r['original_policy_slots'] == 4 and r['paired_completion_slots'] == 4 for r in group)))
    for name, value in [('per_WT_behavior.json', rows), ('all_attempts.json', trial_rows), ('behavior_summary.json', summary)]:
        write(out/name, value)
    for name, value in [('per_WT_behavior.csv', rows), ('all_attempts.csv', trial_rows), ('behavior_summary.csv', summary)]:
        with (out/name).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(value[0]))
            writer.writeheader();writer.writerows(value)
    text = ['# 相同种子补全对照：行为端点', '',
        '固定20个WT，来源、种子和锚点在新结构结果前冻结；两臂共用尝试槽、原Policy候选及同一Scorer。',
        '', '|补全规则|WT|尝试数|合法率|平均补全删除数|最终采用种子补全|完整4+4|', '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        text.append('|'+r['method']+'|20|'+str(r['attempted_completions'])+'|'+f"{r['pooled_legal_rate']:.3f}"+'|'+f"{r['completion_deleted_mean']:.2f}"+'|'+str(r['selected_seed_completion'])+'/20|'+str(r['fixed_original4_and_completion4_WT'])+'/20|')
    text += ['', '合法率是满足预算及锚点的行为指标，不是结构成功率。均匀合法补全在可行自由集合中抽样，其合法性由约束构造保证。',
        '两臂共同入池只使用合法性与mask去重，不读取Scorer或结构成绩；共享拒绝可使实际尝试数高于单臂独立去重，故保留每次尝试的双臂状态。',
        '来源贡献严格来自赢家携带的retrieval_seed_spans；Policy来源赢家即使偶然覆盖同源缺失位置，也不计为同源来源贡献。',
        '完整逐WT、每次尝试、种子删除量与新增删除量，以及最终采用和身份均见同目录CSV/JSON。结构质量须等待全部缺失预测与原指标评价完成。']
    (out/'behavior_result_zh.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    names = ['per_WT_behavior.json', 'all_attempts.json', 'behavior_summary.json',
        'per_WT_behavior.csv', 'all_attempts.csv', 'behavior_summary.csv', 'behavior_result_zh.md']
    sources = {str(WINNERS): sha(root/WINNERS), 'analysis/recomb_confirm_v2_E9_behavior.py': sha(source)}
    sources.update({str(INPUT/n): sha(root/INPUT/n) for n in ['completion.json', 'behavior.json', 'candidate_pools.json', 'selected_structure_outputs.json']})
    write(out/'completion.json', dict(status='E9_ALL20_BEHAVIOR_ENDPOINTS_COMPLETE_NOT_STRUCTURE_QUALITY_COMPLETION',
        WT=20, arms=2, rows=40, trial_rows=len(trial_rows), new_model_calls=0, new_structure_predictions=0,
        sources_sha256=sources, outputs={n: sha(out/n) for n in names}))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
