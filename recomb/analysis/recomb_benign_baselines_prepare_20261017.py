"""Prepare screened baseline tasks; reuse only exact safe completed conditions."""
import argparse
import json
from pathlib import Path
from recomb_general100_common_20261017 import sha, write, verify
from recomb_benign_baselines_20261017 import BASE, FREEZE, validate_tasks


def prepare(root):
    parent_path = root / 'configs/recomb_general100_baselines_freeze_20261017.json'
    parent_sha = sha(parent_path)
    verify(root, parent_sha)
    out = root / BASE
    out.mkdir(exist_ok=False)
    scope = root / 'reports/recomb_benign_scope_20261017'
    summary = json.loads((scope / 'completion.json').read_text())
    if summary['policy_sha256'] != sha(root / 'configs/recomb_benign_scope_20261017.json'):
        raise ValueError('scope policy changed after curation')
    roster = json.loads((scope / 'general_panel.json').read_text())
    allowed = {r['protein_id'] for r in roster}
    old_base = root / 'reports/recomb_general100_execution_20261017'
    old_tasks = json.loads((old_base / 'raygun_tasks.json').read_text())
    tasks, reused = [], []
    for i, task in enumerate(old_tasks):
        if task['protein_id'] not in allowed:
            continue
        validate_tasks([task], roster)
        path = old_base / 'raygun' / ('condition_%03d.json' % i)
        if path.exists():
            value = json.loads(path.read_text())
            for key in ['protein_id', 'sequence', 'budget', 'target_length']:
                if value[key] != task[key]:
                    raise ValueError('old condition identity mismatch')
            raw = path.with_name('raw_%03d.jsonl.gz' % i)
            if sha(raw) != value['raw_sha256']:
                raise ValueError('completed raw candidate identity mismatch')
            reused.append(dict(protein_id=task['protein_id'], budget=task['budget'],
                result_path=str(path.relative_to(root)), result_sha256=sha(path), raw_sha256=sha(raw)))
        else:
            tasks.append(task)
    scisor = json.loads((scope / 'scisor_eligible_tasks.json').read_text())
    validate_tasks(tasks, roster)
    validate_tasks(scisor, roster)
    write(out / 'raygun_tasks.json', tasks)
    write(out / 'scisor_tasks.json', scisor)
    write(out / 'raygun_reused_conditions.json', reused)
    write(out / 'raygun_checkpoint_load_contract.json', json.loads((old_base / 'raygun_checkpoint_load_contract.json').read_text()))
    paths = ['configs/recomb_benign_scope_20261017.json',
             'reports/recomb_benign_scope_20261017/completion.json',
             'reports/recomb_benign_scope_20261017/general_panel.json',
             'reports/recomb_benign_scope_20261017/general_audit.json',
             'analysis/recomb_benign_baselines_20261017.py',
             'analysis/recomb_benign_baselines_prepare_20261017.py',
             'analysis/recomb_scope_screen_20261017.py',
             'slurm/recomb_benign_baselines_20261017.sbatch']
    paths += [str(BASE / n) for n in ['raygun_tasks.json', 'scisor_tasks.json', 'raygun_reused_conditions.json', 'raygun_checkpoint_load_contract.json']]
    freeze = dict(status='POSITIVE_SCOPE_BASELINE_INPUTS_FROZEN', parent_freeze_sha256=parent_sha,
        files={p: sha(root / p) for p in paths}, retained_WT=len(roster),
        source_general100=True, source_N_not_new_N=True, new_training=False,
        complete_structure_prediction_limit=None, structure_prediction_allowed=False,
        application_and_RAG_jobs_allowed=False, task_GPU_concurrency_limit=2)
    write(root / FREEZE, freeze)
    result = dict(status='SCREENED_BASELINE_PREPARATION_COMPLETE', scope_WT=len(roster),
        raygun_new_conditions=len(tasks), raygun_reused_conditions=len(reused),
        scisor_conditions=len(scisor), freeze_sha256=sha(root / FREEZE),
        excluded_outputs_not_opened_or_reused=True, new_GPU_calls=0)
    write(out / 'preparation_completion.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    prepare(p.parse_args().root)
