"""Release only identity-checked, positively reviewed ordinary protein tasks."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path('reports/recomb_benign_baselines_20261017')
FREEZE = Path('configs/recomb_benign_baselines_freeze_20261017.json')


def validate_tasks(tasks, roster):
    allowed = {p['protein_id']: p['sequence_sha256'] for p in roster}
    if not tasks:
        raise ValueError('empty task list')
    for task in tasks:
        pid = task['protein_id']
        digest = hashlib.sha256(task['sequence'].encode()).hexdigest()
        if pid not in allowed or digest != allowed[pid]:
            raise ValueError('task not in exact positively reviewed roster')
    return True


def main(args):
    import recomb_general100_common_20261017 as common
    root = args.root
    if common.sha(root / FREEZE) != args.freeze_sha256:
        raise ValueError('scope freeze changed')
    frozen = json.loads((root / FREEZE).read_text())
    for path, expected in frozen['files'].items():
        if common.sha(root / path) != expected:
            raise ValueError('scope input/source changed: ' + path)
    parent = common.verify(root, frozen['parent_freeze_sha256'])
    roster = json.loads((root / 'reports/recomb_benign_scope_20261017/general_panel.json').read_text())
    tasks = json.loads((root / BASE / (args.method + '_tasks.json')).read_text())
    validate_tasks(tasks, roster)
    common.require_gpu()
    import recomb_general100_baselines_20261017 as baseline
    baseline.BASE = BASE
    print(json.dumps(dict(event='BENIGN_SCOPE_GATE_PASSED', method=args.method,
                          scope_WT=len(roster), tasks=len(tasks), scope_sha256=args.freeze_sha256)), flush=True)
    if args.method == 'raygun':
        baseline.run_raygun(root, parent, args.shard_index, args.shards)
    else:
        baseline.run_scisor(root, parent)
        report_path = root / BASE / 'scisor/completion.json'
        report = json.loads(report_path.read_text())
        report.update(actual_new_sampler_calls=sum(t['candidate_count'] - (t['mode'] == 'same_scorer_K8') for t in tasks),
                      native_conditions=sum(t['mode'] == 'native' for t in tasks),
                      K8_conditions=sum(t['mode'] == 'same_scorer_K8' for t in tasks),
                      status='SCREENED_SCISOR_GENERATION_COMPLETE', scope_WT=len(roster))
        common.write(report_path, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    p.add_argument('--method', choices=['raygun', 'scisor'], required=True)
    p.add_argument('--freeze-sha256', required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    main(p.parse_args())
