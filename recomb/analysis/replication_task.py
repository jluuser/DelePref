"""Validate a prospective seed task before launching unchanged frozen training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from common import AUTHORIZED_ROOT, confined_path, sha_file, write_json


def first_seed_identities(root):
    learned = json.loads((root / 'reports/development_review/report.json').read_text())
    expected_runs = {(arm, lr, 20261013) for arm in 'SABCD' for lr in (.0003, .001)}
    runs = learned['runs']
    if (learned['status'] != 'DEVELOPMENT_REVIEW_READY_FOR_HUMAN_OR_AGENT_EVIDENCE_REVIEW'
            or learned['expected_runs'] != 10 or learned['completed_runs'] != 10 or len(runs) != 10
            or {(r['arm'], r['lr'], r['seed']) for r in runs} != expected_runs
            or any(r['state'] != 'COMPLETE' for r in runs)):
        raise ValueError('first-seed review inventory is incomplete')
    if set(learned['selected']) != set('SABCD') or any(v['lr'] != .001 for v in learned['selected'].values()):
        raise ValueError('first-seed selected learning rates differ')
    if len(learned['validation_wt_components']) != 10 or learned['frozen_validation_candidates'] != 420:
        raise ValueError('first-seed validation roster differs')
    seal_path = root / 'configs/execution_freeze.json'
    seal = json.loads(seal_path.read_text())
    fields = {'protocol': 'configs/protocol_v2.json', 'manifest': 'manifests/development_primary.jsonl',
              'splits': 'manifests/splits.json', 'label_summary_sha256': 'reports/label_candidates.jsonl'}
    identities = {}
    for field, relative in fields.items():
        value = sha_file(root / relative)
        if value != seal['input_hashes'].get(relative) or value != learned['identities'].get(field):
            raise ValueError('training input differs from first-seed evidence: ' + relative)
        identities[field] = value
    dispatch_path = root / 'reports/dispatch_after_pilot.json'
    dispatch = json.loads(dispatch_path.read_text())
    identities.update(core_review_sha256=sha_file(root / 'reports/development_review/report.json'),
                      execution_freeze_sha256=sha_file(seal_path), dispatch_sha256=sha_file(dispatch_path),
                      diagnostic_config_sha256=sha_file(root / 'configs/development_diagnostics_first_seed.json'),
                      frozen_code_hashes=dispatch['code_hashes'],
                      cache_index_sha256={str(root / f'caches/embeddings/full_shard{s}_index.jsonl'):
                          sha_file(root / f'caches/embeddings/full_shard{s}_index.jsonl') for s in (0, 1)})
    for key in ('execution_freeze_sha256', 'dispatch_sha256', 'cache_index_sha256'):
        if learned['identities'][key] != identities[key]:
            raise ValueError('first-seed sealed identity changed: ' + key)
    if learned['identities']['code_hashes'] != dispatch['code_hashes']:
        raise ValueError('first-seed computational identity differs')
    return learned, identities


def checked_task(root, freeze_path, task_index, expected_freeze_hash):
    root = Path(root).resolve()
    freeze_path = Path(freeze_path).resolve()
    if root != AUTHORIZED_ROOT.resolve() or freeze_path != root / 'configs/prospective_replication_freeze.json':
        raise ValueError('unexpected replication root/freeze path')
    if not expected_freeze_hash or sha_file(freeze_path) != expected_freeze_hash:
        raise ValueError('replication freeze hash differs')
    freeze = json.loads(freeze_path.read_text())
    if freeze['status'] != 'PROSPECTIVE_SEED_REPLICATION_FROZEN':
        raise ValueError('prospective replication is not frozen')
    if freeze['first_seed_inspected'] != 20261013 or freeze['prospective_seeds'] != [20261014, 20261015]:
        raise ValueError('unfrozen seed identities')
    if freeze['implementation_sha256'] != sha_file(__file__):
        raise ValueError('replication launcher implementation changed')
    if freeze['launcher_sha256'] != sha_file(root / 'slurm/prospective_replication.sbatch'):
        raise ValueError('replication Slurm launcher changed')
    if freeze['scientific_scope'] != 'training_seed_stability_only_no_confirmation_folds':
        raise ValueError('replication scope differs')
    required = {'configs/protocol_v2.json', 'configs/execution_freeze.json',
                'configs/development_diagnostics_first_seed.json', 'manifests/development_primary.jsonl',
                'manifests/splits.json', 'reports/label_candidates.jsonl',
                'caches/embeddings/full_shard0_index.jsonl', 'caches/embeddings/full_shard1_index.jsonl',
                'reports/dispatch_after_pilot.json', 'reports/development_review/report.json'}
    if not required.issubset(freeze['input_hashes']):
        raise ValueError('replication input identities are incomplete')
    for relative, checksum in freeze['input_hashes'].items():
        path = root / relative
        if root not in path.resolve().parents or sha_file(path) != checksum:
            raise ValueError('replication input changed: ' + relative)
    learned, evidence_identities = first_seed_identities(root)
    for field in ('prediction_diagnostic', 'decision_diagnostic'):
        evidence = freeze[field]
        path = root / evidence['report']
        if root not in path.resolve().parents or sha_file(path) != evidence['sha256']:
            raise ValueError('CPU diagnostic evidence changed')
        report = json.loads(path.read_text())
        expected_status = ('CPU_PREDICTION_DIAGNOSTICS_COMPLETE' if field == 'prediction_diagnostic'
                           else 'CPU_DECISION_DIAGNOSTICS_COMPLETE')
        if report['status'] != expected_status or evidence['required_status'] != expected_status:
            raise ValueError('CPU diagnostic evidence is incomplete')
        if (report['validation_candidates'] != 420 or len(report['validation_wt_components']) != 10
                or report['seed'] != 20261013 or report['structure_calls'] != 0 or report['PLM_forward_calls'] != 0
                or report['GPU_used'] is not False or report['variant_structure_files_read'] != 0):
            raise ValueError('CPU diagnostic scope or roster differs')
        if report['validation_wt_components'] != learned['validation_wt_components']:
            raise ValueError('CPU diagnostic WT/component roster differs')
        for key, value in evidence_identities.items():
            if report['identities'].get(key) != value:
                raise ValueError('CPU diagnostic first-seed identity changed: ' + key)
    snapshot = Path(freeze['code_snapshot']).resolve()
    dispatch = json.loads((root / 'reports/dispatch_after_pilot.json').read_text())
    if freeze['code_snapshot'] != dispatch['code_snapshot'] or freeze['code_hashes'] != dispatch['code_hashes']:
        raise ValueError('replication snapshot differs from the first seed')
    if root / 'provenance/frozen_code' not in snapshot.parents:
        raise ValueError('snapshot outside experiment provenance')
    actual = {p.relative_to(snapshot).as_posix(): sha_file(p) for p in snapshot.rglob('*.py')}
    if actual != freeze['code_hashes']:
        raise ValueError('frozen training snapshot changed')
    protocol = json.loads((root / 'configs/protocol_v2.json').read_text())
    if protocol['train']['seeds'] != [20261013, 20261014, 20261015]:
        raise ValueError('protocol seed list differs')
    expected = {(arm, seed) for arm in 'SABCD' for seed in (20261014, 20261015)}
    tasks = freeze['tasks']
    observed = {(r['arm'], r['seed']) for r in tasks}
    if len(tasks) != 10 or observed != expected or len({r['output'] for r in tasks}) != 10:
        raise ValueError('prospective arm/seed inventory differs')
    if type(task_index) is not int or not 0 <= task_index < 10:
        raise ValueError('task index outside frozen inventory')
    for row in tasks:
        if row['lr'] != .001 or freeze['selected_learning_rates'].get(row['arm']) != row['lr']:
            raise ValueError('selected LR changed or reopened')
        expected_output = root / f"checkpoints/{row['arm']}_s{row['seed']}_lr0.001"
        if (root / row['output']).resolve() != expected_output:
            raise ValueError('prospective output differs')
    task = tasks[task_index]
    output = confined_path(root / task['output'])
    if output.exists():
        raise ValueError('prospective output already exists; refusing duplicate or overwrite')
    command = [sys.executable, str(snapshot / 'train.py'), '--protocol', str(root / 'configs/protocol_v2.json'),
               '--manifest', str(root / 'manifests/development_primary.jsonl'), '--splits', str(root / 'manifests/splits.json'),
               '--labels', str(root / 'labels/pairs'), '--label-summary', str(root / 'reports/label_candidates.jsonl'),
               '--cache-index', str(root / 'caches/embeddings/full_shard0_index.jsonl'),
               str(root / 'caches/embeddings/full_shard1_index.jsonl'), '--arm', task['arm'],
               '--lr', str(task['lr']), '--seed', str(task['seed']), '--output', str(output)]
    return task, command


def claim_task(root, freeze_hash, task_index):
    """Permanent exclusive launch receipt prevents duplicate arrays racing."""
    if len(freeze_hash) != 64 or any(x not in '0123456789abcdef' for x in freeze_hash):
        raise ValueError('invalid claim identity')
    path = confined_path(root / f'ledger/prospective_replication/{freeze_hash}/{task_index}.claim')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(json.dumps({'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'task_index': task_index}) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    return path


def claim_array(root, freeze_hash, array_id):
    """Bind one immutable freeze to one Slurm array, including direct duplicates."""
    if not str(array_id).isdigit():
        raise ValueError('explicit Slurm array ID required')
    directory = confined_path(root / f'ledger/prospective_replication/{freeze_hash}')
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix='.array-owner-')
    temporary = Path(temporary)
    owner_path = directory / 'array_owner.json'
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(json.dumps({'slurm_array_job_id': str(array_id)}) + '\n')
            handle.flush(); os.fsync(handle.fileno())
        try:
            os.link(temporary, owner_path)
        except FileExistsError:
            pass
        if json.loads(owner_path.read_text())['slurm_array_job_id'] != str(array_id):
            raise RuntimeError('different array already owns this prospective freeze')
    finally:
        temporary.unlink(missing_ok=True)


def submit_array(root, freeze_path, freeze_hash):
    """Persist a one-shot controller claim before sbatch; never retry ambiguity."""
    for index in range(10):
        checked_task(root, freeze_path, index, freeze_hash)
    directory = confined_path(root / f'ledger/prospective_replication/{freeze_hash}')
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / 'submission.claim'
    receipt = directory / 'submission.json'
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return json.loads(receipt.read_text()) if receipt.exists() else {
            'status': 'SUBMISSION_ALREADY_ATTEMPTED_OUTCOME_UNKNOWN', 'freeze_sha256': freeze_hash}
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(freeze_hash + '\n'); handle.flush(); os.fsync(handle.fileno())
    command = ['/opt/gridview/slurm/bin/sbatch', '--parsable', '--array=0-9%2',
               '--gres=gpu:NVIDIAA40:1', '--partition=gpu_gub', '--mem=48G', '--time=02:00:00',
               '--nodes=1', '--ntasks=1', '--cpus-per-task=4',
               '--export=ALL,REPLICATION_FREEZE_SHA256=' + freeze_hash,
               str(root / 'slurm/prospective_replication.sbatch')]
    submission_env = {key: value for key, value in os.environ.items() if not key.startswith('SBATCH_')}
    submission_env['REPLICATION_FREEZE_SHA256'] = freeze_hash
    result = {'status': 'SUBMISSION_ATTEMPT_STARTED', 'freeze_sha256': freeze_hash, 'command': command}
    write_json(receipt, result)
    try:
        response = subprocess.run(command, check=True, text=True, capture_output=True, env=submission_env)
        job_id = response.stdout.strip().split(';')[0]
        if not job_id.isdigit():
            raise ValueError('Slurm outcome cannot be parsed; do not resubmit')
        result.update(status='PROSPECTIVE_REPLICATION_SUBMITTED', job_id=job_id)
    except Exception as error:
        result.update(status='SUBMISSION_ERROR_OR_UNKNOWN_NO_AUTOMATIC_RETRY', error=repr(error))
        write_json(receipt, result)
        raise
    write_json(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--freeze-sha256', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--submit-array', action='store_true')
    args = parser.parse_args()
    if args.submit_array:
        print(json.dumps(submit_array(args.root, args.freeze, args.freeze_sha256)), flush=True)
        return
    if args.execute and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('replication training requires a Slurm allocation')
    task, command = checked_task(args.root, args.freeze, args.task_index, args.freeze_sha256)
    print(json.dumps({'stage': 'PROSPECTIVE_TASK_VALIDATED', 'task': task, 'execute': args.execute}), flush=True)
    if args.execute:
        claim_array(args.root, args.freeze_sha256, os.environ.get('SLURM_ARRAY_JOB_ID', ''))
        claim_task(args.root, args.freeze_sha256, args.task_index)
        write_json(args.root / f'reports/replication_task_launches/{args.task_index}.json',
                   {'status': 'LAUNCH_VALIDATED', 'task': task, 'command': command,
                    'freeze_sha256': args.freeze_sha256, 'slurm_job_id': os.environ['SLURM_JOB_ID']})
        os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
