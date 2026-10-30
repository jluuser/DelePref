import concurrent.futures
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replication_task as task
import common


class ReplicationTaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.patches = [patch.object(task, 'AUTHORIZED_ROOT', self.root), patch.object(common, 'AUTHORIZED_ROOT', self.root)]
        for item in self.patches: item.start()
        self.snapshot = self.root / 'provenance/frozen_code/fixture/code'
        self.snapshot.mkdir(parents=True)
        (self.snapshot / 'train.py').write_text('raise AssertionError("fixture must never execute training")')
        self.code_hashes = {'train.py': common.sha_file(self.snapshot / 'train.py')}
        protocol = {'train': {'seeds': [20261013, 20261014, 20261015]}}
        self.put('configs/protocol_v2.json', protocol)
        self.put('reports/dispatch_after_pilot.json', {'code_snapshot': str(self.snapshot), 'code_hashes': self.code_hashes})
        for path in ('configs/execution_freeze.json', 'configs/development_diagnostics_first_seed.json',
                     'manifests/development_primary.jsonl', 'manifests/splits.json', 'reports/label_candidates.jsonl',
                     'caches/embeddings/full_shard0_index.jsonl', 'caches/embeddings/full_shard1_index.jsonl',
                     'reports/development_review/report.json'):
            self.put(path, {'fixture': path})
        self.put('slurm/prospective_replication.sbatch', {'fixture': 'launcher'})
        fields = {'protocol': 'configs/protocol_v2.json', 'manifest': 'manifests/development_primary.jsonl',
                  'splits': 'manifests/splits.json', 'label_summary_sha256': 'reports/label_candidates.jsonl'}
        identities = {name: common.sha_file(self.root / path) for name, path in fields.items()}
        self.put('configs/execution_freeze.json', {'input_hashes': {path: identities[name] for name, path in fields.items()}})
        identities.update(execution_freeze_sha256=common.sha_file(self.root / 'configs/execution_freeze.json'),
                          dispatch_sha256=common.sha_file(self.root / 'reports/dispatch_after_pilot.json'),
                          cache_index_sha256={str(self.root / f'caches/embeddings/full_shard{s}_index.jsonl'):
                              common.sha_file(self.root / f'caches/embeddings/full_shard{s}_index.jsonl') for s in (0, 1)},
                          code_hashes=self.code_hashes)
        roster = {str(i): str(i) for i in range(10)}
        learned = {'status': 'DEVELOPMENT_REVIEW_READY_FOR_HUMAN_OR_AGENT_EVIDENCE_REVIEW',
                   'expected_runs': 10, 'completed_runs': 10, 'frozen_validation_candidates': 420,
                   'validation_wt_components': roster, 'identities': identities,
                   'runs': [{'arm': arm, 'seed': 20261013, 'lr': lr, 'state': 'COMPLETE'}
                            for arm in 'SABCD' for lr in (.0003, .001)],
                   'selected': {arm: {'lr': .001} for arm in 'SABCD'}}
        self.put('reports/development_review/report.json', learned)
        diagnostic_ids = {k: v for k, v in identities.items() if k != 'code_hashes'}
        diagnostic_ids.update(frozen_code_hashes=self.code_hashes,
                              core_review_sha256=common.sha_file(self.root / 'reports/development_review/report.json'),
                              diagnostic_config_sha256=common.sha_file(self.root / 'configs/development_diagnostics_first_seed.json'))
        evidence = {}
        for kind in ('prediction', 'decision'):
            path = 'reports/cpu_development_diagnostics/' + kind + '/report.json'
            status = f'CPU_{kind.upper()}_DIAGNOSTICS_COMPLETE'
            self.put(path, {'status': status, 'validation_candidates': 420, 'validation_wt_components': roster,
                            'seed': 20261013, 'structure_calls': 0, 'PLM_forward_calls': 0, 'GPU_used': False,
                            'variant_structure_files_read': 0, 'identities': diagnostic_ids})
            evidence[kind + '_diagnostic'] = {'report': path, 'sha256': common.sha_file(self.root / path), 'required_status': status}
        self.freeze = {'status': 'PROSPECTIVE_SEED_REPLICATION_FROZEN', 'first_seed_inspected': 20261013,
                       'prospective_seeds': [20261014, 20261015], 'implementation_sha256': common.sha_file(task.__file__),
                       'launcher_sha256': common.sha_file(self.root / 'slurm/prospective_replication.sbatch'),
                       'scientific_scope': 'training_seed_stability_only_no_confirmation_folds',
                       'code_snapshot': str(self.snapshot), 'code_hashes': self.code_hashes,
                       'selected_learning_rates': {a: .001 for a in 'SABCD'}, **evidence,
                       'tasks': [{'arm': a, 'seed': seed, 'lr': .001,
                                  'output': f'checkpoints/{a}_s{seed}_lr0.001'} for a in 'SABCD' for seed in (20261014, 20261015)]}
        paths = [p for p in self.root.rglob('*') if p.is_file() and 'configs/prospective_replication_freeze.json' not in str(p)]
        self.freeze['input_hashes'] = {p.relative_to(self.root).as_posix(): common.sha_file(p) for p in paths}
        self.path = self.root / 'configs/prospective_replication_freeze.json'

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()

    def put(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def check(self, index=0):
        self.put(self.path, self.freeze)
        return task.checked_task(self.root, self.path, index, common.sha_file(self.path))

    def test_prospective_inventory_and_unchanged_command(self):
        outputs = []
        for index in range(10):
            row, command = self.check(index)
            self.assertEqual(row['lr'], .001)
            self.assertIn(str(self.snapshot / 'train.py'), command)
            self.assertNotIn('--steps', command)
            self.assertNotIn('--technical', command)
            self.assertEqual(command[command.index('--seed') + 1], str(row['seed']))
            outputs.append(row['output'])
        self.assertEqual(len(set(outputs)), 10)

    def test_missing_input_and_modified_identity_rejected(self):
        del self.freeze['input_hashes']['reports/development_review/report.json']
        with self.assertRaisesRegex(ValueError, 'incomplete'): self.check()

    def test_code_and_evidence_change_rejected(self):
        self.check()
        (self.snapshot / 'train.py').write_text('changed')
        with self.assertRaises(ValueError):
            task.checked_task(self.root, self.path, 0, common.sha_file(self.path))

    def test_freshly_rehashed_training_protocol_still_rejected(self):
        path = 'configs/protocol_v2.json'
        self.put(path, {'train': {'seeds': [20261013, 20261014, 20261015], 'max_steps': 10000}})
        self.freeze['input_hashes'][path] = common.sha_file(self.root / path)
        with self.assertRaisesRegex(ValueError, 'differs from first-seed'): self.check()

    def test_rehashed_diagnostic_wrong_original_roster_or_identity_rejected(self):
        path = 'reports/cpu_development_diagnostics/prediction/report.json'
        report = json.loads((self.root / path).read_text())
        report['validation_wt_components']['0'] = 'different_family'
        self.put(path, report)
        checksum = common.sha_file(self.root / path)
        self.freeze['input_hashes'][path] = checksum
        self.freeze['prediction_diagnostic']['sha256'] = checksum
        with self.assertRaisesRegex(ValueError, 'roster differs'): self.check()
        report['validation_wt_components']['0'] = '0'
        report['identities']['protocol'] = 'stale'
        self.put(path, report)
        checksum = common.sha_file(self.root / path)
        self.freeze['input_hashes'][path] = checksum
        self.freeze['prediction_diagnostic']['sha256'] = checksum
        with self.assertRaisesRegex(ValueError, 'first-seed identity'): self.check()

    def test_failed_cpu_diagnostic_cannot_launch(self):
        path = 'reports/cpu_development_diagnostics/prediction/report.json'
        report = json.loads((self.root / path).read_text())
        report['status'] = 'FAILED'
        self.put(path, report)
        checksum = common.sha_file(self.root / path)
        self.freeze['prediction_diagnostic'].update(sha256=checksum, required_status='FAILED')
        self.freeze['input_hashes'][path] = checksum
        with self.assertRaisesRegex(ValueError, 'incomplete'): self.check()

    def test_diagnostic_partial_roster_rejected_even_with_fresh_hashes(self):
        path = 'reports/cpu_development_diagnostics/prediction/report.json'
        report = json.loads((self.root / path).read_text())
        report['validation_candidates'] = 419
        self.put(path, report)
        checksum = common.sha_file(self.root / path)
        self.freeze['prediction_diagnostic']['sha256'] = checksum
        self.freeze['input_hashes'][path] = checksum
        with self.assertRaisesRegex(ValueError, 'scope or roster'): self.check()

    def test_existing_output_and_duplicate_inventory_rejected(self):
        (self.root / self.freeze['tasks'][0]['output']).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'already exists'): self.check()
        self.freeze['tasks'][0] = self.freeze['tasks'][1].copy()
        with self.assertRaisesRegex(ValueError, 'inventory'): self.check()

    def test_lr_and_seed_reopening_rejected(self):
        self.freeze['tasks'][0]['lr'] = .0003
        with self.assertRaisesRegex(ValueError, 'LR'): self.check()
        self.freeze['prospective_seeds'] = [20261013, 20261014]
        with self.assertRaisesRegex(ValueError, 'seed'): self.check()

    def test_concurrent_claims_allow_exactly_one_launch(self):
        def claim(_):
            try:
                task.claim_task(self.root, 'a' * 64, 0)
                return True
            except FileExistsError:
                return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            result = list(pool.map(claim, range(16)))
        self.assertEqual(sum(result), 1)

    def test_one_freeze_cannot_be_owned_by_two_arrays(self):
        task.claim_array(self.root, 'a' * 64, 123)
        task.claim_array(self.root, 'a' * 64, 123)
        with self.assertRaisesRegex(RuntimeError, 'different array'):
            task.claim_array(self.root, 'a' * 64, 456)

    def test_submission_controller_sends_sbatch_once(self):
        self.check()
        checksum = common.sha_file(self.path)
        with patch.object(task.subprocess, 'run', return_value=SimpleNamespace(stdout='12345\n')) as sbatch:
            first = task.submit_array(self.root, self.path, checksum)
            again = task.submit_array(self.root, self.path, checksum)
        self.assertEqual(first['job_id'], '12345')
        self.assertEqual(again, first)
        self.assertEqual(sbatch.call_count, 1)

    def test_unknown_submission_outcome_is_not_retried(self):
        self.check()
        checksum = common.sha_file(self.path)
        with patch.object(task.subprocess, 'run', return_value=SimpleNamespace(stdout='unparseable')) as sbatch:
            with self.assertRaises(ValueError): task.submit_array(self.root, self.path, checksum)
            result = task.submit_array(self.root, self.path, checksum)
        self.assertEqual(result['status'], 'SUBMISSION_ERROR_OR_UNKNOWN_NO_AUTOMATIC_RETRY')
        self.assertEqual(sbatch.call_count, 1)

    def test_inherited_submission_overrides_cannot_change_resource_budget(self):
        self.check()
        checksum = common.sha_file(self.path)
        inherited = {'SBATCH_ARRAY_INX': '0-9%10', 'SBATCH_GRES': 'gpu:8',
                     'SBATCH_TIMELIMIT': '48:00:00', 'REPLICATION_FREEZE_SHA256': 'stale'}
        with patch.dict(task.os.environ, inherited), patch.object(task.subprocess, 'run', return_value=SimpleNamespace(stdout='12345\n')) as sbatch:
            task.submit_array(self.root, self.path, checksum)
        command = sbatch.call_args.args[0]
        environment = sbatch.call_args.kwargs['env']
        self.assertIn('--array=0-9%2', command)
        self.assertIn('--gres=gpu:NVIDIAA40:1', command)
        self.assertIn('--time=02:00:00', command)
        self.assertIn('--cpus-per-task=4', command)
        self.assertFalse(any(key.startswith('SBATCH_') for key in environment))
        self.assertEqual(environment['REPLICATION_FREEZE_SHA256'], checksum)


if __name__ == '__main__': unittest.main()
