"""Tests for the fixed, paired D-ranking development trial."""
import concurrent.futures
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import d_ranking_trial as trial


def rows_fixture():
    return [dict(candidate_id=f'{p}_{b}_{g}_{i}', protein_id=p, budget=b,
                 generator=g) for p in ('train1', 'train2')
            for b in trial.BUDGETS for g in trial.SOURCES for i in range(7)]


class RankingTests(unittest.TestCase):
    def test_correct_order_and_gradient(self):
        good = trial.ranking_auxiliary(torch.tensor(.7), torch.tensor(.3), .6, .4)
        a, b = torch.tensor(.3, requires_grad=True), torch.tensor(.7, requires_grad=True)
        bad = trial.ranking_auxiliary(a, b, .6, .4)
        self.assertLess(good.item(), bad.item())
        bad.backward()
        self.assertLess(a.grad.item(), 0)
        self.assertGreater(b.grad.item(), 0)

    def test_ties_keep_batch_denominator_and_zero_gradient(self):
        a = torch.tensor([.2, .2], requires_grad=True)
        b = torch.tensor([.6, .6], requires_grad=True)
        losses = trial.ranking_auxiliary(a, b, torch.tensor([.4, .4]), torch.tensor([.4, .2]))
        self.assertEqual(losses[0].item(), 0)
        self.assertAlmostEqual(losses.mean().item(), losses[1].item()/2)
        losses.mean().backward()
        self.assertEqual(a.grad[0].item(), 0)

    def test_full_coverage_scales_differentiable_T(self):
        logits = torch.tensor([[1., 0., 0., 0., 0.]]*128, requires_grad=True)
        hi = trial.predicted_t(logits, .8)
        lo = trial.predicted_t(logits, .4)
        self.assertAlmostEqual(hi.item(), 2*lo.item())
        self.assertLess(trial.ranking_auxiliary(hi, lo, .64, .32).item(),
                        trial.ranking_auxiliary(lo, hi, .64, .32).item())
        loss = trial.ranking_auxiliary(lo, hi, .64, .32)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)


class SamplerTests(unittest.TestCase):
    def setUp(self):
        self.rows = rows_fixture()
        self.assignment = {'train1': 'train', 'train2': 'train', 'val': 'validation'}
        self.labels = {r['candidate_id']: {'pair_ids': np.arange(10, 300, dtype=np.int64)} for r in self.rows}

    def ledger(self, seed):
        sampler = trial.PairedSampler(self.rows, self.assignment, seed)
        return [sampler.step(self.labels, step)[1] for step in range(1, 5)]

    def test_seed_ledger_same_across_modes_and_external_rng(self):
        expected = self.ledger(20261013)
        torch.manual_seed(2); torch.rand(1024); np.random.seed(45)
        self.assertEqual(expected, self.ledger(20261013))
        self.assertNotEqual(expected, self.ledger(20261014))
        for step in expected:
            self.assertEqual(step['relation_updates'], 1024)
            self.assertEqual(len(step['conditions']), 4)
            for condition in step['conditions']:
                self.assertEqual(len(set(x['candidate_id'] for x in condition['candidates'])), 2)

    def test_reject_validation_and_mixed_condition(self):
        bad = dict(self.rows[0], protein_id='val')
        with self.assertRaisesRegex(ValueError, 'train'):
            trial.PairedSampler([bad], self.assignment, 1)
        with self.assertRaisesRegex(ValueError, 'condition'):
            trial.validate_pair(self.rows[0], self.rows[7], self.assignment)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            trial.validate_pair(self.rows[0], self.rows[0], self.assignment)


class LabelAuditTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(candidate_id='x', protein_id='train1', source='e1_exact', budget=.2)
        self.ref_i = np.array([0, 0, 1, 1]); self.ref_j = np.array([3, 4, 4, 5])
        self.kept = np.array([0, 1, 3, 4])
        self.arrays = dict(kept=self.kept, reference_i=self.ref_i, reference_j=self.ref_j,
            pair_ids=np.array([0, 1, 2]), target=np.array([0, 1, 2]),
            wt_i=np.array([0, 0, 1]), wt_j=np.array([3, 4, 4]),
            edited_i=np.array([0, 0, 1]), edited_j=np.array([2, 3, 3]),
            retained_mask=np.array([True, True, True, False]), unknown_pair_ids=np.array([], dtype=int))
        self.summary = dict(candidate_id='x', protein_id='train1', source='e1_exact', budget=.2,
            reference_pairs=4, retained_pairs=3, known_pairs=3, unknown_pairs=0,
            C=.75, G_lower=.75, G_upper=.75, T_lower=.5625, T_upper=.5625,
            class_counts=[1, 1, 1, 0, 0])

    def check(self):
        return trial.verify_candidate_labels(self.row, self.summary, self.arrays,
                                             self.ref_i, self.ref_j, self.kept)

    def test_full_reference_denominator_and_truth(self):
        verified = self.check()
        self.assertEqual(verified['C'], .75)
        self.assertEqual(verified['T'], .5625)

    def test_unknown_partial_mapping_target_and_identity_rejected(self):
        for key, value in [('unknown_pair_ids', np.array([3])), ('pair_ids', np.array([0, 1, 3])),
                           ('edited_j', np.array([2, 3, 2])), ('target', np.array([0, 1, 5]))]:
            original = self.arrays[key]; self.arrays[key] = value
            with self.assertRaises(ValueError): self.check()
            self.arrays[key] = original
        self.summary['protein_id'] = 'val'
        with self.assertRaisesRegex(ValueError, 'identity'): self.check()


@unittest.skipUnless(trial.AUTHORIZED_ROOT.is_dir(), 'requires the sealed Linux CPU snapshot')
class PinnedInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = trial.load_runtime(trial.AUTHORIZED_ROOT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.path = self.root/'caches/embeddings/a.npy'; self.path.parent.mkdir(parents=True)
        self.sequence = 'AC'; self.protocol = 'f'*64
        np.save(self.path, np.zeros((2, 1152), dtype=np.float32))
        self.key = trial.hashlib.sha256(self.sequence.encode()).hexdigest()
        self.meta = dict(sequence_sha256=self.key, protocol_sha256=self.protocol,
                         output_sha256=trial.sha_file(self.path), length=2)
        trial.write_json(self.path.with_suffix('.json'), self.meta)
        self.index = self.root/'index.jsonl'
        self.index.write_text(trial.canonical(dict(key='fixture', path=str(self.path),
            sequence_sha256=self.key, protocol_sha256=self.protocol, length=2))+'\n')
        pins = {p.relative_to(self.root).as_posix(): trial.sha_file(p)
                for p in (self.path, self.path.with_suffix('.json'))}
        self.audit = dict(asset_hashes=pins, assets_sha256=trial.identity(pins))

    def tearDown(self): self.temp.cleanup()

    def inputs(self):
        return trial.make_pinned_inputs(self.runtime['train'].Inputs, [self.index],
                                       self.protocol, self.root, self.audit)

    def mutate_both_files(self):
        np.save(self.path, np.ones((2, 1152), dtype=np.float32))
        self.meta['output_sha256'] = trial.sha_file(self.path)
        trial.write_json(self.path.with_suffix('.json'), self.meta)

    def test_unchanged_embedding_and_first_load_only_pin_validation(self):
        inputs = self.inputs()
        with patch.object(trial, 'sha_file', wraps=trial.sha_file) as hasher:
            first = inputs.embedding(self.sequence)
            self.assertEqual(hasher.call_count, 4)
            self.assertIs(inputs.embedding(self.sequence), first)
            self.assertEqual(hasher.call_count, 4)
        self.assertTrue(np.array_equal(first, np.zeros((2, 1152), dtype=np.float32)))

    def test_synchronized_embedding_and_sidecar_mutation_rejected(self):
        self.mutate_both_files()
        # This was the escape: the unchanged original Inputs accepts both
        # rewritten files because their current output_sha256 agrees.
        original = self.runtime['train'].Inputs([self.index], self.protocol)
        self.assertTrue(original.embedding(self.sequence).all())
        with self.assertRaisesRegex(ValueError, 'prepare-time.*pin'):
            self.inputs().embedding(self.sequence)

    def test_missing_pin_rejected(self):
        del self.audit['asset_hashes'][self.path.with_suffix('.json').relative_to(self.root).as_posix()]
        self.audit['assets_sha256'] = trial.identity(self.audit['asset_hashes'])
        with self.assertRaisesRegex(ValueError, 'pin missing'):
            self.inputs().embedding(self.sequence)

    def test_mutation_during_original_load_rejected_and_cache_discarded(self):
        inputs = self.inputs(); original = self.runtime['train'].Inputs.embedding
        def load_then_mutate(instance, sequence):
            result = original(instance, sequence); self.mutate_both_files(); return result
        with patch.object(self.runtime['train'].Inputs, 'embedding', load_then_mutate):
            with self.assertRaisesRegex(ValueError, 'during first load'):
                inputs.embedding(self.sequence)
        self.assertNotIn(self.key, inputs.arrays)
        self.assertNotIn(self.key, inputs.pinned_sequences)

    def test_actual_candidate_original_head_cpu_integration(self):
        root = trial.AUTHORIZED_ROOT
        row = trial.read_jsonl(root/'manifests/development_primary.jsonl')[0]
        protocol_hash = trial.sha_file(root/'configs/protocol_v2.json')
        audit = trial.audit_embeddings(root, [row], protocol_hash)
        inputs = trial.make_pinned_inputs(self.runtime['train'].Inputs,
            [root/f'caches/embeddings/full_shard{s}_index.jsonl' for s in (0, 1)], protocol_hash, root, audit)
        data, ids, total = inputs.candidate(row)
        torch.set_num_threads(4)
        model = self.runtime['model'].PairResponseHead().eval()
        with torch.no_grad():
            prediction = self.runtime['train'].logits(model, data, 'D', 'cpu', np.arange(min(128, len(ids))))
        self.assertEqual(prediction.shape, (min(128, len(ids)), 5))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertGreater(total, len(ids))
        self.assertEqual(len(inputs.pinned_sequences), 2)
        summaries = {r['candidate_id']: r for r in trial.read_jsonl(root/'reports/label_candidates.jsonl')}
        output = self.root/'actual_candidate_evaluation'
        with patch.object(self.runtime['common'], 'AUTHORIZED_ROOT', self.root):
            report = self.runtime['evaluate'].evaluate(model, [row], inputs, summaries,
                root/'labels/pairs', 'D', 'cpu', output)
        self.assertEqual(report['candidates'], 1)
        self.assertTrue(np.isfinite(report['hierarchical_ce']))
        with np.load(output/(row['candidate_id']+'.npz')) as prediction_file:
            self.assertTrue(np.array_equal(prediction_file['pair_ids'], ids))
            self.assertEqual(prediction_file['probabilities'].shape, (len(ids), 5))


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_claim_race_and_array_owner(self):
        def claim(_):
            try: trial.claim_task(self.root, 'a'*64, 0); return True
            except FileExistsError: return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(claim, range(16))), 1)
        trial.claim_array(self.root, 'a'*64, '123')
        trial.claim_array(self.root, 'a'*64, '123')
        with self.assertRaises(RuntimeError): trial.claim_array(self.root, 'a'*64, '456')

    def test_existing_output_is_never_reused(self):
        path = self.root/'output'; trial.fresh_output(path)
        with self.assertRaises(FileExistsError): trial.fresh_output(path)

    def test_fixed_nine_task_inventory(self):
        tasks = trial.task_inventory()
        self.assertEqual(len(tasks), 9)
        self.assertEqual(len({r['output'] for r in tasks}), 9)
        self.assertEqual({(r['mode'], r['seed']) for r in tasks},
                         {(mode, seed) for mode in trial.MODES for seed in trial.SEEDS})

    def test_submission_calls_sbatch_once_and_never_retries_unknown(self):
        checksum = 'b'*64
        fake = {'tasks': trial.task_inventory(), 'preflight': {'verified': True, 'embeddings': {'sealed': True}}}
        with patch.object(trial, 'AUTHORIZED_ROOT', self.root), patch.object(trial, 'checked_freeze', return_value=fake), \
             patch.object(trial, 'preflight', return_value={'audit': fake['preflight']}) as preflight, \
             patch.object(trial.subprocess, 'run', return_value=SimpleNamespace(stdout='12345\n')) as sbatch:
            first = trial.submit_array(self.root, checksum)
            again = trial.submit_array(self.root, checksum)
        self.assertEqual(first, again); self.assertEqual(sbatch.call_count, 1)
        preflight.assert_called_once_with(self.root.resolve(), sealed_embeddings=fake['preflight']['embeddings'])
        command = sbatch.call_args.args[0]
        for value in ('--array=0-8%2', '--time=00:45:00', '--cpus-per-task=4', '--mem=48G', '--no-requeue'):
            self.assertIn(value, command)
        with patch.object(trial, 'AUTHORIZED_ROOT', self.root), patch.object(trial, 'checked_freeze', return_value=fake), \
             patch.object(trial, 'preflight', return_value={'audit': fake['preflight']}), \
             patch.object(trial.subprocess, 'run', return_value=SimpleNamespace(stdout='unknown')) as sbatch:
            with self.assertRaises(ValueError): trial.submit_array(self.root, 'c'*64)
            result = trial.submit_array(self.root, 'c'*64)
        self.assertEqual(result['status'], 'SUBMISSION_ERROR_OR_UNKNOWN_NO_AUTOMATIC_RETRY')
        self.assertEqual(sbatch.call_count, 1)

    def test_fast_task_waits_for_matching_submission_receipt(self):
        checksum = 'd'*64; directory = trial.ledger_directory(self.root, checksum)
        trial.exclusive_json(directory/'submission.claim', {'freeze_sha256': checksum})
        trial.write_json(directory/'submission.json', {'status': 'SUBMISSION_ATTEMPT_STARTED'})
        final = dict(status='D_RANKING_TRIAL_SUBMITTED', freeze_sha256=checksum, job_id='123')
        timer = threading.Timer(.05, lambda: trial.write_json(directory/'submission.json', final))
        timer.start()
        try: self.assertEqual(trial.submission_receipt(self.root, checksum, '123', timeout=1.), final)
        finally: timer.join()
        with self.assertRaises(ValueError): trial.submission_receipt(self.root, checksum, '456', timeout=0.)


class SlurmResourceTests(unittest.TestCase):
    def fields(self, name='TRES', memory='49152M', gpu='gres/gpu=1'):
        return dict(Partition='gpu_gub', TimeLimit='00:45:00', NumCPUs='4', NumNodes='1', Requeue='0',
                    **{name: f'cpu=4,mem={memory},node=1,billing=4,{gpu}'})

    def test_modern_and_legacy_tres_memory_units(self):
        trial.validate_slurm_resources(self.fields())
        trial.validate_slurm_resources(self.fields('ReqTRES', '48G'))
        trial.validate_slurm_resources(self.fields('ReqTRES', '50331648K', 'gres/gpu:NVIDIAA40=1'))
        trial.validate_slurm_resources(self.fields('TRES', '0.046875T', 'gres/gpu=1,gres/gpu:NVIDIAA40=1'))

    def test_wrong_memory_gpu_cpu_or_node_rejected(self):
        for memory in ('47G', '49153M', '48K', '48GB'):
            with self.subTest(memory=memory), self.assertRaises(ValueError):
                trial.validate_slurm_resources(self.fields(memory=memory))
        for gpu in ('gres/gpu=2', 'gres/gpu:NVIDIAA40=2', 'gres/gpu=1,gres/gpu:NVIDIAA40=2',
                    'gres/gpu:NVIDIAA40=1,gres/gpu:NVIDIAA100=1', 'gres/nic=1'):
            with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                trial.validate_slurm_resources(self.fields(gpu=gpu))
        for field, value in [('NumCPUs', '8'), ('NumNodes', '2'), ('TimeLimit', '01:00:00'), ('Requeue', '1')]:
            fields = self.fields(); fields[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): trial.validate_slurm_resources(fields)
        fields = self.fields(); fields['TRES'] = fields['TRES'].replace('cpu=4', 'cpu=8')
        with self.assertRaises(ValueError): trial.validate_slurm_resources(fields)


if __name__ == '__main__': unittest.main()
