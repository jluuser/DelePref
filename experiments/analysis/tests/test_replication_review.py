"""Small artifact fixtures exercise the CPU audit without inference or torch."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replication_review as review


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(v) + '\n' for v in values))


class ReviewFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / 'reports/replication_review'
        self.addCleanup(patch.stopall)
        patch.object(review.common, 'AUTHORIZED_ROOT', self.root).start()
        self.valid = [dict(candidate_id=f'{wt}_{i}', protein_id=wt, budget=.2, generator='g')
                      for wt in ('w0', 'w1') for i in range(2)]
        self.labels = {r['candidate_id']: dict(retained_mask=np.array([True, True, False]),
                         pair_ids=np.array([0]), target=np.array([0]), unknown_pair_ids=np.array([1])) for r in self.valid}
        self.summaries = {r['candidate_id']: dict(protein_id=r['protein_id'], budget=.2,
            reference_pairs=3, retained_pairs=2, known_pairs=1, unknown_pairs=1,
            C=2/3, G_lower=.5, G_upper=1., T_lower=1/3, T_upper=2/3) for r in self.valid}
        self.context = dict(root=self.root, output=self.output, valid=self.valid, summaries=self.summaries,
            roster={'w0': 'c0', 'w1': 'c1'}, profiles=['profile'], snapshot=self.root,
            paths={'labels': self.root / 'labels/pairs'}, identities={
                'protocol': 'p', 'manifest': 'm', 'splits': 's', 'label_summary_sha256': 'l',
                'cache_index_sha256': {str(self.root / 'caches/i'): 'cache'}},
            freeze={'code_hashes': {'train.py': 'train'}, 'tasks': []},
            first_seed_predictions={}, protocol={'train': {'max_steps': 3000, 'eval_every': 100, 'patience': 5},
                      'statistics': {'confidence': .95}}, learned={'selected': {}, 'runs': []})

    def make_run(self, arm='C', seed=20261014, probabilities=None):
        directory = self.root / f'checkpoints/{arm}_s{seed}_lr0.001'
        run = dict(arm=arm, seed=seed, lr=.001, directory=str(directory))
        config = dict(arm=arm, seed=seed, lr=.001, technical=False, steps=None, device='cuda',
            protocol=str(self.root / 'configs/protocol_v2.json'),
            manifest=str(self.root / 'manifests/development_primary.jsonl'),
            splits=str(self.root / 'manifests/splits.json'),
            label_summary=str(self.root / 'reports/label_candidates.jsonl'), labels=str(self.root / 'labels/pairs'),
            output=str(directory), cache_index=list(self.context['identities']['cache_index_sha256']))
        dump(directory / 'run.json', self.context['identities'] | {'config': config, 'code': {'train.py': 'train'}})
        # Best 100, followed by exactly five stale evaluations: the true frozen cadence.
        lines(directory / 'train_trace.jsonl', [dict(step=i, ce=1.) for i in range(1, 601)])
        p = (np.array([[.8, .05, .05, .05, .05], [.2, .2, .2, .2, .2]])
             if probabilities is None else probabilities)
        first = self.valid[0]
        best = review.candidate_metrics(first, self.summaries[first['candidate_id']], self.labels[first['candidate_id']], p)['ce']
        lines(directory / 'validation_trace.jsonl', [dict(step=100*i, ce=best if i == 1 else best+.1,
                                                       improved=i == 1) for i in range(1, 7)])
        dump(directory / 'status.json', dict(status='DEVELOPMENT_RUN_COMPLETE', arm=arm, best_ce=best,
            steps=600, stopped_by_patience=True, convergence_review_required=arm in ('S', 'B'),
            esmc_profile_hashes=['profile']))
        (directory / 'best.pt').write_bytes(b'opaque checkpoint bytes')
        evaluation = directory / 'validation_100'
        candidates = []
        for row in self.valid:
            cid = row['candidate_id']
            record = review.candidate_metrics(row, self.summaries[cid], self.labels[cid], p)
            candidates.append(record)
            evaluation.mkdir(parents=True, exist_ok=True)
            np.savez(evaluation / (cid + '.npz'), pair_ids=np.array([0, 1]), probabilities=p)
            if seed == review.SEEDS[0]:
                self.context['first_seed_predictions'][(arm, cid)] = dict(
                    kind='prediction', arm=arm, candidate_id=cid, path=str(evaluation / (cid + '.npz')),
                    sha256=review.common.sha_file(evaluation / (cid + '.npz')))
        decisions = review.core.expected_decisions(candidates)
        lines(evaluation / 'candidates.jsonl', candidates)
        lines(evaluation / 'proteins.jsonl', [dict(protein_id=wt, ce=best) for wt in self.context['roster']])
        lines(evaluation / 'decisions.jsonl', decisions)
        dump(evaluation / 'report.json', dict(status='DEVELOPMENT_EVALUATED', arm=arm, candidates=4, proteins=2,
            hierarchical_ce=best, budget_decisions=review.core.budget_decisions(decisions)))
        return run

    def test_pending_and_all_missing_have_no_efficacy(self):
        for arm in review.ARMS:
            for seed in review.SEEDS[1:]:
                self.context['freeze']['tasks'].append(dict(arm=arm, seed=seed, lr=.001,
                    output=f'checkpoints/{arm}_s{seed}_lr0.001'))
            original = dict(arm=arm, seed=review.SEEDS[0], lr=.001, best_ce=.1)
            self.context['learned']['selected'][arm] = original
            self.context['learned']['runs'].append(original)
        # build_review must audit every inventory member and must not bootstrap
        # or create the ensemble while even one is incomplete.
        def pending(context, run, labels):
            return run | {'state': 'PENDING'}
        with patch.object(review, 'read_context', return_value=self.context), \
             patch.object(review.prediction, 'load_labels', return_value=(self.labels, {}, [])), \
             patch.object(review, 'verify_run', side_effect=pending) as verify, \
             patch.object(review, 'ensemble_evaluation') as ensemble, \
             patch.object(review.core, 'load_statistics') as statistics:
            report = review.build_review(root=self.root, freeze=self.root / 'freeze', output=self.output)
        self.assertEqual(verify.call_count, 15)
        self.assertEqual(report['status'], review.WAITING)
        self.assertEqual(report['completed_runs'], 0)
        self.assertNotIn('per_seed', report)
        self.assertNotIn('ensemble', report)
        ensemble.assert_not_called()
        statistics.assert_not_called()
        self.assertFalse(report['statistics']['seeds_are_independent_WT_units'])
        self.assertEqual(report['statistics']['n_wts'], 2)

    def test_completed_run_and_checkpoint_seal(self):
        run = self.make_run()
        observed = review.verify_run(self.context, run, self.labels)
        self.assertEqual(observed['best_step'], 100)
        self.assertEqual(observed['steps'], 600)
        self.assertNotIn('train_trace', observed)
        sealed = review.public_run(observed)
        (Path(run['directory']) / 'best.pt').write_bytes(b'changed bytes')
        with self.assertRaisesRegex(ValueError, 'checkpoint hash'):
            review.verify_run(self.context, run | {'_sealed': sealed}, self.labels)

    def test_direct_pending_partial_and_corrupt_completed(self):
        directory = self.root / 'checkpoints/C_s20261014_lr0.001'
        run = dict(arm='C', seed=20261014, lr=.001, directory=str(directory))
        self.assertEqual(review.verify_run(self.context, run, self.labels)['state'], 'PENDING')
        directory.mkdir(parents=True)
        self.assertEqual(review.verify_run(self.context, run, self.labels)['state'], 'INCOMPLETE')
        dump(directory / 'status.json', {'status': 'RUNNING'})
        self.assertEqual(review.verify_run(self.context, run, self.labels)['state'], 'INCOMPLETE')
        run = self.make_run()
        (directory / 'validation_100/report.json').unlink()
        with self.assertRaises(FileNotFoundError):
            review.verify_run(self.context, run, self.labels)

    def test_frozen_trace_cadence_patience_and_earliest_minimum(self):
        run = self.make_run()
        path = Path(run['directory']) / 'validation_trace.jsonl'
        good = review.common.read_jsonl(path)
        changed = copy.deepcopy(good)
        changed[1]['step'] = 201
        lines(path, changed)
        with self.assertRaisesRegex(ValueError, 'cadence'):
            review.verify_run(self.context, run, self.labels)
        changed = copy.deepcopy(good)
        changed[1]['ce'] = changed[0]['ce']
        # Ties are non-improvements and preserve the earliest checkpoint.
        lines(path, changed)
        self.assertEqual(review.verify_run(self.context, run, self.labels)['best_step'], 100)
        changed[1]['improved'] = True
        lines(path, changed)
        with self.assertRaisesRegex(ValueError, 'improvement'):
            review.verify_run(self.context, run, self.labels)

    def test_identity_seed_lr_code_and_output_rejected(self):
        run = self.make_run()
        path = Path(run['directory']) / 'run.json'
        good = json.loads(path.read_text())
        for field, bad in [('seed', 20261015), ('lr', .0003), ('arm', 'A'), ('technical', True), ('steps', 100), ('output', str(self.root))]:
            value = copy.deepcopy(good)
            value['config'][field] = bad
            dump(path, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                review.verify_run(self.context, run, self.labels)
        value = copy.deepcopy(good)
        value['code']['train.py'] = 'changed'
        dump(path, value)
        with self.assertRaisesRegex(ValueError, 'code'):
            review.verify_run(self.context, run, self.labels)

    def test_missing_candidate_or_wt_rejected(self):
        run = self.make_run()
        path = Path(run['directory']) / 'validation_100/candidates.jsonl'
        original = review.common.read_jsonl(path)
        lines(path, original[:-1])
        with self.assertRaisesRegex(ValueError, 'candidate roster'):
            review.verify_run(self.context, run, self.labels)
        lines(path, original)
        path = path.parent / 'proteins.jsonl'
        lines(path, [dict(protein_id='w0', ce=-np.log(.8))])
        with self.assertRaisesRegex(ValueError, 'protein roster'):
            review.verify_run(self.context, run, self.labels)

    def test_prediction_ids_shape_probabilities_and_hash(self):
        path = self.root / 'p.npz'
        lab = self.labels['w0_0']
        p = np.array([[.8, .05, .05, .05, .05], [.2]*5])
        np.savez(path, pair_ids=np.array([0, 1]), probabilities=p)
        _, checksum = review.load_probability(path, lab, self.output)
        for ids, probs in [(np.array([1, 0]), p), (np.array([0., 1.]), p), (np.array([0, 1]), p[:, :4]),
                            (np.array([0, 1]), p * .5), (np.array([0, 1]), p * np.nan)]:
            np.savez(path, pair_ids=ids, probabilities=probs)
            with self.assertRaises(ValueError):
                review.load_probability(path, lab, self.output)
        np.savez(path, pair_ids=np.array([0, 1]), probabilities=p * .5)
        with self.assertRaisesRegex(ValueError, 'hash'):
            review.load_probability(path, lab, self.output, checksum)

    def test_original_seed_probability_edit_preserving_ce_and_ghat_is_rejected(self):
        old = np.array([[.8, 0, 0, 0, .2], [.2, 0, 0, 0, .8]])
        edited = np.array([[.4, 2/15, 0, 0, 7/15]] * 2)
        for row in self.valid:
            cid = row['candidate_id']
            self.labels[cid]['pair_ids'] = np.array([0, 1])
            self.labels[cid]['target'] = np.array([0, 0])
            self.labels[cid]['unknown_pair_ids'] = np.array([], dtype=int)
            self.summaries[cid].update(known_pairs=2, unknown_pairs=0,
                G_lower=1., G_upper=1., T_lower=2/3, T_upper=2/3)
        run = self.make_run(seed=review.SEEDS[0], probabilities=old)
        review.verify_run(self.context, run, self.labels)
        row = self.valid[0]
        cid = row['candidate_id']
        before = review.candidate_metrics(row, self.summaries[cid], self.labels[cid], old)
        after = review.candidate_metrics(row, self.summaries[cid], self.labels[cid], edited)
        self.assertAlmostEqual(before['ce'], after['ce'])
        self.assertAlmostEqual(before['Ghat'], after['Ghat'])
        self.assertNotAlmostEqual(
            review.candidate_metrics(row, self.summaries[cid], self.labels[cid], review.mean_probabilities([old]*3))['ce'],
            review.candidate_metrics(row, self.summaries[cid], self.labels[cid], review.mean_probabilities([old, old, edited]))['ce'])
        path = Path(run['directory']) / 'validation_100' / (cid + '.npz')
        np.savez(path, pair_ids=np.array([0, 1]), probabilities=edited)
        with self.assertRaisesRegex(ValueError, 'prediction hash'):
            review.verify_run(self.context, run, self.labels)

    def test_prior_prediction_ledger_requires_hash_exact_roster_and_paths(self):
        learned = {'selected': {arm: {'validation_output': str(self.root / f'checkpoints/{arm}/validation_100')}
                                for arm in review.ARMS}}
        report_path = Path('reports/cpu_development_diagnostics/prediction/report.json')
        ledger = self.root / report_path.parent / 'artifact_ledger.jsonl'
        rows = [dict(kind='prediction', arm=arm, candidate_id=row['candidate_id'],
                     path=str(Path(learned['selected'][arm]['validation_output']) / (row['candidate_id'] + '.npz')),
                     sha256='a'*64) for arm in review.ARMS for row in self.valid]
        def seal(values):
            lines(ledger, values)
            return {'artifact_sha256': {'artifact_ledger.jsonl': review.common.sha_file(ledger)}}
        report = seal(rows)
        observed, _ = review.sealed_first_seed_predictions(report_path, report, learned, self.valid, self.root, self.output)
        self.assertEqual(len(observed), 20)
        for changed, message in [(rows[:-1], 'roster'), (rows + [rows[0]], 'duplicate'),
                                 (rows + [rows[0] | {'candidate_id': 'extra'}], 'roster'),
                                 ([rows[0] | {'path': str(self.root / 'wrong.npz')}] + rows[1:], 'path')]:
            report = seal(changed)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                review.sealed_first_seed_predictions(report_path, report, learned, self.valid, self.root, self.output)
        report = seal(rows)
        lines(ledger, rows[:-1])
        with self.assertRaisesRegex(ValueError, 'ledger hash'):
            review.sealed_first_seed_predictions(report_path, report, learned, self.valid, self.root, self.output)

    def test_probability_mixture_ce_and_unknown_bounds(self):
        arrays = [np.array([[x, 0, 0, 0, 1-x], [.1, .2, .3, .2, .2]]) for x in (.9, .1, .6)]
        mixture = review.mean_probabilities(arrays)
        row = self.valid[0]
        result = review.candidate_metrics(row, self.summaries[row['candidate_id']], self.labels[row['candidate_id']], mixture)
        self.assertAlmostEqual(result['ce'], -np.log((.9+.1+.6)/3))
        self.assertNotAlmostEqual(result['ce'], np.mean([-np.log(x) for x in (.9, .1, .6)]))
        self.assertAlmostEqual(result['Ghat'], (mixture @ review.prediction.VALUES).mean())
        self.assertEqual((result['G_lower'], result['G_upper']), (.5, 1.))
        self.assertEqual((result['T_lower'], result['T_upper']), (1/3, 2/3))
        self.assertEqual(result['C'], 2/3)
        self.assertAlmostEqual(result['That'], 2/3 * result['Ghat'])

    def test_mixture_top1_can_differ_from_every_seed_winner(self):
        # Each seed's specialist wins individually; consensus wins the mixture.
        scores = [[.95, .05, .05, .6], [.05, .95, .05, .6], [.05, .05, .95, .6]]
        winners = []
        for seed_scores in scores:
            candidates = [dict(candidate_id=str(i), protein_id='w', budget=.2, generator='g',
                That=s, Ghat=s, C=1., G_lower=.1, G_upper=.9, T_lower=.1, T_upper=.9) for i, s in enumerate(seed_scores)]
            winners.append(review.core.expected_decisions(candidates)[0]['selected'])
        averaged = [review.mean_probabilities([np.array([[s[i], 0, 0, 0, 1-s[i]]]) for s in scores])[0, 0] for i in range(4)]
        for i, candidate in enumerate(candidates):
            candidate['That'] = float(averaged[i])
        self.assertEqual(winners, ['0', '1', '2'])
        self.assertEqual(review.core.expected_decisions(candidates)[0]['selected'], '3')

    def test_ensemble_outputs_and_checkpoint_identity_tuple(self):
        runs = [review.verify_run(self.context, self.make_run(seed=s), self.labels) for s in review.SEEDS]
        result = review.ensemble_evaluation(self.context, 'C', runs, self.labels)
        self.assertEqual(result['checkpoint_sha256']['seeds'], list(review.SEEDS))
        self.assertEqual(len(result['checkpoint_sha256']['sha256']), 3)
        self.assertEqual(len(result['_evaluation']['candidates']), 4)
        paired = review.core.paired_inputs({'C': result}, self.context['roster'])
        self.assertEqual({r['wt_id'] for r in paired}, {'w0', 'w1'})
        self.assertEqual(len([r for r in paired if r['endpoint'] == 'CE_overall']), 2)

    def test_all_complete_reports_every_seed_and_five_identical_ensembles(self):
        all_runs = []
        for arm in review.ARMS:
            arm_runs = []
            for seed in review.SEEDS:
                run = review.verify_run(self.context, self.make_run(arm, seed), self.labels)
                all_runs.append(run)
                arm_runs.append(run)
                if seed != review.SEEDS[0]:
                    self.context['freeze']['tasks'].append(dict(arm=arm, seed=seed, lr=.001,
                        output=f'checkpoints/{arm}_s{seed}_lr0.001'))
            original = review.public_run(arm_runs[0])
            self.context['learned']['selected'][arm] = original
            self.context['learned']['runs'].append(original)
        by_key = {(r['arm'], r['seed']): r for r in all_runs}
        observed_calls = []
        def comparisons(rows, roster, statistics, confidence):
            observed_calls.append(rows)
            self.assertEqual(roster, self.context['roster'])
            self.assertEqual({r['method'] for r in rows}, set(review.ARMS))
            self.assertEqual(len([r for r in rows if r['endpoint'] == 'CE_overall']), 10)
            return {comparison: {'fixture': {'n_wts': 2}} for comparison in ('C-B', 'C-S', 'C-A', 'C-D')}
        with patch.object(review, 'read_context', return_value=self.context), \
             patch.object(review.prediction, 'load_labels', return_value=(self.labels, {}, [])), \
             patch.object(review, 'verify_run', side_effect=lambda c, r, l: by_key[(r['arm'], r['seed'])]), \
             patch.object(review.core, 'load_statistics', return_value=object()), \
             patch.object(review.core, 'compare_inputs', side_effect=comparisons):
            report = review.build_review(root=self.root, freeze=self.root / 'freeze', output=self.output)
        self.assertEqual(report['status'], review.READY)
        self.assertEqual(report['completed_runs'], 15)
        self.assertEqual(set(report['per_seed']), {str(s) for s in review.SEEDS})
        self.assertEqual(set(report['ensemble']['arms']), set(review.ARMS))
        self.assertEqual(len(observed_calls), 4)  # three seeds plus one ensemble, never 30 WT
        self.assertIsNone(report['scientific_pass_decision'])
        self.assertFalse(report['confirmation_folding_authorized'])

    def test_repeat_audit_refuses_changed_completed_bytes(self):
        run = review.verify_run(self.context, self.make_run(), self.labels)
        ledger = self.output / 'artifact_ledger.jsonl'
        lines(ledger, run['_prediction_ledger'])
        previous = dict(status=review.WAITING, identities=self.context['identities'],
                        runs=[review.public_run(run)], artifact_ledger_sha256=review.common.sha_file(ledger))
        dump(self.output / 'report.json', previous)
        review.verify_previous_audit(self.context, [run])
        changed = copy.deepcopy(run)
        changed['checkpoint_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
            review.verify_previous_audit(self.context, [changed])
        changed = copy.deepcopy(run)
        changed['_prediction_ledger'][0]['sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'prediction hash'):
            review.verify_previous_audit(self.context, [changed])

    def test_link_and_input_overlap_protection(self):
        self.assertEqual(review.protected_output(self.root, self.output), self.output)
        with self.assertRaises(ValueError):
            review.protected_output(self.root, self.root / 'reports/development_review')
        with self.assertRaises(ValueError):
            review.protected_output(self.root, self.output, [self.output / 'input.json'])
        self.output.mkdir(parents=True)
        target = self.root / 'sealed.json'
        target.write_text('sealed')
        try:
            (self.output / 'report.json').hardlink_to(target)
        except OSError:
            self.skipTest('hard links unavailable')
        with self.assertRaisesRegex(ValueError, 'hard-linked'):
            review.protected_output(self.root, self.output)
        self.assertEqual(target.read_text(), 'sealed')


if __name__ == '__main__':
    unittest.main()
