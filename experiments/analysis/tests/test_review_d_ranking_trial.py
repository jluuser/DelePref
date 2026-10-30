"""Synthetic CPU artifacts exercise final-only selection, audit, and WT statistics."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import review_d_ranking_trial as review

trial = review.trial
core, diagnostics = review.helpers()
STATISTICS = core.load_statistics(Path(__file__).resolve().parents[2]/'code')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(trial.canonical(r)+'\n' for r in rows))


def embedding_fixture(root):
    path = root/'caches/embeddings/fixture.npy'; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'pinned embedding bytes')
    profile = dict(model='esmc_600m', layer=36, dtype='float32', output_dimension=1152)
    dump(path.with_suffix('.json'), profile | dict(sequence_sha256='sequence', length=4, output_sha256=trial.sha_file(path), gpu='A40', batch_seconds=.1))
    pins = {p.relative_to(root).as_posix(): trial.sha_file(p) for p in (path, path.with_suffix('.json'))}
    return dict(profiles=[trial.identity(profile)], asset_hashes=pins, assets_sha256=trial.identity(pins)), profile


def decision_rows(value, selected='a'):
    return [dict(protein_id=wt, budget=str(b), generator=s, seed=seed, selected=selected,
                 T_lower=value(wt, b, seed), raw_regret_lower=1-value(wt, b, seed))
            for wt in ('w0', 'w1') for b in trial.BUDGETS for s in (*trial.SOURCES, 'mixed') for seed in trial.SEEDS]


class StatisticsTests(unittest.TestCase):
    def test_budgets_then_seeds_inside_wt_bootstrap_and_negative_changes(self):
        left = decision_rows(lambda wt, b, seed: .5+(.03 if wt == 'w0' else -.01)+(seed-trial.SEEDS[0])*.003, 'changed')
        right = decision_rows(lambda wt, b, seed: .5)
        result = review.compare(left, right, {'w0': 'w0', 'w1': 'w1'}, STATISTICS)
        primary = next(r for r in result if r['budget'] == 'all' and r['source'] == 'mixed')
        self.assertAlmostEqual(primary['selected_T']['difference_bounds'][0], .013)
        self.assertAlmostEqual(primary['selected_T']['confidence_interval'][0], -.007)
        self.assertAlmostEqual(primary['regret_difference'], -.013)
        self.assertEqual(primary['top1_changes'], dict(changed=18, good=9, bad=9, same_truth=0, unchanged=0))
        self.assertEqual(primary['selected_T']['n_wts'], 2)
        self.assertEqual(primary['selected_T']['replicates'], 10000)

    def test_all_declared_guards_and_zero_is_not_positive_seed(self):
        def comparison(delta=.02):
            return [dict(budget=b, source=s, selected_T=dict(difference_bounds=[delta, delta], confidence_interval=[.001, .03]))
                    for b in ('all', *map(str, trial.BUDGETS)) for s in (*trial.SOURCES, 'mixed')]
        healthy = comparison(); seeds = [comparison(), comparison(), comparison(0)]
        self.assertEqual(review.continuation(healthy, seeds)['status'], 'CONTINUE_SUPPORTED')
        for key, value, failure in [(('0.2', 'mixed'), -.006, 'mixed_budget_0.2'),
                                    (('all', 'scisor'), -.001, 'source_scisor')]:
            changed = copy.deepcopy(healthy)
            next(r for r in changed if (r['budget'], r['source']) == key)['selected_T']['difference_bounds'] = [value, value]
            result = review.continuation(changed, seeds)
            self.assertEqual(result['status'], 'LIMITED_DIRECTIONAL')
            self.assertTrue(any(failure in name for name in result['failed']))
        result = review.continuation(healthy, [comparison(), comparison(0), comparison(-.01)])
        self.assertIn('positive_seed_count_at_least_2', result['failed'])

    def test_identical_prediction_tie_uses_max_candidate_id(self):
        rows = [dict(candidate_id=cid, protein_id='w0', budget=.2, generator='scisor', That=.4,
                     Ghat=.8, C=.5, G_lower=t, G_upper=t, T_lower=t/2, T_upper=t/2) for cid, t in [('a', .8), ('z', .2)]]
        self.assertTrue(all(r['selected'] == 'z' for r in core.expected_decisions(rows)))
        rows[1]['T_lower'] = rows[1]['T_upper'] = rows[0]['T_lower']
        self.assertTrue(all(r['selected'] == 'z' for r in core.expected_decisions(rows)))

    def test_diagnostic_weighting_does_not_pool_long_candidates(self):
        rows = []
        for wt, count, p in [('w0', 1000, .9), ('w1', 1, .1)]:
            rows.append(dict(protein_id=wt, budget=.2, generator='scisor', statistics=diagnostics.sufficient_statistics(
                np.tile([p, 0, 0, 0, 1-p], (count, 1)), np.zeros(count, dtype=int))))
        result = diagnostics.finish_statistics(diagnostics.hierarchy_statistics(rows))
        self.assertAlmostEqual(result['ce'], (-np.log(.9)-np.log(.1))/2)
        self.assertAlmostEqual(result['multiclass_brier'], .82)


class FixtureHead(torch.nn.Module):
    def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(133701))


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name).resolve()
        self.valid = [dict(candidate_id=f'{wt}_{b}_{s}_{i}', protein_id=wt, budget=b, generator=s,
                           sequence='AA', wt_sequence='AAAA') for wt in ('w0', 'w1') for b in trial.BUDGETS for s in trial.SOURCES for i in range(2)]
        self.labels = {r['candidate_id']: dict(pair_ids=np.array([0, 2]), target=np.array([0, 4]), C=.5, T=.25) for r in self.valid}
        self.summaries = {r['candidate_id']: dict(protein_id=r['protein_id'], budget=r['budget'], known_pairs=2, C=.5,
            G_lower=.5, G_upper=.5, T_lower=.25, T_upper=.25) for r in self.valid}
        self.ledger = review.Ledger(self.root); embeddings, self.profile = embedding_fixture(self.root)
        self.audit = {'embeddings': embeddings}; self.profile_ids = review.embedding_profile_identities(self.root, embeddings, self.ledger)
        self.context = dict(root=self.root, freeze_hash='a'*64, ledger=self.ledger, roster={'w0': 'w0', 'w1': 'w1'},
            embedding_profile_identities=self.profile_ids,
            freeze=dict(preflight=self.audit, source_hashes={'runner': 'hash'}), bundle=dict(valid=self.valid, train=self.valid,
                labels=self.labels, summaries=self.summaries, runtime={'model': SimpleNamespace(PairResponseHead=FixtureHead)}))
        self.task = trial.task_inventory()[0]; self.directory = self.root/self.task['output']; self.directory.mkdir(parents=True)
        self.make_run()

    def evaluation(self, step, p):
        directory = self.directory/f'validation_{step}'; directory.mkdir()
        records = []
        for row in self.valid:
            record = review.candidate(row, self.labels[row['candidate_id']], p)
            records.append({k: v for k, v in record.items() if k not in ('statistics', 'G_MAE', 'T_MAE')})
            np.savez_compressed(directory/(row['candidate_id']+'.npz'), pair_ids=np.array([0, 2]), probabilities=p)
        decisions = core.expected_decisions(records); ce = core.hierarchical(records)
        lines(directory/'candidates.jsonl', records); lines(directory/'decisions.jsonl', decisions)
        lines(directory/'proteins.jsonl', [dict(protein_id=wt, ce=ce) for wt in self.context['roster']])
        pooled = np.concatenate([p for _ in self.valid]); y = np.concatenate([self.labels[r['candidate_id']]['target'] for r in self.valid])
        confusion = np.zeros((5, 5), dtype=int); np.add.at(confusion, (y, pooled.argmax(1)), 1)
        calibration = []
        confidence = pooled.max(1); correct = pooled.argmax(1) == y
        for i in range(10):
            mask = (confidence >= i/10) & (confidence < (i+1)/10 if i < 9 else confidence <= 1)
            calibration.append(dict(lower=i/10, upper=(i+1)/10, count=int(mask.sum()), confidence=float(confidence[mask].mean()) if mask.any() else None,
                                    accuracy=float(correct[mask].mean()) if mask.any() else None))
        np.savez_compressed(directory/'pooled_probabilities.npz', probabilities=pooled, targets=y)
        dump(directory/'report.json', dict(status='DEVELOPMENT_EVALUATED', arm='D', candidates=len(records), proteins=2,
             hierarchical_ce=ce, budget_decisions=core.budget_decisions(decisions), confusion=confusion.tolist(), calibration=calibration))
        return ce

    def make_run(self):
        ledgerdir = self.root/'ledger/d_ranking_trial'/self.context['freeze_hash']
        for name, value in [('submission.claim', {}), ('submission.json', dict(status='D_RANKING_TRIAL_SUBMITTED', freeze_sha256='a'*64, job_id='123')),
                            ('array_owner.json', dict(slurm_array_job_id='123')), ('0.claim', dict(task_index=0, slurm_job_id='123_0', slurm_array_job_id='123'))]: dump(ledgerdir/name, value)
        model = FixtureHead(); state_hash = trial.state_sha256(model)
        run = dict(status='D_RANKING_TRIAL_RUNNING', technical=False, mode='D0', seed=trial.SEEDS[0], parameter_count=133701,
            freeze_sha256='a'*64, primary_step=2000, contract=trial.CONTRACT, input_audit=self.audit, new_source_hashes={'runner': 'hash'},
            arm='D', ranking_lambda=0., training_wt=['w0', 'w1'], validation_wt=['w0', 'w1'], initialization_sha256=state_hash,
            PLM_forward_calls=0, structure_calls=0, variant_structure_files_read=0, slurm=dict(gpu_name='NVIDIA A40',
                environment=dict(SLURM_ARRAY_TASK_ID='0', SLURM_ARRAY_TASK_COUNT='9', SLURM_ARRAY_JOB_ID='123', SLURM_JOB_ID='123_0'),
                job_fields=dict(Partition='gpu_gub', TimeLimit='00:45:00', NumCPUs='4', NumNodes='1', Requeue='0', TRES='cpu=4,mem=48G,node=1,gres/gpu=1')))
        lines(self.directory/'sampling_trace.jsonl', [dict(step=1)])
        lines(self.directory/'train_trace.jsonl', [dict(step=i, ce=1., ranking=0., loss=1., gradient_norm=.1, relation_updates=1024) for i in range(1, 2001)])
        validations = []
        for step in (500, 1000, 1500, 2000):
            # Float32 0.7/0.9 expose evaluator-vs-diagnostic bin-edge differences.
            p = np.array([[.7, 0, 0, 0, .3], [.1, 0, 0, 0, .9]], dtype=np.float32)
            if step == 500: p = np.array([[.9, 0, 0, 0, .1], [.1, 0, 0, 0, .9]], dtype=np.float32)
            validations.append(dict(step=step, hierarchical_ce=self.evaluation(step, p), primary=step == 2000))
        dump(self.directory/'validation_trace.json', validations); dump(self.directory/'run.json', run)
        status = dict(status='D_RANKING_TRIAL_COMPLETE', technical=False, mode='D0', seed=trial.SEEDS[0], parameter_count=133701,
            freeze_sha256='a'*64, primary_step=2000, steps=2000, relation_updates=2048000, candidate_forwards=16000,
            checkpoint_selection=False, early_stopping=False, initialization_sha256=state_hash, final_state_sha256=state_hash,
            sampling_trace_sha256=trial.sha_file(self.directory/'sampling_trace.jsonl'), validations=validations, esmc_profile_hashes=self.profile_ids['legacy_profiles'],
            total_wall_seconds=3., actual_gpu_seconds=2., elapsed=dict(training_device_seconds=1., validation_device_seconds=1.),
            actual_projection_counts=dict(training=dict(project_calls=16000, project_residues=32000), validation=dict(project_calls=96, project_residues=192)))
        dump(self.directory/'status.json', status)
        torch.save(dict(step=2000, mode='D0', arm='D', model=model.state_dict(), provenance=run,
                        esmc_profile_hashes=self.profile_ids['legacy_profiles'], validation_ce=validations[-1]['hierarchical_ce']), self.directory/'final.pt')

    def fresh(self): self.context['ledger'] = review.Ledger(self.root)

    def test_valid_fixed_final_run_and_float32_calibration_boundaries(self):
        run = review.verify_run(self.context, self.task, 0)
        self.assertGreater(run['curves'][-1]['hierarchical_ce'], run['curves'][0]['hierarchical_ce'])
        final_ce = core.hierarchical(run['evaluation']['candidates'])
        self.assertAlmostEqual(final_ce, run['curves'][-1]['hierarchical_ce'])
        self.assertEqual(run['status']['actual_projection_counts']['training']['project_calls'], 16000)
        self.context['ledger'].check()

    def test_tampered_probabilities_pair_ids_roster_and_saved_winner_rejected(self):
        final = self.directory/'validation_2000'; row = self.valid[0]; path = final/(row['candidate_id']+'.npz'); original = path.read_bytes()
        for ids, p, message in [(np.array([0, 1]), np.ones((2, 5))/.5, 'pair IDs'),
                                (np.array([0, 2]), np.array([[np.nan]*5]*2), 'probabilities'),
                                (np.array([0, 2]), np.ones((2, 5)), 'row sum')]:
            np.savez_compressed(path, pair_ids=ids, probabilities=p); self.fresh()
            with self.assertRaisesRegex(ValueError, message): review.verify_run(self.context, self.task, 0)
        path.write_bytes(original); np.savez_compressed(final/'unexpected.npz', probabilities=np.zeros((1, 5))); self.fresh()
        with self.assertRaisesRegex(ValueError, 'NPZ roster'): review.verify_run(self.context, self.task, 0)
        (final/'unexpected.npz').unlink(); decisions = trial.read_jsonl(final/'decisions.jsonl'); decisions[0]['selected'] = 'wrong'; lines(final/'decisions.jsonl', decisions); self.fresh()
        with self.assertRaisesRegex(ValueError, 'selected'): review.verify_run(self.context, self.task, 0)

    def test_trace_freeze_checkpoint_projection_and_after_read_tampering_rejected(self):
        status_path = self.directory/'status.json'; good = trial.read_json(status_path)
        for field, value, message in [('freeze_sha256', 'b'*64, 'freeze_sha256'), ('steps', 1999, 'incomplete'),
                                      ('actual_projection_counts', dict(training={'project_calls': 1}, validation={'project_calls': 1}), 'projection')]:
            dump(status_path, good | {field: value}); self.fresh()
            with self.assertRaisesRegex(ValueError, message): review.verify_run(self.context, self.task, 0)
        dump(status_path, good); trace = trial.read_jsonl(self.directory/'train_trace.jsonl'); trace[8]['step'] = 1
        lines(self.directory/'train_trace.jsonl', trace); self.fresh()
        with self.assertRaisesRegex(ValueError, 'sequential'): review.verify_run(self.context, self.task, 0)
        trace[8]['step'] = 9; lines(self.directory/'train_trace.jsonl', trace); self.fresh(); review.verify_run(self.context, self.task, 0)
        (self.directory/'final.pt').write_bytes(b'tamper after read')
        with self.assertRaisesRegex(ValueError, 'changed during audit'): self.context['ledger'].check()

    def test_explicit_pin_allows_readonly_external_assets(self):
        with tempfile.TemporaryDirectory() as outside:
            path = Path(outside).resolve()/'native.json'; path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'unpinned external'): self.ledger.add(path)
            self.ledger.add(path, trial.sha_file(path)); self.ledger.check()
            with self.assertRaisesRegex(ValueError, 'hash differs'): self.ledger.add(path, 'b'*64)

    def test_explicit_freeze_digest_checked_before_any_context_reads(self):
        dump(self.root/trial.NEW_FREEZE, {'status': 'tampered'})
        with patch.object(trial, 'AUTHORIZED_ROOT', self.root), patch.object(trial, 'checked_freeze') as check:
            with self.assertRaisesRegex(ValueError, 'input hash differs'):
                review.read_context(self.root, 'b'*64, self.ledger)
        check.assert_not_called()

    def test_real_json_serializations_bridge_only_the_pinned_profile(self):
        import hashlib
        canonical = trial.identity(self.profile)
        legacy = hashlib.sha256(json.dumps(self.profile, sort_keys=True).encode()).hexdigest()
        self.assertNotEqual(canonical, legacy)
        self.assertEqual(self.profile_ids['canonical_to_legacy'], {canonical: legacy})
        self.assertEqual(self.audit['embeddings']['profiles'], [canonical])
        self.assertEqual(trial.read_json(self.directory/'status.json')['esmc_profile_hashes'], [legacy])
        changed = copy.deepcopy(self.audit['embeddings']); changed['profiles'] = ['b'*64]
        with self.assertRaisesRegex(ValueError, 'canonical embedding profile'): review.embedding_profile_identities(self.root, changed, review.Ledger(self.root))
        path = self.root/'caches/embeddings/fixture.json'; meta = trial.read_json(path); meta['layer'] = 35; dump(path, meta)
        with self.assertRaisesRegex(ValueError, 'input hash differs'): review.embedding_profile_identities(self.root, self.audit['embeddings'], review.Ledger(self.root))
        changed = copy.deepcopy(self.audit['embeddings']); changed['asset_hashes']['caches/embeddings/fixture.json'] = trial.sha_file(path)
        changed['assets_sha256'] = trial.identity(changed['asset_hashes'])
        with self.assertRaisesRegex(ValueError, 'canonical embedding profile'): review.embedding_profile_identities(self.root, changed, review.Ledger(self.root))

    def test_status_and_checkpoint_must_use_derived_legacy_profiles(self):
        path = self.directory/'status.json'; original = trial.read_json(path)
        dump(path, original | {'esmc_profile_hashes': self.audit['embeddings']['profiles']}); self.fresh()
        with self.assertRaisesRegex(ValueError, 'embedding profiles differ'): review.verify_run(self.context, self.task, 0)
        dump(path, original); checkpoint = torch.load(self.directory/'final.pt', map_location='cpu', weights_only=True)
        checkpoint['esmc_profile_hashes'] = self.audit['embeddings']['profiles']; torch.save(checkpoint, self.directory/'final.pt'); self.fresh()
        with self.assertRaisesRegex(ValueError, 'checkpoint metadata'): review.verify_run(self.context, self.task, 0)

    def test_nine_run_review_ensembles_probabilities_and_publishes_once(self):
        runs = []
        for relative in review.SOURCES:
            path = self.root/relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# synthetic source\n')
        for row in self.valid:
            cid = row['candidate_id']; target = 0 if cid.endswith('_0') else 4
            self.labels[cid]['target'] = np.array([target, target]); self.labels[cid]['T'] = .5 if target == 0 else 0.
        for task in trial.task_inventory():
            directory = self.root/task['output']; (directory/'validation_2000').mkdir(parents=True, exist_ok=True); records = []
            for row in self.valid:
                g = [.9, .1, .9][task['seed']-trial.SEEDS[0]] if row['candidate_id'].endswith('_0') else .8
                p = np.tile([g, 0., 0., 0., 1-g], (2, 1)); label = self.labels[row['candidate_id']]
                np.savez_compressed(directory/'validation_2000'/(row['candidate_id']+'.npz'), pair_ids=label['pair_ids'], probabilities=p)
                records.append(review.candidate(row, label, p))
            runs.append(dict(mode=task['mode'], seed=task['seed'], directory=str(directory), status=dict(actual_gpu_seconds=1., esmc_profile_hashes=self.profile_ids['legacy_profiles']),
                             evaluation=dict(candidates=records, decisions=core.expected_decisions(records))))
        def context(root, checksum, ledger):
            for relative in review.SOURCES: ledger.add(root/relative)
            profiles = review.embedding_profile_identities(root, self.audit['embeddings'], ledger)
            return self.context | dict(ledger=ledger, embedding_profile_identities=profiles, freeze=self.context['freeze'] | {'tasks': trial.task_inventory()})
        output = self.root/'reports/d_ranking_trial/review'
        with patch.object(trial, 'AUTHORIZED_ROOT', self.root), patch.object(review, 'read_context', side_effect=context), \
             patch.object(review, 'verify_run', side_effect=runs), patch.object(review, 'verify_sampling'), \
             patch.object(core, 'load_statistics', return_value=STATISTICS):
            report = review.build_review(self.root, 'a'*64, output)
            selected = lambda part: next(r for r in part['selection'] if r['budget'] == 'all' and r['source'] == 'mixed')['selected_T']
            self.assertAlmostEqual(selected(report['mode_averages']['D0']), 1/3)
            self.assertEqual(selected(report['ensemble']['modes']['D0']), 0.)
            self.assertEqual(report['status'], 'D_RANKING_REVIEW_COMPLETE')
            self.assertEqual(report['embedding_profile_identities'], self.profile_ids)
            self.assertEqual(report['resources']['actual_gpu_seconds'], 9.)
            self.assertEqual(len(trial.read_jsonl(output/'per_wt.jsonl')), 216)
            for name, checksum in report['artifact_sha256'].items(): self.assertEqual(trial.sha_file(output/name), checksum)
            with self.assertRaisesRegex(ValueError, 'already exists'): review.build_review(self.root, 'a'*64, output)


class SamplingTests(unittest.TestCase):
    def test_independent_replay_rejects_training_id_hash_and_validation_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rows = [dict(candidate_id=f'{wt}_{b}_{s}_{i}', protein_id=wt, budget=b, generator=s, sequence='AA', wt_sequence='AAAA')
                for wt in ('t0', 't1', 't2', 't3') for b in trial.BUDGETS for s in trial.SOURCES for i in range(7)]
            labels = {r['candidate_id']: {'pair_ids': np.arange(150)} for r in rows}; assignment = {r['protein_id']: 'train' for r in rows}; runs = []
            for seed in trial.SEEDS:
                sampler = trial.PairedSampler(rows, assignment, seed); trace = [sampler.step(labels, i)[1] for i in range(1, 2001)]
                for mode in trial.MODES:
                    directory = root/f'{mode}_{seed}'; lines(directory/'sampling_trace.jsonl', trace)
                    runs.append(dict(mode=mode, seed=seed, directory=str(directory), initialization_sha256=str(seed),
                        status=dict(sampling_trace_sha256=trial.sha_file(directory/'sampling_trace.jsonl'), actual_projection_counts=dict(training=dict(project_residues=96000 if mode == 'C0' else 32000)))))
            context = dict(bundle=dict(train=rows, labels=labels, split={'assignment': assignment}))
            review.verify_sampling(context, runs)
            directory = Path(runs[0]['directory']); bad = trial.read_jsonl(directory/'sampling_trace.jsonl'); bad[0]['conditions'][0]['candidates'][0]['relation_id_sha256'] = 'bad'; lines(directory/'sampling_trace.jsonl', bad)
            with self.assertRaisesRegex(ValueError, 'replayed'): review.verify_sampling(context, runs)


if __name__ == '__main__': unittest.main()
