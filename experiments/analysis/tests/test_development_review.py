"""CPU synthetic contract tests; no training, model imports, or real artifacts."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'analysis'))
import development_review as review


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_root = review.common.AUTHORIZED_ROOT
        review.common.AUTHORIZED_ROOT = self.root
        self.addCleanup(setattr, review.common, 'AUTHORIZED_ROOT', self.old_root)
        self.addCleanup(self.temp.cleanup)
        self.dump('configs/protocol.json', {
            'root': str(self.root), 'seed': 20261013,
            'model': {'arms': list('SABCD')},
            'train': {'learning_rates': [.0003, .001], 'seeds': [20261013],
                      'max_steps': 4, 'eval_every': 2, 'patience': 2,
                      'selection': 'hierarchical_protein_mean_validation_ce'},
            'statistics': {'bootstrap_replicates': 10000, 'confidence': .95}})
        manifest = [{'candidate_id': p + g, 'protein_id': p, 'budget': .2,
                     'generator': g, 'source': 'historical'}
                    for p in ('p', 'q', 'r') for g in ('g1', 'g2')]
        self.lines('manifests/primary.jsonl', manifest)
        self.dump('manifests/splits.json', {
            'status': 'SPLIT_FROZEN', 'protocol_sha256': self.sha('configs/protocol.json'),
            'assignment': {p: 'validation' for p in ('p', 'q', 'r')},
            'component': {'p': 'family1', 'q': 'family1', 'r': 'family2'}})
        summaries = [{**r, 'known_pairs': 2, 'C': .8, 'G_lower': .25,
                      'G_upper': .75, 'T_lower': .2, 'T_upper': .6} for r in manifest]
        self.lines('reports/labels.jsonl', summaries)
        self.lines('caches/index.jsonl', [{'key': 'embedding'}])
        snapshot = self.root / 'snapshot'
        snapshot.mkdir()
        for name in ('common.py', 'paired_statistics.py'):
            shutil.copyfile(REPO / 'code' / name, snapshot / name)
        self.code = {p.name: review.common.sha_file(p) for p in snapshot.iterdir()}
        self.dump('reports/dispatch_after_pilot.json', {
            'code_snapshot': str(snapshot), 'code_hashes': self.code})
        paths = ['configs/protocol.json', 'manifests/primary.jsonl',
                 'manifests/splits.json', 'reports/labels.jsonl']
        self.dump('configs/execution_freeze.json', {'input_hashes': {p: self.sha(p) for p in paths}})
        self.paths = dict(root=self.root, config=self.root/'configs/protocol.json',
                          manifest=self.root/'manifests/primary.jsonl',
                          splits=self.root/'manifests/splits.json',
                          label_summary=self.root/'reports/labels.jsonl',
                          cache_indices=[self.root/'caches/index.jsonl'],
                          output=self.root/'reports/development_review')
        for arm in 'SABCD':
            for lr in (.0003, .001):
                self.make_run(arm, lr, manifest, summaries)

    def dump(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def lines(self, path, rows):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def sha(self, path):
        return review.common.sha_file(self.root/path)

    def directory(self, arm='C', lr=.0003):
        return Path('checkpoints') / f'{arm}_s20261013_lr{lr:g}'

    def make_run(self, arm, lr, manifest, summaries):
        d = self.directory(arm, lr)
        config = {k: str(v) for k, v in self.paths.items()
                  if k in ('manifest', 'splits', 'label_summary')}
        config.update(protocol=str(self.paths['config']), arm=arm, lr=lr,
                      seed=20261013, technical=False, steps=None,
                      output=str(self.root/d), cache_index=[str(self.paths['cache_indices'][0])])
        self.dump(d/'run.json', {'protocol': self.sha('configs/protocol.json'),
                  'manifest': self.sha('manifests/primary.jsonl'),
                  'splits': self.sha('manifests/splits.json'),
                  'label_summary_sha256': self.sha('reports/labels.jsonl'),
                  'cache_index_sha256': {str(self.paths['cache_indices'][0]): self.sha('caches/index.jsonl')},
                  'code': self.code, 'config': config})
        # Per-WT means 1,3,10: family means 2 and10, overall family mean6.
        ces = {'p': 1., 'q': 3., 'r': 10.}
        candidates = [{k: r[k] for k in ('candidate_id','protein_id','budget','generator')}
                      | {k: s[k] for k in ('C','G_lower','G_upper','T_lower','T_upper')}
                      | {'Ghat': .5, 'That': .4, 'ce': ces[r['protein_id']], 'valid_pairs': 2}
                      for r, s in zip(manifest, summaries)]
        ce = 14/3
        self.lines(d/'train_trace.jsonl', [{'step': i, 'ce': 1.} for i in range(1,5)])
        self.lines(d/'validation_trace.jsonl', [{'step': 2, 'ce': ce, 'improved': True},
                                                {'step': 4, 'ce': ce+.1, 'improved': False}])
        self.dump(d/'status.json', {'status': 'DEVELOPMENT_RUN_COMPLETE', 'arm': arm,
                  'best_ce': ce, 'steps': 4, 'stopped_by_patience': False,
                  'convergence_review_required': arm in ('S','B'), 'esmc_profile_hashes': ['profile']})
        checkpoint = self.root/d/'best.pt'
        checkpoint.write_bytes(b'opaque checkpoint - never loaded')
        self.lines(d/'validation_2/candidates.jsonl', candidates)
        self.lines(d/'validation_2/proteins.jsonl', [{'protein_id': p, 'ce': ces[p]} for p in ces])
        decisions = review.expected_decisions(candidates)
        self.lines(d/'validation_2/decisions.jsonl', decisions)
        self.dump(d/'validation_2/report.json', {
            'status': 'DEVELOPMENT_EVALUATED', 'arm': arm, 'hierarchical_ce': ce,
            'proteins': 3, 'candidates': 6,
            'budget_decisions': review.budget_decisions(decisions)})

    def test_missing_run_waits_without_comparisons(self):
        shutil.rmtree(self.root/self.directory('D', .001))
        result = review.build_review(**self.paths)
        self.assertEqual(result['status'], 'DEVELOPMENT_REVIEW_WAITING')
        self.assertEqual(len(result['runs']), 10)
        self.assertNotIn('comparisons', result)
        self.assertEqual(result['runs'][-1]['state'], 'PENDING')

    def test_complete_tie_selects_lower_lr_and_family_pairing(self):
        result = review.build_review(**self.paths)
        self.assertEqual(result['status'], 'DEVELOPMENT_REVIEW_READY_FOR_HUMAN_OR_AGENT_EVIDENCE_REVIEW')
        self.assertEqual(result['selected']['C']['lr'], .0003)
        ce = result['comparisons']['C-B']['CE_overall']
        self.assertEqual(ce['n_wts'], 3)
        self.assertEqual(ce['n_groups'], 2)
        self.assertEqual(ce['difference_bounds'], (0., 0.))
        bounds = result['comparisons']['C-B']['T|0.2|g1']['difference_bounds']
        self.assertAlmostEqual(bounds[0], -.4)
        self.assertAlmostEqual(bounds[1], .4)
        self.assertTrue((self.root/'reports/development_review/paired_inputs.jsonl').is_file())

    def test_missing_candidate_rejected(self):
        path = self.root/self.directory()/'validation_2/candidates.jsonl'
        rows = review.common.read_jsonl(path)
        self.lines(path, rows[:-1])
        with self.assertRaisesRegex(ValueError, 'candidate roster'):
            review.build_review(**self.paths)

    def test_invalid_hash_rejected_even_when_other_run_pending(self):
        path = self.root/self.directory()/'run.json'
        data = json.loads(path.read_text()); data['protocol'] = 'changed'
        self.dump(path, data)
        (self.root/self.directory('D', .001)/'status.json').unlink()
        with self.assertRaisesRegex(ValueError, 'protocol'):
            review.build_review(**self.paths)

    def test_duplicate_decision_rejected(self):
        path = self.root/self.directory()/'validation_2/decisions.jsonl'
        rows = review.common.read_jsonl(path)
        self.lines(path, rows + rows[:1])
        with self.assertRaisesRegex(ValueError, 'duplicate decision'):
            review.build_review(**self.paths)

    def test_best_step_is_earliest_equal_minimum(self):
        path = self.root/self.directory()/'validation_trace.jsonl'
        rows = review.common.read_jsonl(path); rows[1]['ce'] = rows[0]['ce']
        self.lines(path, rows)
        result = review.build_review(**self.paths)
        self.assertEqual(result['selected']['C']['best_step'], 2)

    def test_invalid_frozen_code_rejected(self):
        (self.root/'snapshot/paired_statistics.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'snapshot hashes'):
            review.build_review(**self.paths)

    def test_output_cannot_overwrite_existing_validation_report(self):
        directory = self.root/self.directory()/'validation_2'
        existing = (directory/'report.json').read_bytes()
        paths = self.paths | {'output':directory}
        with self.assertRaisesRegex(ValueError, 'dedicated development review subtree'):
            review.build_review(**paths)
        self.assertEqual((directory/'report.json').read_bytes(), existing)
        self.assertFalse((directory/'paired_inputs.jsonl').exists())

    def test_undefined_predicted_g_with_positive_pairs_rejected(self):
        directory = self.root/self.directory()/'validation_2'
        candidates = review.common.read_jsonl(directory/'candidates.jsonl')
        for row in candidates:
            row['Ghat'] = None
            row['That'] = 0.
        decisions = review.expected_decisions(candidates)
        self.lines(directory/'candidates.jsonl', candidates)
        self.lines(directory/'decisions.jsonl', decisions)
        report = json.loads((directory/'report.json').read_text())
        report['budget_decisions'] = review.budget_decisions(decisions)
        self.dump(directory/'report.json', report)
        with self.assertRaisesRegex(ValueError, 'Ghat must be finite'):
            review.build_review(**self.paths)

    def test_dispatch_override_cannot_be_overwritten(self):
        path = self.paths['output']/'report.json'
        self.dump(path, json.loads((self.root/'reports/dispatch_after_pilot.json').read_text()))
        existing = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'output contains an input'):
            review.build_review(**(self.paths | {'dispatch':path}))
        self.assertEqual(path.read_bytes(), existing)

    def test_run_root_override_cannot_overlap_output(self):
        base = self.paths['output']/'checkpoints'
        run = base/self.directory().name
        shutil.copytree(self.root/self.directory(), run)
        provenance = json.loads((run/'run.json').read_text())
        provenance['config']['output'] = str(run)
        self.dump(run/'run.json', provenance)
        directory = run/'validation_2'
        existing = (directory/'report.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'output overlaps run root'):
            review.build_review(**(self.paths | {'run_root':base,'output':directory}))
        self.assertEqual((directory/'report.json').read_bytes(), existing)

    def test_linked_text_outputs_cannot_overwrite_protocol(self):
        output = self.paths['output']
        output.mkdir(parents=True)
        protocol = self.paths['config']
        existing = protocol.read_bytes()
        for name in ('engineering.csv','ENGINEERING.md'):
            os.link(protocol, output/name)
        result = review.build_review(**self.paths)
        self.assertEqual(result['status'], review.READY)
        self.assertEqual(protocol.read_bytes(), existing)
        for name in ('engineering.csv','ENGINEERING.md'):
            self.assertNotEqual((output/name).read_bytes(), existing)
            self.assertFalse(os.path.samefile(protocol, output/name))

    def test_symlinked_text_outputs_preserve_protocol(self):
        output = self.paths['output']
        output.mkdir(parents=True)
        protocol = self.paths['config']
        existing = protocol.read_bytes()
        try:
            for name in ('engineering.csv','ENGINEERING.md'):
                (output/name).symlink_to(protocol)
        except OSError as error:
            self.skipTest('symlink creation unavailable: '+str(error))
        result = review.build_review(**self.paths)
        self.assertEqual(result['status'], review.READY)
        self.assertEqual(protocol.read_bytes(), existing)
        for name in ('engineering.csv','ENGINEERING.md'):
            self.assertFalse((output/name).is_symlink())
            self.assertNotEqual((output/name).read_bytes(), existing)

    def test_bad_trace_best_and_patience_rejected(self):
        path = self.root/self.directory()/'status.json'
        status = json.loads(path.read_text())
        status['best_ce'] = .1
        self.dump(path, status)
        with self.assertRaisesRegex(ValueError, 'best CE'):
            review.build_review(**self.paths)
        status['best_ce'] = 14/3
        status['stopped_by_patience'] = True
        self.dump(path, status)
        with self.assertRaisesRegex(ValueError, 'patience'):
            review.build_review(**self.paths)

    def test_missing_protein_rejected(self):
        path = self.root/self.directory()/'validation_2/proteins.jsonl'
        self.lines(path, review.common.read_jsonl(path)[:-1])
        with self.assertRaisesRegex(ValueError, 'protein roster'):
            review.build_review(**self.paths)

    def test_candidate_metadata_and_decision_selection_rejected(self):
        path = self.root/self.directory()/'validation_2/candidates.jsonl'
        rows = review.common.read_jsonl(path)
        rows[0]['generator'] = 'other'
        self.lines(path, rows)
        with self.assertRaisesRegex(ValueError, 'candidate metadata'):
            review.build_review(**self.paths)
        rows[0]['generator'] = 'g1'
        self.lines(path, rows)
        path = self.root/self.directory()/'validation_2/decisions.jsonl'
        rows = review.common.read_jsonl(path); rows[0]['selected'] = 'not-present'
        self.lines(path, rows)
        with self.assertRaisesRegex(ValueError, 'selected'):
            review.build_review(**self.paths)

    def test_missing_pool_retains_wt_and_source_budget_strata(self):
        run = {'lr': .0003, 'checkpoint_sha256': 'hash', '_evaluation': {
            'proteins': [{'protein_id':p,'ce':1.} for p in ('p','q')],
            'candidates': [{'protein_id':'p','generator':'g1','budget':.2,'ce':1.}],
            'decisions': [{'protein_id':'p','generator':'g1','budget':'.2',
                           'T_lower':.2,'T_upper':.6,'G_lower':.3,'G_upper':.8,
                           'selected':'p1','pool_size':1}]}}
        rows = review.paired_inputs({'C':run}, {'p':'f1','q':'f2'})
        missing = [r for r in rows if r['wt_id']=='q' and r['endpoint']=='T|.2|g1']
        self.assertEqual(len(missing),1)
        self.assertEqual(missing[0]['status'],'UNDEFINED_NO_POOL')
        self.assertIsNone(missing[0]['lower'])

    def test_unfavorable_comparisons_are_included(self):
        rows = [{'method':arm,'wt_id':'p','group_id':'family','endpoint':'CE_overall',
                 'lower':2. if arm=='C' else 1.,'upper':2. if arm=='C' else 1.}
                for arm in 'SABCD']
        comparisons = review.compare_inputs(rows, {'p':'family'}, review.load_statistics(self.root/'snapshot'), .95)
        self.assertEqual(set(comparisons), {'C-B','C-S','C-A','C-D'})
        self.assertTrue(all(v['CE_overall']['difference_bounds']==(1.,1.) for v in comparisons.values()))

    def test_unknown_bounds_keep_full_wt_roster(self):
        rows = [{'method': arm, 'wt_id': p, 'group_id': g,
                 'endpoint': 'G|0.2|g1', 'lower': None if p=='r' else .1,
                 'upper': None if p=='r' else .8}
                for arm in ('C','B') for p,g in {'p':'f1','q':'f1','r':'f2'}.items()]
        result = review.compare_inputs(rows, {'p':'f1','q':'f1','r':'f2'}, review.load_statistics(self.root/'snapshot'), .95)
        self.assertEqual(result['C-B']['G|0.2|g1']['status'], 'UNDEFINED_ENDPOINT_BOUNDS')
        self.assertEqual(result['C-B']['G|0.2|g1']['undefined_wts'], ['r'])

    def test_unbalanced_variants_aggregate_within_wt_before_family(self):
        stats = review.load_statistics(self.root/'snapshot')
        rows = [{'wt_id': p, 'group_id': g, 'method': arm, 'endpoint':'CE_overall',
                 'lower': ce if arm=='C' else 0., 'upper':ce if arm=='C' else 0.}
                for arm in ('C','B') for p,g,ce,n in [('p','f1',1.,20),('q','f1',3.,1),('r','f2',10.,1)]
                for _ in range(n)]
        result = review.compare_inputs(rows, {'p':'f1','q':'f1','r':'f2'}, stats, .95)
        self.assertEqual(result['C-B']['CE_overall']['difference_bounds'], (6.,6.))


if __name__ == '__main__':
    unittest.main()
