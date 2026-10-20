"""Synthetic scalar review contracts, entirely on CPU."""
import json
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import development_review as core
import test_development_review as fixtures
import review_scalar_baselines as scalar


class ScalarReviewTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ReviewTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture; self.root = fixture.root
        primary = core.common.read_jsonl(fixture.paths['manifest'])
        self.full = primary + [{'candidate_id':'sensitivity','protein_id':'train',
                               'budget':.3,'generator':'extra','source':'other'}]
        fixture.lines('manifests/development.jsonl',self.full)
        freeze = core.read_json(self.root/'configs/execution_freeze.json')
        freeze['input_hashes']['manifests/development.jsonl'] = fixture.sha('manifests/development.jsonl')
        fixture.dump('configs/execution_freeze.json',freeze)
        fixture.dump('analysis/prepare_utility_baseline.py',{'synthetic':'CPU scalar implementation'})
        sources = {}
        for name in ('teacher_scoring','proposal_validation'):
            path = self.root/'provenance'/f'{name}.py'
            fixture.dump(path,{'synthetic':name})
            sources[name] = {'original_path':str(path),'snapshot':str(path),'sha256':core.common.sha_file(path)}
        fixture.dump('provenance/utility_baseline_sources.json',sources)
        records = [{k:r[k] for k in ('candidate_id','protein_id','budget','generator')}
                   | {'utility':.5,'local_cosine_similarity':.4,
                      'static_relationship_coverage':.8,'direct_contact_loss':.2}
                   for r in self.full]
        fixture.lines('reports/utility_baseline/candidates.jsonl',records)
        fixture.dump('reports/utility_baseline/report.json',{
            'status':'CPU_UTILITY_BASELINE_COMPLETE','candidates':len(records),'WT':4,
            'protocol_sha256':fixture.sha('configs/protocol.json'),
            'manifest_sha256':fixture.sha('manifests/development.jsonl'),
            'cache_index_sha256':{str(fixture.paths['cache_indices'][0]):fixture.sha('caches/index.jsonl')},
            'embedding_profile_hashes':['profile'],'sources':sources,
            'implementation_sha256':fixture.sha('analysis/prepare_utility_baseline.py'),
            'output_sha256':fixture.sha('reports/utility_baseline/candidates.jsonl'),
            'weights':{'global_similarity':.35,'retained_similarity':.45,'junction_plausibility':.20},
            'GPU_used':False,'PLM_forward_calls':0,'structure_calls':0})
        core.build_review(**fixture.paths)
        self.paths = dict(root=self.root,config=fixture.paths['config'],
            full_manifest=self.root/'manifests/development.jsonl',
            primary_manifest=fixture.paths['manifest'],splits=fixture.paths['splits'],
            label_summary=fixture.paths['label_summary'],cache_indices=fixture.paths['cache_indices'],
            baseline_output=self.root/'reports/utility_baseline',
            core_report=self.root/'reports/development_review/report.json',
            output=self.root/'reports/development_review/scalar_baselines')

    def test_wait_until_core_ready(self):
        self.fixture.dump(self.paths['core_report'],{'status':core.WAITING})
        self.paths['baseline_output'].rename(self.root/'reports/baseline_unavailable')
        result = scalar.build_review(**self.paths)
        self.assertEqual(result['status'],scalar.WAITING)
        self.assertNotIn('comparisons',result)

    def test_malformed_ready_inventory_rejected_before_effects(self):
        original = core.read_json(self.paths['core_report'])
        for mutation in ('non_C_pending','C_only','bad_counts','duplicate_run'):
            with self.subTest(mutation=mutation):
                report = json.loads(json.dumps(original))
                if mutation=='non_C_pending':
                    next(r for r in report['runs'] if r['arm']=='D')['state']='PENDING'
                elif mutation=='C_only':
                    report['runs']=[r for r in report['runs'] if r['arm']=='C']
                elif mutation=='bad_counts':
                    report['completed_runs']=9
                else:
                    report['runs'][-1]=report['runs'][0]
                self.fixture.dump(self.paths['core_report'],report)
                with self.assertRaisesRegex(ValueError,'core run inventory'):
                    scalar.build_review(**self.paths)
                self.assertFalse((self.paths['output']/'report.json').exists())

    def test_score_selection_tie_and_complete_pairing(self):
        result = scalar.build_review(**self.paths)
        self.assertEqual(result['status'],scalar.READY)
        self.assertEqual(set(result['comparisons']),{'C-U','C-L','C-V'})
        rows = core.common.read_jsonl(self.paths['output']/'paired_inputs.jsonl')
        selected = [r for r in rows if r['method']=='U' and r['wt_id']=='p'
                    and r['generator']=='mixed' and r['endpoint']=='T|0.2|mixed']
        self.assertEqual(selected[0]['selected_candidate'],'pg2')
        summary = result['comparisons']['C-U']['T|0.2|mixed']
        self.assertEqual((summary['n_wts'],summary['n_groups']),(3,2))
        self.assertAlmostEqual(summary['difference_bounds'][0],-.4)

    def test_missing_sensitivity_candidate_rejected(self):
        path = self.paths['baseline_output']/'candidates.jsonl'
        self.fixture.lines(path,core.common.read_jsonl(path)[:-1])
        report = core.read_json(self.paths['baseline_output']/'report.json')
        report['output_sha256'] = core.common.sha_file(path)
        self.fixture.dump(self.paths['baseline_output']/'report.json',report)
        with self.assertRaisesRegex(ValueError,'full candidate roster'):
            scalar.build_review(**self.paths)

    def test_invalid_baseline_identity_rejected(self):
        path = self.paths['baseline_output']/'report.json'
        report = core.read_json(path); report['manifest_sha256']='invalid'
        self.fixture.dump(path,report)
        with self.assertRaisesRegex(ValueError,'baseline manifest'):
            scalar.build_review(**self.paths)

    def test_selected_c_output_hash_rejected(self):
        core_report = core.read_json(self.paths['core_report'])
        path = Path(core_report['selected']['C']['validation_output'])/'decisions.jsonl'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'selected C output hash'):
            scalar.build_review(**self.paths)

    def test_invalid_metadata_rejected(self):
        path = self.paths['baseline_output']/'candidates.jsonl'
        rows = core.common.read_jsonl(path); rows[-1]['generator']='changed'
        self.fixture.lines(path,rows)
        report = core.read_json(self.paths['baseline_output']/'report.json')
        report['output_sha256']=core.common.sha_file(path)
        self.fixture.dump(self.paths['baseline_output']/'report.json',report)
        with self.assertRaisesRegex(ValueError,'baseline candidate metadata'):
            scalar.build_review(**self.paths)

    def test_each_baseline_uses_its_own_fixed_score(self):
        manifest = core.common.read_jsonl(self.paths['primary_manifest'])
        scores = {r['candidate_id']:r for r in core.common.read_jsonl(self.paths['baseline_output']/'candidates.jsonl')}
        summaries = {r['candidate_id']:r for r in core.common.read_jsonl(self.paths['label_summary'])}
        scores['pg1']['local_cosine_similarity'] = .9
        choices = {}
        for method in scalar.SELECTORS:
            rows = scalar.scalar_decisions(manifest,scores,summaries,method)
            choices[method] = next(r['selected'] for r in rows if r['protein_id']=='p' and r['generator']=='mixed')
        self.assertEqual(choices,{'U':'pg2','L':'pg1','V':'pg2'})

    def test_invalid_realized_bounds_rejected_after_hash_refresh(self):
        report = core.read_json(self.paths['core_report'])
        selected = report['selected']['C']
        path = Path(selected['validation_output'])/'candidates.jsonl'
        rows = core.common.read_jsonl(path)
        rows[0]['T_lower'] = .9
        self.fixture.lines(path,rows)
        selected['output_hashes']['candidates.jsonl'] = core.common.sha_file(path)
        self.fixture.dump(self.paths['core_report'],report)
        with self.assertRaisesRegex(ValueError,'frozen endpoint'):
            scalar.build_review(**self.paths)

    def test_unknown_bounds_keep_full_roster(self):
        roster = {'p':'f1','q':'f1','r':'f2'}
        rows = [{'method':arm,'wt_id':wt,'group_id':group,'endpoint':'G|0.2|mixed',
                 'lower':None if wt=='r' else .2,'upper':None if wt=='r' else .6}
                for arm in ('C','U','L','V') for wt,group in roster.items()]
        result = scalar.compare_inputs(rows,roster,core.load_statistics(self.root/'snapshot'),.95)
        self.assertEqual(result['C-U']['G|0.2|mixed']['undefined_wts'],['r'])
        self.assertEqual(result['C-U']['G|0.2|mixed']['n_wts'],3)
        self.assertNotIn('difference_bounds',result['C-U']['G|0.2|mixed'])

    def test_family_aggregation_and_unfavorable_results(self):
        roster = {'p':'f1','q':'f1','r':'f2'}
        rows = [{'method':arm,'wt_id':wt,'group_id':group,'endpoint':'T|0.2|mixed',
                 'lower':0. if arm=='C' else value,'upper':0. if arm=='C' else value}
                for arm in ('C','U','L','V') for wt,group,value in
                [('p','f1',.1),('q','f1',.3),('r','f2',1.)]]
        result = scalar.compare_inputs(rows,roster,core.load_statistics(self.root/'snapshot'),.95)
        self.assertTrue(all(abs(x['T|0.2|mixed']['difference_bounds'][0]+.6)<1e-12 for x in result.values()))

    def test_protected_output_and_hardlink(self):
        with self.assertRaisesRegex(ValueError,'dedicated scalar baseline subtree'):
            scalar.build_review(**(self.paths|{'output':self.root/'reports/utility_baseline'}))
        self.paths['output'].mkdir(parents=True)
        original = self.paths['config'].read_bytes()
        os.link(self.paths['config'],self.paths['output']/'ENGINEERING.md')
        scalar.build_review(**self.paths)
        self.assertEqual(self.paths['config'].read_bytes(),original)


if __name__=='__main__':
    unittest.main()
