"""Synthetic CPU diagnostics checks; no model or structure imports."""
import sys
import json
import os
import unittest
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cpu_prediction_diagnostics as pred
import cpu_decision_diagnostics as dec
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_development_review as fixtures


def record(wt, budget, generator, probability, n=1):
    return dict(protein_id=wt, budget=budget, generator=generator,
                candidate_id=f'{wt}-{budget}-{generator}-{probability}',
                statistics=pred.sufficient_statistics(np.tile([probability, 1-probability, 0, 0, 0], (n, 1)), np.zeros(n, dtype=int)))


class PredictionTests(unittest.TestCase):
    def test_hierarchy_does_not_pool_pairs_or_candidates(self):
        rows = [record('a', 1, 'g', .9, 100), record('a', 1, 'h', .1),
                record('a', 1, 'h', .1), record('b', 1, 'g', .7)]
        result = pred.aggregate(rows)
        self.assertAlmostEqual(result['mean_class_probability'][0], .6)
        self.assertEqual(result['n_candidates'], 4)

    def test_ece_after_aggregation(self):
        a = pred.sufficient_statistics(np.array([[.6,.4,0,0,0]]), np.array([0]))
        b = pred.sufficient_statistics(np.array([[.6,.4,0,0,0]]), np.array([1]))
        rows = [dict(protein_id='a',budget=1,generator='g',candidate_id='a',statistics=s) for s in (a,b)]
        self.assertAlmostEqual(pred.aggregate(rows)['top_label_reliability']['ece'], .1)
        self.assertAlmostEqual((pred.finish_statistics(a)['top_label_reliability']['ece']+pred.finish_statistics(b)['top_label_reliability']['ece'])/2, .5)

    def test_bins_include_zero_and_one(self):
        self.assertEqual(pred.bin_indices(np.array([0., .1, .999, 1.])).tolist(), [0,1,9,9])
        result = pred.finish_statistics(pred.sufficient_statistics(np.eye(5), np.arange(5)))
        self.assertAlmostEqual(result['top_label_reliability']['bins'][9]['weight'], 1.)

    def test_invalid_probabilities(self):
        for p in (np.array([[1.1,-.1,0,0,0]]), np.zeros((1,5)), np.array([[np.nan,0,0,0,0]])):
            with self.assertRaises(ValueError): pred.validate_probabilities(p, np.array([0]), np.array([0]))
        with self.assertRaises(ValueError): pred.validate_probabilities(np.eye(5),np.arange(5),np.arange(4))

    def test_confusion_zero_denominator(self):
        value = pred.finish_statistics(pred.sufficient_statistics(np.array([[1.,0,0,0,0]]),np.array([0])))
        self.assertIsNone(value['confusion_recall'][1])
        self.assertIsNone(value['confusion_precision'][1])

    def test_empty_stratum_preserves_roster(self):
        rows = [record('a',1,'g',.9), dict(protein_id='b',budget=1,generator='g',candidate_id='b',statistics=None)]
        result = pred.aggregate(rows, expected_wts=['a','b'])
        self.assertEqual(result['status'], 'UNDEFINED_ZERO_ELIGIBLE_KNOWN_PAIRS')
        self.assertEqual(result['expected_wts'], ['a','b'])
        self.assertEqual(result['n_candidates'], 2)
        with self.assertRaises(ValueError): pred.aggregate(rows[:1], expected_wts=['a','b'])

    def test_hash_rejects_modified_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.json'; path.write_text('{}')
            identity=pred.common.sha_file(path)
            path.write_text('{"changed":true}')
            with self.assertRaises(ValueError): pred.checked_hash(path,identity,'sealed input')

    def test_legacy_label_path_has_frozen_creation_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve(); labels=root/'labels/pairs'
            # This intentionally differs from the historical ROOT/code cwd.
            previous=Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(pred.resolve_label_path(root,labels,'x','../labels/pairs/x.npz'),labels/'x.npz')
                self.assertEqual(pred.resolve_label_path(root,labels,'x',str(labels/'x.npz')),labels/'x.npz')
                with self.assertRaises(ValueError): pred.resolve_label_path(root,labels,'x','../labels/pairs/other.npz')
            finally: os.chdir(previous)

    def test_labels_mapping_partition_and_thresholds(self):
        row = dict(candidate_id='x',wt_sequence='AAAAA',sequence='AAAAA',mask=[0]*5)
        lab = dict(kept=np.arange(5),reference_i=np.array([0,0,1]),reference_j=np.array([3,4,4]),
                   retained_mask=np.ones(3,dtype=bool),pair_ids=np.array([0,2]),unknown_pair_ids=np.array([1]),
                   wt_i=np.array([0,1]),wt_j=np.array([3,4]),edited_i=np.array([0,1]),edited_j=np.array([3,4]),
                   error=np.array([.5,4.]),target=np.array([1,4]))
        summary=dict(reference_pairs=3,retained_pairs=3,known_pairs=2,unknown_pairs=1,C=1.,
                     G_lower=.25,G_upper=7/12,T_lower=.25,T_upper=7/12,class_counts=[0,1,0,0,1])
        pred.validate_labels(lab,row,summary)
        for key,value in [('unknown_pair_ids',np.array([0])),('edited_i',np.array([1,1])),('target',np.array([0,4]))]:
            broken={**lab,key:value}
            with self.assertRaises(ValueError): pred.validate_labels(broken,row,summary)


class DecisionTests(unittest.TestCase):
    def test_common_retained_keeps_unknown(self):
        labels={'a':dict(retained_mask=np.array([True,True,False]),pair_ids=np.array([0]),target=np.array([0])),
                'b':dict(retained_mask=np.array([False,True,True]),pair_ids=np.array([2]),target=np.array([0]))}
        ids=dec.common_retained(['a','b'],labels)
        self.assertEqual(ids.tolist(),[1])
        self.assertEqual(dec.common_bounds(labels['a'],ids),(0.,1.,0,1))

    def test_empty_intersection_is_undefined(self):
        lab=dict(pair_ids=np.array([0]),target=np.array([0]))
        self.assertEqual(dec.common_bounds(lab,np.array([],dtype=int)),(None,None,0,0))
        rows=[dict(wt_id='a',group_id='a',method=m,endpoint='G|1|mixed',lower=None,upper=None) for m in ['C','B']]
        result=dec.compare_common(rows,{'a':'a'},None,.95,comparators=['B'])
        self.assertEqual(result['C-B']['G|1|mixed']['n_wts'],1)

    def test_strict_matching(self):
        base=dict(protein_id='a',budget=.2,generator='g',mask=[0,1,1,0,0],candidate_id='a')
        score=dict(static_relationship_coverage=.099,direct_contact_loss=.2)
        key=dec.matching_key(base,score)
        self.assertNotEqual(key,dec.matching_key(base,{**score,'static_relationship_coverage':.1}))
        self.assertNotEqual(key,dec.matching_key({**base,'mask':[0,1,0,1,0]},score))
        self.assertNotEqual(key,dec.matching_key({**base,'mask':[1,1,0,0,0]},score))
        self.assertEqual(dec.fixed_bin(.15),3)

    def test_choices_never_changed(self):
        choices={'C':[dict(protein_id='a',budget='1',generator=g,selected='low',pool_size=2) for g in ['g','mixed']]}
        rows=[dict(protein_id='a',budget=1,generator='g',candidate_id=x) for x in ['low','high']]
        lab={x:dict(retained_mask=np.ones(1,dtype=bool),pair_ids=np.array([0]),target=np.array([y])) for x,y in [('low',4),('high',0)]}
        common=dec.common_decision_rows(rows,lab,choices,{'a':'family'})
        self.assertEqual(common[0]['selected_candidate'],'low')
        self.assertEqual(common[0]['lower'],0.)

    def test_family_weight_influence(self):
        rows=[dict(wt_id=w,group_id=g,method=m,endpoint='CE_overall',lower=v,upper=v)
              for w,g,c in [('a','x',2),('b','x',4),('c','y',10)] for m,v in [('C',c),('B',0)]]
        result=dec.ce_influence(rows,{'a':'x','b':'x','c':'y'},comparators=['B'])
        self.assertAlmostEqual(result['C-B']['CE_overall']['mean'],6.5)
        self.assertAlmostEqual(result['C-B']['CE_overall']['leave_one_WT_out']['c'],3.)


class InventoryIdentityTests(unittest.TestCase):
    """Exercise the complete context path with ten opaque synthetic checkpoints."""
    def setUp(self):
        fixture=fixtures.ReviewTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups); self.fixture=fixture; root=fixture.root.resolve()
        fixture.root=root
        fixture.paths={key:([p.resolve() for p in value] if isinstance(value,list) else value.resolve())
                       for key,value in fixture.paths.items()}
        protocol=pred.core.read_json(fixture.paths['config'])
        protocol.update(primary_source='historical',labels=dict(ca_cutoff_angstrom=15.,
            min_sequence_separation=3,contact_cutoff_angstrom=8.,error_thresholds=[.5,1.,2.,4.],
            preservation_values=pred.VALUES.tolist()))
        fixture.dump(fixture.paths['config'],protocol)
        split=pred.core.read_json(fixture.paths['splits']); split['protocol_sha256']=fixture.sha('configs/protocol.json')
        fixture.dump(fixture.paths['splits'],split)
        seal=pred.core.read_json(root/'configs/execution_freeze.json')
        seal['input_hashes']={path:fixture.sha(path) for path in seal['input_hashes']}
        fixture.dump('configs/execution_freeze.json',seal)
        manifest=pred.common.read_jsonl(fixture.paths['manifest'])
        summaries=pred.common.read_jsonl(fixture.paths['label_summary'])
        for arm in pred.core.ARMS:
            for lr in (.0003,.001): fixture.make_run(arm,lr,manifest,summaries)
        pred.core.build_review(**fixture.paths)
        cfg=pred.core.read_json(Path(__file__).resolve().parents[2]/'configs/development_diagnostics_first_seed.json')
        cfg.update(root=str(root),primary_validation_candidates=6,primary_validation_WT=3)
        cfg['inputs'].update(protocol='configs/protocol.json',primary_manifest='manifests/primary.jsonl',
            splits='manifests/splits.json',label_summary='reports/labels.jsonl',cache_indices=['caches/index.jsonl'])
        fixture.dump('configs/diagnostics.json',cfg); self.config=root/'configs/diagnostics.json'

    def test_valid_inventory_accepts_opaque_checkpoints(self):
        context=pred.load_context(self.config,'prediction')
        self.assertEqual(context['learned']['selected']['D']['lr'],.0003)
        self.assertEqual(len(context['learned']['runs']),10)

    def test_nonselected_checkpoint_corruption_rejected(self):
        path=self.fixture.root/self.fixture.directory('D',.001)/'best.pt'
        path.write_bytes(b'changed opaque checkpoint')
        with self.assertRaisesRegex(ValueError,'checkpoint D hash differs'):
            pred.load_context(self.config,'prediction')
        self.assertFalse((self.fixture.root/'reports/cpu_development_diagnostics/prediction/report.json').exists())

    def test_nonselected_validation_corruption_rejected(self):
        path=self.fixture.root/self.fixture.directory('D',.001)/'validation_2/report.json'
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'run validation output report.json hash differs'):
            pred.load_context(self.config,'prediction')
        self.assertFalse((self.fixture.root/'reports/cpu_development_diagnostics/prediction/report.json').exists())

    def test_nested_run_symlinks_into_output_rejected_before_hashing(self):
        directory=self.fixture.root/self.fixture.directory('D',.001)
        output=self.fixture.root/'reports/cpu_development_diagnostics/prediction'
        output.mkdir(parents=True)
        for relative in ('run.json','best.pt','validation_2/report.json'):
            with self.subTest(relative=relative):
                source=directory/relative; data=source.read_bytes()
                target=output/relative.replace('/','_'); target.write_bytes(data)
                source.unlink()
                try:
                    try: os.symlink(target,source)
                    except (OSError,NotImplementedError) as error:
                        self.skipTest('symlink creation unavailable: '+str(error))
                    # Identical bytes make this a path-boundary regression, not
                    # a checksum failure or malformed JSON test.
                    with self.assertRaisesRegex(ValueError,'output contains an input'):
                        pred.load_context(self.config,'prediction')
                finally:
                    source.unlink(missing_ok=True); source.write_bytes(data)


if __name__=='__main__': unittest.main()
