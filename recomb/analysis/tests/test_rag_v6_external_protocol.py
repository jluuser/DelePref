import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rag_v6_external_evaluate as e
import rag_v6_external_protocol as p


class ExternalProtocolTests(unittest.TestCase):
    def test_ambiguous_map_uses_uniform_mean_and_preserves_observation(self):
        rows=[dict(row_id='r',possible_deleted_positions=[1,2],measurement=-999)]
        entries=[dict(row_id='r',deleted_position=1,score=0),dict(row_id='r',deleted_position=2,score=1)]
        self.assertEqual(e.predictions(rows,entries,'mapping'),{'r':.5})
        rows[0]['measurement']=999
        self.assertEqual(e.predictions(rows,entries,'mapping'),{'r':.5})

    def test_missing_extra_duplicate_or_nonfinite_predictions_fail(self):
        rows=[dict(row_id='a',possible_deleted_positions=[0])]
        for entries in [[],[dict(row_id='b',score=1)],[dict(row_id='a',score=1)]*2,[dict(row_id='a',score=float('nan'))]]:
            with self.assertRaises(ValueError):e.predictions(rows,entries,'row')

    def test_mapping_cannot_pick_best_scored_map(self):
        rows=[dict(row_id='r',possible_deleted_positions=[0,1])]
        with self.assertRaises(ValueError):e.predictions(rows,[dict(row_id='r',deleted_position=1,score=1)],'mapping')

    def test_ties_and_constant_correlations(self):
        np.testing.assert_array_equal(e.average_ranks([3,1,1,4]),[3,1.5,1.5,4])
        self.assertAlmostEqual(e.spearman([1,2,2,4],[4,3,3,1]),-1)
        self.assertIsNone(e.spearman([1,2],[1,1]))
        self.assertIsNone(e.spearman([1],[2]))

    def test_duplicate_physical_measurements_merge_without_favorable_filtering(self):
        rows=[dict(row_id='a',assay_id='x',wt_sequence='AACD',sequence='ACD',target='T',measurement=0),
              dict(row_id='b',assay_id='x',wt_sequence='AACD',sequence='ACD',target='T',measurement=2),
              dict(row_id='legacy',assay_id='urn:mavedb:00000100-b-1',wt_sequence='AACD',sequence='ACD',target='T',measurement=100)]
        result=p.physical_rows(rows)
        self.assertEqual(len(result),1);self.assertEqual(result[0]['measurement'],1)
        self.assertEqual(result[0]['possible_deleted_positions'],[0,1])

    def test_assays_then_proteins_are_balanced_not_rows(self):
        rows=[];scores={}
        for target,assay,values in [('A','a1',[1,2,3]),('A','a2',[3,2,1]),('B','b',[1,2,3]),('C','singleton',[1])]:
            for i,value in enumerate(values):
                key=target+assay+str(i);rows.append(dict(row_id=key,target=target,assay_id=assay,measurement=value));scores[key]=i
        r=e.evaluate(rows,scores)
        self.assertAlmostEqual(r['target_mean_Spearman']['A'],0)
        self.assertAlmostEqual(r['macro_Spearman'],.5)
        self.assertEqual(r['eligible_targets'],2);self.assertEqual(r['total_targets'],3)

    def test_constant_method_does_not_silently_drop_a_target(self):
        rows=[dict(row_id=str(i),target='a' if i<2 else 'b',assay_id='x',measurement=i%2) for i in range(4)]
        r=e.evaluate(rows,{'0':0,'1':1,'2':1,'3':1})
        self.assertIsNone(r['macro_Spearman']);self.assertEqual(r['constant_prediction_targets'],['b'])

    def test_bootstrap_resamples_physical_sequences_across_conditions(self):
        rows=[];scores={}
        for assay in ['a','b']:
            for i,sequence in enumerate(['AA','AC','AD']):
                key=assay+str(i);rows.append(dict(row_id=key,target='T',assay_id=assay,sequence=sequence,measurement=i));scores[key]=i
        result=e.target_cluster_intervals(rows,scores,replicates=20)['T']
        self.assertEqual(result['sequence_clusters'],3);self.assertEqual(result['fixed_conditions'],2)
        self.assertEqual(result['defined_replicates']+result['undefined_replicates'],20)
        np.testing.assert_allclose(result['conditional_CI95'],[1,1])


if __name__=='__main__':unittest.main()
