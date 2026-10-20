from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import delepref_v8_external_inputs as p


class ExternalMappingTests(unittest.TestCase):
    def test_all_ambiguous_maps_retained_and_repeated_conditions_share_computation(self):
        rows=[dict(row_id=x,target='t',wt_sequence='AAAC',sequence='AAC',possible_deleted_positions=[0,1,2],measurement=y)
              for x,y in [('a',-100),('b',100)]]
        wt,edits,links=p.prepare_mappings(rows)
        self.assertEqual(len(wt),1);self.assertEqual(len(edits),3);self.assertEqual(len(links),6)
        pred=p.mapping_predictions(links,{x['edit_key']:i for i,x in enumerate(edits)})
        self.assertEqual([x['score'] for x in pred[:3]],[x['score'] for x in pred[3:]])
        changed=[dict(x,measurement=-x['measurement']) for x in rows]
        self.assertEqual(p.prepare_mappings(rows),p.prepare_mappings(changed))

    def test_wrong_mapping_and_missing_score_fail(self):
        row=dict(row_id='a',target='t',wt_sequence='ACDE',sequence='ADE',possible_deleted_positions=[1])
        _,edits,links=p.prepare_mappings([row])
        with self.assertRaises(ValueError):p.mapping_predictions(links,{})
        with self.assertRaises(ValueError):p.prepare_mappings([dict(row,possible_deleted_positions=[0])])
        with self.assertRaises(ValueError):p.prepare_mappings([row,row])


if __name__=='__main__':unittest.main()
