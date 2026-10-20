import unittest
import numpy as np
from recomb_confirm_v2_mechanisms import morphology,shape,source_balanced


class Interventions(unittest.TestCase):
    def test_full_seed_position_changes_preserving_exact_shape(self):
        parent=dict(spans=((5,7),),seed_spans=((5,7),),source_target='reviewed-example',kind='retrieval_direct')
        items,d=morphology(np.arange(20,dtype=float),.15,[parent],13)
        self.assertTrue(d[0]['effective_final_shape_matched_intervention'])
        self.assertNotEqual(items[0]['spans'],parent['spans'])
        self.assertEqual(shape(items[0]['spans'],20),shape(parent['spans'],20))

    def test_unmovable_terminal_shape_remains_explicitly_unperturbed(self):
        parent=dict(spans=((0,2),),seed_spans=((0,2),),source_target='reviewed-example',kind='retrieval_direct')
        items,d=morphology(np.arange(20,dtype=float),.15,[parent],13)
        self.assertFalse(d[0]['effective_final_shape_matched_intervention'])
        self.assertEqual(items,[parent])

    def test_input_selection_ignores_any_quality_field(self):
        rows=[dict(protein_id=str(i),WT_sha256=str(i),stratum='chain' if i%2 else 'domain',quality=i) for i in range(20)]
        a=source_balanced(rows,8,'fixed')
        b=source_balanced([dict(r,quality=-r['quality']) for r in reversed(rows)],8,'fixed')
        self.assertEqual([r['protein_id'] for r in a],[r['protein_id'] for r in b])


if __name__=='__main__':unittest.main()
