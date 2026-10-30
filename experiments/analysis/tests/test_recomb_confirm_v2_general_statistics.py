import unittest
from recomb_confirm_v2_general_evaluate import effect


class ComponentStatistics(unittest.TestCase):
    def test_constant_effect_with_unequal_component_sizes(self):
        r=effect([('a',.2),('b',.2),('c',.2)],{'a':'x','b':'x','c':'y'})
        self.assertEqual(r['N'],3);self.assertEqual(r['components'],2)
        self.assertAlmostEqual(r['mean_difference'],.2)
        for value in r['CI95']+r['joint_Bonferroni_CI97_5']:self.assertAlmostEqual(value,.2)

    def test_missing_case_is_not_removed_to_produce_a_CI(self):
        r=effect([('a',.2),('b',None)],{'a':'x','b':'y'})
        self.assertEqual(r['N'],2);self.assertEqual(r['N_defined'],1)
        self.assertNotIn('CI95',r)


if __name__=='__main__':unittest.main()
