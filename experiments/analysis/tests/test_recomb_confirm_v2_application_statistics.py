import unittest
from recomb_confirm_v2_application_evaluate import seed_mean,summary_entry,endpoint_bounds


class ApplicationStatisticsTests(unittest.TestCase):
    def test_three_seeds_are_averaged_within_WT(self):
        lo,hi=seed_mean([(.1,.9),(.3,.8),(.8,1.)])
        self.assertAlmostEqual(lo,.4);self.assertAlmostEqual(hi,.9)

    def test_missing_seed_is_not_silently_averaged(self):
        self.assertEqual(seed_mean([(.1,.1),(.2,.2)]),(None,None))
        self.assertEqual(seed_mean([(.1,.1),(None,None),(.3,.3)]),(None,None))

    def test_missing_structural_outcome_keeps_full_denominator(self):
        rows=[dict(common_structure=dict(full_WT_TM=.7)),dict(common_structure={})]
        result=summary_entry(rows,'full_WT_TM')
        self.assertEqual(result['N_all'],2);self.assertEqual(result['N_defined'],1)
        self.assertIsNone(result['mean_lower'])

    def test_identity_ambiguity_is_an_interval(self):
        row=dict(catalytic_identity=dict(all_verified_identities_retained_lower=False,all_verified_identities_retained_upper=True))
        self.assertEqual(endpoint_bounds(row,'complete_catalytic_AA_identity'),(0.,1.))


if __name__=='__main__':unittest.main()
