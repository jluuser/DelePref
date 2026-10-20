import unittest
from recomb_confirm_v2_general_supplement import pool_summary


class PoolSummaryTest(unittest.TestCase):
    def test_incomplete_pool_is_not_an_oracle(self):
        with self.assertRaises(ValueError):pool_summary([.5]*7,.5)

    def test_missing_candidate_keeps_undefined_outcome(self):
        self.assertNotIn('best',pool_summary([.5]*7+[None],.5))

    def test_regret_is_candidate_best_minus_fixed_selection(self):
        r=pool_summary([.1,.2,.3,.4,.5,.6,.7,.8],.3)
        self.assertAlmostEqual(r['regret'],.5)
        with self.assertRaises(ValueError):pool_summary([.5]*8,.6)


if __name__=='__main__':unittest.main()
