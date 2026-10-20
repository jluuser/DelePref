import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recomb_tm_interval_20261017 import stable_interval


class IntervalTest(unittest.TestCase):
    def test_zero_unknown_exactly_collapsed(self):
        r=stable_interval(dict(lower=1/3,upper=(1/3+200-200),unknown_retained_residues=0,full_wt_length=1))
        self.assertEqual(r['lower'],r['upper'])
    def test_known_unknown_fraction(self):
        r=stable_interval(dict(lower=.3,upper=.35,unknown_retained_residues=5,full_wt_length=100))
        self.assertAlmostEqual(r['upper'],.35)
    def test_invalid_interval_not_masked(self):
        with self.assertRaises(ValueError):stable_interval(dict(lower=.3,upper=.2,unknown_retained_residues=0,full_wt_length=100))


if __name__=='__main__':unittest.main()
