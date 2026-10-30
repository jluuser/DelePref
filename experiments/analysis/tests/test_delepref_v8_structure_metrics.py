import os
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from delepref_v8_structure_metrics import relationship_metrics, common_retained_diagnostic, fixed_full_wt_tm


class MetricsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.ref = rng.normal(size=(40, 3)) * 4

    def test_identity_rigid_and_deleted_denominator(self):
        kept = np.arange(40)
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        out = relationship_metrics(self.ref, self.ref @ rotation + 32, kept)
        self.assertEqual(out["T_lower"], 1.)
        self.assertEqual(out["contact_recovery_lower"], 1.)
        kept = np.arange(8, 40)
        out = relationship_metrics(self.ref, self.ref[kept], kept)
        self.assertEqual(out["T_lower"], out["C"])
        self.assertLess(out["T_lower"], 1.)
        self.assertEqual(out["retained_ca_lddt_lower"], 1.)
        self.assertAlmostEqual(out["contact_recovery_lower"], 1. - out["static_contact_loss"])

    def test_missing_and_failure_are_not_dropped(self):
        kept = np.arange(5, 40)
        out = relationship_metrics(self.ref, np.full((35, 3), np.nan), kept)
        self.assertEqual(out["T_lower"], 0.)
        self.assertEqual(out["T_upper"], out["C"])
        self.assertEqual(out["retained_ca_lddt_upper"], 1.)
        pred = self.ref[kept].copy(); pred[4] = np.nan
        out = relationship_metrics(self.ref, pred, kept)
        self.assertGreater(out["unknown_pairs"], 0)
        self.assertGreater(out["T_upper"], out["T_lower"])
        self.assertAlmostEqual(out["T_upper"], out["C"])

    def test_common_relations_and_invalid_mapping(self):
        x, y = np.arange(3, 40), np.arange(0, 35)
        out = common_retained_diagnostic(self.ref, self.ref[x], x, self.ref[y], y)
        self.assertEqual(out["difference_lower"], 0.)
        self.assertEqual(out["common_retained_residues"], 32)
        with self.assertRaises(ValueError):
            relationship_metrics(self.ref, self.ref[[3, 2]], [3, 2])

    def test_per_residue_lddt_is_not_pair_average(self):
        ref = np.zeros((10, 3)); ref[:, 0] = np.arange(10) * 1000
        ref[[0, 3, 6, 9], 0] = [0, 1, 2, 16]
        pred = ref.copy(); pred[[0, 3], 0] = [100, 200]
        out = relationship_metrics(ref, pred, np.arange(10))
        # Triangle plus one leaf: degrees 2,2,3,1; only the leaf edge
        # remains accurate. Residue weighting differs from pair weighting.
        self.assertEqual(out["G_lower"], .25)
        self.assertAlmostEqual(out["retained_ca_lddt_lower"], 1. / 3.)

    @unittest.skipUnless(os.environ.get("V8_USALIGN_BINARY"), "official binary required")
    def test_official_fixed_tm_perfect_deletion_and_missing(self):
        kept = np.arange(8, 40)
        pred = self.ref[kept] @ np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]) + 20
        result = fixed_full_wt_tm(self.ref, pred, kept, os.environ["V8_USALIGN_BINARY"])
        self.assertAlmostEqual(result["lower"], .8, places=6)
        self.assertAlmostEqual(result["upper"], .8, places=6)
        pred[0] = np.nan
        result = fixed_full_wt_tm(self.ref, pred, kept, os.environ["V8_USALIGN_BINARY"])
        self.assertAlmostEqual(result["lower"], 31 / 40, places=6)
        self.assertAlmostEqual(result["upper"], .8, places=6)


if __name__ == "__main__":
    unittest.main()
