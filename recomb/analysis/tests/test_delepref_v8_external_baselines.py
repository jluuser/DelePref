from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from delepref_v8_external_baseline_core import (load_original_teacher, original_utility,
                                              residue_deletion_probabilities, scisor_mapping_views, paired_target_difference)


class ExternalBaselineTests(unittest.TestCase):
    def test_special_tokens_and_shift_invariance(self):
        a = residue_deletion_probabilities([1e6, 1., 2., 3., 1e6], 3)
        b = residue_deletion_probabilities([-1e6, 1001., 1002., 1003., -1e6], 3)
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(a.sum(), 1.)
        self.assertTrue(a[2] > a[1] > a[0])
        with self.assertRaises(ValueError):
            residue_deletion_probabilities([0, 1, 2], 3)

    def test_repeated_residue_event_mean_and_sequence_sum(self):
        links = [dict(row_id="repeat", deleted_position=1, edit_key="a"),
                 dict(row_id="repeat", deleted_position=2, edit_key="b"),
                 dict(row_id="unique", deleted_position=3, edit_key="c")]
        primary, marginal = scisor_mapping_views(links, {"a": .1, "b": .2, "c": .3})
        self.assertAlmostEqual(np.mean([r["score"] for r in primary[:2]]), .15)
        self.assertAlmostEqual(marginal[0]["score"], .3)
        self.assertAlmostEqual(primary[-1]["score"], marginal[-1]["score"])
        with self.assertRaises(ValueError):
            scisor_mapping_views(links, {"a": .1, "b": .2})

    def test_original_teacher_views_on_synthetic_embeddings(self):
        import torch
        module = load_original_teacher(Path(__file__).resolve().parents[2])
        wt = torch.eye(6)
        kept = [0, 1, 3, 4, 5]
        mutant = wt[kept]
        mapping = module.retained_residue_mapping(6, [(2, 2)])
        self.assertEqual(module.mean_retained_distance(torch, wt, mutant, mapping), 0.)
        self.assertEqual(module.junction_plausibility_proxy(torch, wt, mutant, 6, [(2, 2)])[0], 1.)
        self.assertEqual(module.junction_plausibility_proxy(torch, wt, wt[1:], 6, [(0, 0)])[0], 1.)
        self.assertAlmostEqual(module.combine_junction_values([.2, .8]), .35)
        self.assertAlmostEqual(original_utility(dict(scoring_status="ok", global_similarity=.2,
                                                  retained_similarity=.4, junction_plausibility=.6)), .37)

    def test_paired_targets_and_undefined_predictions(self):
        result = paired_target_difference({"a": .1, "b": .5}, {"a": 0., "b": .2})
        self.assertAlmostEqual(result["difference"], .2)
        result = paired_target_difference({"a": .1, "b": None}, {"a": 0., "b": .2})
        self.assertIsNone(result["difference"])
        self.assertEqual(result["targets"], 2)
        with self.assertRaises(ValueError):
            paired_target_difference({"a": .1}, {"b": .2})


if __name__ == "__main__":
    unittest.main()
