import importlib.util
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "audit_preference_pair_similarity.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_preference_pair_similarity", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PreferencePairSimilarityAuditTests(unittest.TestCase):
    def test_closed_inclusive_overlap_and_similarity(self):
        audit = load_audit_module()
        a = [(2, 4), (10, 10)]
        b = [(3, 5), (10, 10)]

        stats = audit.interval_similarity(a, b, wildtype_length=20)

        self.assertEqual(stats["intersection"], 3)
        self.assertEqual(stats["deleted_len_a"], 4)
        self.assertEqual(stats["deleted_len_b"], 4)
        self.assertAlmostEqual(stats["mask_jaccard"], 3 / 5)
        self.assertAlmostEqual(stats["overlap_coverage"], 3 / 4)
        self.assertGreater(stats["boundary_shift_mean"], 0.0)

    def test_relation_and_weight_are_bounded(self):
        audit = load_audit_module()
        features = {
            "budget_diff": 0.0,
            "mask_jaccard": 0.6,
            "overlap_coverage": 0.75,
            "center_distance_norm": 0.03,
            "boundary_shift_mean": 0.05,
            "topology_relation": "shifted_overlap",
            "teacher_margin": 0.12,
            "preference": "a_better",
            "preference_confidence": 0.8,
            "is_valid_pair": True,
        }

        relation = audit.classify_pair_relation(features)
        weight = audit.recommended_weight(features, relation)

        self.assertIn(relation, {"local_boundary_contrast", "local_shift_contrast"})
        self.assertGreaterEqual(weight, 0.1)
        self.assertLessEqual(weight, 1.0)
        self.assertGreater(weight, 0.5)


if __name__ == "__main__":
    unittest.main()
