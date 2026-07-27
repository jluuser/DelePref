import importlib.util
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "audit_embedding_similarity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_embedding_similarity", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EmbeddingSimilarityAuditTests(unittest.TestCase):
    def test_similarity_from_shift(self):
        audit = load_module()

        self.assertAlmostEqual(audit.similarity_from_shift(0.0), 1.0)
        self.assertAlmostEqual(audit.similarity_from_shift(0.01), 0.98)
        self.assertAlmostEqual(audit.similarity_from_shift("0.125"), 0.75)

    def test_parse_policy_and_budget_from_formal_id(self):
        audit = load_module()
        perturbation_id = "UniRef50_X__formalqv1__uniform_random_interval__b10__007__12-21"

        parsed = audit.parse_policy_budget_from_perturbation_id(perturbation_id)

        self.assertEqual(parsed["generation_policy"], "uniform_random_interval")
        self.assertEqual(parsed["budget_bin"], "b10")

    def test_pair_quality_classification(self):
        audit = load_module()

        self.assertEqual(
            audit.classify_pair_quality(
                preference="a_better",
                preference_confidence=0.9,
                preferred_global=0.99,
                preferred_retained=0.96,
                dispreferred_global=0.80,
                dispreferred_retained=0.70,
            ),
            "clean_preference",
        )
        self.assertEqual(
            audit.classify_pair_quality(
                preference="a_better",
                preference_confidence=0.9,
                preferred_global=0.97,
                preferred_retained=0.93,
                dispreferred_global=0.90,
                dispreferred_retained=0.85,
            ),
            "low_vs_lower",
        )
        self.assertEqual(
            audit.classify_pair_quality(
                preference="a_better",
                preference_confidence=0.9,
                preferred_global=0.94,
                preferred_retained=0.89,
                dispreferred_global=0.93,
                dispreferred_retained=0.88,
            ),
            "both_low",
        )
        self.assertEqual(
            audit.classify_pair_quality(
                preference="a_better",
                preference_confidence=0.9,
                preferred_global=0.96,
                preferred_retained=0.91,
                dispreferred_global=0.98,
                dispreferred_retained=0.95,
            ),
            "conflicting",
        )
        self.assertEqual(
            audit.classify_pair_quality(
                preference="tie",
                preference_confidence=0.5,
                preferred_global=0.99,
                preferred_retained=0.96,
                dispreferred_global=0.99,
                dispreferred_retained=0.96,
            ),
            "tie_or_uncertain",
        )


if __name__ == "__main__":
    unittest.main()
