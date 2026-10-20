import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recomb_confirm_v2_diagnostics import incumbent_audit


def pool(prefix, top):
    return [dict(deletion_mask_sha256=prefix+str(i), scorer_score=top-i,
                 scorer_rank=i+1, generation_kind="test") for i in range(8)]


class QuotaCounterfactualTests(unittest.TestCase):
    def test_absent_incumbent_alone_is_not_causal_evidence(self):
        result = incumbent_audit(pool("a", 3), pool("b", 4))
        self.assertFalse(result["original_winner_in_RAG_pool"])
        self.assertFalse(result["reinsertion_strictly_changes_scorer_choice"])

    def test_removed_stronger_incumbent_is_explicit_counterfactual(self):
        self.assertTrue(incumbent_audit(pool("a", 4), pool("b", 3))[
            "reinsertion_strictly_changes_scorer_choice"])

    def test_numerical_tie_does_not_overstate_quota_effect(self):
        result = incumbent_audit(pool("a", 3+1e-6), pool("b", 3))
        self.assertFalse(result["reinsertion_strictly_changes_scorer_choice"])
        self.assertTrue(result["reinsertion_tie_unresolved"])


if __name__ == "__main__":
    unittest.main()
