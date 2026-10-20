import unittest
from recomb_rag_decision_audit_20261017 import category


class DecisionProvenanceTests(unittest.TestCase):
    def test_retrieval_provenance_is_not_a_mask_change(self):
        self.assertEqual(category('retrieval_direct', 1, False, False), 'direct_guidance_selected')

    def test_evidence_proposal_can_change_original_winner_without_winning(self):
        self.assertEqual(category('original_policy_unlimited', 4, True, True), 'retrieval_proposed_policy_selected')

    def test_exact_fallback(self):
        self.assertEqual(category('original_policy_unlimited', 0, False, False), 'no_usable_proposal_fallback')
        with self.assertRaises(ValueError):
            category('original_policy_unlimited', 0, True, False)

    def test_partial_completion_is_not_direct_natural_plan(self):
        for kind in ('retrieval_completed', 'retrieval_refined'):
            self.assertEqual(category(kind, 2, True, True), 'partial_guidance_selected')


if __name__ == '__main__':
    unittest.main()
