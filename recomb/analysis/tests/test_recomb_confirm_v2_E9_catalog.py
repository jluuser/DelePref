import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import delepref_unlimited_planning as planning
import recomb_confirm_v2_E9_catalog as catalog

ROOT = Path(__file__).resolve().parents[2]


class E9FrozenWinnerRelease(unittest.TestCase):
    def test_all20_paired_pools_winners_and_admitted_slots_are_frozen_and_valid(self):
        directory = ROOT/'reports/recomb_confirm_v2_E9_completion'
        pools = json.loads((directory/'candidate_pools.json').read_text())
        winners = json.loads((directory/'selected_structure_outputs.json').read_text())
        behavior = json.loads((directory/'behavior.json').read_text())
        self.assertEqual((len(pools), len(winners), len(behavior)), (40, 40, 20))
        indexed = {(r['protein_id'], r['method']): r for r in winners}
        self.assertEqual(len(indexed), 40)
        by_pool = {(r['protein_id'], r['method']): r for r in pools}
        for d in behavior:
            self.assertEqual(d['policy_completion']['admitted_attempt_indices'],
                             d['simple_legal_completion']['admitted_attempt_indices'])
            self.assertEqual(d['paired_retrieval_slots']+d['common_original_policy_slots'], 8)
            for arm in ['policy_completion', 'simple_legal_completion']:
                pool = by_pool[d['protein_id'], arm]['rows']
                self.assertEqual(len({catalog.spans(r) for r in pool}), 8)
                self.assertEqual(sum(r['generation_kind'].startswith('retrieval') for r in pool), d['paired_retrieval_slots'])
                winner = indexed[d['protein_id'], arm]
                expected = min(pool, key=lambda r: (-r['scorer_score'], catalog.spans(r)))
                self.assertEqual(winner['deletion_mask_sha256'], expected['deletion_mask_sha256'])
                seq = winner['wildtype_sequence']
                edited, mapping = catalog.explicit_correspondence(seq, catalog.spans(winner))
                self.assertEqual(edited, winner['edited_sequence'])
                self.assertEqual(hashlib.sha256(edited.encode()).hexdigest(), winner['edited_sequence_sha256'])
                for row in pool:
                    mask = planning.validate_plan(len(seq), .2, catalog.spans(row))
                    if row['generation_kind'].startswith('retrieval'):
                        self.assertEqual(row['retrieval_seed_spans'], d['seed']['spans'])
                        for a, b in row['retrieval_seed_spans']:
                            self.assertTrue(mask[a:b+1].all())
                            self.assertFalse(mask[a-1] or mask[b+1])

    def test_new_predictor_release_changes_only_the_three_output_namespaces(self):
        old = (ROOT/'analysis/recomb_confirm_v2_general_predict.py').read_text()
        new = catalog.derive_predictor(old)
        self.assertIn("CAT=Path('reports/recomb_confirm_v2_E9_catalog')", new)
        self.assertIn("FREEZE=Path('configs/recomb_confirm_v2_E9_confirmation_freeze.json')", new)
        functions = lambda s: [ast.dump(n, include_attributes=False) for n in ast.parse(s).body if isinstance(n, ast.FunctionDef)]
        self.assertEqual(functions(old), functions(new))

    def test_comparison_is_completion_rule_and_sample_count_not_K12_capacity(self):
        source = (ROOT/'analysis/recomb_confirm_v2_E9_catalog.py').read_text()
        self.assertIn("requested_comparisons=[['policy_completion', 'simple_legal_completion']]", source)
        self.assertIn('logical=40, fixed_WT=20', source)
        evaluator = (ROOT/'analysis/recomb_confirm_v2_E9_evaluate.py').read_text()
        self.assertIn("from recomb_confirm_v2_general_evaluate import metric_pair, effect", evaluator)
        self.assertIn("joint_Bonferroni_CI97_5_conservative_envelope", evaluator)


if __name__ == '__main__':
    unittest.main()
