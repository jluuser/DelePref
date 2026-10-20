import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recomb_confirm_v2_E10_fidelity import microbatch_layout, padding_count, compare_pools


class FidelityIsolationTests(unittest.TestCase):
    def test_original_layout_deduplicates_and_sorts_identity_before_chunking(self):
        rows = [dict(protein_id=p, wildtype_sequence=s) for p, s in [('z', 'A'*10), ('a', 'C'*8), ('b', 'D'*9), ('a', 'C'*8)]]
        blocks = microbatch_layout(rows, 2)
        self.assertEqual([b['protein_ids'] for b in blocks], [['a', 'b'], ['z']])
        self.assertEqual([b['padded_token_width'] for b in blocks], [11, 12])
        self.assertEqual(blocks, microbatch_layout(list(reversed(rows)), 2))

    def test_padding_control_never_changes_sequence_length_or_truncates(self):
        self.assertEqual(padding_count(102, 132), 30)
        self.assertEqual(padding_count(102, 102), 0)
        with self.assertRaises(ValueError):
            padding_count(132, 102)

    def test_mask_pool_and_selection_changes_are_distinct_from_score_drift(self):
        a = [dict(deletion_mask_sha256='a', scorer_score=2., deleted_intervals='[[2,3]]'),
             dict(deletion_mask_sha256='b', scorer_score=1., deleted_intervals='[[4,5]]')]
        drift = [dict(r, scorer_score=r['scorer_score']+.01) for r in a]
        result = compare_pools(a, drift)
        self.assertEqual(result['common_masks'], 2)
        self.assertFalse(result['winner_mask_changed'])
        self.assertAlmostEqual(result['common_mask_score_max_abs_delta'], .01)
        changed = [a[1], dict(a[0], deletion_mask_sha256='c', scorer_score=3.)]
        result = compare_pools(a, changed)
        self.assertTrue(result['winner_mask_changed'])
        self.assertEqual((result['introduced_masks'], result['lost_masks']), (['c'], ['a']))


if __name__ == '__main__':
    unittest.main()
