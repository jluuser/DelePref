import copy
import unittest

from recomb_confirm_v2_K12_supplement import continue_unique, rag_union, selected, selection_order


def row(i, score=0.):
    return dict(deleted_intervals='[[%d,%d]]' % (i, i), scorer_score=score,
                scorer_rank=i+1, method='delepref_unlimited', deletion_mask_sha256=str(i))


class K12Tests(unittest.TestCase):
    def setUp(self):
        self.prefix = [row(i) for i in range(8)]

    def test_resume_skips_old_masks_and_never_redecodes_prefix(self):
        seen = []
        def propose(attempt):
            seen.append(attempt)
            return row(0 if attempt == 11 else attempt)
        original = copy.deepcopy(self.prefix)
        additions, audit = continue_unique(self.prefix, propose, 11)
        self.assertEqual(seen, [11, 12, 13, 14, 15])
        self.assertEqual([r['K12_attempt_index'] for r in additions], [12, 13, 14, 15])
        self.assertEqual(audit['new_duplicate_decodes'], 1)
        self.assertEqual(self.prefix, original)

    def test_attempt_cap_reports_shortfall_without_retries(self):
        seen = []
        def propose(attempt):
            seen.append(attempt)
            return row(0)
        additions, audit = continue_unique(self.prefix, propose, 190)
        self.assertEqual(seen, [190, 191])
        self.assertEqual(additions, [])
        self.assertEqual(audit['returned'], 8)

    def test_rag_union_preserves8_deduplicates_and_fills_in_order(self):
        retrieval = [row(6), row(20), row(21), row(20)]
        extras = [row(i) for i in range(8, 12)]
        combined = rag_union(self.prefix, retrieval, extras)
        self.assertEqual(combined[:8], self.prefix)
        self.assertEqual([r['deletion_mask_sha256'] for r in combined[8:]], ['20', '21', '8', '9'])

    def test_no_retrieval_fallback_is_exact_policy12(self):
        extras = [row(i) for i in range(8, 12)]
        self.assertEqual(rag_union(self.prefix, [], extras), self.prefix+extras)

    def test_scorer_tie_uses_integer_intervals_not_raw_order(self):
        pool = [row(10, .5), row(2, .5), row(1, .4)]
        self.assertEqual(selection_order(pool), [1, 0, 2])
        original = copy.deepcopy(pool)
        winner = selected('ABCDEFGHIJKLMNOPQRSTUVWXYZ', pool, 'rag_K12')
        self.assertEqual(winner['deletion_mask_sha256'], '2')
        self.assertEqual(winner['method'], 'rag_K12')
        self.assertEqual(pool, original)

    def test_malformed_prefix_and_more_than4_retrieval_rows_rejected(self):
        with self.assertRaises(ValueError):
            continue_unique(self.prefix[:7], lambda a: row(a), 8)
        with self.assertRaises(ValueError):
            rag_union(self.prefix, [row(i) for i in range(5)], [])


if __name__ == '__main__':
    unittest.main()
