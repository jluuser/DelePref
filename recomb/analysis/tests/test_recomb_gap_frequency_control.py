import unittest
import numpy as np
from recomb_gap_frequency_control_20261017 import frequencies, proposals
import delepref_unlimited_planning as planning


class GapFrequencyTests(unittest.TestCase):
    def test_unknown_does_not_change_denominator(self):
        profile = dict(representatives=[dict(target='g', projected='AAA--AAA'),
            dict(target='a', projected='AAAAAAAA'), dict(target='u', projected='????????')])
        rates, counts, denominators = frequencies(profile, 8)
        self.assertEqual(rates.tolist(), [0, 0, 0, .5, .5, 0, 0, 0])
        self.assertEqual(denominators.tolist(), [2]*8)

    def test_terminal_or_unanchored_gaps_are_not_evidence(self):
        profile = dict(representatives=[dict(target='terminal', projected='--AAAAAA'),
            dict(target='bad', projected='AA?--?AA')])
        self.assertEqual(frequencies(profile, 8)[0].sum(), 0)

    def test_exact_budget_and_forced_positions(self):
        profile = dict(representatives=[dict(target='g', projected='AA----AA')])
        items, d = proposals(np.arange(8, dtype=float), .5, profile)
        self.assertTrue(items)
        self.assertFalse(d['same_event_cooccurrence_used'])
        for item in items:
            mask = planning.validate_plan(8, .5, item['spans'])
            for a, b in item['seed_spans']:
                self.assertTrue(mask[a:b+1].all())

    def test_no_evidence_uses_identical_original_pool(self):
        items, diagnostic = proposals(np.arange(8, dtype=float), .5, dict(representatives=[]))
        original = [dict(spans=((i, i),)) for i in range(8)]
        mixed, fallback = planning.mix_candidates(original, items)
        self.assertEqual(mixed, original)
        self.assertTrue(fallback)


if __name__ == '__main__':
    unittest.main()
