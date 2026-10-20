import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import delepref_unlimited_planning as planning
import recomb_confirm_v2_E9_completion as e9


class MatchedCompletionInvariants(unittest.TestCase):
    def setUp(self):
        self.seed = dict(spans=((10, 12),), target='reviewed_ordinary_source',
            all_reliable_events_used=True, deleted=3)
        self.logits = np.zeros(60)

    def completed(self, start):
        mask = np.zeros(60, dtype=bool)
        mask[10:13] = True
        mask[start:start+9] = True
        return planning.intervals(mask)

    def originals(self):
        return [dict(deleted_intervals=json.dumps(((i, i+11),)), scorer_score=float(i)) for i in range(8)]

    def test_both_decoders_preserve_seed_anchors_and_exact_budget(self):
        forced, kept, count = e9.constraints(60, self.seed)
        self.assertEqual(count, 9)
        for attempt in range(10):
            for result in [e9.policy_completion(self.logits, self.seed, 13, attempt),
                           e9.legal_completion(60, self.seed, 13, attempt)]:
                mask = planning.validate_plan(60, .2, result)
                self.assertTrue(mask[forced].all())
                self.assertFalse(mask[kept].any())

    def test_legal_sampler_has_no_learned_inputs_and_never_calls_objective(self):
        self.assertEqual(list(inspect.signature(e9.legal_completion).parameters),
                         ['length', 'seed', 'condition_seed', 'attempt', 'beta'])
        with patch.object(planning, 'complete', side_effect=AssertionError('learned completion used')):
            a = e9.legal_completion(60, self.seed, 13, 7)
            self.assertEqual(a, e9.legal_completion(60, copy.deepcopy(self.seed), 13, 7))
        self.assertNotEqual(a, e9.legal_completion(60, self.seed, 13, 8))

    def test_first_two_policy_decodes_match_original_retrieved_completion(self):
        projected = 'A'*10+'---'+'A'*47
        profile = dict(representatives=[dict(target=self.seed['target'], projected=projected)])
        seed = planning.seed_sets(profile, 60, .2)[0]
        proposals, attempts = planning.retrieved_proposals(self.logits, .2, profile, 13)
        self.assertEqual([a['status'] for a in attempts], ['accepted', 'accepted'])
        for i in range(2):
            self.assertEqual(e9.policy_completion(self.logits, seed, 13, i), proposals[i]['spans'])

    def test_duplicate_rejection_is_paired_with_equal_quotas_and_attempt_slots(self):
        old = self.originals()
        old[0]['deleted_intervals'] = json.dumps(self.completed(20))
        choices = [(20, 30), (21, 31), (21, 32), (22, 33)]
        def propose(attempt):
            a, b = choices[attempt]
            return {e9.ARMS[0]: self.completed(a), e9.ARMS[1]: self.completed(b)}
        result, cost = e9.paired_candidates(self.logits, self.seed, 13, old, max_attempts=4, proposer=propose)
        self.assertEqual([r['pair_accepted'] for r in cost['attempt_log']], [False, True, False, True])
        self.assertEqual(cost['attempts'], 4)
        self.assertEqual(cost['paired_retrieval_slots'], 2)
        self.assertEqual(cost['common_original_policy_slots'], 6)
        self.assertEqual(cost['shortfall'], 2)
        for arm in e9.ARMS:
            self.assertEqual([r['paired_attempt_index'] for r in result[arm]], [1, 3])
            self.assertEqual([r['seed_spans'] for r in result[arm]], [self.seed['spans']]*2)

    def test_score_tie_uses_integer_interval_order_and_never_policy_energy(self):
        rows = [dict(deleted_intervals='[[10, 11]]', scorer_score=2., policy_energy=100.),
                dict(deleted_intervals='[[2, 3]]', scorer_score=2., policy_energy=-100.)]
        ranked = e9.ranked(rows)
        self.assertEqual([r['scorer_rank'] for r in ranked], [2, 1])
        self.assertNotIn('scorer_rank', rows[0])

    def test_input_roster_is_invariant_to_order_and_unrelated_outcome_fields(self):
        panel, profiles = [], {}
        for i, letter in enumerate('ACDE'):
            sequence = letter*60
            pid = 'ordinary_'+str(i)
            panel.append(dict(protein_id=pid, sequence=sequence,
                WT_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                WT_encoding_allowed=True, deletion_generation_allowed=False,
                source_scope_review_status='PASS_SOURCE_AND_FUNCTION_METADATA'))
            profiles[pid] = dict(representatives=[dict(target='reviewed_'+str(i), projected=letter*10+'---'+letter*47)])
        first = e9.input_roster(panel, profiles, count=2)
        altered = [dict(r, old_quality=1e9*(i % 2), old_RAG_winner=bool(i % 2)) for i, r in enumerate(reversed(panel))]
        self.assertEqual(first, e9.input_roster(altered, profiles, count=2))

    def test_infeasible_or_complete_seed_is_rejected_without_weakening_constraints(self):
        for s in [dict(self.seed, spans=((0, 2),)), dict(self.seed, spans=((10, 21),)),
                  dict(self.seed, spans=((10, 11), (12, 13)))]:
            with self.assertRaises(ValueError):
                e9.constraints(60, s)


if __name__ == '__main__':
    unittest.main()
