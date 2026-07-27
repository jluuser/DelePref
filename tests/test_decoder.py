import itertools
import unittest

import torch

from delepref.student_model import (
    InfeasibleStrictBudgetError,
    generate_strict_budget_plans,
    strict_budget_dp_decode,
    strict_delete_count,
    validate_strict_plan,
)


def intervals_from_bits(bits):
    intervals = []
    start = None
    for index, bit in enumerate(bits):
        if bit and start is None:
            start = index
        elif not bit and start is not None:
            intervals.append((start, index - 1))
            start = None
    if start is not None:
        intervals.append((start, len(bits) - 1))
    return tuple(intervals)


def brute_force(logits, beta, qmax, min_span_len, max_span_len):
    length = len(logits)
    deleted = strict_delete_count(length, beta)
    best = None
    for selected in itertools.combinations(range(length), deleted):
        bits = [0] * length
        for index in selected:
            bits[index] = 1
        intervals = intervals_from_bits(bits)
        if len(intervals) > qmax:
            continue
        if any(end - start + 1 < min_span_len or end - start + 1 > max_span_len for start, end in intervals):
            continue
        score = sum(float(logits[index]) for index in selected)
        item = (score, intervals)
        if best is None or score > best[0] or (score == best[0] and intervals < best[1]):
            best = item
    if best is None:
        raise InfeasibleStrictBudgetError("brute force found no plan")
    return best[1], best[0]


class StrictDecoderTests(unittest.TestCase):
    def test_dp_matches_brute_force_optimum_for_small_sequences(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(101)
        cases = 0
        for length in range(2, 9):
            logits = torch.randn(length, generator=generator, dtype=torch.float64)
            for beta in (0.01, 0.25, 0.5, 0.99):
                for qmax in (1, 2, 3):
                    for min_len, max_len in ((1, 1), (1, 3), (2, 3)):
                        try:
                            expected_intervals, expected_score = brute_force(
                                logits.tolist(), beta, qmax, min_len, max_len
                            )
                        except InfeasibleStrictBudgetError:
                            with self.assertRaises(InfeasibleStrictBudgetError):
                                strict_budget_dp_decode(
                                    logits,
                                    beta,
                                    qmax=qmax,
                                    min_span_len=min_len,
                                    max_span_len=max_len,
                                )
                            continue
                        actual_intervals, actual_score = strict_budget_dp_decode(
                            logits,
                            beta,
                            qmax=qmax,
                            min_span_len=min_len,
                            max_span_len=max_len,
                        )
                        self.assertEqual(actual_intervals, expected_intervals)
                        self.assertAlmostEqual(actual_score, expected_score, places=12)
                        cases += 1
        self.assertGreater(cases, 100)

    def test_tie_breaking_is_deterministic_and_lexicographic(self):
        logits = torch.zeros(6)
        intervals, score = strict_budget_dp_decode(
            logits, 2.0 / 6.0, qmax=2, min_span_len=1, max_span_len=2
        )
        self.assertEqual(intervals, ((0, 1),))
        self.assertEqual(score, 0.0)
        for _ in range(20):
            repeated, _ = strict_budget_dp_decode(
                logits, 2.0 / 6.0, qmax=2, min_span_len=1, max_span_len=2
            )
            self.assertEqual(repeated, intervals)

    def test_terminal_single_multispan_small_and_D_equals_L_minus_one(self):
        left, _ = strict_budget_dp_decode(torch.tensor([9.0, 1.0, 0.0, 0.0]), 0.25, qmax=1)
        self.assertEqual(left, ((0, 0),))
        right, _ = strict_budget_dp_decode(torch.tensor([0.0, 0.0, 1.0, 9.0]), 0.25, qmax=1)
        self.assertEqual(right, ((3, 3),))
        multi, _ = strict_budget_dp_decode(
            torch.tensor([8.0, -9.0, 7.0, -9.0, 6.0]), 0.4, qmax=2, max_span_len=1
        )
        self.assertEqual(multi, ((0, 0), (2, 2)))
        tiny, _ = strict_budget_dp_decode(torch.tensor([2.0, 1.0]), 1.0, qmax=1)
        self.assertEqual(tiny, ((0, 0),))
        near_full, _ = strict_budget_dp_decode(
            torch.tensor([5.0, 4.0, -100.0, 3.0, 2.0]),
            1.0,
            qmax=2,
            min_span_len=2,
            max_span_len=2,
        )
        self.assertEqual(near_full, ((0, 1), (3, 4)))

    def test_infeasible_constraints_never_fall_back(self):
        with self.assertRaises(InfeasibleStrictBudgetError):
            strict_budget_dp_decode(
                torch.arange(7.0), 1.0, qmax=3, min_span_len=2, max_span_len=2
            )
        result = generate_strict_budget_plans(
            torch.arange(7.0),
            1.0,
            num_candidates=5,
            qmax=3,
            min_span_len=2,
            max_span_len=2,
            seed=3,
        )
        self.assertEqual(result.plans, ())
        self.assertEqual(result.reason, "infeasible_constraints")
        too_short = generate_strict_budget_plans(
            torch.tensor([1.0]), 0.5, num_candidates=3, qmax=1
        )
        self.assertEqual(too_short.plans, ())
        self.assertEqual(too_short.reason, "sequence_too_short")

    def test_K_generation_returns_actual_unique_count_and_diagnostics(self):
        logits = torch.tensor([3.0, 2.0])
        result = generate_strict_budget_plans(
            logits,
            0.5,
            num_candidates=8,
            qmax=1,
            seed=19,
            perturbation_scale=0.0,
            max_attempts=10,
        )
        self.assertEqual(len(result.plans), 1)
        self.assertEqual(result.requested_count, 8)
        self.assertEqual(result.duplicate_count, 9)
        self.assertEqual(result.reason, "unique_plan_space_exhausted_or_attempt_limit")

    def test_exact_budget_span_cap_nonoverlap_and_determinism_100_percent(self):
        for seed in range(25):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            logits = torch.randn(12, generator=generator)
            first = generate_strict_budget_plans(
                logits,
                0.34,
                num_candidates=5,
                qmax=3,
                min_span_len=1,
                max_span_len=4,
                seed=seed,
                perturbation_scale=0.7,
            )
            second = generate_strict_budget_plans(
                logits,
                0.34,
                num_candidates=5,
                qmax=3,
                min_span_len=1,
                max_span_len=4,
                seed=seed,
                perturbation_scale=0.7,
            )
            self.assertEqual(first, second)
            self.assertEqual(len({plan.deleted_intervals for plan in first.plans}), len(first.plans))
            for plan in first.plans:
                self.assertEqual(plan.deleted_length, strict_delete_count(12, 0.34))
                validate_strict_plan(
                    12,
                    0.34,
                    plan.deleted_intervals,
                    qmax=3,
                    min_span_len=1,
                    max_span_len=4,
                )

    def test_rejects_length_one_nonfinite_and_adjacent_spans(self):
        with self.assertRaises(ValueError):
            strict_budget_dp_decode(torch.tensor([1.0]), 0.5, qmax=1)
        with self.assertRaises(ValueError):
            strict_budget_dp_decode(torch.tensor([1.0, float("nan")]), 0.5, qmax=1)
        with self.assertRaises(ValueError):
            strict_budget_dp_decode(torch.tensor([1.0, 2.0]), 0.5, qmax=0)
        with self.assertRaises(ValueError):
            strict_budget_dp_decode(torch.tensor([1.0, 2.0]), 0.5, qmax=1, min_span_len=0)
        with self.assertRaisesRegex(ValueError, "maximal runs"):
            validate_strict_plan(
                5,
                0.4,
                ((0, 0), (1, 1)),
                qmax=2,
                min_span_len=1,
                max_span_len=1,
            )


if __name__ == "__main__":
    unittest.main()
