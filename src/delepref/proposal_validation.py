"""Frozen-dev proposal validation and checkpoint selection for student DelePref."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


PRIMARY_METRIC = "teacher_rescored_selected_utility_equal_weight_per_protein_budget_seed"
BEST_OF_K_METRIC = "teacher_rescored_best_of_k"
STRICT_ACCURACY_METRIC = "strict_pair_accuracy"
FORBIDDEN_SELECTION_METRICS = frozenset(
    {
        "student_scorer_self_evaluation",
        "CATH",
        "Binder",
        "Enzyme",
        "Cas9",
        "confirmation",
    }
)


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(str(sequence).encode("utf-8")).hexdigest()


def strict_delete_count(length: int, beta: float) -> int:
    length = int(length)
    if length < 2:
        raise ValueError("proposal validation requires L >= 2")
    return max(1, min(length - 1, int(round(float(beta) * length))))


def canonical_intervals(value) -> Tuple[Tuple[int, int], ...]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError("deleted intervals must be a non-empty sequence")
    result = []
    previous_end = -2
    for interval in parsed:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("each deleted interval must contain start/end")
        start, end = int(interval[0]), int(interval[1])
        if start < 0 or end < start or start <= previous_end + 1:
            raise ValueError("deleted intervals must be ordered, maximal, and non-overlapping")
        result.append((start, end))
        previous_end = end
    return tuple(result)


def read_frozen_proteins(path, *, expected_count: int = 256):
    rows = []
    seen_proteins = set()
    seen_sequences = set()
    expected_rank = 1
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            protein_id = str(row.get("protein_id") or "")
            sequence = str(row.get("wildtype_sequence") or "")
            sequence_hash = str(row.get("sequence_sha256") or "")
            if int(row.get("selection_rank", -1)) != expected_rank:
                raise ValueError("proposal manifest selection ranks are not contiguous")
            if row.get("split") != "dev":
                raise ValueError("proposal validation may use only frozen dev proteins")
            if not protein_id or protein_id in seen_proteins:
                raise ValueError("duplicate or empty proposal-validation protein")
            if not sequence or len(sequence) != int(row["wildtype_length"]):
                raise ValueError("proposal-validation sequence length mismatch")
            if sequence_sha256(sequence) != sequence_hash or sequence_hash in seen_sequences:
                raise ValueError("proposal-validation sequence hash mismatch or duplicate")
            rows.append(
                {
                    "selection_rank": expected_rank,
                    "protein_id": protein_id,
                    "wildtype_sequence": sequence,
                    "sequence_sha256": sequence_hash,
                    "wildtype_length": len(sequence),
                    "split": "dev",
                }
            )
            seen_proteins.add(protein_id)
            seen_sequences.add(sequence_hash)
            expected_rank += 1
    if len(rows) != int(expected_count):
        raise ValueError("proposal manifest has {} proteins, expected {}".format(len(rows), expected_count))
    return tuple(rows)


def read_cold_start_references(path, *, expected_proteins: int = 256, budgets=(0.10, 0.20)):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("reference_role") != "diagnostic_only_not_checkpoint_selection":
                raise ValueError("cold-start reference role is not diagnostic-only")
            key = (row["protein_id"], round(float(row["beta"]), 8))
            if key in result:
                raise ValueError("duplicate cold-start reference {}".format(key))
            result[key] = {
                "teacher_utility": float(row["best_teacher_utility"]),
                "deleted_length": int(row["deleted_length"]),
                "absolute_gates_pass": str(row["absolute_gates_pass"]).lower() == "true",
                "plan_id": row["best_plan_id"],
            }
    expected = int(expected_proteins) * len(tuple(budgets))
    if len(result) != expected:
        raise ValueError("cold-start reference has {} rows, expected {}".format(len(result), expected))
    return result


def teacher_utility(record: Mapping[str, object]) -> float:
    global_similarity = float(record["global_similarity"])
    retained_similarity = float(record["retained_similarity"])
    junction = float(record["junction_plausibility"])
    if not all(math.isfinite(value) for value in (global_similarity, retained_similarity, junction)):
        raise ValueError("teacher views must be finite")
    return 0.35 * global_similarity + 0.45 * retained_similarity + 0.20 * junction


def _deleted_positions(intervals):
    return frozenset(position for start, end in intervals for position in range(start, end + 1))


def _pairwise_mean(items, distance):
    values = [distance(items[left], items[right]) for left in range(len(items)) for right in range(left + 1, len(items))]
    return sum(values) / len(values) if values else 0.0


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_proposal_records(
    records: Sequence[Mapping[str, object]],
    *,
    protein_ids: Sequence[str],
    budgets: Sequence[float],
    seeds: Sequence[int],
    requested_k: int,
    qmax: int = 3,
    cold_start_references: Optional[Mapping[Tuple[str, float], Mapping[str, object]]] = None,
):
    """Validate all generated plans and compute equal-condition teacher metrics."""

    expected_keys = {
        (str(protein_id), round(float(beta), 8), int(seed))
        for protein_id in protein_ids
        for beta in budgets
        for seed in seeds
    }
    grouped = defaultdict(list)
    failures = []
    for ordinal, record in enumerate(records):
        try:
            protein_id = str(record["protein_id"])
            beta = round(float(record["beta"]), 8)
            seed = int(record["seed"])
            key = (protein_id, beta, seed)
            length = int(record["wildtype_length"])
            if key not in expected_keys:
                raise ValueError("unexpected protein/budget/seed condition")
            if str(record.get("teacher_scoring_status")) != "ok":
                raise ValueError("real teacher scoring status is not ok")
            intervals = canonical_intervals(record["deleted_intervals"])
            if len(intervals) > int(qmax):
                raise ValueError("plan exceeds Qmax")
            if intervals[-1][1] >= length:
                raise ValueError("plan exceeds sequence length")
            positions = _deleted_positions(intervals)
            if len(positions) != strict_delete_count(length, beta):
                raise ValueError("plan violates exact deletion budget")
            score = float(record["student_scorer_score"])
            if not math.isfinite(score):
                raise ValueError("student scorer score is not finite")
            utility = teacher_utility(record)
            grouped[key].append(
                {
                    **record,
                    "deleted_intervals": intervals,
                    "positions": positions,
                    "teacher_utility": utility,
                    "student_scorer_score": score,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            failures.append({"record_ordinal": ordinal, "reason": str(exc)})

    missing = sorted(expected_keys - set(grouped))
    extra = sorted(set(grouped) - expected_keys)
    group_metrics = []
    exact_k = True
    unique_within_group = True
    for key in sorted(expected_keys):
        group = grouped.get(key, [])
        if len(group) != int(requested_k):
            exact_k = False
            continue
        masks = [item["positions"] for item in group]
        unique_masks = len(set(masks))
        if unique_masks != len(group):
            unique_within_group = False
        ordered = sorted(group, key=lambda item: (-item["student_scorer_score"], item["deleted_intervals"]))
        selected = ordered[0]
        best = max(group, key=lambda item: (item["teacher_utility"], tuple(reversed(item["deleted_intervals"]))))
        length = int(group[0]["wildtype_length"])
        centroids = [sum(item["positions"]) / len(item["positions"]) for item in group]
        topologies = {
            (len(item["deleted_intervals"]), tuple(end - start + 1 for start, end in item["deleted_intervals"]))
            for item in group
        }
        mask_diversity = _pairwise_mean(
            masks,
            lambda left, right: 1.0 - len(left & right) / float(len(left | right)),
        )
        position_diversity = _pairwise_mean(
            centroids,
            lambda left, right: abs(left - right) / float(max(1, length - 1)),
        )
        group_metrics.append(
            {
                "protein_id": key[0],
                "beta": key[1],
                "seed": key[2],
                "pool_mean": _mean(item["teacher_utility"] for item in group),
                "best_of_k": best["teacher_utility"],
                "student_selected_utility": selected["teacher_utility"],
                "oracle_gap": best["teacher_utility"] - selected["teacher_utility"],
                "effective_k": unique_masks,
                "mask_diversity": mask_diversity,
                "position_diversity": position_diversity,
                "topology_diversity": len(topologies) / float(max(1, unique_masks)),
                "selected_candidate_id": str(selected["candidate_id"]),
                "oracle_candidate_id": str(best["candidate_id"]),
            }
        )

    gates = {
        "no_record_validation_failures": not failures,
        "all_expected_conditions_present": not missing and not extra and len(grouped) == len(expected_keys),
        "exact_k_per_condition": exact_k,
        "unique_masks_within_condition": unique_within_group,
        "all_plans_exact_budget_qmax_nonoverlap": not failures,
        "all_teacher_scores_real_ok_finite": not failures,
    }
    status = "PASS" if all(gates.values()) and len(group_metrics) == len(expected_keys) else "FAIL"
    aggregate = {
        "teacher_rescored_pool_mean": _mean(item["pool_mean"] for item in group_metrics),
        BEST_OF_K_METRIC: _mean(item["best_of_k"] for item in group_metrics),
        PRIMARY_METRIC: _mean(item["student_selected_utility"] for item in group_metrics),
        "teacher_rescored_oracle_gap": _mean(item["oracle_gap"] for item in group_metrics),
        "effective_k_mean": _mean(item["effective_k"] for item in group_metrics),
        "mask_diversity_mean": _mean(item["mask_diversity"] for item in group_metrics),
        "position_diversity_mean": _mean(item["position_diversity"] for item in group_metrics),
        "topology_diversity_mean": _mean(item["topology_diversity"] for item in group_metrics),
    }
    diagnostic = {
        "role": "diagnostic_only_not_checkpoint_selection",
        "used_for_checkpoint_selection": False,
    }
    if cold_start_references is not None:
        cold_values = []
        selected_deltas = []
        for item in group_metrics:
            reference = cold_start_references[(item["protein_id"], round(item["beta"], 8))]
            cold = float(reference["teacher_utility"])
            cold_values.append(cold)
            selected_deltas.append(item["student_selected_utility"] - cold)
        diagnostic.update(
            {
                "teacher_rescored_cold_start_best_mean": _mean(cold_values),
                "selected_minus_cold_start_mean": _mean(selected_deltas),
            }
        )
    return {
        "schema_version": "delepref_real_teacher_proposal_validation_v1",
        "status": status,
        "real_teacher_rescoring": True,
        "teacher_utility": "0.35*global_similarity+0.45*retained_similarity+0.20*junction_plausibility",
        "equal_weighting_unit": "protein_budget_seed",
        "expected_conditions": len(expected_keys),
        "evaluated_conditions": len(group_metrics),
        "requested_k": int(requested_k),
        "record_count": len(records),
        "legality_gates": gates,
        "failure_examples": failures[:20],
        "missing_condition_examples": missing[:20],
        "metrics": aggregate,
        "cold_start_reference": diagnostic,
        "group_metrics": group_metrics,
        "teacher_preference_is_ground_truth": False,
    }


def checkpoint_candidate(
    *,
    checkpoint_path: str,
    step: int,
    proposal_report: Mapping[str, object],
    strict_pair_accuracy: float,
):
    metrics = proposal_report.get("metrics") or {}
    gates = proposal_report.get("legality_gates") or {}
    required = (PRIMARY_METRIC, BEST_OF_K_METRIC)
    if any(name not in metrics for name in required):
        raise ValueError("proposal report lacks checkpoint-selection metrics")
    values = {
        PRIMARY_METRIC: float(metrics[PRIMARY_METRIC]),
        BEST_OF_K_METRIC: float(metrics[BEST_OF_K_METRIC]),
        STRICT_ACCURACY_METRIC: float(strict_pair_accuracy),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("checkpoint-selection metrics must be finite")
    return {
        "schema_version": "delepref_checkpoint_selection_candidate_v1",
        "checkpoint_path": str(checkpoint_path),
        "step": int(step),
        "legality_gates_pass": proposal_report.get("status") == "PASS" and bool(gates) and all(gates.values()),
        "metrics": values,
        "selection_rule": {
            "legality_gates_first": True,
            "primary": PRIMARY_METRIC,
            "tie_tolerance": 1.0e-6,
            "tiebreakers": [BEST_OF_K_METRIC, STRICT_ACCURACY_METRIC],
            "forbidden_metrics": sorted(FORBIDDEN_SELECTION_METRICS),
        },
        "forbidden_metrics_used": [],
        "confirmation_metrics_used": False,
    }


def choose_checkpoint(candidates: Iterable[Mapping[str, object]], *, tie_tolerance: float = 1.0e-6):
    """Apply the frozen legality/utility/Best-of-K/accuracy rule."""

    tolerance = float(tie_tolerance)
    if tolerance < 0.0:
        raise ValueError("tie tolerance cannot be negative")
    best = None
    for candidate in candidates:
        if candidate.get("forbidden_metrics_used"):
            raise ValueError("forbidden checkpoint-selection metric was supplied")
        if not candidate.get("legality_gates_pass"):
            continue
        metrics = candidate.get("metrics") or {}
        if any(name in metrics for name in FORBIDDEN_SELECTION_METRICS):
            raise ValueError("forbidden checkpoint-selection metric was supplied")
        if best is None:
            best = candidate
            continue
        candidate_primary = float(metrics[PRIMARY_METRIC])
        best_primary = float(best["metrics"][PRIMARY_METRIC])
        if candidate_primary > best_primary + tolerance:
            best = candidate
            continue
        if abs(candidate_primary - best_primary) > tolerance:
            continue
        candidate_bok = float(metrics[BEST_OF_K_METRIC])
        best_bok = float(best["metrics"][BEST_OF_K_METRIC])
        if candidate_bok > best_bok:
            best = candidate
            continue
        if candidate_bok < best_bok:
            continue
        if float(metrics[STRICT_ACCURACY_METRIC]) > float(best["metrics"][STRICT_ACCURACY_METRIC]):
            best = candidate
    return best
