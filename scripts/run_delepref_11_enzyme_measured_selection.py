#!/usr/bin/env python3
"""Frozen CPU-only selection analysis for DelePref measured enzyme intervals.

This script reads already-scored measured candidates.  It never imports model
code, loads a checkpoint, scores a candidate, runs a teacher, or trains/refits
anything.  The pre-registered protocol must exist with its frozen SHA256 before
the numeric analysis can run.
"""

import argparse
import csv
import hashlib
import json
import math
import py_compile
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/experiments/delepref_step004000_enzyme"
OUT = ROOT / "results/experiments/delepref_step004000_delepref_11_enzyme_measured_selection"
PROTOCOL = OUT / "protocol.json"
PROTOCOL_SHA256 = "8860491114dbf01354c6cda49d38a9e02062a9d3dc0c624708d272172bc88268"
PROTOCOL_SOURCE = Path(__file__).resolve()
SCRIPT = Path(__file__).resolve()

PASS_STATUS = "PASS_DelePref_11_ENZYME_MEASURED_SELECTION"
MARKER_START = "DelePref_SUPP_ENZYME_MEASURED_SELECTION_START"
MARKER_END = "DelePref_SUPP_ENZYME_MEASURED_SELECTION_END"
CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion "
    "perturbation preference and optional auxiliary risk evidence."
)
VALIDATION_BOUNDARY = (
    "This is measured deletion/indel interval ranking over fixed measured "
    "candidates, not wet-lab functional validation of freely generated proposals."
)

RANKERS: Tuple[Tuple[str, str], ...] = (
    ("delepref", "delepref_score"),
    ("random", "random_score"),
    ("deletion_length", "deletion_length"),
    ("terminal_distance", "terminal_distance"),
    ("metadata_heuristic", "metadata_heuristic"),
    ("metadata_linear_LOTO", "metadata_linear_LOTO"),
)
KS = (1, 3, 5)
MAIN_METRICS: Tuple[Tuple[str, str], ...] = (
    ("top1_measured_percentile", "higher_is_better"),
    ("top1_ranking_regret", "lower_is_better"),
    ("top1_normalized_regret", "lower_is_better"),
    ("top1_measured_top_decile_hit_rate", "higher_is_better"),
    ("top3_measured_top_decile_hit_rate", "higher_is_better"),
    ("top5_measured_top_decile_hit_rate", "higher_is_better"),
    ("top1_tolerated_hit_rate", "higher_is_better"),
    ("top3_tolerated_hit_rate", "higher_is_better"),
    ("top5_tolerated_hit_rate", "higher_is_better"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".17g")
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    if not rows and fields is None:
        raise ValueError(f"cannot infer CSV fields for empty table: {path}")
    fieldnames = list(fields or rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def as_float(value: Any) -> float:
    result = float(str(value))
    if not math.isfinite(result):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return result


def as_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"invalid frozen boolean: {value!r}")
    return text == "true"


def mean_available(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None


def equal_float(left: Any, right: Any, tolerance: float = 2e-15) -> bool:
    if left in (None, "") or right in (None, ""):
        return left in (None, "") and right in (None, "")
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def canonical_intervals(value: str) -> Tuple[Tuple[int, int], ...]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("deleted_intervals must be a non-empty JSON list")
    output: List[Tuple[int, int]] = []
    previous_end = -1
    for interval in parsed:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError("each deleted interval must contain two coordinates")
        start, end = int(interval[0]), int(interval[1])
        if start < 0 or end < start or start <= previous_end:
            raise ValueError("deleted intervals must be sorted, closed, and non-overlapping")
        output.append((start, end))
        previous_end = end
    return tuple(output)


def recompute_measured_row_hash(row: Mapping[str, str]) -> str:
    intervals = canonical_intervals(row["deleted_intervals"])
    payload = {
        "row_id": str(row.get("row_id", "")),
        "protein_id": str(row.get("protein_id", "")),
        "uniprot_id": str(row.get("uniprot_id", "")),
        "assay_id": str(row.get("assay_id", "")),
        "variant_id": str(row.get("variant_id", "")),
        "sequence_sha256": str(row.get("sequence_sha256", "")),
        "deleted_intervals": [[start, end] for start, end in intervals],
        "measured_score_normalized": str(row.get("measured_score_normalized", "")),
        "tolerated_label": str(row.get("tolerated_label", "")),
        "deleterious_label": str(row.get("deleterious_label", "")),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def verify_protocol() -> Mapping[str, Any]:
    if not PROTOCOL.is_file():
        raise RuntimeError("frozen protocol.json is missing")
    actual = sha256_file(PROTOCOL)
    if actual != PROTOCOL_SHA256:
        raise RuntimeError(f"frozen protocol SHA256 changed: {actual}")
    protocol = read_json(PROTOCOL)
    if protocol.get("protocol_status") != "FROZEN_BEFORE_NUMERIC_ANALYSIS":
        raise RuntimeError("protocol is not frozen")
    return protocol


def preflight_schema_and_alignment(protocol: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
    frozen_hashes = dict(protocol["source"]["frozen_sha256"])
    actual_hashes = {name: sha256_file(SOURCE / name) for name in frozen_hashes}
    if actual_hashes != frozen_hashes:
        raise RuntimeError("authoritative input SHA256 differs from frozen protocol")

    source_summary = read_json(SOURCE / "run_summary.json")
    source_manifest = read_json(SOURCE / "audit_manifest.json")
    if source_summary.get("status") != "PASS_DelePref_ENZYME" or source_manifest.get("status") != "PASS_DelePref_ENZYME":
        raise RuntimeError("authoritative source is not PASS_DelePref_ENZYME")

    rows = read_csv(SOURCE / "measured_row_scores.csv")
    alignment = read_csv(SOURCE / "measured_row_alignment_audit.csv")
    baselines = read_csv(SOURCE / "baselines.csv")
    availability = read_csv(SOURCE / "baseline_availability.csv")
    if len(rows) != 1645 or len(alignment) != 1645 or len(baselines) != 1645:
        raise RuntimeError("expected exactly 1,645 source/alignment/baseline rows")
    if len({row["row_id"] for row in rows}) != 1645:
        raise RuntimeError("measured row_id is not unique")

    by_alignment = {row["row_id"]: row for row in alignment}
    by_baseline = {row["row_id"]: row for row in baselines}
    row_ids = {row["row_id"] for row in rows}
    if set(by_alignment) != row_ids or set(by_baseline) != row_ids:
        raise RuntimeError("source, original audit, and baseline row-id sets differ")

    required_columns = {
        "row_id", "measured_row_hash", "protein_id", "target_name", "uniprot_id", "assay_id",
        "variant_id", "sequence_sha256", "wildtype_sequence", "deleted_intervals",
        "measured_score", "measured_score_normalized", "tolerated_label", "deleterious_label",
        "binary_label_available", "delepref_score", "random_score", "deletion_length",
        "terminal_distance", "metadata_heuristic", "metadata_linear_LOTO",
    }
    if not rows or not required_columns.issubset(rows[0]):
        raise RuntimeError("measured_row_scores.csv lacks required selection columns")

    expected_availability = {name for name, _column in RANKERS}
    registry_names = {
        "delepref" if row["baseline"] == "delepref" else row["baseline"].replace("_score", "")
        for row in availability if as_bool(row["available"])
    }
    if registry_names != expected_availability:
        raise RuntimeError(f"baseline availability registry mismatch: {sorted(registry_names)}")

    baseline_column = {
        "delepref_score": "delepref",
        "random_score": "random_score",
        "deletion_length": "deletion_length",
        "terminal_distance": "terminal_distance",
        "metadata_heuristic": "metadata_heuristic",
        "metadata_linear_LOTO": "metadata_linear_LOTO",
    }
    audit_rows: List[Dict[str, Any]] = []
    for row in rows:
        row_id = row["row_id"]
        original = by_alignment[row_id]
        baseline = by_baseline[row_id]
        recomputed_sequence_sha = sha256_text(row["wildtype_sequence"])
        recomputed_row_hash = recompute_measured_row_hash(row)
        flags: Dict[str, bool] = {
            "original_alignment_status_pass": original.get("alignment_status") == "PASS" and not original.get("alignment_issues"),
            "measured_row_hash_match_original_audit": row["measured_row_hash"] == original["measured_row_hash"],
            "measured_row_hash_recomputed_match": row["measured_row_hash"] == recomputed_row_hash,
            "sequence_sha256_match_original_audit": row["sequence_sha256"] == original["sequence_sha256"],
            "sequence_sha256_recomputed_match": row["sequence_sha256"] == recomputed_sequence_sha,
            "baseline_measured_row_hash_match": row["measured_row_hash"] == baseline["measured_row_hash"],
            "baseline_sequence_sha256_match": row["sequence_sha256"] == baseline["sequence_sha256"],
            "measured_score_match_original_audit": row["measured_score"] == original["measured_score"],
            "measured_score_normalized_match_original_audit": row["measured_score_normalized"] == original["measured_score_normalized"],
            "tolerated_label_match_original_audit": row["tolerated_label"] == original["tolerated_label"],
            "deleterious_label_match_original_audit": row["deleterious_label"] == original["deleterious_label"],
            "binary_label_valid": as_bool(row["binary_label_available"]) and row["tolerated_label"] in {"0", "1"} and row["deleterious_label"] in {"0", "1"},
        }
        for _ranker, column in RANKERS:
            flags[column + "_finite"] = math.isfinite(as_float(row[column]))
            flags[column + "_baseline_match"] = equal_float(row[column], baseline[baseline_column[column]])
        status = "PASS" if all(flags.values()) else "FAIL"
        audit_rows.append({
            "row_id": row_id,
            "protein_id": row["protein_id"],
            "target_name": row["target_name"],
            "assay_id": row["assay_id"],
            "measured_row_hash": row["measured_row_hash"],
            "original_audit_measured_row_hash": original["measured_row_hash"],
            "recomputed_measured_row_hash": recomputed_row_hash,
            "sequence_sha256": row["sequence_sha256"],
            "original_audit_sequence_sha256": original["sequence_sha256"],
            "recomputed_sequence_sha256": recomputed_sequence_sha,
            **flags,
            "alignment_status": status,
        })
        if status != "PASS":
            raise RuntimeError(f"source alignment failed for {row_id}")

    targets = sorted({row["protein_id"] for row in rows})
    assays = sorted({row["assay_id"] for row in rows})
    groups = sorted({(row["protein_id"], row["assay_id"]) for row in rows})
    if (len(targets), len(assays), len(groups)) != (5, 9, 9):
        raise RuntimeError("expected 5 targets, 9 assays, and 9 target/assay groups")
    coverage = {
        column: sum(1 for row in rows if math.isfinite(as_float(row[column])))
        for _ranker, column in RANKERS
    }
    if any(count != 1645 for count in coverage.values()):
        raise RuntimeError("one or more rankers lacks full measured-row coverage")

    manifest_hash = next(
        item["sha256"] for item in source_manifest["artifacts"]
        if item["relative_path"] == "measured_row_scores.csv"
    )
    if manifest_hash != actual_hashes["measured_row_scores.csv"]:
        raise RuntimeError("source manifest measured_row_scores SHA256 mismatch")

    metadata = {
        "status": "PASS",
        "input_sha256": actual_hashes,
        "measured_intervals": len(rows),
        "independent_targets": len(targets),
        "assays": len(assays),
        "target_assay_groups": len(groups),
        "targets": targets,
        "group_sizes": [
            {
                "protein_id": protein_id,
                "assay_id": assay_id,
                "n_candidates": sum(
                    1 for row in rows
                    if row["protein_id"] == protein_id and row["assay_id"] == assay_id
                ),
            }
            for protein_id, assay_id in groups
        ],
        "binary_label_available_rows": sum(as_bool(row["binary_label_available"]) for row in rows),
        "ranker_finite_coverage": coverage,
    }
    return rows, audit_rows, metadata


def measured_percentile(values: Sequence[float], selected: float) -> float:
    if len(values) == 1:
        return 1.0
    lower = sum(value < selected for value in values)
    equal = sum(value == selected for value in values)
    return (lower + (equal - 1) / 2.0) / (len(values) - 1)


def calculate_group_tables(rows: Sequence[Mapping[str, str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["protein_id"], row["assay_id"])].append(row)

    selection_rows: List[Dict[str, Any]] = []
    topk_rows: List[Dict[str, Any]] = []
    regret_rows: List[Dict[str, Any]] = []
    for (protein_id, assay_id), group in sorted(grouped.items()):
        measured = [as_float(row["measured_score_normalized"]) for row in group]
        n = len(group)
        best = max(measured)
        minimum = min(measured)
        measured_range = best - minimum
        nominal_decile_k = max(1, int(math.ceil(0.10 * n)))
        decile_threshold = sorted(measured, reverse=True)[nominal_decile_k - 1]
        top_decile_hashes = {
            row["measured_row_hash"] for row in group
            if as_float(row["measured_score_normalized"]) >= decile_threshold
        }
        labels_available = all(
            as_bool(row["binary_label_available"]) and row["tolerated_label"] in {"0", "1"}
            for row in group
        )
        target_name = group[0]["target_name"]

        for ranker, column in RANKERS:
            ordered = sorted(group, key=lambda row: (-as_float(row[column]), row["measured_row_hash"]))
            selected = ordered[0]
            selected_measured = as_float(selected["measured_score_normalized"])
            raw_regret = best - selected_measured
            normalized_regret = raw_regret / measured_range if measured_range > 0.0 else None
            percentile = measured_percentile(measured, selected_measured)
            common = {
                "ranker": ranker,
                "score_column": column,
                "protein_id": protein_id,
                "target_name": target_name,
                "assay_id": assay_id,
                "n_candidates": n,
                "selected_top1_row_id": selected["row_id"],
                "selected_top1_measured_row_hash": selected["measured_row_hash"],
                "selected_top1_variant_id": selected["variant_id"],
                "selected_top1_ranker_score": as_float(selected[column]),
                "selected_top1_measured_score_normalized": selected_measured,
                "top1_measured_percentile": percentile,
                "group_best_measured_score_normalized": best,
                "group_min_measured_score_normalized": minimum,
                "group_measured_range": measured_range,
                "top1_ranking_regret": raw_regret,
                "top1_normalized_regret": normalized_regret,
                "constant_measured_range": measured_range == 0.0,
                "measured_top_decile_nominal_k": nominal_decile_k,
                "measured_top_decile_realized_n": len(top_decile_hashes),
                "binary_label_available": labels_available,
                "selected_top1_tolerated_label": int(selected["tolerated_label"]) if labels_available else None,
                "selection_tie_rule": "ranker_score_desc_then_measured_row_hash_asc",
                "percentile_tie_rule": "midrank",
            }
            selection_rows.append(common)
            regret_rows.append({
                key: common[key] for key in (
                    "ranker", "score_column", "protein_id", "target_name", "assay_id", "n_candidates",
                    "selected_top1_row_id", "selected_top1_measured_row_hash",
                    "selected_top1_measured_score_normalized", "group_best_measured_score_normalized",
                    "group_min_measured_score_normalized", "group_measured_range", "top1_ranking_regret",
                    "top1_normalized_regret", "constant_measured_range",
                )
            })

            for requested_k in KS:
                effective_k = min(requested_k, n)
                selected_k = ordered[:effective_k]
                decile_hits = sum(row["measured_row_hash"] in top_decile_hashes for row in selected_k)
                tolerated_hits = (
                    sum(int(row["tolerated_label"]) == 1 for row in selected_k)
                    if labels_available else None
                )
                topk_rows.append({
                    "ranker": ranker,
                    "score_column": column,
                    "protein_id": protein_id,
                    "target_name": target_name,
                    "assay_id": assay_id,
                    "n_candidates": n,
                    "requested_k": requested_k,
                    "effective_k": effective_k,
                    "selected_row_ids": ";".join(row["row_id"] for row in selected_k),
                    "selected_measured_row_hashes": ";".join(row["measured_row_hash"] for row in selected_k),
                    "measured_top_decile_nominal_k": nominal_decile_k,
                    "measured_top_decile_realized_n": len(top_decile_hashes),
                    "measured_top_decile_threshold": decile_threshold,
                    "measured_top_decile_hits": decile_hits,
                    "measured_top_decile_hit_rate": decile_hits / effective_k,
                    "measured_top_decile_any_hit": decile_hits > 0,
                    "binary_label_available": labels_available,
                    "tolerated_hits": tolerated_hits,
                    "tolerated_hit_rate": tolerated_hits / effective_k if tolerated_hits is not None else None,
                    "tolerated_any_hit": tolerated_hits > 0 if tolerated_hits is not None else None,
                })
    return selection_rows, topk_rows, regret_rows


def calculate_target_summaries(selection_rows: Sequence[Mapping[str, Any]], topk_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped_selection: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in selection_rows:
        grouped_selection[(str(row["ranker"]), str(row["protein_id"]))].append(row)
    topk_index = {
        (str(row["ranker"]), str(row["protein_id"]), str(row["assay_id"]), int(row["requested_k"])): row
        for row in topk_rows
    }

    output: List[Dict[str, Any]] = []
    for (ranker, protein_id), assay_rows in sorted(grouped_selection.items()):
        metric_values: Dict[str, List[Optional[float]]] = {
            "top1_measured_percentile": [float(row["top1_measured_percentile"]) for row in assay_rows],
            "top1_ranking_regret": [float(row["top1_ranking_regret"]) for row in assay_rows],
            "top1_normalized_regret": [
                float(row["top1_normalized_regret"]) if row["top1_normalized_regret"] is not None else None
                for row in assay_rows
            ],
        }
        for requested_k in KS:
            k_rows = [topk_index[(ranker, protein_id, str(row["assay_id"]), requested_k)] for row in assay_rows]
            metric_values[f"top{requested_k}_measured_top_decile_hit_rate"] = [
                float(row["measured_top_decile_hit_rate"]) for row in k_rows
            ]
            metric_values[f"top{requested_k}_measured_top_decile_any_hit_rate"] = [
                float(bool(row["measured_top_decile_any_hit"])) for row in k_rows
            ]
            metric_values[f"top{requested_k}_tolerated_hit_rate"] = [
                float(row["tolerated_hit_rate"]) if row["tolerated_hit_rate"] is not None else None
                for row in k_rows
            ]
            metric_values[f"top{requested_k}_tolerated_any_hit_rate"] = [
                float(bool(row["tolerated_any_hit"])) if row["tolerated_any_hit"] is not None else None
                for row in k_rows
            ]

        record: Dict[str, Any] = {
            "ranker": ranker,
            "protein_id": protein_id,
            "target_name": assay_rows[0]["target_name"],
            "n_assays": len(assay_rows),
            "n_candidates": sum(int(row["n_candidates"]) for row in assay_rows),
            "aggregation": "unweighted_mean_across_assays_within_target",
        }
        for metric, values in metric_values.items():
            record[metric] = mean_available(values)
            record[metric + "_n_assays"] = sum(value is not None for value in values)
        output.append(record)
    return output


def calculate_baseline_comparison(target_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_ranker: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        by_ranker[str(row["ranker"])].append(row)
    student = {metric: mean_available(row.get(metric) for row in by_ranker["delepref"]) for metric, _direction in MAIN_METRICS}
    output: List[Dict[str, Any]] = []
    for ranker, _column in RANKERS:
        rows = by_ranker[ranker]
        for metric, direction in MAIN_METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            value = mean_available(values)
            reference = student[metric]
            output.append({
                "ranker": ranker,
                "metric": metric,
                "direction": direction,
                "target_macro_value": value,
                "n_targets_available": len(values),
                "n_independent_targets_total": 5,
                "delepref_target_macro_value": reference,
                "delta_ranker_minus_delepref": value - reference if value is not None and reference is not None else None,
                "aggregation": "unweighted_mean_across_target_summaries",
                "uncertainty": "exact_leave_one_target_out_reported_separately",
            })
    return output


def calculate_loto_sensitivity(target_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    targets = sorted({str(row["protein_id"]) for row in target_rows})
    by_ranker: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        by_ranker[str(row["ranker"])].append(row)
    output: List[Dict[str, Any]] = []
    for ranker, _column in RANKERS:
        rows = by_ranker[ranker]
        for metric, direction in MAIN_METRICS:
            full_values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            full = mean_available(full_values)
            for excluded in targets:
                remaining = [
                    float(row[metric]) for row in rows
                    if str(row["protein_id"]) != excluded and row.get(metric) is not None
                ]
                without = mean_available(remaining)
                output.append({
                    "ranker": ranker,
                    "metric": metric,
                    "direction": direction,
                    "excluded_target": excluded,
                    "remaining_targets_total": 4,
                    "remaining_targets_available": len(remaining),
                    "full_target_macro_value": full,
                    "without_excluded_target_macro_value": without,
                    "difference_without_minus_full": without - full if without is not None and full is not None else None,
                    "sensitivity_type": "exact_leave_one_target_out_aggregation",
                    "delepref_retraining": False,
                    "baseline_refitting": False,
                })
    return output


def markdown_number(value: Any) -> str:
    if value in (None, ""):
        return "NA"
    return f"{float(value):.4f}"


def make_payload_and_summary(
    metadata: Mapping[str, Any], target_rows: Sequence[Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]], status: str,
) -> Tuple[str, Mapping[str, Any], str]:
    student_targets = sorted(
        (row for row in target_rows if row["ranker"] == "delepref"),
        key=lambda row: str(row["protein_id"]),
    )
    comparison_index = {
        (str(row["ranker"]), str(row["metric"])): row for row in comparison
    }
    lines = [
        f"<!-- {MARKER_START} -->",
        "### DelePref-11: DelePref measured-candidate selection on enzyme assays",
        "",
        (
            "Frozen-result analysis ranks 1,645 fixed measured deletion/indel intervals across "
            "9 assays and 5 independent protein targets. Targets, rather than rows, are the "
            "independent units; exact leave-one-target-out sensitivity is reported."
        ),
        "",
        "| target | assays | top-1 measured percentile | top-1 regret | normalized regret | top-1/top-3/top-5 measured-top-decile hit rate | top-1/top-3/top-5 tolerated hit rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in student_targets:
        decile = "/".join(markdown_number(row[f"top{k}_measured_top_decile_hit_rate"]) for k in KS)
        tolerated = "/".join(markdown_number(row[f"top{k}_tolerated_hit_rate"]) for k in KS)
        lines.append(
            f"| {row['protein_id']} | {row['n_assays']} | {markdown_number(row['top1_measured_percentile'])} | "
            f"{markdown_number(row['top1_ranking_regret'])} | {markdown_number(row['top1_normalized_regret'])} | "
            f"{decile} | {tolerated} |"
        )
    lines.extend([
        "",
        (
            "The random, deletion-length, terminal-distance, metadata heuristic, and existing "
            "metadata_linear_LOTO baselines use the identical selection metrics and aggregation."
        ),
        "",
        (
            "Boundary: this evaluates ranking within fixed measured intervals. It does not validate "
            "wet-lab activity of freely generated DelePref proposals and does not prove that deleted regions have no function."
        ),
        f"<!-- {MARKER_END} -->",
        "",
    ])
    payload_md = "\n".join(lines)

    target_macro_student = {
        metric: comparison_index[("delepref", metric)]["target_macro_value"]
        for metric, _direction in MAIN_METRICS
    }
    payload_json: Mapping[str, Any] = {
        "schema_version": "delepref_11_enzyme_measured_selection_payload_v1",
        "status": status,
        "marker_start": MARKER_START,
        "marker_end": MARKER_END,
        "protocol_source_modified": False,
        "analysis_scope": "fixed_measured_deletion_indel_interval_ranking",
        "counts": {
            "measured_intervals": metadata["measured_intervals"],
            "assays": metadata["assays"],
            "independent_targets": metadata["independent_targets"],
        },
        "delepref_target_wise": [dict(row) for row in student_targets],
        "delepref_target_macro": target_macro_student,
        "baseline_target_macro": [dict(row) for row in comparison],
        "uncertainty": "exact leave-one-target-out target sensitivity; no row-level bootstrap",
        "validation_boundary": VALIDATION_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
    }

    summary_lines = [
        "# DelePref-11 Enzyme Measured-Candidate Selection",
        "",
        f"Status: `{status}`.",
        "",
        "## Scope and independent unit",
        "",
        (
            "This CPU-only frozen-result analysis evaluates selection within 1,645 already scored, fixed "
            "measured deletion/indel intervals. They span 9 assays but only 5 independent protein targets. "
            "All target-macro results therefore weight targets equally rather than treating rows as independent."
        ),
        "",
        "## DelePref target-wise results",
        "",
        "| target | assays | candidates | top-1 percentile | regret | normalized regret | top-1/top-3/top-5 top-decile hit rate | top-1/top-3/top-5 tolerated hit rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in student_targets:
        decile = "/".join(markdown_number(row[f"top{k}_measured_top_decile_hit_rate"]) for k in KS)
        tolerated = "/".join(markdown_number(row[f"top{k}_tolerated_hit_rate"]) for k in KS)
        summary_lines.append(
            f"| {row['protein_id']} | {row['n_assays']} | {row['n_candidates']} | "
            f"{markdown_number(row['top1_measured_percentile'])} | {markdown_number(row['top1_ranking_regret'])} | "
            f"{markdown_number(row['top1_normalized_regret'])} | {decile} | {tolerated} |"
        )
    summary_lines.extend([
        "",
        "## Frozen definitions",
        "",
        (
            "Top-1 measured percentile uses empirical midranks with 1.0 best (a singleton is 1.0). "
            "A measured top-decile set starts at `max(1, ceil(0.10*n))` and includes threshold ties. "
            "Top-k uses `min(k,n)`. Tolerated hit rates use the original frozen binary label and never "
            "substitute measured top-decile membership. Normalized regret is NA for constant measured ranges."
        ),
        "",
        (
            "Every baseline uses the same definitions. Leave-one-target-out rows are exact aggregation "
            "sensitivity checks only; neither DelePref nor metadata_linear_LOTO is retrained/refit."
        ),
        "",
        "## Claim boundary",
        "",
        VALIDATION_BOUNDARY,
        "It does not prove that a deleted region has no function. Teacher preference is not a ground-truth functional label.",
        "",
    ])
    return payload_md, payload_json, "\n".join(summary_lines)


def independent_validation(
    protocol: Mapping[str, Any], before_input_hashes: Mapping[str, str], protocol_source_hash_before: str
) -> Mapping[str, Any]:
    """Re-read inputs and independently recompute selections and aggregations."""
    checks: Dict[str, bool] = {}
    details: Dict[str, Any] = {}

    after_input_hashes = {
        name: sha256_file(SOURCE / name) for name in protocol["source"]["frozen_sha256"]
    }
    checks["input_hashes_recomputed_match_protocol"] = after_input_hashes == protocol["source"]["frozen_sha256"]
    checks["input_hashes_unchanged_during_analysis"] = after_input_hashes == dict(before_input_hashes)
    checks["protocol_source_unchanged"] = sha256_file(PROTOCOL_SOURCE) == protocol_source_hash_before

    rows = read_csv(SOURCE / "measured_row_scores.csv")
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["protein_id"], row["assay_id"])].append(row)
    checks["row_count_1645"] = len(rows) == 1645
    checks["group_counts_exact"] = (
        len({row["protein_id"] for row in rows}) == 5
        and len({row["assay_id"] for row in rows}) == 9
        and len(groups) == 9
    )
    checks["measured_row_hashes_recomputed"] = all(
        recompute_measured_row_hash(row) == row["measured_row_hash"] for row in rows
    )
    checks["sequence_sha256_recomputed"] = all(
        sha256_text(row["wildtype_sequence"]) == row["sequence_sha256"] for row in rows
    )

    alignment_by_id = {
        row["row_id"]: row for row in read_csv(SOURCE / "measured_row_alignment_audit.csv")
    }
    checks["original_audit_hash_alignment"] = all(
        row["measured_row_hash"] == alignment_by_id[row["row_id"]]["measured_row_hash"]
        and row["sequence_sha256"] == alignment_by_id[row["row_id"]]["sequence_sha256"]
        for row in rows
    )
    coverage = {
        column: sum(math.isfinite(as_float(row[column])) for row in rows)
        for _ranker, column in RANKERS
    }
    checks["baseline_coverage_complete"] = all(value == 1645 for value in coverage.values())
    details["baseline_finite_rows"] = coverage

    selection_table = read_csv(OUT / "target_assay_selection_metrics.csv")
    selection_index = {
        (row["ranker"], row["protein_id"], row["assay_id"]): row for row in selection_table
    }
    regret_table = read_csv(OUT / "ranking_regret.csv")
    regret_index = {
        (row["ranker"], row["protein_id"], row["assay_id"]): row for row in regret_table
    }
    topk_table = read_csv(OUT / "topk_measured_hit_rates.csv")
    topk_index = {
        (row["ranker"], row["protein_id"], row["assay_id"], int(row["requested_k"])): row
        for row in topk_table
    }
    independently_computed: Dict[Tuple[str, str, str], Dict[str, Optional[float]]] = {}
    selection_ok = True
    topk_ok = True
    percentile_ok = True
    regret_ok = True

    for (protein_id, assay_id), group in sorted(groups.items()):
        outcomes = [as_float(row["measured_score_normalized"]) for row in group]
        n = len(group)
        maximum, minimum = max(outcomes), min(outcomes)
        spread = maximum - minimum
        decile_k = max(1, int(math.ceil(n / 10.0)))
        threshold = sorted(outcomes, reverse=True)[decile_k - 1]
        top_decile = {
            row["measured_row_hash"] for row in group
            if as_float(row["measured_score_normalized"]) >= threshold
        }
        binary_ok = all(
            row["binary_label_available"].lower() == "true" and row["tolerated_label"] in {"0", "1"}
            for row in group
        )
        for ranker, column in RANKERS:
            ordered = list(group)
            ordered.sort(key=lambda item: item["measured_row_hash"])
            ordered.sort(key=lambda item: as_float(item[column]), reverse=True)
            chosen = ordered[0]
            chosen_y = as_float(chosen["measured_score_normalized"])
            if n == 1:
                pct = 1.0
            else:
                occupied_low = sum(value < chosen_y for value in outcomes)
                occupied_equal = sum(value == chosen_y for value in outcomes)
                pct = (occupied_low + (occupied_equal - 1.0) / 2.0) / (n - 1.0)
            raw = maximum - chosen_y
            normalized = raw / spread if spread > 0.0 else None
            key = (ranker, protein_id, assay_id)
            produced = selection_index.get(key)
            produced_regret = regret_index.get(key)
            selection_ok = selection_ok and produced is not None and produced["selected_top1_measured_row_hash"] == chosen["measured_row_hash"]
            percentile_ok = percentile_ok and produced is not None and equal_float(produced["top1_measured_percentile"], pct)
            regret_ok = (
                regret_ok and produced_regret is not None
                and equal_float(produced_regret["top1_ranking_regret"], raw)
                and equal_float(produced_regret["top1_normalized_regret"], normalized)
            )
            independently_computed[key] = {
                "top1_measured_percentile": pct,
                "top1_ranking_regret": raw,
                "top1_normalized_regret": normalized,
            }
            for requested_k in KS:
                effective = min(requested_k, n)
                selected = ordered[:effective]
                decile_hits = sum(item["measured_row_hash"] in top_decile for item in selected)
                tolerated_hits = sum(int(item["tolerated_label"]) == 1 for item in selected) if binary_ok else None
                produced_k = topk_index.get((ranker, protein_id, assay_id, requested_k))
                topk_ok = (
                    topk_ok and produced_k is not None
                    and int(produced_k["effective_k"]) == effective
                    and produced_k["selected_measured_row_hashes"] == ";".join(item["measured_row_hash"] for item in selected)
                    and int(produced_k["measured_top_decile_hits"]) == decile_hits
                    and equal_float(produced_k["measured_top_decile_hit_rate"], decile_hits / effective)
                    and (
                        tolerated_hits is None
                        or (
                            int(produced_k["tolerated_hits"]) == tolerated_hits
                            and equal_float(produced_k["tolerated_hit_rate"], tolerated_hits / effective)
                        )
                    )
                )
                independently_computed[key][f"top{requested_k}_measured_top_decile_hit_rate"] = decile_hits / effective
                independently_computed[key][f"top{requested_k}_tolerated_hit_rate"] = (
                    tolerated_hits / effective if tolerated_hits is not None else None
                )

    checks["top1_selection_independently_recomputed"] = selection_ok
    checks["topk_selection_independently_recomputed"] = topk_ok
    checks["midrank_percentile_independently_recomputed"] = percentile_ok
    checks["regret_independently_recomputed"] = regret_ok

    target_table = read_csv(OUT / "target_selection_summary.csv")
    target_index = {(row["ranker"], row["protein_id"]): row for row in target_table}
    aggregation_ok = True
    target_values: Dict[Tuple[str, str, str], Optional[float]] = {}
    for ranker, _column in RANKERS:
        for protein_id in sorted({row["protein_id"] for row in rows}):
            assays = sorted({assay for protein, assay in groups if protein == protein_id})
            target_row = target_index.get((ranker, protein_id))
            aggregation_ok = aggregation_ok and target_row is not None
            for metric, _direction in MAIN_METRICS:
                vals = [independently_computed[(ranker, protein_id, assay)][metric] for assay in assays]
                expected = mean_available(vals)
                available_n = sum(value is not None for value in vals)
                if target_row is None:
                    aggregation_ok = False
                    continue
                aggregation_ok = (
                    aggregation_ok
                    and equal_float(target_row[metric], expected)
                    and int(target_row[metric + "_n_assays"]) == available_n
                )
                target_values[(ranker, protein_id, metric)] = expected
    checks["target_level_aggregation_independently_recomputed"] = aggregation_ok

    comparison_table = read_csv(OUT / "baseline_selection_comparison.csv")
    comparison_index = {(row["ranker"], row["metric"]): row for row in comparison_table}
    target_macro_ok = True
    for ranker, _column in RANKERS:
        for metric, _direction in MAIN_METRICS:
            vals = [
                target_values[(ranker, protein_id, metric)]
                for protein_id in sorted({row["protein_id"] for row in rows})
            ]
            expected = mean_available(vals)
            produced = comparison_index.get((ranker, metric))
            target_macro_ok = (
                target_macro_ok and produced is not None
                and equal_float(produced["target_macro_value"], expected)
                and int(produced["n_targets_available"]) == sum(value is not None for value in vals)
            )
    checks["target_macro_aggregation_independently_recomputed"] = target_macro_ok

    source_audit = read_csv(OUT / "source_alignment_audit.csv")
    checks["source_alignment_audit_all_pass"] = len(source_audit) == 1645 and all(row["alignment_status"] == "PASS" for row in source_audit)
    checks["selection_output_coverage_exact"] = (
        len(selection_table) == 54 and len(regret_table) == 54 and len(topk_table) == 162
        and len(target_table) == 30 and len(comparison_table) == 54
    )

    json_files = [PROTOCOL, OUT / "result_summary.json"]
    csv_files = [
        OUT / "source_alignment_audit.csv", OUT / "target_assay_selection_metrics.csv",
        OUT / "target_selection_summary.csv", OUT / "topk_measured_hit_rates.csv",
        OUT / "ranking_regret.csv", OUT / "baseline_selection_comparison.csv",
        OUT / "leave_one_target_out_sensitivity.csv",
    ]
    json_parse_ok = True
    csv_parse_ok = True
    for path in json_files:
        try:
            read_json(path)
        except Exception:
            json_parse_ok = False
    for path in csv_files:
        try:
            parsed = read_csv(path)
            csv_parse_ok = csv_parse_ok and bool(parsed)
        except Exception:
            csv_parse_ok = False
    checks["json_parse_pass"] = json_parse_ok
    checks["csv_parse_pass"] = csv_parse_ok

    core_artifacts = [
        PROTOCOL,
        OUT / "source_alignment_audit.csv",
        OUT / "target_assay_selection_metrics.csv",
        OUT / "target_selection_summary.csv",
        OUT / "topk_measured_hit_rates.csv",
        OUT / "ranking_regret.csv",
        OUT / "baseline_selection_comparison.csv",
        OUT / "leave_one_target_out_sensitivity.csv",
        OUT / "result_summary.md",
        OUT / "result_summary.json",
        OUT / "SUMMARY.md",
        SCRIPT,
    ]
    artifact_hashes = {str(path.relative_to(ROOT)): sha256_file(path) for path in core_artifacts}
    checks["artifact_sha256_recomputed_and_nonempty"] = all(
        path.stat().st_size > 0 and sha256_file(path) == artifact_hashes[str(path.relative_to(ROOT))]
        for path in core_artifacts
    )
    details["artifact_sha256"] = artifact_hashes

    with tempfile.TemporaryDirectory(prefix="delepref_11_pycompile_") as temporary:
        try:
            py_compile.compile(str(SCRIPT), cfile=str(Path(temporary) / "validated.pyc"), doraise=True)
            checks["py_compile_pass"] = True
            details["py_compile"] = "py_compile.compile with temporary cfile"
        except py_compile.PyCompileError as error:
            checks["py_compile_pass"] = False
            details["py_compile_error"] = str(error)

    diff = subprocess.run(
        ["git", "diff", "--check"], cwd=str(ROOT), universal_newlines=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    checks["git_diff_check_pass"] = diff.returncode == 0
    details["git_diff_check_returncode"] = diff.returncode
    details["git_diff_check_output"] = (diff.stdout + diff.stderr).strip()

    all_pass = all(checks.values())
    return {
        "schema_version": "delepref_11_enzyme_measured_selection_independent_validation_v1",
        "status": "PASS" if all_pass else "FAIL",
        "validated_at": utc_now(),
        "independent_recomputation": True,
        "checks": checks,
        "details": details,
        "counts": {
            "measured_intervals": len(rows),
            "assays": len({row["assay_id"] for row in rows}),
            "independent_targets": len({row["protein_id"] for row in rows}),
            "target_assay_groups": len(groups),
            "rankers": len(RANKERS),
        },
        "statistical_boundary": "Targets are independent units; no row-level bootstrap was used.",
        "validation_boundary": VALIDATION_BOUNDARY,
    }


def artifact_record(path: Path) -> Mapping[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "nonempty": path.stat().st_size > 0,
    }


def final_read_only_verification(protocol: Mapping[str, Any], protocol_source_hash_before: str) -> Mapping[str, Any]:
    manifest = read_json(OUT / "audit_manifest.json")
    run_summary = read_json(OUT / "run_summary.json")
    validation = read_json(OUT / "independent_validation.json")
    hash_ok = all(
        (ROOT / item["path"]).is_file()
        and sha256_file(ROOT / item["path"]) == item["sha256"]
        and (ROOT / item["path"]).stat().st_size == int(item["size_bytes"])
        for item in manifest["artifacts"]
    )
    csv_ok = all(
        bool(read_csv(OUT / name)) for name in (
            "source_alignment_audit.csv", "target_assay_selection_metrics.csv",
            "target_selection_summary.csv", "topk_measured_hit_rates.csv", "ranking_regret.csv",
            "baseline_selection_comparison.csv", "leave_one_target_out_sensitivity.csv",
        )
    )
    input_ok = all(
        sha256_file(SOURCE / name) == expected
        for name, expected in protocol["source"]["frozen_sha256"].items()
    )
    diff = subprocess.run(
        ["git", "diff", "--check"], cwd=str(ROOT), universal_newlines=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    checks = {
        "final_json_parse": bool(manifest and run_summary and validation),
        "final_csv_parse": csv_ok,
        "final_manifest_artifact_sha256": hash_ok,
        "final_statuses": (
            manifest.get("status") == PASS_STATUS
            and run_summary.get("status") == PASS_STATUS
            and validation.get("status") == "PASS"
        ),
        "final_input_hashes": input_ok,
        "final_protocol_source_unchanged": sha256_file(PROTOCOL_SOURCE) == protocol_source_hash_before,
        "final_git_diff_check": diff.returncode == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def run_analysis() -> None:
    protocol = verify_protocol()
    pending = {
        "schema_version": "delepref_11_enzyme_measured_selection_pending_v1",
        "status": "VALIDATION_PENDING",
        "updated_at": utc_now(),
        "note": "A prior PASS, if any, is invalidated before recomputation begins.",
    }
    write_json(OUT / "run_summary.json", pending)
    write_json(OUT / "audit_manifest.json", pending)
    write_json(OUT / "independent_validation.json", pending)
    protocol_source_hash_before = sha256_file(PROTOCOL_SOURCE)
    rows, alignment_audit, metadata = preflight_schema_and_alignment(protocol)
    before_input_hashes = dict(metadata["input_sha256"])

    selection_rows, topk_rows, regret_rows = calculate_group_tables(rows)
    target_rows = calculate_target_summaries(selection_rows, topk_rows)
    comparison_rows = calculate_baseline_comparison(target_rows)
    sensitivity_rows = calculate_loto_sensitivity(target_rows)

    write_csv(OUT / "source_alignment_audit.csv", alignment_audit)
    write_csv(OUT / "target_assay_selection_metrics.csv", selection_rows)
    write_csv(OUT / "target_selection_summary.csv", target_rows)
    write_csv(OUT / "topk_measured_hit_rates.csv", topk_rows)
    write_csv(OUT / "ranking_regret.csv", regret_rows)
    write_csv(OUT / "baseline_selection_comparison.csv", comparison_rows)
    write_csv(OUT / "leave_one_target_out_sensitivity.csv", sensitivity_rows)

    payload_md, payload_json, summary_md = make_payload_and_summary(
        metadata, target_rows, comparison_rows, "VALIDATION_PENDING"
    )
    (OUT / "result_summary.md").write_text(payload_md, encoding="utf-8")
    write_json(OUT / "result_summary.json", payload_json)
    (OUT / "SUMMARY.md").write_text(summary_md, encoding="utf-8")

    pre_validation = independent_validation(protocol, before_input_hashes, protocol_source_hash_before)
    if pre_validation["status"] != "PASS":
        raise RuntimeError("independent validation failed; PASS run summary was not written")

    # Only after a complete recomputation pass may any final artifact carry the
    # PASS status.  Recompute once more over those final payloads so their parse
    # and SHA256 checks describe the exact frozen deliverables.
    payload_md, payload_json, summary_md = make_payload_and_summary(
        metadata, target_rows, comparison_rows, PASS_STATUS
    )
    (OUT / "result_summary.md").write_text(payload_md, encoding="utf-8")
    write_json(OUT / "result_summary.json", payload_json)
    (OUT / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
    validation = independent_validation(protocol, before_input_hashes, protocol_source_hash_before)
    write_json(OUT / "independent_validation.json", validation)
    if validation["status"] != "PASS":
        raise RuntimeError("final-payload independent validation failed; PASS run summary was not written")

    student_macro = {
        row["metric"]: row["target_macro_value"]
        for row in comparison_rows if row["ranker"] == "delepref"
    }
    run_summary = {
        "schema_version": "delepref_11_enzyme_measured_selection_run_summary_v1",
        "status": PASS_STATUS,
        "created_at": utc_now(),
        "analysis_id": "DelePref-11",
        "analysis_scope": "fixed_measured_deletion_indel_interval_ranking",
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_frozen_before_numeric_analysis": True,
        "source_status": "PASS_DelePref_ENZYME",
        "source_directory": str(SOURCE.relative_to(ROOT)),
        "input_sha256": before_input_hashes,
        "input_immutability_pass": all(
            sha256_file(SOURCE / name) == expected for name, expected in before_input_hashes.items()
        ),
        "protocol_source_modified": sha256_file(PROTOCOL_SOURCE) != protocol_source_hash_before,
        "protocol_source_sha256_before": protocol_source_hash_before,
        "protocol_source_sha256_after": sha256_file(PROTOCOL_SOURCE),
        "counts": {
            "measured_intervals": metadata["measured_intervals"],
            "independent_targets": metadata["independent_targets"],
            "assays": metadata["assays"],
            "target_assay_groups": metadata["target_assay_groups"],
            "rankers": len(RANKERS),
            "target_assay_metric_rows": len(selection_rows),
            "topk_rows": len(topk_rows),
            "target_summary_rows": len(target_rows),
            "loto_sensitivity_rows": len(sensitivity_rows),
        },
        "execution": {
            "device": "cpu",
            "checkpoint_loaded": False,
            "scorer_run": False,
            "teacher_run": False,
            "training_run": False,
            "proposal_generation_run": False,
            "metadata_linear_LOTO_refit": False,
        },
        "selection_definitions": {
            "top1_percentile": "empirical midrank, 1.0 best, singleton 1.0",
            "top_decile": "max(1,ceil(0.10*n))-th measured threshold with boundary ties included",
            "topk": "effective_k=min(k,n), k in {1,3,5}",
            "tolerated": "original frozen tolerated_label only",
            "regret": "group best measured score minus selected top-1 measured score",
            "normalized_regret": "regret / measured range; NA for constant range",
        },
        "aggregation": "assay macro within target, then equal-weight target macro",
        "uncertainty": "exact leave-one-target-out target sensitivity; no row-level bootstrap",
        "delepref_target_macro": student_macro,
        "independent_validation_status": validation["status"],
        "py_compile_pass": validation["checks"]["py_compile_pass"],
        "git_diff_check_pass": validation["checks"]["git_diff_check_pass"],
        "payload_marker_start": MARKER_START,
        "payload_marker_end": MARKER_END,
        "docs_protocol_source_modified_by_task": False,
        "free_generation": False,
        "wet_lab_validation_of_free_proposals": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "validation_boundary": VALIDATION_BOUNDARY,
    }
    if run_summary["protocol_source_modified"] or not run_summary["input_immutability_pass"]:
        raise RuntimeError("input or master document changed; PASS run summary was not written")
    write_json(OUT / "run_summary.json", run_summary)

    manifest_paths = [
        PROTOCOL,
        OUT / "source_alignment_audit.csv",
        OUT / "target_assay_selection_metrics.csv",
        OUT / "target_selection_summary.csv",
        OUT / "topk_measured_hit_rates.csv",
        OUT / "ranking_regret.csv",
        OUT / "baseline_selection_comparison.csv",
        OUT / "leave_one_target_out_sensitivity.csv",
        OUT / "independent_validation.json",
        OUT / "result_summary.md",
        OUT / "result_summary.json",
        OUT / "SUMMARY.md",
        OUT / "run_summary.json",
        SCRIPT,
    ]
    manifest = {
        "schema_version": "delepref_11_enzyme_measured_selection_audit_manifest_v1",
        "status": PASS_STATUS,
        "created_at": utc_now(),
        "audit_manifest_self_hash_excluded": True,
        "artifacts": [artifact_record(path) for path in manifest_paths],
        "input_artifacts": [
            {
                "path": str((SOURCE / name).relative_to(ROOT)),
                "sha256": expected,
                "size_bytes": (SOURCE / name).stat().st_size,
            }
            for name, expected in sorted(before_input_hashes.items())
        ],
        "counts": run_summary["counts"],
        "source_alignment_all_pass": all(row["alignment_status"] == "PASS" for row in alignment_audit),
        "input_immutability_pass": run_summary["input_immutability_pass"],
        "protocol_source_unchanged": not run_summary["protocol_source_modified"],
        "independent_validation_status": validation["status"],
        "claim_boundary": CLAIM_BOUNDARY,
        "validation_boundary": VALIDATION_BOUNDARY,
    }
    write_json(OUT / "audit_manifest.json", manifest)

    final = final_read_only_verification(protocol, protocol_source_hash_before)
    if final["status"] != "PASS":
        run_summary["status"] = "FAIL_FINAL_READ_ONLY_VERIFICATION"
        run_summary["final_read_only_verification"] = final
        write_json(OUT / "run_summary.json", run_summary)
        raise RuntimeError(f"final read-only verification failed: {final}")
    print(json.dumps({"status": PASS_STATUS, "counts": run_summary["counts"], "final_verification": final}, indent=2))


def schema_audit_only() -> None:
    protocol = verify_protocol()
    _rows, _audit, metadata = preflight_schema_and_alignment(protocol)
    print(json.dumps(metadata, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("schema-audit", "analyze"), default="analyze",
        help="schema-audit is read-only; analyze requires the already frozen protocol SHA",
    )
    args = parser.parse_args()
    if args.stage == "schema-audit":
        schema_audit_only()
    else:
        run_analysis()


if __name__ == "__main__":
    main()
