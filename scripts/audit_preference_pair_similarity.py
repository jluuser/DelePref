#!/usr/bin/env python3
"""Audit deletion-plan similarity in teacher preference pairs.

This is a CPU-only audit utility. It reads existing deletion perturbation and
teacher preference pair CSVs, annotates pair geometry/similarity, and writes
summary tables without modifying the source preference data.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


ANNOTATED_COLUMNS = [
    "protein_id",
    "pair_id",
    "budget_bin",
    "candidate_a",
    "candidate_b",
    "preferred_candidate",
    "budget_a",
    "budget_b",
    "budget_diff",
    "deleted_len_a",
    "deleted_len_b",
    "mask_jaccard",
    "overlap_coverage",
    "center_distance_norm",
    "boundary_shift_mean",
    "topology_relation",
    "teacher_margin",
    "preference_margin",
    "preference_confidence",
    "pair_relation_type",
    "recommended_weight",
    "preference",
    "is_valid_pair",
    "construction_policy",
    "source_pairs_csv",
]


RELATION_TYPES = [
    "local_boundary_contrast",
    "local_shift_contrast",
    "regional_alternative",
    "topology_contrast",
    "hard_negative_or_far",
    "too_far_uninformative",
    "unknown",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit deletion-plan similarity for teacher preference pairs.")
    parser.add_argument("--perturbations_csv", required=True, help="Deletion perturbation CSV, shard index, directory, or glob.")
    parser.add_argument("--pairs_csv", required=True, help="Teacher preference pair CSV, directory, or glob.")
    parser.add_argument("--teacher_scores_csv", default=None, help="Optional teacher impact score CSV, directory, or glob.")
    parser.add_argument("--out_dir", default="data/processed/pair_audit")
    parser.add_argument("--max_pairs", type=int, default=None, help="Optional smoke limit over pair rows.")
    parser.add_argument("--high_quality_weight_min", type=float, default=0.70)
    parser.add_argument("--high_quality_max_rows", type=int, default=200000)
    parser.add_argument("--progress_every", type=int, default=1000000)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def is_blank(value: object) -> bool:
    return value is None or str(value).strip() == ""


def safe_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if is_blank(value):
        return default
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_int(value: object, default: int = 0) -> int:
    if is_blank(value):
        return int(default)
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return int(default)


def parse_bool(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def parse_intervals(value: object) -> Tuple[Tuple[int, int], ...]:
    if isinstance(value, str):
        data = json.loads(value)
    else:
        data = value
    if not isinstance(data, (list, tuple)):
        raise ValueError("deleted_intervals must be a list")
    intervals: List[Tuple[int, int]] = []
    previous_end: Optional[int] = None
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("interval must be [start, end]")
        start = int(item[0])
        end = int(item[1])
        if start < 0 or end < start:
            raise ValueError("invalid interval [{}, {}]".format(start, end))
        if previous_end is not None and start <= previous_end:
            raise ValueError("intervals overlap or are unsorted")
        intervals.append((start, end))
        previous_end = end
    if not intervals:
        raise ValueError("empty intervals")
    return tuple(intervals)


def deleted_length(intervals: Sequence[Tuple[int, int]]) -> int:
    return sum(int(end) - int(start) + 1 for start, end in intervals)


def interval_intersection_len(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> int:
    i = 0
    j = 0
    total = 0
    a_sorted = sorted(a)
    b_sorted = sorted(b)
    while i < len(a_sorted) and j < len(b_sorted):
        start = max(a_sorted[i][0], b_sorted[j][0])
        end = min(a_sorted[i][1], b_sorted[j][1])
        if start <= end:
            total += end - start + 1
        if a_sorted[i][1] < b_sorted[j][1]:
            i += 1
        else:
            j += 1
    return int(total)


def interval_center(intervals: Sequence[Tuple[int, int]]) -> float:
    total_len = deleted_length(intervals)
    if total_len <= 0:
        return 0.0
    weighted = 0.0
    for start, end in intervals:
        length = end - start + 1
        weighted += 0.5 * (start + end) * length
    return weighted / float(total_len)


def boundary_shift(intervals_a: Sequence[Tuple[int, int]], intervals_b: Sequence[Tuple[int, int]], wildtype_length: int) -> float:
    if not intervals_a or not intervals_b or wildtype_length <= 0:
        return 1.0
    if len(intervals_a) == len(intervals_b):
        shifts = []
        for (a_start, a_end), (b_start, b_end) in zip(sorted(intervals_a), sorted(intervals_b)):
            shifts.append(abs(a_start - b_start) / float(wildtype_length))
            shifts.append(abs(a_end - b_end) / float(wildtype_length))
        return sum(shifts) / float(len(shifts)) if shifts else 0.0
    return (
        abs(min(start for start, _end in intervals_a) - min(start for start, _end in intervals_b))
        + abs(max(end for _start, end in intervals_a) - max(end for _start, end in intervals_b))
    ) / float(max(1, 2 * wildtype_length))


def interval_similarity(
    intervals_a: Sequence[Tuple[int, int]],
    intervals_b: Sequence[Tuple[int, int]],
    wildtype_length: int,
) -> Dict[str, float]:
    len_a = deleted_length(intervals_a)
    len_b = deleted_length(intervals_b)
    intersection = interval_intersection_len(intervals_a, intervals_b)
    union = len_a + len_b - intersection
    center_distance_norm = abs(interval_center(intervals_a) - interval_center(intervals_b)) / float(max(1, wildtype_length))
    return {
        "deleted_len_a": int(len_a),
        "deleted_len_b": int(len_b),
        "intersection": int(intersection),
        "mask_jaccard": intersection / float(union) if union > 0 else 0.0,
        "overlap_coverage": intersection / float(max(1, min(len_a, len_b))),
        "center_distance_norm": center_distance_norm,
        "boundary_shift_mean": boundary_shift(intervals_a, intervals_b, wildtype_length),
    }


def is_terminal(intervals: Sequence[Tuple[int, int]], wildtype_length: int) -> bool:
    return bool(intervals) and (intervals[0][0] == 0 or intervals[-1][1] == int(wildtype_length) - 1)


def topology_relation(
    intervals_a: Sequence[Tuple[int, int]],
    intervals_b: Sequence[Tuple[int, int]],
    wildtype_length: int,
    stats: Mapping[str, float],
) -> str:
    multi_a = len(intervals_a) > 1
    multi_b = len(intervals_b) > 1
    terminal_a = is_terminal(intervals_a, wildtype_length)
    terminal_b = is_terminal(intervals_b, wildtype_length)
    if multi_a != multi_b:
        return "multi_vs_single"
    if terminal_a != terminal_b:
        return "terminal_vs_internal"
    if stats["mask_jaccard"] >= 0.95:
        return "same_interval_like"
    if stats["overlap_coverage"] > 0.0:
        return "shifted_overlap"
    if stats["deleted_len_a"] == stats["deleted_len_b"]:
        return "disjoint_same_budget"
    return "other"


def classify_pair_relation(features: Mapping[str, object]) -> str:
    topology = str(features.get("topology_relation") or "")
    budget_diff = float(features.get("budget_diff") or 0.0)
    jaccard = float(features.get("mask_jaccard") or 0.0)
    coverage = float(features.get("overlap_coverage") or 0.0)
    center = float(features.get("center_distance_norm") or 1.0)
    boundary = float(features.get("boundary_shift_mean") or 1.0)
    if str(features.get("is_valid_pair", "true")).lower() in {"false", "0"}:
        return "unknown"
    if budget_diff > 0.08:
        return "too_far_uninformative"
    if topology in {"multi_vs_single", "terminal_vs_internal"}:
        return "topology_contrast"
    if coverage >= 0.80 and boundary <= 0.03:
        return "local_boundary_contrast"
    if coverage >= 0.30 and (jaccard >= 0.20 or center <= 0.10):
        return "local_shift_contrast"
    if center <= 0.25 and budget_diff <= 0.03:
        return "regional_alternative"
    if center > 0.35 and coverage == 0.0:
        return "too_far_uninformative"
    if coverage == 0.0:
        return "hard_negative_or_far"
    return "unknown"


def recommended_weight(features: Mapping[str, object], relation_type: str) -> float:
    if str(features.get("is_valid_pair", "true")).lower() in {"false", "0"}:
        return 0.1
    preference = str(features.get("preference") or "")
    confidence = safe_float(features.get("preference_confidence"), 0.5) or 0.5
    margin = safe_float(features.get("teacher_margin"), None)
    if margin is None:
        margin = safe_float(features.get("preference_margin"), 0.0) or 0.0
    budget_diff = float(features.get("budget_diff") or 0.0)
    jaccard = float(features.get("mask_jaccard") or 0.0)
    coverage = float(features.get("overlap_coverage") or 0.0)
    center = float(features.get("center_distance_norm") or 1.0)
    topology = str(features.get("topology_relation") or "")

    relation_base = {
        "local_boundary_contrast": 0.82,
        "local_shift_contrast": 0.78,
        "regional_alternative": 0.62,
        "topology_contrast": 0.58,
        "hard_negative_or_far": 0.42,
        "too_far_uninformative": 0.18,
        "unknown": 0.30,
    }.get(relation_type, 0.30)
    if preference == "tie":
        relation_base *= 0.70
    elif preference == "invalid":
        relation_base = 0.10
    locality_bonus = 0.12 * clamp(coverage, 0.0, 1.0) + 0.08 * clamp(jaccard, 0.0, 1.0)
    margin_bonus = min(0.10, max(0.0, float(margin)) * 0.35)
    confidence_factor = 0.65 + 0.35 * clamp(confidence, 0.0, 1.0)
    budget_penalty = min(0.35, max(0.0, budget_diff) * 4.0)
    far_penalty = 0.12 if center > 0.35 and coverage == 0.0 else 0.0
    topology_bonus = 0.04 if topology in {"shifted_overlap", "terminal_vs_internal", "multi_vs_single"} else 0.0
    weight = (relation_base + locality_bonus + margin_bonus + topology_bonus - budget_penalty - far_penalty) * confidence_factor
    return round(clamp(weight, 0.1, 1.0), 6)


def expand_input_paths(spec: str, *, kind: str) -> List[str]:
    if not spec:
        return []
    specs = [item.strip() for item in str(spec).split(",") if item.strip()]
    paths: List[str] = []
    for item in specs:
        if any(char in item for char in "*?[]"):
            paths.extend(glob.glob(item))
        elif os.path.isdir(item):
            patterns = {
                "perturbations": ["deletion_perturbations_shard_*.csv", "deletion_perturbations.csv"],
                "pairs": ["teacher_preference_pairs_shard_*.csv", "teacher_preference_pairs.csv"],
                "teacher_scores": ["teacher_impact_scores_shard_*.csv", "teacher_impact_scores.csv"],
            }.get(kind, ["*.csv"])
            for pattern in patterns:
                paths.extend(glob.glob(os.path.join(item, pattern)))
        elif os.path.isfile(item):
            if kind == "perturbations" and os.path.basename(item) == "deletion_perturbations_shard_index.csv":
                with open(item, newline="") as handle:
                    for row in csv.DictReader(handle):
                        shard_path = row.get("shard_path") or row.get("source_deletion_perturbations_csv")
                        if shard_path:
                            paths.append(shard_path)
            else:
                paths.append(item)
        else:
            paths.extend(glob.glob(item))
    unique = sorted(dict.fromkeys(paths))
    return unique


def shard_number(path: str) -> Optional[int]:
    match = re.search(r"shard_(\d+)\.csv$", os.path.basename(path))
    return int(match.group(1)) if match else None


def aligned_shard_map(paths: Sequence[str]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for path in paths:
        number = shard_number(path)
        if number is not None:
            out[number] = path
    return out


def iter_csv_rows(paths: Sequence[str]) -> Iterator[Tuple[str, Dict[str, str]]]:
    for path in paths:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield path, dict(row)


def collect_pair_perturbation_ids(paths: Sequence[str], max_pairs: Optional[int] = None) -> Tuple[set, int]:
    ids = set()
    total_pairs = 0
    stop = False
    for path in paths:
        if stop:
            break
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                total_pairs += 1
                if row.get("perturbation_id_a"):
                    ids.add(row["perturbation_id_a"])
                if row.get("perturbation_id_b"):
                    ids.add(row["perturbation_id_b"])
                if max_pairs is not None and total_pairs >= int(max_pairs):
                    stop = True
                    break
    return ids, total_pairs


def load_perturbation_index(paths: Sequence[str], needed_ids: Optional[set] = None) -> Tuple[Dict[str, Dict[str, object]], Dict[str, int]]:
    index: Dict[str, Dict[str, object]] = {}
    stats = Counter()
    for path, row in iter_csv_rows(paths):
        stats["rows"] += 1
        perturbation_id = row.get("perturbation_id", "")
        if not perturbation_id:
            stats["missing_id"] += 1
            continue
        if needed_ids is not None and perturbation_id not in needed_ids:
            stats["skipped_not_needed"] += 1
            continue
        try:
            intervals = parse_intervals(row.get("deleted_intervals", ""))
        except Exception:
            stats["parse_failed"] += 1
            continue
        wildtype_length = safe_int(row.get("wildtype_length"), 0)
        deleted_len = safe_int(row.get("deleted_length"), deleted_length(intervals))
        deletion_fraction = safe_float(row.get("deletion_fraction"), None)
        if deletion_fraction is None and wildtype_length > 0:
            deletion_fraction = deleted_len / float(wildtype_length)
        index[perturbation_id] = {
            "protein_id": row.get("protein_id", ""),
            "plan_id": row.get("plan_id", ""),
            "budget_bin": row.get("budget_bin", ""),
            "wildtype_length": wildtype_length,
            "intervals": intervals,
            "deleted_length": deleted_len,
            "deletion_fraction": float(deletion_fraction or 0.0),
            "generation_policy": row.get("generation_policy", ""),
            "source_path": path,
        }
        stats["loaded"] += 1
    return index, dict(stats)


def load_teacher_score_index(paths: Sequence[str]) -> Dict[str, Dict[str, str]]:
    if not paths:
        return {}
    index: Dict[str, Dict[str, str]] = {}
    for _path, row in iter_csv_rows(paths):
        perturbation_id = row.get("perturbation_id", "")
        if perturbation_id:
            index[perturbation_id] = row
    return index


def teacher_desirability(row: Mapping[str, str]) -> Optional[float]:
    if not row:
        return None
    global_shift = safe_float(row.get("global_shift"), None)
    retained = safe_float(row.get("retained_region_shift"), None)
    junction = safe_float(row.get("junction_plausibility"), None)
    if global_shift is None or retained is None or junction is None:
        return None
    return float(junction) - float(global_shift) - float(retained)


def pair_teacher_margin(pair_row: Mapping[str, str], score_index: Mapping[str, Mapping[str, str]]) -> Optional[float]:
    margin = safe_float(pair_row.get("preference_margin"), None)
    if margin is not None:
        return abs(float(margin))
    if not score_index:
        return None
    a = teacher_desirability(score_index.get(pair_row.get("perturbation_id_a", ""), {}))
    b = teacher_desirability(score_index.get(pair_row.get("perturbation_id_b", ""), {}))
    if a is None or b is None:
        return None
    return abs(a - b)


def preferred_candidate(row: Mapping[str, str]) -> str:
    pref = row.get("preference", "")
    if pref == "a_better":
        return row.get("perturbation_id_a", "")
    if pref == "b_better":
        return row.get("perturbation_id_b", "")
    return ""


def annotate_pair(
    pair_row: Mapping[str, str],
    perturbations: Mapping[str, Mapping[str, object]],
    teacher_scores: Mapping[str, Mapping[str, str]],
    source_path: str,
) -> Tuple[Optional[Dict[str, object]], str]:
    a_id = pair_row.get("perturbation_id_a", "")
    b_id = pair_row.get("perturbation_id_b", "")
    a = perturbations.get(a_id)
    b = perturbations.get(b_id)
    if not a or not b:
        return None, "missing_perturbation"
    try:
        intervals_a = a["intervals"]  # type: ignore[assignment]
        intervals_b = b["intervals"]  # type: ignore[assignment]
        wildtype_length = max(int(a.get("wildtype_length", 0)), int(b.get("wildtype_length", 0)), 1)
        stats = interval_similarity(intervals_a, intervals_b, wildtype_length)  # type: ignore[arg-type]
    except Exception:
        return None, "similarity_failed"
    budget_a = float(a.get("deletion_fraction", 0.0))
    budget_b = float(b.get("deletion_fraction", 0.0))
    teacher_margin = pair_teacher_margin(pair_row, teacher_scores)
    features: Dict[str, object] = {
        "budget_diff": abs(budget_a - budget_b),
        "mask_jaccard": stats["mask_jaccard"],
        "overlap_coverage": stats["overlap_coverage"],
        "center_distance_norm": stats["center_distance_norm"],
        "boundary_shift_mean": stats["boundary_shift_mean"],
        "topology_relation": topology_relation(intervals_a, intervals_b, wildtype_length, stats),  # type: ignore[arg-type]
        "teacher_margin": teacher_margin,
        "preference_margin": safe_float(pair_row.get("preference_margin"), None),
        "preference_confidence": safe_float(pair_row.get("preference_confidence"), 0.5),
        "preference": pair_row.get("preference", ""),
        "is_valid_pair": parse_bool(pair_row.get("is_valid_pair", "true")),
    }
    relation = classify_pair_relation(features)
    weight = recommended_weight(features, relation)
    return {
        "protein_id": pair_row.get("protein_id", ""),
        "pair_id": pair_row.get("pair_id", ""),
        "budget_bin": pair_row.get("budget_bin", ""),
        "candidate_a": a_id,
        "candidate_b": b_id,
        "preferred_candidate": preferred_candidate(pair_row),
        "budget_a": "{:.10f}".format(budget_a),
        "budget_b": "{:.10f}".format(budget_b),
        "budget_diff": "{:.10f}".format(float(features["budget_diff"])),
        "deleted_len_a": int(stats["deleted_len_a"]),
        "deleted_len_b": int(stats["deleted_len_b"]),
        "mask_jaccard": "{:.10f}".format(float(stats["mask_jaccard"])),
        "overlap_coverage": "{:.10f}".format(float(stats["overlap_coverage"])),
        "center_distance_norm": "{:.10f}".format(float(stats["center_distance_norm"])),
        "boundary_shift_mean": "{:.10f}".format(float(stats["boundary_shift_mean"])),
        "topology_relation": features["topology_relation"],
        "teacher_margin": "" if teacher_margin is None else "{:.10f}".format(float(teacher_margin)),
        "preference_margin": pair_row.get("preference_margin", ""),
        "preference_confidence": pair_row.get("preference_confidence", ""),
        "pair_relation_type": relation,
        "recommended_weight": "{:.6f}".format(weight),
        "preference": pair_row.get("preference", ""),
        "is_valid_pair": str(features["is_valid_pair"]).lower(),
        "construction_policy": pair_row.get("construction_policy", ""),
        "source_pairs_csv": source_path,
    }, ""


class Reservoir:
    def __init__(self, max_rows: int, key: str = "recommended_weight") -> None:
        self.max_rows = max(0, int(max_rows))
        self.key = key
        self.rows: List[Dict[str, object]] = []

    def add(self, row: Dict[str, object]) -> None:
        if self.max_rows <= 0:
            return
        self.rows.append(row)
        if len(self.rows) > self.max_rows * 2:
            self.trim()

    def trim(self) -> None:
        if len(self.rows) > self.max_rows:
            self.rows.sort(key=lambda item: float(item.get(self.key, 0.0)), reverse=True)
            del self.rows[self.max_rows :]


def percentile(sorted_values: Sequence[float], q: float) -> Optional[float]:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * float(q)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(sorted_values[lower])
    weight = pos - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    return {
        "count": len(clean),
        "p05": percentile(clean, 0.05),
        "p25": percentile(clean, 0.25),
        "p50": percentile(clean, 0.50),
        "p75": percentile(clean, 0.75),
        "p95": percentile(clean, 0.95),
        "mean": (sum(clean) / float(len(clean))) if clean else None,
    }


def bin_label(value: float, bins: Sequence[float]) -> str:
    previous = 0.0
    for edge in bins:
        if value <= edge:
            return "{:.2f}-{:.2f}".format(previous, edge)
        previous = edge
    return "{:.2f}-1.00".format(previous)


def judgement(summary: Mapping[str, object]) -> str:
    relation = summary.get("relation_type_fraction", {})
    weight = summary.get("recommended_weight_quantiles", {})
    high_quality = int(summary.get("high_quality_pairs", 0) or 0)
    total = int(summary.get("total_pairs", 0) or 0)
    parse_success = float(summary.get("parse_success_rate", 0.0) or 0.0)
    local_fraction = float(relation.get("local_boundary_contrast", 0.0) or 0.0) + float(relation.get("local_shift_contrast", 0.0) or 0.0)
    uninformative = float(relation.get("too_far_uninformative", 0.0) or 0.0)
    median_weight = float(weight.get("p50", 0.0) or 0.0)
    high_quality_fraction = high_quality / float(max(1, total))
    if parse_success < 0.98 or high_quality_fraction < 0.05 or uninformative > 0.55:
        return "NEED_PAIR_RECONSTRUCTION"
    if local_fraction < 0.25 or median_weight < 0.45:
        return "NEED_LOCAL_PAIR_FINETUNE"
    return "OK_FOR_FULL_PRETRAIN"


def write_csv(path: str, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: str, payload: Mapping[str, object]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    ensure_dir(args.out_dir)
    perturbation_paths = expand_input_paths(args.perturbations_csv, kind="perturbations")
    pair_paths = expand_input_paths(args.pairs_csv, kind="pairs")
    teacher_paths = expand_input_paths(args.teacher_scores_csv, kind="teacher_scores") if args.teacher_scores_csv else []
    if not perturbation_paths:
        raise SystemExit("no perturbation CSVs resolved")
    if not pair_paths:
        raise SystemExit("no pair CSVs resolved")

    perturbation_shards = aligned_shard_map(perturbation_paths)
    pair_shards = aligned_shard_map(pair_paths)
    shard_aligned = bool(perturbation_shards) and bool(pair_shards) and set(pair_shards).issubset(set(perturbation_shards))
    teacher_shards = aligned_shard_map(teacher_paths) if teacher_paths else {}
    teacher_scores = {} if shard_aligned else (load_teacher_score_index(teacher_paths) if teacher_paths else {})
    planned_pair_rows = None
    needed_ids = set()
    if shard_aligned:
        perturbation_stats = Counter()
    else:
        needed_ids, planned_pair_rows = collect_pair_perturbation_ids(pair_paths, args.max_pairs)
        perturbations, perturbation_stats = load_perturbation_index(perturbation_paths, needed_ids=needed_ids)

    annotated_path = os.path.join(args.out_dir, "pair_similarity_annotated.csv")
    high_quality_path = os.path.join(args.out_dir, "recommended_high_quality_pairs.csv")
    relation_counts = Counter()
    topology_counts = Counter()
    bin_counts = Counter()
    parse_failures = Counter()
    preference_counts = Counter()
    jaccards: List[float] = []
    coverages: List[float] = []
    centers: List[float] = []
    weights: List[float] = []
    total_pairs = 0
    annotated_pairs = 0
    high_quality_count = 0
    high_quality = Reservoir(args.high_quality_max_rows)

    with open(annotated_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATED_COLUMNS)
        writer.writeheader()
        stop = False
        for path in pair_paths:
            if stop:
                break
            if shard_aligned:
                number = shard_number(path)
                deletion_path = perturbation_shards.get(number) if number is not None else None
                if not deletion_path:
                    parse_failures["missing_aligned_deletion_shard"] += 1
                    continue
                local_needed_ids, _local_pair_rows = collect_pair_perturbation_ids([path], None)
                perturbations, local_stats = load_perturbation_index([deletion_path], needed_ids=local_needed_ids)
                perturbation_stats.update(local_stats)
                local_teacher_scores = load_teacher_score_index([teacher_shards[number]]) if number in teacher_shards else {}
            else:
                local_teacher_scores = teacher_scores
            with open(path, newline="") as pair_handle:
                reader = csv.DictReader(pair_handle)
                for pair_row in reader:
                    total_pairs += 1
                    annotated, reason = annotate_pair(pair_row, perturbations, local_teacher_scores, path)
                    if annotated is None:
                        parse_failures[reason] += 1
                    else:
                        writer.writerow({field: annotated.get(field, "") for field in ANNOTATED_COLUMNS})
                        annotated_pairs += 1
                        relation = str(annotated["pair_relation_type"])
                        topology = str(annotated["topology_relation"])
                        weight = float(annotated["recommended_weight"])
                        jaccard = float(annotated["mask_jaccard"])
                        coverage = float(annotated["overlap_coverage"])
                        center = float(annotated["center_distance_norm"])
                        relation_counts[relation] += 1
                        topology_counts[topology] += 1
                        preference_counts[str(annotated["preference"])] += 1
                        bin_counts[("mask_jaccard", bin_label(jaccard, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]))] += 1
                        bin_counts[("overlap_coverage", bin_label(coverage, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]))] += 1
                        bin_counts[("center_distance_norm", bin_label(center, [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]))] += 1
                        bin_counts[("recommended_weight", bin_label(weight, [0.2, 0.4, 0.6, 0.7, 0.8, 0.9]))] += 1
                        jaccards.append(jaccard)
                        coverages.append(coverage)
                        centers.append(center)
                        weights.append(weight)
                        if weight >= float(args.high_quality_weight_min):
                            high_quality_count += 1
                            high_quality.add(annotated)
                    if args.progress_every and total_pairs % int(args.progress_every) == 0:
                        print("processed_pairs={}".format(total_pairs), flush=True)
                    if args.max_pairs is not None and total_pairs >= int(args.max_pairs):
                        stop = True
                        break

    high_quality.trim()
    write_csv(high_quality_path, ANNOTATED_COLUMNS, high_quality.rows)
    relation_rows = [
        {
            "pair_relation_type": key,
            "count": relation_counts.get(key, 0),
            "fraction": relation_counts.get(key, 0) / float(max(1, annotated_pairs)),
        }
        for key in RELATION_TYPES
    ]
    write_csv(os.path.join(args.out_dir, "pair_relation_type_counts.csv"), ["pair_relation_type", "count", "fraction"], relation_rows)
    similarity_rows = [
        {"metric": metric, "bin": label, "count": count, "fraction": count / float(max(1, annotated_pairs))}
        for (metric, label), count in sorted(bin_counts.items())
    ]
    write_csv(os.path.join(args.out_dir, "similarity_bins.csv"), ["metric", "bin", "count", "fraction"], similarity_rows)

    relation_fraction = {key: relation_counts.get(key, 0) / float(max(1, annotated_pairs)) for key in RELATION_TYPES}
    summary: Dict[str, object] = {
        "status": "ok",
        "total_pairs": total_pairs,
        "annotated_pairs": annotated_pairs,
        "parse_success_rate": annotated_pairs / float(max(1, total_pairs)),
        "parse_failures": dict(parse_failures),
        "perturbation_index": perturbation_stats,
        "num_perturbation_csvs": len(perturbation_paths),
        "num_pair_csvs": len(pair_paths),
        "num_teacher_score_csvs": len(teacher_paths),
        "shard_aligned_processing": shard_aligned,
        "planned_pair_rows": planned_pair_rows,
        "needed_perturbation_ids": len(needed_ids),
        "mask_jaccard_quantiles": summarize_values(jaccards),
        "overlap_coverage_quantiles": summarize_values(coverages),
        "center_distance_norm_quantiles": summarize_values(centers),
        "recommended_weight_quantiles": summarize_values(weights),
        "relation_type_counts": dict(relation_counts),
        "relation_type_fraction": relation_fraction,
        "topology_relation_counts": dict(topology_counts),
        "preference_counts": dict(preference_counts),
        "high_quality_pairs": high_quality_count,
        "high_quality_weight_min": float(args.high_quality_weight_min),
        "high_quality_rows_written": len(high_quality.rows),
        "judgement_logic": (
            "NEED_PAIR_RECONSTRUCTION if parse_success<0.98 or high_quality_fraction<0.05 "
            "or too_far_uninformative>0.55; NEED_LOCAL_PAIR_FINETUNE if local_boundary+local_shift<0.25 "
            "or median recommended_weight<0.45; otherwise OK_FOR_FULL_PRETRAIN."
        ),
        "inputs": {
            "perturbations_csv": args.perturbations_csv,
            "pairs_csv": args.pairs_csv,
            "teacher_scores_csv": args.teacher_scores_csv or "",
        },
        "outputs": {
            "pair_similarity_annotated_csv": annotated_path,
            "recommended_high_quality_pairs_csv": high_quality_path,
        },
    }
    summary["judgement"] = judgement(summary)
    write_json(os.path.join(args.out_dir, "pair_similarity_summary.json"), summary)
    print(json.dumps({"status": "ok", "total_pairs": total_pairs, "annotated_pairs": annotated_pairs, "judgement": summary["judgement"], "out_dir": args.out_dir}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
