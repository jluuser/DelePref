#!/usr/bin/env python3
"""Audit WT-vs-deletion-mutant PLM embedding similarity.

This CPU-only utility streams teacher impact and preference-pair CSV shards.
It audits absolute WT-mutant embedding similarity, then checks whether pairwise
preferences compare high-similarity candidates or merely choose low-vs-lower
alternatives.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


QUANTILES = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
SIMILARITY_THRESHOLDS = [0.99, 0.98, 0.95, 0.90]
PAIR_QUALITY_TYPES = [
    "clean_preference",
    "both_high",
    "low_vs_lower",
    "both_low",
    "conflicting",
    "tie_or_uncertain",
    "unknown",
]

LOW_SIMILARITY_COLUMNS = [
    "protein_id",
    "perturbation_id",
    "budget_bin",
    "generation_policy",
    "global_shift",
    "global_similarity",
    "retained_region_shift",
    "retained_similarity",
    "junction_plausibility",
    "impact_confidence",
    "low_similarity_reason",
]

PROBLEM_PAIR_COLUMNS = [
    "protein_id",
    "budget_bin",
    "pair_id",
    "perturbation_id_a",
    "perturbation_id_b",
    "preferred_candidate",
    "dispreferred_candidate",
    "preferred_global_similarity",
    "preferred_retained_similarity",
    "dispreferred_global_similarity",
    "dispreferred_retained_similarity",
    "delta_global_similarity",
    "delta_retained_similarity",
    "preference",
    "preference_margin",
    "preference_confidence",
    "pair_embedding_quality_type",
    "generation_policy_a",
    "generation_policy_b",
    "source_pairs_csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit WT-mutant embedding similarity in teacher impact and preference pair tables.")
    parser.add_argument("--teacher_impacts_glob", required=True)
    parser.add_argument("--pairs_glob", default=None)
    parser.add_argument("--out_dir", default="data/processed/pair_audit/embedding_similarity")
    parser.add_argument("--max_impact_rows", type=int, default=None)
    parser.add_argument("--max_pair_rows", type=int, default=None)
    parser.add_argument("--pair_stride", type=int, default=1)
    parser.add_argument("--low_similarity_max_rows", type=int, default=200000)
    parser.add_argument("--problem_pair_max_rows", type=int, default=200000)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def safe_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return default
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def similarity_from_shift(value: object) -> Optional[float]:
    shift = safe_float(value, None)
    if shift is None:
        return None
    return 1.0 - 2.0 * float(shift)


def parse_policy_budget_from_perturbation_id(perturbation_id: str) -> Dict[str, str]:
    parts = str(perturbation_id or "").split("__")
    result = {"generation_policy": "", "budget_bin": ""}
    if len(parts) >= 4 and parts[1].startswith("formalqv"):
        result["generation_policy"] = parts[2]
        result["budget_bin"] = parts[3]
        return result
    for part in parts:
        if re.match(r"^b\d", part):
            result["budget_bin"] = part
            break
    if len(parts) >= 3:
        result["generation_policy"] = parts[2]
    return result


def expand_paths(spec: Optional[str]) -> List[str]:
    if not spec:
        return []
    paths: List[str] = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if os.path.isdir(item):
            paths.extend(glob.glob(os.path.join(item, "*.csv")))
        else:
            paths.extend(glob.glob(item))
    return sorted(dict.fromkeys(paths))


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
    out: Dict[str, Optional[float]] = {"count": len(clean)}
    for q in QUANTILES:
        label = "p{}".format(str(q * 100).rstrip("0").rstrip("."))
        out[label] = percentile(clean, q)
    out["mean"] = sum(clean) / float(len(clean)) if clean else None
    return out


class GroupStats:
    def __init__(self) -> None:
        self.n = 0
        self.global_similarity: List[float] = []
        self.retained_similarity: List[float] = []
        self.global_low_counts = Counter()
        self.retained_low_counts = Counter()

    def add(self, global_similarity: Optional[float], retained_similarity: Optional[float]) -> None:
        self.n += 1
        if global_similarity is not None:
            self.global_similarity.append(global_similarity)
            for threshold in SIMILARITY_THRESHOLDS:
                if global_similarity < threshold:
                    self.global_low_counts[str(threshold)] += 1
        if retained_similarity is not None:
            self.retained_similarity.append(retained_similarity)
            for threshold in SIMILARITY_THRESHOLDS:
                if retained_similarity < threshold:
                    self.retained_low_counts[str(threshold)] += 1

    def row(self, group_name: str, group_value: str) -> Dict[str, object]:
        row: Dict[str, object] = {group_name: group_value, "n": self.n}
        for prefix, values in [("global_similarity", self.global_similarity), ("retained_similarity", self.retained_similarity)]:
            summary = summarize_values(values)
            for key in ["p5", "p10", "p25", "p50", "p75", "p90", "p95", "mean"]:
                row["{}_{}".format(prefix, key)] = summary.get(key)
        for threshold in SIMILARITY_THRESHOLDS:
            row["global_similarity_lt_{}".format(str(threshold).replace(".", "_"))] = self.global_low_counts[str(threshold)] / float(max(1, self.n))
            row["retained_similarity_lt_{}".format(str(threshold).replace(".", "_"))] = self.retained_low_counts[str(threshold)] / float(max(1, self.n))
        return row


def csv_index(header: Sequence[str], name: str) -> int:
    try:
        return list(header).index(name)
    except ValueError:
        return -1


def iter_csv_rows_by_index(paths: Sequence[str]) -> Iterator[Tuple[str, List[str], List[str]]]:
    for path in paths:
        with open(path, newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                continue
            for row in reader:
                yield path, header, row


def value_at(row: Sequence[str], index: int) -> str:
    return row[index] if 0 <= index < len(row) else ""


def low_similarity_reason(global_similarity: Optional[float], retained_similarity: Optional[float]) -> str:
    reasons = []
    if global_similarity is not None and global_similarity < 0.95:
        reasons.append("global_similarity_lt_0.95")
    if retained_similarity is not None and retained_similarity < 0.90:
        reasons.append("retained_similarity_lt_0.90")
    return ";".join(reasons)


def impact_judgement(summary: Mapping[str, object]) -> str:
    global_low_95 = float(summary.get("global_similarity_fraction_below", {}).get("0.95", 0.0))
    global_low_90 = float(summary.get("global_similarity_fraction_below", {}).get("0.90", 0.0))
    retained_low_95 = float(summary.get("retained_similarity_fraction_below", {}).get("0.95", 0.0))
    retained_low_90 = float(summary.get("retained_similarity_fraction_below", {}).get("0.90", 0.0))
    if global_low_90 > 0.05 or retained_low_90 > 0.10 or global_low_95 > 0.20:
        return "FAILED_TOO_MANY_LOW_SIMILARITY"
    if global_low_95 > 0.05 or retained_low_90 > 0.02:
        return "WARNING_LOW_SIMILARITY_TAIL"
    if retained_low_95 > 0.05:
        return "OK_GLOBAL_BUT_RETAINED_TAIL"
    return "OK_HIGH_SIMILARITY"


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


def audit_impacts(paths: Sequence[str], out_dir: str, max_rows: Optional[int], low_similarity_max_rows: int) -> Dict[str, object]:
    total_rows = 0
    ok_rows = 0
    parse_success = 0
    low_similarity_total = 0
    low_rows_written = 0
    global_shift_values: List[float] = []
    retained_shift_values: List[float] = []
    global_similarity_values: List[float] = []
    retained_similarity_values: List[float] = []
    junction_values: List[float] = []
    global_low_counts = Counter()
    retained_low_counts = Counter()
    by_budget: Dict[str, GroupStats] = defaultdict(GroupStats)
    by_policy: Dict[str, GroupStats] = defaultdict(GroupStats)
    low_path = os.path.join(out_dir, "low_similarity_candidates.csv")
    ensure_dir(out_dir)
    with open(low_path, "w", newline="") as low_handle:
        low_writer = csv.DictWriter(low_handle, fieldnames=LOW_SIMILARITY_COLUMNS)
        low_writer.writeheader()
        cached_indices: Dict[Tuple[str, ...], Dict[str, int]] = {}
        stop = False
        for _path, header, row in iter_csv_rows_by_index(paths):
            if stop:
                break
            key = tuple(header)
            if key not in cached_indices:
                cached_indices[key] = {name: csv_index(header, name) for name in [
                    "protein_id",
                    "perturbation_id",
                    "global_shift",
                    "retained_region_shift",
                    "junction_plausibility",
                    "impact_confidence",
                    "scoring_status",
                    "metadata_json",
                ]}
            idx = cached_indices[key]
            total_rows += 1
            scoring_status = value_at(row, idx["scoring_status"])
            if scoring_status == "ok":
                ok_rows += 1
            global_shift = safe_float(value_at(row, idx["global_shift"]), None)
            retained_shift = safe_float(value_at(row, idx["retained_region_shift"]), None)
            junction = safe_float(value_at(row, idx["junction_plausibility"]), None)
            global_similarity = similarity_from_shift(global_shift)
            retained_similarity = similarity_from_shift(retained_shift)
            if global_similarity is not None and retained_similarity is not None:
                parse_success += 1
            if global_shift is not None:
                global_shift_values.append(global_shift)
            if retained_shift is not None:
                retained_shift_values.append(retained_shift)
            if global_similarity is not None:
                global_similarity_values.append(global_similarity)
                for threshold in SIMILARITY_THRESHOLDS:
                    if global_similarity < threshold:
                        global_low_counts[str(threshold)] += 1
            if retained_similarity is not None:
                retained_similarity_values.append(retained_similarity)
                for threshold in SIMILARITY_THRESHOLDS:
                    if retained_similarity < threshold:
                        retained_low_counts[str(threshold)] += 1
            if junction is not None:
                junction_values.append(junction)
            perturbation_id = value_at(row, idx["perturbation_id"])
            parsed = parse_policy_budget_from_perturbation_id(perturbation_id)
            generation_policy = parsed["generation_policy"]
            budget_bin = parsed["budget_bin"]
            if (not generation_policy or not budget_bin) and idx["metadata_json"] >= 0:
                try:
                    meta = json.loads(value_at(row, idx["metadata_json"]) or "{}")
                    generation_policy = generation_policy or str(meta.get("generation_policy", ""))
                except Exception:
                    pass
            by_budget[budget_bin or "unknown"].add(global_similarity, retained_similarity)
            by_policy[generation_policy or "unknown"].add(global_similarity, retained_similarity)
            reason = low_similarity_reason(global_similarity, retained_similarity)
            if reason:
                low_similarity_total += 1
                if low_rows_written < int(low_similarity_max_rows):
                    low_writer.writerow(
                        {
                            "protein_id": value_at(row, idx["protein_id"]),
                            "perturbation_id": perturbation_id,
                            "budget_bin": budget_bin,
                            "generation_policy": generation_policy,
                            "global_shift": value_at(row, idx["global_shift"]),
                            "global_similarity": "" if global_similarity is None else "{:.10f}".format(global_similarity),
                            "retained_region_shift": value_at(row, idx["retained_region_shift"]),
                            "retained_similarity": "" if retained_similarity is None else "{:.10f}".format(retained_similarity),
                            "junction_plausibility": value_at(row, idx["junction_plausibility"]),
                            "impact_confidence": value_at(row, idx["impact_confidence"]),
                            "low_similarity_reason": reason,
                        }
                    )
                    low_rows_written += 1
            if max_rows is not None and total_rows >= int(max_rows):
                stop = True
    budget_rows = [stats.row("budget_bin", key) for key, stats in sorted(by_budget.items())]
    policy_rows = [stats.row("generation_policy", key) for key, stats in sorted(by_policy.items())]
    write_csv(os.path.join(out_dir, "embedding_similarity_by_budget.csv"), list(budget_rows[0].keys()) if budget_rows else ["budget_bin", "n"], budget_rows)
    write_csv(os.path.join(out_dir, "embedding_similarity_by_generation_policy.csv"), list(policy_rows[0].keys()) if policy_rows else ["generation_policy", "n"], policy_rows)
    summary: Dict[str, object] = {
        "status": "ok",
        "total_rows": total_rows,
        "ok_rows": ok_rows,
        "parse_success_rate": parse_success / float(max(1, total_rows)),
        "global_shift_quantiles": summarize_values(global_shift_values),
        "global_similarity_quantiles": summarize_values(global_similarity_values),
        "retained_region_shift_quantiles": summarize_values(retained_shift_values),
        "retained_similarity_quantiles": summarize_values(retained_similarity_values),
        "junction_plausibility_quantiles": summarize_values(junction_values),
        "global_similarity_fraction_below": {str(t): global_low_counts[str(t)] / float(max(1, len(global_similarity_values))) for t in SIMILARITY_THRESHOLDS},
        "retained_similarity_fraction_below": {str(t): retained_low_counts[str(t)] / float(max(1, len(retained_similarity_values))) for t in SIMILARITY_THRESHOLDS},
        "low_similarity_candidates": low_similarity_total,
        "low_similarity_candidates_fraction": low_similarity_total / float(max(1, total_rows)),
        "low_similarity_rows_written": low_rows_written,
        "judgement_logic": (
            "FAILED_TOO_MANY_LOW_SIMILARITY if global<0.90 fraction>0.05 or retained<0.90 fraction>0.10 "
            "or global<0.95 fraction>0.20; WARNING_LOW_SIMILARITY_TAIL if global<0.95 fraction>0.05 "
            "or retained<0.90 fraction>0.02; OK_GLOBAL_BUT_RETAINED_TAIL if retained<0.95 fraction>0.05; "
            "otherwise OK_HIGH_SIMILARITY."
        ),
    }
    summary["judgement"] = impact_judgement(summary)
    write_json(os.path.join(out_dir, "embedding_similarity_summary.json"), summary)
    return summary


def pick_preferred_side(preference: str) -> Tuple[Optional[str], Optional[str]]:
    if preference == "a_better":
        return "a", "b"
    if preference == "b_better":
        return "b", "a"
    return None, None


def classify_pair_quality(
    *,
    preference: str,
    preference_confidence: float,
    preferred_global: Optional[float],
    preferred_retained: Optional[float],
    dispreferred_global: Optional[float],
    dispreferred_retained: Optional[float],
) -> str:
    if preference == "tie" or preference_confidence < 0.55:
        return "tie_or_uncertain"
    if None in {preferred_global, preferred_retained, dispreferred_global, dispreferred_retained}:
        return "unknown"
    pg = float(preferred_global)
    pr = float(preferred_retained)
    dg = float(dispreferred_global)
    dr = float(dispreferred_retained)
    if pg < dg and pr < dr:
        return "conflicting"
    if (pg < 0.95 or pr < 0.90) and (dg < 0.95 or dr < 0.90):
        return "both_low"
    if (pg < 0.98 or pr < 0.95) and (pg >= dg or pr >= dr):
        return "low_vs_lower"
    if pg >= 0.98 and dg >= 0.98 and pr >= 0.95 and dr >= 0.95:
        return "both_high"
    if pg >= 0.98 and pr >= 0.95:
        return "clean_preference"
    return "unknown"


def side_similarity(side: Mapping[str, object], key: str) -> Optional[float]:
    return similarity_from_shift(side.get(key))


def pair_budget_row(budget_bin: str, counts: Mapping[str, int]) -> Dict[str, object]:
    total = sum(int(value) for value in counts.values())
    row: Dict[str, object] = {"budget_bin": budget_bin, "sampled_pair_rows": total}
    for quality in PAIR_QUALITY_TYPES:
        row["{}_count".format(quality)] = int(counts.get(quality, 0))
        row["{}_fraction".format(quality)] = int(counts.get(quality, 0)) / float(max(1, total))
    return row


def audit_pairs(paths: Sequence[str], out_dir: str, max_rows: Optional[int], stride: int, problem_pair_max_rows: int) -> Dict[str, object]:
    if not paths:
        return {"status": "skipped_no_pairs_glob"}
    stride = max(1, int(stride))
    total_pair_rows_seen = 0
    sampled_pair_rows = 0
    parse_failures = Counter()
    quality_counts = Counter()
    by_budget_counts: Dict[str, Counter] = defaultdict(Counter)
    preferred_global_values: List[float] = []
    preferred_retained_values: List[float] = []
    dispreferred_global_values: List[float] = []
    dispreferred_retained_values: List[float] = []
    problem_rows_written = 0
    problem_path = os.path.join(out_dir, "problem_pairs.csv")
    ensure_dir(out_dir)
    with open(problem_path, "w", newline="") as problem_handle:
        writer = csv.DictWriter(problem_handle, fieldnames=PROBLEM_PAIR_COLUMNS)
        writer.writeheader()
        stop = False
        cached_indices: Dict[Tuple[str, ...], Dict[str, int]] = {}
        for path, header, row in iter_csv_rows_by_index(paths):
            if stop:
                break
            total_pair_rows_seen += 1
            if (total_pair_rows_seen - 1) % stride != 0:
                continue
            sampled_pair_rows += 1
            key = tuple(header)
            if key not in cached_indices:
                cached_indices[key] = {name: csv_index(header, name) for name in [
                    "protein_id",
                    "budget_bin",
                    "pair_id",
                    "perturbation_id_a",
                    "perturbation_id_b",
                    "preference",
                    "preference_margin",
                    "preference_confidence",
                    "comparison_basis_json",
                ]}
            idx = cached_indices[key]
            try:
                basis = json.loads(value_at(row, idx["comparison_basis_json"]))
                a = basis.get("a") or {}
                b = basis.get("b") or {}
            except Exception:
                parse_failures["comparison_basis_json"] += 1
                quality = "unknown"
                quality_counts[quality] += 1
                by_budget_counts[value_at(row, idx["budget_bin"])][quality] += 1
                continue
            preference = value_at(row, idx["preference"])
            confidence = safe_float(value_at(row, idx["preference_confidence"]), 0.0) or 0.0
            preferred_side, dispreferred_side = pick_preferred_side(preference)
            side_map = {"a": a, "b": b}
            if preferred_side is None or dispreferred_side is None:
                preferred = a
                dispreferred = b
            else:
                preferred = side_map[preferred_side]
                dispreferred = side_map[dispreferred_side]
            preferred_global = side_similarity(preferred, "global_shift")
            preferred_retained = side_similarity(preferred, "retained_region_shift")
            dispreferred_global = side_similarity(dispreferred, "global_shift")
            dispreferred_retained = side_similarity(dispreferred, "retained_region_shift")
            for target, values in [
                (preferred_global, preferred_global_values),
                (preferred_retained, preferred_retained_values),
                (dispreferred_global, dispreferred_global_values),
                (dispreferred_retained, dispreferred_retained_values),
            ]:
                if target is not None:
                    values.append(float(target))
            quality = classify_pair_quality(
                preference=preference,
                preference_confidence=confidence,
                preferred_global=preferred_global,
                preferred_retained=preferred_retained,
                dispreferred_global=dispreferred_global,
                dispreferred_retained=dispreferred_retained,
            )
            quality_counts[quality] += 1
            budget_bin = value_at(row, idx["budget_bin"]) or str(a.get("budget_bin") or b.get("budget_bin") or "")
            by_budget_counts[budget_bin or "unknown"][quality] += 1
            if quality in {"low_vs_lower", "both_low", "conflicting"} and problem_rows_written < int(problem_pair_max_rows):
                delta_global = None if preferred_global is None or dispreferred_global is None else preferred_global - dispreferred_global
                delta_retained = None if preferred_retained is None or dispreferred_retained is None else preferred_retained - dispreferred_retained
                writer.writerow(
                    {
                        "protein_id": value_at(row, idx["protein_id"]),
                        "budget_bin": budget_bin,
                        "pair_id": value_at(row, idx["pair_id"]),
                        "perturbation_id_a": value_at(row, idx["perturbation_id_a"]),
                        "perturbation_id_b": value_at(row, idx["perturbation_id_b"]),
                        "preferred_candidate": str(preferred.get("perturbation_id", "")),
                        "dispreferred_candidate": str(dispreferred.get("perturbation_id", "")),
                        "preferred_global_similarity": "" if preferred_global is None else "{:.10f}".format(preferred_global),
                        "preferred_retained_similarity": "" if preferred_retained is None else "{:.10f}".format(preferred_retained),
                        "dispreferred_global_similarity": "" if dispreferred_global is None else "{:.10f}".format(dispreferred_global),
                        "dispreferred_retained_similarity": "" if dispreferred_retained is None else "{:.10f}".format(dispreferred_retained),
                        "delta_global_similarity": "" if delta_global is None else "{:.10f}".format(delta_global),
                        "delta_retained_similarity": "" if delta_retained is None else "{:.10f}".format(delta_retained),
                        "preference": preference,
                        "preference_margin": value_at(row, idx["preference_margin"]),
                        "preference_confidence": value_at(row, idx["preference_confidence"]),
                        "pair_embedding_quality_type": quality,
                        "generation_policy_a": str(a.get("generation_policy", "")),
                        "generation_policy_b": str(b.get("generation_policy", "")),
                        "source_pairs_csv": path,
                    }
                )
                problem_rows_written += 1
            if max_rows is not None and sampled_pair_rows >= int(max_rows):
                stop = True
    by_budget_rows = [pair_budget_row(key, counts) for key, counts in sorted(by_budget_counts.items())]
    fieldnames = list(by_budget_rows[0].keys()) if by_budget_rows else ["budget_bin", "sampled_pair_rows"]
    write_csv(os.path.join(out_dir, "pair_embedding_quality_by_budget.csv"), fieldnames, by_budget_rows)
    fractions = {quality: quality_counts.get(quality, 0) / float(max(1, sampled_pair_rows)) for quality in PAIR_QUALITY_TYPES}
    summary = {
        "status": "ok",
        "total_pair_rows_seen": total_pair_rows_seen,
        "sampled_pair_rows": sampled_pair_rows,
        "pair_stride": stride,
        "parse_failures": dict(parse_failures),
        "quality_type_counts": dict(quality_counts),
        "quality_type_fraction": fractions,
        "preferred_global_similarity_quantiles": summarize_values(preferred_global_values),
        "preferred_retained_similarity_quantiles": summarize_values(preferred_retained_values),
        "dispreferred_global_similarity_quantiles": summarize_values(dispreferred_global_values),
        "dispreferred_retained_similarity_quantiles": summarize_values(dispreferred_retained_values),
        "low_vs_lower_fraction": fractions.get("low_vs_lower", 0.0),
        "both_low_fraction": fractions.get("both_low", 0.0),
        "conflicting_fraction": fractions.get("conflicting", 0.0),
        "problem_pairs_written": problem_rows_written,
        "by_budget_quality_type_fraction": {
            key: {quality: counts.get(quality, 0) / float(max(1, sum(counts.values()))) for quality in PAIR_QUALITY_TYPES}
            for key, counts in sorted(by_budget_counts.items())
        },
    }
    write_json(os.path.join(out_dir, "pair_embedding_quality_summary.json"), summary)
    return summary


def main() -> int:
    args = parse_args()
    ensure_dir(args.out_dir)
    impact_paths = expand_paths(args.teacher_impacts_glob)
    pair_paths = expand_paths(args.pairs_glob)
    if not impact_paths:
        raise SystemExit("no teacher impact CSV files resolved")
    impact_summary = audit_impacts(impact_paths, args.out_dir, args.max_impact_rows, args.low_similarity_max_rows)
    pair_summary = audit_pairs(pair_paths, args.out_dir, args.max_pair_rows, args.pair_stride, args.problem_pair_max_rows) if pair_paths else {"status": "skipped_no_pairs_glob"}
    run_summary = {
        "status": "ok",
        "teacher_impact_csv_count": len(impact_paths),
        "pair_csv_count": len(pair_paths),
        "impact_summary_path": os.path.join(args.out_dir, "embedding_similarity_summary.json"),
        "pair_summary_path": os.path.join(args.out_dir, "pair_embedding_quality_summary.json") if pair_paths else "",
        "impact_judgement": impact_summary.get("judgement"),
        "pair_status": pair_summary.get("status"),
        "inputs": {
            "teacher_impacts_glob": args.teacher_impacts_glob,
            "pairs_glob": args.pairs_glob or "",
            "max_impact_rows": args.max_impact_rows,
            "max_pair_rows": args.max_pair_rows,
            "pair_stride": args.pair_stride,
        },
    }
    write_json(os.path.join(args.out_dir, "embedding_similarity_run_summary.json"), run_summary)
    print(json.dumps(run_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
