#!/usr/bin/env python3
"""Independently audit strict same-count DelePref pair shards on CPU."""

import argparse
import csv
import importlib.util
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BUILDER_PATH = SCRIPT_DIR / "build_pair_same_count.py"
SPEC = importlib.util.spec_from_file_location("pair_builder_for_audit", str(BUILDER_PATH))
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)

DEFAULT_OUTPUT = Path("data/processed/preference_training/delepref/pair_same_count")
FLOAT_TOLERANCE = 5.0e-8
MAX_FAILURE_EXAMPLES = 8

VALIDATION_INDEX_COLUMNS = (
    "shard_index", "source_candidate_csv", "source_impact_csv", "pair_csv",
    "source_candidates", "source_impacts", "source_proteins", "pairs",
    "full_proteins", "shortage_proteins", "failure_count", "status",
    "runtime_seconds",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-index", type=Path, default=builder.DEFAULT_CANDIDATE_INDEX)
    parser.add_argument("--impact-index", type=Path, default=builder.DEFAULT_IMPACT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shards", default="", help="Comma-separated shard indices; empty means all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="CPU shard workers; each holds only one local shard")
    return parser.parse_args()


def bool_value(value):
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise ValueError("not a canonical boolean: %s" % value)


def close_float(a, b, tolerance=FLOAT_TOLERANCE):
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def histogram_add(counter, value, edges):
    value = float(value)
    lower = float("-inf")
    for upper in edges:
        if value <= upper:
            counter["(%s,%s]" % ("-inf" if lower == float("-inf") else "%g" % lower, "%g" % upper)] += 1
            return
        lower = upper
    counter["(%g,inf)" % edges[-1]] += 1


class NumericSummary(object):
    def __init__(self, edges):
        self.edges = tuple(edges)
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None
        self.histogram = Counter()

    def add(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("non-finite distribution value")
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        histogram_add(self.histogram, value, self.edges)

    def payload(self):
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.total / self.count if self.count else None,
            "histogram": dict(self.histogram),
        }


class FailureCollector(object):
    def __init__(self):
        self.counts = Counter()
        self.examples = defaultdict(list)

    def add(self, reason, row_number=None, detail=""):
        self.counts[reason] += 1
        if len(self.examples[reason]) < MAX_FAILURE_EXAMPLES:
            self.examples[reason].append({"row_number": row_number, "detail": str(detail)[:500]})

    @property
    def total(self):
        return sum(self.counts.values())

    def payload(self):
        return {"counts": dict(self.counts), "examples": dict(self.examples), "total": self.total}


def load_sources(candidate_path, impact_path):
    impacts, impact_rows, invalid_impacts = builder.load_impacts(impact_path)
    state = builder.new_shard_counters()
    candidates = {}
    protein_candidates = defaultdict(list)
    protein_splits = {}
    for protein_id, group in builder.iter_candidate_groups(candidate_path, impacts, state):
        for candidate in group:
            key = (candidate["protein_id"], candidate["perturbation_id"], candidate["plan_id"])
            candidates[key] = candidate
            protein_candidates[protein_id].append(candidate)
            split = candidate["source_split"]
            if protein_id in protein_splits and protein_splits[protein_id] != split:
                raise ValueError("source protein has multiple splits")
            protein_splits[protein_id] = split
    return {
        "impacts": impacts,
        "impact_rows": impact_rows,
        "invalid_impacts": invalid_impacts,
        "state": state,
        "candidates": candidates,
        "protein_candidates": protein_candidates,
        "protein_splits": protein_splits,
    }


def annotated_candidate_map(protein_candidates):
    result = {}
    group_stats = {}
    duplicate_counter = Counter()
    by_length = defaultdict(list)
    for candidate in protein_candidates:
        by_length[candidate["deleted_length"]].append(candidate)
    for deleted_length, raw_group in by_length.items():
        group = builder.unique_masks(raw_group, duplicate_counter)
        if len(group) < 2:
            continue
        stats = builder.annotate_group(group)
        group_stats[deleted_length] = stats
        for candidate in group:
            result[(candidate["perturbation_id"], candidate["plan_id"])] = candidate
    return result, group_stats


def compare_source_fields(row, suffix, candidate, failures, row_number):
    exact_fields = {
        "perturbation_id": "perturbation_id",
        "plan_id": "plan_id",
        "deleted_intervals": "intervals_json",
        "num_intervals": "num_intervals",
        "budget_bin": "budget_bin",
        "generation_policy": "generation_policy",
        "candidate_quality": "candidate_quality",
    }
    for output_name, source_name in exact_fields.items():
        if str(row[output_name + "_" + suffix]) != str(candidate[source_name]):
            failures.add("source_field_mismatch", row_number, "%s_%s" % (output_name, suffix))
    numeric_fields = {
        "deletion_fraction": "deletion_fraction",
        "global_similarity": "global_similarity",
        "retained_similarity": "retained_similarity",
        "junction_plausibility": "junction",
    }
    for output_name, source_name in numeric_fields.items():
        try:
            if not close_float(row[output_name + "_" + suffix], candidate[source_name]):
                failures.add("source_numeric_mismatch", row_number, "%s_%s" % (output_name, suffix))
        except Exception as exc:
            failures.add("invalid_numeric_output", row_number, "%s_%s: %s" % (output_name, suffix, exc))


def audit_pair_row(row, row_number, shard_index, sources, annotated_by_protein, group_stats_by_protein,
                   thresholds, pair_ids, unordered, protein_pair_counts, failures, distributions, counters):
    if set(row) != set(builder.PAIR_COLUMNS):
        failures.add("pair_schema_mismatch", row_number)
        return
    protein_id = row["protein_id"]
    a_key = (protein_id, row["perturbation_id_a"], row["plan_id_a"])
    b_key = (protein_id, row["perturbation_id_b"], row["plan_id_b"])
    candidate_a = sources["candidates"].get(a_key)
    candidate_b = sources["candidates"].get(b_key)
    counters["source_references"] += 2
    if candidate_a is not None:
        counters["matched_source_candidate_references"] += 1
    if candidate_b is not None:
        counters["matched_source_candidate_references"] += 1
    if a_key in sources["impacts"]:
        counters["matched_source_impact_references"] += 1
    if b_key in sources["impacts"]:
        counters["matched_source_impact_references"] += 1
    if candidate_a is None or candidate_b is None:
        failures.add("missing_source_candidate_link", row_number)
        return
    if a_key not in sources["impacts"] or b_key not in sources["impacts"]:
        failures.add("missing_source_impact_link", row_number)
        return
    if candidate_a["protein_id"] != candidate_b["protein_id"] or candidate_a["protein_id"] != protein_id:
        failures.add("cross_protein_pair", row_number)
    if candidate_a["deleted_length"] != candidate_b["deleted_length"]:
        failures.add("unequal_deleted_length", row_number)
        counters["unequal_count_pairs"] += 1
    else:
        counters["same_count_pairs"] += 1
    if row["perturbation_id_a"] == row["perturbation_id_b"]:
        failures.add("self_pair", row_number)
    unordered_key = (protein_id,) + tuple(sorted((row["perturbation_id_a"], row["perturbation_id_b"])))
    if unordered_key in unordered:
        failures.add("duplicate_unordered_pair", row_number)
    unordered.add(unordered_key)
    if row["pair_id"] in pair_ids:
        failures.add("duplicate_pair_id", row_number)
    pair_ids.add(row["pair_id"])
    protein_pair_counts[protein_id] += 1
    if row["construction_policy"] != "strict_same_deleted_length_pair":
        failures.add("construction_policy_not_strict_same_count", row_number)
    if row["selection_mode"] not in ("quota", "same_deleted_length_refill"):
        failures.add("invalid_selection_mode", row_number)
    if str(row["source_shard_index"]) != str(shard_index):
        failures.add("source_shard_index_mismatch", row_number)
    if row["source_split"] != candidate_a["source_split"] or row["source_split"] != candidate_b["source_split"]:
        failures.add("source_split_mismatch", row_number)
    if int(row["deleted_length"]) != candidate_a["deleted_length"]:
        failures.add("output_deleted_length_mismatch", row_number)
    if row["pair_budget_key"] != "deleted_length:%d" % candidate_a["deleted_length"]:
        failures.add("pair_budget_key_mismatch", row_number)
    compare_source_fields(row, "a", candidate_a, failures, row_number)
    compare_source_fields(row, "b", candidate_b, failures, row_number)
    if candidate_a["candidate_quality"] == candidate_b["candidate_quality"] == "acceptable":
        failures.add("both_low_pair_present", row_number)
    try:
        confidence = float(row["preference_confidence"])
        weight = float(row["scorer_training_weight"])
        agreement = float(row["view_agreement"])
        if not (0.0 <= confidence <= 1.0 and 0.0 <= agreement <= 1.0 and 0.0 <= weight <= 1.0):
            failures.add("bounded_metric_out_of_range", row_number)
        distributions["confidence"].add(confidence)
        distributions["utility_margin"].add(float(row["preference_margin"]))
        distributions["mask_similarity"].add(float(row["mask_similarity"]))
    except Exception as exc:
        failures.add("invalid_pair_numeric", row_number, exc)
    try:
        conflicting = bool_value(row["view_conflicting"])
        policy_eligible = bool_value(row["policy_positive_eligible"])
        if conflicting:
            counters["view_conflicting_pairs"] += 1
            if policy_eligible:
                failures.add("view_conflict_policy_eligible", row_number)
            if row["pair_quality_bucket"] != "diagnostic_view_conflict" or not close_float(row["scorer_training_weight"], 0.05):
                failures.add("view_conflict_not_low_weight_diagnostic", row_number)
        if row["preference"] == "tie":
            counters["tie_pairs"] += 1
            if policy_eligible:
                failures.add("tie_policy_eligible", row_number)
        if policy_eligible:
            counters["policy_positive_eligible_pairs"] += 1
    except Exception as exc:
        failures.add("invalid_boolean_output", row_number, exc)
    direct_a = annotated_by_protein.get(protein_id, {}).get((candidate_a["perturbation_id"], candidate_a["plan_id"]))
    direct_b = annotated_by_protein.get(protein_id, {}).get((candidate_b["perturbation_id"], candidate_b["plan_id"]))
    stats = group_stats_by_protein.get(protein_id, {}).get(candidate_a["deleted_length"])
    if direct_a is None or direct_b is None or stats is None:
        failures.add("pair_candidate_removed_or_group_unavailable", row_number)
    else:
        expected, excluded = builder.make_pair(
            protein_id, candidate_a["deleted_length"], int(row["group_size"]),
            direct_a, direct_b, stats, thresholds, shard_index,
        )
        if excluded:
            failures.add("pair_should_have_been_excluded", row_number, excluded)
        else:
            exact_derived = (
                "preference", "selection_category", "pair_quality_bucket",
                "policy_positive_eligible", "view_conflicting",
            )
            for field in exact_derived:
                if str(row[field]) != str(expected[field]):
                    failures.add("derived_field_mismatch", row_number, field)
            numeric_derived = (
                "preference_margin", "preference_confidence", "tie_margin",
                "utility_a", "utility_b", "utility_margin_signed",
                "view_agreement", "absolute_similarity_score", "mask_similarity",
                "centroid_distance_fraction", "scorer_training_weight",
            )
            for field in numeric_derived:
                if not close_float(row[field], expected[field]):
                    failures.add("derived_numeric_mismatch", row_number, field)
    try:
        comparison = json.loads(row["comparison_basis_json"])
        if comparison.get("source_preference_pairs_used") is not False:
            failures.add("comparison_basis_old_pair_source", row_number)
        if comparison.get("teacher_preference_is_ground_truth") is not False:
            failures.add("unsafe_teacher_ground_truth_claim", row_number)
        if comparison.get("group_scope") != [protein_id, candidate_a["deleted_length"]]:
            failures.add("comparison_group_scope_mismatch", row_number)
    except Exception as exc:
        failures.add("invalid_comparison_basis_json", row_number, exc)
    counters["category:" + row["selection_category"]] += 1
    counters["quality:" + row["pair_quality_bucket"]] += 1
    counters["preference:" + row["preference"]] += 1
    counters["generation_policy:" + row["generation_policy_a"] + "|" + row["generation_policy_b"]] += 1
    counters["span_count:" + row["num_intervals_a"] + "|" + row["num_intervals_b"]] += 1
    counters["budget_bin_equal:" + str(row["budget_bin_a"] == row["budget_bin_b"]).lower()] += 1


def audit_shard(shard_index, candidate_info, impact_info, pair_path, thresholds):
    started = time.time()
    failures = FailureCollector()
    counters = Counter()
    sources = load_sources(candidate_info["path"], impact_info["path"])
    source_state = sources["state"]
    source_candidate_keys = source_state["seen_candidate_keys"]
    source_impact_keys = set(sources["impacts"])
    if source_candidate_keys != source_impact_keys:
        failures.add(
            "source_candidate_impact_key_mismatch", None,
            "missing=%d orphan=%d" % (
                len(source_candidate_keys - source_impact_keys),
                len(source_impact_keys - source_candidate_keys),
            ),
        )
    if source_state["source_candidate_rows"] != candidate_info["expected_rows"]:
        failures.add("source_candidate_index_count_mismatch")
    if sources["impact_rows"] != impact_info["expected_rows"]:
        failures.add("source_impact_index_count_mismatch")
    annotated_by_protein = {}
    group_stats_by_protein = {}
    for protein_id, protein_candidates in sources["protein_candidates"].items():
        annotated, group_stats = annotated_candidate_map(protein_candidates)
        annotated_by_protein[protein_id] = annotated
        group_stats_by_protein[protein_id] = group_stats
    pair_ids = set()
    unordered = set()
    protein_pair_counts = Counter()
    distributions = {
        "utility_margin": NumericSummary((0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)),
        "mask_similarity": NumericSummary((0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)),
        "confidence": NumericSummary((0.10, 0.25, 0.50, 0.75, 0.90, 1.0)),
    }
    pair_rows = 0
    with Path(pair_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != builder.PAIR_COLUMNS:
            failures.add("pair_schema_mismatch", 1, reader.fieldnames)
        for row_number, row in enumerate(reader, 2):
            pair_rows += 1
            audit_pair_row(
                row, row_number, shard_index, sources, annotated_by_protein,
                group_stats_by_protein, thresholds, pair_ids, unordered,
                protein_pair_counts, failures, distributions, counters,
            )
    for protein_id in sources["protein_candidates"]:
        protein_pair_counts.setdefault(protein_id, 0)
    over_target = sum(count > builder.TARGET_PAIRS for count in protein_pair_counts.values())
    if over_target:
        failures.add("protein_exceeds_64_pairs", None, over_target)
    pair_count_histogram = Counter(protein_pair_counts.values())
    full_proteins = pair_count_histogram[builder.TARGET_PAIRS]
    source_proteins = len(sources["protein_candidates"])
    report = {
        "schema_version": "pair_validation_shard_report_v1",
        "status": "PASS" if failures.total == 0 else "FAIL",
        "shard_index": shard_index,
        "source_candidate_csv": str(candidate_info["path"]),
        "source_impact_csv": str(impact_info["path"]),
        "pair_csv": str(pair_path),
        "source_candidate_size_bytes": candidate_info["path"].stat().st_size,
        "source_impact_size_bytes": impact_info["path"].stat().st_size,
        "pair_size_bytes": Path(pair_path).stat().st_size,
        "source_candidates": source_state["source_candidate_rows"],
        "source_impacts": sources["impact_rows"],
        "source_proteins": source_proteins,
        "valid_source_candidates": source_state["valid_candidates"],
        "pairs": pair_rows,
        "full_proteins": full_proteins,
        "shortage_proteins": source_proteins - full_proteins,
        "pair_count_histogram": dict(pair_count_histogram),
        "source_reference_count": counters["source_references"],
        "matched_source_candidate_references": counters["matched_source_candidate_references"],
        "matched_source_impact_references": counters["matched_source_impact_references"],
        "same_count_pairs": counters["same_count_pairs"],
        "unequal_count_pairs": counters["unequal_count_pairs"],
        "tie_pairs": counters["tie_pairs"],
        "view_conflicting_pairs": counters["view_conflicting_pairs"],
        "policy_positive_eligible_pairs": counters["policy_positive_eligible_pairs"],
        "category_counts": {key.split(":", 1)[1]: value for key, value in counters.items() if key.startswith("category:")},
        "quality_counts": {key.split(":", 1)[1]: value for key, value in counters.items() if key.startswith("quality:")},
        "preference_counts": {key.split(":", 1)[1]: value for key, value in counters.items() if key.startswith("preference:")},
        "generation_policy_pair_counts": {key.split(":", 1)[1]: value for key, value in counters.items() if key.startswith("generation_policy:")},
        "span_count_pair_counts": {key.split(":", 1)[1]: value for key, value in counters.items() if key.startswith("span_count:")},
        "budget_bin_equal_counts": {key.split(":", 1)[1]: value for key, value in counters.items() if key.startswith("budget_bin_equal:")},
        "distributions": {key: value.payload() for key, value in distributions.items()},
        "source_invalid_candidate_counts": dict(source_state["invalid_candidates"]),
        "source_invalid_impact_counts": dict(sources["invalid_impacts"]),
        "protein_splits": sources["protein_splits"],
        "failures": failures.payload(),
        "runtime_seconds": time.time() - started,
    }
    return report


def validation_report_path(output_dir, shard_index):
    return Path(output_dir) / "validation_shards" / ("shard_%06d.json" % shard_index)


def resume_validation(output_dir, shard_index, candidate_path, impact_path, pair_path):
    path = validation_report_path(output_dir, shard_index)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report["status"] != "PASS":
            return None
        if report["source_candidate_size_bytes"] != candidate_path.stat().st_size:
            return None
        if report["source_impact_size_bytes"] != impact_path.stat().st_size:
            return None
        if report["pair_size_bytes"] != pair_path.stat().st_size:
            return None
        return report
    except Exception:
        return None


def merge_dict_counter(reports, field):
    counter = Counter()
    for report in reports:
        counter.update({str(key): int(value) for key, value in report.get(field, {}).items()})
    return counter


def merge_distributions(reports, name):
    count = 0
    total = 0.0
    minimum = None
    maximum = None
    histogram = Counter()
    for report in reports:
        payload = report["distributions"][name]
        count += payload["count"]
        if payload["count"]:
            total += payload["mean"] * payload["count"]
            minimum = payload["min"] if minimum is None else min(minimum, payload["min"])
            maximum = payload["max"] if maximum is None else max(maximum, payload["max"])
        histogram.update(payload["histogram"])
    return {"count": count, "min": minimum, "max": maximum, "mean": total / count if count else None, "histogram": dict(histogram)}


def write_validation_outputs(args, reports, selected_count, total_available):
    reports = sorted(reports, key=lambda item: item["shard_index"])
    index_rows = []
    for report in reports:
        index_rows.append({
            "shard_index": report["shard_index"],
            "source_candidate_csv": report["source_candidate_csv"],
            "source_impact_csv": report["source_impact_csv"],
            "pair_csv": report["pair_csv"],
            "source_candidates": report["source_candidates"],
            "source_impacts": report["source_impacts"],
            "source_proteins": report["source_proteins"],
            "pairs": report["pairs"],
            "full_proteins": report["full_proteins"],
            "shortage_proteins": report["shortage_proteins"],
            "failure_count": report["failures"]["total"],
            "status": report["status"],
            "runtime_seconds": "%.6f" % report["runtime_seconds"],
        })
    builder.atomic_write_csv(args.output_dir / "indexes/validation_shard_index.csv", VALIDATION_INDEX_COLUMNS, index_rows)
    protein_splits = {}
    leakage = Counter()
    for report in reports:
        for protein_id, split in report["protein_splits"].items():
            if protein_id in protein_splits and protein_splits[protein_id] != split:
                leakage[protein_id] += 1
            protein_splits[protein_id] = split
    pairs = sum(report["pairs"] for report in reports)
    source_references = sum(report["source_reference_count"] for report in reports)
    same_count = sum(report["same_count_pairs"] for report in reports)
    unequal_count = sum(report["unequal_count_pairs"] for report in reports)
    source_candidates = sum(report["source_candidates"] for report in reports)
    source_impacts = sum(report["source_impacts"] for report in reports)
    proteins = sum(report["source_proteins"] for report in reports)
    full = sum(report["full_proteins"] for report in reports)
    cross_protein_failures = sum(report["failures"]["counts"].get("cross_protein_pair", 0) for report in reports)
    failure_count = sum(report["failures"]["total"] for report in reports) + sum(leakage.values())
    all_requested = len(reports) == selected_count
    all_available = len(reports) == total_available
    status = "PASS" if failure_count == 0 and all_requested else "FAIL"
    report = {
        "schema_version": "pair_esmc600m_same_count_validation_v1",
        "status": status,
        "scope": "full" if all_available else "partial",
        "claim_boundary": "Teacher-derived deletion perturbation preference validation; not functional ground truth or proof of functional preservation.",
        "shards_available": total_available,
        "shards_requested": selected_count,
        "shards_validated": len(reports),
        "source_candidate_rows": source_candidates,
        "source_impact_rows": source_impacts,
        "proteins": proteins,
        "pairs": pairs,
        "same_protein_pairs": pairs - cross_protein_failures,
        "same_protein_rate": (pairs - cross_protein_failures) / float(pairs) if pairs else None,
        "same_deleted_length_pairs": same_count,
        "same_deleted_length_rate": same_count / float(pairs) if pairs else None,
        "unequal_count_pairs": unequal_count,
        "source_candidate_linkage": {
            "matched_references": sum(report["matched_source_candidate_references"] for report in reports),
            "total_references": source_references,
            "rate": sum(report["matched_source_candidate_references"] for report in reports) / float(source_references) if source_references else None,
        },
        "source_impact_linkage": {
            "matched_references": sum(report["matched_source_impact_references"] for report in reports),
            "total_references": source_references,
            "rate": sum(report["matched_source_impact_references"] for report in reports) / float(source_references) if source_references else None,
        },
        "self_pairs": sum(report["failures"]["counts"].get("self_pair", 0) for report in reports),
        "duplicate_unordered_pairs": sum(report["failures"]["counts"].get("duplicate_unordered_pair", 0) for report in reports),
        "coordinate_deleted_length_fraction_consistency": "PASS" if not any(
            report["failures"]["counts"].get(key, 0)
            for report in reports
            for key in ("source_field_mismatch", "source_numeric_mismatch", "output_deleted_length_mismatch")
        ) else "FAIL",
        "protein_split_leakage_count": sum(leakage.values()),
        "protein_split_counts": dict(Counter(protein_splits.values())),
        "proteins_with_64_pairs": full,
        "proteins_with_64_pairs_rate": full / float(proteins) if proteins else None,
        "shortage_proteins": proteins - full,
        "pair_count_histogram": dict(merge_dict_counter(reports, "pair_count_histogram")),
        "category_counts": dict(merge_dict_counter(reports, "category_counts")),
        "quality_counts": dict(merge_dict_counter(reports, "quality_counts")),
        "preference_counts": dict(merge_dict_counter(reports, "preference_counts")),
        "generation_policy_pair_counts": dict(merge_dict_counter(reports, "generation_policy_pair_counts")),
        "span_count_pair_counts": dict(merge_dict_counter(reports, "span_count_pair_counts")),
        "budget_bin_equal_counts": dict(merge_dict_counter(reports, "budget_bin_equal_counts")),
        "tie_pairs": sum(report["tie_pairs"] for report in reports),
        "view_conflicting_pairs": sum(report["view_conflicting_pairs"] for report in reports),
        "policy_positive_eligible_pairs": sum(report["policy_positive_eligible_pairs"] for report in reports),
        "utility_margin_distribution": merge_distributions(reports, "utility_margin"),
        "mask_similarity_distribution": merge_distributions(reports, "mask_similarity"),
        "confidence_distribution": merge_distributions(reports, "confidence"),
        "shorter_deletion_rule": {
            "deleted_length_difference_nonzero_pairs": unequal_count,
            "informative_pairs": 0 if unequal_count == 0 else unequal_count,
            "informative_rate": 0.0 if pairs and unequal_count == 0 else (unequal_count / float(pairs) if pairs else None),
            "conclusion": "No label information: both sides have identical deleted_length." if unequal_count == 0 else "FAIL: unequal-count pairs exist.",
        },
        "source_preference_pairs_used": False,
        "old_pair_labels_read": False,
        "teacher_run": False,
        "gpu_used": False,
        "student_training": False,
        "failure_count": failure_count,
        "failed_shards": [report["shard_index"] for report in reports if report["status"] != "PASS"],
        "runtime_seconds_sum": sum(report["runtime_seconds"] for report in reports),
    }
    builder.atomic_write_json(args.output_dir / "validation_report.json", report)
    failures = []
    for shard_report in reports:
        if shard_report["status"] != "PASS":
            failures.append({
                "shard_index": shard_report["shard_index"],
                "stage": "validation",
                "reason": builder.canonical_json(shard_report["failures"]),
            })
    builder.atomic_write_csv(args.output_dir / "failed_shards.csv", ("shard_index", "stage", "reason"), failures)
    return report


def main():
    args = parse_args()
    candidate_rows = builder.parse_index(args.candidate_index, "candidate")
    impact_rows = builder.parse_index(args.impact_index, "impact")
    if set(candidate_rows) != set(impact_rows):
        raise ValueError("candidate and impact indexes have different shard sets")
    available = sorted(candidate_rows)
    selected = builder.parse_shard_selection(args.shards, available)
    with (args.output_dir / "calibration_summary.json").open(encoding="utf-8") as handle:
        calibration = json.load(handle)
    thresholds = calibration["thresholds"]
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    reports = []

    def missing_pair_report(shard_index, pair_path, reason="missing_pair_shard"):
        return {
            "shard_index": shard_index, "status": "FAIL", "pairs": 0,
            "source_candidate_csv": str(candidate_rows[shard_index]["path"]),
            "source_impact_csv": str(impact_rows[shard_index]["path"]),
            "pair_csv": str(pair_path), "source_candidates": 0, "source_impacts": 0,
            "source_proteins": 0, "full_proteins": 0, "shortage_proteins": 0,
            "runtime_seconds": 0.0, "failures": {"counts": {reason: 1}, "examples": {}, "total": 1},
            "protein_splits": {}, "pair_count_histogram": {}, "distributions": {
                name: {"count": 0, "min": None, "max": None, "mean": None, "histogram": {}}
                for name in ("utility_margin", "mask_similarity", "confidence")
            },
            "source_reference_count": 0, "matched_source_candidate_references": 0,
            "matched_source_impact_references": 0, "same_count_pairs": 0,
            "unequal_count_pairs": 0, "tie_pairs": 0, "view_conflicting_pairs": 0,
            "policy_positive_eligible_pairs": 0, "category_counts": {}, "quality_counts": {},
            "preference_counts": {}, "generation_policy_pair_counts": {}, "span_count_pair_counts": {},
            "budget_bin_equal_counts": {},
        }

    def consume_report(shard_index, report):
        reports.append(report)
        aggregate = write_validation_outputs(args, reports, len(selected), len(available))
        print(builder.canonical_json({
            "event": "validation_shard_complete", "ordinal": len(reports),
            "requested": len(selected), "shard_index": shard_index,
            "status": report["status"], "pairs": report["pairs"],
            "failure_count": report["failures"]["total"],
            "validated_so_far": len(reports), "aggregate_status": aggregate["status"],
        }), flush=True)

    futures = {}
    pool = ProcessPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    for shard_index in selected:
        pair_path = builder.output_path(args.output_dir, shard_index)
        if not pair_path.exists():
            consume_report(shard_index, missing_pair_report(shard_index, pair_path))
            continue
        report = None
        if args.resume:
            report = resume_validation(
                args.output_dir, shard_index, candidate_rows[shard_index]["path"],
                impact_rows[shard_index]["path"], pair_path,
            )
        if report is not None:
            consume_report(shard_index, report)
        elif pool is None:
            try:
                report = audit_shard(
                    shard_index, candidate_rows[shard_index], impact_rows[shard_index],
                    pair_path, thresholds,
                )
            except Exception as exc:
                report = missing_pair_report(shard_index, pair_path, "audit_exception:%s" % type(exc).__name__)
                report["failures"]["examples"] = {"exception": [{"detail": str(exc)[:500]}]}
            builder.atomic_write_json(validation_report_path(args.output_dir, shard_index), report)
            consume_report(shard_index, report)
        else:
            future = pool.submit(
                audit_shard, shard_index, candidate_rows[shard_index], impact_rows[shard_index],
                pair_path, thresholds,
            )
            futures[future] = (shard_index, pair_path)
    if pool is not None:
        for future in as_completed(futures):
            shard_index, pair_path = futures[future]
            try:
                report = future.result()
            except Exception as exc:
                report = missing_pair_report(shard_index, pair_path, "audit_exception:%s" % type(exc).__name__)
                report["failures"]["examples"] = {"exception": [{"detail": str(exc)[:500]}]}
            builder.atomic_write_json(validation_report_path(args.output_dir, shard_index), report)
            consume_report(shard_index, report)
        pool.shutdown(wait=True)
    final = write_validation_outputs(args, reports, len(selected), len(available))
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if final["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
