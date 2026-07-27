#!/usr/bin/env python3
"""Build strict same-deleted-count ESMC-600M preference pairs on CPU.

This program consumes only deletion-candidate and frozen teacher-impact shards.
It never reads prior preference labels, runs a teacher, uses a GPU, or trains a
student.  Input is processed one shard at a time and output shards are replaced
atomically only after local validation succeeds.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


DEFAULT_BASE = Path("data/processed/preference_training/delepref/raw_teacher_assets")
DEFAULT_CANDIDATE_INDEX = DEFAULT_BASE / "indexes/deletion_perturbations_shard_index.csv"
DEFAULT_IMPACT_INDEX = DEFAULT_BASE / "indexes/teacher_impact_shard_index.csv"
DEFAULT_OUTPUT = Path("data/processed/preference_training/delepref/pair_same_count")

COORDINATE_SYSTEM = "0_based_closed_inclusive"
GLOBAL_GATE = 0.95
RETAINED_GATE = 0.90
HIGH_GLOBAL = 0.98
HIGH_RETAINED = 0.95
EPSILON = 1.0e-12
MAD_NORMALIZER = 1.4826
IQR_NORMALIZER = 1.349
Z_CLIP = 8.0
UTILITY_WEIGHTS = {"global": 0.35, "retained": 0.45, "junction": 0.20}
DEFAULT_SEED = 7302026
TARGET_PAIRS = 64
QUOTAS = {"local_hard": 24, "position_contrast": 16, "topology_contrast": 16, "tie_near_tie": 8}

CANDIDATE_REQUIRED = (
    "protein_id", "perturbation_id", "plan_id", "wildtype_length",
    "deleted_intervals", "num_intervals", "deleted_length",
    "deletion_fraction", "budget_bin", "generation_policy",
    "coordinate_system", "wildtype_sequence", "mutant_sequence",
)
IMPACT_REQUIRED = (
    "protein_id", "perturbation_id", "plan_id", "teacher_model",
    "teacher_scoring_mode", "global_shift", "retained_region_shift",
    "junction_plausibility", "impact_confidence", "scoring_status",
    "global_similarity", "retained_similarity", "teacher_backend",
)

PAIR_COLUMNS = (
    "protein_id", "source_split", "source_shard_index", "pair_id",
    "selection_category", "selection_mode", "construction_policy",
    "deleted_length", "group_size", "pair_budget_key", "preference",
    "preference_margin", "preference_confidence", "tie_margin",
    "utility_a", "utility_b", "utility_margin_signed", "view_agreement",
    "view_conflicting", "absolute_similarity_score", "mask_similarity",
    "centroid_distance_fraction", "pair_quality_bucket",
    "scorer_training_weight", "policy_positive_eligible",
    "perturbation_id_a", "plan_id_a", "perturbation_id_b", "plan_id_b",
    "deleted_intervals_a", "deleted_intervals_b", "num_intervals_a",
    "num_intervals_b", "deletion_fraction_a", "deletion_fraction_b",
    "budget_bin_a", "budget_bin_b", "generation_policy_a",
    "generation_policy_b", "global_similarity_a", "global_similarity_b",
    "retained_similarity_a", "retained_similarity_b",
    "junction_plausibility_a", "junction_plausibility_b",
    "candidate_quality_a", "candidate_quality_b", "comparison_basis_json",
)

INDEX_COLUMNS = (
    "shard_id", "shard_index", "source_candidate_csv", "source_impact_csv",
    "output_csv", "num_source_candidates", "num_source_impacts",
    "num_proteins", "num_pairs", "num_full_proteins",
    "num_shortage_proteins", "status", "runtime_seconds", "output_size_bytes",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "build", "toy"), default="build")
    parser.add_argument("--candidate-index", type=Path, default=DEFAULT_CANDIDATE_INDEX)
    parser.add_argument("--impact-index", type=Path, default=DEFAULT_IMPACT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shards", default="", help="Comma-separated shard indices; empty means all")
    parser.add_argument("--calibration-shards", default="", help="Default: ten uniformly spaced available shards")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-pairs", type=int, default=TARGET_PAIRS)
    parser.add_argument("--workers", type=int, default=1, help="CPU shard workers; each holds only one local shard")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rebuild owned delepref output shards")
    parser.add_argument("--calibration-reservoir", type=int, default=500000)
    parser.add_argument("--calibration-pairs-per-group", type=int, default=128)
    return parser.parse_args()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.%d" % os.getpid())
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.%d" % os.getpid())
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            tmp.unlink()


def parse_index(path, kind):
    path = Path(path)
    if "preference_pair" in path.name.lower():
        raise ValueError("prior preference-pair indexes are forbidden inputs: %s" % path)
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for ordinal, row in enumerate(reader):
            shard_index = int(row.get("shard_index") or ordinal)
            if shard_index in rows:
                raise ValueError("duplicate %s shard index %d" % (kind, shard_index))
            if kind == "candidate":
                shard_path = row.get("shard_path")
                expected = int(float(row.get("num_candidates") or 0))
            else:
                shard_path = row.get("output_csv")
                expected = int(float(row.get("num_rows") or 0))
            if not shard_path:
                raise ValueError("%s index row %d has no source path" % (kind, ordinal + 2))
            rows[shard_index] = {"path": Path(shard_path), "expected_rows": expected, "index_row": row}
    return rows


def parse_shard_selection(value, available):
    if not value.strip():
        return list(sorted(available))
    selected = []
    for token in value.split(","):
        token = token.strip()
        if token:
            selected.append(int(token))
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError("unknown shard indices: %s" % unknown)
    return sorted(set(selected))


def uniform_shards(available, count=10):
    available = sorted(available)
    if len(available) <= count:
        return available
    chosen = []
    for offset in range(count):
        position = int(round(offset * (len(available) - 1) / float(count - 1)))
        chosen.append(available[position])
    return sorted(set(chosen))


def finite_float(value, field):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s is not numeric" % field)
    if not math.isfinite(number):
        raise ValueError("%s is not finite" % field)
    return number


def integer_value(value, field):
    number = finite_float(value, field)
    integer = int(number)
    if number != integer:
        raise ValueError("%s is not integral" % field)
    return integer


def parse_intervals(value, sequence_length):
    try:
        parsed = json.loads(value)
    except Exception as exc:
        raise ValueError("deleted_intervals is not JSON: %s" % exc)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("deleted_intervals must be a non-empty list")
    intervals = []
    previous_end = -1
    for item in parsed:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("interval must be [start,end]")
        start = integer_value(item[0], "interval_start")
        end = integer_value(item[1], "interval_end")
        if start < 0 or end < start or end >= sequence_length:
            raise ValueError("interval [%d,%d] out of bounds for %d" % (start, end, sequence_length))
        if start <= previous_end:
            raise ValueError("intervals overlap or are not strictly ordered")
        intervals.append((start, end))
        previous_end = end
    return tuple(intervals)


def deleted_mask(intervals):
    mask = set()
    for start, end in intervals:
        mask.update(range(start, end + 1))
    return frozenset(mask)


def apply_deletion(sequence, intervals):
    pieces = []
    cursor = 0
    for start, end in intervals:
        pieces.append(sequence[cursor:start])
        cursor = end + 1
    pieces.append(sequence[cursor:])
    return "".join(pieces)


def source_split(candidate):
    try:
        metadata = json.loads(candidate.get("metadata_json") or "{}")
    except Exception:
        metadata = {}
    return str(metadata.get("source_split") or "train")


def compact_impact(row):
    for field in IMPACT_REQUIRED:
        if not str(row.get(field, "")).strip():
            raise ValueError("missing impact field %s" % field)
    if row["scoring_status"] != "ok":
        raise ValueError("impact scoring_status is not ok")
    values = {
        "global_shift": finite_float(row["global_shift"], "global_shift"),
        "retained_shift": finite_float(row["retained_region_shift"], "retained_region_shift"),
        "junction": finite_float(row["junction_plausibility"], "junction_plausibility"),
        "impact_confidence": finite_float(row["impact_confidence"], "impact_confidence"),
        "global_similarity": finite_float(row["global_similarity"], "global_similarity"),
        "retained_similarity": finite_float(row["retained_similarity"], "retained_similarity"),
    }
    if not 0.0 <= values["impact_confidence"] <= 1.0:
        raise ValueError("impact_confidence outside [0,1]")
    for field in ("global_similarity", "retained_similarity", "junction"):
        if not -1.000001 <= values[field] <= 1.000001:
            raise ValueError("%s outside cosine range" % field)
    if abs(values["global_similarity"] - (1.0 - 2.0 * values["global_shift"])) > 5.0e-5:
        raise ValueError("global shift/similarity mismatch")
    if abs(values["retained_similarity"] - (1.0 - 2.0 * values["retained_shift"])) > 5.0e-5:
        raise ValueError("retained shift/similarity mismatch")
    return values


def load_impacts(path):
    impacts = {}
    invalid = Counter()
    total = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, 2):
            total += 1
            key = (row.get("protein_id", ""), row.get("perturbation_id", ""), row.get("plan_id", ""))
            if key in impacts:
                raise ValueError("duplicate impact key at %s:%d" % (path, row_number))
            try:
                impacts[key] = compact_impact(row)
            except ValueError as exc:
                invalid[str(exc)] += 1
                impacts[key] = {"invalid_reason": str(exc)}
    return impacts, total, invalid


def prepare_candidate(row, impact):
    for field in CANDIDATE_REQUIRED:
        if not str(row.get(field, "")).strip():
            raise ValueError("missing candidate field %s" % field)
    if impact is None:
        raise ValueError("missing linked teacher impact")
    if "invalid_reason" in impact:
        raise ValueError("invalid linked teacher impact: %s" % impact["invalid_reason"])
    length = integer_value(row["wildtype_length"], "wildtype_length")
    if length <= 1:
        raise ValueError("wildtype_length must exceed one")
    wildtype = row["wildtype_sequence"]
    mutant = row["mutant_sequence"]
    if len(wildtype) != length:
        raise ValueError("wildtype_length disagrees with sequence")
    intervals = parse_intervals(row["deleted_intervals"], length)
    count = sum(end - start + 1 for start, end in intervals)
    if count != integer_value(row["deleted_length"], "deleted_length"):
        raise ValueError("deleted_length disagrees with coordinates")
    if len(intervals) != integer_value(row["num_intervals"], "num_intervals"):
        raise ValueError("num_intervals disagrees with coordinates")
    fraction = finite_float(row["deletion_fraction"], "deletion_fraction")
    if abs(fraction - count / float(length)) > 5.0e-8:
        raise ValueError("deletion_fraction disagrees with deleted_length/wildtype_length")
    if row["coordinate_system"] != COORDINATE_SYSTEM:
        raise ValueError("unexpected coordinate_system")
    if apply_deletion(wildtype, intervals) != mutant:
        raise ValueError("mutant_sequence is not the coordinate deletion")
    if impact["global_similarity"] < GLOBAL_GATE:
        raise ValueError("below_global_similarity_gate")
    if impact["retained_similarity"] < RETAINED_GATE:
        raise ValueError("below_retained_similarity_gate")
    quality = "high" if (
        impact["global_similarity"] >= HIGH_GLOBAL and impact["retained_similarity"] >= HIGH_RETAINED
    ) else "acceptable"
    return {
        "protein_id": row["protein_id"],
        "perturbation_id": row["perturbation_id"],
        "plan_id": row["plan_id"],
        "wildtype_length": length,
        "intervals": intervals,
        "intervals_json": canonical_json([list(item) for item in intervals]),
        "mask": deleted_mask(intervals),
        "num_intervals": len(intervals),
        "deleted_length": count,
        "deletion_fraction": fraction,
        "budget_bin": row["budget_bin"],
        "generation_policy": row["generation_policy"],
        "source_split": source_split(row),
        "global_shift": impact["global_shift"],
        "retained_shift": impact["retained_shift"],
        "junction": impact["junction"],
        "impact_confidence": impact["impact_confidence"],
        "global_similarity": impact["global_similarity"],
        "retained_similarity": impact["retained_similarity"],
        "candidate_quality": quality,
    }


def iter_candidate_groups(path, impacts, counters):
    current_protein = None
    current_rows = []
    seen_proteins = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        counters["candidate_schema"] = list(reader.fieldnames or [])
        for row_number, row in enumerate(reader, 2):
            counters["source_candidate_rows"] += 1
            protein_id = row.get("protein_id", "")
            if current_protein is None:
                current_protein = protein_id
            if protein_id != current_protein:
                if protein_id in seen_proteins:
                    raise ValueError("protein %s is not local/contiguous in %s" % (protein_id, path))
                seen_proteins.add(current_protein)
                yield current_protein, current_rows
                current_protein = protein_id
                current_rows = []
            key = (protein_id, row.get("perturbation_id", ""), row.get("plan_id", ""))
            if key in counters["seen_candidate_keys"]:
                raise ValueError("duplicate candidate key at %s:%d" % (path, row_number))
            counters["seen_candidate_keys"].add(key)
            impact = impacts.get(key)
            if impact is not None:
                counters["linked_impact_keys"].add(key)
            try:
                current_rows.append(prepare_candidate(row, impact))
                counters["valid_candidates"] += 1
            except ValueError as exc:
                counters["invalid_candidates"][str(exc)] += 1
        if current_protein is not None:
            if current_protein in seen_proteins:
                raise ValueError("protein %s is not local/contiguous in %s" % (current_protein, path))
            yield current_protein, current_rows


def percentile(values, probability):
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def robust_center_scale(values):
    values = [finite_float(value, "robust_stat_value") for value in values]
    if not values:
        raise ValueError("empty robust-stat group")
    center = statistics.median(values)
    if len(values) == 1:
        return center, 1.0, "singleton_unit_scale"
    mad = statistics.median([abs(value - center) for value in values])
    scale = MAD_NORMALIZER * mad
    if scale > EPSILON:
        return center, scale, "median_mad"
    if len(values) >= 4:
        q1 = percentile(values, 0.25)
        q3 = percentile(values, 0.75)
        scale = (q3 - q1) / IQR_NORMALIZER
        if scale > EPSILON:
            return center, scale, "median_iqr_fallback"
    scale = statistics.pstdev(values)
    if scale > EPSILON:
        return center, scale, "median_population_std_fallback"
    return center, 1.0, "constant_unit_scale"


def clipped_z(value, center, scale):
    return max(-Z_CLIP, min(Z_CLIP, (value - center) / max(scale, EPSILON)))


def teacher_utility(candidate):
    """Return the fixed absolute teacher utility, never a group-normalized sum.

    Robust within-group statistics may calibrate ties, confidence, weights, and
    sampling, but must not change the ordering encoded by this utility.
    """

    return (
        UTILITY_WEIGHTS["global"] * finite_float(candidate["global_similarity"], "global_similarity")
        + UTILITY_WEIGHTS["retained"] * finite_float(candidate["retained_similarity"], "retained_similarity")
        + UTILITY_WEIGHTS["junction"] * finite_float(candidate["junction"], "junction")
    )


def annotate_group(group):
    views = {
        "global": [candidate["global_similarity"] for candidate in group],
        "retained": [candidate["retained_similarity"] for candidate in group],
        "junction": [candidate["junction"] for candidate in group],
    }
    stats = {}
    for view_name, values in views.items():
        center, scale, fallback = robust_center_scale(values)
        stats[view_name] = {"center": center, "scale": scale, "fallback": fallback}
    for index, candidate in enumerate(group):
        z = {}
        for view_name in ("global", "retained", "junction"):
            z[view_name] = clipped_z(views[view_name][index], stats[view_name]["center"], stats[view_name]["scale"])
        candidate["view_z"] = z
        candidate["utility"] = teacher_utility(candidate)
    utility_center, utility_scale, utility_fallback = robust_center_scale([item["utility"] for item in group])
    tie_margin = max(EPSILON, 0.15 * utility_scale)
    stats["utility"] = {
        "center": utility_center,
        "scale": utility_scale,
        "fallback": utility_fallback,
        "tie_margin": tie_margin,
    }
    return stats


def mask_similarity(a, b):
    union = a["mask"] | b["mask"]
    return len(a["mask"] & b["mask"]) / float(len(union)) if union else 1.0


def centroid_distance_fraction(a, b):
    centroid_a = sum(a["mask"]) / float(len(a["mask"]))
    centroid_b = sum(b["mask"]) / float(len(b["mask"]))
    return abs(centroid_a - centroid_b) / float(a["wildtype_length"])


def absolute_candidate_score(candidate):
    global_score = min(1.0, max(0.0, (candidate["global_similarity"] - GLOBAL_GATE) / (1.0 - GLOBAL_GATE)))
    retained_score = min(1.0, max(0.0, (candidate["retained_similarity"] - RETAINED_GATE) / (1.0 - RETAINED_GATE)))
    junction_score = min(1.0, max(0.0, (candidate["junction"] + 1.0) / 2.0))
    return 0.35 * global_score + 0.45 * retained_score + 0.20 * junction_score


def category_for_pair(tie, spans_differ, similarity, thresholds):
    if tie:
        return "tie_near_tie"
    if spans_differ:
        return "topology_contrast"
    if similarity >= thresholds["local_high_mask_similarity"]:
        return "local_hard"
    if similarity <= thresholds["position_low_mask_similarity"]:
        return "position_contrast"
    return "same_count_refill"


def make_pair(protein_id, deleted_length, group_size, a, b, stats, thresholds, shard_index):
    if a["perturbation_id"] == b["perturbation_id"]:
        return None, "self_pair"
    if a["candidate_quality"] == "acceptable" and b["candidate_quality"] == "acceptable":
        return None, "both_low_absolute_quality"
    if a["mask"] == b["mask"]:
        return None, "identical_deletion_mask"
    if b["perturbation_id"] < a["perturbation_id"]:
        a, b = b, a
    delta = a["utility"] - b["utility"]
    tie_margin = stats["utility"]["tie_margin"]
    tie = abs(delta) <= tie_margin
    preference = "tie" if tie else ("a_better" if delta > 0 else "b_better")
    view_deltas = [a["view_z"][name] - b["view_z"][name] for name in ("global", "retained", "junction")]
    signs = [1 if value > EPSILON else (-1 if value < -EPSILON else 0) for value in view_deltas]
    positive = sum(sign > 0 for sign in signs)
    negative = sum(sign < 0 for sign in signs)
    active = positive + negative
    conflicting = positive > 0 and negative > 0
    if active == 0:
        agreement = 1.0
    elif tie:
        agreement = max(positive, negative) / float(active)
    else:
        preferred_sign = 1 if delta > 0 else -1
        agreement = sum(sign == preferred_sign for sign in signs) / float(active)
    similarity = mask_similarity(a, b)
    centroid_distance = centroid_distance_fraction(a, b)
    spans_differ = a["num_intervals"] != b["num_intervals"]
    category = category_for_pair(tie, spans_differ, similarity, thresholds)
    absolute_score = 0.5 * (absolute_candidate_score(a) + absolute_candidate_score(b))
    if tie:
        margin_score = max(0.0, 1.0 - abs(delta) / max(tie_margin, EPSILON))
    else:
        margin_score = min(1.0, abs(delta) / max(2.0 * tie_margin, EPSILON))
    if category == "local_hard":
        relation_score = similarity
    elif category == "position_contrast":
        relation_score = 1.0 - similarity
    elif category == "topology_contrast":
        relation_score = 1.0
    else:
        relation_score = 0.5
    confidence = max(0.0, min(1.0, 0.40 * margin_score + 0.25 * agreement + 0.25 * absolute_score + 0.10 * relation_score))
    if conflicting:
        quality = "diagnostic_view_conflict"
        weight = 0.05
    elif tie:
        quality = "tie_or_near_tie"
        weight = 0.25
    elif confidence >= 0.75 and a["candidate_quality"] == b["candidate_quality"] == "high":
        quality = "high"
        weight = 1.0
    elif confidence >= 0.50:
        quality = "medium"
        weight = 0.70
    else:
        quality = "low"
        weight = 0.25
    preferred = None if tie else (a if delta > 0 else b)
    policy_eligible = bool(
        preferred is not None and not conflicting and quality in ("high", "medium")
        and preferred["candidate_quality"] == "high" and confidence >= 0.50
    )
    pair_key = stable_hash(protein_id, a["perturbation_id"], b["perturbation_id"], DEFAULT_SEED)[:24]
    comparison = {
        "formula_version": "pair_absolute_teacher_utility",
        "teacher_preference_is_ground_truth": False,
        "source_preference_pairs_used": False,
        "group_scope": [protein_id, deleted_length],
        "teacher_utility": "0.35*global_similarity+0.45*retained_similarity+0.20*junction_plausibility",
        "raw_view_deltas": {
            "global": a["global_similarity"] - b["global_similarity"],
            "retained": a["retained_similarity"] - b["retained_similarity"],
            "junction": a["junction"] - b["junction"],
        },
        "view_z_deltas": dict(zip(("global", "retained", "junction"), view_deltas)),
        "robust_statistics_role": "tie_margin_confidence_weight_and_pair_sampling_only",
        "scale_fallbacks": {name: stats[name]["fallback"] for name in ("global", "retained", "junction", "utility")},
    }
    return {
        "protein_id": protein_id,
        "source_split": a["source_split"],
        "source_shard_index": shard_index,
        "pair_id": "%s__pairdelepref__%s" % (protein_id, pair_key),
        "selection_category": category,
        "selection_mode": "",
        "construction_policy": "strict_same_deleted_length_pair",
        "deleted_length": deleted_length,
        "group_size": group_size,
        "pair_budget_key": "deleted_length:%d" % deleted_length,
        "preference": preference,
        "preference_margin": "%.10g" % abs(delta),
        "preference_confidence": "%.10g" % confidence,
        "tie_margin": "%.10g" % tie_margin,
        "utility_a": "%.10g" % a["utility"],
        "utility_b": "%.10g" % b["utility"],
        "utility_margin_signed": "%.10g" % delta,
        "view_agreement": "%.10g" % agreement,
        "view_conflicting": str(conflicting).lower(),
        "absolute_similarity_score": "%.10g" % absolute_score,
        "mask_similarity": "%.10g" % similarity,
        "centroid_distance_fraction": "%.10g" % centroid_distance,
        "pair_quality_bucket": quality,
        "scorer_training_weight": "%.10g" % weight,
        "policy_positive_eligible": str(policy_eligible).lower(),
        "perturbation_id_a": a["perturbation_id"],
        "plan_id_a": a["plan_id"],
        "perturbation_id_b": b["perturbation_id"],
        "plan_id_b": b["plan_id"],
        "deleted_intervals_a": a["intervals_json"],
        "deleted_intervals_b": b["intervals_json"],
        "num_intervals_a": a["num_intervals"],
        "num_intervals_b": b["num_intervals"],
        "deletion_fraction_a": "%.10g" % a["deletion_fraction"],
        "deletion_fraction_b": "%.10g" % b["deletion_fraction"],
        "budget_bin_a": a["budget_bin"],
        "budget_bin_b": b["budget_bin"],
        "generation_policy_a": a["generation_policy"],
        "generation_policy_b": b["generation_policy"],
        "global_similarity_a": "%.10g" % a["global_similarity"],
        "global_similarity_b": "%.10g" % b["global_similarity"],
        "retained_similarity_a": "%.10g" % a["retained_similarity"],
        "retained_similarity_b": "%.10g" % b["retained_similarity"],
        "junction_plausibility_a": "%.10g" % a["junction"],
        "junction_plausibility_b": "%.10g" % b["junction"],
        "candidate_quality_a": a["candidate_quality"],
        "candidate_quality_b": b["candidate_quality"],
        "comparison_basis_json": canonical_json(comparison),
        "_rank": (0 if policy_eligible else 1, confidence, stable_hash(DEFAULT_SEED, protein_id, a["perturbation_id"], b["perturbation_id"])),
    }, None


def unique_masks(group, counters):
    kept = {}
    for candidate in sorted(group, key=lambda item: item["perturbation_id"]):
        key = candidate["mask"]
        if key in kept:
            counters["duplicate_masks_removed"] += 1
        else:
            kept[key] = candidate
    return list(kept.values())


def candidate_pair_pools(protein_id, candidates, thresholds, shard_index, counters):
    by_length = defaultdict(list)
    for candidate in candidates:
        by_length[candidate["deleted_length"]].append(candidate)
    pools = defaultdict(lambda: defaultdict(list))
    for deleted_length in sorted(by_length):
        group = unique_masks(by_length[deleted_length], counters)
        if len(group) < 2:
            counters["groups_with_fewer_than_two"] += 1
            continue
        stats = annotate_group(group)
        counters["scale_fallbacks"].update(stats[name]["fallback"] for name in stats)
        for a, b in itertools.combinations(group, 2):
            pair, excluded = make_pair(protein_id, deleted_length, len(group), a, b, stats, thresholds, shard_index)
            if excluded:
                counters["excluded_pairs"][excluded] += 1
                continue
            pools[pair["selection_category"]][deleted_length].append(pair)
    for category in pools:
        for deleted_length in pools[category]:
            if category == "local_hard":
                pools[category][deleted_length].sort(key=lambda item: (item["_rank"][0], abs(float(item["utility_margin_signed"])), item["_rank"][2]))
            elif category == "tie_near_tie":
                pools[category][deleted_length].sort(key=lambda item: (item["_rank"][0], float(item["preference_margin"]), item["_rank"][2]))
            else:
                pools[category][deleted_length].sort(key=lambda item: (item["_rank"][0], -item["_rank"][1], item["_rank"][2]))
    return pools


def round_robin_take(grouped, limit, protein_id, seed, already_selected):
    lengths = sorted(grouped)
    if not lengths or limit <= 0:
        return []
    rotation = int(stable_hash(seed, protein_id, "length_rotation")[:12], 16) % len(lengths)
    lengths = lengths[rotation:] + lengths[:rotation]
    positions = {length: 0 for length in lengths}
    selected = []
    while len(selected) < limit:
        progress = False
        for length in lengths:
            pool = grouped[length]
            while positions[length] < len(pool):
                pair = pool[positions[length]]
                positions[length] += 1
                if pair["pair_id"] not in already_selected:
                    already_selected.add(pair["pair_id"])
                    selected.append(pair)
                    progress = True
                    break
            if len(selected) >= limit:
                break
        if not progress:
            break
    return selected


def select_pairs_for_protein(protein_id, candidates, thresholds, shard_index, seed, target_pairs, counters):
    pools = candidate_pair_pools(protein_id, candidates, thresholds, shard_index, counters)
    selected = []
    selected_ids = set()
    for category in ("local_hard", "position_contrast", "topology_contrast", "tie_near_tie"):
        quota = min(QUOTAS[category], max(0, target_pairs - len(selected)))
        chosen = round_robin_take(pools.get(category, {}), quota, protein_id, seed, selected_ids)
        for pair in chosen:
            pair["selection_mode"] = "quota"
        selected.extend(chosen)
        counters["quota_selected"][category] += len(chosen)
        counters["quota_shortfall"][category] += QUOTAS[category] - len(chosen)
    remaining = max(0, target_pairs - len(selected))
    if remaining:
        refill = defaultdict(list)
        for category, by_length in pools.items():
            for deleted_length, rows in by_length.items():
                refill[deleted_length].extend(rows)
        for deleted_length in refill:
            refill[deleted_length].sort(key=lambda item: (item["_rank"][0], -item["_rank"][1], item["_rank"][2]))
        chosen = round_robin_take(refill, remaining, protein_id, seed + 1, selected_ids)
        for pair in chosen:
            pair["selection_mode"] = "same_deleted_length_refill"
        selected.extend(chosen)
        counters["same_count_refill_selected"] += len(chosen)
    selected.sort(key=lambda item: (item["deleted_length"], item["selection_category"], item["pair_id"]))
    for pair in selected:
        pair.pop("_rank", None)
    return selected


class Reservoir(object):
    def __init__(self, capacity, seed):
        self.capacity = int(capacity)
        self.random = random.Random(seed)
        self.values = []
        self.seen = 0

    def add(self, value):
        self.seen += 1
        if len(self.values) < self.capacity:
            self.values.append(value)
        else:
            replacement = self.random.randrange(self.seen)
            if replacement < self.capacity:
                self.values[replacement] = value


def calibration_for_shards(candidate_rows, impact_rows, shard_indices, seed, reservoir_size, pairs_per_group):
    all_values = Reservoir(reservoir_size, seed)
    positive_values = Reservoir(reservoir_size, seed + 1)
    counters = Counter()
    invalid = Counter()
    for shard_index in shard_indices:
        impacts, impact_count, invalid_impacts = load_impacts(impact_rows[shard_index]["path"])
        counters["impact_rows"] += impact_count
        invalid.update(invalid_impacts)
        state = {
            "source_candidate_rows": 0, "valid_candidates": 0,
            "invalid_candidates": Counter(), "seen_candidate_keys": set(),
            "linked_impact_keys": set(), "candidate_schema": [],
        }
        for protein_id, candidates in iter_candidate_groups(candidate_rows[shard_index]["path"], impacts, state):
            counters["proteins"] += 1
            by_length = defaultdict(list)
            local_counter = Counter()
            for candidate in candidates:
                by_length[candidate["deleted_length"]].append(candidate)
            for deleted_length, group in by_length.items():
                group = unique_masks(group, local_counter)
                combinations = []
                for a, b in itertools.combinations(group, 2):
                    if a["candidate_quality"] == b["candidate_quality"] == "acceptable":
                        continue
                    combinations.append((stable_hash(seed, protein_id, deleted_length, a["perturbation_id"], b["perturbation_id"]), a, b))
                combinations.sort(key=lambda item: item[0])
                for _, a, b in combinations[:pairs_per_group]:
                    value = mask_similarity(a, b)
                    all_values.add(value)
                    if value > 0.0:
                        positive_values.add(value)
        counters["candidate_rows"] += state["source_candidate_rows"]
        counters["valid_candidates"] += state["valid_candidates"]
        invalid.update(state["invalid_candidates"])
        if state["seen_candidate_keys"] != set(impacts):
            raise ValueError("calibration shard %d candidate/impact key sets differ" % shard_index)
    if not all_values.values:
        raise ValueError("calibration found no eligible exact-count candidate pairs")
    position_threshold = percentile(all_values.values, 0.25)
    if positive_values.values:
        local_threshold = percentile(positive_values.values, 0.75)
        local_fallback = "none"
    else:
        local_threshold = 1.0
        local_fallback = "no_positive_overlap_pairs_identity_only"
    return {
        "schema_version": "pair_mask_similarity_calibration_v1",
        "seed": seed,
        "representative_shards": shard_indices,
        "sampling": {
            "pairs_per_exact_count_group": pairs_per_group,
            "reservoir_capacity": reservoir_size,
            "all_pairs_seen": all_values.seen,
            "positive_overlap_pairs_seen": positive_values.seen,
            "all_pairs_sampled": len(all_values.values),
            "positive_overlap_pairs_sampled": len(positive_values.values),
        },
        "quantiles": {
            "all_mask_similarity_q25": position_threshold,
            "all_mask_similarity_q50": percentile(all_values.values, 0.50),
            "all_mask_similarity_q75": percentile(all_values.values, 0.75),
            "positive_mask_similarity_q75": local_threshold,
        },
        "thresholds": {
            "position_low_mask_similarity": position_threshold,
            "local_high_mask_similarity": local_threshold,
        },
        "local_threshold_fallback": local_fallback,
        "source_counts": dict(counters),
        "invalid_candidate_or_impact_counts": dict(invalid),
        "source_preference_pairs_used": False,
    }


def calibration_path(output_dir):
    return Path(output_dir) / "calibration_summary.json"


def ensure_calibration(args, candidate_rows, impact_rows, available):
    path = calibration_path(args.output_dir)
    if path.exists() and not args.force:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("seed") != args.seed:
            raise ValueError("existing calibration seed differs; use --force within owned delepref output")
        return payload
    if args.calibration_shards.strip():
        shards = parse_shard_selection(args.calibration_shards, available)
    else:
        shards = uniform_shards(available, 10)
    payload = calibration_for_shards(
        candidate_rows, impact_rows, shards, args.seed,
        args.calibration_reservoir, args.calibration_pairs_per_group,
    )
    payload["candidate_index"] = str(args.candidate_index)
    payload["impact_index"] = str(args.impact_index)
    payload["candidate_index_sha256"] = sha256_file(args.candidate_index)
    payload["impact_index_sha256"] = sha256_file(args.impact_index)
    atomic_write_json(path, payload)
    return payload


def new_shard_counters():
    return {
        "source_candidate_rows": 0,
        "valid_candidates": 0,
        "invalid_candidates": Counter(),
        "seen_candidate_keys": set(),
        "linked_impact_keys": set(),
        "candidate_schema": [],
        "duplicate_masks_removed": 0,
        "groups_with_fewer_than_two": 0,
        "scale_fallbacks": Counter(),
        "excluded_pairs": Counter(),
        "quota_selected": Counter(),
        "quota_shortfall": Counter(),
        "same_count_refill_selected": 0,
    }


def validate_output_shard(path, expected_rows=None):
    pair_ids = set()
    unordered = set()
    counts = Counter()
    categories = Counter()
    qualities = Counter()
    preferences = Counter()
    rows = 0
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PAIR_COLUMNS:
            raise ValueError("unexpected pair-delepref output schema")
        for row_number, row in enumerate(reader, 2):
            rows += 1
            pair_id = row["pair_id"]
            if pair_id in pair_ids:
                raise ValueError("duplicate pair_id at row %d" % row_number)
            pair_ids.add(pair_id)
            if row["perturbation_id_a"] == row["perturbation_id_b"]:
                raise ValueError("self-pair at row %d" % row_number)
            key = (row["protein_id"],) + tuple(sorted((row["perturbation_id_a"], row["perturbation_id_b"])))
            if key in unordered:
                raise ValueError("duplicate unordered pair at row %d" % row_number)
            unordered.add(key)
            deleted_length = integer_value(row["deleted_length"], "deleted_length")
            if deleted_length <= 0 or row["pair_budget_key"] != "deleted_length:%d" % deleted_length:
                raise ValueError("invalid exact-count budget key at row %d" % row_number)
            if row["construction_policy"] != "strict_same_deleted_length_pair":
                raise ValueError("unexpected construction policy at row %d" % row_number)
            if row["view_conflicting"] == "true" and row["policy_positive_eligible"] != "false":
                raise ValueError("view-conflicting pair is policy eligible at row %d" % row_number)
            if row["candidate_quality_a"] == row["candidate_quality_b"] == "acceptable":
                raise ValueError("both-low pair survived at row %d" % row_number)
            counts[row["protein_id"]] += 1
            categories[row["selection_category"]] += 1
            qualities[row["pair_quality_bucket"]] += 1
            preferences[row["preference"]] += 1
    if expected_rows is not None and rows != expected_rows:
        raise ValueError("output row count %d != expected %d" % (rows, expected_rows))
    if counts and max(counts.values()) > TARGET_PAIRS:
        raise ValueError("a protein exceeds the 64-pair target")
    return {
        "rows": rows,
        "protein_pair_counts": counts,
        "category_counts": categories,
        "quality_counts": qualities,
        "preference_counts": preferences,
    }


def report_path(output_dir, shard_index):
    return Path(output_dir) / "shard_reports" / ("shard_%06d.json" % shard_index)


def output_path(output_dir, shard_index):
    return Path(output_dir) / "preference_pairs" / ("teacher_preference_pairs_shard_%06d.csv" % shard_index)


def resume_report_is_valid(output_dir, shard_index, candidate_path, impact_path):
    report_file = report_path(output_dir, shard_index)
    pair_file = output_path(output_dir, shard_index)
    if not report_file.exists() or not pair_file.exists():
        return None
    try:
        with report_file.open(encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("status") not in ("written", "resumed_validated"):
            return None
        if report.get("source_candidate_size_bytes") != candidate_path.stat().st_size:
            return None
        if report.get("source_impact_size_bytes") != impact_path.stat().st_size:
            return None
        validate_output_shard(pair_file, int(report["num_pairs"]))
        report["status"] = "resumed_validated"
        atomic_write_json(report_file, report)
        return report
    except Exception:
        return None


def process_shard(shard_index, candidate_info, impact_info, output_dir, thresholds, seed, target_pairs, resume, force):
    candidate_path = candidate_info["path"]
    impact_path = impact_info["path"]
    if not candidate_path.exists() or not impact_path.exists():
        raise FileNotFoundError("missing source shard %d" % shard_index)
    if resume and not force:
        report = resume_report_is_valid(output_dir, shard_index, candidate_path, impact_path)
        if report:
            return report
    started = time.time()
    impacts, impact_count, invalid_impacts = load_impacts(impact_path)
    counters = new_shard_counters()
    counters["invalid_candidates"].update({"invalid impact: " + key: value for key, value in invalid_impacts.items()})
    pair_rows = []
    pair_count_histogram = Counter()
    category_counts = Counter()
    quality_counts = Counter()
    preference_counts = Counter()
    generation_policy_counts = Counter()
    span_count_counts = Counter()
    num_proteins = 0
    full_proteins = 0
    shortage_proteins = 0
    for protein_id, candidates in iter_candidate_groups(candidate_path, impacts, counters):
        num_proteins += 1
        selected = select_pairs_for_protein(
            protein_id, candidates, thresholds, shard_index, seed, target_pairs, counters,
        )
        pair_count_histogram[len(selected)] += 1
        if len(selected) == target_pairs:
            full_proteins += 1
        else:
            shortage_proteins += 1
        for pair in selected:
            category_counts[pair["selection_category"]] += 1
            quality_counts[pair["pair_quality_bucket"]] += 1
            preference_counts[pair["preference"]] += 1
            generation_policy_counts[pair["generation_policy_a"] + "|" + pair["generation_policy_b"]] += 1
            span_count_counts[str(pair["num_intervals_a"]) + "|" + str(pair["num_intervals_b"])] += 1
        pair_rows.extend(selected)
    candidate_keys = counters["seen_candidate_keys"]
    impact_keys = set(impacts)
    if candidate_keys != impact_keys:
        missing_impacts = len(candidate_keys - impact_keys)
        orphan_impacts = len(impact_keys - candidate_keys)
        raise ValueError("source linkage mismatch: missing impacts=%d orphan impacts=%d" % (missing_impacts, orphan_impacts))
    if counters["source_candidate_rows"] != candidate_info["expected_rows"]:
        raise ValueError("candidate source rows disagree with index")
    if impact_count != impact_info["expected_rows"]:
        raise ValueError("impact source rows disagree with index")
    pair_file = output_path(output_dir, shard_index)
    pair_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = pair_file.with_name(pair_file.name + ".building.%d" % os.getpid())
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PAIR_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in pair_rows:
                writer.writerow({column: row.get(column, "") for column in PAIR_COLUMNS})
            handle.flush()
            os.fsync(handle.fileno())
        validate_output_shard(tmp, len(pair_rows))
        os.replace(str(tmp), str(pair_file))
    finally:
        if tmp.exists():
            tmp.unlink()
    report = {
        "schema_version": "pair_build_shard_report_v1",
        "status": "written",
        "shard_index": shard_index,
        "source_candidate_csv": str(candidate_path),
        "source_impact_csv": str(impact_path),
        "output_csv": str(pair_file),
        "source_candidate_size_bytes": candidate_path.stat().st_size,
        "source_impact_size_bytes": impact_path.stat().st_size,
        "num_source_candidates": counters["source_candidate_rows"],
        "num_source_impacts": impact_count,
        "num_linked_source_keys": len(counters["linked_impact_keys"]),
        "num_proteins": num_proteins,
        "num_valid_candidates": counters["valid_candidates"],
        "num_pairs": len(pair_rows),
        "num_full_proteins": full_proteins,
        "num_shortage_proteins": shortage_proteins,
        "pair_count_histogram": dict(pair_count_histogram),
        "category_counts": dict(category_counts),
        "quality_counts": dict(quality_counts),
        "preference_counts": dict(preference_counts),
        "generation_policy_pair_counts": dict(generation_policy_counts),
        "span_count_pair_counts": dict(span_count_counts),
        "invalid_candidate_counts": dict(counters["invalid_candidates"]),
        "excluded_pair_counts": dict(counters["excluded_pairs"]),
        "duplicate_masks_removed": counters["duplicate_masks_removed"],
        "groups_with_fewer_than_two": counters["groups_with_fewer_than_two"],
        "scale_fallback_counts": dict(counters["scale_fallbacks"]),
        "quota_selected": dict(counters["quota_selected"]),
        "quota_shortfall": dict(counters["quota_shortfall"]),
        "same_count_refill_selected": counters["same_count_refill_selected"],
        "runtime_seconds": time.time() - started,
        "output_size_bytes": pair_file.stat().st_size,
        "source_preference_pairs_used": False,
        "gpu_used": False,
        "teacher_run": False,
        "student_training": False,
    }
    atomic_write_json(report_path(output_dir, shard_index), report)
    return report


def merge_counter_from_reports(reports, field):
    counter = Counter()
    for report in reports:
        counter.update({str(key): int(value) for key, value in report.get(field, {}).items()})
    return counter


def write_indexes_and_summary(args, reports, calibration, total_available):
    reports = sorted(reports, key=lambda item: item["shard_index"])
    index_rows = []
    for report in reports:
        index_rows.append({
            "shard_id": "pair_%06d" % report["shard_index"],
            "shard_index": report["shard_index"],
            "source_candidate_csv": report["source_candidate_csv"],
            "source_impact_csv": report["source_impact_csv"],
            "output_csv": report["output_csv"],
            "num_source_candidates": report["num_source_candidates"],
            "num_source_impacts": report["num_source_impacts"],
            "num_proteins": report["num_proteins"],
            "num_pairs": report["num_pairs"],
            "num_full_proteins": report["num_full_proteins"],
            "num_shortage_proteins": report["num_shortage_proteins"],
            "status": report["status"],
            "runtime_seconds": "%.6f" % report["runtime_seconds"],
            "output_size_bytes": report["output_size_bytes"],
        })
    atomic_write_csv(Path(args.output_dir) / "indexes/preference_pair_shard_index.csv", INDEX_COLUMNS, index_rows)
    total_proteins = sum(item["num_proteins"] for item in reports)
    full_proteins = sum(item["num_full_proteins"] for item in reports)
    total_pairs = sum(item["num_pairs"] for item in reports)
    summary = {
        "schema_version": "delepref/pair_same_count_summary_v1",
        "status": "complete" if len(reports) == total_available else "partial",
        "identity": "strict same-deleted-count teacher-derived deletion perturbation preference dataset",
        "claim_boundary": "Ranks deletion plans by teacher-derived deletion perturbation preference; not functional ground truth or proof of functional preservation.",
        "source_candidate_index": str(args.candidate_index),
        "source_impact_index": str(args.impact_index),
        "source_candidate_index_sha256": sha256_file(args.candidate_index),
        "source_impact_index_sha256": sha256_file(args.impact_index),
        "source_preference_pairs_used": False,
        "old_pair_labels_read": False,
        "teacher_run": False,
        "gpu_used": False,
        "student_training": False,
        "streaming_scope": "one 64k-candidate/64k-impact source shard at a time",
        "resume": True,
        "atomic_output": True,
        "per_shard_validation": True,
        "seed": args.seed,
        "target_pairs_per_protein": args.target_pairs,
        "quota_targets": QUOTAS,
        "hard_gates": {"global_similarity_min": GLOBAL_GATE, "retained_similarity_min": RETAINED_GATE},
        "absolute_quality": {
            "high": "global_similarity >= 0.98 AND retained_similarity >= 0.95",
            "acceptable": "passes hard gates but is not high",
            "both_acceptable_pair": "excluded as both-low absolute quality",
        },
        "grouping": ["protein_id", "deleted_length"],
        "budget_matching": "exact numeric deleted_length only; budget_bin is never used for matching",
        "generation_policy_role": "descriptive provenance only and separate from numeric budget",
        "utility_formula": {
            "views": {"global": "global_similarity", "retained": "retained_similarity", "junction": "junction_plausibility"},
            "formula": "0.35*global_similarity + 0.45*retained_similarity + 0.20*junction_plausibility",
            "weights": UTILITY_WEIGHTS,
            "ordering_scope": "absolute utility; no per-view normalization before weighting",
            "robust_statistics_scope": "within each (protein_id, deleted_length) group for tie margin, confidence/weight, and pair sampling only",
            "robust_center": "median",
            "robust_primary_scale": "1.4826 * median absolute deviation",
            "robust_fallbacks": ["IQR / 1.349 when n>=4", "population standard deviation", "unit scale for singleton/constant group"],
            "epsilon": EPSILON,
            "confidence_z_clip": [-Z_CLIP, Z_CLIP],
            "tie_margin": "max(epsilon, 0.15 * robust absolute-utility scale) within the same exact-count group",
        },
        "pair_quality_formula": {
            "confidence": "0.40*teacher_margin + 0.25*view_agreement + 0.25*absolute_similarity + 0.10*mask_or_topology_relation",
            "view_conflict": "opposite non-neutral teacher-view directions; diagnostic weight 0.05 and never policy-positive eligible",
            "tie_weight": 0.25,
            "both_low": "excluded",
        },
        "mask_similarity_calibration": calibration,
        "shards_available": total_available,
        "shards_completed": len(reports),
        "source_candidate_rows": sum(item["num_source_candidates"] for item in reports),
        "source_impact_rows": sum(item["num_source_impacts"] for item in reports),
        "proteins": total_proteins,
        "pairs": total_pairs,
        "proteins_with_64_pairs": full_proteins,
        "proteins_with_64_pairs_rate": full_proteins / float(total_proteins) if total_proteins else 0.0,
        "shortage_proteins": total_proteins - full_proteins,
        "pair_count_histogram": dict(merge_counter_from_reports(reports, "pair_count_histogram")),
        "category_counts": dict(merge_counter_from_reports(reports, "category_counts")),
        "quality_counts": dict(merge_counter_from_reports(reports, "quality_counts")),
        "preference_counts": dict(merge_counter_from_reports(reports, "preference_counts")),
        "generation_policy_pair_counts": dict(merge_counter_from_reports(reports, "generation_policy_pair_counts")),
        "span_count_pair_counts": dict(merge_counter_from_reports(reports, "span_count_pair_counts")),
        "invalid_candidate_counts": dict(merge_counter_from_reports(reports, "invalid_candidate_counts")),
        "excluded_pair_counts": dict(merge_counter_from_reports(reports, "excluded_pair_counts")),
        "scale_fallback_counts": dict(merge_counter_from_reports(reports, "scale_fallback_counts")),
        "quota_selected": dict(merge_counter_from_reports(reports, "quota_selected")),
        "quota_shortfall": dict(merge_counter_from_reports(reports, "quota_shortfall")),
        "same_count_refill_selected": sum(item.get("same_count_refill_selected", 0) for item in reports),
        "runtime_seconds_sum": sum(item["runtime_seconds"] for item in reports),
    }
    atomic_write_json(Path(args.output_dir) / "pair_summary.json", summary)
    failed_path = Path(args.output_dir) / "failed_shards.csv"
    if not failed_path.exists():
        atomic_write_csv(failed_path, ("shard_index", "stage", "reason"), [])
    return summary


def load_all_reports(output_dir):
    reports = []
    directory = Path(output_dir) / "shard_reports"
    if directory.exists():
        for path in sorted(directory.glob("shard_*.json")):
            with path.open(encoding="utf-8") as handle:
                report = json.load(handle)
            if report.get("status") in ("written", "resumed_validated"):
                reports.append(report)
    return reports


def append_failure(output_dir, shard_index, stage, reason):
    path = Path(output_dir) / "failed_shards.csv"
    existing = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    existing = [row for row in existing if int(row["shard_index"]) != shard_index]
    existing.append({"shard_index": shard_index, "stage": stage, "reason": reason})
    atomic_write_csv(path, ("shard_index", "stage", "reason"), sorted(existing, key=lambda row: int(row["shard_index"])))


def run_toy(seed):
    def toy_candidate(identifier, intervals, views, quality="high"):
        mask = deleted_mask(intervals)
        return {
            "protein_id": "toy_protein", "perturbation_id": identifier, "plan_id": identifier,
            "wildtype_length": 20, "intervals": intervals,
            "intervals_json": canonical_json([list(item) for item in intervals]),
            "mask": mask, "num_intervals": len(intervals), "deleted_length": len(mask),
            "deletion_fraction": len(mask) / 20.0, "budget_bin": "ignored_bucket",
            "generation_policy": "toy_policy", "source_split": "train",
            "global_shift": views[0], "retained_shift": views[1], "junction": views[2],
            "impact_confidence": 0.7, "global_similarity": 1.0 - 2.0 * views[0],
            "retained_similarity": 1.0 - 2.0 * views[1], "candidate_quality": quality,
        }
    candidates = [
        toy_candidate("toy_a", ((1, 2),), (0.005, 0.010, 0.95)),
        toy_candidate("toy_b", ((5, 6),), (0.010, 0.020, 0.90)),
        toy_candidate("toy_c", ((1, 1), (8, 8)), (0.015, 0.025, 0.85)),
        toy_candidate("toy_d", ((10, 11),), (0.020, 0.030, 0.80)),
    ]
    counters = new_shard_counters()
    thresholds = {"local_high_mask_similarity": 0.5, "position_low_mask_similarity": 0.0}
    selected = select_pairs_for_protein("toy_protein", candidates, thresholds, 0, seed, TARGET_PAIRS, counters)
    assert selected
    assert all(int(row["deleted_length"]) == 2 for row in selected)
    assert len({tuple(sorted((row["perturbation_id_a"], row["perturbation_id_b"]))) for row in selected}) == len(selected)
    assert all(row["construction_policy"] == "strict_same_deleted_length_pair" for row in selected)
    return {
        "status": "PASS", "pairs": len(selected),
        "categories": dict(Counter(row["selection_category"] for row in selected)),
        "shortage": TARGET_PAIRS - len(selected), "gpu_used": False,
        "teacher_run": False, "student_training": False,
    }


def main():
    args = parse_args()
    if args.mode == "toy":
        print(json.dumps(run_toy(args.seed), indent=2, sort_keys=True))
        return 0
    candidate_rows = parse_index(args.candidate_index, "candidate")
    impact_rows = parse_index(args.impact_index, "impact")
    if set(candidate_rows) != set(impact_rows):
        raise ValueError("candidate and impact indexes have different shard sets")
    available = sorted(candidate_rows)
    calibration = ensure_calibration(args, candidate_rows, impact_rows, available)
    if args.mode == "calibrate":
        print(json.dumps(calibration, indent=2, sort_keys=True))
        return 0
    thresholds = calibration["thresholds"]
    selected_shards = parse_shard_selection(args.shards, available)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    failed = []

    def consume_result(ordinal, shard_index, report=None, exc=None):
        if exc is not None:
            reason = "%s: %s" % (type(exc).__name__, exc)
            append_failure(args.output_dir, shard_index, "build", reason)
            failed.append((shard_index, reason))
            print(canonical_json({"event": "shard_failed", "shard_index": shard_index, "reason": reason}), flush=True)
            return
        reports = load_all_reports(args.output_dir)
        summary = write_indexes_and_summary(args, reports, calibration, len(available))
        print(canonical_json({
            "event": "shard_complete", "ordinal": ordinal, "requested": len(selected_shards),
            "shard_index": shard_index, "status": report["status"],
            "pairs": report["num_pairs"], "shortage_proteins": report["num_shortage_proteins"],
            "shards_completed_total": summary["shards_completed"],
        }), flush=True)

    if args.workers == 1:
        for ordinal, shard_index in enumerate(selected_shards, 1):
            try:
                report = process_shard(
                    shard_index, candidate_rows[shard_index], impact_rows[shard_index],
                    args.output_dir, thresholds, args.seed, args.target_pairs,
                    args.resume, args.force,
                )
                consume_result(ordinal, shard_index, report=report)
            except Exception as exc:
                consume_result(ordinal, shard_index, exc=exc)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for shard_index in selected_shards:
                future = pool.submit(
                    process_shard, shard_index, candidate_rows[shard_index], impact_rows[shard_index],
                    args.output_dir, thresholds, args.seed, args.target_pairs, args.resume, args.force,
                )
                futures[future] = shard_index
            for ordinal, future in enumerate(as_completed(futures), 1):
                shard_index = futures[future]
                try:
                    consume_result(ordinal, shard_index, report=future.result())
                except Exception as exc:
                    consume_result(ordinal, shard_index, exc=exc)
    reports = load_all_reports(args.output_dir)
    summary = write_indexes_and_summary(args, reports, calibration, len(available))
    print(json.dumps({
        "status": "FAILED" if failed else summary["status"],
        "failed_shards": failed,
        "shards_completed": summary["shards_completed"],
        "pairs": summary["pairs"],
    }, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
