#!/usr/bin/env python3
"""Independent recomputation, integrity audit, and sealing for DelePref-14."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/experiments/delepref_step004000_delepref_14_confirmation_evaluation"
AUDIT = OUT / "audit"
WORK = AUDIT / "work"
CONFIRMATION = REPO / "data/processed/preference_training/delepref/splits/delepref_sequence_hash_v1/confirmation_manifest.csv"
CONFIRMATION_LOCK = REPO / "data/processed/preference_training/delepref/splits/delepref_sequence_hash_v1/confirmation_manifest_lock.json"
TRAIN_DEV = REPO / "data/processed/preference_training/delepref/splits/delepref_sequence_hash_v1/train_dev_manifest.csv"
CHECKPOINT = REPO / "results/experiments/delepref_evaluation_checkpoint/final_freeze/delepref_evaluation_step_00004000.pt"
PROTOCOL_SOURCE = Path(__file__).resolve()
EXPECTED_MANIFEST_SHA256 = "817fec50caf8c9b8634b5e783bc03ac3113f3272caf569efc0d21ec4d3d7a7de"
EXPECTED_CHECKPOINT_SHA256 = "1bc7694be77e24e4b671773be39331daed241dcfe101fd6d2d26f8e23fe7ff4d"
EXPECTED_PROTEINS = 15_053
BOOTSTRAPS = 10_000
BASE_SEED = 20260716
HIGH_CONFIDENCE_THRESHOLD = 0.70
ECE_BINS = 15
PAIR_SEED = 7_302_026
CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion perturbation preference "
    "and optional auxiliary risk evidence. This confirmation evaluation is not functional ground "
    "truth or wet-lab validation."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(*parts: object) -> str:
    return sha256_text("\x1f".join(str(part) for part in parts))


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def close(left: object, right: object, tolerance: float = 2.0e-10) -> bool:
    a, b = parse_float(left), parse_float(right)
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def expected_worker(protein: str, sequence_hash: str) -> int:
    return int(sha256_text("delepref-14-worker\0{}\0{}".format(protein, sequence_hash))[:16], 16) % 2


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return math.nan
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    dx, dy = [x - xm for x in xs], [y - ym for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denominator if denominator > 0 else math.nan


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(average_ranks(xs), average_ranks(ys)) if len(xs) >= 2 else math.nan


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = average_ranks(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def ece(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    if not labels:
        return math.nan
    result = 0.0
    for index in range(ECE_BINS):
        low, high = index / ECE_BINS, (index + 1) / ECE_BINS
        offsets = [i for i, probability in enumerate(probabilities) if probability >= low and (probability < high or (index == ECE_BINS - 1 and probability <= high))]
        if offsets:
            result += len(offsets) / len(labels) * abs(
                statistics.mean(probabilities[i] for i in offsets) - statistics.mean(labels[i] for i in offsets)
            )
    return result


def metrics_from_frame(frame: pd.DataFrame, score_field: str) -> dict[str, float]:
    non_ties = frame[frame["preference"].isin(["a_better", "b_better"])]
    labels = (non_ties["preference"] == "a_better").astype(int).tolist()
    scores = non_ties[score_field].astype(float).tolist()
    weights = non_ties["preference_confidence"].astype(float).tolist()
    teacher = non_ties["utility_margin_signed"].astype(float).tolist()
    correct = [int((score > 0) == bool(label)) for score, label in zip(scores, labels)]
    high = [index for index, weight in enumerate(weights) if weight >= HIGH_CONFIDENCE_THRESHOLD]
    probabilities = [sigmoid(score) for score in scores]
    return {
        "non_tie_accuracy": statistics.mean(correct) if correct else math.nan,
        "roc_auc": auc(labels, scores),
        "weighted_accuracy": sum(w * c for w, c in zip(weights, correct)) / sum(weights) if sum(weights) else math.nan,
        "high_confidence_accuracy": statistics.mean(correct[index] for index in high) if high else math.nan,
        "score_margin_pearson": pearson(teacher, scores),
        "score_margin_spearman": spearman(teacher, scores),
        "brier_score": statistics.mean((probability - label) ** 2 for probability, label in zip(probabilities, labels)) if labels else math.nan,
        "ece_15": ece(labels, probabilities),
        "tie_rate": float((frame["preference"] == "tie").sum()) / len(frame) if len(frame) else math.nan,
    }


def ndcg(utilities: Sequence[float], scores: Sequence[float], identities: Sequence[str]) -> float:
    if not utilities:
        return math.nan
    predicted = sorted(range(len(utilities)), key=lambda index: (-scores[index], identities[index]))
    ideal = sorted(range(len(utilities)), key=lambda index: (-utilities[index], identities[index]))
    def dcg(order: Sequence[int]) -> float:
        return sum((2.0 ** utilities[index] - 1.0) / math.log2(rank + 2.0) for rank, index in enumerate(order))
    denominator = dcg(ideal)
    return dcg(predicted) / denominator if denominator > 0 else math.nan


def independent_protein_values(rows: Sequence[Mapping[str, str]], source: str) -> dict[str, float]:
    subset = list(rows) if source == "ALL" else [row for row in rows if row["source"] == source]
    frame = pd.DataFrame(subset)
    output: dict[str, float] = {}
    fields = {"delepref": "score_delta", "random_score": "random_score_delta", "joint_metadata": "metadata_score_delta"}
    if subset:
        for method, field in fields.items():
            for metric, value in metrics_from_frame(frame, field).items():
                output["{}_{}".format(method, metric)] = value
    candidates: dict[str, tuple[float, float]] = {}
    for row in subset:
        for side in ("a", "b"):
            identity = "{}\0{}\0{}".format(row["perturbation_id_" + side], row["plan_id_" + side], row["deleted_intervals_" + side])
            value = (float(row["utility_" + side]), float(row["score_" + side]))
            previous = candidates.get(identity)
            if previous is not None and (not close(previous[0], value[0], 1.0e-7) or not close(previous[1], value[1], 1.0e-6)):
                raise RuntimeError("independent candidate consistency failure")
            candidates[identity] = value
    identities = sorted(candidates)
    utilities = [candidates[identity][0] for identity in identities]
    scores = [candidates[identity][1] for identity in identities]
    output["unique_candidate_count"] = float(len(identities))
    output["delepref_listwise_spearman"] = spearman(utilities, scores)
    output["delepref_ndcg"] = ndcg(utilities, scores, identities)
    if identities:
        top = min(range(len(identities)), key=lambda index: (-scores[index], identities[index]))
        oracle = min(range(len(identities)), key=lambda index: (-utilities[index], identities[index]))
        output["scorer_top1_teacher_utility"] = utilities[top]
        output["teacher_oracle_utility"] = utilities[oracle]
        output["oracle_gap"] = utilities[oracle] - utilities[top]
        output["top1_oracle_agreement"] = float(top == oracle)
    return output


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:
    checks.append({"name": name, "pass": bool(passed), "details": details})


def markdown_summary(main: Mapping[tuple[str, str, str], float], manifest_rows: Sequence[Mapping[str, str]], pair_count: int) -> str:
    def value(method: str, source: str, metric: str) -> float:
        return main[(method, source, metric)]
    missing_base = sum(int(row["base_pair_count"]) == 0 for row in manifest_rows)
    missing_b20 = sum(int(row["b20_pair_count"]) == 0 for row in manifest_rows)
    return """# DelePref-14 One-Time Confirmation Evaluation

Status: `PASS_DelePref_14_CONFIRMATION_EVALUATION`
Opening state: `DelePref_CONFIRMATION_SET_OPENED_ONCE`

The protocol was durably frozen before the locked manifest was opened. The evaluation used all 15,053 confirmation proteins, the fixed step-4000 checkpoint, the frozen scorer only, and {pairs:,} legal strict same-count pair rows. No training, fine-tuning, checkpoint/threshold selection, proposal generation, or teacher rescoring occurred.

| method | source | non-tie accuracy | ROC-AUC | weighted accuracy | high-confidence accuracy | margin Spearman | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DelePref | base | {base_acc:.6f} | {base_auc:.6f} | {base_weighted:.6f} | {base_high:.6f} | {base_spearman:.6f} | {base_brier:.6f} | {base_ece:.6f} |
| DelePref | b20 | {b20_acc:.6f} | {b20_auc:.6f} | {b20_weighted:.6f} | {b20_high:.6f} | {b20_spearman:.6f} | {b20_brier:.6f} | {b20_ece:.6f} |
| DelePref | ALL | {all_acc:.6f} | {all_auc:.6f} | {all_weighted:.6f} | {all_high:.6f} | {all_spearman:.6f} | {all_brier:.6f} | {all_ece:.6f} |

Protein-outer confidence intervals use 10,000 replicates with seed 20260716. Random and joint-metadata baselines reuse their DelePref-1 frozen definitions; the metadata model was not refit. Development values in `baseline_comparison.csv` are labeled `EXTERNAL_REFERENCE` and are excluded from confirmation intervals.

Coverage retains all 15,053 manifest proteins. Proteins without a legal source pair are explicit rather than dropped: base `{missing_base}`, B20 `{missing_b20}`. Detailed pair, per-protein, listwise, oracle-gap, stratified, calibration, bootstrap, and missingness results are in the auditable CSV artifacts.

Claim boundary: {claim}
""".format(
        pairs=pair_count,
        base_acc=value("delepref", "base", "non_tie_accuracy"), base_auc=value("delepref", "base", "roc_auc"),
        base_weighted=value("delepref", "base", "weighted_accuracy"), base_high=value("delepref", "base", "high_confidence_accuracy"),
        base_spearman=value("delepref", "base", "score_margin_spearman"), base_brier=value("delepref", "base", "brier_score"), base_ece=value("delepref", "base", "ece_15"),
        b20_acc=value("delepref", "b20", "non_tie_accuracy"), b20_auc=value("delepref", "b20", "roc_auc"),
        b20_weighted=value("delepref", "b20", "weighted_accuracy"), b20_high=value("delepref", "b20", "high_confidence_accuracy"),
        b20_spearman=value("delepref", "b20", "score_margin_spearman"), b20_brier=value("delepref", "b20", "brier_score"), b20_ece=value("delepref", "b20", "ece_15"),
        all_acc=value("delepref", "ALL", "non_tie_accuracy"), all_auc=value("delepref", "ALL", "roc_auc"),
        all_weighted=value("delepref", "ALL", "weighted_accuracy"), all_high=value("delepref", "ALL", "high_confidence_accuracy"),
        all_spearman=value("delepref", "ALL", "score_margin_spearman"), all_brier=value("delepref", "ALL", "brier_score"), all_ece=value("delepref", "ALL", "ece_15"),
        missing_base=missing_base, missing_b20=missing_b20, claim=CLAIM_BOUNDARY,
    )


def main() -> int:
    checks: list[dict[str, Any]] = []
    protocol = read_json(OUT / "protocol.json")
    protocol_hash = sha256_file(OUT / "protocol.json")
    protocol_done = read_json(OUT / "protocol.done.json")
    opened = read_json(OUT / "confirmation_open_record.json")
    add_check(checks, "protocol_hash_and_done_sentinel", protocol_done.get("protocol_sha256") == protocol_hash and protocol_done.get("status") == "PROTOCOL_FROZEN_AND_FSYNCED", {"protocol_sha256": protocol_hash})
    code_mismatches = {
        path: {"expected": expected, "actual": sha256_file(REPO / path)}
        for path, expected in protocol["code"].items()
        if not (REPO / path).is_file() or sha256_file(REPO / path) != expected
    }
    add_check(checks, "protocol_frozen_code_hashes", not code_mismatches, {"files": len(protocol["code"]), "mismatches": code_mismatches})
    add_check(checks, "protocol_precedes_first_open", protocol["frozen_at_utc"] <= opened["first_opened_at_utc"] and protocol_done["frozen_at_utc"] <= opened["first_opened_at_utc"], {"frozen": protocol["frozen_at_utc"], "opened": opened["first_opened_at_utc"]})
    add_check(checks, "opening_record_identity", opened.get("status") == "DelePref_CONFIRMATION_SET_OPENED_ONCE" and opened.get("logical_reveal_count") == 1 and opened.get("not_for_checkpoint_selection") is True, opened)
    add_check(checks, "locked_input_hashes", sha256_file(CONFIRMATION) == EXPECTED_MANIFEST_SHA256 and sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA256 and sha256_file(CONFIRMATION_LOCK) == opened["manifest_lock_sha256_before_open"], {"manifest": sha256_file(CONFIRMATION), "checkpoint": sha256_file(CHECKPOINT), "lock": sha256_file(CONFIRMATION_LOCK)})

    manifest_rows = read_csv(CONFIRMATION)
    manifest = {row["protein_id"]: row for row in manifest_rows}
    manifest_ids = set(manifest)
    manifest_sequences = {row["sequence_sha256"] for row in manifest_rows}
    audit_rows = read_csv(OUT / "confirmation_manifest_audit.csv")
    audit_ids = {row["protein_id"] for row in audit_rows}
    add_check(checks, "all_confirmation_membership", len(manifest_rows) == EXPECTED_PROTEINS == len(manifest_ids) == len(manifest_sequences) and audit_ids == manifest_ids, {"manifest_rows": len(manifest_rows), "ids": len(manifest_ids), "sequence_groups": len(manifest_sequences), "audit_ids": len(audit_ids)})
    train_dev = read_csv(TRAIN_DEV)
    overlap_ids = manifest_ids & {row["protein_id"] for row in train_dev}
    overlap_sequences = manifest_sequences & {row["sequence_sha256"] for row in train_dev}
    add_check(checks, "train_dev_zero_overlap", not overlap_ids and not overlap_sequences, {"protein_overlap": len(overlap_ids), "sequence_overlap": len(overlap_sequences)})
    source_audit = read_csv(OUT / "source_shard_audit.csv")
    add_check(checks, "all_600_source_shards_audited", len(source_audit) == 600 and all(row["status"] == "PASS" for row in source_audit) and sum(int(row["confirmation_manifest_proteins"]) for row in source_audit) == 2 * EXPECTED_PROTEINS, {"rows": len(source_audit), "manifest_assignments": sum(int(row["confirmation_manifest_proteins"]) for row in source_audit)})

    pair_path = OUT / "confirmation_pair_scores.csv"
    protein_metric_rows = read_csv(OUT / "confirmation_protein_metrics.csv")
    protein_lookup = {(row["protein_id"], row["source"]): row for row in protein_metric_rows}
    protein_failures = 0
    verified_groups = 0

    def validate_finished_protein(protein: str, rows: Sequence[Mapping[str, str]]) -> None:
        nonlocal protein_failures, verified_groups
        for source in ("base", "b20", "ALL"):
            expected = independent_protein_values(rows, source)
            observed = protein_lookup[(protein, source)]
            for field, value in expected.items():
                if not close(observed.get(field), value, 2.0e-8):
                    protein_failures += 1
            verified_groups += 1

    pair_ids = set()
    pair_count = 0
    worker_counts = defaultdict(int)
    pair_proteins = set()
    pair_identity_failures = 0
    previous_worker_protein: tuple[int, str] | None = None
    contiguous_seen = set()
    current_protein = None
    current_rows: list[dict[str, str]] = []
    with pair_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pair_count += 1
            protein = row["protein_id"]
            pair_proteins.add(protein)
            worker = int(row["worker_id"])
            worker_counts[worker] += 1
            if protein not in manifest or worker != expected_worker(protein, manifest[protein]["sequence_sha256"]):
                pair_identity_failures += 1
            expected_id = "{}__pairdelepref__{}".format(protein, stable_hash(protein, row["perturbation_id_a"], row["perturbation_id_b"], PAIR_SEED)[:24])
            delta = float(row["utility_a"]) - float(row["utility_b"])
            expected_preference = "tie" if abs(delta) <= float(row["tie_margin"]) else ("a_better" if delta > 0 else "b_better")
            if row["pair_id"] != expected_id or row["pair_id"] in pair_ids or row["preference"] != expected_preference or not close(row["utility_margin_signed"], delta, 1.0e-7) or row["metric_scope"] != "teacher_preference_fidelity_not_biological_ground_truth":
                pair_identity_failures += 1
            pair_ids.add(row["pair_id"])
            key = (worker, protein)
            if previous_worker_protein is not None and key != previous_worker_protein:
                contiguous_seen.add(previous_worker_protein)
                if key in contiguous_seen:
                    pair_identity_failures += 1
            previous_worker_protein = key
            if current_protein is None:
                current_protein = protein
            if protein != current_protein:
                validate_finished_protein(current_protein, current_rows)
                current_protein, current_rows = protein, []
            current_rows.append(row)
    if current_protein is not None:
        validate_finished_protein(current_protein, current_rows)
    manifest_expected_pairs = sum(int(row["all_pair_count"]) for row in audit_rows)
    worker_summaries = [read_json(WORK / "worker_{}_summary.json".format(worker)) for worker in (0, 1)]
    add_check(checks, "pair_identity_membership_and_partition", pair_identity_failures == 0 and pair_count == manifest_expected_pairs == len(pair_ids) == sum(int(row["pairs"]) for row in worker_summaries) and set(worker_counts) == {0, 1}, {"pairs": pair_count, "unique_pair_ids": len(pair_ids), "expected": manifest_expected_pairs, "worker_counts": dict(worker_counts), "failures": pair_identity_failures})
    add_check(checks, "dual_gpu_worker_no_overlap_or_omission", set(row["protein_id"] for row in audit_rows if int(row["all_pair_count"]) > 0) == pair_proteins and all(row["status"] == "PASS" and row["shared_encoder_identity"] and not row["policy_invoked"] and not row["teacher_invoked"] for row in worker_summaries), {"scored_proteins": len(pair_proteins), "workers": worker_summaries})

    expected_protein_rows = EXPECTED_PROTEINS * 3
    missing_rows = [row for row in protein_metric_rows if row["status"].startswith("MISSING")]
    add_check(checks, "all_per_protein_pair_listwise_ndcg_oracle_metrics", len(protein_metric_rows) == expected_protein_rows and protein_failures == 0, {"rows": len(protein_metric_rows), "verified_nonempty_groups": verified_groups, "missing_rows": len(missing_rows), "numeric_mismatches": protein_failures})

    usecols = [
        "protein_id", "source", "evaluation_budget", "wildtype_length", "length_bin", "preference",
        "preference_confidence", "utility_margin_signed", "score_delta", "random_score_delta", "metadata_score_delta",
    ]
    frame = pd.read_csv(pair_path, usecols=usecols)
    fidelity_rows = read_csv(OUT / "confirmation_fidelity_summary.csv")
    fidelity = {(row["method"], row["source"], row["metric"]): float(row["value"]) for row in fidelity_rows}
    fields = {"delepref": "score_delta", "random_score": "random_score_delta", "joint_metadata": "metadata_score_delta"}
    aggregate_failures = 0
    aggregate_recomputed = 0
    for method, field in fields.items():
        for source in ("base", "b20", "ALL"):
            subset = frame if source == "ALL" else frame[frame["source"] == source]
            for metric, value in metrics_from_frame(subset, field).items():
                if not close(fidelity[(method, source, metric)], value, 2.0e-10):
                    aggregate_failures += 1
                aggregate_recomputed += 1
    add_check(checks, "all_pair_fidelity_and_calibration_metrics", aggregate_failures == 0, {"metrics_recomputed": aggregate_recomputed, "mismatches": aggregate_failures})

    stratified_rows = read_csv(OUT / "confirmation_stratified_metrics.csv")
    stratified_failures = 0
    for row in stratified_rows:
        subset = frame
        if row["stratum_type"] == "budget":
            subset = subset[np.isclose(subset["evaluation_budget"].astype(float), float(row["stratum_value"]))]
        elif row["stratum_type"] == "length_bin":
            subset = subset[subset["length_bin"] == row["stratum_value"]]
        else:
            stratified_failures += 1
            continue
        if row["source"] != "ALL":
            subset = subset[subset["source"] == row["source"]]
        recomputed = metrics_from_frame(subset, fields[row["method"]])[row["metric"]]
        if not close(row["value"], recomputed, 2.0e-10) or int(row["pair_count"]) != len(subset):
            stratified_failures += 1
    add_check(checks, "all_budget_and_length_stratified_metrics", stratified_failures == 0, {"rows": len(stratified_rows), "mismatches": stratified_failures})

    bootstrap_rows = read_csv(OUT / "confirmation_protein_bootstrap_ci.csv")
    protein_frame = pd.read_csv(OUT / "confirmation_protein_metrics.csv")
    bootstrap_failures = 0
    for row in bootstrap_rows:
        method, source, metric = row["method"], row["source"], row["metric"]
        field = metric if metric in {"scorer_top1_teacher_utility", "teacher_oracle_utility", "oracle_gap", "top1_oracle_agreement"} else "{}_{}".format(method, metric)
        values = protein_frame.loc[protein_frame["source"] == source, field].dropna().astype(float).to_numpy()
        seed = int(row["metric_seed"])
        rng = np.random.default_rng(seed)
        samples = np.empty(BOOTSTRAPS, dtype=np.float64)
        for start in range(0, BOOTSTRAPS, 128):
            end = min(BOOTSTRAPS, start + 128)
            samples[start:end] = values[rng.integers(0, len(values), size=(end - start, len(values)))].mean(axis=1)
        expected = (float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))
        if int(row["bootstrap_n"]) != BOOTSTRAPS or int(row["base_seed"]) != BASE_SEED or row["resample_unit"] != "protein_id" or int(row["protein_count"]) != len(values) or any(not close(observed, wanted, 2.0e-10) for observed, wanted in zip((row["observed_macro_mean"], row["ci95_low"], row["ci95_high"]), expected)):
            bootstrap_failures += 1
    add_check(checks, "all_10000_replicate_protein_bootstraps", bootstrap_failures == 0 and len(bootstrap_rows) > 0, {"rows": len(bootstrap_rows), "mismatches": bootstrap_failures, "replicates": BOOTSTRAPS, "unit": "protein_id"})

    baseline_rows = read_csv(OUT / "baseline_comparison.csv")
    baseline_checks = (
        any(row["comparison_type"] == "CONFIRMATION" and row["baseline"] == "random_score" for row in baseline_rows)
        and any(row["comparison_type"] == "CONFIRMATION" and row["baseline"] == "joint_metadata" for row in baseline_rows)
        and all(row["included_in_confirmation_ci"] == "False" for row in baseline_rows if row["comparison_type"] == "EXTERNAL_REFERENCE")
        and all(row["fit_on_confirmation"] == "False" for row in baseline_rows)
    )
    add_check(checks, "frozen_baselines_and_external_reference_boundary", baseline_checks, {"rows": len(baseline_rows), "external_reference_rows": sum(row["comparison_type"] == "EXTERNAL_REFERENCE" for row in baseline_rows)})
    checkpoint = read_json(OUT / "checkpoint_provenance.json")
    add_check(checks, "checkpoint_before_after_immutable", checkpoint["evaluation_checkpoint_sha256_before"] == checkpoint["evaluation_checkpoint_sha256_after"] == EXPECTED_CHECKPOINT_SHA256 == sha256_file(CHECKPOINT) and checkpoint["training_or_finetuning_run"] is False and checkpoint["checkpoint_selection_run"] is False, checkpoint)
    resource = read_json(OUT / "resource_provenance.json")
    add_check(checks, "designated_resource_provenance", resource.get("status") == "PASS" and resource["pre_execution_gate"]["status"] == "PASS" and len(resource["workers"]) == 2, resource)

    py_compile = subprocess.run([sys.executable, "-m", "py_compile", str(REPO / "scripts/run_delepref_14_confirmation_evaluation.py"), str(Path(__file__).resolve())], cwd=str(REPO), text=True, capture_output=True)
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=str(REPO), text=True, capture_output=True)
    index_check = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(REPO))
    add_check(checks, "py_compile", py_compile.returncode == 0, {"stdout": py_compile.stdout, "stderr": py_compile.stderr})
    add_check(checks, "git_diff_check", diff_check.returncode == 0, {"stdout": diff_check.stdout, "stderr": diff_check.stderr})
    add_check(checks, "git_index_empty", index_check.returncode == 0, {"returncode": index_check.returncode})
    protocol_source_hash = sha256_file(PROTOCOL_SOURCE)
    frozen_protocol_source_hash = protocol["frozen_inputs"]["protocol_source_sha256_at_freeze"]
    add_check(checks, "protocol_source_untouched", protocol_source_hash == frozen_protocol_source_hash, {"frozen": frozen_protocol_source_hash, "current": protocol_source_hash})

    preliminary_pass = all(row["pass"] for row in checks)
    if not preliminary_pass:
        failure = {
            "schema_version": "delepref_14_independent_validation_v1", "status": "FAIL",
            "validated_at_utc": utc_now(), "checks": checks, "all_numeric_values_independently_recomputed": False,
        }
        atomic_json(OUT / "independent_validation.json", failure)
        raise RuntimeError("independent validation failed: {}".format([row["name"] for row in checks if not row["pass"]]))

    summary_lookup = {(row["method"], row["source"], row["metric"]): float(row["value"]) for row in fidelity_rows}
    summary_md = markdown_summary(summary_lookup, audit_rows, pair_count)
    atomic_text(OUT / "SUMMARY.md", summary_md)
    payload_md = """<!-- DelePref_POSTFREEZE_CONFIRMATION_START -->
## DelePref-14: One-Time Full Confirmation-Set Evaluation

- Status: `PASS_DelePref_14_CONFIRMATION_EVALUATION`
- Opening: `DelePref_CONFIRMATION_SET_OPENED_ONCE`
- Frozen protocol SHA-256: `{protocol_hash}`
- Evaluation checkpoint: step 4000, SHA-256 `{checkpoint_hash}`
- Scope: all 15,053 locked confirmation proteins and {pairs:,} legal teacher-derived strict same-count pair rows.
- Statistics: protein-outer macro bootstrap, 10,000 replicates, seed 20260716; no pair-row bootstrap.
- Baselines: DelePref-1 frozen random and train-only joint-metadata definitions; no confirmation fit.
- Negative, null, missing, stratified, calibration, and oracle-gap results are retained in the result package.
- No training, fine-tuning, checkpoint/threshold selection, proposal generation, or teacher rescoring occurred.
- Claim boundary: {claim}
<!-- DelePref_POSTFREEZE_CONFIRMATION_END -->
""".format(protocol_hash=protocol_hash, checkpoint_hash=EXPECTED_CHECKPOINT_SHA256, pairs=pair_count, claim=CLAIM_BOUNDARY)
    atomic_text(OUT / "result_summary.md", payload_md)
    payload_json = {
        "schema_version": "delepref_14_result_summary_v1",
        "start_marker": "DelePref_POSTFREEZE_CONFIRMATION_START", "end_marker": "DelePref_POSTFREEZE_CONFIRMATION_END",
        "status": "PASS_DelePref_14_CONFIRMATION_EVALUATION", "opening_status": "DelePref_CONFIRMATION_SET_OPENED_ONCE",
        "protocol_sha256": protocol_hash, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "confirmation_proteins": EXPECTED_PROTEINS, "confirmation_pair_rows": pair_count,
        "fidelity": {"{}|{}|{}".format(*key): value for key, value in summary_lookup.items()},
        "bootstrap": {"unit": "protein_id", "replicates": BOOTSTRAPS, "seed": BASE_SEED},
        "claim_boundary": CLAIM_BOUNDARY, "negative_results_hidden": False,
        "direct_master_modification_performed": False,
    }
    atomic_json(OUT / "result_summary.json", payload_json)
    scientific = read_json(AUDIT / "scientific_summary.json")
    run_summary = {
        "schema_version": "delepref_14_confirmation_run_summary_v1",
        "status": "PASS_DelePref_14_CONFIRMATION_EVALUATION", "opening_status": "DelePref_CONFIRMATION_SET_OPENED_ONCE",
        "finished_at_utc": utc_now(), "protocol_sha256": protocol_hash,
        "checkpoint_sha256_before": EXPECTED_CHECKPOINT_SHA256, "checkpoint_sha256_after": sha256_file(CHECKPOINT),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256, "manifest_lock_sha256": sha256_file(CONFIRMATION_LOCK),
        "confirmation_proteins": EXPECTED_PROTEINS, "confirmation_sequence_groups": EXPECTED_PROTEINS,
        "valid_pair_rows": pair_count, "valid_pair_proteins": len(pair_proteins),
        "missing_base_proteins": sum(int(row["base_pair_count"]) == 0 for row in audit_rows),
        "missing_b20_proteins": sum(int(row["b20_pair_count"]) == 0 for row in audit_rows),
        "bootstrap": {"unit": "protein_id", "replicates": BOOTSTRAPS, "seed": BASE_SEED},
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD, "ece_bins": ECE_BINS,
        "training_or_refit_run": False, "checkpoint_or_threshold_selection_run": False,
        "proposal_generation_run": False, "teacher_rescoring_run": False,
        "all_negative_and_missing_results_retained": True, "scientific_summary": scientific,
        "independent_validation": "PASS", "protocol_source_modified": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    atomic_json(OUT / "run_summary.json", run_summary)

    validation = {
        "schema_version": "delepref_14_independent_validation_v1", "status": "PASS",
        "validated_at_utc": utc_now(), "checks": checks,
        "all_confirmation_membership_rechecked": True, "all_pair_identities_rechecked": True,
        "all_numeric_values_independently_recomputed": True,
        "all_10000_replicate_protein_bootstraps_recomputed": True,
        "json_csv_parse_check": "PENDING_FINAL_PACKAGE_PARSE_IN_SAME_VALIDATOR",
        "artifact_hash_manifest_written_after_validation": True,
    }
    atomic_json(OUT / "independent_validation.json", validation)
    atomic_text(OUT / "PASS_DelePref_14_CONFIRMATION_EVALUATION", "PASS_DelePref_14_CONFIRMATION_EVALUATION\n")
    atomic_text(OUT / "DelePref_CONFIRMATION_SET_OPENED_ONCE", "DelePref_CONFIRMATION_SET_OPENED_ONCE\n")

    required = protocol["required_outputs"]
    missing = [name for name in required if name != "audit_manifest.json" and not (OUT / name).is_file()]
    if missing:
        raise RuntimeError("required package outputs missing after sealing: {}".format(missing))
    parse_failures = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name in {"audit_manifest.json", "formal_launcher.log"}:
            continue
        try:
            if path.suffix == ".json":
                read_json(path)
            elif path.suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader, None)
                    if not header:
                        raise ValueError("empty CSV")
                    for _ in reader:
                        pass
        except Exception as exc:
            parse_failures.append({"path": str(path.relative_to(REPO)), "error": str(exc)})
    if parse_failures:
        raise RuntimeError("final JSON/CSV parse failures: {}".format(parse_failures[:10]))
    validation["json_csv_parse_check"] = "PASS"
    atomic_json(OUT / "independent_validation.json", validation)

    artifacts = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name in {"audit_manifest.json", "formal_launcher.log"}:
            continue
        artifacts.append({
            "path": str(path.relative_to(REPO)), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "role": "result_artifact",
        })
    manifest_payload = {
        "schema_version": "delepref_14_audit_manifest_v1", "status": "PASS",
        "created_at_utc": utc_now(), "artifact_count": len(artifacts), "artifacts": artifacts,
        "protocol_sha256": protocol_hash, "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "confirmation_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "sentinels": ["PASS_DelePref_14_CONFIRMATION_EVALUATION", "DelePref_CONFIRMATION_SET_OPENED_ONCE"],
    }
    atomic_json(OUT / "audit_manifest.json", manifest_payload)
    stored = read_json(OUT / "audit_manifest.json")
    mismatches = []
    for item in stored["artifacts"]:
        path = REPO / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            mismatches.append(item["path"])
    if mismatches:
        raise RuntimeError("postwrite artifact hash mismatch: {}".format(mismatches[:10]))
    print(json.dumps({
        "status": "PASS_DelePref_14_CONFIRMATION_EVALUATION",
        "opening_status": "DelePref_CONFIRMATION_SET_OPENED_ONCE",
        "checks": len(checks), "artifacts": len(artifacts), "pairs": pair_count,
        "proteins": EXPECTED_PROTEINS,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
