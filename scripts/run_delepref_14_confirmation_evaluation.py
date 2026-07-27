#!/usr/bin/env python3
"""Run the one-time post-freeze DelePref confirmation evaluation.

The confirmation manifest is inaccessible to every mode except ``--run`` and
post-open packaging/validation.  ``--freeze-protocol`` only records file
existence, byte size, and SHA-256 for the locked confirmation assets.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/experiments/delepref_step004000_delepref_14_confirmation_evaluation"
AUDIT = OUT / "audit"
ROUTING = AUDIT / "routing"
WORK = AUDIT / "work"
CORRECTED = REPO / "data/processed/preference_training/delepref"
CONFIRMATION = CORRECTED / "splits/delepref_sequence_hash_v1/confirmation_manifest.csv"
CONFIRMATION_LOCK = CORRECTED / "splits/delepref_sequence_hash_v1/confirmation_manifest_lock.json"
TRAIN_DEV = CORRECTED / "splits/delepref_sequence_hash_v1/train_dev_manifest.csv"
BASE_INDEX = CORRECTED / "base_pairs/indexes/preference_pair_shard_index.csv"
B20_INDEX = CORRECTED / "b20_supplement/pairs/indexes/preference_pair_shard_index.csv"
CONFIG = REPO / "configs/deletion_preference_student.yaml"
CHECKPOINT = REPO / "results/experiments/delepref_evaluation_checkpoint/final_freeze/delepref_evaluation_step_00004000.pt"
DelePref_1 = REPO / "results/experiments/delepref_step004000_exp0_exp1_exp2"
DelePref_1_COEFFICIENTS = DelePref_1 / "exp01_metadata_coefficients.csv"
DelePref_1_METADATA_AUDIT = DelePref_1 / "exp01_metadata_baseline_audit.json"
DelePref_1_METRICS = DelePref_1 / "exp01_main_metrics.csv"
DelePref_13 = REPO / "results/experiments/delepref_step004000_delepref_13_supplemental_external_rollup"
PROTOCOL_SOURCE = Path(__file__).resolve()
VALIDATOR = REPO / "scripts/validate_delepref_14_confirmation_evaluation.py"
LAUNCHER = ROUTING / "launcher.sh"

EXPECTED_CHECKPOINT_SHA256 = "1bc7694be77e24e4b671773be39331daed241dcfe101fd6d2d26f8e23fe7ff4d"
EXPECTED_MANIFEST_SHA256 = "817fec50caf8c9b8634b5e783bc03ac3113f3272caf569efc0d21ec4d3d7a7de"
EXPECTED_PROTEINS = 15_053
EXPECTED_SEQUENCE_GROUPS = 15_053
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260716
HIGH_CONFIDENCE_THRESHOLD = 0.70
ECE_BINS = 15
PAIR_SEED = 7_302_026
SOURCES = {"base": 0.10, "b20": 0.20}
LENGTH_BINS = (
    (2, 79, "002_079"),
    (80, 127, "080_127"),
    (128, 191, "128_191"),
    (192, 255, "192_255"),
    (256, 383, "256_383"),
    (384, 511, "384_511"),
    (512, 800, "512_800"),
    (801, 1022, "801_1022"),
)
CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion "
    "perturbation preference and optional auxiliary risk evidence. This "
    "confirmation evaluation is not functional ground truth or wet-lab validation."
)

for entry in (REPO / "scripts", REPO / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from delepref.preference_core import (  # noqa: E402
    COORDINATE_SYSTEM,
    DeletionCandidate,
    parse_deleted_intervals,
)
from delepref.student_model import strict_delete_count  # noqa: E402
from evaluation_runtime import load_runtime  # noqa: E402


PAIR_COLUMNS = [
    "worker_id", "source", "evaluation_budget", "pair_id", "protein_id",
    "sequence_sha256", "wildtype_length", "preference", "preference_margin",
    "preference_confidence", "tie_margin", "utility_a", "utility_b",
    "utility_margin_signed", "deleted_length", "perturbation_id_a", "plan_id_a",
    "perturbation_id_b", "plan_id_b", "deleted_intervals_a", "deleted_intervals_b",
    "num_intervals_a", "num_intervals_b", "generation_policy_a",
    "generation_policy_b", "budget_bin_a", "budget_bin_b", "selection_category",
    "selection_mode", "pair_quality_bucket", "scorer_training_weight",
    "score_a", "score_b", "score_delta", "random_score_a", "random_score_b",
    "random_score_delta", "metadata_score_delta", "student_predicted_preference",
    "random_predicted_preference", "metadata_predicted_preference", "length_bin",
    "metric_scope",
]

EXTRACTION_COLUMNS = [column for column in PAIR_COLUMNS if column not in {
    "score_a", "score_b", "score_delta", "random_score_a", "random_score_b",
    "random_score_delta", "metadata_score_delta", "student_predicted_preference",
    "random_predicted_preference", "metadata_predicted_preference", "metric_scope",
}]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(*parts: object) -> str:
    return sha256_text("\x1f".join(str(part) for part in parts))


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def append_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]], first: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if first else "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        if first:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def file_metadata_only(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": relative(path),
        "exists": path.is_file(),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def parse_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def close(left: object, right: object, tolerance: float = 5.0e-8) -> bool:
    a, b = float(left), float(right)
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def length_bin(length: int) -> str:
    for low, high, label in LENGTH_BINS:
        if low <= int(length) <= high:
            return label
    raise ValueError("length outside preregistered bins: {}".format(length))


def parse_intervals(value: object, length: int) -> tuple[tuple[int, int], ...]:
    intervals = tuple(parse_deleted_intervals(value, length))
    if not intervals or len(intervals) > 3:
        raise ValueError("interval count outside 1..3")
    previous = -2
    for start, end in intervals:
        if start < 0 or end >= length or start > end or start <= previous + 1:
            raise ValueError("invalid/non-separated interval geometry")
        previous = end
    return intervals


def canonical_intervals(intervals: Sequence[tuple[int, int]]) -> str:
    return json.dumps([[int(a), int(b)] for a, b in intervals], separators=(",", ":"))


def freeze_protocol() -> int:
    """Freeze the complete analysis protocol without opening confirmation rows."""

    OUT.mkdir(parents=True, exist_ok=True)
    ROUTING.mkdir(parents=True, exist_ok=True)
    if (OUT / "confirmation_open_record.json").exists():
        raise RuntimeError("confirmation was already opened; protocol cannot be rewritten")
    if (OUT / "protocol.json").exists():
        raise FileExistsError("refusing to overwrite frozen protocol")
    delepref_13_summary = read_json(DelePref_13 / "run_summary.json")
    delepref_13_validation = read_json(DelePref_13 / "audit/final_validation.json")
    if delepref_13_summary.get("status") != "PASS_DelePref_13_SUPPLEMENTAL_EXTERNAL_ROLLUP":
        raise RuntimeError("DelePref-13 prerequisite is not PASS")
    if not bool(delepref_13_validation.get("status") == "PASS" or all(row.get("pass") for row in delepref_13_validation.get("checks", []))):
        raise RuntimeError("DelePref-13 independent validation is not PASS")
    checkpoint_meta = file_metadata_only(CHECKPOINT)
    lock_meta = file_metadata_only(CONFIRMATION_LOCK)
    manifest_meta = file_metadata_only(CONFIRMATION)
    if checkpoint_meta["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("evaluation checkpoint hash mismatch")
    if manifest_meta["sha256"] != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("locked confirmation manifest hash mismatch")
    required_scripts = [Path(__file__).resolve(), VALIDATOR, LAUNCHER]
    missing = [str(path) for path in required_scripts if not path.is_file()]
    if missing:
        raise RuntimeError("protocol code/launcher missing: {}".format(missing))
    payload = {
        "schema_version": "delepref_14_confirmation_protocol_v1",
        "status": "FROZEN_BEFORE_CONFIRMATION_OPEN",
        "frozen_at_utc": utc_now(),
        "irreversible_one_time_reveal": True,
        "objective": "DelePref one-time full confirmation-set evaluation",
        "claim_boundary": CLAIM_BOUNDARY,
        "prohibitions": {
            "training": True,
            "fine_tuning": True,
            "checkpoint_reselection": True,
            "threshold_reselection": True,
            "protocol_change_after_results": True,
            "proposal_generation": True,
            "teacher_rescoring": True,
            "functional_ground_truth_claim": True,
        },
        "prerequisite": {
            "required_status": "PASS_DelePref_13_SUPPLEMENTAL_EXTERNAL_ROLLUP",
            "observed_status": delepref_13_summary["status"],
            "run_summary_sha256": sha256_file(DelePref_13 / "run_summary.json"),
            "independent_validation_sha256": sha256_file(DelePref_13 / "audit/final_validation.json"),
        },
        "pre_open_confirmation_checks": {
            "rule": "existence, byte size, and SHA-256 only; no CSV parsing, protein IDs, pair rows, candidate metrics, or lock JSON parsing",
            "confirmation_manifest": manifest_meta,
            "confirmation_lock": lock_meta,
            "expected_proteins": EXPECTED_PROTEINS,
            "expected_sequence_groups": EXPECTED_SEQUENCE_GROUPS,
        },
        "frozen_inputs": {
            "evaluation_checkpoint": checkpoint_meta,
            "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "config": {**file_metadata_only(CONFIG), "role": "frozen DelePref architecture/evaluation config"},
            "base_pair_index": file_metadata_only(BASE_INDEX),
            "b20_pair_index": file_metadata_only(B20_INDEX),
            "delepref_1_metadata_coefficients": file_metadata_only(DelePref_1_COEFFICIENTS),
            "delepref_1_metadata_audit": file_metadata_only(DelePref_1_METADATA_AUDIT),
            "delepref_1_development_metrics": file_metadata_only(DelePref_1_METRICS),
            "protocol_source_sha256_at_freeze": sha256_file(PROTOCOL_SOURCE),
        },
        "model_inference": {
            "student": "DelePref scorer from evaluation-only step 4000 checkpoint",
            "shared_encoder": "one frozen ESMC encoder object shared by DelePref scorer/policy heads within each GPU worker; policy is not invoked",
            "wildtype_encoding": "deduplicate by protein_id and exact sequence SHA-256; encode each WT once per worker scoring chunk",
            "gpu_partition": "sha256('delepref-14-worker\\0' + protein_id + '\\0' + sequence_sha256) first 64 bits modulo 2",
            "devices": "exactly two visible CUDA GPUs",
        },
        "data_extraction": {
            "membership": "all 15,053 confirmation proteins; no subsampling or length/result/resource filtering",
            "sources": {"base": relative(BASE_INDEX), "b20": relative(B20_INDEX)},
            "eligible_rows": "pair protein_id must be in confirmation manifest; strict same-protein, same-WT, same-deleted-length pairs at exact round(beta*L) count",
            "identity_validation": "recompute pair_id from protein_id, canonical ordered perturbation IDs, and frozen seed 7302026; validate plan/candidate linkage, intervals, WT, utilities, label, confidence, and source shard",
            "train_dev_metric_rows_forbidden": True,
            "missingness": "retain every confirmation protein and report zero-pair/source-linkage reasons explicitly",
        },
        "metrics": {
            "pair_scope": "teacher-derived same-budget deletion preference; ties excluded from classification/correlation/calibration but counted diagnostically",
            "primary": {
                "non_tie_pair_accuracy": "I(score_delta > 0) for a_better else I(score_delta <= 0) for b_better; mean over non-ties",
                "roc_auc": "Mann-Whitney rank AUC with average ranks for tied scores; positive=a_better",
                "weighted_accuracy": "accuracy weighted by frozen preference_confidence",
                "high_confidence_accuracy": "non-tie accuracy where preference_confidence >= 0.70 (DelePref-1 frozen threshold)",
                "score_margin_pearson": "Pearson(score_delta, utility_margin_signed) over non-ties",
                "score_margin_spearman": "Pearson of average ranks over non-ties",
            },
            "calibration": {
                "included": True,
                "probability": "sigmoid(raw score_delta), with no fit, scaling, clipping, or confirmation-set recalibration",
                "brier": "mean((probability - I[a_better])^2) over non-ties",
                "ece": "15 equal-width bins [j/15,(j+1)/15), final bin right-closed; sum n_bin/N * abs(mean_probability-mean_label)",
            },
            "strata": {
                "sources": ["base", "b20", "ALL"],
                "budgets": [0.10, 0.20],
                "length_bins_closed": [{"low": a, "high": b, "label": label} for a, b, label in LENGTH_BINS],
            },
            "per_protein": {
                "pair_metrics": "same definitions within protein/source, with undefined values retained as missing",
                "listwise_spearman": "average-rank Spearman of unique serialized candidate teacher utility versus scorer score",
                "ndcg": "NDCG@all on unique candidates using gain 2^utility-1 and log2(rank+1), deterministic candidate-ID tie-break",
                "top1_teacher_utility": "teacher utility for highest-scoring unique candidate, candidate-ID tie-break",
                "teacher_oracle_utility": "maximum serialized teacher utility among unique candidates",
                "oracle_gap": "teacher_oracle_utility - scorer_top1_teacher_utility",
                "candidate_scope": "all unique candidates appearing in legal confirmation pair rows; no proposal generation",
            },
        },
        "baselines": {
            "random": "exact DelePref-1 SHA-256 stable candidate score definition; no fitting",
            "joint_metadata": "exact DelePref-1 feature vocabulary, intercept, train-only means/scales/coefficients from frozen artifacts; no confirmation fitting",
            "development_values": "EXTERNAL_REFERENCE only and excluded from confirmation confidence intervals",
        },
        "statistics": {
            "outer_resampling_unit": "protein_id",
            "aggregation": "DelePref-1-compatible macro mean of finite per-protein metrics",
            "replicates": BOOTSTRAPS,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 2.5% and 97.5%",
            "pair_row_bootstrap_forbidden": True,
            "metric_specific_seed": "first 64 bits sha256('delepref-14-bootstrap\\0' + seed + '\\0' + method + '\\0' + source + '\\0' + metric)",
        },
        "hardware_gate": {"gpu_count": 2},
        "retry_policy": "one formal launcher; at most one corrected retry only after pre-model zero-scientific-row failure or packaging/validator-only failure; never alter protocol/sample/model/metrics",
        "code": {relative(path): sha256_file(path) for path in required_scripts},
        "required_outputs": [
            "protocol.json", "confirmation_open_record.json", "confirmation_manifest_audit.csv",
            "source_shard_audit.csv", "confirmation_pair_scores.csv", "confirmation_protein_metrics.csv",
            "confirmation_fidelity_summary.csv", "confirmation_stratified_metrics.csv",
            "confirmation_protein_bootstrap_ci.csv", "baseline_comparison.csv",
            "checkpoint_provenance.json", "resource_provenance.json", "independent_validation.json",
            "result_summary.md", "result_summary.json", "SUMMARY.md", "run_summary.json",
            "audit_manifest.json",
        ],
        "payload_markers": ["DelePref_POSTFREEZE_CONFIRMATION_START", "DelePref_POSTFREEZE_CONFIRMATION_END"],
    }
    protocol_path = OUT / "protocol.json"
    atomic_json(protocol_path, payload)
    protocol_hash = sha256_file(protocol_path)
    atomic_text(OUT / "protocol.sha256", protocol_hash + "  protocol.json\n")
    atomic_json(OUT / "protocol.done.json", {
        "status": "PROTOCOL_FROZEN_AND_FSYNCED",
        "protocol_sha256": protocol_hash,
        "frozen_at_utc": payload["frozen_at_utc"],
        "confirmation_manifest_parsed": False,
        "fsync_completed": True,
    })
    fsync_directory(OUT)
    print(json.dumps({"status": "PROTOCOL_FROZEN_AND_FSYNCED", "protocol_sha256": protocol_hash}, sort_keys=True))
    return 0


def assert_frozen_protocol() -> dict[str, Any]:
    protocol_path = OUT / "protocol.json"
    done_path = OUT / "protocol.done.json"
    if not protocol_path.is_file() or not done_path.is_file():
        raise RuntimeError("protocol and durable done sentinel must exist before confirmation open")
    done = read_json(done_path)
    actual = sha256_file(protocol_path)
    if done.get("status") != "PROTOCOL_FROZEN_AND_FSYNCED" or done.get("protocol_sha256") != actual:
        raise RuntimeError("protocol sentinel/hash mismatch")
    return {"payload": read_json(protocol_path), "sha256": actual}


def resolve_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO / path


def load_indexes() -> dict[str, dict[int, dict[str, Any]]]:
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for source, path in (("base", BASE_INDEX), ("b20", B20_INDEX)):
        entries: dict[int, dict[str, Any]] = {}
        for ordinal, row in enumerate(read_csv(path)):
            index = int(row.get("shard_index") or ordinal)
            if index in entries:
                raise RuntimeError("duplicate source shard index")
            entries[index] = {
                "pair": resolve_path(row["output_csv"]),
                "candidate": resolve_path(row["source_candidate_csv"]),
                "expected_pairs": int(row["num_pairs"]),
            }
        if set(entries) != set(range(300)):
            raise RuntimeError("{} index is not exactly shards 0..299".format(source))
        result[source] = entries
    return result


def expected_worker(protein: str, sequence_hash: str) -> int:
    material = "delepref-14-worker\0{}\0{}".format(protein, sequence_hash)
    return int(sha256_text(material)[:16], 16) % 2


def validate_pair_row(
    row: Mapping[str, str], source: str, shard_index: int, manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]]:
    protein = str(row.get("protein_id") or "")
    length = int(manifest["wildtype_length"])
    budget = SOURCES[source]
    if int(row.get("source_shard_index") or -1) != shard_index:
        raise ValueError("source_shard_index mismatch")
    if row.get("construction_policy") != "strict_same_deleted_length_pair":
        raise ValueError("construction policy is not strict same count")
    deleted = int(row["deleted_length"])
    if deleted != strict_delete_count(length, budget):
        raise ValueError("deleted length is not exact frozen budget count")
    if row.get("pair_budget_key") != "deleted_length:{}".format(deleted):
        raise ValueError("pair budget key mismatch")
    left = parse_intervals(row["deleted_intervals_a"], length)
    right = parse_intervals(row["deleted_intervals_b"], length)
    if canonical_intervals(left) != row["deleted_intervals_a"] or canonical_intervals(right) != row["deleted_intervals_b"]:
        raise ValueError("interval serialization is not canonical")
    if left == right:
        raise ValueError("identical pair masks")
    if sum(b - a + 1 for a, b in left) != deleted or sum(b - a + 1 for a, b in right) != deleted:
        raise ValueError("side deleted length mismatch")
    if int(row["num_intervals_a"]) != len(left) or int(row["num_intervals_b"]) != len(right):
        raise ValueError("side interval count mismatch")
    perturbation_a = row["perturbation_id_a"]
    perturbation_b = row["perturbation_id_b"]
    if not perturbation_a or not perturbation_b or perturbation_a >= perturbation_b:
        raise ValueError("pair perturbation ordering/identity mismatch")
    expected_id = "{}__pairdelepref__{}".format(protein, stable_hash(protein, perturbation_a, perturbation_b, PAIR_SEED)[:24])
    if row["pair_id"] != expected_id:
        raise ValueError("pair identity mismatch")
    utility_a, utility_b = float(row["utility_a"]), float(row["utility_b"])
    delta = utility_a - utility_b
    tie_margin = float(row["tie_margin"])
    expected_preference = "tie" if abs(delta) <= tie_margin else ("a_better" if delta > 0 else "b_better")
    if row["preference"] != expected_preference:
        raise ValueError("teacher preference label mismatch")
    if not close(row["utility_margin_signed"], delta) or not close(row["preference_margin"], abs(delta)):
        raise ValueError("teacher utility margin mismatch")
    confidence = float(row["preference_confidence"])
    weight = float(row["scorer_training_weight"])
    if not (0.0 <= confidence <= 1.0 and 0.0 <= weight <= 1.0 and tie_margin > 0.0):
        raise ValueError("confidence/weight/tie margin invalid")
    if row.get("preference") not in {"a_better", "b_better", "tie"}:
        raise ValueError("invalid preference")
    worker = expected_worker(protein, manifest["sequence_sha256"])
    compact = {column: row.get(column, "") for column in EXTRACTION_COLUMNS}
    compact.update({
        "worker_id": worker,
        "source": source,
        "evaluation_budget": budget,
        "sequence_sha256": manifest["sequence_sha256"],
        "wildtype_length": length,
        "length_bin": length_bin(length),
    })
    expected_candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for side, intervals, utility in (("a", left, utility_a), ("b", right, utility_b)):
        key = (protein, row["perturbation_id_" + side], row["plan_id_" + side])
        expected_candidates[key] = {
            "deleted_intervals": canonical_intervals(intervals),
            "deleted_length": deleted,
            "num_intervals": len(intervals),
            "deletion_fraction": deleted / float(length),
            "budget_bin": row["budget_bin_" + side],
            "generation_policy": row["generation_policy_" + side],
            "teacher_utility": utility,
        }
    return compact, expected_candidates


def validate_candidate_row(
    row: Mapping[str, str], manifest: Mapping[str, Any], expected: Mapping[str, Any] | None,
) -> str:
    sequence = str(row.get("wildtype_sequence") or "").strip().upper()
    length = int(row["wildtype_length"])
    if length != int(manifest["wildtype_length"]) or len(sequence) != length:
        raise ValueError("candidate WT length mismatch")
    sequence_hash = sha256_text(sequence)
    if sequence_hash != manifest["sequence_sha256"]:
        raise ValueError("candidate WT SHA-256 mismatch")
    stored = str(row.get("wildtype_sha256") or "").strip()
    if stored and stored != sequence_hash:
        raise ValueError("candidate stored WT SHA-256 mismatch")
    intervals = parse_intervals(row["deleted_intervals"], length)
    deleted = sum(b - a + 1 for a, b in intervals)
    if deleted != int(row["deleted_length"]) or len(intervals) != int(row["num_intervals"]):
        raise ValueError("candidate geometry mismatch")
    if str(row.get("coordinate_system") or "") != COORDINATE_SYSTEM:
        raise ValueError("candidate coordinate system mismatch")
    if not close(row["deletion_fraction"], deleted / float(length)):
        raise ValueError("candidate deletion fraction mismatch")
    if expected is not None:
        if canonical_intervals(intervals) != expected["deleted_intervals"]:
            raise ValueError("pair/candidate interval linkage mismatch")
        for field in ("deleted_length", "num_intervals"):
            if int(row[field]) != int(expected[field]):
                raise ValueError("pair/candidate {} mismatch".format(field))
        if not close(row["deletion_fraction"], expected["deletion_fraction"]):
            raise ValueError("pair/candidate deletion fraction mismatch")
        for field in ("budget_bin", "generation_policy"):
            if row[field] != expected[field]:
                raise ValueError("pair/candidate {} mismatch".format(field))
    return sequence


def open_and_extract_confirmation() -> dict[str, Any]:
    protocol = assert_frozen_protocol()
    if (OUT / "confirmation_open_record.json").exists():
        raise RuntimeError("one-time confirmation open record already exists; refusing a second scientific run")
    if sha256_file(CONFIRMATION) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("confirmation manifest changed after protocol freeze")
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint changed after protocol freeze")
    lock_hash_before = sha256_file(CONFIRMATION_LOCK)
    lock = read_json(CONFIRMATION_LOCK)  # First permitted parse: protocol is frozen.
    if lock.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("confirmation lock points to an unexpected manifest hash")
    first_opened_at = utc_now()
    manifest_rows = read_csv(CONFIRMATION)  # Irreversible first membership reveal.
    if len(manifest_rows) != EXPECTED_PROTEINS:
        raise RuntimeError("confirmation manifest protein count mismatch")
    manifest: dict[str, dict[str, Any]] = {}
    sequences = set()
    for row in manifest_rows:
        protein = row["protein_id"]
        if protein in manifest or row["sequence_sha256"] in sequences:
            raise RuntimeError("confirmation protein/sequence identity is not unique")
        if row.get("split") != "confirmation":
            raise RuntimeError("non-confirmation row in confirmation manifest")
        item = dict(row)
        item["wildtype_length"] = int(row["wildtype_length"])
        item["base_shard_index"] = int(row["base_shard_index"])
        item["b20_shard_index"] = int(row["b20_shard_index"])
        manifest[protein] = item
        sequences.add(row["sequence_sha256"])
    if len(sequences) != EXPECTED_SEQUENCE_GROUPS:
        raise RuntimeError("confirmation sequence-group count mismatch")
    open_record = {
        "schema_version": "delepref_14_confirmation_open_record_v1",
        "status": "DelePref_CONFIRMATION_SET_OPENED_ONCE",
        "logical_reveal_count": 1,
        "first_opened_at_utc": first_opened_at,
        "record_written_at_utc": utc_now(),
        "manifest_path": relative(CONFIRMATION),
        "manifest_sha256": sha256_file(CONFIRMATION),
        "manifest_lock_path": relative(CONFIRMATION_LOCK),
        "manifest_lock_sha256_before_open": lock_hash_before,
        "protocol_sha256": protocol["sha256"],
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "proteins": len(manifest),
        "sequence_groups": len(sequences),
        "opening_purpose": "one-time post-freeze confirmation evaluation only",
        "not_for_checkpoint_selection": True,
        "not_for_threshold_selection": True,
        "not_for_protocol_change": True,
        "original_lock_remains_byte_unchanged": True,
    }
    atomic_json(OUT / "confirmation_open_record.json", open_record)

    train_dev_rows = read_csv(TRAIN_DEV)
    train_ids = {row["protein_id"] for row in train_dev_rows}
    train_sequences = {row["sequence_sha256"] for row in train_dev_rows}
    if set(manifest) & train_ids or sequences & train_sequences:
        raise RuntimeError("confirmation overlaps train/dev split")

    indexes = load_indexes()
    expected_by_shard: dict[tuple[str, int], set[str]] = defaultdict(set)
    for protein, row in manifest.items():
        expected_by_shard[("base", row["base_shard_index"])].add(protein)
        expected_by_shard[("b20", row["b20_shard_index"])].add(protein)

    WORK.mkdir(parents=True, exist_ok=True)
    worker_inputs = [WORK / "worker_0_pairs.csv", WORK / "worker_1_pairs.csv"]
    for path in worker_inputs:
        if path.exists():
            raise FileExistsError("refusing to overwrite scientific worker input {}".format(path))
    worker_first = [True, True]
    source_audit: list[dict[str, Any]] = []
    pair_counts: dict[str, Counter[str]] = {"base": Counter(), "b20": Counter()}
    pair_rows_total = Counter()
    candidate_rows_total = Counter()
    wildtypes_by_source: dict[str, dict[str, str]] = {"base": {}, "b20": {}}
    started = time.time()
    for source in ("base", "b20"):
        for shard_index in range(300):
            entry = indexes[source][shard_index]
            expected_proteins = expected_by_shard[(source, shard_index)]
            shard_rows: list[dict[str, Any]] = []
            expected_candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
            pair_ids = set()
            observed_pair_rows = 0
            with entry["pair"].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    observed_pair_rows += 1
                    protein = row["protein_id"]
                    if protein not in expected_proteins:
                        continue
                    compact, side_expectations = validate_pair_row(row, source, shard_index, manifest[protein])
                    if compact["pair_id"] in pair_ids:
                        raise RuntimeError("duplicate confirmation pair identity within shard")
                    pair_ids.add(compact["pair_id"])
                    for key, expectation in side_expectations.items():
                        previous = expected_candidates.get(key)
                        if previous is not None and any(
                            previous[field] != expectation[field] for field in (
                                "deleted_intervals", "deleted_length", "num_intervals",
                                "budget_bin", "generation_policy",
                            )
                        ):
                            raise RuntimeError("candidate linkage changes across pair rows")
                        if previous is not None and not close(previous["teacher_utility"], expectation["teacher_utility"]):
                            raise RuntimeError("candidate teacher utility changes across pair rows")
                        expected_candidates[key] = expectation
                    shard_rows.append(compact)
                    pair_counts[source][protein] += 1
            if observed_pair_rows != int(entry["expected_pairs"]):
                raise RuntimeError("source pair shard row count mismatch")
            pair_rows_total[source] += observed_pair_rows

            found_candidates = set()
            found_proteins = set()
            observed_candidate_rows = 0
            with entry["candidate"].open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    observed_candidate_rows += 1
                    protein = row["protein_id"]
                    if protein not in expected_proteins:
                        continue
                    key = (protein, row["perturbation_id"], row["plan_id"])
                    sequence = validate_candidate_row(row, manifest[protein], expected_candidates.get(key))
                    previous_sequence = wildtypes_by_source[source].get(protein)
                    if previous_sequence is not None and previous_sequence != sequence:
                        raise RuntimeError("protein has inconsistent WT within source")
                    wildtypes_by_source[source][protein] = sequence
                    found_proteins.add(protein)
                    if key in expected_candidates:
                        found_candidates.add(key)
            candidate_rows_total[source] += observed_candidate_rows
            missing_proteins = expected_proteins - found_proteins
            missing_candidates = set(expected_candidates) - found_candidates
            if missing_proteins or missing_candidates:
                raise RuntimeError("source candidate linkage incomplete: proteins={} candidates={}".format(len(missing_proteins), len(missing_candidates)))

            by_worker = {0: [], 1: []}
            for row in shard_rows:
                by_worker[int(row["worker_id"])].append(row)
            for worker in (0, 1):
                if by_worker[worker]:
                    append_csv(worker_inputs[worker], EXTRACTION_COLUMNS, by_worker[worker], worker_first[worker])
                    worker_first[worker] = False
            source_audit.append({
                "source": source,
                "budget": SOURCES[source],
                "shard_index": shard_index,
                "pair_path": relative(entry["pair"]),
                "candidate_path": relative(entry["candidate"]),
                "pair_size_bytes": entry["pair"].stat().st_size,
                "candidate_size_bytes": entry["candidate"].stat().st_size,
                "index_expected_pair_rows": entry["expected_pairs"],
                "observed_pair_rows": observed_pair_rows,
                "confirmation_manifest_proteins": len(expected_proteins),
                "confirmation_pair_proteins": len({row["protein_id"] for row in shard_rows}),
                "confirmation_pair_rows": len(shard_rows),
                "confirmation_candidate_references": len(expected_candidates),
                "matched_candidate_references": len(found_candidates),
                "candidate_rows_scanned": observed_candidate_rows,
                "wt_proteins_resolved": len(found_proteins),
                "status": "PASS",
            })
            if (shard_index + 1) % 10 == 0:
                print(json.dumps({
                    "event": "confirmation_extraction_progress", "source": source,
                    "shards": shard_index + 1, "confirmation_pairs": sum(pair_counts[source].values()),
                    "elapsed_seconds": time.time() - started,
                }, sort_keys=True), flush=True)
    if any(worker_first):
        raise RuntimeError("one deterministic GPU worker received zero pairs")
    if set(wildtypes_by_source["base"]) != set(manifest) or set(wildtypes_by_source["b20"]) != set(manifest):
        raise RuntimeError("not all confirmation WT sequences were resolved from both sources")
    for protein in manifest:
        if wildtypes_by_source["base"][protein] != wildtypes_by_source["b20"][protein]:
            raise RuntimeError("base/B20 WT mismatch for confirmation protein")

    protein_rows = []
    for protein in sorted(manifest):
        row = manifest[protein]
        base_count = pair_counts["base"][protein]
        b20_count = pair_counts["b20"][protein]
        reasons = []
        if base_count == 0:
            reasons.append("no_legal_base_same_count_pairs")
        if b20_count == 0:
            reasons.append("no_legal_b20_same_count_pairs")
        protein_rows.append({
            **row,
            "worker_id": expected_worker(protein, row["sequence_sha256"]),
            "base_pair_count": base_count,
            "b20_pair_count": b20_count,
            "all_pair_count": base_count + b20_count,
            "base_wt_sha256_match": True,
            "b20_wt_sha256_match": True,
            "train_dev_protein_overlap": False,
            "train_dev_sequence_overlap": False,
            "missing_reason": ";".join(reasons),
            "status": "PASS" if base_count + b20_count else "MISSING_NO_LEGAL_PAIRS",
            "wildtype_sequence": wildtypes_by_source["base"][protein],
        })
    manifest_audit_columns = [
        "protein_id", "sequence_sha256", "wildtype_length", "split", "assignment_bucket",
        "assignment_digest", "base_shard_index", "b20_shard_index", "worker_id",
        "base_pair_count", "b20_pair_count", "all_pair_count", "base_wt_sha256_match",
        "b20_wt_sha256_match", "train_dev_protein_overlap", "train_dev_sequence_overlap",
        "missing_reason", "status",
    ]
    write_csv(OUT / "confirmation_manifest_audit.csv", manifest_audit_columns, protein_rows)
    write_csv(WORK / "confirmation_proteins_with_sequences.csv", manifest_audit_columns + ["wildtype_sequence"], protein_rows)
    write_csv(OUT / "source_shard_audit.csv", list(source_audit[0]), source_audit)
    lock_hash_after = sha256_file(CONFIRMATION_LOCK)
    if lock_hash_after != lock_hash_before:
        raise RuntimeError("original confirmation lock changed during opening/extraction")
    extraction_summary = {
        "status": "PASS",
        "finished_at_utc": utc_now(),
        "protocol_sha256": protocol["sha256"],
        "manifest_sha256": sha256_file(CONFIRMATION),
        "manifest_lock_sha256_before": lock_hash_before,
        "manifest_lock_sha256_after": lock_hash_after,
        "proteins": len(manifest),
        "sequence_groups": len(sequences),
        "pair_rows": dict(pair_rows_total),
        "confirmation_pair_rows": {source: sum(pair_counts[source].values()) for source in SOURCES},
        "confirmation_pair_proteins": {source: sum(pair_counts[source][protein] > 0 for protein in manifest) for source in SOURCES},
        "candidate_rows_scanned": dict(candidate_rows_total),
        "worker_pair_rows": {
            str(worker): sum(1 for _ in csv.DictReader(worker_inputs[worker].open("r", encoding="utf-8", newline="")))
            for worker in (0, 1)
        },
        "train_dev_protein_overlap": 0,
        "train_dev_sequence_overlap": 0,
        "all_15053_proteins_retained": True,
    }
    atomic_json(AUDIT / "extraction_summary.json", extraction_summary)
    return extraction_summary


NUMERIC_METADATA = [
    "span_count", "mean_span_length", "max_span_length", "min_span_length",
    "terminal_span_fraction", "internal_span_fraction", "is_terminal_any",
    "is_terminal_only", "is_multi_span", "relative_start", "relative_end",
    "relative_centroid", "relative_max_gap", "relative_mean_gap", "fragmentation_index",
]


def side_geometry(row: Mapping[str, Any], side: str) -> dict[str, Any]:
    length = int(row["wildtype_length"])
    intervals = parse_intervals(row["deleted_intervals_" + side], length)
    spans = [end - start + 1 for start, end in intervals]
    positions = [position for start, end in intervals for position in range(start, end + 1)]
    terminals = sum(int(start == 0 or end == length - 1) for start, end in intervals)
    gaps = [intervals[index + 1][0] - intervals[index][1] - 1 for index in range(len(intervals) - 1)]
    return {
        "span_count": float(len(intervals)),
        "mean_span_length": statistics.mean(spans),
        "max_span_length": float(max(spans)),
        "min_span_length": float(min(spans)),
        "terminal_span_fraction": terminals / float(len(intervals)),
        "internal_span_fraction": (len(intervals) - terminals) / float(len(intervals)),
        "is_terminal_any": float(terminals > 0),
        "is_terminal_only": float(terminals == len(intervals)),
        "is_multi_span": float(len(intervals) > 1),
        "relative_start": intervals[0][0] / float(max(1, length - 1)),
        "relative_end": intervals[-1][1] / float(max(1, length - 1)),
        "relative_centroid": statistics.mean(positions) / float(max(1, length - 1)),
        "relative_max_gap": max(gaps) / float(length) if gaps else 0.0,
        "relative_mean_gap": statistics.mean(gaps) / float(length) if gaps else 0.0,
        "fragmentation_index": (len(intervals) - 1) / float(max(1, int(row["deleted_length"]) - 1)),
        "generation_policy": str(row["generation_policy_" + side]),
        "budget_bin": str(row["budget_bin_" + side]),
    }


def metadata_feature_map(row: Mapping[str, Any]) -> dict[str, float]:
    left, right = side_geometry(row, "a"), side_geometry(row, "b")
    features = {"delta_" + name: float(left[name]) - float(right[name]) for name in NUMERIC_METADATA}
    for category in ("generation_policy", "budget_bin"):
        for value in {str(left[category]), str(right[category])}:
            features["delta_onehot_{}={}".format(category, value)] = float(left[category] == value) - float(right[category] == value)
    return features


def load_frozen_metadata_model() -> tuple[float, dict[str, tuple[float, float, float]]]:
    audit = read_json(DelePref_1_METADATA_AUDIT)
    coefficients = {}
    for row in read_csv(DelePref_1_COEFFICIENTS):
        coefficients[row["feature"]] = (
            float(row["mean"]), float(row["scale"]), float(row["standardized_coefficient"]),
        )
    if set(coefficients) != set(audit["features"]):
        raise RuntimeError("frozen metadata coefficient feature vocabulary mismatch")
    return float(audit["intercept"]), coefficients


def metadata_score(row: Mapping[str, Any], model: tuple[float, Mapping[str, tuple[float, float, float]]]) -> float:
    intercept, coefficients = model
    features = metadata_feature_map(row)
    return intercept + sum(coef * (features.get(name, 0.0) - mean) / scale for name, (mean, scale, coef) in coefficients.items())


def stable_random_candidate_score(row: Mapping[str, Any], side: str) -> float:
    identity = "delepref-exp01-random\0{}\0{:.2f}\0{}\0{}".format(
        row["protein_id"], float(row["evaluation_budget"]),
        row["perturbation_id_" + side], row["deleted_intervals_" + side],
    )
    return int(sha256_text(identity)[:16], 16) / float(16**16 - 1)


def predicted_preference(delta: float) -> str:
    return "a_better" if delta > 0 else ("b_better" if delta < 0 else "tie")


def candidate_for_row(row: Mapping[str, Any], side: str, sequence: str) -> DeletionCandidate:
    intervals = parse_intervals(row["deleted_intervals_" + side], len(sequence))
    deleted = sum(end - start + 1 for start, end in intervals)
    return DeletionCandidate(
        protein_id=str(row["protein_id"]),
        perturbation_id=str(row["perturbation_id_" + side]),
        plan_id=str(row["plan_id_" + side]),
        wildtype_sequence=sequence,
        deleted_intervals=intervals,
        deleted_length=deleted,
        deletion_fraction=deleted / float(len(sequence)),
        target_delete_fraction=float(row["evaluation_budget"]),
        budget_bin="delepref_confirmation_b{:02d}".format(int(round(float(row["evaluation_budget"]) * 100))),
        generation_policy=str(row["generation_policy_" + side]),
    )


def run_worker(worker_id: int) -> int:
    import resource
    import torch

    assert_frozen_protocol()
    if worker_id not in (0, 1):
        raise ValueError("worker ID must be 0 or 1")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_available():
        raise RuntimeError("each worker must see exactly one CUDA GPU")
    input_path = WORK / "worker_{}_pairs.csv".format(worker_id)
    output_path = WORK / "worker_{}_scores.csv".format(worker_id)
    summary_path = WORK / "worker_{}_summary.json".format(worker_id)
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite worker scientific results")
    checkpoint_before = sha256_file(CHECKPOINT)
    if checkpoint_before != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("worker checkpoint hash mismatch")
    protein_rows = read_csv(WORK / "confirmation_proteins_with_sequences.csv")
    sequence_by_protein = {
        row["protein_id"]: row["wildtype_sequence"] for row in protein_rows if int(row["worker_id"]) == worker_id
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            protein = row["protein_id"]
            if int(row["worker_id"]) != worker_id or protein not in sequence_by_protein:
                raise RuntimeError("worker input partition mismatch")
            grouped[protein].append(dict(row))
    proteins = sorted(grouped)
    if not proteins:
        raise RuntimeError("worker has no proteins")
    metadata_model = load_frozen_metadata_model()
    started_at = utc_now()
    started = time.time()
    runtime = load_runtime(CHECKPOINT, config_path=CONFIG, device="cuda", k=8)
    model_loaded_at = utc_now()
    properties = torch.cuda.get_device_properties(0)
    device_uuid = str(getattr(properties, "uuid", ""))
    if device_uuid and not device_uuid.startswith("GPU-"):
        device_uuid = "GPU-" + device_uuid
    if not properties.name:
        raise RuntimeError("worker has no visible CUDA device")
    first = True
    scored_pairs = unique_candidates_total = 0
    chunk_size = 24
    torch.cuda.reset_peak_memory_stats()
    for chunk_start in range(0, len(proteins), chunk_size):
        chunk_proteins = proteins[chunk_start:chunk_start + chunk_size]
        encoded = runtime.model.encode_unique_wildtypes(
            [(protein, sequence_by_protein[protein]) for protein in chunk_proteins],
            microbatch_size=runtime.encoder_microbatch_size,
        )
        unique: dict[tuple[str, float, str], DeletionCandidate] = {}
        row_keys: dict[tuple[str, int, str], tuple[str, float, str]] = {}
        chunk_rows: list[dict[str, Any]] = []
        offsets: dict[str, int] = defaultdict(int)
        for protein in chunk_proteins:
            sequence = sequence_by_protein[protein]
            for row in grouped[protein]:
                item = dict(row)
                item["random_score_a"] = stable_random_candidate_score(item, "a")
                item["random_score_b"] = stable_random_candidate_score(item, "b")
                item["random_score_delta"] = item["random_score_a"] - item["random_score_b"]
                item["metadata_score_delta"] = metadata_score(item, metadata_model)
                row_index = offsets[protein]
                offsets[protein] += 1
                for side in ("a", "b"):
                    key = (protein, float(item["evaluation_budget"]), item["deleted_intervals_" + side])
                    row_keys[(protein, row_index, side)] = key
                    if key not in unique:
                        unique[key] = candidate_for_row(item, side, sequence)
                item["_row_index"] = row_index
                chunk_rows.append(item)
        ordered_keys = sorted(unique)
        candidates = [unique[key] for key in ordered_keys]
        with torch.no_grad():
            values = runtime.model.score_plans_from_embeddings(candidates, encoded)["preference_score"]
        scores = {key: float(value) for key, value in zip(ordered_keys, values.detach().float().cpu().tolist())}
        for item in chunk_rows:
            protein, row_index = item["protein_id"], int(item.pop("_row_index"))
            item["score_a"] = scores[row_keys[(protein, row_index, "a")]]
            item["score_b"] = scores[row_keys[(protein, row_index, "b")]]
            item["score_delta"] = item["score_a"] - item["score_b"]
            item["student_predicted_preference"] = predicted_preference(item["score_delta"])
            item["random_predicted_preference"] = predicted_preference(float(item["random_score_delta"]))
            item["metadata_predicted_preference"] = predicted_preference(float(item["metadata_score_delta"]))
            item["metric_scope"] = "teacher_preference_fidelity_not_biological_ground_truth"
        append_csv(output_path, PAIR_COLUMNS, chunk_rows, first)
        first = False
        scored_pairs += len(chunk_rows)
        unique_candidates_total += len(candidates)
        del encoded, candidates, values, scores, unique, chunk_rows
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({
            "event": "delepref_14_worker_progress", "worker_id": worker_id,
            "proteins_scored": min(len(proteins), chunk_start + chunk_size),
            "proteins_total": len(proteins), "pairs_scored": scored_pairs,
            "elapsed_seconds": time.time() - started,
        }, sort_keys=True), flush=True)
    checkpoint_after = sha256_file(CHECKPOINT)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("checkpoint changed during worker inference")
    summary = {
        "status": "PASS",
        "worker_id": worker_id,
        "started_at_utc": started_at,
        "model_loaded_at_utc": model_loaded_at,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "proteins": len(proteins),
        "pairs": scored_pairs,
        "unique_candidate_scores": unique_candidates_total,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "source_checkpoint_sha256": runtime.source_checkpoint_sha256,
        "selected_step": runtime.selected_step,
        "shared_encoder_identity": runtime.model.scorer.encoder is runtime.model.encoder and runtime.model.policy.encoder is runtime.model.encoder,
        "policy_invoked": False,
        "teacher_invoked": False,
        "training_invoked": False,
        "torch_visible_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "device_uuid": device_uuid,
        "device_total_memory_bytes": properties.total_memory,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    if not summary["shared_encoder_identity"]:
        raise RuntimeError("DelePref encoder identity is not shared")
    atomic_json(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


def launch_workers() -> list[dict[str, Any]]:
    python = sys.executable
    processes = []
    logs = []
    for worker in (0, 1):
        log_path = ROUTING / "worker_{}.log".format(worker)
        log_handle = log_path.open("w", encoding="utf-8")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(worker)
        process = subprocess.Popen(
            [python, str(Path(__file__).resolve()), "--worker", str(worker)],
            cwd=str(REPO), env=environment, stdout=log_handle, stderr=subprocess.STDOUT,
        )
        processes.append((worker, process, log_handle))
        logs.append(log_path)
    failures = []
    for worker, process, handle in processes:
        returncode = process.wait()
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        if returncode:
            failures.append({"worker": worker, "returncode": returncode, "log": relative(logs[worker])})
    if failures:
        raise RuntimeError("scientific GPU worker failure: {}".format(failures))
    return [read_json(WORK / "worker_{}_summary.json".format(worker)) for worker in (0, 1)]


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
    xmean, ymean = statistics.mean(xs), statistics.mean(ys)
    dx, dy = [x - xmean for x in xs], [y - ymean for y in ys]
    denominator = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / denominator if denominator > 0 else math.nan


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(average_ranks(xs), average_ranks(ys)) if len(xs) >= 2 else math.nan


def rank_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = average_ranks(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (rank_sum - positives * (positives + 1) / 2.0) / float(positives * negatives)


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def ece(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    if not labels:
        return math.nan
    total = len(labels)
    value = 0.0
    for index in range(ECE_BINS):
        low, high = index / float(ECE_BINS), (index + 1) / float(ECE_BINS)
        selected = [offset for offset, probability in enumerate(probabilities) if probability >= low and (probability < high or (index == ECE_BINS - 1 and probability <= high))]
        if selected:
            confidence = statistics.mean(probabilities[offset] for offset in selected)
            accuracy = statistics.mean(labels[offset] for offset in selected)
            value += len(selected) / float(total) * abs(confidence - accuracy)
    return value


def metric_for_rows(rows: Sequence[Mapping[str, Any]], score_field: str) -> dict[str, Any]:
    labels, scores, teacher, weights = [], [], [], []
    correct, high_correct, high_total = [], 0, 0
    ties = 0
    for row in rows:
        preference = str(row["preference"])
        if preference == "tie":
            ties += 1
            continue
        if preference not in {"a_better", "b_better"}:
            continue
        score = float(row[score_field])
        label = int(preference == "a_better")
        ok = int((score > 0) == bool(label))
        labels.append(label)
        scores.append(score)
        teacher.append(float(row["utility_margin_signed"]))
        weights.append(float(row["preference_confidence"]))
        correct.append(ok)
        if float(row["preference_confidence"]) >= HIGH_CONFIDENCE_THRESHOLD:
            high_total += 1
            high_correct += ok
    probabilities = [sigmoid(score) for score in scores]
    return {
        "num_pairs": len(rows),
        "num_non_tie_pairs": len(labels),
        "num_teacher_ties": ties,
        "non_tie_accuracy": statistics.mean(correct) if correct else math.nan,
        "roc_auc": rank_auc(labels, scores),
        "weighted_accuracy": sum(weight * ok for weight, ok in zip(weights, correct)) / sum(weights) if sum(weights) > 0 else math.nan,
        "high_confidence_accuracy": high_correct / float(high_total) if high_total else math.nan,
        "high_confidence_pairs": high_total,
        "score_margin_pearson": pearson(teacher, scores),
        "score_margin_spearman": spearman(teacher, scores),
        "brier_score": statistics.mean((probability - label) ** 2 for probability, label in zip(probabilities, labels)) if labels else math.nan,
        "ece_15": ece(labels, probabilities),
        "tie_rate": ties / float(len(rows)) if rows else math.nan,
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


def protein_metric_row(protein: str, source: str, rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, str]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "protein_id": protein,
        "sequence_sha256": manifest["sequence_sha256"],
        "wildtype_length": int(manifest["wildtype_length"]),
        "length_bin": length_bin(int(manifest["wildtype_length"])),
        "source": source,
        "evaluation_budget": "ALL" if source == "ALL" else SOURCES[source],
        "status": "PASS" if rows else "MISSING_NO_LEGAL_PAIRS",
        "missing_reason": "" if rows else "no_legal_{}_pairs".format(source.lower()),
    }
    methods = {
        "delepref": "score_delta",
        "random_score": "random_score_delta",
        "joint_metadata": "metadata_score_delta",
    }
    for method, field in methods.items():
        metrics = metric_for_rows(rows, field) if rows else {
            name: (0 if name in {"num_pairs", "num_non_tie_pairs", "num_teacher_ties", "high_confidence_pairs"} else math.nan)
            for name in (
                "num_pairs", "num_non_tie_pairs", "num_teacher_ties", "non_tie_accuracy", "roc_auc",
                "weighted_accuracy", "high_confidence_accuracy", "high_confidence_pairs",
                "score_margin_pearson", "score_margin_spearman", "brier_score", "ece_15", "tie_rate",
            )
        }
        for name, value in metrics.items():
            base["{}_{}".format(method, name)] = value
    candidates: dict[str, tuple[float, float]] = {}
    for row in rows:
        for side in ("a", "b"):
            identity = "{}\0{}\0{}".format(row["perturbation_id_" + side], row["plan_id_" + side], row["deleted_intervals_" + side])
            value = (float(row["utility_" + side]), float(row["score_" + side]))
            previous = candidates.get(identity)
            if previous is not None and (not close(previous[0], value[0], 1.0e-7) or not close(previous[1], value[1], 1.0e-6)):
                raise RuntimeError("candidate utility/score inconsistent across protein pairs")
            candidates[identity] = value
    identities = sorted(candidates)
    utilities = [candidates[identity][0] for identity in identities]
    scores = [candidates[identity][1] for identity in identities]
    base["unique_candidate_count"] = len(identities)
    base["delepref_listwise_spearman"] = spearman(utilities, scores)
    base["delepref_ndcg"] = ndcg(utilities, scores, identities)
    if identities:
        top = min(range(len(identities)), key=lambda index: (-scores[index], identities[index]))
        oracle = min(range(len(identities)), key=lambda index: (-utilities[index], identities[index]))
        base["scorer_top1_teacher_utility"] = utilities[top]
        base["teacher_oracle_utility"] = utilities[oracle]
        base["oracle_gap"] = utilities[oracle] - utilities[top]
        base["top1_oracle_agreement"] = float(top == oracle)
    else:
        for field in ("scorer_top1_teacher_utility", "teacher_oracle_utility", "oracle_gap", "top1_oracle_agreement"):
            base[field] = math.nan
    return base


def finalize_scientific_results(worker_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    score_path = OUT / "confirmation_pair_scores.csv"
    if score_path.exists():
        raise FileExistsError("refusing to overwrite merged scientific pair scores")
    with score_path.open("wb") as destination:
        for worker in (0, 1):
            source_path = WORK / "worker_{}_scores.csv".format(worker)
            with source_path.open("rb") as source:
                if worker == 1:
                    source.readline()
                shutil.copyfileobj(source, destination, 16 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    fsync_directory(OUT)

    manifest_rows = {row["protein_id"]: row for row in read_csv(OUT / "confirmation_manifest_audit.csv")}
    score_rows = read_csv(score_path)
    expected_pairs = sum(int(row["all_pair_count"]) for row in manifest_rows.values())
    if len(score_rows) != expected_pairs or len(score_rows) != sum(int(item["pairs"]) for item in worker_summaries):
        raise RuntimeError("merged scored pair count mismatch")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in score_rows:
        grouped[row["protein_id"]].append(row)
    protein_metrics = []
    for protein in sorted(manifest_rows):
        rows = grouped.get(protein, [])
        by_source = {source: [row for row in rows if row["source"] == source] for source in SOURCES}
        for source in ("base", "b20", "ALL"):
            subset = rows if source == "ALL" else by_source[source]
            protein_metrics.append(protein_metric_row(protein, source, subset, manifest_rows[protein]))
    columns = sorted({key for row in protein_metrics for key in row})
    write_csv(OUT / "confirmation_protein_metrics.csv", columns, protein_metrics)

    methods = {"delepref": "score_delta", "random_score": "random_score_delta", "joint_metadata": "metadata_score_delta"}
    fidelity_rows = []
    metric_names = [
        "non_tie_accuracy", "roc_auc", "weighted_accuracy", "high_confidence_accuracy",
        "score_margin_pearson", "score_margin_spearman", "brier_score", "ece_15", "tie_rate",
    ]
    for method, field in methods.items():
        for source in ("base", "b20", "ALL"):
            subset = score_rows if source == "ALL" else [row for row in score_rows if row["source"] == source]
            metrics = metric_for_rows(subset, field)
            for metric in metric_names:
                fidelity_rows.append({
                    "method": method, "source": source, "metric": metric, "value": metrics[metric],
                    "pair_count": metrics["num_pairs"], "non_tie_pair_count": metrics["num_non_tie_pairs"],
                    "high_confidence_pair_count": metrics["high_confidence_pairs"],
                    "protein_count": len({row["protein_id"] for row in subset}),
                    "scope": "confirmation_teacher_preference_fidelity",
                })
    ranking_fields = [
        "delepref_listwise_spearman", "delepref_ndcg", "scorer_top1_teacher_utility",
        "teacher_oracle_utility", "oracle_gap", "top1_oracle_agreement",
    ]
    for source in ("base", "b20", "ALL"):
        rows = [row for row in protein_metrics if row["source"] == source]
        for metric in ranking_fields:
            values = [float(row[metric]) for row in rows if math.isfinite(parse_float(row[metric]))]
            fidelity_rows.append({
                "method": "delepref", "source": source, "metric": metric,
                "value": statistics.mean(values) if values else math.nan,
                "pair_count": sum(int(float(row["delepref_num_pairs"])) for row in rows),
                "non_tie_pair_count": sum(int(float(row["delepref_num_non_tie_pairs"])) for row in rows),
                "high_confidence_pair_count": sum(int(float(row["delepref_high_confidence_pairs"])) for row in rows),
                "protein_count": len(values), "scope": "confirmation_unique_pair_candidate_ranking",
            })
    write_csv(OUT / "confirmation_fidelity_summary.csv", list(fidelity_rows[0]), fidelity_rows)

    stratified = []
    strata: list[tuple[str, str, list[dict[str, str]]]] = []
    for source in ("base", "b20"):
        strata.append(("budget", "{:.2f}".format(SOURCES[source]), [row for row in score_rows if row["source"] == source]))
    for _low, _high, label in LENGTH_BINS:
        strata.append(("length_bin", label, [row for row in score_rows if row["length_bin"] == label]))
    for stratum_type, stratum_value, rows in strata:
        for method, field in methods.items():
            for source in ("base", "b20", "ALL"):
                subset = rows if source == "ALL" else [row for row in rows if row["source"] == source]
                if not subset:
                    continue
                metrics = metric_for_rows(subset, field)
                for metric in metric_names:
                    stratified.append({
                        "method": method, "source": source, "stratum_type": stratum_type,
                        "stratum_value": stratum_value, "metric": metric, "value": metrics[metric],
                        "pair_count": metrics["num_pairs"], "non_tie_pair_count": metrics["num_non_tie_pairs"],
                        "protein_count": len({row["protein_id"] for row in subset}),
                    })
    write_csv(OUT / "confirmation_stratified_metrics.csv", list(stratified[0]), stratified)

    bootstrap_rows = []
    bootstrap_fields = {
        "delepref": [
            "non_tie_accuracy", "roc_auc", "weighted_accuracy", "high_confidence_accuracy",
            "score_margin_pearson", "score_margin_spearman", "brier_score", "ece_15",
            "listwise_spearman", "ndcg",
        ],
        "random_score": [
            "non_tie_accuracy", "roc_auc", "weighted_accuracy", "high_confidence_accuracy",
            "score_margin_pearson", "score_margin_spearman", "brier_score", "ece_15",
        ],
        "joint_metadata": [
            "non_tie_accuracy", "roc_auc", "weighted_accuracy", "high_confidence_accuracy",
            "score_margin_pearson", "score_margin_spearman", "brier_score", "ece_15",
        ],
    }
    for method, metrics in bootstrap_fields.items():
        for source in ("base", "b20", "ALL"):
            source_rows = [row for row in protein_metrics if row["source"] == source]
            for metric in metrics:
                field = "{}_{}".format(method, metric)
                values = [float(row[field]) for row in source_rows if math.isfinite(parse_float(row.get(field)))]
                if not values:
                    continue
                seed = int(sha256_text("delepref-14-bootstrap\0{}\0{}\0{}\0{}".format(BOOTSTRAP_SEED, method, source, metric))[:16], 16)
                import numpy as np
                rng = np.random.default_rng(seed)
                array = np.asarray(values, dtype=np.float64)
                samples = np.empty(BOOTSTRAPS, dtype=np.float64)
                for start in range(0, BOOTSTRAPS, 128):
                    end = min(BOOTSTRAPS, start + 128)
                    indices = rng.integers(0, len(array), size=(end - start, len(array)))
                    samples[start:end] = array[indices].mean(axis=1)
                bootstrap_rows.append({
                    "method": method, "source": source, "metric": metric,
                    "protein_count": len(values), "bootstrap_n": BOOTSTRAPS,
                    "base_seed": BOOTSTRAP_SEED, "metric_seed": seed,
                    "observed_macro_mean": float(array.mean()),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "resample_unit": "protein_id", "aggregation": "macro_mean_finite_protein_metrics",
                })
    for metric in ("scorer_top1_teacher_utility", "teacher_oracle_utility", "oracle_gap", "top1_oracle_agreement"):
        for source in ("base", "b20", "ALL"):
            source_rows = [row for row in protein_metrics if row["source"] == source]
            values = [float(row[metric]) for row in source_rows if math.isfinite(parse_float(row.get(metric)))]
            if not values:
                continue
            import numpy as np
            seed = int(sha256_text("delepref-14-bootstrap\0{}\0delepref\0{}\0{}".format(BOOTSTRAP_SEED, source, metric))[:16], 16)
            rng = np.random.default_rng(seed)
            array = np.asarray(values, dtype=np.float64)
            samples = np.empty(BOOTSTRAPS, dtype=np.float64)
            for start in range(0, BOOTSTRAPS, 128):
                end = min(BOOTSTRAPS, start + 128)
                samples[start:end] = array[rng.integers(0, len(array), size=(end - start, len(array)))].mean(axis=1)
            bootstrap_rows.append({
                "method": "delepref", "source": source, "metric": metric,
                "protein_count": len(values), "bootstrap_n": BOOTSTRAPS, "base_seed": BOOTSTRAP_SEED,
                "metric_seed": seed, "observed_macro_mean": float(array.mean()),
                "ci95_low": float(np.quantile(samples, 0.025)), "ci95_high": float(np.quantile(samples, 0.975)),
                "resample_unit": "protein_id", "aggregation": "macro_mean_finite_protein_metrics",
            })
    write_csv(OUT / "confirmation_protein_bootstrap_ci.csv", list(bootstrap_rows[0]), bootstrap_rows)

    fidelity_lookup = {(row["method"], row["source"], row["metric"]): row for row in fidelity_rows}
    baseline_rows = []
    for source in ("base", "b20", "ALL"):
        for metric in metric_names:
            student = float(fidelity_lookup[("delepref", source, metric)]["value"])
            for baseline in ("random_score", "joint_metadata"):
                value = float(fidelity_lookup[(baseline, source, metric)]["value"])
                baseline_rows.append({
                    "comparison_type": "CONFIRMATION", "source": source, "metric": metric,
                    "method": "delepref", "baseline": baseline, "method_value": student,
                    "baseline_value": value, "method_minus_baseline": student - value,
                    "fit_on_confirmation": False, "included_in_confirmation_ci": True,
                })
    for row in read_csv(DelePref_1_METRICS):
        for metric in metric_names:
            if metric not in row or row[metric] == "":
                continue
            baseline_rows.append({
                "comparison_type": "EXTERNAL_REFERENCE", "source": row["source"], "metric": metric,
                "method": row["method"], "baseline": "DelePref_1_DEVELOPMENT_REFERENCE",
                "method_value": row[metric], "baseline_value": "", "method_minus_baseline": "",
                "fit_on_confirmation": False, "included_in_confirmation_ci": False,
            })
    write_csv(OUT / "baseline_comparison.csv", list(baseline_rows[0]), baseline_rows)

    extraction = read_json(AUDIT / "extraction_summary.json")
    checkpoint_provenance = {
        "schema_version": "delepref_14_checkpoint_provenance_v1", "status": "PASS",
        "evaluation_checkpoint": relative(CHECKPOINT),
        "evaluation_checkpoint_sha256_before": EXPECTED_CHECKPOINT_SHA256,
        "evaluation_checkpoint_sha256_after": sha256_file(CHECKPOINT),
        "selected_step": 4000, "config_path": relative(CONFIG), "config_sha256": sha256_file(CONFIG),
        "source_checkpoint_sha256": worker_summaries[0]["source_checkpoint_sha256"],
        "workers_agree_source_checkpoint": len({row["source_checkpoint_sha256"] for row in worker_summaries}) == 1,
        "training_or_finetuning_run": False, "checkpoint_selection_run": False,
        "proposal_generation_run": False, "teacher_rescoring_run": False,
    }
    atomic_json(OUT / "checkpoint_provenance.json", checkpoint_provenance)
    gate = read_json(AUDIT / "resource_gate.json")
    atomic_json(OUT / "resource_provenance.json", {
        "schema_version": "delepref_14_resource_provenance_v1", "status": "PASS",
        "pre_execution_gate": gate, "workers": list(worker_summaries),
        "deterministic_partition": "sha256(delepref-14-worker NUL protein_id NUL sequence_sha256) modulo 2",
        "worker_pair_rows": extraction["worker_pair_rows"],
    })
    scientific_summary = {
        "status": "SCIENTIFIC_RESULTS_COMPLETE_PENDING_INDEPENDENT_VALIDATION",
        "finished_at_utc": utc_now(), "confirmation_proteins": EXPECTED_PROTEINS,
        "confirmation_sequence_groups": EXPECTED_SEQUENCE_GROUPS, "pair_rows": len(score_rows),
        "source_pair_rows": {source: sum(row["source"] == source for row in score_rows) for source in SOURCES},
        "source_valid_proteins": {source: len({row["protein_id"] for row in score_rows if row["source"] == source}) for source in SOURCES},
        "worker_summaries": list(worker_summaries), "negative_results_hidden": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    atomic_json(AUDIT / "scientific_summary.json", scientific_summary)
    return scientific_summary


def resource_gate() -> int:
    import torch

    protocol = assert_frozen_protocol()
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    gpu_rows = []
    for line in query.stdout.strip().splitlines():
        fields = [piece.strip() for piece in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError("unexpected nvidia-smi query schema")
        gpu_rows.append({
            "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
            "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]),
            "utilization_gpu_percent": int(fields[5]),
        })
    compute = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    compute_processes = [line.strip() for line in compute.stdout.splitlines() if line.strip() and "No running" not in line]
    torch_devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        uuid = str(getattr(properties, "uuid", ""))
        if uuid and not uuid.startswith("GPU-"):
            uuid = "GPU-" + uuid
        torch_devices.append({
            "index": index, "name": properties.name, "uuid": uuid,
            "total_memory_bytes": properties.total_memory,
        })
    payload = {
        "schema_version": "delepref_14_resource_gate_v1", "checked_at_utc": utc_now(),
        "protocol_sha256": protocol["sha256"], "hostname": socket.gethostname(),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "python": sys.executable, "python_version": sys.version,
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(), "torch_visible_count": torch.cuda.device_count(),
        "torch_devices": torch_devices, "nvidia_smi_gpus": gpu_rows,
        "compute_processes": compute_processes,
        "checks": {
            "two_nvidia_gpus": len(gpu_rows) == 2,
            "two_torch_gpus": torch.cuda.device_count() == 2,
            "gpu_names_recorded": len(gpu_rows) == 2 and all(bool(row["name"]) for row in gpu_rows),
        },
    }
    payload["status"] = "PASS" if all(payload["checks"].values()) else "FAIL"
    atomic_json(AUDIT / "resource_gate.json", payload)
    print(json.dumps(payload, sort_keys=True))
    if payload["status"] != "PASS":
        raise RuntimeError("designated pane/GPU resource gate failed")
    return 0


def run_all() -> int:
    assert_frozen_protocol()
    gate = read_json(AUDIT / "resource_gate.json")
    if gate.get("status") != "PASS":
        raise RuntimeError("resource gate must pass before formal launcher run")
    extraction = open_and_extract_confirmation()
    workers = launch_workers()
    scientific = finalize_scientific_results(workers)
    validator = subprocess.run([sys.executable, str(VALIDATOR)], cwd=str(REPO))
    if validator.returncode:
        raise RuntimeError("independent validation/packaging failed with code {}".format(validator.returncode))
    print(json.dumps({
        "status": "PASS_DelePref_14_CONFIRMATION_EVALUATION",
        "opening": "DelePref_CONFIRMATION_SET_OPENED_ONCE",
        "extraction": extraction, "scientific": scientific,
    }, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze-protocol", action="store_true")
    modes.add_argument("--resource-gate", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--worker", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.freeze_protocol:
        return freeze_protocol()
    if args.resource_gate:
        return resource_gate()
    if args.worker is not None:
        return run_worker(args.worker)
    return run_all()


if __name__ == "__main__":
    raise SystemExit(main())
