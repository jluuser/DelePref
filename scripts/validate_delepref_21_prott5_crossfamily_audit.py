#!/usr/bin/env python3
"""Independent CPU-only validator for DelePref-21.

This file intentionally does not import the DelePref-21 runner, Torch, Transformers,
or a ProtT5 model.  It independently reconstructs identities, tokenization,
representation views, ranks, tie-aware selections, pair orders, and all
protein bootstraps from sealed disk artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import py_compile
import re
import socket
import statistics
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/experiments/delepref_step004000_delepref_21_prott5_crossfamily_audit"
V20 = REPO / "results/experiments/delepref_step004000_delepref_20_direct_plm_cross_teacher"
PROTOCOL_SOURCE = Path(__file__).resolve()
RUNNER = REPO / "scripts/run_delepref_21_prott5_crossfamily_audit.py"
VALIDATOR = Path(__file__).resolve()
BOOTSTRAP = REPO / "scripts/bootstrap_prott5.py"
VENDOR_SITE = Path("models/prott5_xl_half_uniref50_enc/runtime_vendor/site-packages")
PASS_SENTINEL = "PASS_DelePref_21_PROTT5_CROSSFAMILY_AUDIT"
CLAIM_BOUNDARY = "The method ranks candidate deletion plans by teacher-derived deletion perturbation preference and optional auxiliary risk evidence."

PROTEINS = 500
GROUPS = 3000
CANDIDATES = 24000
K = 8
METHODS = ("learned", "cold_start", "random")
BUDGETS = ("0.10", "0.20")
BOOTSTRAP_SEED = 20260722
BOOTSTRAP_REPLICATES = 10000
TIE_ATOL = 1.0e-12
EPSILON = 1.0e-12
MAD_NORMALIZER = 1.4826
IQR_NORMALIZER = 1.349
Z_CLIP = 8.0
VIEW_WEIGHTS = (0.35, 0.45, 0.20)
NORMALIZE_PATTERN = re.compile(r"[UZOB]")
CATH_METRICS = ("domain_core_deletion_content_ratio", "core_damage", "boundary_overlap", "span_count", "fragmentation")
REFERENCE_FIELDS = {
    "delepref": "scorer_score",
    "esmc": "teacher_utility",
    "direct": "direct_delete_nll_mean",
    "esm2": "esm2_proxy_utility",
    "prott5": "prott5_representation_proxy_utility",
}
ALIGNMENT_METRICS = (
    "group_esmc_prott5_spearman", "group_delepref_prott5_spearman", "group_esm2_prott5_spearman",
    "delepref_prott5_top1_set_exact_agreement", "esmc_prott5_top1_set_exact_agreement",
    "direct_prott5_top1_set_exact_agreement", "esm2_prott5_top1_set_exact_agreement",
    "all_four_reference_top1_sets_equal_prott5", "esmc_prott5_pairwise_order_agreement",
    "delepref_prott5_pairwise_order_agreement",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path: Path, payload: bytes) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def close(left: Any, right: Any, tolerance: float = 2.5e-10) -> bool:
    a, b = finite(left), finite(right)
    if not math.isfinite(a) and not math.isfinite(b):
        return True
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def parse_intervals(value: str) -> Tuple[Tuple[int, int], ...]:
    raw = json.loads(value)
    intervals = tuple((int(item[0]), int(item[1])) for item in raw)
    if not intervals:
        raise ValueError("empty intervals")
    previous = -2
    for start, end in intervals:
        if start < 0 or end < start or start <= previous + 1:
            raise ValueError("invalid sorted/nonoverlap/separator intervals")
        previous = end
    return intervals


def apply_deletion(sequence: str, intervals: Sequence[Tuple[int, int]]) -> str:
    pieces: List[str] = []
    cursor = 0
    for start, end in intervals:
        pieces.append(sequence[cursor:start])
        cursor = end + 1
    pieces.append(sequence[cursor:])
    return "".join(pieces)


def mask_hash(length: int, intervals: Sequence[Tuple[int, int]]) -> str:
    mask = bytearray(length)
    for start, end in intervals:
        mask[start : end + 1] = b"\x01" * (end - start + 1)
    return sha256_bytes(bytes(mask))


def retained_mapping(length: int, intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    deleted = {position for start, end in intervals for position in range(start, end + 1)}
    result = []
    edited = 0
    for original in range(length):
        if original in deleted:
            continue
        result.append((original, edited))
        edited += 1
    return result


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low, high = int(math.floor(position)), int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def robust_center_scale(values: Sequence[float]) -> Tuple[float, float, str]:
    center = statistics.median(values)
    if len(values) == 1:
        return center, 1.0, "singleton_unit_scale"
    mad = statistics.median(abs(value - center) for value in values)
    scale = MAD_NORMALIZER * mad
    if scale > EPSILON:
        return center, scale, "median_mad"
    if len(values) >= 4:
        scale = (percentile(values, 0.75) - percentile(values, 0.25)) / IQR_NORMALIZER
        if scale > EPSILON:
            return center, scale, "median_iqr_fallback"
    scale = statistics.pstdev(values)
    if scale > EPSILON:
        return center, scale, "median_population_std_fallback"
    return center, 1.0, "constant_unit_scale"


def cosine_shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    left, right = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0:
        return math.nan, math.nan
    cosine = float(np.dot(left, right) / denominator)
    shift = float(np.clip((1.0 - cosine) / 2.0, 0.0, 1.0))
    return shift, float(np.clip(1.0 - 2.0 * shift, 0.0, 1.0))


def recompute_view(base: Mapping[str, str], wt: np.ndarray, edited: np.ndarray) -> Dict[str, Any]:
    intervals = parse_intervals(base["deleted_intervals"])
    length = int(base["sequence_length"])
    mapping = retained_mapping(length, intervals)
    if wt.shape[0] != length or edited.shape[0] != int(base["edited_length"]) or len(mapping) != edited.shape[0]:
        raise ValueError("retained mapping/shape mismatch")
    global_shift, global_similarity = cosine_shift(wt.astype(np.float32).mean(axis=0), edited.astype(np.float32).mean(axis=0))
    retained_shifts = [cosine_shift(wt[original], edited[mutant])[0] for original, mutant in mapping]
    retained_shift = float(np.mean(np.asarray(retained_shifts, dtype=np.float32)))
    retained_similarity = float(np.clip(1.0 - 2.0 * retained_shift, 0.0, 1.0))
    by_original = dict(mapping)
    junction_values, details = [], []
    for start, end in intervals:
        left, right = start - 1, end + 1
        if left in by_original and right in by_original:
            left_shift, left_similarity = cosine_shift(wt[left], edited[by_original[left]])
            right_shift, right_similarity = cosine_shift(wt[right], edited[by_original[right]])
            value = float(np.clip(1.0 - left_shift - right_shift, 0.0, 1.0))
            details.append({"aggregation": "two_sided", "left_original": left, "right_original": right, "left_shift": left_shift, "right_shift": right_shift, "left_similarity": left_similarity, "right_similarity": right_similarity})
        elif left in by_original or right in by_original:
            anchor = left if left in by_original else right
            anchor_shift, anchor_similarity = cosine_shift(wt[anchor], edited[by_original[anchor]])
            value = float(np.clip(1.0 - 2.0 * anchor_shift, 0.0, 1.0))
            details.append({"aggregation": "one_sided_terminal", "anchor_original": anchor, "anchor_shift": anchor_shift, "anchor_similarity": anchor_similarity})
        else:
            value = 0.0
            details.append({"aggregation": "no_retained_anchor"})
        junction_values.append(value)
    junction_mean = float(np.mean(junction_values))
    junction_min = min(junction_values)
    junction = junction_values[0] if len(junction_values) == 1 else 0.5 * junction_mean + 0.5 * junction_min
    junction = float(np.clip(junction, 0.0, 1.0))
    utility = 0.35 * global_similarity + 0.45 * retained_similarity + 0.20 * junction
    return {
        "prott5_global_shift": global_shift,
        "prott5_global_similarity": global_similarity,
        "prott5_retained_region_shift": retained_shift,
        "prott5_retained_similarity": retained_similarity,
        "prott5_junction_mean": junction_mean,
        "prott5_junction_min": junction_min,
        "prott5_junction_plausibility": junction,
        "prott5_representation_proxy_utility": utility,
        "junction_values": junction_values,
        "junction_details": details,
    }


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> Tuple[float, int, str]:
    clean = [(left, right) for left, right in zip(values_a, values_b) if math.isfinite(left) and math.isfinite(right)]
    if len(clean) < 2:
        return math.nan, len(clean), "FEWER_THAN_TWO_ELIGIBLE"
    left_values = [left for left, _right in clean]
    right_values = [right for _left, right in clean]
    if max(left_values) - min(left_values) <= TIE_ATOL or max(right_values) - min(right_values) <= TIE_ATOL:
        return math.nan, len(clean), "CONSTANT_INPUT"
    from scipy.stats import spearmanr  # type: ignore

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = spearmanr(left_values, right_values)
    value = float(result.statistic if hasattr(result, "statistic") else result[0])
    return (value, len(clean), "") if math.isfinite(value) else (math.nan, len(clean), "NONFINITE_RESULT")


def top_ids(group: Sequence[Mapping[str, Any]], field: str) -> List[str]:
    values = [(row["candidate_id"], finite(row.get(field))) for row in group if math.isfinite(finite(row.get(field)))]
    if not values:
        return []
    best = max(value for _candidate, value in values)
    return sorted(candidate for candidate, value in values if abs(value - best) <= TIE_ATOL)


def order_sign(left: float, right: float) -> str:
    if not math.isfinite(left) or not math.isfinite(right):
        return "NA"
    if abs(left - right) <= TIE_ATOL:
        return "TIE"
    return "A_GT_B" if left > right else "A_LT_B"


def order_status(reference: str, proxy: str) -> str:
    if reference == "NA" or proxy == "NA":
        return "NA"
    if reference == proxy:
        return "AGREE_BOTH_TIE" if reference == "TIE" else "AGREE_DIRECTION"
    if reference == "TIE":
        return "DISAGREE_REFERENCE_TIE_ONLY"
    if proxy == "TIE":
        return "DISAGREE_PROTT5_TIE_ONLY"
    return "DISAGREE_OPPOSITE_DIRECTION"


def tie_mean(group: Sequence[Mapping[str, Any]], ids: Sequence[str], field: str) -> float:
    by_id = {row["candidate_id"]: row for row in group}
    values = [finite(by_id[candidate].get(field)) for candidate in ids]
    return float(np.mean(values)) if values and all(math.isfinite(value) for value in values) else math.nan


def scope_definitions() -> List[Tuple[str, str, str]]:
    return [(f"{method}:{budget}", method, budget) for method in METHODS for budget in BUDGETS] + [("ALL", "ALL", "ALL")]


def in_scope(row: Mapping[str, Any], method: str, budget: str) -> bool:
    return (method == "ALL" or row.get("method") == method) and (budget == "ALL" or row.get("budget") == budget)


def metric_float(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool):
        return float(value)
    if str(value).lower() in {"true", "false"}:
        return 1.0 if str(value).lower() == "true" else 0.0
    return finite(value)


def protein_values(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    grouped: MutableMapping[str, List[float]] = defaultdict(list)
    for row in rows:
        value = metric_float(row, field)
        if math.isfinite(value):
            grouped[row["protein_id"]].append(value)
    return np.asarray([float(np.mean(grouped[protein])) for protein in sorted(grouped) if grouped[protein]], dtype=np.float64)


def bootstrap_mean(values: np.ndarray) -> Tuple[float, float, float]:
    if values.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    samples = values[indices].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return float(values.mean()), float(low), float(high)


def validate_environment_and_protocol() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    checks = {
        "cuda_visible_devices_empty": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "torch_not_imported": "torch" not in sys.modules,
        "transformers_not_imported": "transformers" not in sys.modules,
        "protocol_present": (OUT / "protocol.json").is_file(),
        "protocol_hash": sha256_file(OUT / "protocol.json") == (OUT / "protocol.sha256").read_text(encoding="utf-8").strip(),
    }
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    checks.update(
        {
            "protocol_id": protocol.get("protocol_id") == "DelePref_21_PROTT5_CROSSFAMILY_AUDIT",
            "model_repo": protocol.get("model", {}).get("repo_id") == "Rostlab/prot_t5_xl_half_uniref50-enc",
            "model_revision_full": protocol.get("model", {}).get("full_revision") == "94a6abc029ae13029317b140b7424e012bf8dfbf",
            "cohort_counts": protocol.get("cohort", {}).get("protein_count") == PROTEINS and protocol.get("cohort", {}).get("group_count") == GROUPS and protocol.get("cohort", {}).get("candidate_count") == CANDIDATES and protocol.get("cohort", {}).get("K") == K,
            "bootstrap_rules": protocol.get("statistics", {}).get("bootstrap_seed") == BOOTSTRAP_SEED and protocol.get("statistics", {}).get("bootstrap_replicates") == BOOTSTRAP_REPLICATES and protocol.get("statistics", {}).get("bootstrap_unit") == "protein_id only",
            "declarations": all(protocol.get("prohibitions_and_declarations", {}).get(name) is False for name in ("training", "student_checkpoint_load", "student_scorer_inference", "teacher_rescoring", "candidate_regeneration", "proposal_inference", "delepref_14_opened", "network_after_bootstrap", "cpu_representation_fallback", "zero_fill", "second_gpu")),
            "claim_boundary": protocol.get("claim_boundary") == CLAIM_BOUNDARY,
            "runner_hash_locked": protocol.get("code_hashes", {}).get("runner") == sha256_file(RUNNER),
            "validator_hash_locked": protocol.get("code_hashes", {}).get("validator") == sha256_file(VALIDATOR),
            "bootstrap_hash_locked": protocol.get("code_hashes", {}).get("bootstrap") == sha256_file(BOOTSTRAP),
        }
    )
    scientific = json.loads((OUT / "formal_scientific_complete.json").read_text(encoding="utf-8"))
    gpu_complete = json.loads((OUT / "formal_gpu_representation_complete.json").read_text(encoding="utf-8"))
    checks["scientific_outputs_sealed"] = scientific.get("status") == "PASS_SCIENTIFIC_OUTPUTS_SEALED_AWAITING_VALIDATION"
    checks["formal_gpu_complete"] = gpu_complete.get("status") == "PASS" and gpu_complete.get("wt_representation_rows") == PROTEINS and gpu_complete.get("candidate_representation_rows") == CANDIDATES
    return checks, protocol


def verify_model_runtime_and_inputs(protocol: Mapping[str, Any]) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    model_files = protocol["model"]["files"]
    checks["model_file_count"] = len(model_files) == 7
    checks["model_files_unchanged"] = all(
        Path(item["path"]).is_file() and Path(item["path"]).stat().st_size == int(item["size"]) and sha256_file(Path(item["path"])) == item["sha256"]
        for item in model_files
    )
    model_provenance = json.loads((OUT / "model_provenance.json").read_text(encoding="utf-8"))
    checks["model_provenance_offline_immutable"] = model_provenance.get("status") == "PASS_FORMAL_GPU_REPRESENTATIONS_COMPLETE" and model_provenance.get("snapshot_immutable_during_formal_forward") is True and model_provenance["snapshot"]["aggregate_sha256"] == protocol["model"]["snapshot_sha256"]
    vendor = json.loads((OUT / "runtime_vendor_provenance.json").read_text(encoding="utf-8"))
    checks["runtime_vendor_files_unchanged"] = all(
        Path(item["path"]).is_file() and Path(item["path"]).stat().st_size == int(item["size"]) and sha256_file(Path(item["path"])) == item["sha256"]
        for item in vendor["inventory"]["files"]
    ) and vendor["inventory"]["aggregate_sha256"] == protocol["runtime_vendor"]["aggregate_sha256"]
    checks["sentencepiece_wheel_hash"] = sha256_file(Path(vendor["wheel"]["path"])) == protocol["runtime_vendor"]["sentencepiece_wheel_sha256"]
    source_checks = []
    checkpoint_checks = []
    v20_checks = []
    for item in protocol["authoritative_input_hashes"]:
        path = Path(item["path"])
        current_ok = path.is_file() and path.stat().st_size == int(item["size"]) and sha256_file(path) == item["current_sha256"] == item["expected_sha256"]
        source_checks.append(current_ok)
        if "checkpoint" in item["name"]:
            checkpoint_checks.append(current_ok)
        if "delepref_20" in item["name"]:
            v20_checks.append(current_ok)
    checks["all_authoritative_inputs_unchanged"] = all(source_checks)
    checks["frozen_student_checkpoint_unchanged"] = bool(checkpoint_checks) and all(checkpoint_checks)
    checks["delepref_20_source_package_and_model_unchanged"] = bool(v20_checks) and all(v20_checks)
    checks["no_delepref_14_source"] = not any("delepref_14" in item["path"].lower() for item in protocol["authoritative_input_hashes"])
    checks["input_hash_validation_csv"] = len(read_csv(OUT / "input_hash_validation.csv")) == len(protocol["authoritative_input_hashes"]) and all(row["status"] == "PASS" for row in read_csv(OUT / "input_hash_validation.csv"))
    return checks


def validate_tokenization(base_by_id: Mapping[str, Mapping[str, str]], base_rows: Sequence[Mapping[str, str]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(VENDOR_SITE))
    import sentencepiece as spm  # type: ignore

    spiece_path = Path(protocol["model"]["snapshot_path"]) / "spiece.model"
    processor = spm.SentencePieceProcessor(model_file=str(spiece_path))
    records = read_csv(OUT / "input_tokenization_preflight.csv")
    if len(records) != PROTEINS + CANDIDATES:
        raise ValueError("tokenization preflight row count mismatch")
    wt_by_protein: Dict[str, Mapping[str, str]] = {}
    for row in base_rows:
        wt_by_protein.setdefault(row["protein_id"], row)
    checked = 0
    changed = 0
    replacements = 0
    for record in records:
        if record["record_kind"] == "WT":
            source = wt_by_protein[record["protein_id"]]
            sequence, sequence_hash = source["wildtype_sequence"], source["sequence_sha256"]
        else:
            source = base_by_id[record["candidate_id"]]
            sequence, sequence_hash = source["edited_sequence"], source["edited_sequence_sha256"]
        normalized = NORMALIZE_PATTERN.sub("X", sequence)
        expected_replacements = sum(residue in "UZOB" for residue in sequence)
        ids = list(processor.encode(" ".join(normalized), out_type=int)) + [int(processor.eos_id())]
        checks = (
            record["original_sequence_sha256"] == sequence_hash,
            record["normalized_sequence_sha256"] == sha256_text(normalized),
            int(record["original_length"]) == len(sequence),
            int(record["normalized_length"]) == len(normalized),
            truthy(record["normalization_changed"]) == (normalized != sequence),
            int(record["replacement_count"]) == expected_replacements,
            int(record["token_count_with_eos"]) == len(ids) == len(normalized) + 1,
            int(record["residue_token_count"]) == len(normalized),
            record["token_ids_sha256"] == sha256_text(json.dumps(ids, separators=(",", ":"))),
            not truthy(record["truncation"]),
            record["status"] == "PASS",
        )
        if not all(checks):
            raise ValueError(f"tokenization mismatch for {record['record_kind']}:{record['protein_id']}:{record['candidate_id']}")
        checked += 1
        changed += normalized != sequence
        replacements += expected_replacements
    return {
        "records_checked": checked,
        "normalization_changed_records": changed,
        "replacement_count": replacements,
        "sentencepiece_version": getattr(spm, "__version__", "unknown"),
        "sentencepiece_import_file": str(Path(spm.__file__).resolve()),
        "transformers_imported": "transformers" in sys.modules,
        "torch_imported": "torch" in sys.modules,
        "status": "PASS",
    }


def validate_representations_and_views(base_rows: Sequence[Mapping[str, str]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    base_by_id = {row["candidate_id"]: row for row in base_rows}
    views = read_csv(OUT / "prott5_candidate_views.csv")
    view_by_id = {row["candidate_id"]: row for row in views}
    wt_inventory = {row["protein_id"]: row for row in read_csv(OUT / "prott5_wt_representation_inventory.csv")}
    candidate_inventory = {row["candidate_id"]: row for row in read_csv(OUT / "prott5_candidate_representation_inventory.csv")}
    shard_rows = read_csv(OUT / "prott5_representation_shard_index.csv")
    if len(view_by_id) != CANDIDATES or len(wt_inventory) != PROTEINS or len(candidate_inventory) != CANDIDATES or len(shard_rows) != PROTEINS:
        raise ValueError("representation/view inventory counts mismatch")
    numeric_fields = (
        "prott5_global_shift", "prott5_global_similarity", "prott5_retained_region_shift",
        "prott5_retained_similarity", "prott5_junction_mean", "prott5_junction_min",
        "prott5_junction_plausibility", "prott5_representation_proxy_utility",
    )
    checked_candidates = 0
    max_abs_error = 0.0
    identities = []
    for shard_row in shard_rows:
        protein = shard_row["protein_id"]
        path = Path(shard_row["representation_shard_path"])
        if sha256_file(path) != shard_row["representation_shard_sha256"]:
            raise ValueError(f"raw representation shard hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            wt = np.asarray(data["wt_embedding"], dtype=np.float16)
            concatenated = np.asarray(data["candidate_embeddings"], dtype=np.float16)
            offsets = [int(value) for value in data["candidate_offsets"].tolist()]
            candidate_ids = [str(value) for value in data["candidate_ids"].tolist()]
            if len(candidate_ids) != 48 or len(offsets) != 49 or offsets[0] != 0 or offsets[-1] != concatenated.shape[0]:
                raise ValueError(f"raw shard offsets/candidate count mismatch: {protein}")
            wt_row = wt_inventory[protein]
            if sha256_bytes(np.ascontiguousarray(wt, dtype=np.float16).tobytes(order="C")) != wt_row["embedding_sha256"] or wt.shape[0] != int(wt_row["residue_count"]):
                raise ValueError(f"WT representation inventory mismatch: {protein}")
            for index, candidate_id in enumerate(candidate_ids):
                base = base_by_id[candidate_id]
                inventory = candidate_inventory[candidate_id]
                observed = view_by_id[candidate_id]
                edited = concatenated[offsets[index] : offsets[index + 1]]
                if sha256_bytes(np.ascontiguousarray(edited, dtype=np.float16).tobytes(order="C")) != inventory["embedding_sha256"]:
                    raise ValueError(f"candidate embedding hash mismatch: {candidate_id}")
                if int(inventory["offset_start"]) != offsets[index] or int(inventory["offset_end_exclusive"]) != offsets[index + 1] or int(inventory["residue_count"]) != edited.shape[0]:
                    raise ValueError(f"candidate representation offset mismatch: {candidate_id}")
                identity_fields = ("protein_id", "candidate_id", "budget", "method", "sequence_sha256", "edited_sequence_sha256", "deletion_mask_sha256", "deleted_intervals")
                if any(observed[field] != base[field] for field in identity_fields):
                    raise ValueError(f"candidate view identity mismatch: {candidate_id}")
                recomputed = recompute_view(base, wt, edited)
                for field in numeric_fields:
                    error = abs(finite(observed[field]) - float(recomputed[field]))
                    max_abs_error = max(max_abs_error, error)
                    if error > 2.5e-10:
                        raise ValueError(f"view recomputation mismatch {field}:{candidate_id}:{error}")
                observed_values = json.loads(observed["prott5_junction_values_json"])
                observed_details = json.loads(observed["prott5_junction_detail_json"])
                if len(observed_values) != len(recomputed["junction_values"]) or len(observed_details) != len(recomputed["junction_details"]):
                    raise ValueError(f"junction count mismatch: {candidate_id}")
                if any(left.get("aggregation") != right.get("aggregation") for left, right in zip(observed_details, recomputed["junction_details"])):
                    raise ValueError(f"junction mapping detail mismatch: {candidate_id}")
                identities.append(candidate_id)
                checked_candidates += 1
    if checked_candidates != CANDIDATES or len(set(identities)) != CANDIDATES:
        raise ValueError("raw representation candidate coverage mismatch")
    by_robust: MutableMapping[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in views:
        by_robust[(row["protein_id"], int(row["deleted_length"]))].append(row)
    fallback_counts = Counter()
    for group in by_robust.values():
        values = [finite(row["prott5_representation_proxy_utility"]) for row in group]
        center, scale, method = robust_center_scale(values)
        fallback_counts[method] += 1
        for row, value in zip(group, values):
            z = max(-Z_CLIP, min(Z_CLIP, (value - center) / max(scale, EPSILON)))
            if not close(row["prott5_proxy_robust_center"], center) or not close(row["prott5_proxy_robust_scale"], scale) or row["prott5_proxy_robust_scale_method"] != method or not close(row["prott5_proxy_robust_z_clipped"], z):
                raise ValueError(f"robust scale mismatch: {row['candidate_id']}")
    merged = []
    for base in base_rows:
        row = dict(base)
        row.update(view_by_id[base["candidate_id"]])
        merged.append(row)
    return {
        "wt_representations_checked": len(wt_inventory),
        "candidate_representations_checked": checked_candidates,
        "candidate_views_recomputed": len(views),
        "raw_shards_checked": len(shard_rows),
        "max_abs_view_error": max_abs_error,
        "robust_groups_checked": len(by_robust),
        "robust_fallback_counts": dict(fallback_counts),
        "status": "PASS",
    }, merged


def recompute_group_metrics(scored: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: MutableMapping[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[(row["protein_id"], row["budget"], row["method"])].append(row)
    metrics, pair_rows, selected_rows = [], [], []
    for (protein, budget, method), group in sorted(groups.items()):
        ids = {selector: top_ids(group, field) for selector, field in REFERENCE_FIELDS.items()}
        if any(not values for values in ids.values()):
            raise ValueError("no eligible top1")
        esmc_corr = spearman([finite(row["teacher_utility"]) for row in group], [finite(row["prott5_representation_proxy_utility"]) for row in group])[0]
        delepref_corr = spearman([finite(row["scorer_score"]) for row in group], [finite(row["prott5_representation_proxy_utility"]) for row in group])[0]
        esm2_corr = spearman([finite(row["esm2_proxy_utility"]) for row in group], [finite(row["prott5_representation_proxy_utility"]) for row in group])[0]
        counters = {"esmc": Counter(), "delepref": Counter()}
        for pair_index, (left, right) in enumerate(itertools.combinations(sorted(group, key=lambda row: row["candidate_id"]), 2)):
            proxy_order = order_sign(finite(left["prott5_representation_proxy_utility"]), finite(right["prott5_representation_proxy_utility"]))
            esmc_order = order_sign(finite(left["teacher_utility"]), finite(right["teacher_utility"]))
            delepref_order = order_sign(finite(left["scorer_score"]), finite(right["scorer_score"]))
            esmc_status, delepref_status = order_status(esmc_order, proxy_order), order_status(delepref_order, proxy_order)
            counters["esmc"][esmc_status] += 1
            counters["delepref"][delepref_status] += 1
            pair_rows.append({"protein_id": protein, "budget": budget, "method": method, "pair_index": pair_index, "candidate_id_a": left["candidate_id"], "candidate_id_b": right["candidate_id"], "prott5_order": proxy_order, "esmc_order": esmc_order, "delepref_order": delepref_order, "esmc_vs_prott5_status": esmc_status, "delepref_vs_prott5_status": delepref_status})
        esmc_eligible = 28 - counters["esmc"]["NA"]
        delepref_eligible = 28 - counters["delepref"]["NA"]
        row: Dict[str, Any] = {
            "protein_id": protein,
            "budget": budget,
            "method": method,
            "group_esmc_prott5_spearman": esmc_corr,
            "group_delepref_prott5_spearman": delepref_corr,
            "group_esm2_prott5_spearman": esm2_corr,
            "delepref_prott5_top1_set_exact_agreement": ids["delepref"] == ids["prott5"],
            "esmc_prott5_top1_set_exact_agreement": ids["esmc"] == ids["prott5"],
            "direct_prott5_top1_set_exact_agreement": ids["direct"] == ids["prott5"],
            "esm2_prott5_top1_set_exact_agreement": ids["esm2"] == ids["prott5"],
            "all_four_reference_top1_sets_equal_prott5": all(ids[selector] == ids["prott5"] for selector in ("delepref", "esmc", "direct", "esm2")),
            "esmc_prott5_pairwise_order_agreement": (counters["esmc"]["AGREE_DIRECTION"] + counters["esmc"]["AGREE_BOTH_TIE"]) / esmc_eligible,
            "delepref_prott5_pairwise_order_agreement": (counters["delepref"]["AGREE_DIRECTION"] + counters["delepref"]["AGREE_BOTH_TIE"]) / delepref_eligible,
        }
        by_id = {item["candidate_id"]: item for item in group}
        for selector in REFERENCE_FIELDS:
            row[f"{selector}_top1_candidate_ids_json"] = json.dumps(ids[selector], separators=(",", ":"))
            row[f"{selector}_top1_tie_count"] = len(ids[selector])
            for metric in CATH_METRICS:
                row[f"{selector}_top1_{metric}"] = tie_mean(group, ids[selector], "cath_" + metric)
            for candidate_id in ids[selector]:
                selected_rows.append({"selector": selector, "protein_id": protein, "budget": budget, "method": method, "candidate_id": candidate_id, "top1_candidate_ids_json": row[f"{selector}_top1_candidate_ids_json"], "top1_tie_count": len(ids[selector])})
        metrics.append(row)
    return metrics, pair_rows, selected_rows


def validate_metric_outputs(scored: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    expected_groups, expected_pairs, expected_selected = recompute_group_metrics(scored)
    observed_groups = {(row["protein_id"], row["budget"], row["method"]): row for row in read_csv(OUT / "prott5_group_metrics.csv")}
    observed_pairs = {(row["protein_id"], row["budget"], row["method"], int(row["pair_index"])): row for row in read_csv(OUT / "prott5_pairwise_order_diagnostics.csv")}
    observed_selected = {(row["selector"], row["protein_id"], row["budget"], row["method"], row["candidate_id"]) for row in read_csv(OUT / "prott5_top1_selected_plans.csv")}
    if len(observed_groups) != GROUPS or len(observed_pairs) != GROUPS * 28:
        raise ValueError("observed group/pair counts mismatch")
    numeric = (
        "group_esmc_prott5_spearman", "group_delepref_prott5_spearman", "group_esm2_prott5_spearman",
        "esmc_prott5_pairwise_order_agreement", "delepref_prott5_pairwise_order_agreement",
    )
    bools = (
        "delepref_prott5_top1_set_exact_agreement", "esmc_prott5_top1_set_exact_agreement",
        "direct_prott5_top1_set_exact_agreement", "esm2_prott5_top1_set_exact_agreement",
        "all_four_reference_top1_sets_equal_prott5",
    )
    for expected in expected_groups:
        key = (expected["protein_id"], expected["budget"], expected["method"])
        observed = observed_groups[key]
        if any(not close(observed[field], expected[field]) for field in numeric):
            raise ValueError(f"group numeric metric mismatch: {key}")
        if any(truthy(observed[field]) != bool(expected[field]) for field in bools):
            raise ValueError(f"group top1 agreement mismatch: {key}")
        for selector in REFERENCE_FIELDS:
            if observed[f"{selector}_top1_candidate_ids_json"] != expected[f"{selector}_top1_candidate_ids_json"] or int(observed[f"{selector}_top1_tie_count"]) != expected[f"{selector}_top1_tie_count"]:
                raise ValueError(f"group tie-aware top1 mismatch: {key}:{selector}")
            for metric in CATH_METRICS:
                if not close(observed[f"{selector}_top1_{metric}"], expected[f"{selector}_top1_{metric}"]):
                    raise ValueError(f"group CATH top1 metric mismatch: {key}:{selector}:{metric}")
    for expected in expected_pairs:
        key = (expected["protein_id"], expected["budget"], expected["method"], expected["pair_index"])
        observed = observed_pairs[key]
        fields = ("candidate_id_a", "candidate_id_b", "prott5_order", "esmc_order", "delepref_order", "esmc_vs_prott5_status", "delepref_vs_prott5_status")
        if any(observed[field] != str(expected[field]) for field in fields):
            raise ValueError(f"pairwise order mismatch: {key}")
    expected_selected_set = {(row["selector"], row["protein_id"], row["budget"], row["method"], row["candidate_id"]) for row in expected_selected}
    if observed_selected != expected_selected_set:
        raise ValueError("tie-aware selected-plan set mismatch")
    return {
        "group_metrics_recomputed": len(expected_groups),
        "pairwise_orders_recomputed": len(expected_pairs),
        "tie_aware_selected_rows_recomputed": len(expected_selected),
        "computed_group_rows": expected_groups,
        "status": "PASS",
    }


def validate_candidate_metrics(scored: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    observed = {(row["scope"], row["reference"]): row for row in read_csv(OUT / "prott5_candidate_level_metrics.csv")}
    checked = 0
    for scope, method, budget in scope_definitions():
        subset = [row for row in scored if in_scope(row, method, budget)]
        for reference, field in (("esmc", "teacher_utility"), ("delepref", "scorer_score"), ("esm2", "esm2_proxy_utility")):
            correlation, eligible, reason = spearman([finite(row[field]) for row in subset], [finite(row["prott5_representation_proxy_utility"]) for row in subset])
            row = observed[(scope, reference)]
            if int(row["N_total"]) != len(subset) or int(row["N_eligible"]) != eligible or not close(row["spearman"], correlation) or row["spearman_na_reason"] != reason:
                raise ValueError(f"candidate metric mismatch: {scope}:{reference}")
            checked += 1
    return {"candidate_metric_rows_recomputed": checked, "status": "PASS"}


def validate_bootstraps(computed_groups: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    alignment = {(row["scope"], row["metric"]): row for row in read_csv(OUT / "prott5_alignment_bootstrap_ci.csv")}
    context = {(row["scope"], row["comparison"], row["metric"]): row for row in read_csv(OUT / "prott5_contextual_cath_bootstrap_ci.csv")}
    alignment_checked = 0
    context_checked = 0
    for scope, method, budget in scope_definitions():
        subset = [row for row in computed_groups if in_scope(row, method, budget)]
        for metric in ALIGNMENT_METRICS:
            values = protein_values(subset, metric)
            point, low, high = bootstrap_mean(values)
            row = alignment[(scope, metric)]
            if int(row["bootstrap_replicates"]) != BOOTSTRAP_REPLICATES or int(row["bootstrap_seed"]) != BOOTSTRAP_SEED or row["bootstrap_unit"] != "protein_id" or not all((close(row["point_estimate"], point), close(row["ci95_low"], low), close(row["ci95_high"], high))):
                raise ValueError(f"alignment bootstrap mismatch: {scope}:{metric}")
            alignment_checked += 1
        for reference in ("delepref", "direct", "esm2"):
            for metric in CATH_METRICS:
                delta_rows = []
                field = "delta"
                for source in subset:
                    row = dict(source)
                    row[field] = finite(source[f"prott5_top1_{metric}"]) - finite(source[f"{reference}_top1_{metric}"])
                    delta_rows.append(row)
                point, low, high = bootstrap_mean(protein_values(delta_rows, field))
                row = context[(scope, f"prott5_top1_minus_{reference}_top1", metric)]
                if row["comparison_status"] != "NA_DIFFERENT_PROTOCOL" or int(row["bootstrap_replicates"]) != BOOTSTRAP_REPLICATES or int(row["bootstrap_seed"]) != BOOTSTRAP_SEED or not all((close(row["point_estimate_left_minus_right"], point), close(row["ci95_low"], low), close(row["ci95_high"], high))):
                    raise ValueError(f"context bootstrap mismatch: {scope}:{reference}:{metric}")
                context_checked += 1
    if alignment_checked != 70 or context_checked != 105:
        raise ValueError("bootstrap output row counts mismatch")
    return {"alignment_rows_recomputed": alignment_checked, "contextual_rows_recomputed": context_checked, "replicates_each": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "unit": "protein_id", "status": "PASS"}


def verify_scientific_manifest() -> Dict[str, Any]:
    manifest = json.loads((OUT / "scientific_manifest.json").read_text(encoding="utf-8"))
    failures = []
    total_bytes = 0
    for item in manifest["artifacts"]:
        path = OUT / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["size"]) or sha256_file(path) != item["sha256"]:
            failures.append(item["path"])
        else:
            total_bytes += path.stat().st_size
    if failures:
        raise ValueError(f"scientific manifest hash failures: {failures[:10]}")
    return {"artifacts_verified": len(manifest["artifacts"]), "total_bytes_verified": total_bytes, "raw_representation_shards": manifest["raw_representation_shards"], "raw_scoring_shards": manifest["raw_scoring_shards"], "status": "PASS"}


def check_git_and_resources(protocol: Mapping[str, Any]) -> Dict[str, Any]:
    diff = subprocess.run(["git", "diff", "--check"], cwd=REPO, capture_output=True, text=True, check=False)
    cached = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO, check=False)
    gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    processes = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    return {
        "git_diff_check": diff.returncode == 0,
        "git_diff_check_output": diff.stdout + diff.stderr,
        "git_index_empty": cached.returncode == 0,
        "protocol_source_sha256_before": protocol["protected_protocol_source_sha256_before"],
        "protocol_source_sha256_after": sha256_file(PROTOCOL_SOURCE),
        "protocol_source_unchanged": sha256_file(PROTOCOL_SOURCE) == protocol["protected_protocol_source_sha256_before"],
        "nvidia_smi_query_exit_code": gpu.returncode,
        "gpu_compute_processes": processes,
        "no_gpu_process_residual": gpu.returncode == 0 and not processes,
        "torch_not_imported": "torch" not in sys.modules,
        "transformers_not_imported": "transformers" not in sys.modules,
    }


def json_csv_parse_checks() -> Dict[str, Any]:
    json_files = sorted(OUT.rglob("*.json"))
    csv_files = sorted(OUT.rglob("*.csv"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    for path in csv_files:
        read_csv(path)
    return {"json_files_parsed": len(json_files), "csv_files_parsed": len(csv_files), "status": "PASS"}


def required_outputs() -> List[str]:
    return [
        "SUMMARY.md", "protocol.json", "protocol.sha256", "run_summary.json", "model_provenance.json",
        "runtime_vendor_provenance.json", "gpu_resource_provenance.json", "input_hash_validation.csv",
        "prott5_wt_representation_inventory.csv", "prott5_candidate_representation_inventory.csv",
        "prott5_candidate_views.csv", "prott5_group_metrics.csv", "prott5_top1_selected_plans.csv",
        "prott5_alignment_bootstrap_ci.csv", "prott5_contextual_cath_bootstrap_ci.csv",
        "missingness_and_tie_report.csv", "independent_validation.json", "audit_manifest.json",
        "result_summary.md", PASS_SENTINEL,
    ]


def finalize(validation: Dict[str, Any], protocol: Mapping[str, Any]) -> None:
    validation_path = OUT / "independent_validation.json"
    atomic_json(validation_path, validation)
    run_summary = json.loads((OUT / "run_summary.json").read_text(encoding="utf-8"))
    run_summary.update(
        {
            "status": PASS_SENTINEL,
            "independent_validation": "PASS",
            "independent_validation_sha256": sha256_file(validation_path),
            "protected_protocol_source_sha256_after": sha256_file(PROTOCOL_SOURCE),
            "protocol_source_unchanged": sha256_file(PROTOCOL_SOURCE) == protocol["protected_protocol_source_sha256_before"],
        }
    )
    atomic_json(OUT / "run_summary.json", run_summary)
    summary_text = (OUT / "SUMMARY.md").read_text(encoding="utf-8")
    summary_text = summary_text.replace("Status: `AWAITING_INDEPENDENT_VALIDATION`", f"Status: `{PASS_SENTINEL}`")
    atomic_text(OUT / "SUMMARY.md", summary_text)

    artifacts = []
    volatile = {
        "audit/routing/formal_launcher.stdout_stderr.log",
        "audit/routing/formal_launcher.exit_code",
        "audit/routing/formal_launcher.done.json",
    }
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path == OUT / "audit_manifest.json" or path == OUT / PASS_SENTINEL:
            continue
        relative = str(path.relative_to(OUT))
        if relative in volatile:
            continue
        artifacts.append({"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)})
    missing_before_sentinel = [name for name in required_outputs() if name not in {PASS_SENTINEL, "audit_manifest.json"} and not (OUT / name).is_file()]
    if missing_before_sentinel:
        raise ValueError(f"required output missing before final manifest: {missing_before_sentinel}")
    audit = {
        "schema_version": "delepref_21_audit_manifest_v1",
        "status": "PASS",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "required_outputs": required_outputs(),
        "required_outputs_missing_before_sentinel": missing_before_sentinel,
        "audit_manifest_self_hash_excluded": True,
        "pass_sentinel_hash_excluded": True,
        "volatile_launcher_evidence_excluded_because_launcher_trap_finishes_after_validator": sorted(volatile),
        "claim_boundary": CLAIM_BOUNDARY,
        "created_at_utc": utc_now(),
    }
    atomic_json(OUT / "audit_manifest.json", audit)
    atomic_text(OUT / PASS_SENTINEL, PASS_SENTINEL + "\n")
    if any(not (OUT / name).is_file() for name in required_outputs()):
        raise ValueError("required output inventory incomplete after sentinel")


def full_validation() -> int:
    checks, protocol = validate_environment_and_protocol()
    checks.update(verify_model_runtime_and_inputs(protocol))
    if not all(bool(value) for value in checks.values()):
        raise ValueError("protocol/model/input checks failed: " + json.dumps(checks, sort_keys=True))
    manifest_validation = verify_scientific_manifest()
    base_rows = read_csv(V20 / "input_manifest.csv")
    base_by_id = {row["candidate_id"]: row for row in base_rows}
    if len(base_rows) != CANDIDATES or len(base_by_id) != CANDIDATES:
        raise ValueError("frozen base cohort identity/count mismatch")
    token_validation = validate_tokenization(base_by_id, base_rows, protocol)
    if token_validation["torch_imported"] or token_validation["transformers_imported"]:
        raise ValueError("validator imported forbidden Torch/Transformers module")
    representation_validation, scored = validate_representations_and_views(base_rows)
    metric_validation = validate_metric_outputs(scored)
    candidate_validation = validate_candidate_metrics(scored)
    bootstrap_validation = validate_bootstraps(metric_validation.pop("computed_group_rows"))
    missing_rows = read_csv(OUT / "missingness_and_tie_report.csv")
    if len(missing_rows) != 70:
        raise ValueError(f"missingness/tie report row count {len(missing_rows)} != 70")
    parse_validation = json_csv_parse_checks()
    resource_git = check_git_and_resources(protocol)
    if not all(resource_git[name] for name in ("git_diff_check", "git_index_empty", "protocol_source_unchanged", "no_gpu_process_residual", "torch_not_imported", "transformers_not_imported")):
        raise ValueError("git/master/GPU residual gates failed: " + json.dumps(resource_git, sort_keys=True))
    py_compile.compile(str(RUNNER), doraise=True)
    py_compile.compile(str(VALIDATOR), doraise=True)
    validation = {
        "schema_version": "delepref_21_independent_validation_v1",
        "status": "PASS",
        "cpu_only": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "protT5_model_loaded": False,
        "torch_imported": "torch" in sys.modules,
        "transformers_imported": "transformers" in sys.modules,
        "sentencepiece_tokenizer_only": True,
        "checks": checks,
        "scientific_manifest": manifest_validation,
        "tokenization": token_validation,
        "representations_and_views": representation_validation,
        "metrics": metric_validation,
        "candidate_metrics": candidate_validation,
        "bootstrap": bootstrap_validation,
        "missingness_and_tie_rows": len(missing_rows),
        "parse": parse_validation,
        "git_master_gpu": resource_git,
        "code_hashes": {
            "runner": sha256_file(RUNNER),
            "validator": sha256_file(VALIDATOR),
            "bootstrap": sha256_file(BOOTSTRAP),
        },
        "protocol_sha256": sha256_file(OUT / "protocol.json"),
        "claim_boundary": CLAIM_BOUNDARY,
        "created_at_utc": utc_now(),
    }
    finalize(validation, protocol)
    print(json.dumps({"event": PASS_SENTINEL, "status": "PASS", "checks": len(checks)}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    try:
        return full_validation()
    except BaseException as exc:
        failure = {
            "schema_version": "delepref_21_independent_validation_v1",
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": repr(exc),
            "created_at_utc": utc_now(),
            "protT5_model_loaded": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "pass_sentinel_written": (OUT / PASS_SENTINEL).exists(),
        }
        atomic_json(OUT / "independent_validation.json", failure)
        atomic_json(OUT / "audit/failure_or_recovery_history/independent_validation_failure.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
