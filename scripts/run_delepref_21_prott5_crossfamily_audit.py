#!/usr/bin/env python3
"""DelePref-21 frozen ProtT5 representation-proxy alignment audit.

The program consumes only the already-frozen DelePref-20 CATH candidate package.
It never generates candidates, loads the DelePref checkpoint, calls an ESMC
teacher, trains a model, or opens DelePref-14 data.  The sole neural forward stage is
the one protocol-locked ProtT5 encoder representation pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/experiments/delepref_step004000_delepref_21_prott5_crossfamily_audit"
V20 = REPO / "results/experiments/delepref_step004000_delepref_20_direct_plm_cross_teacher"
DelePref7 = REPO / "results/experiments/delepref_step004000_evidence_controls_policy_diagnostic"
MODEL_ROOT = Path("models/prott5_xl_half_uniref50_enc")
VENDOR_SITE = MODEL_ROOT / "runtime_vendor/site-packages"
CHECKPOINT = REPO / "results/experiments/delepref_evaluation_checkpoint/final_freeze/delepref_evaluation_step_00004000.pt"
PROTOCOL_SOURCE = Path(__file__).resolve()
SCRIPT = Path(__file__).resolve()
BOOTSTRAP_SCRIPT = REPO / "scripts/bootstrap_prott5.py"
VALIDATOR = REPO / "scripts/validate_delepref_21_prott5_crossfamily_audit.py"

PASS_SENTINEL = "PASS_DelePref_21_PROTT5_CROSSFAMILY_AUDIT"
CLAIM_BOUNDARY = "The method ranks candidate deletion plans by teacher-derived deletion perturbation preference and optional auxiliary risk evidence."
REPO_ID = "Rostlab/prot_t5_xl_half_uniref50-enc"
REVISION = "94a6abc029ae13029317b140b7424e012bf8dfbf"
EXPECTED_SNAPSHOT_SHA256 = "31f0f19a619401bde542280940521ff3d6d6ee93e2b492e0dd2140f2b78c5432"
EXPECTED_VENDOR_SHA256 = "c6c3a7edd94bb3576a46172a0ff02d9efe4acf0b58dba58a233dc543c5f4dddf"
EXPECTED_CHECKPOINT_SHA256 = "1bc7694be77e24e4b671773be39331daed241dcfe101fd6d2d26f8e23fe7ff4d"
EXPECTED_DelePref7_DEFINITION_SHA256 = "bc77fa32ca5721f3c2b148b679be85744ee86e3269cda13085233c5a10711b03"

PROTEINS = 500
GROUPS = 3000
CANDIDATES = 24000
K = 8
GENERATION_SEED = "101011"
METHODS = ("learned", "cold_start", "random")
BUDGETS = ("0.10", "0.20")
COORDINATES = "0_based_closed_inclusive"
BOOTSTRAP_SEED = 20260722
BOOTSTRAP_REPLICATES = 10000
TIE_ATOL = 1.0e-12
EPSILON = 1.0e-12
MAD_NORMALIZER = 1.4826
IQR_NORMALIZER = 1.349
Z_CLIP = 8.0
MAX_BATCH_SIZE = 16
TARGET_TOKENS_SQUARED = 1_500_000
NORMALIZE_PATTERN = re.compile(r"[UZOB]")
ALLOWED_RAW_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYXUZOB")
VIEW_WEIGHTS = (0.35, 0.45, 0.20)
CATH_METRICS = (
    "domain_core_deletion_content_ratio",
    "core_damage",
    "boundary_overlap",
    "span_count",
    "fragmentation",
)
REFERENCE_FIELDS = {
    "esmc": "teacher_utility",
    "delepref": "scorer_score",
    "direct": "direct_delete_nll_mean",
    "esm2": "esm2_proxy_utility",
    "prott5": "prott5_representation_proxy_utility",
}


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


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


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
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def fmt(value: Any, digits: int = 12) -> str:
    number = finite(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def parse_intervals(value: Any) -> Tuple[Tuple[int, int], ...]:
    raw = json.loads(value) if isinstance(value, str) else value
    intervals: List[Tuple[int, int]] = []
    previous = -2
    if not isinstance(raw, list) or not raw:
        raise ValueError("deleted_intervals must be a non-empty JSON list")
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("invalid deletion interval")
        start, end = int(item[0]), int(item[1])
        if start < 0 or end < start or start <= previous + 1:
            raise ValueError("intervals must be sorted, non-overlapping, and separated by a retained residue")
        intervals.append((start, end))
        previous = end
    return tuple(intervals)


def canonical_intervals(intervals: Sequence[Tuple[int, int]]) -> str:
    return json.dumps([[start, end] for start, end in intervals], separators=(",", ":"))


def deleted_length(intervals: Sequence[Tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in intervals)


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
    deleted = set(position for start, end in intervals for position in range(start, end + 1))
    mapping: List[Tuple[int, int]] = []
    edited_position = 0
    for original_position in range(length):
        if original_position in deleted:
            continue
        mapping.append((original_position, edited_position))
        edited_position += 1
    return mapping


def normalize_sequence(sequence: str) -> Tuple[str, List[Dict[str, Any]]]:
    illegal = sorted(set(sequence) - ALLOWED_RAW_RESIDUES)
    if illegal:
        raise ValueError(f"unsupported residues outside the locked mapping rule: {illegal}")
    replacements = [
        {"position": index, "original": residue, "normalized": "X"}
        for index, residue in enumerate(sequence)
        if residue in "UZOB"
    ]
    return NORMALIZE_PATTERN.sub("X", sequence), replacements


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute percentile of empty values")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def robust_center_scale(values: Sequence[float]) -> Tuple[float, float, str]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        raise ValueError("empty robust-stat group")
    center = statistics.median(clean)
    if len(clean) == 1:
        return center, 1.0, "singleton_unit_scale"
    mad = statistics.median(abs(value - center) for value in clean)
    scale = MAD_NORMALIZER * mad
    if scale > EPSILON:
        return center, scale, "median_mad"
    if len(clean) >= 4:
        scale = (percentile(clean, 0.75) - percentile(clean, 0.25)) / IQR_NORMALIZER
        if scale > EPSILON:
            return center, scale, "median_iqr_fallback"
    scale = statistics.pstdev(clean)
    if scale > EPSILON:
        return center, scale, "median_population_std_fallback"
    return center, 1.0, "constant_unit_scale"


def clipped_z(value: float, center: float, scale: float) -> float:
    return max(-Z_CLIP, min(Z_CLIP, (value - center) / max(scale, EPSILON)))


def source_paths() -> Dict[str, Path]:
    return {
        "delepref_20_protocol": V20 / "protocol.json",
        "delepref_20_run_summary": V20 / "run_summary.json",
        "delepref_20_audit_manifest": V20 / "audit_manifest.json",
        "delepref_20_source_hashes": V20 / "source_hashes.json",
        "delepref_20_input_manifest": V20 / "input_manifest.csv",
        "delepref_20_candidate_reconstruction": V20 / "candidate_reconstruction_audit.csv",
        "delepref_20_cath_join": V20 / "cath_join_audit.csv",
        "delepref_20_esm2_views": V20 / "esm2_cross_teacher_candidate_views.csv",
        "delepref_20_direct_scores": V20 / "direct_plm_candidate_scores.csv",
        "delepref_20_direct_selected": V20 / "direct_plm_selected_plans.csv",
        "delepref_20_esm2_selected": V20 / "esm2_proxy_selected_plans.csv",
        "delepref_20_group_metrics": V20 / "direct_plm_and_cross_teacher_group_metrics.csv",
        "delepref_20_model_provenance": V20 / "model_provenance.json",
        "delepref_20_independent_validation": V20 / "independent_validation.json",
        "delepref_20_pass_sentinel": V20 / "PASS_DelePref_20_DIRECT_PLM_CROSS_TEACHER",
        "delepref_7_teacher_view_definitions": DelePref7 / "teacher_view_definitions.json",
        "delepref_7_teacher_view_scoring_source": REPO / "scripts/teacher_scoring.py",
        "preference_robust_scaling_source": REPO / "scripts/build_pair_same_count.py",
        "delepref_frozen_checkpoint_hash_only": CHECKPOINT,
        "protected_protocol_source": PROTOCOL_SOURCE,
    }


def build_input_hash_validation() -> List[Dict[str, Any]]:
    paths = source_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"required frozen source files are missing: {missing}")
    v20_manifest = json.loads((V20 / "audit_manifest.json").read_text(encoding="utf-8"))
    manifest_hashes = {item["path"]: item["sha256"] for item in v20_manifest["artifacts"]}
    rows: List[Dict[str, Any]] = []

    def add(name: str, path: Path, expected: str, authority: str, used: bool = True) -> None:
        current = sha256_file(path)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "size": path.stat().st_size,
                "expected_sha256": expected,
                "current_sha256": current,
                "authority": authority,
                "used_by_delepref_21": used,
                "status": "PASS" if current == expected else "FAIL",
            }
        )

    for name, path in paths.items():
        if str(path).startswith(str(V20)):
            relative = str(path.relative_to(V20))
            expected = manifest_hashes.get(relative, sha256_file(path))
            authority = "DelePref-20 audit_manifest" if relative in manifest_hashes else "DelePref-21 preflight self-pin for DelePref-20 manifest/sentinel"
        elif path == CHECKPOINT:
            expected, authority = EXPECTED_CHECKPOINT_SHA256, "frozen DelePref evaluation checkpoint provenance"
        elif path == DelePref7 / "teacher_view_definitions.json":
            expected, authority = EXPECTED_DelePref7_DEFINITION_SHA256, "DelePref-7 authoritative teacher-view definition"
        else:
            expected, authority = sha256_file(path), "DelePref-21 preflight code/document pin"
        add(name, path, expected, authority)

    v20_sources = json.loads((V20 / "source_hashes.json").read_text(encoding="utf-8"))
    for item in v20_sources["sources"]:
        path = Path(item["path"])
        if not path.is_file():
            raise RuntimeError(f"DelePref-20 transitive source missing: {path}")
        add("delepref_20_transitive_source:" + item["name"], path, item["sha256"], "DelePref-20 source_hashes.json", False)
    for item in v20_sources["model"]["files"]:
        path = Path(item["path"])
        if not path.is_file():
            raise RuntimeError(f"DelePref-20 ESM2 model file missing: {path}")
        add("delepref_20_esm2_model_file:" + item["name"], path, item["sha256"], "DelePref-20 source_hashes model inventory", False)

    if any(row["status"] != "PASS" for row in rows):
        failed = [row["name"] for row in rows if row["status"] != "PASS"]
        raise RuntimeError(f"frozen input hash validation failed: {failed}")
    if any("delepref_14" in row["path"].lower() for row in rows):
        raise RuntimeError("DelePref-14 path appeared in DelePref-21 source inventory")
    write_csv(
        OUT / "input_hash_validation.csv",
        ("name", "path", "size", "expected_sha256", "current_sha256", "authority", "used_by_delepref_21", "status"),
        rows,
    )
    return rows


def validate_v20_package_status() -> Dict[str, Any]:
    protocol = json.loads((V20 / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((V20 / "run_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((V20 / "independent_validation.json").read_text(encoding="utf-8"))
    audit = json.loads((V20 / "audit_manifest.json").read_text(encoding="utf-8"))
    source_hashes = json.loads((V20 / "source_hashes.json").read_text(encoding="utf-8"))
    checks = {
        "pass_sentinel_exact": (V20 / "PASS_DelePref_20_DIRECT_PLM_CROSS_TEACHER").read_text(encoding="utf-8").strip() == "PASS_DelePref_20_DIRECT_PLM_CROSS_TEACHER",
        "run_summary_pass": summary.get("status") == "PASS_DelePref_20_DIRECT_PLM_CROSS_TEACHER",
        "protocol_id": protocol.get("protocol_id") == "DelePref_20_DIRECT_PLM_CROSS_TEACHER",
        "protocol_hash": sha256_file(V20 / "protocol.json") == (V20 / "protocol.sha256").read_text(encoding="utf-8").strip(),
        "independent_validation_pass": validation.get("status") == "PASS",
        "audit_manifest_pass": audit.get("status") == "PASS",
        "source_hashes_counts": source_hashes.get("cohort") == {"K": 8, "candidates": 24000, "groups": 3000, "proteins": 500, "seed": "101011"},
        "protocol_counts": protocol.get("cohort", {}).get("protein_count") == PROTEINS and protocol.get("cohort", {}).get("group_count") == GROUPS and protocol.get("cohort", {}).get("candidate_count") == CANDIDATES and protocol.get("cohort", {}).get("K") == K,
        "summary_counts": summary.get("cohort", {}).get("protein_count") == PROTEINS and summary.get("cohort", {}).get("group_count") == GROUPS and summary.get("cohort", {}).get("candidate_count") == CANDIDATES and summary.get("cohort", {}).get("K") == K,
    }
    if not all(checks.values()):
        raise RuntimeError("DelePref-20 package status gate failed: " + json.dumps(checks, sort_keys=True))
    return checks


def validate_frozen_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    manifest = read_csv(V20 / "input_manifest.csv")
    reconstruction = {row["candidate_id"]: row for row in read_csv(V20 / "candidate_reconstruction_audit.csv")}
    cath = {row["candidate_id"]: row for row in read_csv(V20 / "cath_join_audit.csv")}
    esm2 = {row["candidate_id"]: row for row in read_csv(V20 / "esm2_cross_teacher_candidate_views.csv")}
    direct = {row["candidate_id"]: row for row in read_csv(V20 / "direct_plm_candidate_scores.csv")}
    expected_ids = {row["candidate_id"] for row in manifest}
    if len(manifest) != CANDIDATES or len(expected_ids) != CANDIDATES:
        raise RuntimeError("DelePref-20 input manifest does not have 24,000 unique candidate IDs")
    if any(set(index) != expected_ids for index in (reconstruction, cath, esm2, direct)):
        raise RuntimeError("DelePref-20 authoritative candidate inventories differ")

    groups: MutableMapping[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    proteins: Dict[str, Tuple[str, str]] = {}
    normalized_rows: List[Dict[str, Any]] = []
    for raw in manifest:
        candidate_id = raw["candidate_id"]
        protein = raw["protein_id"]
        method = raw["method"]
        budget = f"{float(raw['budget']):.2f}"
        if method not in METHODS or budget not in BUDGETS or raw["generation_seed"] != GENERATION_SEED:
            raise RuntimeError(f"out-of-scope method/budget/seed for {candidate_id}")
        if raw["coordinate_system"] != COORDINATES:
            raise RuntimeError(f"coordinate-system mismatch for {candidate_id}")
        sequence = raw["wildtype_sequence"]
        length = int(raw["sequence_length"])
        if len(sequence) != length or sha256_text(sequence) != raw["sequence_sha256"]:
            raise RuntimeError(f"WT sequence identity mismatch for {candidate_id}")
        intervals = parse_intervals(raw["deleted_intervals"])
        if intervals[-1][1] >= length or canonical_intervals(intervals) != raw["deleted_intervals"]:
            raise RuntimeError(f"interval identity mismatch for {candidate_id}")
        edited = apply_deletion(sequence, intervals)
        if edited != raw["edited_sequence"] or sha256_text(edited) != raw["edited_sequence_sha256"]:
            raise RuntimeError(f"edited sequence mismatch for {candidate_id}")
        if mask_hash(length, intervals) != raw["deletion_mask_sha256"]:
            raise RuntimeError(f"deletion mask hash mismatch for {candidate_id}")
        if deleted_length(intervals) != int(raw["deleted_length"]) or len(edited) != int(raw["edited_length"]):
            raise RuntimeError(f"deletion length mismatch for {candidate_id}")
        g, r, j, utility = (finite(raw[name]) for name in ("global_similarity", "retained_similarity", "junction_plausibility", "teacher_utility"))
        if not all(math.isfinite(value) for value in (g, r, j, utility)) or abs(0.35 * g + 0.45 * r + 0.20 * j - utility) > 2.0e-6:
            raise RuntimeError(f"frozen ESMC utility mismatch for {candidate_id}")
        if not math.isfinite(finite(raw["scorer_score"])):
            raise RuntimeError(f"missing frozen DelePref score for {candidate_id}")
        rec, cath_row, esm2_row, direct_row = reconstruction[candidate_id], cath[candidate_id], esm2[candidate_id], direct[candidate_id]
        if rec.get("status") != "PASS" or not all(truthy(rec[name]) for name in ("intervals_valid", "mask_hash_valid", "sequence_hash_valid", "edited_reconstruction_valid")):
            raise RuntimeError(f"DelePref-20 reconstruction audit failed for {candidate_id}")
        if cath_row.get("status") != "PASS" or not truthy(cath_row.get("cath_joined")):
            raise RuntimeError(f"DelePref-20 CATH join failed for {candidate_id}")
        identity_fields = ("protein_id", "sequence_sha256", "candidate_id", "deleted_intervals", "deletion_mask_sha256", "edited_sequence_sha256")
        for source_row in (esm2_row, direct_row):
            if any(source_row.get(field) != raw.get(field) for field in identity_fields):
                raise RuntimeError(f"DelePref-20 diagnostic identity mismatch for {candidate_id}")
        if not math.isfinite(finite(esm2_row.get("esm2_proxy_utility"))) or not math.isfinite(finite(direct_row.get("direct_delete_nll_mean"))):
            raise RuntimeError(f"missing frozen DelePref-20 diagnostics for {candidate_id}")
        if proteins.setdefault(protein, (sequence, raw["sequence_sha256"])) != (sequence, raw["sequence_sha256"]):
            raise RuntimeError(f"inconsistent WT identity within protein {protein}")
        row: Dict[str, Any] = dict(raw)
        row.update(
            {
                "budget": budget,
                "esm2_proxy_utility": esm2_row["esm2_proxy_utility"],
                "direct_delete_nll_mean": direct_row["direct_delete_nll_mean"],
            }
        )
        normalized_rows.append(row)
        groups[(protein, budget, method)].append(row)

    if len(proteins) != PROTEINS or len(groups) != GROUPS:
        raise RuntimeError(f"frozen cohort count mismatch proteins={len(proteins)} groups={len(groups)}")
    if any(len(group) != K or len({row["deletion_mask_sha256"] for row in group}) != K for group in groups.values()):
        raise RuntimeError("one or more frozen groups are not unique-mask K=8")
    for group in groups.values():
        ranked = sorted(group, key=lambda row: int(row["scorer_rank"]))
        if [int(row["scorer_rank"]) for row in ranked] != list(range(1, K + 1)):
            raise RuntimeError("frozen scorer rank permutation is invalid")
    normalized_rows.sort(key=lambda row: int(row["manifest_index"]))
    if [int(row["manifest_index"]) for row in normalized_rows] != list(range(CANDIDATES)):
        raise RuntimeError("DelePref-20 canonical manifest indexes are not complete")
    diagnostics = {
        "protein_count": len(proteins),
        "group_count": len(groups),
        "candidate_count": len(normalized_rows),
        "K": K,
        "methods": dict(Counter(row["method"] for row in normalized_rows)),
        "budgets": dict(Counter(row["budget"] for row in normalized_rows)),
        "generation_seeds": dict(Counter(row["generation_seed"] for row in normalized_rows)),
        "unique_candidate_ids": len({row["candidate_id"] for row in normalized_rows}),
        "unique_group_masks": True,
        "authoritative_candidate_sequence_source": str(V20 / "input_manifest.csv"),
    }
    return normalized_rows, diagnostics


def load_tokenizer():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(VENDOR_SITE))
    from transformers import T5Tokenizer  # type: ignore

    provenance = json.loads((OUT / "model_provenance.json").read_text(encoding="utf-8"))
    snapshot = Path(provenance["snapshot"]["snapshot_path"])
    return T5Tokenizer.from_pretrained(str(snapshot), local_files_only=True, do_lower_case=False), snapshot


def tokenization_record(tokenizer: Any, kind: str, protein: str, candidate_id: str, sequence: str, sequence_hash: str) -> Dict[str, Any]:
    normalized, replacements = normalize_sequence(sequence)
    spaced = " ".join(normalized)
    encoded = tokenizer(spaced, add_special_tokens=True, truncation=False, return_attention_mask=True)
    ids = [int(value) for value in encoded["input_ids"]]
    attention = [int(value) for value in encoded["attention_mask"]]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    if len(ids) != len(normalized) + 1 or len(attention) != len(ids) or sum(attention) != len(ids):
        raise RuntimeError(f"tokenizer residue/no-truncation contract failed for {kind}:{protein}:{candidate_id}")
    if tokens[-1] != "</s>" or any(token == "<unk>" for token in tokens[:-1]):
        raise RuntimeError(f"tokenizer unknown/EOS contract failed for {kind}:{protein}:{candidate_id}")
    return {
        "record_kind": kind,
        "protein_id": protein,
        "candidate_id": candidate_id,
        "original_sequence_sha256": sequence_hash,
        "normalized_sequence_sha256": sha256_text(normalized),
        "original_length": len(sequence),
        "normalized_length": len(normalized),
        "normalization_changed": normalized != sequence,
        "replacement_count": len(replacements),
        "replacements_json": json.dumps(replacements, separators=(",", ":")),
        "preprocessing": "single spaces between normalized amino-acid tokens",
        "token_count_with_eos": len(ids),
        "residue_token_count": len(ids) - 1,
        "token_ids_sha256": sha256_text(json.dumps(ids, separators=(",", ":"))),
        "eos_token_id": ids[-1],
        "truncation": False,
        "status": "PASS",
        "_normalized_sequence": normalized,
        "_token_ids": ids,
    }


def tokenization_preflight(rows: Sequence[MutableMapping[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    tokenizer, snapshot = load_tokenizer()
    wt_records: Dict[str, Dict[str, Any]] = {}
    candidate_records: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        protein = row["protein_id"]
        if protein not in wt_records:
            wt_records[protein] = tokenization_record(tokenizer, "WT", protein, "", row["wildtype_sequence"], row["sequence_sha256"])
        record = tokenization_record(tokenizer, "edited_candidate", protein, row["candidate_id"], row["edited_sequence"], row["edited_sequence_sha256"])
        candidate_records[row["candidate_id"]] = record
        row["_normalized_wt_sequence"] = wt_records[protein]["_normalized_sequence"]
        row["_normalized_edited_sequence"] = record["_normalized_sequence"]
        row["_wt_token_ids_sha256"] = wt_records[protein]["token_ids_sha256"]
        row["_edited_token_ids_sha256"] = record["token_ids_sha256"]
    output_records = []
    for record in list(wt_records.values()) + list(candidate_records.values()):
        output_records.append({key: value for key, value in record.items() if not key.startswith("_")})
    columns = (
        "record_kind", "protein_id", "candidate_id", "original_sequence_sha256", "normalized_sequence_sha256",
        "original_length", "normalized_length", "normalization_changed", "replacement_count", "replacements_json",
        "preprocessing", "token_count_with_eos", "residue_token_count", "token_ids_sha256", "eos_token_id",
        "truncation", "status",
    )
    write_csv(OUT / "input_tokenization_preflight.csv", columns, output_records)
    summary = {
        "snapshot": str(snapshot),
        "wt_records": len(wt_records),
        "candidate_records": len(candidate_records),
        "total_records": len(output_records),
        "normalization_changed_records": sum(truthy(record["normalization_changed"]) for record in output_records),
        "total_uzob_replacements": sum(int(record["replacement_count"]) for record in output_records),
        "max_original_length": max(int(record["original_length"]) for record in output_records),
        "max_token_count_with_eos": max(int(record["token_count_with_eos"]) for record in output_records),
        "all_untruncated": all(not truthy(record["truncation"]) for record in output_records),
        "tokenization_preflight_sha256": sha256_file(OUT / "input_tokenization_preflight.csv"),
    }
    del tokenizer
    return wt_records, candidate_records, summary


def nvidia_query() -> Dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode != 0:
        raise RuntimeError("nvidia-smi query failed: " + query.stderr)
    gpus = []
    fields = ("index", "name", "uuid", "driver_version", "memory_total_mib", "memory_free_mib", "memory_used_mib", "utilization_gpu_percent")
    for line in query.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != len(fields):
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        gpus.append(dict(zip(fields, values)))
    process_query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    processes = [line.strip() for line in process_query.stdout.splitlines() if line.strip()]
    return {"gpus": gpus, "compute_processes": processes}


def resource_preflight() -> Dict[str, Any]:
    import torch  # type: ignore
    import transformers  # type: ignore

    gpu = nvidia_query()
    env = {key: os.environ.get(key, "") for key in ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "CONDA_DEFAULT_ENV", "CONDA_PREFIX")}
    disk = shutil.disk_usage(REPO)
    one = len(gpu["gpus"]) == 1
    row = gpu["gpus"][0] if one else {}
    checks = {
        "bootstrap_complete": json.loads((OUT / "bootstrap_status.json").read_text(encoding="utf-8")).get("status") == "PASS_BOOTSTRAP_COMPLETE",
        "protocol_not_yet_created": not (OUT / "protocol.json").exists(),
        "formal_science_not_started": not (OUT / "formal_scientific_started.json").exists(),
        "one_visible_gpu_smi": one,
        "public_free_at_least_40_gib": disk.free >= 40 * 1024**3,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_one_gpu": int(torch.cuda.device_count()) == 1,
        "torch_gpu_detected": bool(torch.cuda.is_available()) and bool(torch.cuda.get_device_name(0)),
        "hf_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
    }
    if not all(checks.values()):
        raise RuntimeError("scientific resource preflight failed: " + json.dumps(checks, sort_keys=True))
    result = {
        "schema_version": "delepref_21_gpu_resource_provenance_v1",
        "status": "PASS_PREFLIGHT",
        "phase": "scientific_preflight",
        "created_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "environment": env,
        "nvidia_smi": gpu,
        "disk": {"path": str(REPO), "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free, "free_gib": disk.free / 1024**3},
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "checks": checks,
        "bootstrap_offline_load_check_sha256": sha256_file(OUT / "audit/bootstrap_offline_load_check.json"),
    }
    atomic_json(OUT / "gpu_resource_provenance.json", result)
    return result


def protocol_lock(
    rows: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    hash_rows: Sequence[Mapping[str, Any]],
    token_summary: Mapping[str, Any],
    resource: Mapping[str, Any],
) -> Dict[str, Any]:
    if (OUT / "protocol.json").exists() or (OUT / "protocol.sha256").exists():
        raise RuntimeError("DelePref-21 protocol already exists; refusing to relock")
    model_provenance = json.loads((OUT / "model_provenance.json").read_text(encoding="utf-8"))
    vendor_provenance = json.loads((OUT / "runtime_vendor_provenance.json").read_text(encoding="utf-8"))
    if model_provenance.get("status") != "PASS_BOOTSTRAP_COMPLETE":
        raise RuntimeError("ProtT5 bootstrap is not complete")
    if model_provenance["snapshot"]["aggregate_sha256"] != EXPECTED_SNAPSHOT_SHA256 or vendor_provenance["inventory"]["aggregate_sha256"] != EXPECTED_VENDOR_SHA256:
        raise RuntimeError("model or task-local runtime hash changed after bootstrap")
    definitions = json.loads((DelePref7 / "teacher_view_definitions.json").read_text(encoding="utf-8"))
    canonical = definitions["views"]["canonical_full"]["weights"]
    if (canonical["global_similarity"], canonical["retained_similarity"], canonical["junction_plausibility"]) != VIEW_WEIGHTS:
        raise RuntimeError("DelePref-7 canonical view weights do not match locked DelePref-21 weights")
    protocol = {
        "schema_version": "delepref_21_prott5_crossfamily_audit_v1",
        "protocol_id": "DelePref_21_PROTT5_CROSSFAMILY_AUDIT",
        "created_at_utc": utc_now(),
        "claim_boundary": CLAIM_BOUNDARY,
        "scientific_label": "frozen cross-family representation-proxy alignment audit",
        "cohort": {
            "protein_count": PROTEINS,
            "group_count": GROUPS,
            "candidate_count": CANDIDATES,
            "K": K,
            "methods": list(METHODS),
            "budgets": [0.10, 0.20],
            "generation_seed": 101011,
            "diagnostics": diagnostics,
        },
        "authoritative_candidate_source": str(V20 / "input_manifest.csv"),
        "authoritative_input_hashes": [dict(row) for row in hash_rows],
        "input_hash_validation_sha256": sha256_file(OUT / "input_hash_validation.csv"),
        "delepref_7_teacher_view_definitions": {
            "path": str(DelePref7 / "teacher_view_definitions.json"),
            "sha256": EXPECTED_DelePref7_DEFINITION_SHA256,
            "component_transforms": definitions["component_transforms"],
            "canonical_weights": canonical,
        },
        "model": {
            "repo_id": REPO_ID,
            "full_revision": REVISION,
            "snapshot_path": model_provenance["snapshot"]["snapshot_path"],
            "snapshot_sha256": model_provenance["snapshot"]["aggregate_sha256"],
            "files": model_provenance["snapshot"]["files"],
            "architecture": "T5EncoderModel",
            "dtype": "frozen fp16",
            "eval": True,
            "inference_mode": True,
            "local_files_only": True,
            "offline": True,
            "training": False,
            "quantized": False,
            "adapter": False,
            "lora": False,
            "decoder": False,
        },
        "runtime_vendor": {
            "sentencepiece_version": vendor_provenance["imported_version"],
            "sentencepiece_wheel_sha256": vendor_provenance["wheel"]["sha256"],
            "install_root": vendor_provenance["install_root"],
            "aggregate_sha256": vendor_provenance["inventory"]["aggregate_sha256"],
            "base_environment_modified": False,
        },
        "tokenization": {
            "model_card_preprocessing": "amino-acid tokens separated by one ASCII space",
            "normalization": "replace each U/Z/O/B residue with X; preserve every other residue",
            "truncation": False,
            "cropping": False,
            "dropping": False,
            "special_tokens": "one terminal </s>; no residue token omitted",
            "preflight": dict(token_summary),
        },
        "coordinates": COORDINATES,
        "retained_mapping": "edited_index = original_index - count(deleted positions < original_index); only retained WT positions",
        "proxy_view_definitions": {
            "global_shift": "clamp01((1 - cosine(mean(WT residue reps), mean(edited residue reps))) / 2)",
            "global_similarity": "clamp01(1 - 2*global_shift)",
            "retained_region_shift": "mean clamp01((1-cosine(mapped WT residue, edited residue))/2) over retained residues",
            "retained_similarity": "clamp01(1 - 2*retained_region_shift)",
            "junction_values": "two-sided: clamp01(1-left_shift-right_shift); terminal: clamp01(1-2*anchor_shift); no anchor: 0",
            "junction_plausibility": "single span value; multi-span clamp01(0.5*mean(values)+0.5*min(values))",
            "prott5_representation_proxy_utility": "0.35*global_similarity + 0.45*retained_similarity + 0.20*junction_plausibility",
            "representation_storage": "raw fp16 residue representations; all cosine and aggregate calculations in float32",
        },
        "robust_scaling": {
            "scope": "within each (protein_id, exact deleted_length) group, equivalent to per-protein/budget for this frozen cohort",
            "ordering": "absolute prott5_representation_proxy_utility; robust scaling never changes candidate order",
            "center": "median",
            "primary_scale": "1.4826 * median absolute deviation",
            "fallbacks": ["IQR / 1.349 when n>=4", "population standard deviation", "unit scale for singleton/constant group"],
            "epsilon": EPSILON,
            "z_clip": [-Z_CLIP, Z_CLIP],
        },
        "ranking_and_missingness": {
            "direction": "higher is preferred for ESMC utility, DelePref score, Direct-PLM NLL, ESM2 proxy, and ProtT5 proxy",
            "tie_rule": f"values within absolute tolerance {TIE_ATOL:.1e} are tied; top-1 is the complete candidate-id set",
            "no_arbitrary_tie_break": True,
            "spearman": "average ranks for ties; pairwise-finite candidates only; NA when fewer than 2 or either variable is constant",
            "missingness": "retain NA, no-eligible, ties, and disagreement; never zero-fill or drop a group",
            "pairwise_order": "28 unordered candidate pairs per K=8 group; agreement only when reference and ProtT5 ternary order signs match",
            "contextual_tie_selection": "mean CATH metric across every candidate in the selector's tie-aware top-1 set",
        },
        "batching": {
            "deterministic_sort": "(floor(residue_length/32), residue_length, normalized_sequence_sha256, candidate_id)",
            "dynamic_batch_size": f"max(1,min({MAX_BATCH_SIZE},floor({TARGET_TOKENS_SQUARED}/max_token_length^2)))",
            "max_batch_size": MAX_BATCH_SIZE,
            "target_tokens_squared": TARGET_TOKENS_SQUARED,
            "padding": True,
            "raw_shard_fsync_before_aggregate": True,
        },
        "statistics": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "protein_id only",
            "ci": "percentile 95% with NumPy linear quantiles",
            "candidate_scopes": "six method-by-budget strata and ALL",
            "group_macro": "protein macro mean",
            "contextual_status": "NA_DIFFERENT_PROTOCOL; descriptive only",
        },
        "resource": {
            "hostname": resource["hostname"],
            "gpu_name": resource["nvidia_smi"]["gpus"][0]["name"],
            "cuda_visible_devices": resource["environment"]["CUDA_VISIBLE_DEVICES"],
            "torch_version": resource["torch_version"],
            "transformers_version": resource["transformers_version"],
        },
        "prohibitions_and_declarations": {
            "training": False,
            "student_checkpoint_load": False,
            "student_scorer_inference": False,
            "teacher_rescoring": False,
            "candidate_regeneration": False,
            "proposal_inference": False,
            "delepref_14_opened": False,
            "network_after_bootstrap": False,
            "cpu_representation_fallback": False,
            "zero_fill": False,
            "second_gpu": False,
            "second_formal_gpu_retry": False,
        },
        "protected_protocol_source_sha256_before": sha256_file(PROTOCOL_SOURCE),
        "code_hashes": {
            "runner": sha256_file(SCRIPT),
            "bootstrap": sha256_file(BOOTSTRAP_SCRIPT),
            "validator": sha256_file(VALIDATOR),
        },
        "safety": {
            "prott5_role": "cross-ESM-family frozen sequence-representation proxy",
            "not_truth": ["function", "stability", "binding", "retained function", "ground-truth teacher", "training label"],
            "cannot_prove": ["a deleted region lacks function", "functional preservation", "retained binding"],
            "retain_negative_mixed_results": True,
        },
    }
    atomic_json(OUT / "protocol.json", protocol)
    atomic_text(OUT / "protocol.sha256", sha256_file(OUT / "protocol.json") + "\n")
    atomic_json(
        OUT / "protocol_locked.json",
        {
            "status": "LOCKED_BEFORE_ANY_CANDIDATE_REPRESENTATION_FORWARD",
            "protocol_sha256": sha256_file(OUT / "protocol.json"),
            "input_hash_validation_sha256": sha256_file(OUT / "input_hash_validation.csv"),
            "tokenization_preflight_sha256": sha256_file(OUT / "input_tokenization_preflight.csv"),
            "created_at_utc": utc_now(),
        },
    )
    return protocol


def dynamic_batch(max_residues: int) -> int:
    token_length = max(1, int(max_residues) + 1)
    return max(1, min(MAX_BATCH_SIZE, TARGET_TOKENS_SQUARED // (token_length * token_length)))


def load_model():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(VENDOR_SITE))
    import torch  # type: ignore
    from transformers import T5EncoderModel, T5Tokenizer  # type: ignore

    provenance = json.loads((OUT / "model_provenance.json").read_text(encoding="utf-8"))
    snapshot = Path(provenance["snapshot"]["snapshot_path"])
    tokenizer = T5Tokenizer.from_pretrained(str(snapshot), local_files_only=True, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(str(snapshot), local_files_only=True, torch_dtype=torch.float16)
    model.eval()
    model.to(torch.device("cuda:0"))
    if model.__class__.__name__ != "T5EncoderModel" or model.training or next(model.parameters()).dtype != torch.float16:
        raise RuntimeError("loaded model violates the locked frozen fp16 encoder contract")
    return torch, tokenizer, model, torch.device("cuda:0"), snapshot


def embedding_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float16)
    return sha256_bytes(contiguous.tobytes(order="C"))


def encode_records(torch: Any, tokenizer: Any, model: Any, device: Any, records: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, np.ndarray], List[int]]:
    ordered = sorted(
        records,
        key=lambda record: (
            len(record["normalized_sequence"]) // 32,
            len(record["normalized_sequence"]),
            record["normalized_sequence_sha256"],
            record["record_id"],
        ),
    )
    output: Dict[str, np.ndarray] = {}
    batch_sizes: List[int] = []
    cursor = 0
    while cursor < len(ordered):
        bucket = len(ordered[cursor]["normalized_sequence"]) // 32
        end = cursor + 1
        while end < len(ordered) and len(ordered[end]["normalized_sequence"]) // 32 == bucket:
            end += 1
        bucket_rows = ordered[cursor:end]
        bucket_cursor = 0
        while bucket_cursor < len(bucket_rows):
            remaining = bucket_rows[bucket_cursor:]
            max_length = max(len(record["normalized_sequence"]) for record in remaining[:MAX_BATCH_SIZE])
            batch_size = dynamic_batch(max_length)
            batch = remaining[:batch_size]
            sequences = [" ".join(record["normalized_sequence"]) for record in batch]
            encoded = tokenizer(sequences, add_special_tokens=True, truncation=False, padding=True, return_tensors="pt")
            for index, record in enumerate(batch):
                active = int(encoded["attention_mask"][index].sum().item())
                ids = encoded["input_ids"][index, :active].tolist()
                if active != len(record["normalized_sequence"]) + 1 or sha256_text(json.dumps(ids, separators=(",", ":"))) != record["token_ids_sha256"]:
                    raise RuntimeError(f"formal tokenization mismatch for {record['record_id']}")
            inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
            hidden_cpu = hidden.detach().cpu()
            for index, record in enumerate(batch):
                length = len(record["normalized_sequence"])
                residue = hidden_cpu[index, :length, :].contiguous().numpy().astype(np.float16, copy=True)
                if residue.shape[0] != length or residue.ndim != 2 or not np.isfinite(residue).all():
                    raise RuntimeError(f"invalid ProtT5 residue representation for {record['record_id']}")
                output[record["record_id"]] = residue
            batch_sizes.append(len(batch))
            del inputs, encoded, hidden, hidden_cpu
            bucket_cursor += len(batch)
        cursor = end
    if len(output) != len(records):
        raise RuntimeError("representation output record inventory mismatch")
    return output, batch_sizes


def cosine_shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return math.nan, math.nan
    cosine = float(np.dot(left, right) / denominator)
    shift = float(np.clip((1.0 - cosine) / 2.0, 0.0, 1.0))
    similarity = float(np.clip(1.0 - 2.0 * shift, 0.0, 1.0))
    return shift, similarity


def compute_proxy_view(row: Mapping[str, Any], wt: np.ndarray, edited: np.ndarray) -> Dict[str, Any]:
    intervals = parse_intervals(row["deleted_intervals"])
    length = int(row["sequence_length"])
    mapping = retained_mapping(length, intervals)
    if wt.shape[0] != length or edited.shape[0] != int(row["edited_length"]) or len(mapping) != edited.shape[0]:
        raise RuntimeError(f"representation/retained mapping mismatch for {row['candidate_id']}")
    global_shift, global_similarity = cosine_shift(wt.astype(np.float32).mean(axis=0), edited.astype(np.float32).mean(axis=0))
    retained_shifts = [cosine_shift(wt[original], edited[mutant])[0] for original, mutant in mapping]
    retained_shift = float(np.mean(np.asarray(retained_shifts, dtype=np.float32))) if retained_shifts else 1.0
    retained_similarity = float(np.clip(1.0 - 2.0 * retained_shift, 0.0, 1.0))
    mapping_dict = dict(mapping)
    junction_values: List[float] = []
    junction_details: List[Dict[str, Any]] = []
    for start, end in intervals:
        left, right = start - 1, end + 1
        if left in mapping_dict and right in mapping_dict:
            left_shift, left_similarity = cosine_shift(wt[left], edited[mapping_dict[left]])
            right_shift, right_similarity = cosine_shift(wt[right], edited[mapping_dict[right]])
            value = float(np.clip(1.0 - left_shift - right_shift, 0.0, 1.0))
            detail = {"aggregation": "two_sided", "left_original": left, "right_original": right, "left_shift": left_shift, "right_shift": right_shift, "left_similarity": left_similarity, "right_similarity": right_similarity}
        elif left in mapping_dict or right in mapping_dict:
            anchor = left if left in mapping_dict else right
            anchor_shift, anchor_similarity = cosine_shift(wt[anchor], edited[mapping_dict[anchor]])
            value = float(np.clip(1.0 - 2.0 * anchor_shift, 0.0, 1.0))
            detail = {"aggregation": "one_sided_terminal", "anchor_original": anchor, "anchor_shift": anchor_shift, "anchor_similarity": anchor_similarity}
        else:
            value = 0.0
            detail = {"aggregation": "no_retained_anchor"}
        junction_values.append(value)
        junction_details.append(detail)
    junction_mean = float(np.mean(junction_values)) if junction_values else 0.0
    junction_min = float(min(junction_values)) if junction_values else 0.0
    junction = float(np.clip(junction_values[0] if len(junction_values) == 1 else 0.5 * junction_mean + 0.5 * junction_min, 0.0, 1.0))
    utility = VIEW_WEIGHTS[0] * global_similarity + VIEW_WEIGHTS[1] * retained_similarity + VIEW_WEIGHTS[2] * junction
    values = (global_shift, global_similarity, retained_shift, retained_similarity, junction, utility)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"non-finite ProtT5 proxy view for {row['candidate_id']}")
    return {
        "prott5_global_shift": fmt(global_shift),
        "prott5_global_similarity": fmt(global_similarity),
        "prott5_retained_region_shift": fmt(retained_shift),
        "prott5_retained_similarity": fmt(retained_similarity),
        "prott5_junction_values_json": json.dumps(junction_values, separators=(",", ":")),
        "prott5_junction_detail_json": json.dumps(junction_details, separators=(",", ":"), sort_keys=True),
        "prott5_junction_mean": fmt(junction_mean),
        "prott5_junction_min": fmt(junction_min),
        "prott5_junction_plausibility": fmt(junction),
        "prott5_representation_proxy_utility": fmt(utility),
        "prott5_view_status": "PASS",
    }


def write_npz_atomic(path: Path, **arrays: Any) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


VIEW_IDENTITY_COLUMNS = (
    "manifest_index", "protein_id", "candidate_id", "budget", "method", "source_method", "generation_seed",
    "sequence_sha256", "edited_sequence_sha256", "deletion_mask_sha256", "deleted_intervals", "deleted_length",
    "span_count", "scorer_score", "teacher_utility", "direct_delete_nll_mean", "esm2_proxy_utility",
    "cath_domain_core_deletion_content_ratio", "cath_core_damage", "cath_boundary_overlap", "cath_span_count", "cath_fragmentation",
)
PROXY_VIEW_COLUMNS = (
    "prott5_global_shift", "prott5_global_similarity", "prott5_retained_region_shift", "prott5_retained_similarity",
    "prott5_junction_values_json", "prott5_junction_detail_json", "prott5_junction_mean", "prott5_junction_min",
    "prott5_junction_plausibility", "prott5_representation_proxy_utility", "prott5_view_status",
)


def formal_representation_pass(
    rows: Sequence[MutableMapping[str, Any]],
    wt_tokens: Mapping[str, Mapping[str, Any]],
    candidate_tokens: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    marker = OUT / "formal_scientific_started.json"
    if marker.exists() or (OUT / "formal_gpu_representation_complete.json").exists():
        raise RuntimeError("formal DelePref-21 GPU science already started; retry is forbidden")
    if sha256_file(OUT / "protocol.json") != (OUT / "protocol.sha256").read_text(encoding="utf-8").strip():
        raise RuntimeError("protocol hash mismatch immediately before formal science")
    atomic_json(
        marker,
        {
            "status": "STARTED",
            "started_at_utc": utc_now(),
            "protocol_sha256": sha256_file(OUT / "protocol.json"),
            "one_formal_launcher_only": True,
            "second_gpu_retry_forbidden": True,
        },
    )
    started = time.time()
    try:
        torch, tokenizer, model, device, snapshot = load_model()
        model_provenance = json.loads((OUT / "model_provenance.json").read_text(encoding="utf-8"))
        before_inventory = model_provenance["snapshot"]
        if before_inventory["aggregate_sha256"] != protocol["model"]["snapshot_sha256"]:
            raise RuntimeError("model snapshot changed before formal forward")
        torch.manual_seed(20260722)
        torch.cuda.manual_seed_all(20260722)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats(device)

        by_protein: MutableMapping[str, List[MutableMapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_protein[row["protein_id"]].append(row)
        proteins = sorted(by_protein)
        wt_encode_records = []
        for protein in proteins:
            token = wt_tokens[protein]
            wt_encode_records.append(
                {
                    "record_id": protein,
                    "normalized_sequence": token["_normalized_sequence"],
                    "normalized_sequence_sha256": token["normalized_sequence_sha256"],
                    "token_ids_sha256": token["token_ids_sha256"],
                }
            )
        wt_started = time.time()
        wt_embeddings, wt_batch_sizes = encode_records(torch, tokenizer, model, device, wt_encode_records)
        torch.cuda.synchronize(device)
        wt_elapsed = time.time() - wt_started
        if len(wt_embeddings) != PROTEINS:
            raise RuntimeError("WT representation count mismatch")

        representation_dir = OUT / "audit/representation_shards"
        scoring_dir = OUT / "audit/scoring_shards"
        ensure_dir(representation_dir)
        ensure_dir(scoring_dir)
        wt_inventory: List[Dict[str, Any]] = []
        candidate_inventory: List[Dict[str, Any]] = []
        shard_index: List[Dict[str, Any]] = []
        score_shard_paths: List[Path] = []
        global_peak = int(torch.cuda.max_memory_allocated(device))

        for protein_index, protein in enumerate(proteins):
            shard_started = time.time()
            torch.cuda.reset_peak_memory_stats(device)
            group = sorted(by_protein[protein], key=lambda row: int(row["manifest_index"]))
            candidate_records = []
            for row in group:
                token = candidate_tokens[row["candidate_id"]]
                candidate_records.append(
                    {
                        "record_id": row["candidate_id"],
                        "normalized_sequence": token["_normalized_sequence"],
                        "normalized_sequence_sha256": token["normalized_sequence_sha256"],
                        "token_ids_sha256": token["token_ids_sha256"],
                    }
                )
            candidate_embeddings, batch_sizes = encode_records(torch, tokenizer, model, device, candidate_records)
            torch.cuda.synchronize(device)
            wt_embedding = wt_embeddings[protein]
            ordered_candidate_ids = [row["candidate_id"] for row in group]
            offsets = [0]
            pieces = []
            for candidate_id in ordered_candidate_ids:
                piece = candidate_embeddings[candidate_id]
                pieces.append(piece)
                offsets.append(offsets[-1] + piece.shape[0])
            concatenated = np.concatenate(pieces, axis=0).astype(np.float16, copy=False)
            representation_path = representation_dir / f"{protein_index:04d}_{safe_id(protein)}.npz"
            write_npz_atomic(
                representation_path,
                wt_embedding=wt_embedding,
                candidate_embeddings=concatenated,
                candidate_offsets=np.asarray(offsets, dtype=np.int64),
                candidate_ids=np.asarray(ordered_candidate_ids),
                wt_sequence_sha256=np.asarray([group[0]["sequence_sha256"]]),
            )
            representation_hash = sha256_file(representation_path)
            if not representation_path.is_file() or representation_path.stat().st_size == 0:
                raise RuntimeError(f"representation shard write failed for {protein}")

            view_rows = []
            for index, row in enumerate(group):
                candidate_id = row["candidate_id"]
                edited_embedding = candidate_embeddings[candidate_id]
                view = compute_proxy_view(row, wt_embedding, edited_embedding)
                view_rows.append({**{column: row.get(column, "") for column in VIEW_IDENTITY_COLUMNS}, **view})
                token = candidate_tokens[candidate_id]
                candidate_inventory.append(
                    {
                        "manifest_index": row["manifest_index"],
                        "protein_id": protein,
                        "candidate_id": candidate_id,
                        "budget": row["budget"],
                        "method": row["method"],
                        "sequence_sha256": row["edited_sequence_sha256"],
                        "normalized_sequence_sha256": token["normalized_sequence_sha256"],
                        "normalization_changed": token["normalization_changed"],
                        "replacement_count": token["replacement_count"],
                        "replacements_json": token["replacements_json"],
                        "token_count_with_eos": token["token_count_with_eos"],
                        "residue_count": edited_embedding.shape[0],
                        "embedding_dimension": edited_embedding.shape[1],
                        "embedding_dtype": "float16",
                        "embedding_sha256": embedding_sha256(edited_embedding),
                        "representation_shard_path": str(representation_path),
                        "representation_shard_sha256": representation_hash,
                        "offset_start": offsets[index],
                        "offset_end_exclusive": offsets[index + 1],
                        "identity_status": "PASS",
                    }
                )
            score_path = scoring_dir / f"{protein_index:04d}_{safe_id(protein)}.csv"
            write_csv(score_path, VIEW_IDENTITY_COLUMNS + PROXY_VIEW_COLUMNS, view_rows)
            score_hash = sha256_file(score_path)
            shard_peak = int(torch.cuda.max_memory_allocated(device))
            global_peak = max(global_peak, shard_peak)
            shard_elapsed = time.time() - shard_started
            token = wt_tokens[protein]
            wt_inventory.append(
                {
                    "protein_id": protein,
                    "sequence_sha256": group[0]["sequence_sha256"],
                    "normalized_sequence_sha256": token["normalized_sequence_sha256"],
                    "normalization_changed": token["normalization_changed"],
                    "replacement_count": token["replacement_count"],
                    "replacements_json": token["replacements_json"],
                    "token_count_with_eos": token["token_count_with_eos"],
                    "residue_count": wt_embedding.shape[0],
                    "embedding_dimension": wt_embedding.shape[1],
                    "embedding_dtype": "float16",
                    "embedding_sha256": embedding_sha256(wt_embedding),
                    "representation_shard_path": str(representation_path),
                    "representation_shard_sha256": representation_hash,
                    "identity_status": "PASS",
                }
            )
            shard_index.append(
                {
                    "protein_index": protein_index,
                    "protein_id": protein,
                    "representation_shard_path": str(representation_path),
                    "representation_shard_sha256": representation_hash,
                    "representation_shard_size": representation_path.stat().st_size,
                    "scoring_shard_path": str(score_path),
                    "scoring_shard_sha256": score_hash,
                    "candidate_rows": len(group),
                    "candidate_forward_rows": len(candidate_records),
                    "candidate_batch_sizes_json": json.dumps(batch_sizes, separators=(",", ":")),
                    "peak_gpu_memory_bytes": shard_peak,
                    "elapsed_seconds": fmt(shard_elapsed, 6),
                    "fsync_complete": True,
                    "status": "PASS",
                }
            )
            score_shard_paths.append(score_path)
            del candidate_embeddings, concatenated, pieces
            if (protein_index + 1) % 10 == 0 or protein_index + 1 == len(proteins):
                print(json.dumps({"event": "DelePref_21_GPU_PROGRESS", "proteins_completed": protein_index + 1, "proteins_total": len(proteins), "elapsed_seconds": round(time.time() - started, 2)}, sort_keys=True), flush=True)

        write_csv(
            OUT / "prott5_wt_representation_inventory.csv",
            tuple(wt_inventory[0].keys()),
            wt_inventory,
        )
        candidate_inventory.sort(key=lambda row: int(row["manifest_index"]))
        write_csv(OUT / "prott5_candidate_representation_inventory.csv", tuple(candidate_inventory[0].keys()), candidate_inventory)
        write_csv(OUT / "prott5_representation_shard_index.csv", tuple(shard_index[0].keys()), shard_index)

        candidate_views: List[Dict[str, Any]] = []
        for item, score_path in zip(shard_index, score_shard_paths):
            if sha256_file(score_path) != item["scoring_shard_sha256"]:
                raise RuntimeError(f"scoring shard changed before aggregate: {score_path}")
            candidate_views.extend(read_csv(score_path))
        candidate_views.sort(key=lambda row: int(row["manifest_index"]))
        if len(candidate_views) != CANDIDATES or {row["candidate_id"] for row in candidate_views} != {row["candidate_id"] for row in rows}:
            raise RuntimeError("candidate view aggregate identity/count mismatch")
        by_exact_count: MutableMapping[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
        for row in candidate_views:
            by_exact_count[(row["protein_id"], int(row["deleted_length"]))].append(row)
        for group in by_exact_count.values():
            center, scale, method = robust_center_scale([finite(row["prott5_representation_proxy_utility"]) for row in group])
            for row in group:
                value = finite(row["prott5_representation_proxy_utility"])
                row["prott5_proxy_robust_center"] = fmt(center)
                row["prott5_proxy_robust_scale"] = fmt(scale)
                row["prott5_proxy_robust_scale_method"] = method
                row["prott5_proxy_robust_z_clipped"] = fmt(clipped_z(value, center, scale))
        final_view_columns = VIEW_IDENTITY_COLUMNS + PROXY_VIEW_COLUMNS + (
            "prott5_proxy_robust_center", "prott5_proxy_robust_scale", "prott5_proxy_robust_scale_method", "prott5_proxy_robust_z_clipped",
        )
        write_csv(OUT / "prott5_candidate_views.csv", final_view_columns, candidate_views)

        current_model_hashes = []
        for item in protocol["model"]["files"]:
            path = Path(item["path"])
            current_model_hashes.append({"relative_path": item["relative_path"], "sha256": sha256_file(path), "size": path.stat().st_size})
        snapshot_unchanged = all(
            current["sha256"] == expected["sha256"] and current["size"] == expected["size"]
            for current, expected in zip(current_model_hashes, protocol["model"]["files"])
        )
        if not snapshot_unchanged:
            raise RuntimeError("ProtT5 snapshot changed during formal forward")
        complete = {
            "schema_version": "delepref_21_formal_gpu_representation_complete_v1",
            "status": "PASS",
            "finished_at_utc": utc_now(),
            "elapsed_seconds": time.time() - started,
            "wt_representation_rows": len(wt_inventory),
            "candidate_representation_rows": len(candidate_inventory),
            "candidate_view_rows": len(candidate_views),
            "representation_shards": len(shard_index),
            "scoring_shards": len(score_shard_paths),
            "wt_batch_sizes_json": json.dumps(wt_batch_sizes, separators=(",", ":")),
            "wt_encoding_elapsed_seconds": wt_elapsed,
            "peak_gpu_memory_bytes": global_peak,
            "model_snapshot_immutable": snapshot_unchanged,
            "network_used": False,
            "cpu_fallback": False,
            "zero_fill": False,
            "candidate_rows_forwarded_without_deduplication": CANDIDATES,
            "no_second_gpu_retry": True,
        }
        atomic_json(OUT / "formal_gpu_representation_complete.json", complete)
        model_provenance.update(
            {
                "status": "PASS_FORMAL_GPU_REPRESENTATIONS_COMPLETE",
                "formal_model_path": str(snapshot),
                "formal_dtype": "torch.float16",
                "formal_model_eval": True,
                "formal_inference_mode": True,
                "training": False,
                "checkpoint_immutable": True,
                "snapshot_after_formal_forward": current_model_hashes,
                "snapshot_immutable_during_formal_forward": snapshot_unchanged,
                "formal_gpu_representation_complete_sha256": sha256_file(OUT / "formal_gpu_representation_complete.json"),
            }
        )
        atomic_json(OUT / "model_provenance.json", model_provenance)
        resource = json.loads((OUT / "gpu_resource_provenance.json").read_text(encoding="utf-8"))
        resource.update(
            {
                "status": "PASS_FORMAL_GPU_REPRESENTATIONS_COMPLETE",
                "phase": "formal_gpu_complete",
                "formal_peak_gpu_memory_bytes": global_peak,
                "formal_elapsed_seconds": complete["elapsed_seconds"],
                "representation_shards": len(shard_index),
                "candidate_representation_rows": len(candidate_inventory),
            }
        )
        atomic_json(OUT / "gpu_resource_provenance.json", resource)
        del model, tokenizer, wt_embeddings
        torch.cuda.empty_cache()
        return candidate_views
    except BaseException as exc:
        failure = {
            "schema_version": "delepref_21_formal_gpu_failure_v1",
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "created_at_utc": utc_now(),
            "second_gpu_retry_forbidden": True,
            "pass_sentinel_written": False,
        }
        atomic_json(OUT / "audit/failure_or_recovery_history/formal_gpu_failure.json", failure)
        raise


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> Tuple[float, int, str]:
    clean = [(float(left), float(right)) for left, right in zip(values_a, values_b) if math.isfinite(float(left)) and math.isfinite(float(right))]
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


def tie_statistics(values: Sequence[float]) -> Dict[str, int]:
    ordered = sorted(value for value in values if math.isfinite(value))
    groups: List[List[float]] = []
    for value in ordered:
        if not groups or abs(value - groups[-1][-1]) > TIE_ATOL:
            groups.append([value])
        else:
            groups[-1].append(value)
    tied = [group for group in groups if len(group) > 1]
    return {
        "finite_values": len(ordered),
        "unique_value_groups": len(groups),
        "tie_value_groups": len(tied),
        "tied_candidate_rows": sum(len(group) for group in tied),
        "tie_excess_rows": sum(len(group) - 1 for group in tied),
    }


def scope_definitions() -> List[Tuple[str, str, str]]:
    return [(f"{method}:{budget}", method, budget) for method in METHODS for budget in BUDGETS] + [("ALL", "ALL", "ALL")]


def in_scope(row: Mapping[str, Any], method: str, budget: str) -> bool:
    return (method == "ALL" or row["method"] == method or row.get("source_method") == method) and (budget == "ALL" or row["budget"] == budget)


def candidate_level_metrics(scored: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    comparisons = (("esmc", "teacher_utility"), ("delepref", "scorer_score"), ("esm2", "esm2_proxy_utility"))
    for scope, method, budget in scope_definitions():
        subset = [row for row in scored if in_scope(row, method, budget)]
        for reference, field in comparisons:
            reference_values = [finite(row.get(field)) for row in subset]
            proxy_values = [finite(row.get("prott5_representation_proxy_utility")) for row in subset]
            correlation, eligible, reason = spearman(reference_values, proxy_values)
            reference_ties = tie_statistics(reference_values)
            proxy_ties = tie_statistics(proxy_values)
            output.append(
                {
                    "scope": scope,
                    "method": method,
                    "budget": budget,
                    "reference": reference,
                    "reference_field": field,
                    "proxy_field": "prott5_representation_proxy_utility",
                    "N_total": len(subset),
                    "N_eligible": eligible,
                    "missing_reference": sum(not math.isfinite(value) for value in reference_values),
                    "missing_prott5": sum(not math.isfinite(value) for value in proxy_values),
                    "no_eligible": eligible == 0,
                    "spearman": fmt(correlation),
                    "spearman_na_reason": reason,
                    "reference_tie_value_groups": reference_ties["tie_value_groups"],
                    "reference_tied_candidate_rows": reference_ties["tied_candidate_rows"],
                    "prott5_tie_value_groups": proxy_ties["tie_value_groups"],
                    "prott5_tied_candidate_rows": proxy_ties["tied_candidate_rows"],
                    "status": "PASS" if not reason else "NA_RETAINED",
                }
            )
    return output


def top1_set(group: Sequence[Mapping[str, Any]], field: str) -> Tuple[List[Mapping[str, Any]], str]:
    finite_rows = [(row, finite(row.get(field))) for row in group if math.isfinite(finite(row.get(field)))]
    if not finite_rows:
        return [], "NO_ELIGIBLE_CANDIDATE"
    best = max(value for _row, value in finite_rows)
    selected = sorted((row for row, value in finite_rows if abs(value - best) <= TIE_ATOL), key=lambda row: row["candidate_id"])
    return selected, ""


def candidate_set(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted(row["candidate_id"] for row in rows)


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
    if reference == "TIE" and proxy != "TIE":
        return "DISAGREE_REFERENCE_TIE_ONLY"
    if reference != "TIE" and proxy == "TIE":
        return "DISAGREE_PROTT5_TIE_ONLY"
    return "DISAGREE_OPPOSITE_DIRECTION"


def cath_value(row: Mapping[str, Any], metric: str) -> float:
    field = "cath_" + metric
    return finite(row.get(field))


def tie_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [finite(row.get(field)) for row in rows]
    finite_values = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite_values)) if len(finite_values) == len(rows) and rows else math.nan


def build_group_outputs(scored: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: MutableMapping[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in scored:
        groups[(row["protein_id"], row["budget"], row["method"])].append(row)
    group_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    selected_rows: List[Dict[str, Any]] = []
    selectors = ("delepref", "esmc", "direct", "esm2", "prott5")
    for (protein, budget, method), group in sorted(groups.items()):
        if len(group) != K:
            raise RuntimeError(f"group is not K=8: {(protein, budget, method)}")
        selected: Dict[str, List[Mapping[str, Any]]] = {}
        selection_na: Dict[str, str] = {}
        for selector in selectors:
            selected[selector], selection_na[selector] = top1_set(group, REFERENCE_FIELDS[selector])
            if selection_na[selector]:
                raise RuntimeError(f"no eligible top-1 for {selector}:{protein}:{budget}:{method}")
        top_sets = {selector: candidate_set(values) for selector, values in selected.items()}
        esmc_corr, esmc_n, esmc_na = spearman(
            [finite(row["teacher_utility"]) for row in group],
            [finite(row["prott5_representation_proxy_utility"]) for row in group],
        )
        delepref_corr, delepref_n, delepref_na = spearman(
            [finite(row["scorer_score"]) for row in group],
            [finite(row["prott5_representation_proxy_utility"]) for row in group],
        )
        esm2_corr, esm2_n, esm2_na = spearman(
            [finite(row["esm2_proxy_utility"]) for row in group],
            [finite(row["prott5_representation_proxy_utility"]) for row in group],
        )
        pair_counters = {
            "esmc": Counter(),
            "delepref": Counter(),
        }
        ordered_group = sorted(group, key=lambda row: row["candidate_id"])
        for pair_index, (left, right) in enumerate(itertools.combinations(ordered_group, 2)):
            proxy_order = order_sign(finite(left["prott5_representation_proxy_utility"]), finite(right["prott5_representation_proxy_utility"]))
            esmc_order = order_sign(finite(left["teacher_utility"]), finite(right["teacher_utility"]))
            delepref_order = order_sign(finite(left["scorer_score"]), finite(right["scorer_score"]))
            esmc_status = order_status(esmc_order, proxy_order)
            delepref_status = order_status(delepref_order, proxy_order)
            pair_counters["esmc"][esmc_status] += 1
            pair_counters["delepref"][delepref_status] += 1
            pair_rows.append(
                {
                    "protein_id": protein,
                    "budget": budget,
                    "method": method,
                    "pair_index": pair_index,
                    "candidate_id_a": left["candidate_id"],
                    "candidate_id_b": right["candidate_id"],
                    "prott5_order": proxy_order,
                    "esmc_order": esmc_order,
                    "delepref_order": delepref_order,
                    "esmc_vs_prott5_status": esmc_status,
                    "delepref_vs_prott5_status": delepref_status,
                    "esmc_a": left["teacher_utility"],
                    "esmc_b": right["teacher_utility"],
                    "delepref_a": left["scorer_score"],
                    "delepref_b": right["scorer_score"],
                    "prott5_a": left["prott5_representation_proxy_utility"],
                    "prott5_b": right["prott5_representation_proxy_utility"],
                }
            )
        if sum(pair_counters["esmc"].values()) != 28 or sum(pair_counters["delepref"].values()) != 28:
            raise RuntimeError("pairwise diagnostics did not produce 28 pairs")
        esmc_eligible_pairs = 28 - pair_counters["esmc"]["NA"]
        delepref_eligible_pairs = 28 - pair_counters["delepref"]["NA"]
        esmc_agree = pair_counters["esmc"]["AGREE_DIRECTION"] + pair_counters["esmc"]["AGREE_BOTH_TIE"]
        delepref_agree = pair_counters["delepref"]["AGREE_DIRECTION"] + pair_counters["delepref"]["AGREE_BOTH_TIE"]
        output: Dict[str, Any] = {
            "protein_id": protein,
            "budget": budget,
            "method": method,
            "K": K,
            "group_esmc_prott5_spearman": fmt(esmc_corr),
            "group_esmc_prott5_N_eligible": esmc_n,
            "group_esmc_prott5_na_reason": esmc_na,
            "group_delepref_prott5_spearman": fmt(delepref_corr),
            "group_delepref_prott5_N_eligible": delepref_n,
            "group_delepref_prott5_na_reason": delepref_na,
            "group_esm2_prott5_spearman": fmt(esm2_corr),
            "group_esm2_prott5_N_eligible": esm2_n,
            "group_esm2_prott5_na_reason": esm2_na,
            "delepref_prott5_top1_set_exact_agreement": top_sets["delepref"] == top_sets["prott5"],
            "esmc_prott5_top1_set_exact_agreement": top_sets["esmc"] == top_sets["prott5"],
            "direct_prott5_top1_set_exact_agreement": top_sets["direct"] == top_sets["prott5"],
            "esm2_prott5_top1_set_exact_agreement": top_sets["esm2"] == top_sets["prott5"],
            "all_four_reference_top1_sets_equal_prott5": all(top_sets[selector] == top_sets["prott5"] for selector in ("delepref", "esmc", "direct", "esm2")),
            "esmc_prott5_pairwise_order_agreement": fmt(esmc_agree / esmc_eligible_pairs if esmc_eligible_pairs else math.nan),
            "delepref_prott5_pairwise_order_agreement": fmt(delepref_agree / delepref_eligible_pairs if delepref_eligible_pairs else math.nan),
            "esmc_prott5_pairwise_eligible": esmc_eligible_pairs,
            "delepref_prott5_pairwise_eligible": delepref_eligible_pairs,
        }
        for selector in selectors:
            output[f"{selector}_top1_candidate_ids_json"] = json.dumps(top_sets[selector], separators=(",", ":"))
            output[f"{selector}_top1_tie_count"] = len(top_sets[selector])
            output[f"{selector}_top1_no_eligible"] = False
            for metric in CATH_METRICS:
                output[f"{selector}_top1_{metric}"] = fmt(tie_mean(selected[selector], "cath_" + metric))
            output[f"{selector}_top1_teacher_utility"] = fmt(tie_mean(selected[selector], "teacher_utility"))
            output[f"{selector}_top1_delepref_score"] = fmt(tie_mean(selected[selector], "scorer_score"))
            output[f"{selector}_top1_direct_plm"] = fmt(tie_mean(selected[selector], "direct_delete_nll_mean"))
            output[f"{selector}_top1_esm2_proxy"] = fmt(tie_mean(selected[selector], "esm2_proxy_utility"))
            output[f"{selector}_top1_prott5_proxy"] = fmt(tie_mean(selected[selector], "prott5_representation_proxy_utility"))
            for selected_row in selected[selector]:
                selected_rows.append(
                    {
                        "selector": selector,
                        "protein_id": protein,
                        "budget": budget,
                        "method": method,
                        "candidate_id": selected_row["candidate_id"],
                        "top1_candidate_ids_json": output[f"{selector}_top1_candidate_ids_json"],
                        "top1_tie_count": len(selected[selector]),
                        "tie_member_weight": fmt(1.0 / len(selected[selector])),
                        "deleted_intervals": selected_row["deleted_intervals"],
                        "deletion_mask_sha256": selected_row["deletion_mask_sha256"],
                        "edited_sequence_sha256": selected_row["edited_sequence_sha256"],
                        "teacher_utility": selected_row["teacher_utility"],
                        "scorer_score": selected_row["scorer_score"],
                        "direct_delete_nll_mean": selected_row["direct_delete_nll_mean"],
                        "esm2_proxy_utility": selected_row["esm2_proxy_utility"],
                        "prott5_representation_proxy_utility": selected_row["prott5_representation_proxy_utility"],
                        "domain_core_deletion_content_ratio": selected_row["cath_domain_core_deletion_content_ratio"],
                        "core_damage": selected_row["cath_core_damage"],
                        "boundary_overlap": selected_row["cath_boundary_overlap"],
                        "span_count": selected_row["cath_span_count"],
                        "fragmentation": selected_row["cath_fragmentation"],
                        "cross_proxy_context_status": "NA_DIFFERENT_PROTOCOL",
                    }
                )
        for reference in ("esmc", "delepref"):
            for status in (
                "AGREE_DIRECTION", "AGREE_BOTH_TIE", "DISAGREE_REFERENCE_TIE_ONLY",
                "DISAGREE_PROTT5_TIE_ONLY", "DISAGREE_OPPOSITE_DIRECTION", "NA",
            ):
                output[f"{reference}_prott5_pairs_{status.lower()}"] = pair_counters[reference][status]
        group_rows.append(output)
    if len(group_rows) != GROUPS or len(pair_rows) != GROUPS * 28:
        raise RuntimeError("group or pairwise output count mismatch")
    return group_rows, pair_rows, selected_rows


ALIGNMENT_METRICS = (
    "group_esmc_prott5_spearman",
    "group_delepref_prott5_spearman",
    "group_esm2_prott5_spearman",
    "delepref_prott5_top1_set_exact_agreement",
    "esmc_prott5_top1_set_exact_agreement",
    "direct_prott5_top1_set_exact_agreement",
    "esm2_prott5_top1_set_exact_agreement",
    "all_four_reference_top1_sets_equal_prott5",
    "esmc_prott5_pairwise_order_agreement",
    "delepref_prott5_pairwise_order_agreement",
)


def metric_float(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if str(value).lower() in {"true", "false"}:
        return 1.0 if str(value).lower() == "true" else 0.0
    return finite(value)


def protein_values(rows: Sequence[Mapping[str, Any]], field: str) -> Tuple[List[str], np.ndarray]:
    grouped: MutableMapping[str, List[float]] = defaultdict(list)
    for row in rows:
        value = metric_float(row, field)
        if math.isfinite(value):
            grouped[row["protein_id"]].append(value)
    proteins = sorted(protein for protein, values in grouped.items() if values)
    values = np.asarray([float(np.mean(grouped[protein])) for protein in proteins], dtype=np.float64)
    return proteins, values


def bootstrap_mean(values: np.ndarray) -> Tuple[float, float, float]:
    if values.size == 0:
        return math.nan, math.nan, math.nan
    point = float(values.mean())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    replicates = values[indices].mean(axis=1)
    low, high = np.quantile(replicates, [0.025, 0.975], method="linear")
    return point, float(low), float(high)


def build_alignment_outputs(group_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    macro_rows: List[Dict[str, Any]] = []
    bootstrap_rows: List[Dict[str, Any]] = []
    for scope, method, budget in scope_definitions():
        subset = [row for row in group_rows if in_scope(row, method, budget)]
        macro: Dict[str, Any] = {
            "scope": scope,
            "method": method,
            "budget": budget,
            "groups": len(subset),
            "proteins": len({row["protein_id"] for row in subset}),
        }
        for metric in ALIGNMENT_METRICS:
            proteins, values = protein_values(subset, metric)
            point, low, high = bootstrap_mean(values)
            macro[metric] = fmt(point)
            macro[metric + "_eligible_proteins"] = len(proteins)
            macro[metric + "_na_groups"] = sum(not math.isfinite(metric_float(row, metric)) for row in subset)
            bootstrap_rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "budget": budget,
                    "metric": metric,
                    "point_estimate": fmt(point),
                    "ci95_low": fmt(low),
                    "ci95_high": fmt(high),
                    "eligible_proteins": len(proteins),
                    "missing_proteins": PROTEINS - len(proteins),
                    "bootstrap_unit": "protein_id",
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "ci_method": "percentile_linear",
                    "failure_or_na": "" if math.isfinite(point) else "NA_NO_ELIGIBLE_PROTEINS",
                }
            )
        macro_rows.append(macro)
    return macro_rows, bootstrap_rows


def build_contextual_bootstrap(group_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for scope, method, budget in scope_definitions():
        subset = [row for row in group_rows if in_scope(row, method, budget)]
        for reference in ("delepref", "direct", "esm2"):
            for metric in CATH_METRICS:
                delta_field = f"_context_delta_{reference}_{metric}"
                delta_rows = []
                for row in subset:
                    copy = dict(row)
                    left = finite(row[f"prott5_top1_{metric}"])
                    right = finite(row[f"{reference}_top1_{metric}"])
                    copy[delta_field] = left - right if math.isfinite(left) and math.isfinite(right) else math.nan
                    delta_rows.append(copy)
                proteins, values = protein_values(delta_rows, delta_field)
                point, low, high = bootstrap_mean(values)
                output.append(
                    {
                        "comparison_status": "NA_DIFFERENT_PROTOCOL",
                        "scope": scope,
                        "method": method,
                        "budget": budget,
                        "comparison": f"prott5_top1_minus_{reference}_top1",
                        "left_selector": "prott5",
                        "right_selector": reference,
                        "metric": metric,
                        "point_estimate_left_minus_right": fmt(point),
                        "ci95_low": fmt(low),
                        "ci95_high": fmt(high),
                        "eligible_proteins": len(proteins),
                        "bootstrap_unit": "protein_id",
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "ci_method": "percentile_linear",
                        "interpretation": "descriptive contextual comparison only; no significant-better claim is permitted",
                        "failure_or_na": "" if math.isfinite(point) else "NA_NO_ELIGIBLE_PROTEINS",
                    }
                )
    return output


def build_missingness_report(
    candidate_metrics: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in candidate_metrics:
        output.append(
            {
                "record_type": "candidate_level_spearman",
                "scope": row["scope"],
                "method": row["method"],
                "budget": row["budget"],
                "metric_or_selector": row["reference"] + "_vs_prott5",
                "N_total": row["N_total"],
                "N_eligible": row["N_eligible"],
                "missing_count": int(row["missing_reference"]) + int(row["missing_prott5"]),
                "tie_groups": int(row["reference_tie_value_groups"]) + int(row["prott5_tie_value_groups"]),
                "tied_rows": int(row["reference_tied_candidate_rows"]) + int(row["prott5_tied_candidate_rows"]),
                "no_eligible_count": 1 if truthy(row["no_eligible"]) else 0,
                "na_reason": row["spearman_na_reason"],
            }
        )
    for scope, method, budget in scope_definitions():
        groups = [row for row in group_rows if in_scope(row, method, budget)]
        pairs = [row for row in pair_rows if in_scope(row, method, budget)]
        for selector in ("delepref", "esmc", "direct", "esm2", "prott5"):
            output.append(
                {
                    "record_type": "group_top1_ties",
                    "scope": scope,
                    "method": method,
                    "budget": budget,
                    "metric_or_selector": selector,
                    "N_total": len(groups),
                    "N_eligible": sum(not truthy(row[f"{selector}_top1_no_eligible"]) for row in groups),
                    "missing_count": 0,
                    "tie_groups": sum(int(row[f"{selector}_top1_tie_count"]) > 1 for row in groups),
                    "tied_rows": sum(int(row[f"{selector}_top1_tie_count"]) for row in groups if int(row[f"{selector}_top1_tie_count"]) > 1),
                    "no_eligible_count": sum(truthy(row[f"{selector}_top1_no_eligible"]) for row in groups),
                    "na_reason": "",
                }
            )
        for reference, status_field in (("esmc", "esmc_vs_prott5_status"), ("delepref", "delepref_vs_prott5_status")):
            statuses = Counter(row[status_field] for row in pairs)
            output.append(
                {
                    "record_type": "pairwise_order",
                    "scope": scope,
                    "method": method,
                    "budget": budget,
                    "metric_or_selector": reference + "_vs_prott5",
                    "N_total": len(pairs),
                    "N_eligible": len(pairs) - statuses["NA"],
                    "missing_count": statuses["NA"],
                    "tie_groups": statuses["AGREE_BOTH_TIE"] + statuses["DISAGREE_REFERENCE_TIE_ONLY"] + statuses["DISAGREE_PROTT5_TIE_ONLY"],
                    "tied_rows": statuses["AGREE_BOTH_TIE"] + statuses["DISAGREE_REFERENCE_TIE_ONLY"] + statuses["DISAGREE_PROTT5_TIE_ONLY"],
                    "no_eligible_count": 1 if len(pairs) == statuses["NA"] else 0,
                    "na_reason": json.dumps(dict(statuses), separators=(",", ":"), sort_keys=True),
                }
            )
    return output


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


def write_summary(
    candidate_metrics: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
    status: str,
) -> None:
    lines = [
        "# DelePref-21 ProtT5 Cross-Family Representation-Proxy Audit",
        "",
        f"Status: `{status}`",
        "",
        "This is a frozen cross-family representation-proxy alignment audit over the exact DelePref-20 500-protein, 3,000-group, 24,000-candidate CATH cohort. It is not training, functional validation, binding validation, stability validation, teacher replacement, or a new unseen evaluation.",
        "",
        f"ProtT5 checkpoint: `{REPO_ID}` at full revision `{REVISION}`; snapshot aggregate SHA256 `{EXPECTED_SNAPSHOT_SHA256}`.",
        "",
        "## Candidate-level Spearman",
        "",
        "| scope | reference | N | Spearman | NA reason |",
        "|---|---|---:|---:|---|",
    ]
    for row in candidate_metrics:
        lines.append(f"| {row['scope']} | {row['reference']} | {row['N_eligible']} | {row['spearman']} | {row['spearman_na_reason']} |")
    lines.extend(
        [
            "",
            "## Protein-macro group alignment",
            "",
            "| scope | ESMC rho | DelePref rho | ESM2 rho | DelePref top-1 | ESMC top-1 | Direct top-1 | ESM2 top-1 | all four refs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_scope = {row["scope"]: row for row in macro_rows}
    for scope, _method, _budget in scope_definitions():
        row = by_scope[scope]
        lines.append(
            "| {scope} | {e} | {v} | {x} | {vt} | {et} | {dt} | {xt} | {all4} |".format(
                scope=scope,
                e=row["group_esmc_prott5_spearman"],
                v=row["group_delepref_prott5_spearman"],
                x=row["group_esm2_prott5_spearman"],
                vt=row["delepref_prott5_top1_set_exact_agreement"],
                et=row["esmc_prott5_top1_set_exact_agreement"],
                dt=row["direct_prott5_top1_set_exact_agreement"],
                xt=row["esm2_prott5_top1_set_exact_agreement"],
                all4=row["all_four_reference_top1_sets_equal_prott5"],
            )
        )
    lines.extend(
        [
            "",
            "All ties use complete top-1 candidate-ID sets; no tie is broken arbitrarily. Negative, low-agreement, mixed, tied, and NA outcomes are retained.",
            "",
            "CATH core-content, core-damage, boundary, span, and fragmentation comparisons across proxy-selected plans are `NA_DIFFERENT_PROTOCOL` descriptive context only; they do not establish that one selector is significantly better.",
            "",
            "ProtT5 is a frozen sequence-representation proxy from outside the ESM family. It is not truth about function, stability, binding, or retained function. These results cannot prove that a deleted region lacks function, functional preservation, or retained binding.",
            "",
            f"Safe conclusion: {CLAIM_BOUNDARY}",
            "",
        ]
    )
    atomic_text(OUT / "SUMMARY.md", "\n".join(lines))


def write_master_payload() -> None:
    start = "DelePref_21_PROTT5_CROSSFAMILY_AUDIT_START"
    end = "DelePref_21_PROTT5_CROSSFAMILY_AUDIT_END"
    text = "\n".join(
        [
            start,
            "",
            "## DelePref-21 ProtT5 Cross-Family Representation-Proxy Alignment Audit",
            "",
            "The exact frozen DelePref-20 500-protein / 3,000 K=8 group / 24,000-candidate CATH cohort was evaluated with the frozen non-ESM ProtT5 encoder representation proxy. No candidates, ESMC teacher utilities, DelePref scores, Direct-PLM diagnostics, ESM2 diagnostics, or checkpoints were regenerated or rescored.",
            "",
            "ProtT5 is a frozen sequence-representation proxy, not functional, stability, binding, retained-function, or biological ground truth. Cross-proxy CATH selections are `NA_DIFFERENT_PROTOCOL` descriptive context only. Low agreement, negative results, mixed results, ties, and NA outcomes are retained.",
            "",
            CLAIM_BOUNDARY,
            "",
            "The audit cannot prove that a deleted region lacks function, functional preservation, or retained binding.",
            "",
            end,
            "",
        ]
    )
    atomic_text(OUT / "result_summary.md", text)


def scientific_manifest(science_files: Sequence[Path]) -> Dict[str, Any]:
    artifacts = []
    seen = set()
    for path in science_files:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            raise RuntimeError(f"scientific artifact missing before seal: {path}")
        artifacts.append({"path": str(path.relative_to(OUT)), "size": path.stat().st_size, "sha256": sha256_file(path)})
    for root in (OUT / "audit/representation_shards", OUT / "audit/scoring_shards"):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path not in seen:
                artifacts.append({"path": str(path.relative_to(OUT)), "size": path.stat().st_size, "sha256": sha256_file(path)})
    result = {
        "schema_version": "delepref_21_scientific_manifest_v1",
        "status": "SEALED_BEFORE_INDEPENDENT_VALIDATION",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "raw_representation_shards": sum(item["path"].startswith("audit/representation_shards/") for item in artifacts),
        "raw_scoring_shards": sum(item["path"].startswith("audit/scoring_shards/") for item in artifacts),
        "created_at_utc": utc_now(),
    }
    atomic_json(OUT / "scientific_manifest.json", result)
    return result


def package_scientific_outputs(base_rows: Sequence[Mapping[str, Any]], candidate_views: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    by_id = {row["candidate_id"]: row for row in candidate_views}
    if len(by_id) != CANDIDATES:
        raise RuntimeError("candidate view inventory is not unique")
    scored: List[Dict[str, Any]] = []
    for base in base_rows:
        candidate_id = base["candidate_id"]
        view = by_id[candidate_id]
        identity = ("protein_id", "budget", "method", "sequence_sha256", "edited_sequence_sha256", "deletion_mask_sha256", "deleted_intervals")
        if any(str(base[field]) != str(view[field]) for field in identity):
            raise RuntimeError(f"candidate view/base identity mismatch for {candidate_id}")
        merged = dict(base)
        merged.update(view)
        scored.append(merged)
    candidate_metrics = candidate_level_metrics(scored)
    group_rows, pair_rows, selected_rows = build_group_outputs(scored)
    macro_rows, alignment_ci = build_alignment_outputs(group_rows)
    context_ci = build_contextual_bootstrap(group_rows)
    missingness = build_missingness_report(candidate_metrics, group_rows, pair_rows)

    write_csv(OUT / "prott5_candidate_level_metrics.csv", tuple(candidate_metrics[0].keys()), candidate_metrics)
    write_csv(OUT / "prott5_group_metrics.csv", tuple(group_rows[0].keys()), group_rows)
    write_csv(OUT / "prott5_pairwise_order_diagnostics.csv", tuple(pair_rows[0].keys()), pair_rows)
    write_csv(OUT / "prott5_top1_selected_plans.csv", tuple(selected_rows[0].keys()), selected_rows)
    write_csv(OUT / "prott5_group_macro_summary.csv", tuple(macro_rows[0].keys()), macro_rows)
    write_csv(OUT / "prott5_alignment_bootstrap_ci.csv", tuple(alignment_ci[0].keys()), alignment_ci)
    write_csv(OUT / "prott5_contextual_cath_bootstrap_ci.csv", tuple(context_ci[0].keys()), context_ci)
    write_csv(OUT / "missingness_and_tie_report.csv", tuple(missingness[0].keys()), missingness)
    write_summary(candidate_metrics, macro_rows, "AWAITING_INDEPENDENT_VALIDATION")
    write_master_payload()

    science_files = [
        OUT / "protocol.json", OUT / "protocol.sha256", OUT / "input_hash_validation.csv", OUT / "input_tokenization_preflight.csv",
        OUT / "model_provenance.json", OUT / "runtime_vendor_provenance.json", OUT / "gpu_resource_provenance.json",
        OUT / "formal_gpu_representation_complete.json", OUT / "prott5_wt_representation_inventory.csv",
        OUT / "prott5_candidate_representation_inventory.csv", OUT / "prott5_representation_shard_index.csv",
        OUT / "prott5_candidate_views.csv", OUT / "prott5_candidate_level_metrics.csv", OUT / "prott5_group_metrics.csv",
        OUT / "prott5_pairwise_order_diagnostics.csv", OUT / "prott5_top1_selected_plans.csv", OUT / "prott5_group_macro_summary.csv",
        OUT / "prott5_alignment_bootstrap_ci.csv", OUT / "prott5_contextual_cath_bootstrap_ci.csv", OUT / "missingness_and_tie_report.csv",
    ]
    manifest = scientific_manifest(science_files)
    negative_mixed = {
        "negative_candidate_spearman_rows": sum(math.isfinite(finite(row["spearman"])) and finite(row["spearman"]) < 0 for row in candidate_metrics),
        "candidate_spearman_na_rows": sum(not math.isfinite(finite(row["spearman"])) for row in candidate_metrics),
        "top1_alignment_below_half_rows": sum(
            math.isfinite(metric_float(row, metric)) and metric_float(row, metric) < 0.5
            for row in macro_rows
            for metric in (
                "delepref_prott5_top1_set_exact_agreement", "esmc_prott5_top1_set_exact_agreement",
                "direct_prott5_top1_set_exact_agreement", "esm2_prott5_top1_set_exact_agreement",
            )
        ),
        "alignment_ci_rows_containing_zero": sum(
            math.isfinite(finite(row["ci95_low"])) and finite(row["ci95_low"]) <= 0 <= finite(row["ci95_high"])
            for row in alignment_ci
        ),
        "retained_without_filtering": True,
    }
    summary = {
        "schema_version": "delepref_21_run_summary_v1",
        "status": "AWAITING_INDEPENDENT_VALIDATION",
        "claim_boundary": CLAIM_BOUNDARY,
        "scientific_label": "frozen cross-family representation-proxy alignment audit",
        "cohort": protocol["cohort"],
        "model": {"repo_id": REPO_ID, "revision": REVISION, "snapshot_sha256": EXPECTED_SNAPSHOT_SHA256},
        "counts": {
            "wt_representations": PROTEINS,
            "candidate_representations": CANDIDATES,
            "candidate_views": len(candidate_views),
            "groups": len(group_rows),
            "pairwise_candidate_pairs": len(pair_rows),
            "top1_selection_rows_including_ties": len(selected_rows),
            "candidate_metric_rows": len(candidate_metrics),
            "alignment_bootstrap_rows": len(alignment_ci),
            "contextual_cath_bootstrap_rows": len(context_ci),
        },
        "candidate_level_metrics": candidate_metrics,
        "group_macro_summary": macro_rows,
        "negative_and_mixed_results": negative_mixed,
        "bootstrap": {"unit": "protein_id", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "scientific_manifest_sha256": sha256_file(OUT / "scientific_manifest.json"),
        "protocol_sha256": sha256_file(OUT / "protocol.json"),
        "required_outputs": required_outputs(),
        "declarations": {
            "training": False,
            "checkpoint_loaded": False,
            "teacher_rescoring": False,
            "student_rescoring": False,
            "candidate_regeneration": False,
            "delepref_14_opened": False,
            "protocol_source_modified": False,
            "contextual_cath_status": "NA_DIFFERENT_PROTOCOL",
        },
    }
    atomic_json(OUT / "run_summary.json", summary)
    atomic_json(
        OUT / "formal_scientific_complete.json",
        {
            "schema_version": "delepref_21_formal_scientific_complete_v1",
            "status": "PASS_SCIENTIFIC_OUTPUTS_SEALED_AWAITING_VALIDATION",
            "created_at_utc": utc_now(),
            "scientific_manifest_sha256": sha256_file(OUT / "scientific_manifest.json"),
            "run_summary_sha256": sha256_file(OUT / "run_summary.json"),
            "cpu_packaging_retry_eligible_only_if_validator_or_nonscientific_packaging_fails": True,
            "scientific_metric_recomputation_in_retry_forbidden": True,
        },
    )
    return summary


def run() -> None:
    ensure_dir(OUT / "audit/failure_or_recovery_history")
    ensure_dir(OUT / "audit/routing")
    if json.loads((OUT / "bootstrap_status.json").read_text(encoding="utf-8")).get("status") != "PASS_BOOTSTRAP_COMPLETE":
        raise RuntimeError("bootstrap must pass before cohort access")
    if (OUT / "protocol.json").exists() or (OUT / "formal_scientific_started.json").exists():
        raise RuntimeError("DelePref-21 protocol or formal marker already exists; formal retry forbidden")
    v20_checks = validate_v20_package_status()
    hash_rows = build_input_hash_validation()
    rows, diagnostics = validate_frozen_rows()
    wt_tokens, candidate_tokens, token_summary = tokenization_preflight(rows)
    resource = resource_preflight()
    protocol = protocol_lock(rows, diagnostics, hash_rows, token_summary, resource)
    atomic_json(
        OUT / "pre_scientific_gate.json",
        {
            "status": "PASS",
            "delepref_20_status_checks": v20_checks,
            "cohort": diagnostics,
            "tokenization": token_summary,
            "resource_checks": resource["checks"],
            "protocol_sha256": sha256_file(OUT / "protocol.json"),
            "created_at_utc": utc_now(),
        },
    )
    candidate_views = formal_representation_pass(rows, wt_tokens, candidate_tokens, protocol)
    summary = package_scientific_outputs(rows, candidate_views, protocol)
    print(
        json.dumps(
            {
                "event": "DelePref_21_SCIENTIFIC_OUTPUTS_SEALED",
                "status": summary["status"],
                "candidate_views": len(candidate_views),
                "groups": GROUPS,
                "protocol_sha256": sha256_file(OUT / "protocol.json"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="single formal scientific run; no resume or GPU retry")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
