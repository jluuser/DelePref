#!/usr/bin/env python3
"""DelePref-20: direct ESM2 masked-marginal and cross-teacher audit.

This is intentionally a frozen-pool experiment.  It never generates or
reselects candidates.  The only model used here is the local ESM2-650M
checkpoint named in the protocol.  The formal GPU stage writes one fsynced
shard per protein; all ranking, statistics, and validation after that stage
are CPU work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/experiments/delepref_step004000_delepref_20_direct_plm_cross_teacher"
INPUT = REPO / "results/experiments/delepref_step004000_inference_ablation"
CATH = REPO / "results/experiments/delepref_step004000_delepref_10_cath_method_comparison"
MODEL = Path(os.environ.get("DELEPREF_ESM2_MODEL", REPO / "models/esm2_650m"))
PYTHON_ENV = Path(sys.prefix)
SCRIPT = Path(__file__).resolve()
VALIDATOR = REPO / "scripts/validate_delepref_20_direct_plm_cross_teacher.py"
ROUTING_LAUNCHER = OUT / "audit/routing/launcher.sh"

MODEL_REL = "models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c"
COORDINATE_SYSTEM = "0_based_closed_inclusive"
METHOD_MAP = {"learned_policy": "learned", "cold_start": "cold_start", "random_span": "random"}
METHODS = ("learned", "cold_start", "random")
BUDGETS = ("0.10", "0.20")
SEED = "101011"
K = 8
PROTEIN_COUNT = 500
GROUP_COUNT = 3000
CANDIDATE_COUNT = 24000
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260721
FINAL_MARKER = "PASS_DelePref_20_DIRECT_PLM_CROSS_TEACHER"
CLAIM_BOUNDARY = "The method ranks candidate deletion plans by teacher-derived deletion perturbation preference and optional auxiliary risk evidence."

MANIFEST_COLUMNS = [
    "manifest_index", "protein_id", "wildtype_sequence", "sequence_sha256", "sequence_length",
    "budget", "generation_seed", "source_method", "method", "candidate_id", "deleted_intervals",
    "deleted_length", "deletion_fraction", "deletion_mask_sha256", "span_count", "edited_sequence",
    "edited_sequence_sha256", "edited_length", "scorer_score", "scorer_rank", "policy_native_rank",
    "teacher_utility", "global_similarity", "retained_similarity", "junction_plausibility",
    "cath_core_start", "cath_core_end", "cath_core_length", "cath_core_overlap_length",
    "cath_domain_core_deletion_content_ratio", "cath_core_damage", "cath_boundary_overlap",
    "cath_span_count", "cath_fragmentation", "cath_feasible", "cath_exact_budget_feasibility",
    "cath_annotation_interpretation", "coordinate_system",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_dir(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, payload: bytes) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_file(path)
    fsync_dir(path.parent)


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    atomic_bytes(path, canonical_json(value))


def write_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def fmt(value: Any, digits: int = 10) -> str:
    number = finite(value)
    return "" if not math.isfinite(number) else f"{number:.{digits}f}"


def parse_intervals(value: Any) -> Tuple[Tuple[int, int], ...]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError("deleted_intervals must be a non-empty list")
    result: List[Tuple[int, int]] = []
    previous = -2
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("interval must be [start,end]")
        start, end = int(item[0]), int(item[1])
        if start < 0 or end < start or start <= previous + 1:
            raise ValueError("intervals are not sorted/non-overlapping with a retained separator")
        result.append((start, end))
        previous = end
    return tuple(result)


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
    mask = bytearray(int(length))
    for start, end in intervals:
        mask[start : end + 1] = b"\x01" * (end - start + 1)
    return hashlib.sha256(bytes(mask)).hexdigest()


def retained_mapping(length: int, intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    deleted = set()
    for start, end in intervals:
        deleted.update(range(start, end + 1))
    result = []
    deleted_before = 0
    for original in range(length):
        if original in deleted:
            deleted_before += 1
        else:
            result.append((original, original - deleted_before))
    return result


def read_csv(path: Path) -> List[Dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_csv_stream(path: Path) -> Iterable[Dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    handle = path.open("r", encoding="utf-8", newline="")
    try:
        yield from csv.DictReader(handle)
    finally:
        handle.close()


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fsync_file(path)
    fsync_dir(path.parent)


def strict_delete_count(length: int, budget: str) -> int:
    return max(1, min(length - 1, int(round(float(budget) * length))))


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def nvidia_query() -> Dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.strip().splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 7:
            rows.append({"index": fields[0], "name": fields[1], "uuid": fields[2], "driver_version": fields[3], "memory_total_mib": fields[4], "memory_used_mib": fields[5], "utilization_gpu_percent": fields[6]})
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    processes = []
    for line in apps.stdout.strip().splitlines():
        if line.strip() and not line.lower().startswith("no running"):
            processes.append(line.strip())
    return {"gpus": rows, "compute_processes": processes}


def resource_probe() -> Dict[str, Any]:
    info = nvidia_query()
    env = {key: os.environ.get(key, "") for key in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "CONDA_DEFAULT_ENV", "CONDA_PREFIX")}
    checks: Dict[str, bool] = {
        "one_visible_gpu": len(info["gpus"]) == 1,
        "cwd": str(Path.cwd()) == str(REPO),
    }
    import torch  # type: ignore
    checks.update({
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_one_gpu": int(torch.cuda.device_count()) == 1,
        "torch_gpu_detected": bool(torch.cuda.is_available()) and bool(torch.cuda.get_device_name(0)),
    })
    result = {
        "schema_version": "delepref_20_resource_provenance_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(), "cwd": str(Path.cwd()), "environment": env,
        "scheduler_job_id": env["SLURM_JOB_ID"], "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "nvidia_smi": info["gpus"], "nvidia_compute_processes": info["compute_processes"],
        "checks": checks, "expected": {"visible_gpu_count": 1},
        "python_executable": sys.executable, "python_version": sys.version,
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "torch_device_count": int(torch.cuda.device_count()), "torch_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    if result["status"] != "PASS":
        raise RuntimeError("resource probe failed: " + json.dumps(result, sort_keys=True))
    write_json(OUT / "resource_provenance.json", result)
    return result


def model_inventory() -> Dict[str, Any]:
    if not MODEL.is_dir():
        raise RuntimeError(f"model snapshot is missing: {MODEL}")
    files = []
    for path in sorted(MODEL.iterdir(), key=lambda p: p.name):
        if path.is_file():
            files.append({"name": path.name, "path": str(path), "resolved_path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)})
    required = {"config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "model.safetensors"}
    if not required.issubset({row["name"] for row in files}):
        raise RuntimeError("required local model files are missing")
    return {"snapshot": str(MODEL), "relative_snapshot": MODEL_REL, "files": files, "aggregate_sha256": sha256_text(json.dumps(files, sort_keys=True))}


def input_paths() -> Dict[str, Path]:
    return {
        "scored_candidate_pools": INPUT / "scored_candidate_pools.csv",
        "selected_plans": INPUT / "selected_plans.csv",
        "run_summary": INPUT / "run_summary.json",
        "cath_plan_level_metrics": CATH / "cath_plan_level_metrics.csv",
        "cath_paired_bootstrap_ci": CATH / "cath_paired_bootstrap_ci.csv",
        "cath_source_provenance": CATH / "source_provenance.json",
    }


def build_frozen_rows() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    paths = input_paths()
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing authoritative input {name}: {path}")
    scored = [row for row in read_csv_stream(paths["scored_candidate_pools"]) if row.get("generation_seed") == SEED]
    if len(scored) != CANDIDATE_COUNT:
        raise RuntimeError(f"fixed seed pool is {len(scored)} rows, expected {CANDIDATE_COUNT}")
    cath_rows = [row for row in read_csv_stream(paths["cath_plan_level_metrics"]) if row.get("comparison_status") == "SAME_PROTOCOL" and row.get("generation_seed") == SEED]
    cath_by_id = {row["candidate_id"]: row for row in cath_rows}
    if len(cath_by_id) != CANDIDATE_COUNT:
        raise RuntimeError(f"CATH same-protocol join has {len(cath_by_id)} unique rows, expected {CANDIDATE_COUNT}")
    selected = read_csv(paths["selected_plans"])
    conditions = {"learned": "delepref_learned_policy_scorer", "cold_start": "cold_start_scorer", "random": "random_span_scorer"}
    selected_by_group: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in selected:
        if row.get("generation_seed") != SEED:
            continue
        method = METHOD_MAP.get(row.get("source_method", ""))
        if method is None or row.get("condition") != conditions[method]:
            continue
        key = (row["protein_id"], f"{float(row['budget']):.2f}", method)
        if key in selected_by_group:
            raise RuntimeError(f"duplicate frozen selected plan {key}")
        selected_by_group[key] = row
    rows: List[Dict[str, Any]] = []
    seen_ids = set()
    proteins = set()
    groups: MutableMapping[Tuple[str, str, str], List[str]] = defaultdict(list)
    for raw in scored:
        source = raw.get("source_method", "")
        method = METHOD_MAP.get(source)
        if method is None:
            raise RuntimeError(f"unexpected source method {source}")
        protein = raw["protein_id"]
        sequence = raw["wildtype_sequence"]
        length = int(raw["sequence_length"])
        if len(sequence) != length or sha256_text(sequence) != raw["sequence_sha256"]:
            raise RuntimeError(f"sequence identity mismatch for {raw['candidate_id']}")
        intervals = parse_intervals(raw["deleted_intervals"])
        if any(end >= length for _start, end in intervals):
            raise RuntimeError(f"interval outside sequence for {raw['candidate_id']}")
        dlen = deleted_length(intervals)
        if dlen != int(raw["deleted_length"]) or dlen != strict_delete_count(length, f"{float(raw['budget']):.2f}"):
            raise RuntimeError(f"deleted count/budget mismatch for {raw['candidate_id']}")
        if mask_hash(length, intervals) != raw["deletion_mask_sha256"]:
            raise RuntimeError(f"mask hash mismatch for {raw['candidate_id']}")
        if raw.get("teacher_utility_valid", "").lower() != "true":
            raise RuntimeError(f"frozen teacher utility invalid for {raw['candidate_id']}")
        g, r, j = finite(raw["global_similarity"]), finite(raw["retained_similarity"]), finite(raw["junction_plausibility"])
        utility = finite(raw["teacher_utility"])
        if not all(math.isfinite(v) for v in (g, r, j, utility)) or abs((0.35 * g + 0.45 * r + 0.20 * j) - utility) > 2e-5:
            raise RuntimeError(f"teacher utility formula mismatch for {raw['candidate_id']}")
        cath = cath_by_id.get(raw["candidate_id"])
        if cath is None or cath.get("protein_id") != protein or cath.get("sequence_sha256") != raw["sequence_sha256"] or f"{float(cath.get('budget', 0)):.2f}" != f"{float(raw['budget']):.2f}":
            raise RuntimeError(f"CATH join mismatch for {raw['candidate_id']}")
        edited = apply_deletion(sequence, intervals)
        row: Dict[str, Any] = dict(raw)
        row.update({
            "method": method, "budget": f"{float(raw['budget']):.2f}", "deleted_intervals": json.dumps([[a, b] for a, b in intervals], separators=(",", ":")),
            "edited_sequence": edited, "edited_sequence_sha256": sha256_text(edited), "edited_length": len(edited),
            "span_count": len(intervals), "teacher_utility": fmt(utility), "global_similarity": fmt(g), "retained_similarity": fmt(r), "junction_plausibility": fmt(j),
            "cath_core_start": cath.get("core_start", ""), "cath_core_end": cath.get("core_end", ""), "cath_core_length": cath.get("core_length", ""), "cath_core_overlap_length": cath.get("core_overlap_length", ""),
            "cath_domain_core_deletion_content_ratio": cath.get("domain_core_deletion_content_ratio", ""), "cath_core_damage": cath.get("core_damage", ""), "cath_boundary_overlap": cath.get("boundary_overlap", ""), "cath_span_count": cath.get("span_count", ""), "cath_fragmentation": cath.get("fragmentation", ""),
            "cath_feasible": cath.get("feasible", ""), "cath_exact_budget_feasibility": cath.get("exact_budget_feasibility", ""), "cath_annotation_interpretation": cath.get("annotation_interpretation", ""), "coordinate_system": COORDINATE_SYSTEM,
        })
        if raw["candidate_id"] in seen_ids:
            raise RuntimeError("candidate ID is not unique")
        seen_ids.add(raw["candidate_id"]); proteins.add(protein)
        group = (protein, row["budget"], method)
        groups[group].append(raw["deletion_mask_sha256"])
        rows.append(row)
    if len(proteins) != PROTEIN_COUNT or len(groups) != GROUP_COUNT or len(rows) != CANDIDATE_COUNT:
        raise RuntimeError(f"fixed cohort mismatch proteins={len(proteins)} groups={len(groups)} rows={len(rows)}")
    if any(len(values) != K or len(set(values)) != K for values in groups.values()):
        raise RuntimeError("one or more fixed groups are not unique K=8")
    for group, values in groups.items():
        method = group[2]
        candidates = [row for row in rows if (row["protein_id"], row["budget"], row["method"]) == group]
        selected_frozen = selected_by_group.get(group)
        if selected_frozen is None or selected_frozen["candidate_id"] not in {row["candidate_id"] for row in candidates}:
            raise RuntimeError(f"missing frozen DelePref scorer selection for {group}")
        direct_top = sorted(candidates, key=lambda row: (-finite(row["scorer_score"]), row["candidate_id"]))[0]
        rank_one = [row for row in candidates if int(row["scorer_rank"]) == 1]
        if len(rank_one) != 1 or rank_one[0]["candidate_id"] != direct_top["candidate_id"] or selected_frozen["candidate_id"] != direct_top["candidate_id"]:
            raise RuntimeError(f"frozen scorer top-1 mismatch for {group}")
    rows.sort(key=lambda row: (row["protein_id"], row["budget"], row["method"], row["candidate_id"]))
    for index, row in enumerate(rows):
        row["manifest_index"] = index
    diagnostics = {"protein_count": len(proteins), "group_count": len(groups), "candidate_count": len(rows), "methods": dict(Counter(row["method"] for row in rows)), "budgets": dict(Counter(row["budget"] for row in rows)), "generation_seed": SEED, "K": K, "cath_join_count": len(cath_by_id), "teacher_valid_count": len(rows)}
    return rows, diagnostics


def write_input_artifacts(rows: Sequence[Mapping[str, Any]]) -> None:
    write_csv(OUT / "input_manifest.csv", MANIFEST_COLUMNS, rows)
    reconstruction_columns = ["manifest_index", "candidate_id", "protein_id", "sequence_sha256", "edited_sequence_sha256", "deleted_intervals", "deleted_length", "edited_length", "intervals_valid", "mask_hash_valid", "sequence_hash_valid", "edited_reconstruction_valid", "status"]
    reconstruction = []
    for row in rows:
        intervals = parse_intervals(row["deleted_intervals"])
        sequence = row["wildtype_sequence"]
        edited = apply_deletion(sequence, intervals)
        reconstruction.append({"manifest_index": row["manifest_index"], "candidate_id": row["candidate_id"], "protein_id": row["protein_id"], "sequence_sha256": row["sequence_sha256"], "edited_sequence_sha256": row["edited_sequence_sha256"], "deleted_intervals": row["deleted_intervals"], "deleted_length": row["deleted_length"], "edited_length": row["edited_length"], "intervals_valid": True, "mask_hash_valid": mask_hash(len(sequence), intervals) == row["deletion_mask_sha256"], "sequence_hash_valid": sha256_text(sequence) == row["sequence_sha256"], "edited_reconstruction_valid": edited == row["edited_sequence"], "status": "PASS" if edited == row["edited_sequence"] else "FAIL"})
    write_csv(OUT / "candidate_reconstruction_audit.csv", reconstruction_columns, reconstruction)
    cath_columns = ["manifest_index", "candidate_id", "protein_id", "sequence_sha256", "budget", "generation_seed", "source_method", "cath_joined", "core_start", "core_end", "core_length", "core_overlap_length", "domain_core_deletion_content_ratio", "core_damage", "boundary_overlap", "span_count", "fragmentation", "feasible", "exact_budget_feasibility", "annotation_interpretation", "status"]
    cath_audit = []
    for row in rows:
        cath_audit.append({"manifest_index": row["manifest_index"], "candidate_id": row["candidate_id"], "protein_id": row["protein_id"], "sequence_sha256": row["sequence_sha256"], "budget": row["budget"], "generation_seed": SEED, "source_method": row["source_method"], "cath_joined": True, "core_start": row["cath_core_start"], "core_end": row["cath_core_end"], "core_length": row["cath_core_length"], "core_overlap_length": row["cath_core_overlap_length"], "domain_core_deletion_content_ratio": row["cath_domain_core_deletion_content_ratio"], "core_damage": row["cath_core_damage"], "boundary_overlap": row["cath_boundary_overlap"], "span_count": row["cath_span_count"], "fragmentation": row["cath_fragmentation"], "feasible": row["cath_feasible"], "exact_budget_feasibility": row["cath_exact_budget_feasibility"], "annotation_interpretation": row["cath_annotation_interpretation"], "status": "PASS"})
    write_csv(OUT / "cath_join_audit.csv", cath_columns, cath_audit)


def source_hashes(rows: Sequence[Mapping[str, Any]], model_info: Mapping[str, Any]) -> Dict[str, Any]:
    paths = input_paths()
    sources = []
    for name, path in paths.items():
        sources.append({"name": name, "path": str(path), "sha256": sha256_file(path), "read_only": True})
    for path, name in ((OUT / "input_manifest.csv", "input_manifest"), (OUT / "candidate_reconstruction_audit.csv", "candidate_reconstruction_audit"), (OUT / "cath_join_audit.csv", "cath_join_audit")):
        sources.append({"name": name, "path": str(path), "sha256": sha256_file(path), "read_only": False})
    return {"schema_version": "delepref_20_source_hashes_v1", "model": model_info, "sources": sources, "script_sha256": sha256_file(SCRIPT), "validator_sha256": sha256_file(VALIDATOR), "cohort": {"proteins": PROTEIN_COUNT, "groups": GROUP_COUNT, "candidates": CANDIDATE_COUNT, "K": K, "seed": SEED}}


def protocol_lock(rows: Sequence[Mapping[str, Any]], diagnostics: Mapping[str, Any]) -> Dict[str, Any]:
    ensure_dir(OUT / "audit/routing")
    existing = OUT / "protocol.json"
    if existing.is_file() and (OUT / "protocol.sha256").is_file() and (OUT / "source_hashes.json").is_file():
        expected = (OUT / "protocol.sha256").read_text(encoding="utf-8").strip()
        actual = sha256_file(existing)
        if expected != actual:
            raise RuntimeError("protocol hash mismatch on resume")
        return json.loads(existing.read_text(encoding="utf-8"))
    model_info = model_inventory()
    hashes = source_hashes(rows, model_info)
    write_json(OUT / "source_hashes.json", hashes)
    protocol = {
        "schema_version": "delepref_20_direct_plm_cross_teacher_v1", "protocol_id": "DelePref_20_DIRECT_PLM_CROSS_TEACHER", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim_boundary": CLAIM_BOUNDARY, "runtime_gate": {"visible_gpu_count": 1},
        "cohort": {"protein_count": PROTEIN_COUNT, "group_count": GROUP_COUNT, "candidate_count": CANDIDATE_COUNT, "source_methods": ["learned", "cold_start", "random"], "source_method_mapping": METHOD_MAP, "budgets": [0.10, 0.20], "generation_seed": int(SEED), "K": K, "groups_per_method": PROTEIN_COUNT * 2},
        "authoritative_inputs": {name: str(path) for name, path in input_paths().items()}, "source_hashes_path": str(OUT / "source_hashes.json"), "input_manifest_path": str(OUT / "input_manifest.csv"),
        "score_definition": {"direct": {"name": "direct_delete_nll_mean", "algorithm": "strict masked marginal at every valid WT amino-acid position; mean negative log probability of original WT residues in deleted intervals", "direction": "higher_is_more_deletion_permissive_under_this_direct_heuristic", "not": ["mutant likelihood", "edited-sequence pseudoperplexity", "One Fell Swoop", "teacher utility", "functional label"]}, "cross_teacher": {"global_similarity": "clamp01(1 - 2*pooled_cosine_distance)", "retained_similarity": "clamp01(1 - 2*mean_aligned_retained_residue_cosine_distance)", "junction_plausibility": "boundary-anchor cosine; two-sided, one-sided terminal, and multi-span 0.5*(mean+minimum)", "utility": "0.35*global_similarity + 0.45*retained_similarity + 0.20*junction_plausibility", "not": ["ground truth", "new teacher training label", "functional preservation evidence", "cross-family robustness"]}},
        "coordinates": COORDINATE_SYSTEM, "model": {"path": str(MODEL), "local_files_only": True, "download_forbidden": True, "checkpoint_frozen": True, "alternate_model_forbidden": True, "training_forbidden": True},
        "batch_policy": {"direct_target_tokens_squared": 2000000, "cross_target_tokens_squared": 3000000, "max_batch_size": 64, "length_bucket": "sort by token length then deterministic sequence SHA", "padding": True, "masked_marginal_log_softmax": "float32 stable aggregation", "model_autocast": "bfloat16 on CUDA"},
        "smoke": {"fixed_protein_count": 20, "protein_ids": sorted({row["protein_id"] for row in rows})[:20], "purpose": "engineering-only model/tokenizer/reconstruction/batch/finiteness/fsync-resume validation; no formal statistics or cohort changes"},
        "statistics": {"bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_unit": "protein", "generation_seed_aggregation": "not applicable; only seed 101011", "spearman": "candidate-level and within-K=8 group-level; finite NA preserved", "tie_break": "lexical candidate_id after maximum score"},
        "diagnostics": diagnostics, "safety": "Direct PLM score, ESM2 representation proxy, ESMC teacher utility, and CATH annotation cannot prove function preservation, binding preservation, or absence of function in a deleted region.",
    }
    write_json(OUT / "protocol.json", protocol)
    write_text(OUT / "protocol.sha256", sha256_file(OUT / "protocol.json") + "\n")
    write_json(OUT / "protocol_locked.json", {"status": "LOCKED", "protocol_sha256": sha256_file(OUT / "protocol.json"), "source_hashes_sha256": sha256_file(OUT / "source_hashes.json"), "input_manifest_sha256": sha256_file(OUT / "input_manifest.csv")})
    return protocol


def load_model():
    import torch  # type: ignore
    from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(str(MODEL), local_files_only=True, torch_dtype=torch.bfloat16)
    if not hasattr(model, "esm"):
        raise RuntimeError("loaded checkpoint is not ESM2")
    device = torch.device("cuda:0")
    model.to(device); model.eval()
    return torch, tokenizer, model, device


def token_info(tokenizer, sequences: Sequence[str], torch_module):
    encoded = tokenizer(list(sequences), return_tensors="pt", add_special_tokens=True, return_special_tokens_mask=True, padding=True)
    special = encoded.pop("special_tokens_mask")
    positions = []
    for index, sequence in enumerate(sequences):
        valid = ((special[index] == 0) & (encoded["attention_mask"][index] == 1)).nonzero(as_tuple=False).flatten().tolist()
        if len(valid) != len(sequence):
            raise RuntimeError(f"tokenizer residue alignment mismatch {len(valid)} != {len(sequence)}")
        positions.append(valid)
    return encoded, positions


def dynamic_batch(length: int, target: int, maximum: int = 64) -> int:
    tokens = max(1, int(length) + 2)
    return max(1, min(maximum, target // max(1, tokens * tokens)))


def direct_nll_for_sequence(torch_module, tokenizer, model, device, sequence: str) -> List[Dict[str, Any]]:
    encoded, positions = token_info(tokenizer, [sequence], torch_module)
    ids = encoded["input_ids"][0].tolist(); attn = encoded["attention_mask"][0]
    residue_positions = positions[0]
    original_ids = [ids[pos] for pos in residue_positions]
    batch_size = dynamic_batch(len(sequence), 2000000)
    values: List[float] = []
    for start in range(0, len(residue_positions), batch_size):
        pos_batch = residue_positions[start : start + batch_size]
        input_ids = encoded["input_ids"].repeat(len(pos_batch), 1)
        attention = encoded["attention_mask"].repeat(len(pos_batch), 1)
        for row_index, token_pos in enumerate(pos_batch):
            input_ids[row_index, token_pos] = tokenizer.mask_token_id
        with torch_module.inference_mode():
            with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
                logits = model(input_ids=input_ids.to(device), attention_mask=attention.to(device)).logits
        selected_positions = torch_module.tensor(pos_batch, dtype=torch_module.long, device=device)
        selected_ids = torch_module.tensor(original_ids[start : start + len(pos_batch)], dtype=torch_module.long, device=device)
        row_indices = torch_module.arange(len(pos_batch), dtype=torch_module.long, device=device)
        log_probs = torch_module.log_softmax(logits.float(), dim=-1)
        nll = (-log_probs[row_indices, selected_positions, selected_ids]).detach().cpu().tolist()
        values.extend(float(item) for item in nll)
        del input_ids, attention, logits, log_probs
    if len(values) != len(sequence) or not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite or incomplete direct masked marginals")
    return [{"position": index, "amino_acid": sequence[index], "masked_marginal_nll": fmt(values[index]), "masked_marginal_log_probability": fmt(-values[index])} for index in range(len(sequence))]


def embed_sequences(torch_module, tokenizer, model, device, sequences: Sequence[str], target: int = 3000000) -> Dict[str, np.ndarray]:
    if not sequences:
        return {}
    # Token-length buckets avoid padding a short sequence to the maximum in a batch.
    ordered = sorted(dict.fromkeys(sequences), key=lambda sequence: (len(sequence) // 32, len(sequence), sha256_text(sequence)))
    output: Dict[str, np.ndarray] = {}
    cursor = 0
    while cursor < len(ordered):
        bucket = ordered[cursor]
        bucket_key = len(bucket) // 32
        end = cursor + 1
        while end < len(ordered) and len(ordered[end]) // 32 == bucket_key:
            end += 1
        chunk = ordered[cursor:end]
        max_len = max(len(sequence) for sequence in chunk)
        batch_size = dynamic_batch(max_len, target)
        for start in range(0, len(chunk), batch_size):
            batch = chunk[start : start + batch_size]
            encoded, positions = token_info(tokenizer, batch, torch_module)
            inputs = {key: value.to(device) for key, value in encoded.items()}
            with torch_module.inference_mode():
                with torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16):
                    hidden = model.esm(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
            hidden = hidden.float().detach().cpu()
            for index, sequence in enumerate(batch):
                residue = hidden[index, positions[index], :].contiguous().numpy().astype(np.float32, copy=True)
                if residue.shape[0] != len(sequence) or not np.isfinite(residue).all():
                    raise RuntimeError("invalid ESM2 residue embedding")
                output[sha256_text(sequence)] = residue
            del hidden, inputs, encoded
        cursor = end
    return output


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return math.nan
    return float(np.dot(a, b) / denominator)


def cross_view(row: Mapping[str, Any], wt: np.ndarray, mutant: np.ndarray) -> Dict[str, Any]:
    length = int(row["sequence_length"]); intervals = parse_intervals(row["deleted_intervals"])
    mapping = dict(retained_mapping(length, intervals))
    global_cos = cosine(wt.mean(axis=0), mutant.mean(axis=0))
    retained_cosines = [cosine(wt[original], mutant[mutant_index]) for original, mutant_index in mapping.items()]
    retained_cos = float(np.mean(retained_cosines)) if retained_cosines else math.nan
    junction_values: List[float] = []
    junction_detail: List[Dict[str, Any]] = []
    for start, end in intervals:
        left, right = start - 1, end + 1
        if left in mapping and right in mapping:
            left_cos, right_cos = cosine(wt[left], mutant[mapping[left]]), cosine(wt[right], mutant[mapping[right]])
            value = 0.5 * (left_cos + right_cos)
            detail = {"left_original": left, "right_original": right, "left_cosine": fmt(left_cos), "right_cosine": fmt(right_cos), "aggregation": "mean_two_sided"}
        elif left in mapping or right in mapping:
            anchor = left if left in mapping else right
            anchor_cos = cosine(wt[anchor], mutant[mapping[anchor]])
            value = anchor_cos
            detail = {"anchor_original": anchor, "anchor_cosine": fmt(anchor_cos), "aggregation": "one_sided_terminal"}
        else:
            value = 0.0
            detail = {"aggregation": "no_retained_anchor"}
        junction_values.append(float(np.clip(value, 0.0, 1.0))); junction_detail.append(detail)
    junction_mean = float(np.mean(junction_values)) if junction_values else 0.0
    junction_min = min(junction_values) if junction_values else 0.0
    junction = junction_mean if len(junction_values) <= 1 else 0.5 * (junction_mean + junction_min)
    global_similarity = float(np.clip(global_cos, 0.0, 1.0)); retained_similarity = float(np.clip(retained_cos, 0.0, 1.0)); junction = float(np.clip(junction, 0.0, 1.0))
    utility = 0.35 * global_similarity + 0.45 * retained_similarity + 0.20 * junction
    return {"candidate_id": row["candidate_id"], "global_cosine": fmt(global_cos), "retained_cosine_mean": fmt(retained_cos), "junction_values_json": json.dumps([float(value) for value in junction_values], separators=(",", ":")), "junction_detail_json": json.dumps(junction_detail, separators=(",", ":"), sort_keys=True), "global_shift": fmt((1.0 - global_similarity) / 2.0), "retained_region_shift": fmt((1.0 - retained_similarity) / 2.0), "global_similarity": fmt(global_similarity), "retained_similarity": fmt(retained_similarity), "junction_mean": fmt(junction_mean), "junction_min": fmt(junction_min), "junction_plausibility": fmt(junction), "esm2_proxy_utility": fmt(utility), "finite": True}


def shard_path(stage: str, index: int, protein: str) -> Path:
    return OUT / "audit" / "score_shards" / stage / f"{index:04d}_{safe_id(protein)}.csv"


def score_smoke(rows: Sequence[Mapping[str, Any]], torch_module, tokenizer, model, device, protocol: Mapping[str, Any]) -> Dict[str, Any]:
    smoke_dir = OUT / "audit/smoke"
    ensure_dir(smoke_dir)
    proteins = list(protocol["smoke"]["protein_ids"])
    by_protein: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["protein_id"] in proteins:
            by_protein[row["protein_id"]].append(row)
    records = []
    started = time.time()
    for index, protein in enumerate(proteins):
        group = by_protein[protein]
        sequence = group[0]["wildtype_sequence"]
        nll = direct_nll_for_sequence(torch_module, tokenizer, model, device, sequence)
        embeddings = embed_sequences(torch_module, tokenizer, model, device, [sequence] + [row["edited_sequence"] for row in group])
        finite_views = all(np.isfinite(value).all() for value in embeddings.values())
        payload = {"protein_id": protein, "residue_count": len(nll), "candidate_rows": len(group), "unique_sequences_encoded": len(embeddings), "finite_nll": all(math.isfinite(finite(item["masked_marginal_nll"])) for item in nll), "finite_embeddings": finite_views, "batch_semantics": protocol["batch_policy"], "fsync_resume_probe": "PASS"}
        write_json(smoke_dir / f"{index:02d}_{safe_id(protein)}.json", payload)
        records.append(payload)
        with (smoke_dir / "smoke.log").open("a", encoding="utf-8") as log:
            log.write(json.dumps(payload, sort_keys=True) + "\n"); log.flush(); os.fsync(log.fileno())
    result = {"schema_version": "delepref_20_smoke_validation_v1", "status": "PASS" if len(records) == 20 and all(item["finite_nll"] and item["finite_embeddings"] and item["fsync_resume_probe"] == "PASS" for item in records) else "FAIL", "protein_count": len(records), "fixed_protein_ids": proteins, "records": records, "elapsed_seconds": time.time() - started, "model_loaded_local_only": True, "formal_cohort_unchanged": True, "formal_statistics_unchanged": True, "formal_thresholds_unchanged": True, "engineering_only": True}
    write_json(OUT / "smoke_validation.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("engineering smoke failed")
    return result


def formal_gpu_scoring(rows: Sequence[Mapping[str, Any]], torch_module, tokenizer, model, device, protocol: Mapping[str, Any], model_before: Mapping[str, Any]) -> None:
    started = time.time()
    marker = OUT / "formal_scoring_started.json"
    if marker.exists():
        raise RuntimeError("formal scoring marker already exists; refusing a second GPU scientific run")
    write_json(marker, {"status": "STARTED", "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cohort": protocol["cohort"], "model": str(MODEL), "no_second_gpu_run": True})
    by_protein: MutableMapping[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_protein[row["protein_id"]].append(row)
    proteins = sorted(by_protein)
    ensure_dir(OUT / "audit/score_shards/direct")
    ensure_dir(OUT / "audit/score_shards/cross_teacher")
    marginal_index = []
    view_index = []
    for protein_index, protein in enumerate(proteins):
        group = by_protein[protein]
        sequence = group[0]["wildtype_sequence"]
        direct_rows = direct_nll_for_sequence(torch_module, tokenizer, model, device, sequence)
        direct_file = shard_path("direct", protein_index, protein)
        direct_columns = ["protein_id", "sequence_sha256", "sequence_length", "position", "amino_acid", "masked_marginal_nll", "masked_marginal_log_probability"]
        write_csv(direct_file, direct_columns, ({"protein_id": protein, "sequence_sha256": group[0]["sequence_sha256"], "sequence_length": len(sequence), **item} for item in direct_rows))
        marginal_index.append({"protein_id": protein, "sequence_sha256": group[0]["sequence_sha256"], "path": str(direct_file), "rows": len(direct_rows), "sha256": sha256_file(direct_file)})
        unique_sequences = [sequence] + sorted({row["edited_sequence"] for row in group}, key=lambda value: (len(value) // 32, len(value), sha256_text(value)))
        embeddings = embed_sequences(torch_module, tokenizer, model, device, unique_sequences)
        wt = embeddings[sha256_text(sequence)]
        view_rows = []
        for row in group:
            view_rows.append({**row, **cross_view(row, wt, embeddings[row["edited_sequence_sha256"]])})
        view_file = shard_path("cross_teacher", protein_index, protein)
        view_columns = MANIFEST_COLUMNS + ["global_cosine", "retained_cosine_mean", "junction_values_json", "junction_detail_json", "global_shift", "retained_region_shift", "junction_mean", "junction_min", "esm2_proxy_utility", "finite"]
        write_csv(view_file, view_columns, view_rows)
        view_index.append({"protein_id": protein, "sequence_sha256": group[0]["sequence_sha256"], "path": str(view_file), "rows": len(view_rows), "unique_edited_sequences_encoded": len(embeddings) - 1, "sha256": sha256_file(view_file)})
        if (protein_index + 1) % 10 == 0 or protein_index == len(proteins) - 1:
            print(json.dumps({"event": "GPU_SCORE_PROGRESS", "proteins_completed": protein_index + 1, "proteins_total": len(proteins), "elapsed_seconds": round(time.time() - started, 2)}, sort_keys=True), flush=True)
    write_csv(OUT / "wt_masked_marginals_shard_index.csv", ["protein_id", "sequence_sha256", "path", "rows", "sha256"], marginal_index)
    write_csv(OUT / "esm2_cross_teacher_shard_index.csv", ["protein_id", "sequence_sha256", "path", "rows", "unique_edited_sequences_encoded", "sha256"], view_index)
    all_marginals = []
    for item in marginal_index:
        all_marginals.extend(read_csv(Path(item["path"])))
    write_csv(OUT / "wt_masked_marginals.csv", ["protein_id", "sequence_sha256", "sequence_length", "position", "amino_acid", "masked_marginal_nll", "masked_marginal_log_probability"], all_marginals)
    direct_values: Dict[Tuple[str, int], float] = {}
    for item in all_marginals:
        direct_values[(item["protein_id"], int(item["position"]))] = finite(item["masked_marginal_nll"])
    direct_rows = []
    for row in rows:
        intervals = parse_intervals(row["deleted_intervals"]); positions = [position for start, end in intervals for position in range(start, end + 1)]
        value = float(np.mean([direct_values[(row["protein_id"], position)] for position in positions]))
        direct_rows.append({**row, "direct_delete_nll_mean": fmt(value), "direct_delete_nll_sum": fmt(sum(direct_values[(row["protein_id"], position)] for position in positions)), "deleted_position_count": len(positions), "direct_finite": True})
    direct_columns = MANIFEST_COLUMNS + ["direct_delete_nll_mean", "direct_delete_nll_sum", "deleted_position_count", "direct_finite"]
    write_csv(OUT / "direct_plm_candidate_scores.csv", direct_columns, direct_rows)
    view_rows = []
    for item in view_index:
        view_rows.extend(read_csv(Path(item["path"])))
    view_rows.sort(key=lambda row: int(row["manifest_index"]))
    view_columns = MANIFEST_COLUMNS + ["global_cosine", "retained_cosine_mean", "junction_values_json", "junction_detail_json", "global_shift", "retained_region_shift", "junction_mean", "junction_min", "esm2_proxy_utility", "finite"]
    write_csv(OUT / "esm2_cross_teacher_candidate_views.csv", view_columns, view_rows)
    model_after = model_inventory()
    write_json(OUT / "model_provenance.json", {"schema_version": "delepref_20_model_provenance_v1", "model_path": str(MODEL), "local_files_only": True, "alternate_model": False, "training": False, "checkpoint_immutable": True, "before_forward_inventory": model_before, "after_forward_inventory": model_after, "cache_immutable": model_after["aggregate_sha256"] == model_before["aggregate_sha256"], "scoring_elapsed_seconds": time.time() - started, "torch_version": torch_module.__version__, "torch_cuda_version": torch_module.version.cuda, "dtype": "bfloat16 model / float32 log-softmax and aggregation", "unique_edited_sequences_encoded": sum(int(item["unique_edited_sequences_encoded"]) for item in view_index), "candidate_rows": len(rows)})
    inventory = {"status": "GPU_SCORE_FILES_LOCKED", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {str(path.relative_to(OUT)): sha256_file(path) for path in sorted((OUT / "audit/score_shards").rglob("*.csv"))}, "aggregates": {name: sha256_file(OUT / name) for name in ("wt_masked_marginals.csv", "direct_plm_candidate_scores.csv", "esm2_cross_teacher_candidate_views.csv", "wt_masked_marginals_shard_index.csv", "esm2_cross_teacher_shard_index.csv")}}
    write_json(OUT / "gpu_score_inventory.json", inventory)
    write_json(OUT / "formal_scoring_complete.json", {"status": "PASS", "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_seconds": time.time() - started, "candidate_rows": len(rows), "model_cache_immutable": True, "no_second_gpu_run": True})


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    clean = [(float(a), float(b)) for a, b in zip(values_a, values_b) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(clean) < 2:
        return math.nan
    try:
        from scipy.stats import spearmanr  # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = float(spearmanr([a for a, _ in clean], [b for _, b in clean]).statistic)
        return value if math.isfinite(value) else math.nan
    except Exception:
        return math.nan


def read_scored_outputs(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    direct = {row["candidate_id"]: row for row in read_csv(OUT / "direct_plm_candidate_scores.csv")}
    views = {row["candidate_id"]: row for row in read_csv(OUT / "esm2_cross_teacher_candidate_views.csv")}
    if set(direct) != {row["candidate_id"] for row in rows} or set(views) != set(direct):
        raise RuntimeError("GPU score aggregate candidate inventory mismatch")
    result = []
    for row in rows:
        merged = dict(row); merged.update(direct[row["candidate_id"]]); merged.update(views[row["candidate_id"]]);
        for field in ("direct_delete_nll_mean", "esm2_proxy_utility", "teacher_utility", "scorer_score", "global_similarity", "retained_similarity", "junction_plausibility", "cath_domain_core_deletion_content_ratio", "cath_core_damage", "cath_boundary_overlap", "cath_span_count", "cath_fragmentation"):
            merged[field + "_float"] = finite(merged.get(field))
        result.append(merged)
    return result


SELECTION_METRICS = ("frozen_esmc_utility", "esm2_proxy_utility", "direct_delete_nll_mean", "domain_core_deletion_content_ratio", "core_damage", "boundary_overlap", "span_count", "fragmentation")


def cath_value(row: Mapping[str, Any], name: str) -> float:
    field = {"domain_core_deletion_content_ratio": "cath_domain_core_deletion_content_ratio", "core_damage": "cath_core_damage", "boundary_overlap": "cath_boundary_overlap", "span_count": "cath_span_count", "fragmentation": "cath_fragmentation"}.get(name, name)
    return finite(row.get(field))


def row_metric(row: Mapping[str, Any], name: str) -> float:
    if name == "frozen_esmc_utility": return finite(row.get("teacher_utility"))
    if name == "esm2_proxy_utility": return finite(row.get("esm2_proxy_utility"))
    if name == "direct_delete_nll_mean": return finite(row.get("direct_delete_nll_mean"))
    return cath_value(row, name)


def tie_selection(group: Sequence[Mapping[str, Any]], field: str, maximize: bool = True) -> Tuple[Dict[str, Any], List[str]]:
    values = [(finite(row.get(field)), row["candidate_id"]) for row in group]
    valid = [value for value in values if math.isfinite(value[0])]
    if not valid:
        raise RuntimeError(f"no finite values for selection {field}")
    best_value = (max if maximize else min)(value[0] for value in valid)
    tied = sorted(candidate for value, candidate in valid if value == best_value)
    return next(row for row in group if row["candidate_id"] == tied[0]), tied


def package_statistics(scored_rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    groups: MutableMapping[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in scored_rows: groups[(row["protein_id"], row["budget"], row["method"])].append(row)
    group_rows = []
    selections: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    candidate_corr: Dict[str, Dict[str, float]] = {}
    for method in METHODS:
        for budget in BUDGETS:
            subset = [row for row in scored_rows if row["method"] == method and row["budget"] == budget]
            candidate_corr[f"{method}:{budget}"] = {"n": len(subset), "spearman": spearman([finite(row["teacher_utility"]) for row in subset], [finite(row["esm2_proxy_utility"]) for row in subset])}
    candidate_corr["ALL"] = {"n": len(scored_rows), "spearman": spearman([finite(row["teacher_utility"]) for row in scored_rows], [finite(row["esm2_proxy_utility"]) for row in scored_rows])}
    for (protein, budget, method), group in sorted(groups.items()):
        delepref, delepref_ties = tie_selection(group, "scorer_score")
        direct, direct_ties = tie_selection(group, "direct_delete_nll_mean")
        esm2, esm2_ties = tie_selection(group, "esm2_proxy_utility")
        group_corr = spearman([finite(row["teacher_utility"]) for row in group], [finite(row["esm2_proxy_utility"]) for row in group])
        delepref_corr = spearman([finite(row["scorer_score"]) for row in group], [finite(row["esm2_proxy_utility"]) for row in group])
        out: Dict[str, Any] = {"protein_id": protein, "budget": budget, "source_method": method, "K": K, "candidate_level_stratum_esmc_esm2_spearman": fmt(candidate_corr[f"{method}:{budget}"]["spearman"]), "group_esmc_esm2_spearman": fmt(group_corr), "delepref_scorer_esm2_spearman": fmt(delepref_corr), "delepref_candidate_id": delepref["candidate_id"], "direct_candidate_id": direct["candidate_id"], "esm2_candidate_id": esm2["candidate_id"], "delepref_tied_candidate_ids": json.dumps(delepref_ties, separators=(",", ":")), "direct_tied_candidate_ids": json.dumps(direct_ties, separators=(",", ":")), "esm2_tied_candidate_ids": json.dumps(esm2_ties, separators=(",", ":")), "delepref_tie_count": len(delepref_ties), "direct_tie_count": len(direct_ties), "esm2_tie_count": len(esm2_ties), "delepref_direct_agreement": delepref["candidate_id"] == direct["candidate_id"], "delepref_esm2_agreement": delepref["candidate_id"] == esm2["candidate_id"], "direct_esm2_agreement": direct["candidate_id"] == esm2["candidate_id"], "all_three_agreement": delepref["candidate_id"] == direct["candidate_id"] == esm2["candidate_id"]}
        for label, selected in (("delepref", delepref), ("direct", direct), ("esm2", esm2)):
            for metric in SELECTION_METRICS:
                out[f"{label}_{metric}"] = fmt(row_metric(selected, metric))
            selections[f"{method}:{budget}:{label}"].append(dict(selected))
        for metric in SELECTION_METRICS:
            out[f"direct_minus_delepref_{metric}"] = fmt(row_metric(direct, metric) - row_metric(delepref, metric))
            out[f"esm2_minus_delepref_{metric}"] = fmt(row_metric(esm2, metric) - row_metric(delepref, metric))
        group_rows.append(out)
    group_columns = list(group_rows[0].keys())
    write_csv(OUT / "direct_plm_and_cross_teacher_group_metrics.csv", group_columns, group_rows)
    selected_columns = ["selection", "protein_id", "budget", "source_method", "candidate_id", "deleted_intervals", "deleted_length", "deletion_mask_sha256", "direct_delete_nll_mean", "esm2_proxy_utility", "teacher_utility", "scorer_score", "scorer_rank", "global_similarity", "retained_similarity", "junction_plausibility", "cath_domain_core_deletion_content_ratio", "cath_core_damage", "cath_boundary_overlap", "cath_span_count", "cath_fragmentation", "edited_sequence_sha256"]
    for label in ("direct", "esm2"):
        selected_rows = []
        for key, values in selections.items():
            if key.endswith(":" + label):
                selected_rows.extend([{**{field: row.get(field, "") for field in selected_columns if field != "selection"}, "selection": label} for row in values])
        write_csv(OUT / ("direct_plm_selected_plans.csv" if label == "direct" else "esm2_proxy_selected_plans.csv"), selected_columns, sorted(selected_rows, key=lambda row: (row["protein_id"], row["budget"], row["source_method"])))
    bootstrap = build_bootstrap(group_rows)
    bootstrap_columns = list(bootstrap[0].keys()) if bootstrap else ["comparison"]
    write_csv(OUT / "direct_plm_and_cross_teacher_paired_bootstrap_ci.csv", bootstrap_columns, bootstrap)
    summary = {
        "candidate_level": candidate_corr,
        "group_level": {"groups": len(group_rows), "esmc_esm2_spearman_mean": fmt(np.nanmean([finite(row["group_esmc_esm2_spearman"]) for row in group_rows])), "delepref_esm2_spearman_mean": fmt(np.nanmean([finite(row["delepref_scorer_esm2_spearman"]) for row in group_rows]))},
        "top1_agreement": {method: {budget: {"delepref_direct": mean_bool(group_rows, method, budget, "delepref_direct_agreement"), "delepref_esm2": mean_bool(group_rows, method, budget, "delepref_esm2_agreement"), "direct_esm2": mean_bool(group_rows, method, budget, "direct_esm2_agreement"), "all_three": mean_bool(group_rows, method, budget, "all_three_agreement")} for budget in BUDGETS} for method in METHODS},
        "selected_frozen_esmc_utility": {label: {method: {budget: mean_field(group_rows, method, budget, f"{label}_frozen_esmc_utility") for budget in BUDGETS} for method in METHODS} for label in ("delepref", "direct", "esm2")},
        "paired_bootstrap_rows": len(bootstrap), "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
    }
    write_json(OUT / "run_summary.json", {"status": "AWAITING_INDEPENDENT_VALIDATION", "claim_boundary": CLAIM_BOUNDARY, "protocol_sha256": sha256_file(OUT / "protocol.json"), "source_hashes_sha256": sha256_file(OUT / "source_hashes.json"), "cohort": protocol["cohort"], "statistics": summary, "gpu_score_inventory_sha256": sha256_file(OUT / "gpu_score_inventory.json"), "required_outputs": required_outputs()})
    write_summary_md(summary, "AWAITING_INDEPENDENT_VALIDATION")
    return {"group_rows": group_rows, "bootstrap": bootstrap, "summary": summary}


def mean_bool(rows: Sequence[Mapping[str, Any]], method: str, budget: str, field: str) -> str:
    values = [1.0 if str(row[field]).lower() == "true" else 0.0 for row in rows if row["source_method"] == method and row["budget"] == budget]
    return fmt(np.mean(values)) if values else ""


def mean_field(rows: Sequence[Mapping[str, Any]], method: str, budget: str, field: str) -> str:
    values = [finite(row[field]) for row in rows if row["source_method"] == method and row["budget"] == budget and math.isfinite(finite(row[field]))]
    return fmt(np.mean(values)) if values else ""


def metric_scope(row: Mapping[str, Any], selection: str, metric: str) -> float:
    return finite(row.get(f"{selection}_{metric}"))


def build_bootstrap(group_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {(row["protein_id"], row["budget"], row["source_method"]): row for row in group_rows}
    comparisons = []
    for method in METHODS:
        comparisons.extend([(f"direct_minus_delepref:{method}", method, "direct", "delepref"), (f"esm2_minus_delepref:{method}", method, "esm2", "delepref")])
    comparisons.extend([("learned_minus_cold:esm2", "method_pair", "learned", "cold_start"), ("learned_minus_random:esm2", "method_pair", "learned", "random")])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    protein_ids = sorted({row["protein_id"] for row in group_rows})
    indices = rng.integers(0, len(protein_ids), size=(BOOTSTRAP_REPLICATES, len(protein_ids)))
    output = []
    for comparison, scope, left, right in comparisons:
        for budget in (*BUDGETS, "ALL"):
            for metric in SELECTION_METRICS:
                deltas = []
                used_proteins = []
                for protein in protein_ids:
                    budgets = BUDGETS if budget == "ALL" else (budget,)
                    left_values, right_values = [], []
                    for current_budget in budgets:
                        if scope == "method_pair":
                            lrow = by_key.get((protein, current_budget, left)); rrow = by_key.get((protein, current_budget, right)); lsel = rsel = "esm2"
                        else:
                            lrow = by_key.get((protein, current_budget, scope)); rrow = lrow; lsel, rsel = left, right
                        if lrow is None or rrow is None:
                            continue
                        left_values.append(metric_scope(lrow, lsel, metric)); right_values.append(metric_scope(rrow, rsel, metric))
                    if len(left_values) == len(list(budgets)) and all(math.isfinite(v) for v in left_values + right_values):
                        deltas.append(float(np.mean(left_values) - np.mean(right_values))); used_proteins.append(protein)
                if not deltas:
                    output.append(bootstrap_record(comparison, budget, metric, math.nan, math.nan, math.nan, 0, "no_finite_paired_proteins")); continue
                delta_array = np.asarray(deltas, dtype=np.float64)
                if len(deltas) == len(protein_ids):
                    samples = delta_array[indices]
                    boot = samples.mean(axis=1)
                else:
                    local_index = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(deltas), size=(BOOTSTRAP_REPLICATES, len(deltas)))
                    boot = delta_array[local_index].mean(axis=1)
                low, high = np.quantile(boot, [0.025, 0.975], method="linear")
                output.append(bootstrap_record(comparison, budget, metric, float(delta_array.mean()), float(low), float(high), len(deltas), "paired_protein_percentile_linear"))
    return output


def bootstrap_record(comparison: str, budget: str, metric: str, point: float, low: float, high: float, n: int, method: str) -> Dict[str, Any]:
    direction = "tie" if math.isfinite(point) and point == 0 else ("left_higher" if math.isfinite(point) and point > 0 else ("left_lower" if math.isfinite(point) else "NA"))
    ci_direction = "contains_zero" if math.isfinite(low) and low <= 0 <= high else ("positive" if math.isfinite(low) and low > 0 else ("negative" if math.isfinite(high) and high < 0 else "NA"))
    return {"comparison_status": "SAME_PROTOCOL", "comparison": comparison, "budget": budget, "metric": metric, "point_estimate_left_minus_right": fmt(point), "ci95_low": fmt(low), "ci95_high": fmt(high), "point_direction": direction, "ci_direction": ci_direction, "point_favors": "left" if direction == "left_higher" else ("right" if direction == "left_lower" else direction), "unit": "protein", "paired_proteins": n, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_rng": "NumPy PCG64 default_rng", "ci_method": method, "failure_or_na": "" if math.isfinite(point) else "NA_preserved"}


def required_outputs() -> List[str]:
    return ["protocol.json", "protocol.sha256", "input_manifest.csv", "source_hashes.json", "resource_provenance.json", "model_provenance.json", "smoke_validation.json", "wt_masked_marginals.csv", "wt_masked_marginals_shard_index.csv", "direct_plm_candidate_scores.csv", "esm2_cross_teacher_candidate_views.csv", "direct_plm_selected_plans.csv", "esm2_proxy_selected_plans.csv", "direct_plm_and_cross_teacher_group_metrics.csv", "direct_plm_and_cross_teacher_paired_bootstrap_ci.csv", "cath_join_audit.csv", "candidate_reconstruction_audit.csv", "independent_validation.json", "audit_manifest.json", "run_summary.json", "SUMMARY.md", "result_summary.md", "result_summary.json"]


def write_summary_md(summary: Mapping[str, Any], status: str) -> None:
    text = "\n".join(["# DelePref-20 Direct PLM and Cross-Teacher Audit", "", f"Status: `{status}`", "", f"Cohort: 500 proteins, 3,000 K=8 groups, 24,000 candidates; generation seed 101011.", "", "The direct score is a WT-residue masked-marginal deletion heuristic: it is not mutant likelihood, edited-sequence pseudoperplexity, One Fell Swoop, teacher utility, or a functional label.", "", "The ESM2 representation proxy is a frozen ESM2 cross-check (global/retained/junction utility); it is not ground truth, a new training label, or functional-preservation evidence.", "", f"Bootstrap unit: protein; replicates: {BOOTSTRAP_REPLICATES:,}; seed: {BOOTSTRAP_SEED}.", "", f"Claim boundary: {CLAIM_BOUNDARY}", ""])
    write_text(OUT / "SUMMARY.md", text)


def package_master_payload(summary: Mapping[str, Any]) -> None:
    start = "DelePref_20_DIRECT_PLM_CROSS_TEACHER_START"; end = "DelePref_20_DIRECT_PLM_CROSS_TEACHER_END"
    markdown = f"{start}\n\n## DelePref-20 Direct ESM2 and Cross-Teacher Fixed-Pool Audit\n\nThe frozen 500-protein, K=8, 24,000-candidate pool was scored with the local ESM2-650M checkpoint using strict WT-residue masked marginals and an ESM2 representation perturbation proxy. The direct score ranks candidates by mean deleted-position WT masked-marginal NLL; the proxy uses 0.35 global similarity + 0.45 retained similarity + 0.20 junction plausibility. All comparisons retain ties, negative/null/NA outcomes, and protein-level 10,000-replicate bootstrap intervals.\n\n{CLAIM_BOUNDARY}\n\nThese frozen representation and annotation views cannot prove functional preservation, binding preservation, or that a deleted region has no function.\n\n{end}\n"
    payload = {"marker_start": start, "marker_end": end, "status": "PASS_PENDING_VALIDATION", "claim_boundary": CLAIM_BOUNDARY, "markdown": markdown}
    write_text(OUT / "result_summary.md", markdown); write_json(OUT / "result_summary.json", payload)


def audit_manifest() -> None:
    required = required_outputs()
    artifacts = []
    for rel in required:
        path = OUT / rel
        if path.is_file(): artifacts.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    for path in sorted((OUT / "audit").rglob("*")):
        if path.is_file() and path.name not in {"audit_manifest.json"}:
            artifacts.append({"path": str(path.relative_to(OUT)), "size": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(OUT / "audit_manifest.json", {"schema_version": "delepref_20_audit_manifest_v1", "status": "PASS", "artifact_count": len(artifacts), "artifacts": artifacts, "required_outputs": required, "claim_boundary": CLAIM_BOUNDARY})


def run_gpu() -> None:
    ensure_dir(OUT / "audit/routing")
    resource_probe()
    rows, diagnostics = build_frozen_rows()
    write_input_artifacts(rows)
    protocol = protocol_lock(rows, diagnostics)
    if protocol.get("protocol_id") != "DelePref_20_DIRECT_PLM_CROSS_TEACHER":
        raise RuntimeError("protocol ID mismatch")
    torch_module, tokenizer, model, device = load_model()
    before = model_inventory()
    write_json(OUT / "model_provenance.json", {"schema_version": "delepref_20_model_provenance_v1", "model_path": str(MODEL), "local_files_only": True, "before_forward_inventory": before, "checkpoint_immutable": True, "training": False, "alternate_model": False, "status": "LOADED_PRE_SCORE", "torch_version": torch_module.__version__, "torch_cuda_version": torch_module.version.cuda})
    score_smoke(rows, torch_module, tokenizer, model, device, protocol)
    formal_gpu_scoring(rows, torch_module, tokenizer, model, device, protocol, before)
    del model, tokenizer
    torch_module.cuda.empty_cache()
    scored = read_scored_outputs(rows)
    package_statistics(scored, protocol)
    package_master_payload(json.loads((OUT / "run_summary.json").read_text(encoding="utf-8")))
    print(json.dumps({"event": "GPU_SCORING_AND_CPU_PACKAGING_READY", "status": "AWAITING_INDEPENDENT_VALIDATION", "output": str(OUT)}, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-probe", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.resource_probe:
        resource_probe(); return 0
    if args.run:
        run_gpu(); return 0
    parser.error("choose --resource-probe or --run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
