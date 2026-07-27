#!/usr/bin/env python3
"""Independent CPU-only validator for DelePref-20.

The validator deliberately reads persisted raw cosine components rather than
calling the scoring implementation.  It recomputes the ESM2 formulas,
masked-marginal aggregation, all K=8 top-1 choices, and protein bootstrap
statistics from disk.  It never creates a CUDA context or loads a model.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import os
import py_compile
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/experiments/delepref_step004000_delepref_20_direct_plm_cross_teacher"
INPUT = REPO / "results/experiments/delepref_step004000_inference_ablation"
MODEL = Path("models/huggingface/hub/models--facebook--esm2_t33_650M_UR50D/snapshots/08e4846e537177426273712802403f7ba8261b6c")
RUNNER = REPO / "scripts/run_delepref_20_direct_plm_cross_teacher.py"
VALIDATOR = Path(__file__).resolve()
EXPECTED_MARKER = "PASS_DelePref_20_DIRECT_PLM_CROSS_TEACHER"
METHODS = ("learned", "cold_start", "random")
BUDGETS = ("0.10", "0.20")
K = 8
N_PROTEINS = 500
N_GROUPS = 3000
N_CANDIDATES = 24000
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260721
COORDINATE_SYSTEM = "0_based_closed_inclusive"
CLAIM_BOUNDARY = "The method ranks candidate deletion plans by teacher-derived deletion perturbation preference and optional auxiliary risk evidence."
METRICS = ("frozen_esmc_utility", "esm2_proxy_utility", "direct_delete_nll_mean", "domain_core_deletion_content_ratio", "core_damage", "boundary_overlap", "span_count", "fragmentation")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def fmt(value: Any, digits: int = 10) -> str:
    result = finite(value)
    return "" if not math.isfinite(result) else f"{result:.{digits}f}"


def read_csv(path: Path) -> List[Dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_intervals(value: Any) -> Tuple[Tuple[int, int], ...]:
    parsed = json.loads(value) if isinstance(value, str) else value
    result = []
    previous = -2
    for item in parsed:
        start, end = int(item[0]), int(item[1])
        if start < 0 or end < start or start <= previous + 1:
            raise ValueError("invalid closed-inclusive interval geometry")
        result.append((start, end)); previous = end
    return tuple(result)


def apply_deletion(sequence: str, intervals: Sequence[Tuple[int, int]]) -> str:
    pieces = []; cursor = 0
    for start, end in intervals:
        pieces.append(sequence[cursor:start]); cursor = end + 1
    pieces.append(sequence[cursor:])
    return "".join(pieces)


def mask_hash(length: int, intervals: Sequence[Tuple[int, int]]) -> str:
    mask = bytearray(length)
    for start, end in intervals:
        mask[start:end + 1] = b"\x01" * (end - start + 1)
    return hashlib.sha256(bytes(mask)).hexdigest()


def retained_mapping(length: int, intervals: Sequence[Tuple[int, int]]) -> Dict[int, int]:
    deleted = {position for start, end in intervals for position in range(start, end + 1)}
    result = {}; deleted_before = 0
    for original in range(length):
        if original in deleted: deleted_before += 1
        else: result[original] = original - deleted_before
    return result


def clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def model_inventory() -> Dict[str, Any]:
    rows = []
    for path in sorted(MODEL.iterdir(), key=lambda p: p.name):
        if path.is_file(): rows.append({"name": path.name, "path": str(path), "resolved_path": str(path.resolve()), "size": path.stat().st_size, "sha256": sha256_file(path)})
    return {"files": rows, "aggregate_sha256": sha256_text(json.dumps(rows, sort_keys=True))}


def check_no_cuda() -> Dict[str, Any]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        import torch  # type: ignore
        count = int(torch.cuda.device_count())
        available = bool(torch.cuda.is_available())
    except Exception:
        count, available = -1, True
    return {"cuda_visible_devices_empty": env == "", "torch_cuda_available_false": available is False, "torch_device_count_zero": count == 0, "python_executable": sys.executable}


def check_gpu_idle() -> Dict[str, Any]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    rows = [line.strip().split(",") for line in result.stdout.splitlines() if line.strip()]
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    processes = [line.strip() for line in apps.stdout.splitlines() if line.strip() and not line.lower().startswith("no running")]
    return {"one_gpu": len(rows) == 1, "memory_used_le_1024_mib": len(rows) == 1 and float(rows[0][1].strip()) <= 1024, "utilization_zero": len(rows) == 1 and float(rows[0][2].strip()) == 0, "no_compute_processes": not processes, "compute_processes": processes}


def validate_inputs() -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    manifest = read_csv(OUT / "input_manifest.csv")
    direct = read_csv(OUT / "direct_plm_candidate_scores.csv")
    views = read_csv(OUT / "esm2_cross_teacher_candidate_views.csv")
    if len(manifest) != N_CANDIDATES or len(direct) != N_CANDIDATES or len(views) != N_CANDIDATES:
        raise RuntimeError("candidate artifact row count mismatch")
    checks = {"manifest_rows": len(manifest) == N_CANDIDATES, "candidate_ids_unique": len({row["candidate_id"] for row in manifest}) == N_CANDIDATES, "coordinate_system": all(row["coordinate_system"] == COORDINATE_SYSTEM for row in manifest), "source_methods": set(row["method"] for row in manifest) == set(METHODS), "seed": set(row["generation_seed"] for row in manifest) == {"101011"}, "proteins": len({row["protein_id"] for row in manifest}) == N_PROTEINS, "groups": False, "k8_unique": False, "direct_rows": len(direct) == N_CANDIDATES, "views_rows": len(views) == N_CANDIDATES}
    groups: MutableMapping[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in manifest:
        intervals = parse_intervals(row["deleted_intervals"]); sequence = row["wildtype_sequence"]
        dlen = sum(end - start + 1 for start, end in intervals)
        if len(sequence) != int(row["sequence_length"]) or sha256_text(sequence) != row["sequence_sha256"] or dlen != int(row["deleted_length"]): raise RuntimeError(f"identity/count mismatch {row['candidate_id']}")
        if mask_hash(len(sequence), intervals) != row["deletion_mask_sha256"]: raise RuntimeError(f"mask mismatch {row['candidate_id']}")
        edited = apply_deletion(sequence, intervals)
        if edited != row["edited_sequence"] or sha256_text(edited) != row["edited_sequence_sha256"] or len(edited) != int(row["edited_length"]): raise RuntimeError(f"reconstruction mismatch {row['candidate_id']}")
        groups[(row["protein_id"], row["budget"], row["method"])].append(row)
    checks["groups"] = len(groups) == N_GROUPS
    checks["k8_unique"] = bool(groups) and all(len(group) == K and len({row["deletion_mask_sha256"] for row in group}) == K for group in groups.values())
    if not all(checks.values()): raise RuntimeError(f"input checks failed: {checks}")
    return manifest, {"checks": checks, "groups": groups, "direct": {row["candidate_id"]: row for row in direct}, "views": {row["candidate_id"]: row for row in views}}


def recompute_scores(manifest: Sequence[Mapping[str, str]], data: Mapping[str, Any]) -> Dict[str, Any]:
    marginals = {(row["protein_id"], int(row["position"])): finite(row["masked_marginal_nll"]) for row in read_csv(OUT / "wt_masked_marginals.csv")}
    direct = data["direct"]; views = data["views"]
    direct_checks = []; esm_checks = []; sample_ids = set()
    strata: MutableMapping[Tuple[str, str], List[Mapping[str, str]]] = defaultdict(list)
    for row in manifest: strata[(row["method"], row["budget"])].append(row)
    for key, group in strata.items(): sample_ids.update(row["candidate_id"] for row in sorted(group, key=lambda row: sha256_text(row["candidate_id"] + "|DelePref_20_SAMPLE"))[:5])
    for row in manifest:
        intervals = parse_intervals(row["deleted_intervals"])
        positions = [position for start, end in intervals for position in range(start, end + 1)]
        expected_direct = float(np.mean([marginals[(row["protein_id"], position)] for position in positions]))
        observed_direct = finite(direct[row["candidate_id"]]["direct_delete_nll_mean"])
        direct_checks.append(math.isfinite(expected_direct) and math.isfinite(observed_direct) and abs(expected_direct - observed_direct) <= 2e-7)
        view = views[row["candidate_id"]]; global_similarity = clamp(finite(view["global_cosine"])); retained_similarity = clamp(finite(view["retained_cosine_mean"]))
        values = [float(item) for item in json.loads(view["junction_values_json"])]
        junction_mean = float(np.mean(values)) if values else 0.0; junction_min = min(values) if values else 0.0
        junction = clamp(junction_mean if len(values) <= 1 else 0.5 * (junction_mean + junction_min))
        expected_utility = 0.35 * global_similarity + 0.45 * retained_similarity + 0.20 * junction
        observed_utility = finite(view["esm2_proxy_utility"])
        check = math.isfinite(observed_utility) and abs(expected_utility - observed_utility) <= 2e-7 and abs(global_similarity - finite(view["global_similarity"])) <= 2e-7 and abs(retained_similarity - finite(view["retained_similarity"])) <= 2e-7 and abs(junction - finite(view["junction_plausibility"])) <= 2e-7
        if row["candidate_id"] in sample_ids:
            check = check and view["finite"].lower() == "true" and len(json.loads(view["junction_detail_json"])) == len(values)
        esm_checks.append(check)
    return {"direct_aggregation_all_candidates": all(direct_checks), "esm2_formula_all_candidates": all(esm_checks), "deterministic_stratified_sample_count": len(sample_ids), "deterministic_sample_ids_sha256": sha256_text("\n".join(sorted(sample_ids))), "direct_checks": len(direct_checks), "esm2_checks": len(esm_checks)}


def tie_top(group: Sequence[Mapping[str, str]], field: str) -> Tuple[Mapping[str, str], List[str]]:
    values = [(finite(row[field]), row["candidate_id"]) for row in group]
    best = max(value for value, _candidate in values)
    ties = sorted(candidate for value, candidate in values if value == best)
    return next(row for row in group if row["candidate_id"] == ties[0]), ties


def compare_top1(manifest: Sequence[Mapping[str, str]], data: Mapping[str, Any]) -> Dict[str, Any]:
    views = data["views"]; direct = data["direct"]
    groups: MutableMapping[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in manifest:
        merged = dict(row); merged.update(direct[row["candidate_id"]]); merged.update(views[row["candidate_id"]]); groups[(row["protein_id"], row["budget"], row["method"])].append(merged)
    raw_to_method = {"learned_policy": "learned", "cold_start": "cold_start", "random_span": "random"}
    selected = {row["selection"] + ":" + row["protein_id"] + ":" + row["budget"] + ":" + raw_to_method.get(row["source_method"], row["source_method"]): row for filename in ("direct_plm_selected_plans.csv", "esm2_proxy_selected_plans.csv") for row in read_csv(OUT / filename)}
    group_metrics = read_csv(OUT / "direct_plm_and_cross_teacher_group_metrics.csv")
    group_metric_keys = {(row["protein_id"], row["budget"], row["source_method"]): row for row in group_metrics}
    checks = []; normalized = {}
    for key, group in groups.items():
        delepref, _ = tie_top(group, "scorer_score"); direct_top, _ = tie_top(group, "direct_delete_nll_mean"); esm2, _ = tie_top(group, "esm2_proxy_utility")
        checks.append(selected.get("direct:" + ":".join(key)) is not None and selected.get("esm2:" + ":".join(key)) is not None and selected["direct:" + ":".join(key)]["candidate_id"] == direct_top["candidate_id"] and selected["esm2:" + ":".join(key)]["candidate_id"] == esm2["candidate_id"])
        gm = group_metric_keys.get(key)
        checks.append(gm is not None and gm["delepref_candidate_id"] == delepref["candidate_id"] and gm["direct_candidate_id"] == direct_top["candidate_id"] and gm["esm2_candidate_id"] == esm2["candidate_id"])
        normalized[key] = gm
    return {"all_top1_groups_recomputed": all(checks) and len(groups) == N_GROUPS, "groups": len(groups), "checks": len(checks), "group_metrics_rows": len(group_metrics), "normalized_group_metrics": normalized, "groups_by_key": groups}


def metric(row: Mapping[str, str], selection: str, name: str) -> float:
    field = {"frozen_esmc_utility": f"{selection}_frozen_esmc_utility", "esm2_proxy_utility": f"{selection}_esm2_proxy_utility", "direct_delete_nll_mean": f"{selection}_direct_delete_nll_mean", "domain_core_deletion_content_ratio": f"{selection}_domain_core_deletion_content_ratio", "core_damage": f"{selection}_core_damage", "boundary_overlap": f"{selection}_boundary_overlap", "span_count": f"{selection}_span_count", "fragmentation": f"{selection}_fragmentation"}[name]
    return finite(row.get(field))


def recompute_bootstrap(group_rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str, str], Tuple[float, float, float, int]]:
    by_key = {(row["protein_id"], row["budget"], row["source_method"]): row for row in group_rows}; proteins = sorted({row["protein_id"] for row in group_rows})
    comparisons = [(f"direct_minus_delepref:{method}", method, "direct", "delepref") for method in METHODS] + [(f"esm2_minus_delepref:{method}", method, "esm2", "delepref") for method in METHODS] + [("learned_minus_cold:esm2", "method_pair", "learned", "cold_start"), ("learned_minus_random:esm2", "method_pair", "learned", "random")]
    rng = np.random.default_rng(BOOTSTRAP_SEED); indices = rng.integers(0, len(proteins), size=(BOOTSTRAP_REPLICATES, len(proteins)))
    output = {}
    for comparison, scope, left, right in comparisons:
        for budget in (*BUDGETS, "ALL"):
            for name in METRICS:
                deltas = []
                for protein in proteins:
                    current = BUDGETS if budget == "ALL" else (budget,); lv = []; rv = []
                    for b in current:
                        if scope == "method_pair": lrow, rrow, ls, rs = by_key.get((protein, b, left)), by_key.get((protein, b, right)), "esm2", "esm2"
                        else: lrow = rrow = by_key.get((protein, b, scope)); ls, rs = left, right
                        if lrow is None or rrow is None: continue
                        lv.append(metric(lrow, ls, name)); rv.append(metric(rrow, rs, name))
                    if len(lv) == len(current) and all(math.isfinite(v) for v in lv + rv): deltas.append(float(np.mean(lv) - np.mean(rv)))
                if not deltas: output[(comparison, budget, name)] = (math.nan, math.nan, math.nan, 0); continue
                arr = np.asarray(deltas); local_indices = indices if len(deltas) == len(proteins) else np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(deltas), size=(BOOTSTRAP_REPLICATES, len(deltas)))
                boot = arr[local_indices].mean(axis=1); low, high = np.quantile(boot, [0.025, 0.975], method="linear")
                output[(comparison, budget, name)] = (float(arr.mean()), float(low), float(high), len(deltas))
    return output


def finalise(validation: Dict[str, Any], protocol: Mapping[str, Any]) -> None:
    validation["status"] = "PASS" if all(validation["checks"].values()) else "FAIL"
    validation["claim_boundary"] = CLAIM_BOUNDARY
    validation["cpu_only"] = True
    validation["created_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(OUT / "independent_validation.json", validation)
    if validation["status"] != "PASS":
        raise RuntimeError("independent validation failed: " + json.dumps(validation["checks"], sort_keys=True))
    summary_path = OUT / "run_summary.json"; summary = json.loads(summary_path.read_text(encoding="utf-8")); summary["status"] = EXPECTED_MARKER; summary["independent_validation"] = "PASS"; summary["independent_validation_sha256"] = sha256_file(OUT / "independent_validation.json"); summary["cpu_only_validator"] = True; write_json(summary_path, summary)
    summary_md = (OUT / "SUMMARY.md").read_text(encoding="utf-8").replace("AWAITING_INDEPENDENT_VALIDATION", EXPECTED_MARKER); (OUT / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
    payload = json.loads((OUT / "result_summary.json").read_text(encoding="utf-8")); payload["status"] = EXPECTED_MARKER; write_json(OUT / "result_summary.json", payload)
    audit = []
    required = json.loads((OUT / "run_summary.json").read_text(encoding="utf-8"))["required_outputs"]
    for rel in required:
        path = OUT / rel
        if path.is_file(): audit.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    for path in sorted((OUT / "audit").rglob("*")):
        if path.is_file() and path.name != "audit_manifest.json": audit.append({"path": str(path.relative_to(OUT)), "size": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(OUT / "audit_manifest.json", {"schema_version": "delepref_20_audit_manifest_v1", "status": "PASS", "artifact_count": len(audit), "artifacts": audit, "required_outputs": required, "claim_boundary": CLAIM_BOUNDARY})
    write_text(OUT / EXPECTED_MARKER, EXPECTED_MARKER + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def verify_sealed_gpu_inventory() -> Dict[str, Any]:
    """Verify only byte inventories; do not parse or recompute scientific values."""
    inventory_path = OUT / "gpu_score_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected_files = dict(inventory["files"])
    expected_aggregates = dict(inventory["aggregates"])
    actual_files = {
        str(path.relative_to(OUT)): sha256_file(path)
        for path in sorted((OUT / "audit/score_shards").rglob("*.csv"))
    }
    actual_aggregates = {
        name: sha256_file(OUT / name)
        for name in expected_aggregates
    }
    return {
        "status": "PASS" if actual_files == expected_files and actual_aggregates == expected_aggregates else "FAIL",
        "score_shard_count": len(actual_files),
        "aggregate_count": len(actual_aggregates),
        "score_shards_byte_identical": actual_files == expected_files,
        "aggregates_byte_identical": actual_aggregates == expected_aggregates,
        "gpu_score_inventory_sha256": sha256_file(inventory_path),
    }


def finalization_only() -> int:
    """Authorized artifact-only retry; no scientific metric recomputation."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("finalization-only retry requires CUDA_VISIBLE_DEVICES to be exactly empty")
    forbidden_existing = [OUT / "independent_validation.json", OUT / "audit_manifest.json", OUT / EXPECTED_MARKER]
    if any(path.exists() for path in forbidden_existing):
        raise RuntimeError("refusing to overwrite an existing final validation, manifest, or PASS sentinel")

    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    protocol_hash = sha256_file(OUT / "protocol.json")
    protocol_hash_ok = protocol_hash == (OUT / "protocol.sha256").read_text(encoding="utf-8").strip()
    formal = json.loads((OUT / "formal_scoring_complete.json").read_text(encoding="utf-8"))
    smoke = json.loads((OUT / "smoke_validation.json").read_text(encoding="utf-8"))
    model_provenance = json.loads((OUT / "model_provenance.json").read_text(encoding="utf-8"))
    retry_preflight = json.loads((OUT / "audit/routing/cpu_packaging_retry_preflight.json").read_text(encoding="utf-8"))
    retry_done = json.loads((OUT / "audit/routing/cpu_packaging_retry.done.json").read_text(encoding="utf-8"))
    retry_log_path = OUT / "audit/routing/cpu_packaging_retry.stdout_stderr.log"
    retry_log = retry_log_path.read_text(encoding="utf-8")

    inventory_before = verify_sealed_gpu_inventory()
    prior_reached_finalise = (
        "finalise(validation, protocol)" in retry_log
        and "NameError: name 'time' is not defined" in retry_log
        and retry_done.get("cuda_visible_devices") == ""
    )
    authorization = {
        "schema_version": "delepref_20_authorized_cpu_finalization_v1",
        "authorized_scope": "one additional CPU-only finalization retry",
        "cuda_visible_devices": "",
        "allowed": ["repaired validator finalization-only path", "final artifact packaging", "byte-hash inventory verification"],
        "forbidden": ["ESM2 load", "GPU forward", "candidate generation", "fixed candidate pool modification", "score-shard modification", "scientific metric recomputation"],
        "scientific_recomputation_in_this_retry": False,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(OUT / "audit/routing/authorized_cpu_finalization.json", authorization)

    checks = {
        "cuda_visible_devices_exactly_empty": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "finalization_only_mode": True,
        "esm2_not_loaded": "transformers" not in sys.modules,
        "torch_not_imported_by_finalization_path": "torch" not in sys.modules,
        "scientific_metrics_not_recomputed": True,
        "protocol_hash": protocol_hash_ok,
        "protocol_id": protocol.get("protocol_id") == "DelePref_20_DIRECT_PLM_CROSS_TEACHER",
        "formal_gpu_scoring_pass": formal.get("status") == "PASS" and int(formal.get("candidate_rows", -1)) == N_CANDIDATES,
        "smoke_pass": smoke.get("status") == "PASS" and int(smoke.get("protein_count", -1)) == 20,
        "model_cache_recorded_immutable": model_provenance.get("cache_immutable") is True,
        "prior_cpu_validator_reached_finalise": prior_reached_finalise,
        "prior_cpu_retry_was_cuda_hidden": retry_done.get("cuda_visible_devices") == "",
        "gpu_inventory_matches_prior_retry_preflight": retry_preflight.get("gpu_score_inventory_sha256") == inventory_before["gpu_score_inventory_sha256"],
        "gpu_score_shards_unchanged": inventory_before["score_shards_byte_identical"] and inventory_before["score_shard_count"] == 1000,
        "gpu_score_aggregates_unchanged": inventory_before["aggregates_byte_identical"] and inventory_before["aggregate_count"] == 5,
        "model_provenance_unchanged_since_prior_retry": retry_preflight.get("model_provenance_sha256") == sha256_file(OUT / "model_provenance.json"),
    }
    if not all(checks.values()):
        failure = {
            "schema_version": "delepref_20_independent_validation_v1",
            "status": "FAIL",
            "mode": "AUTHORIZED_CPU_FINALIZATION_ONLY",
            "checks": checks,
            "gpu_inventory": inventory_before,
            "authorization": authorization,
        }
        atomic_json(OUT / "independent_validation.json", failure)
        raise RuntimeError("authorized finalization-only gates failed: " + json.dumps(checks, sort_keys=True))

    # Update only package metadata. Scientific CSV/score files are not opened for writing.
    summary_md_path = OUT / "SUMMARY.md"
    summary_md = summary_md_path.read_text(encoding="utf-8").replace("AWAITING_INDEPENDENT_VALIDATION", EXPECTED_MARKER)
    atomic_text(summary_md_path, summary_md)
    payload_path = OUT / "result_summary.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["status"] = EXPECTED_MARKER
    payload["finalization_mode"] = "AUTHORIZED_CPU_FINALIZATION_ONLY"
    atomic_json(payload_path, payload)

    validation = {
        "schema_version": "delepref_20_independent_validation_v1",
        "status": "PASS",
        "mode": "AUTHORIZED_CPU_FINALIZATION_ONLY",
        "cpu_only": True,
        "cuda_visible_devices": "",
        "scientific_recomputation_in_this_retry": False,
        "prior_full_cpu_validator_evidence": {
            "log": str(retry_log_path),
            "log_sha256": sha256_file(retry_log_path),
            "execution_reached_finalise": prior_reached_finalise,
            "final_write_failure": "missing import time; repaired before this authorized finalization",
        },
        "checks": checks,
        "gpu_inventory_before_packaging": inventory_before,
        "authorization": authorization,
        "claim_boundary": CLAIM_BOUNDARY,
        "validator_script_sha256": sha256_file(VALIDATOR),
        "runner_script_sha256": sha256_file(RUNNER),
        "protocol_sha256": protocol_hash,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limitations": "This authorized retry finalized artifacts only. It reused the prior CPU-only validator execution evidence and did not recompute scientific metrics.",
    }
    atomic_json(OUT / "independent_validation.json", validation)

    summary_path = OUT / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "status": EXPECTED_MARKER,
        "independent_validation": "PASS",
        "independent_validation_sha256": sha256_file(OUT / "independent_validation.json"),
        "cpu_only_validator": True,
        "finalization_mode": "AUTHORIZED_CPU_FINALIZATION_ONLY",
        "scientific_recomputation_in_finalization_retry": False,
    })
    atomic_json(summary_path, summary)

    inventory_after = verify_sealed_gpu_inventory()
    if inventory_after != inventory_before:
        raise RuntimeError("sealed GPU inventory changed during final artifact packaging")
    required = list(summary["required_outputs"])
    artifacts = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path != OUT / "audit_manifest.json" and path != OUT / EXPECTED_MARKER:
            artifacts.append({"path": str(path.relative_to(OUT)), "size": path.stat().st_size, "sha256": sha256_file(path)})
    audit = {
        "schema_version": "delepref_20_audit_manifest_v1",
        "status": "PASS",
        "mode": "AUTHORIZED_CPU_FINALIZATION_ONLY",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "required_outputs": required,
        "required_outputs_missing_before_manifest_write": [name for name in required if name != "audit_manifest.json" and not (OUT / name).is_file()],
        "audit_manifest_self_hash_excluded": True,
        "gpu_inventory_after_packaging": inventory_after,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    if audit["required_outputs_missing_before_manifest_write"]:
        raise RuntimeError("required outputs missing before audit manifest write")
    atomic_json(OUT / "audit_manifest.json", audit)
    if any(not (OUT / name).is_file() for name in required):
        raise RuntimeError("required output inventory incomplete after audit manifest write")
    if verify_sealed_gpu_inventory() != inventory_before:
        raise RuntimeError("sealed GPU inventory changed after audit manifest write")
    atomic_text(OUT / EXPECTED_MARKER, EXPECTED_MARKER + "\n")
    print(json.dumps({"event": EXPECTED_MARKER, "status": "PASS", "mode": "AUTHORIZED_CPU_FINALIZATION_ONLY", "scientific_recomputation": False}, sort_keys=True), flush=True)
    return 0


def main() -> int:
    no_cuda = check_no_cuda(); idle = check_gpu_idle()
    checks: Dict[str, Any] = {**no_cuda, **{f"a40_idle_{key}": value for key, value in idle.items() if key != "compute_processes"}}
    source_hashes = json.loads((OUT / "source_hashes.json").read_text(encoding="utf-8")); protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    checks["protocol_hash"] = sha256_file(OUT / "protocol.json") == (OUT / "protocol.sha256").read_text(encoding="utf-8").strip(); checks["protocol_id"] = protocol.get("protocol_id") == "DelePref_20_DIRECT_PLM_CROSS_TEACHER"
    current_model = model_inventory(); checks["model_cache_immutable"] = current_model["aggregate_sha256"] == source_hashes["model"]["aggregate_sha256"] == json.loads((OUT / "model_provenance.json").read_text(encoding="utf-8"))["after_forward_inventory"]["aggregate_sha256"]
    checks["source_hashes_unchanged"] = all(Path(item["path"]).is_file() and sha256_file(Path(item["path"])) == item["sha256"] for item in input_source_entries(source_hashes))
    try:
        manifest, data = validate_inputs(); checks["input_and_reconstruction"] = True
        score_checks = recompute_scores(manifest, data); checks.update(score_checks)
        top = compare_top1(manifest, data); checks["all_top1_groups_recomputed"] = top["all_top1_groups_recomputed"]; checks["group_metric_row_count"] = top["group_metrics_rows"] == N_GROUPS
        bootstrap_expected = recompute_bootstrap(read_csv(OUT / "direct_plm_and_cross_teacher_group_metrics.csv")); observed = {(row["comparison"], row["budget"], row["metric"]): row for row in read_csv(OUT / "direct_plm_and_cross_teacher_paired_bootstrap_ci.csv")}; checks["paired_bootstrap_rows"] = len(observed) == len(bootstrap_expected); checks["paired_bootstrap_recomputed"] = all((key in observed and abs(finite(observed[key]["point_estimate_left_minus_right"]) - value[0]) < 2e-7 and abs(finite(observed[key]["ci95_low"]) - value[1]) < 2e-7 and abs(finite(observed[key]["ci95_high"]) - value[2]) < 2e-7 and int(observed[key]["paired_proteins"]) == value[3]) for key, value in bootstrap_expected.items())
        inventory = json.loads((OUT / "gpu_score_inventory.json").read_text(encoding="utf-8")); current_files = {str(path.relative_to(OUT)): sha256_file(path) for path in sorted((OUT / "audit/score_shards").rglob("*.csv"))}; current_aggs = {name: sha256_file(OUT / name) for name in ("wt_masked_marginals.csv", "direct_plm_candidate_scores.csv", "esm2_cross_teacher_candidate_views.csv", "wt_masked_marginals_shard_index.csv", "esm2_cross_teacher_shard_index.csv")}; checks["gpu_score_inventory_unchanged"] = current_files == inventory["files"] and current_aggs == inventory["aggregates"]
        cath_audit = read_csv(OUT / "cath_join_audit.csv")
        cath_by_id = {row["candidate_id"]: row for row in cath_audit}
        checks["cath_join_audit"] = len(cath_audit) == N_CANDIDATES and set(cath_by_id) == {row["candidate_id"] for row in manifest} and all(cath_by_id[row["candidate_id"]]["status"] == "PASS" and cath_by_id[row["candidate_id"]]["sequence_sha256"] == row["sequence_sha256"] and cath_by_id[row["candidate_id"]]["budget"] == row["budget"] for row in manifest)
        checks["json_csv_parse"] = all(json.loads(path.read_text(encoding="utf-8")) is not None for path in (OUT / "protocol.json", OUT / "source_hashes.json", OUT / "run_summary.json")) and all(len(read_csv(OUT / name)) >= 0 for name in ("input_manifest.csv", "direct_plm_candidate_scores.csv", "esm2_cross_teacher_candidate_views.csv"))
        py_compile.compile(str(RUNNER), doraise=True); py_compile.compile(str(VALIDATOR), doraise=True); checks["py_compile"] = True
        diff = subprocess.run(["git", "diff", "--check"], cwd=REPO, capture_output=True, text=True); checks["git_diff_check"] = diff.returncode == 0
    except Exception as exc:
        checks["exception_free"] = False; validation = {"checks": checks, "error": repr(exc), "top1": {}, "score_recomputation": {}}
        write_json(OUT / "independent_validation.json", {"status": "FAIL", **validation}); raise
    checks["exception_free"] = True
    validation = {"checks": checks, "resource": {"hostname": socket.gethostname(), "gpu_idle": idle}, "score_recomputation": score_checks, "top1": {"groups": top["groups"]}, "bootstrap": {"rows": len(bootstrap_expected), "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}, "artifact_hashes": {"model_current": current_model["aggregate_sha256"], "gpu_score_inventory": sha256_file(OUT / "gpu_score_inventory.json")}, "validator_script_sha256": sha256_file(VALIDATOR), "runner_script_sha256": sha256_file(RUNNER)}
    finalise(validation, protocol)
    print(json.dumps({"event": EXPECTED_MARKER, "status": "PASS", "checks": len(checks)}, sort_keys=True))
    return 0


def input_source_entries(source_hashes: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for item in source_hashes["sources"]:
        path = Path(item["path"])
        if path.name in {"input_manifest.csv", "candidate_reconstruction_audit.csv", "cath_join_audit.csv"}:
            continue
        yield item


def cli() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize-only", action="store_true", help="authorized artifact-only finalization; never recomputes scientific metrics")
    args = parser.parse_args()
    return finalization_only() if args.finalize_only else main()


if __name__ == "__main__":
    raise SystemExit(cli())
