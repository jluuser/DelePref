#!/usr/bin/env python3
"""Hard-gated bounded pilot for strict-budget deletion student delepref.

This entry point intentionally has no full-training mode.  A real pilot starts
only after a DelePref validation report says PASS and the pair index,
same-count schema, local ESMC snapshot, CUDA runtime, and isolated output paths
all pass preflight.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "deletion_preference_student_pilot.yaml")
ALLOWED_OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "delepref_esmc600m_same_count", "pilot")
ALLOWED_CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints", "delepref_esmc600m_same_count", "pilot")


def as_dict(value: object) -> Dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def load_config(path: str) -> Dict[str, object]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for delepref") from exc
    with open(path, "r") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError("delepref config must be a mapping")
    return dict(loaded)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict-budget delepref bounded pilot only")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--pilot", action="store_true", help="Run the bounded real CUDA pilot after all hard gates pass.")
    parser.add_argument("--synthetic_smoke", action="store_true", help="Tiny CPU-only strict-pair smoke; never a pilot.")
    parser.add_argument("--pair_index", default=None)
    parser.add_argument("--validation_report", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def _absolute(path: object) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    return os.path.abspath(text if os.path.isabs(text) else os.path.join(REPO_ROOT, text))


def _contains_delepref_same_count_marker(path: str) -> bool:
    normalized = path.lower().replace("-", "_")
    return "pair_same_count" in normalized or "pair_samecount" in normalized


def _status_values(value: object) -> List[str]:
    values: List[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in {"status", "validation_status", "final_status"}:
                values.append(str(child))
            values.extend(_status_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(_status_values(child))
    return values


def validation_report_passes(path: str) -> Tuple[bool, Dict[str, object]]:
    if not path or not os.path.isfile(path):
        return False, {"path": path, "reason": "missing_validation_report", "status_values": []}
    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        return False, {"path": path, "reason": "invalid_validation_json", "error": str(exc), "status_values": []}
    statuses = _status_values(payload)
    passing = [status for status in statuses if status.strip().upper() == "PASS" or status.strip().upper().startswith("PASS_")]
    explicit_failure = [status for status in statuses if status.strip().upper().startswith(("FAIL", "BLOCK", "ERROR"))]

    def safe_int(key: str, default: int) -> int:
        try:
            return int(payload.get(key, default)) if isinstance(payload, Mapping) else default
        except (TypeError, ValueError):
            return default

    def safe_float(key: str, default: float) -> float:
        try:
            return float(payload.get(key, default)) if isinstance(payload, Mapping) else default
        except (TypeError, ValueError):
            return default

    required_checks = {
        "top_level_status": payload.get("status") == "PASS" if isinstance(payload, Mapping) else False,
        "schema_version": payload.get("schema_version") == "delepref/pair_same_count_validation_v1"
        if isinstance(payload, Mapping)
        else False,
        "full_scope": payload.get("scope") == "full" if isinstance(payload, Mapping) else False,
        "all_shards_validated": (
            safe_int("shards_validated", -1) == safe_int("shards_available", -2)
        ),
        "same_deleted_length_rate": safe_float("same_deleted_length_rate", -1.0) == 1.0,
        "unequal_count_pairs_zero": safe_int("unequal_count_pairs", -1) == 0,
        "same_protein_rate": safe_float("same_protein_rate", -1.0) == 1.0,
        "source_candidate_linkage": (
            float(as_dict(payload.get("source_candidate_linkage")).get("rate", -1.0)) == 1.0
            if isinstance(payload, Mapping)
            else False
        ),
        "source_impact_linkage": (
            float(as_dict(payload.get("source_impact_linkage")).get("rate", -1.0)) == 1.0
            if isinstance(payload, Mapping)
            else False
        ),
        "coordinate_consistency": payload.get("coordinate_deleted_length_fraction_consistency") == "PASS"
        if isinstance(payload, Mapping)
        else False,
        "zero_split_leakage": safe_int("protein_split_leakage_count", -1) == 0,
        "zero_failures": safe_int("failure_count", -1) == 0,
        "positive_pairs": safe_int("pairs", 0) > 0,
        "positive_shards": safe_int("shards_validated", 0) > 0,
    }
    ok = bool(passing) and not explicit_failure and all(required_checks.values())
    return ok, {
        "path": path,
        "reason": "pass" if ok else "validation_status_not_pass",
        "status_values": statuses,
        "passing_status_values": passing,
        "failure_status_values": explicit_failure,
        "required_checks": required_checks,
        "pairs": safe_int("pairs", 0),
        "shards_available": safe_int("shards_available", 0),
        "shards_validated": safe_int("shards_validated", 0),
    }


def _resolve_manifest_path(raw_path: str, index_path: str) -> str:
    candidates = []
    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                os.path.join(REPO_ROOT, raw_path),
                os.path.join(os.path.dirname(index_path), raw_path),
                os.path.join(os.path.dirname(os.path.dirname(index_path)), raw_path),
            ]
        )
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0]) if candidates else ""


def read_pair_index(index_path: str) -> Tuple[List[str], Dict[str, object]]:
    if not index_path or not os.path.isfile(index_path):
        return [], {"path": index_path, "reason": "missing_pair_index", "rows": 0, "total_pairs": 0, "errors": []}
    pair_paths: List[str] = []
    errors: List[str] = []
    total_pairs = 0
    entries: List[Dict[str, object]] = []
    seen_shard_ids = set()
    seen_pair_paths = set()
    with open(index_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not fields.intersection({"output_csv", "pair_path", "path", "csv_path", "file"}):
            errors.append("index_missing_pair_path_column")
        for row_number, row in enumerate(reader, start=2):
            raw_path = str(
                row.get("output_csv") or row.get("pair_path") or row.get("path") or row.get("csv_path") or row.get("file") or ""
            ).strip()
            path = _resolve_manifest_path(raw_path, index_path)
            if not raw_path or not os.path.isfile(path):
                errors.append("row_{}_missing_pair_shard".format(row_number))
            else:
                pair_paths.append(path)
            source_candidate_raw = str(row.get("source_candidate_csv") or "").strip()
            source_candidate_path = _resolve_manifest_path(source_candidate_raw, index_path)
            if not source_candidate_raw or not os.path.isfile(source_candidate_path):
                errors.append("row_{}_missing_source_candidate_shard".format(row_number))
            status = str(row.get("status") or "").strip().lower()
            if not status:
                errors.append("row_{}_missing_status".format(row_number))
            if status and status not in {
                "written",
                "resumed_validated",
                "ok",
                "pass",
                "passed",
                "complete",
                "completed",
            }:
                errors.append("row_{}_bad_status_{}".format(row_number, status))
            num_rows = row.get("num_pairs") or row.get("num_rows")
            expected_rows = row.get("expected_rows")
            if num_rows not in (None, ""):
                try:
                    row_count = int(num_rows)
                    if row_count <= 0:
                        errors.append("row_{}_nonpositive_count".format(row_number))
                    total_pairs += row_count
                    if expected_rows not in (None, "") and row_count != int(expected_rows):
                        errors.append("row_{}_count_mismatch".format(row_number))
                except ValueError:
                    errors.append("row_{}_invalid_count".format(row_number))
            else:
                errors.append("row_{}_missing_count".format(row_number))
            shard_id = str(row.get("shard_id") or row.get("shard_index") or "").strip()
            if not shard_id:
                errors.append("row_{}_missing_shard_id".format(row_number))
            elif shard_id in seen_shard_ids:
                errors.append("row_{}_duplicate_shard_id".format(row_number))
            else:
                seen_shard_ids.add(shard_id)
            if path in seen_pair_paths:
                errors.append("row_{}_duplicate_pair_path".format(row_number))
            else:
                seen_pair_paths.add(path)
            if raw_path:
                entries.append(
                    {
                        "pair_path": path,
                        "source_candidate_path": source_candidate_path,
                        "num_pairs": int(num_rows) if num_rows not in (None, "") and str(num_rows).isdigit() else 0,
                        "status": status,
                    }
                )
    pair_paths = sorted(set(pair_paths))
    if not pair_paths:
        errors.append("no_pair_shards")
    return pair_paths, {
        "path": index_path,
        "reason": "ok" if not errors else "invalid_pair_index",
        "rows": len(pair_paths),
        "total_pairs": total_pairs,
        "errors": errors,
        "entries": entries,
    }


def _pair_header_audit(path: str) -> Dict[str, object]:
    required = {
        "protein_id",
        "pair_id",
        "preference",
        "preference_confidence",
        "preference_margin",
        "comparison_basis_json",
        "construction_policy",
        "deleted_length",
        "scorer_training_weight",
        "policy_positive_eligible",
        "pair_quality_bucket",
        "perturbation_id_a",
        "plan_id_a",
        "perturbation_id_b",
        "plan_id_b",
        "deleted_intervals_a",
        "deleted_intervals_b",
        "deletion_fraction_a",
        "deletion_fraction_b",
    }
    sample_row: Mapping[str, str] = {}
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        sample_row = next(reader, {})
    missing = sorted(required.difference(header))
    sample_errors = []
    basis = {}
    try:
        basis = json.loads(sample_row.get("comparison_basis_json") or "{}")
    except (TypeError, ValueError):
        sample_errors.append("sample_comparison_basis_json_invalid")
    if sample_row.get("scorer_training_weight") in (None, ""):
        sample_errors.append("sample_missing_scorer_training_weight")
    if sample_row.get("policy_positive_eligible") in (None, ""):
        sample_errors.append("sample_missing_explicit_policy_positive_eligible")
    if sample_row.get("construction_policy") != "strict_same_deleted_length_pair":
        sample_errors.append("sample_construction_policy_not_strict_same_count")
    if sample_row.get("deletion_fraction_a") != sample_row.get("deletion_fraction_b"):
        try:
            if abs(float(sample_row.get("deletion_fraction_a") or "nan") - float(sample_row.get("deletion_fraction_b") or "nan")) > 1.0e-10:
                sample_errors.append("sample_side_beta_mismatch")
        except ValueError:
            sample_errors.append("sample_side_beta_invalid")
    return {
        "path": path,
        "columns": header,
        "missing_required_columns": missing,
        "sample_errors": sample_errors,
        "ok": not missing and not sample_errors,
    }


def preflight_delepref(
    config: Mapping[str, object],
    *,
    device_request: str = "auto",
    pair_index_override: Optional[str] = None,
    validation_report_override: Optional[str] = None,
) -> Dict[str, object]:
    import torch

    def import_status(module_name: str) -> str:
        try:
            __import__(module_name)
        except Exception as exc:
            return "error: {}".format(exc)
        return "ok"

    model_config = as_dict(config.get("model"))
    policy_config = as_dict(config.get("policy"))
    data_config = as_dict(config.get("data"))
    training_config = as_dict(config.get("training"))
    pair_index = _absolute(pair_index_override or data_config.get("pair_index"))
    validation_report = _absolute(validation_report_override or data_config.get("validation_report"))
    validation_ok, validation_audit = validation_report_passes(validation_report)
    pair_paths, index_audit = read_pair_index(pair_index)
    header_audit = _pair_header_audit(pair_paths[0]) if pair_paths else {"ok": False, "missing_required_columns": []}
    snapshot = _absolute(model_config.get("esmc_snapshot"))
    safetensors_path = os.path.join(snapshot, "model.safetensors") if snapshot else ""
    output_dir = _absolute(training_config.get("output_dir"))
    checkpoint_dir = _absolute(training_config.get("checkpoint_dir"))
    requested = "cuda" if device_request == "auto" and torch.cuda.is_available() else str(device_request)
    max_steps = int(training_config.get("max_steps", 0))
    max_fraction = float(training_config.get("max_data_fraction", 0.0))
    blockers: List[str] = []
    dependencies = {
        "esm.models.esmc": import_status("esm.models.esmc"),
        "safetensors.torch": import_status("safetensors.torch"),
        "numpy": import_status("numpy"),
    }
    if str(data_config.get("pair_format") or "") != "pair_same_count":
        blockers.append("pair_format_is_not_pair_same_count")
    if not bool(data_config.get("require_same_count", False)):
        blockers.append("require_same_count_must_be_true")
    if int(policy_config.get("qmax", 3)) != 3:
        blockers.append("delepref_final_qmax_must_equal_3")
    if not _contains_delepref_same_count_marker(pair_index):
        blockers.append("pair_index_path_lacks_pair_same_count_marker")
    if not _contains_delepref_same_count_marker(validation_report):
        blockers.append("validation_path_lacks_pair_same_count_marker")
    if not validation_ok:
        blockers.append("pair_validation_not_pass")
    if index_audit.get("errors"):
        blockers.append("invalid_pair_index")
    if validation_ok and (
        int(validation_audit.get("pairs", -1)) != int(index_audit.get("total_pairs", -2))
        or int(validation_audit.get("shards_validated", -1)) != int(index_audit.get("rows", -2))
    ):
        blockers.append("validation_report_does_not_match_pair_index")
    if pair_paths and any(not _contains_delepref_same_count_marker(path) for path in pair_paths):
        blockers.append("pair_index_references_non_delepref_or_non_same_count_shard")
    if not bool(header_audit.get("ok")):
        blockers.append("pair_schema_missing_required_columns")
    if requested != "cuda" or not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        blockers.append("cuda_not_visible_in_current_runtime")
    if not os.path.isfile(safetensors_path):
        blockers.append("missing_esmc600m_safetensors")
    if str(model_config.get("encoder_backend") or "") != "esmc_official_safetensors":
        blockers.append("delepref_requires_esmc_official_safetensors_backend")
    if not bool(model_config.get("esmc_requires_cuda", False)):
        blockers.append("delepref_esmc_requires_cuda_must_be_true")
    if dependencies["esm.models.esmc"] != "ok":
        blockers.append("missing_esm_package")
    if dependencies["safetensors.torch"] != "ok":
        blockers.append("missing_safetensors_package")
    if dependencies["numpy"] != "ok":
        blockers.append("missing_numpy_for_strict_dp")
    if not bool(model_config.get("freeze_esm", False)):
        blockers.append("encoder_must_be_frozen")
    if bool(model_config.get("use_lora", False)):
        blockers.append("lora_forbidden_for_delepref")
    if bool(model_config.get("save_encoder_weights", True)):
        blockers.append("encoder_weights_must_not_be_saved")
    if output_dir != ALLOWED_OUTPUT_DIR:
        blockers.append("output_dir_outside_owned_delepref_pilot_path")
    if checkpoint_dir != ALLOWED_CHECKPOINT_DIR:
        blockers.append("checkpoint_dir_outside_owned_delepref_pilot_path")
    if max_steps < 2000 or max_steps > 5000:
        blockers.append("pilot_max_steps_must_be_between_2000_and_5000")
    if not (0.0 < max_fraction <= 0.05):
        blockers.append("pilot_data_fraction_must_be_in_0_to_0.05")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "pair_validation": validation_audit,
        "pair_index": index_audit,
        "pair_schema": header_audit,
        "pair_paths": pair_paths,
        "pair_format": data_config.get("pair_format"),
        "require_same_count": bool(data_config.get("require_same_count", False)),
        "qmax": int(policy_config.get("qmax", 3)),
        "requested_device": requested,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
        "esmc_snapshot": snapshot,
        "safetensors_path": safetensors_path,
        "safetensors_exists": os.path.isfile(safetensors_path),
        "dependencies": dependencies,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "max_steps": max_steps,
        "max_data_fraction": max_fraction,
        "preflight_only_no_model_load": True,
        "preflight_only_no_training": True,
        "full_training_supported": False,
    }


def load_bounded_pairs(
    config: Mapping[str, object],
    index_entries: Sequence[Mapping[str, object]],
    *,
    max_pairs: int,
) -> Tuple[List[object], Dict[str, object]]:
    from delepref.preference_core import (
        DeletionCandidate,
        PreferencePair,
        deleted_length,
        parse_deleted_intervals,
    )
    from delepref.student_model import validate_strict_preference_pair

    pairs: List[object] = []
    shard_stats: List[Mapping[str, object]] = []
    for entry in index_entries:
        remaining = int(max_pairs) - len(pairs)
        if remaining <= 0:
            break
        pair_path = str(entry.get("pair_path") or "")
        source_candidate_path = str(entry.get("source_candidate_path") or "")
        selected_rows: List[Dict[str, str]] = []
        needed = set()
        with open(pair_path, "r", newline="") as handle:
            for row in csv.DictReader(handle):
                if len(selected_rows) >= remaining:
                    break
                selected_rows.append(dict(row))
                for side in ("a", "b"):
                    needed.add((row["protein_id"], row["perturbation_id_" + side], row["plan_id_" + side]))
        source_lookup: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        with open(source_candidate_path, "r", newline="") as handle:
            for source in csv.DictReader(handle):
                key = (source.get("protein_id", ""), source.get("perturbation_id", ""), source.get("plan_id", ""))
                if key in needed:
                    source_lookup[key] = dict(source)
                    if len(source_lookup) == len(needed):
                        break
        if len(source_lookup) != len(needed):
            missing = sorted(needed.difference(source_lookup))
            raise RuntimeError("pair source candidate linkage missing {} keys, first={}".format(len(missing), missing[:3]))

        def side_candidate(row: Mapping[str, str], side: str) -> DeletionCandidate:
            key = (row["protein_id"], row["perturbation_id_" + side], row["plan_id_" + side])
            source = source_lookup[key]
            sequence = str(source.get("wildtype_sequence") or "")
            intervals = parse_deleted_intervals(row["deleted_intervals_" + side], len(sequence))
            actual_deleted = deleted_length(intervals)
            declared_deleted = int(row["deleted_length"])
            if actual_deleted != declared_deleted:
                raise ValueError("pair side {} interval count mismatch".format(side))
            fraction = float(row["deletion_fraction_" + side])
            if abs(fraction - actual_deleted / float(len(sequence))) > 1.0e-8:
                raise ValueError("pair side {} deletion fraction mismatch".format(side))
            return DeletionCandidate(
                protein_id=row["protein_id"],
                perturbation_id=row["perturbation_id_" + side],
                plan_id=row["plan_id_" + side],
                wildtype_sequence=sequence,
                deleted_intervals=intervals,
                deleted_length=actual_deleted,
                deletion_fraction=fraction,
                target_delete_fraction=fraction,
                budget_bin=row.get("pair_budget_key", ""),
                generation_policy=row.get("generation_policy_" + side, ""),
            )

        loaded = []
        for row in selected_rows:
            candidate_a = side_candidate(row, "a")
            candidate_b = side_candidate(row, "b")
            strict_pair = PreferencePair(
                pair_id=row["pair_id"],
                protein_id=row["protein_id"],
                budget_bin=row.get("pair_budget_key", ""),
                preference=row["preference"],
                preference_confidence=float(row["preference_confidence"]),
                loss_weight=float(row["scorer_training_weight"]),
                preference_margin=float(row["preference_margin"]),
                is_valid_pair=True,
                construction_policy=row["construction_policy"],
                quality_status=row["pair_quality_bucket"],
                quality_reason="pair_teacher_derived_preference",
                a=candidate_a,
                b=candidate_b,
                pair_quality_bucket=row["pair_quality_bucket"],
                policy_positive_eligible=str(row["policy_positive_eligible"]).lower() == "true",
                scorer_training_weight=float(row["scorer_training_weight"]),
                metadata={
                    "source_split": row.get("source_split", ""),
                    "selection_category": row.get("selection_category", ""),
                    "view_conflicting": str(row.get("view_conflicting", "")).lower() == "true",
                    "comparison_basis_json": row.get("comparison_basis_json", ""),
                },
            )
            validate_strict_preference_pair(strict_pair)
            loaded.append(strict_pair)
        pairs.extend(loaded)
        shard_stats.append(
            {
                "pair_path": pair_path,
                "source_candidate_path": source_candidate_path,
                "pairs_loaded": len(loaded),
                "source_candidates_linked": len(source_lookup),
            }
        )
    return pairs, {
        "pairs_loaded": len(pairs),
        "pair_paths_read": len(shard_stats),
        "strict_same_count_validated_loaded_rows": len(pairs),
        "shard_stats": shard_stats,
    }


def split_pairs_by_protein(pairs: Sequence[object], validation_fraction: float) -> Tuple[List[object], List[object]]:
    proteins = sorted({str(pair.protein_id) for pair in pairs})
    if len(proteins) < 2:
        raise RuntimeError("protein-independent pilot split requires at least two proteins")
    validation = set()
    threshold = max(1, min(9999, int(round(float(validation_fraction) * 10000.0))))
    for protein_id in proteins:
        bucket = int(hashlib.sha256(protein_id.encode("utf-8")).hexdigest()[:8], 16) % 10000
        if bucket < threshold:
            validation.add(protein_id)
    if not validation:
        validation.add(proteins[-1])
    if len(validation) == len(proteins):
        validation.remove(proteins[0])
    train = [pair for pair in pairs if pair.protein_id not in validation]
    val = [pair for pair in pairs if pair.protein_id in validation]
    if not train or not val:
        raise RuntimeError("protein-independent split produced an empty partition")
    overlap = {pair.protein_id for pair in train}.intersection(pair.protein_id for pair in val)
    if overlap:
        raise RuntimeError("train/validation protein leakage: {}".format(sorted(overlap)[:10]))
    return train, val


def build_delepref(config: Mapping[str, object], device: object):
    from delepref.preference_core import load_esmc_official_safetensors_model
    from delepref.student_model import DeletionPreferenceStudent, FrozenWTEncoder

    model_config = as_dict(config.get("model"))
    policy_config = as_dict(config.get("policy"))
    qmax = int(policy_config.get("qmax", 3))
    if qmax != 3:
        raise ValueError("delepref final model requires Qmax=3")
    esmc = load_esmc_official_safetensors_model(
        _absolute(model_config.get("esmc_snapshot")),
        dtype=str(model_config.get("esmc_dtype") or "bfloat16"),
        device=device,
        max_length=int(model_config.get("max_length", 1022)),
        require_cuda=bool(model_config.get("esmc_requires_cuda", True)),
    )
    encoder = FrozenWTEncoder(
        esmc,
        esmc.tokenizer,
        hidden_size=int(esmc.config.hidden_size),
        max_length=int(model_config.get("max_length", 1022)),
    )
    model = DeletionPreferenceStudent(
        encoder,
        hidden_size=int(esmc.config.hidden_size),
        scorer_hidden_dim=int(model_config.get("scorer_hidden_dim", 256)),
        span_feature_dim=int(model_config.get("span_feature_dim", 128)),
        budget_hidden_dim=int(policy_config.get("budget_hidden_dim", 64)),
        policy_hidden_dim=int(policy_config.get("policy_hidden_dim", 256)),
        policy_span_hidden_dim=int(policy_config.get("span_hidden_dim", 256)),
        qmax=qmax,
        min_span_len=int(policy_config.get("min_span_len", 1)),
        max_span_len_fraction=float(policy_config.get("max_span_len_fraction", 0.5)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    return model.to(device)


def proposal_loss_kwargs(config: Mapping[str, object]) -> Dict[str, float]:
    policy = as_dict(config.get("policy"))
    names = (
        "rank_weight",
        "imitation_weight",
        "mask_weight",
        "activity_weight",
        "center_weight",
        "length_weight",
        "budget_aux_weight",
        "structure_aux_weight",
        "tie_weight_scale",
        "margin",
    )
    return {name: float(policy.get(name, 0.0)) for name in names}


def _batches(items: Sequence[object], batch_size: int) -> Iterable[List[object]]:
    for start in range(0, len(items), int(batch_size)):
        batch = list(items[start : start + int(batch_size)])
        if len(batch) == int(batch_size):
            yield batch


def _scalar(value: object) -> float:
    return float(value.detach().float().cpu()) if hasattr(value, "detach") else float(value)


def train_bounded_pilot(
    model,
    train_pairs: Sequence[object],
    *,
    config: Mapping[str, object],
    effective_steps: int,
    batch_size: int,
    output_dir: str,
    checkpoint_dir: str,
) -> Tuple[object, List[Mapping[str, object]], int, float]:
    import torch

    from delepref.student_model import save_student_checkpoint, trainable_gradient_norm

    training = as_dict(config.get("training"))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training.get("learning_rate", 5.0e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(training.get("seed", 13)))
    order = torch.randperm(len(train_pairs), generator=generator).tolist()
    ordered = [train_pairs[index] for index in order]
    log_path = os.path.join(output_dir, "train_log.jsonl")
    with open(log_path, "w"):
        pass
    metrics: List[Mapping[str, object]] = []
    start_time = time.time()
    model.train()
    for step, batch in enumerate(_batches(ordered, batch_size), start=1):
        if step > int(effective_steps):
            break
        optimizer.zero_grad(set_to_none=True)
        info = model.joint_objective(
            batch,
            scorer_tie_weight_scale=float(training.get("scorer_tie_weight_scale", 0.1)),
            proposal_kwargs=proposal_loss_kwargs(config),
            scorer_loss_weight=float(training.get("scorer_loss_weight", 1.0)),
            proposal_loss_weight=float(training.get("proposal_loss_weight", 1.0)),
        )
        loss = info["loss"]
        if not bool(torch.isfinite(loss.detach()).item()):
            raise RuntimeError("non-finite joint loss at step {}".format(step))
        loss.backward()
        scorer_grad = trainable_gradient_norm(model.scorer.parameters())
        proposal_grad = trainable_gradient_norm(model.policy.parameters())
        if not math.isfinite(scorer_grad) or not math.isfinite(proposal_grad):
            raise RuntimeError("non-finite gradient norm at step {}".format(step))
        clip = float(training.get("gradient_clip_norm", 0.0))
        if clip > 0.0:
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=clip)
        optimizer.step()
        scorer = info["scorer"]
        proposal = info["proposal"]
        record = {
            "step": step,
            "num_pairs": len(batch),
            "num_unique_wildtypes": info["encoded_wildtypes"].unique_count,
            "loss": _scalar(loss),
            "scorer_loss": _scalar(scorer["loss"]),
            "proposal_loss": _scalar(proposal["loss"]),
            "rank_loss": _scalar(proposal["rank_loss"]),
            "tie_loss": _scalar(proposal["tie_loss"]),
            "imitation_loss": _scalar(proposal["imitation_loss"]),
            "mask_bce": _scalar(proposal["mask_bce"]),
            "activity_bce": _scalar(proposal["activity_bce"]),
            "center_loss": _scalar(proposal["center_loss"]),
            "length_loss": _scalar(proposal["length_loss"]),
            "budget_aux_loss": _scalar(proposal["budget_aux_loss"]),
            "structure_aux_loss": _scalar(proposal["structure_aux_loss"]),
            "num_scorer_rank_used": int(scorer["num_rank_used"]),
            "num_scorer_tie_used": int(scorer["num_tie_used"]),
            "num_policy_rank_used": int(proposal["num_rank_used"]),
            "num_policy_tie_used": int(proposal["num_tie_used"]),
            "num_policy_imitation_used": int(proposal["num_imitation_used"]),
            "num_policy_imitation_references": int(proposal["num_imitation_references"]),
            "num_policy_imitation_duplicates_removed": int(proposal["num_imitation_duplicates_removed"]),
            "num_span_count_supervision_used": int(proposal["num_span_count_supervision_used"]),
            "num_budget_conditions": int(proposal["num_budget_conditions"]),
            "num_budget_condition_references": int(proposal["num_budget_condition_references"]),
            "num_budget_duplicates_removed": int(proposal["num_budget_duplicates_removed"]),
            "num_candidate_energies": int(proposal["num_candidate_energies"]),
            "num_negative_candidate_references": int(proposal["num_negative_candidate_references"]),
            "num_negative_supervision_used": int(proposal["num_negative_supervision_used"]),
            "num_aux_candidates": int(proposal["num_aux_candidates"]),
            "num_matched_spans": int(proposal["num_matched_spans"]),
            "num_mask_residues": int(proposal["num_mask_residues"]),
            "scorer_gradient_norm": scorer_grad,
            "proposal_gradient_norm": proposal_grad,
            "elapsed_seconds": time.time() - start_time,
        }
        metrics.append(record)
        with open(log_path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        log_every = int(training.get("log_every", 10))
        if step == 1 or (log_every > 0 and step % log_every == 0):
            print(json.dumps({"event": "delepref_pilot_step", **record}, sort_keys=True), flush=True)
        save_every = int(training.get("save_every", 500))
        if save_every > 0 and step % save_every == 0:
            save_student_checkpoint(
                os.path.join(checkpoint_dir, "checkpoint_step_{:08d}.pt".format(step)),
                model=model,
                optimizer=optimizer,
                step=step,
                config=config,
                metadata={"bounded_pilot": True},
            )
    if len(metrics) != int(effective_steps):
        raise RuntimeError("bounded pilot ended after {} of {} steps".format(len(metrics), effective_steps))
    return optimizer, metrics, len(metrics), time.time() - start_time


def evaluate_pair_fidelity(model, pairs: Sequence[object], max_pairs: int) -> Dict[str, object]:
    import torch

    from delepref.student_model import proposal_candidate_losses

    scorer_correct = 0.0
    proposal_correct = 0.0
    weight_total = 0.0
    used = 0
    tie_abs_scorer: List[float] = []
    tie_abs_proposal: List[float] = []
    selected_pairs = list(pairs)[: int(max_pairs)]
    model.eval()
    with torch.no_grad():
        candidates = []
        for strict_pair in selected_pairs:
            candidates.extend([strict_pair.a, strict_pair.b])
        encoded = model.encode_unique_wildtypes(candidates)
        all_scores = model.score_plans_from_embeddings(candidates, encoded)["preference_score"]
        policy_cache = {}
        for pair_index, strict_pair in enumerate(selected_pairs):
            scores = all_scores[2 * pair_index : 2 * pair_index + 2]
            policy_key = (
                strict_pair.protein_id,
                hashlib.sha256(strict_pair.a.wildtype_sequence.encode("utf-8")).hexdigest(),
                float(strict_pair.a.target_delete_fraction),
            )
            policy_output = policy_cache.get(policy_key)
            if policy_output is None:
                embedding = encoded.for_candidate(strict_pair.a)
                policy_output = model.propose_from_embeddings(
                    [embedding], [strict_pair.a.target_delete_fraction]
                )[0]
                policy_cache[policy_key] = policy_output
            energy_a = proposal_candidate_losses(model.policy, strict_pair.a, policy_output)["normalized_energy"]
            energy_b = proposal_candidate_losses(model.policy, strict_pair.b, policy_output)["normalized_energy"]
            preference = str(strict_pair.preference).lower()
            weight = float(strict_pair.loss_weight)
            if preference == "a_better":
                scorer_correct += weight * float(scores[0] > scores[1])
                proposal_correct += weight * float(energy_a > energy_b)
            elif preference == "b_better":
                scorer_correct += weight * float(scores[1] > scores[0])
                proposal_correct += weight * float(energy_b > energy_a)
            elif preference == "tie":
                tie_abs_scorer.append(abs(float((scores[0] - scores[1]).cpu())))
                tie_abs_proposal.append(abs(float((energy_a - energy_b).cpu())))
                continue
            else:
                continue
            weight_total += weight
            used += 1
    return {
        "non_tie_pairs_used": used,
        "strict_same_count_scorer_weighted_fidelity": scorer_correct / max(1.0e-12, weight_total),
        "strict_same_count_proposal_weighted_fidelity": proposal_correct / max(1.0e-12, weight_total),
        "tie_pairs_used": len(tie_abs_scorer),
        "mean_abs_scorer_tie_delta": sum(tie_abs_scorer) / max(1, len(tie_abs_scorer)),
        "mean_abs_proposal_tie_delta": sum(tie_abs_proposal) / max(1, len(tie_abs_proposal)),
    }


def _position_diversity(plans: Sequence[object], length: int) -> float:
    masks = []
    for plan in plans:
        selected = set()
        for start, end in plan.deleted_intervals:
            selected.update(range(start, end + 1))
        masks.append(selected)
    distances = []
    for index in range(len(masks)):
        for other in range(index + 1, len(masks)):
            union = masks[index].union(masks[other])
            distances.append(1.0 - len(masks[index].intersection(masks[other])) / float(max(1, len(union))))
    return sum(distances) / max(1, len(distances))


def evaluate_generation(model, pairs: Sequence[object], config: Mapping[str, object]) -> Tuple[Dict[str, object], List[Mapping[str, object]]]:
    import torch

    from delepref.student_model import (
        generate_strict_budget_plans,
        proposal_from_decoded_plan,
        validate_strict_plan,
    )

    policy_config = as_dict(config.get("policy"))
    decoder_config = as_dict(config.get("decoder"))
    evaluation = as_dict(config.get("evaluation"))
    qmax = int(policy_config.get("qmax", 3))
    if qmax != 3:
        raise ValueError("delepref final evaluation requires Qmax=3")
    min_span = int(policy_config.get("min_span_len", 1))
    max_fraction = float(policy_config.get("max_span_len_fraction", 0.5))
    fixed_k = int(evaluation.get("fixed_k", decoder_config.get("num_candidates", 8)))
    max_groups = int(evaluation.get("max_generation_groups", 64))
    seed = int(decoder_config.get("seed", 13))
    perturb = float(decoder_config.get("perturbation_scale", 1.0))
    grouped: Dict[Tuple[str, str, float], List[object]] = defaultdict(list)
    for strict_pair in pairs:
        key = (
            strict_pair.protein_id,
            hashlib.sha256(strict_pair.a.wildtype_sequence.encode("utf-8")).hexdigest(),
            float(strict_pair.a.target_delete_fraction),
        )
        grouped[key].append(strict_pair)
    policy_unique_ratios = []
    policy_span_count_diversity = []
    policy_position_diversity = []
    returned_plans = 0
    legal_plans = 0
    fixed_comparison_scores: Dict[str, List[float]] = defaultdict(list)
    fixed_groups = 0
    failures: List[Mapping[str, object]] = []

    def cold_start_unique_count(group_item: Tuple[Tuple[str, str, float], List[object]]) -> int:
        unique = set()
        for strict_pair in group_item[1]:
            unique.add(tuple(strict_pair.a.deleted_intervals))
            unique.add(tuple(strict_pair.b.deleted_intervals))
        return len(unique)

    ordered_groups = sorted(
        grouped.items(),
        key=lambda item: (-cold_start_unique_count(item), item[0]),
    )
    model.eval()
    with torch.no_grad():
        for group_index, ((_protein, _hash, beta), group_pairs) in enumerate(ordered_groups):
            if group_index >= max_groups:
                break
            representative = group_pairs[0].a
            sequence = representative.wildtype_sequence
            length = len(sequence)
            max_span = max(min_span, min(length - 1, int(math.floor(max_fraction * length))))
            encoded = model.encode_unique_wildtypes([representative])
            embedding = encoded.for_candidate(representative)
            output = model.propose_from_embeddings([embedding], [beta])[0]
            policy_result = generate_strict_budget_plans(
                output.residue_logits,
                beta,
                num_candidates=fixed_k,
                qmax=qmax,
                min_span_len=min_span,
                max_span_len=max_span,
                seed=seed + group_index,
                perturbation_scale=perturb,
                max_attempts=fixed_k * int(decoder_config.get("max_attempts_multiplier", 16)),
            )
            returned_plans += len(policy_result.plans)
            for plan in policy_result.plans:
                try:
                    validate_strict_plan(
                        length,
                        beta,
                        plan.deleted_intervals,
                        qmax=qmax,
                        min_span_len=min_span,
                        max_span_len=max_span,
                    )
                    legal_plans += 1
                except ValueError as exc:
                    failures.append({"protein_id": representative.protein_id, "method": "policy", "reason": str(exc)})
            policy_unique_ratios.append(len(policy_result.plans) / float(max(1, fixed_k)))
            policy_span_count_diversity.append(
                len({len(plan.deleted_intervals) for plan in policy_result.plans}) / float(max(1, len(policy_result.plans)))
            )
            policy_position_diversity.append(_position_diversity(policy_result.plans, length))
            cpu_generator = torch.Generator(device="cpu")
            cpu_generator.manual_seed(seed + 100000 + group_index)
            random_logits = torch.randn(length, generator=cpu_generator, dtype=torch.float64)
            terminal_logits = -torch.tensor(
                [min(index, length - 1 - index) for index in range(length)], dtype=torch.float64
            )
            random_result = generate_strict_budget_plans(
                random_logits,
                beta,
                num_candidates=fixed_k,
                qmax=qmax,
                min_span_len=min_span,
                max_span_len=max_span,
                seed=seed + 200000 + group_index,
                perturbation_scale=1.0,
                max_attempts=fixed_k * 16,
            )
            terminal_result = generate_strict_budget_plans(
                terminal_logits,
                beta,
                num_candidates=fixed_k,
                qmax=qmax,
                min_span_len=min_span,
                max_span_len=max_span,
                seed=seed + 300000 + group_index,
                perturbation_scale=1.0,
                max_attempts=fixed_k * 16,
            )
            cold_intervals = []
            for strict_pair in group_pairs:
                for item in (strict_pair.a.deleted_intervals, strict_pair.b.deleted_intervals):
                    try:
                        validate_strict_plan(
                            length,
                            beta,
                            item,
                            qmax=qmax,
                            min_span_len=min_span,
                            max_span_len=max_span,
                        )
                    except ValueError:
                        continue
                    if item not in cold_intervals:
                        cold_intervals.append(item)
            if not (
                len(policy_result.plans) >= fixed_k
                and len(random_result.plans) >= fixed_k
                and len(terminal_result.plans) >= fixed_k
                and len(cold_intervals) >= fixed_k
            ):
                failures.append(
                    {
                        "protein_id": representative.protein_id,
                        "beta": beta,
                        "method": "fixed_k_comparison",
                        "reason": "insufficient_unique_candidates",
                        "counts": {
                            "policy": len(policy_result.plans),
                            "random": len(random_result.plans),
                            "terminal": len(terminal_result.plans),
                            "cold_start": len(cold_intervals),
                        },
                    }
                )
                continue
            method_candidates: Dict[str, List[object]] = {}
            for method, decoded_plans, generation_reason in (
                ("policy_scorer", policy_result.plans[:fixed_k], policy_result.reason),
                ("random_scorer", random_result.plans[:fixed_k], random_result.reason),
                ("terminal_scorer", terminal_result.plans[:fixed_k], terminal_result.reason),
            ):
                method_candidates[method] = [
                    proposal_from_decoded_plan(
                        protein_id=representative.protein_id,
                        wildtype_sequence=sequence,
                        beta=beta,
                        decoded=plan,
                        rank=rank,
                        generation_reason=generation_reason,
                    )
                    for rank, plan in enumerate(decoded_plans, start=1)
                ]
            method_candidates["cold_start_scorer"] = []
            for index, intervals in enumerate(cold_intervals[:fixed_k]):
                source = None
                for strict_pair in group_pairs:
                    for candidate in (strict_pair.a, strict_pair.b):
                        if candidate.deleted_intervals == intervals:
                            source = candidate
                            break
                    if source is not None:
                        break
                method_candidates["cold_start_scorer"].append(source)
            all_candidates = []
            slices = {}
            for method, candidates in method_candidates.items():
                start = len(all_candidates)
                all_candidates.extend(candidates)
                slices[method] = slice(start, len(all_candidates))
            scores = model.score_plans_from_embeddings(all_candidates, encoded)["preference_score"]
            for method, method_slice in slices.items():
                fixed_comparison_scores[method].append(float(scores[method_slice].max().cpu()))
            fixed_groups += 1
    metrics = {
        "generation_groups_evaluated": min(len(grouped), max_groups),
        "returned_policy_plans": returned_plans,
        "legal_policy_plans": legal_plans,
        "exact_budget_span_nonoverlap_feasibility": legal_plans / float(max(1, returned_plans)),
        "mean_policy_unique_ratio_at_k": sum(policy_unique_ratios) / max(1, len(policy_unique_ratios)),
        "mean_policy_span_count_diversity": sum(policy_span_count_diversity) / max(1, len(policy_span_count_diversity)),
        "mean_policy_position_jaccard_diversity": sum(policy_position_diversity) / max(1, len(policy_position_diversity)),
        "fixed_k": fixed_k,
        "fixed_k_comparison_groups": fixed_groups,
        "group_selection": "highest_cold_start_unique_coverage_then_stable_key",
        "fixed_k_selected_scorer_score": {
            method: sum(values) / max(1, len(values)) for method, values in sorted(fixed_comparison_scores.items())
        },
        "teacher_rescoring_performed": False,
        "teacher_rescoring_scope": "offline_diagnostic_only_not_functional_proof",
    }
    return metrics, failures


def _loss_scale_summary(step_metrics: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    names = (
        "loss",
        "scorer_loss",
        "proposal_loss",
        "rank_loss",
        "tie_loss",
        "imitation_loss",
        "mask_bce",
        "activity_bce",
        "center_loss",
        "length_loss",
        "budget_aux_loss",
        "structure_aux_loss",
        "scorer_gradient_norm",
        "proposal_gradient_norm",
    )
    summary = {}
    for name in names:
        values = [float(row[name]) for row in step_metrics if name in row]
        summary[name] = {
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "mean": sum(values) / max(1, len(values)),
            "all_finite": all(math.isfinite(value) for value in values),
        }
    return summary


def run_synthetic_smoke(config: Mapping[str, object], seed: int, device_request: str = "cpu") -> Dict[str, object]:
    import torch

    from delepref.preference_core import DeletionCandidate, PreferencePair
    from delepref.student_model import DeletionPreferenceStudent, trainable_gradient_norm

    torch.manual_seed(seed)
    use_real_cuda = str(device_request) == "cuda"
    if use_real_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA synthetic ESMC smoke requested but CUDA is unavailable")
        model = build_delepref(config, torch.device("cuda"))
        smoke_steps = 1
        encoder_backend = "frozen_esmc600m"
    else:
        model = DeletionPreferenceStudent.tiny_for_tests(qmax=3, dropout=0.0)
        smoke_steps = 3
        encoder_backend = "tiny_test_encoder"
    sequence = "ACDEFGHIKLMNPQRSTVWY"

    def item(name: str, intervals: Sequence[Tuple[int, int]]) -> DeletionCandidate:
        deleted = sum(end - start + 1 for start, end in intervals)
        return DeletionCandidate(
            protein_id="synthetic_delepref",
            perturbation_id=name,
            plan_id=name,
            wildtype_sequence=sequence,
            deleted_intervals=tuple(intervals),
            deleted_length=deleted,
            deletion_fraction=deleted / float(len(sequence)),
            target_delete_fraction=0.2,
        )

    a = item("a", [(0, 3)])
    b = item("b", [(8, 9), (14, 15)])
    strict_pair = PreferencePair(
        pair_id="synthetic_pair",
        protein_id="synthetic_delepref",
        budget_bin="strict",
        preference="a_better",
        preference_confidence=0.9,
        loss_weight=0.8,
        preference_margin=0.1,
        is_valid_pair=True,
        construction_policy="synthetic_same_count",
        quality_status="ok",
        quality_reason="synthetic",
        a=a,
        b=b,
        policy_positive_eligible=True,
    )
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1.0e-3)
    records = []
    for step in range(1, smoke_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        info = model.joint_objective([strict_pair], proposal_kwargs=proposal_loss_kwargs(config))
        info["loss"].backward()
        scorer_grad = trainable_gradient_norm(model.scorer.parameters())
        proposal_grad = trainable_gradient_norm(model.policy.parameters())
        optimizer.step()
        records.append(
            {
                "step": step,
                "loss": _scalar(info["loss"]),
                "scorer_gradient_norm": scorer_grad,
                "proposal_gradient_norm": proposal_grad,
                "unique_wildtypes_encoded": info["encoded_wildtypes"].unique_count,
            }
        )
    status = "PASS" if all(
        math.isfinite(row["loss"])
        and row["scorer_gradient_norm"] > 0.0
        and row["proposal_gradient_norm"] > 0.0
        and row["unique_wildtypes_encoded"] == 1
        for row in records
    ) else "FAIL"
    return {
        "status": status,
        "status_scope": "operational_integrity_and_metric_completeness_not_a_quality_threshold",
        "synthetic_only": True,
        "not_a_pilot": True,
        "encoder_backend": encoder_backend,
        "device": str(next(model.scorer.parameters()).device),
        "encoder_weights_trainable": any(parameter.requires_grad for parameter in model.encoder.parameters()),
        "steps": records,
    }


def _write_json(path: str, payload: object) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    import torch

    from delepref.student_model import save_student_checkpoint, seed_everything

    args = parse_args(argv)
    if sum(bool(value) for value in (args.preflight_only, args.pilot, args.synthetic_smoke)) != 1:
        raise SystemExit("Specify exactly one of --preflight_only, --pilot, or --synthetic_smoke")
    config = load_config(args.config)
    training = as_dict(config.get("training"))
    data = as_dict(config.get("data"))
    if args.max_steps is not None:
        training["max_steps"] = int(args.max_steps)
    if args.max_pairs is not None:
        training["max_pairs"] = int(args.max_pairs)
    if args.batch_size is not None:
        data["batch_size"] = int(args.batch_size)
    config["training"] = training
    config["data"] = data
    seed = int(args.seed if args.seed is not None else training.get("seed", 13))
    training["seed"] = seed
    seed_everything(seed)
    if args.synthetic_smoke:
        report = run_synthetic_smoke(config, seed, device_request=args.device)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["status"] == "PASS" else 1
    preflight = preflight_delepref(
        config,
        device_request=args.device,
        pair_index_override=args.pair_index,
        validation_report_override=args.validation_report,
    )
    output_dir = _absolute(training.get("output_dir"))
    if args.preflight_only:
        os.makedirs(output_dir, exist_ok=True)
        _write_json(os.path.join(output_dir, "preflight_delepref.json"), preflight)
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0 if preflight["status"] == "PASS" else 2
    if preflight["status"] != "PASS":
        print(json.dumps(preflight, indent=2, sort_keys=True))
        raise SystemExit("delepref pilot blocked by preflight")
    requested_steps = int(training.get("max_steps", 2000))
    if requested_steps > 5000:
        raise SystemExit("delepref pilot refuses more than 5000 steps")
    total_pairs = int(as_dict(preflight.get("pair_index")).get("total_pairs", 0))
    if total_pairs <= 0:
        raise SystemExit("pair index must report positive num_rows for the 5% gate")
    max_fraction = float(training.get("max_data_fraction", 0.05))
    max_pairs_by_fraction = max(1, int(math.floor(total_pairs * max_fraction)))
    bounded_pairs = min(int(training.get("max_pairs", max_pairs_by_fraction)), max_pairs_by_fraction)
    pairs, load_stats = load_bounded_pairs(
        config,
        as_dict(preflight.get("pair_index")).get("entries", []),
        max_pairs=bounded_pairs,
    )
    if not pairs:
        raise RuntimeError("pair loader returned zero strict pairs")
    train_pairs, val_pairs = split_pairs_by_protein(pairs, float(data.get("validation_fraction", 0.2)))
    batch_size = int(data.get("batch_size", 1))
    fraction_step_cap = max(1, max_pairs_by_fraction // batch_size)
    data_step_cap = len(train_pairs) // batch_size
    effective_steps = min(requested_steps, fraction_step_cap, data_step_cap)
    if effective_steps < 2000:
        raise RuntimeError("not enough strict pair rows for the minimum 2000-step bounded pilot")
    output_dir = ALLOWED_OUTPUT_DIR
    checkpoint_dir = ALLOWED_CHECKPOINT_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    _write_json(os.path.join(output_dir, "pilot_config.json"), config)
    _write_json(os.path.join(output_dir, "preflight_delepref.json"), preflight)
    device = torch.device("cuda")
    model = build_delepref(config, device)
    optimizer, step_metrics, completed_steps, elapsed = train_bounded_pilot(
        model,
        train_pairs,
        config=config,
        effective_steps=effective_steps,
        batch_size=batch_size,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
    )
    final_checkpoint = os.path.join(checkpoint_dir, "checkpoint.pt")
    save_student_checkpoint(
        final_checkpoint,
        model=model,
        optimizer=optimizer,
        step=completed_steps,
        config=config,
        metadata={
            "bounded_pilot": True,
            "pair_validation_report": preflight["pair_validation"],
            "train_protein_count": len({pair.protein_id for pair in train_pairs}),
            "validation_protein_count": len({pair.protein_id for pair in val_pairs}),
        },
    )
    pair_metrics = evaluate_pair_fidelity(
        model,
        val_pairs,
        max_pairs=int(as_dict(config.get("evaluation")).get("max_validation_pairs", 512)),
    )
    generation_metrics, failure_cases = evaluate_generation(model, val_pairs, config)
    loss_summary = _loss_scale_summary(step_metrics)
    scorer_nonzero = max(float(row["scorer_gradient_norm"]) for row in step_metrics) > 0.0
    proposal_nonzero = max(float(row["proposal_gradient_norm"]) for row in step_metrics) > 0.0
    all_finite = all(item["all_finite"] for item in loss_summary.values())
    loss_scales_stable = all(
        item["all_finite"] and abs(float(item["max"])) < 1.0e6 and abs(float(item["min"])) < 1.0e6
        for item in loss_summary.values()
    )
    feasibility_100 = generation_metrics["exact_budget_span_nonoverlap_feasibility"] == 1.0
    required_baselines = {"policy_scorer", "random_scorer", "terminal_scorer", "cold_start_scorer"}
    selected_scores = as_dict(generation_metrics.get("fixed_k_selected_scorer_score"))
    metric_completeness = (
        int(pair_metrics.get("non_tie_pairs_used", 0)) > 0
        and int(generation_metrics.get("generation_groups_evaluated", 0)) > 0
        and math.isfinite(float(generation_metrics.get("mean_policy_unique_ratio_at_k", float("nan"))))
        and math.isfinite(float(generation_metrics.get("mean_policy_span_count_diversity", float("nan"))))
        and math.isfinite(float(generation_metrics.get("mean_policy_position_jaccard_diversity", float("nan"))))
    )
    fixed_k_comparison_complete = (
        int(generation_metrics["fixed_k_comparison_groups"]) > 0
        and required_baselines.issubset(selected_scores)
        and all(math.isfinite(float(selected_scores[name])) for name in required_baselines)
    )
    status = (
        "PASS"
        if all_finite
        and loss_scales_stable
        and scorer_nonzero
        and proposal_nonzero
        and feasibility_100
        and metric_completeness
        and fixed_k_comparison_complete
        else "FAIL"
    )
    metrics = {
        "status": status,
        "status_scope": "operational_integrity_and_metric_completeness_not_a_quality_threshold",
        "bounded_pilot": True,
        "full_training_started": False,
        "completed_steps": completed_steps,
        "requested_steps": requested_steps,
        "effective_steps_after_5_percent_gate": effective_steps,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "no_nan_inf": all_finite,
        "no_oom": True,
        "loss_scales_stable": loss_scales_stable,
        "scorer_gradient_nonzero": scorer_nonzero,
        "proposal_gradient_nonzero": proposal_nonzero,
        "fixed_k_comparison_complete": fixed_k_comparison_complete,
        "metric_completeness": metric_completeness,
        "loss_scale_summary": loss_summary,
        "pair_metrics": pair_metrics,
        "generation_metrics": generation_metrics,
        "data": {
            "index_total_pairs": total_pairs,
            "maximum_allowed_pairs_at_fraction": max_pairs_by_fraction,
            "pairs_loaded": len(pairs),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(val_pairs),
            "train_proteins": len({pair.protein_id for pair in train_pairs}),
            "validation_proteins": len({pair.protein_id for pair in val_pairs}),
            "protein_overlap": sorted(
                {pair.protein_id for pair in train_pairs}.intersection(pair.protein_id for pair in val_pairs)
            ),
            "load_stats": load_stats,
        },
        "outputs": {
            "checkpoint": final_checkpoint,
            "config": os.path.join(output_dir, "pilot_config.json"),
            "train_log": os.path.join(output_dir, "train_log.jsonl"),
            "metrics": os.path.join(output_dir, "pilot_metrics.json"),
            "failure_cases": os.path.join(output_dir, "failure_cases.json"),
        },
        "claim_boundary": "The method ranks candidate deletion plans by teacher-derived deletion perturbation preference and optional auxiliary risk evidence.",
        "teacher_rescoring_is_functional_proof": False,
    }
    _write_json(os.path.join(output_dir, "failure_cases.json"), failure_cases)
    _write_json(os.path.join(output_dir, "pilot_metrics.json"), metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
