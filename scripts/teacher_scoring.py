#!/usr/bin/env python3
"""Score deletion counterfactuals with a frozen protein language model."""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from collections import defaultdict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (ROOT_DIR, SCRIPT_DIR, SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

COORDINATE_SYSTEM = "0_based_closed_inclusive"

TEACHER_IMPACT_REQUIRED_COLUMNS = [
    "protein_id",
    "perturbation_id",
    "plan_id",
    "teacher_model",
    "teacher_scoring_mode",
    "global_shift",
    "retained_region_shift",
    "junction_plausibility",
    "sequence_manifold_plausibility",
    "risk_overlap",
    "risk_confidence",
    "impact_confidence",
    "scoring_status",
]

TEACHER_IMPACT_OPTIONAL_COLUMNS = [
    "global_similarity",
    "retained_similarity",
    "junction_mean",
    "junction_min",
    "teacher_backend",
    "max_memory_allocated_mb",
    "view_status_json",
    "risk_summary_json",
    "metadata_json",
]

DELETION_PERTURBATION_REQUIRED_COLUMNS = [
    "protein_id",
    "perturbation_id",
    "plan_id",
    "wildtype_length",
    "deleted_intervals",
    "num_intervals",
    "deleted_length",
    "deletion_fraction",
    "budget_bin",
    "generation_policy",
    "coordinate_system",
]

TEACHER_PREFERENCE_REQUIRED_COLUMNS = [
    "protein_id",
    "budget_bin",
    "pair_id",
    "perturbation_id_a",
    "plan_id_a",
    "perturbation_id_b",
    "plan_id_b",
    "preference",
    "preference_margin",
    "preference_confidence",
    "comparison_basis_json",
    "construction_policy",
    "is_valid_pair",
]

TEACHER_PREFERENCE_OPTIONAL_COLUMNS = [
    "invalid_reason",
    "metadata_json",
]

REAL_PLM_TEACHER_MODE = "fast_real_plm"
ESM2_TRANSFORMERS_BACKEND = "esm2_transformers"
ESMC_OFFICIAL_SAFETENSORS_BACKEND = "esmc_official_safetensors"
SUPPORTED_TEACHER_BACKENDS = [ESM2_TRANSFORMERS_BACKEND, ESMC_OFFICIAL_SAFETENSORS_BACKEND]
SUPPORTED_TEACHER_DEVICES = ["auto", "cpu", "cuda"]
BUDGET_BIN_ORDER = ["b001_single", "b02", "b04", "b05", "b06", "b07", "b10", "b15", "b20", "b30"]
REAL_PLM_VIEW_NAMES = [
    "global_shift",
    "retained_region_shift",
    "junction_plausibility",
    "sequence_manifold_plausibility",
    "risk_overlap",
    "risk_confidence",
    "impact_confidence",
]


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def read_csv(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def is_empty(value):
    return value is None or str(value).strip() == ""


def safe_id(value):
    text = str(value).strip()
    out = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            out.append(char)
        else:
            out.append("_")
    safe = "".join(out).strip("_")
    return safe or "unknown"


def parse_deleted_intervals(row, source):
    value = row.get("deleted_intervals")
    if is_empty(value):
        raise ValueError("{}: deleted_intervals is required".format(source))
    try:
        data = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{}: deleted_intervals must be JSON: {}".format(source, exc))
    intervals = []
    for idx, item in enumerate(data):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("{}: interval {} must be [start, end]".format(source, idx))
        start, end = item
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("{}: interval {} start/end must be integers".format(source, idx))
        start = int(start)
        end = int(end)
        if start < 0 or end < start:
            raise ValueError("{}: invalid closed-inclusive interval {}".format(source, item))
        intervals.append((start, end))
    intervals.sort(key=lambda item: item[0])
    previous_end = None
    for start, end in intervals:
        if previous_end is not None and start <= previous_end:
            raise ValueError("{}: intervals overlap under closed-inclusive coordinates".format(source))
        previous_end = end
    return intervals


def deleted_length(intervals):
    return sum(end - start + 1 for start, end in intervals)


def retained_count(wildtype_length, intervals):
    return wildtype_length - deleted_length(intervals)


def boundary_types(wildtype_length, intervals):
    out = []
    for start, end in intervals:
        has_left = start - 1 >= 0
        has_right = end + 1 < wildtype_length
        if has_left and has_right:
            out.append("two_sided")
        elif has_left or has_right:
            out.append("one_sided_boundary")
        else:
            out.append("whole_sequence_boundary")
    return out


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def fmt_score(value):
    if value == "":
        return ""
    return "{:.6f}".format(clamp01(value))


def strict_int(value, source, field):
    if isinstance(value, bool) or value is None or str(value).strip() == "":
        raise ValueError("{}: {} must be an integer".format(source, field))
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{}: {} must be an integer".format(source, field)) from exc
    as_int = int(as_float)
    if as_float != float(as_int):
        raise ValueError("{}: {} must be an integer".format(source, field))
    return as_int


def strict_float(value, source, field):
    if value is None or str(value).strip() == "":
        raise ValueError("{}: {} must be numeric".format(source, field))
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{}: {} must be numeric".format(source, field)) from exc


def parse_deleted_intervals_strict(row, source):
    value = row.get("deleted_intervals")
    if is_empty(value):
        raise ValueError("{}: deleted_intervals is required".format(source))
    try:
        data = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{}: deleted_intervals must be JSON: {}".format(source, exc))
    if not isinstance(data, list) or not data:
        raise ValueError("{}: deleted_intervals must be a non-empty list".format(source))
    intervals = []
    previous_end = None
    for idx, item in enumerate(data):
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("{}: interval {} must be [start, end]".format(source, idx))
        start, end = item
        if isinstance(start, bool) or isinstance(end, bool):
            raise ValueError("{}: interval {} start/end must be integers".format(source, idx))
        start = strict_int(start, source, "deleted_intervals[{}][0]".format(idx))
        end = strict_int(end, source, "deleted_intervals[{}][1]".format(idx))
        if start < 0 or end < start:
            raise ValueError("{}: invalid closed-inclusive interval {}".format(source, item))
        if previous_end is not None and start <= previous_end:
            raise ValueError("{}: intervals must be sorted and non-overlapping under closed-inclusive coordinates".format(source))
        intervals.append((start, end))
        previous_end = end
    return intervals


def apply_deletion(sequence, intervals):
    pieces = []
    cursor = 0
    for start, end in intervals:
        pieces.append(sequence[cursor:start])
        cursor = end + 1
    pieces.append(sequence[cursor:])
    return "".join(pieces)


def retained_residue_mapping(wildtype_length, intervals):
    mapping = []
    interval_idx = 0
    deleted_before = 0
    for original_idx in range(wildtype_length):
        while interval_idx < len(intervals) and intervals[interval_idx][1] < original_idx:
            start, end = intervals[interval_idx]
            deleted_before += end - start + 1
            interval_idx += 1
        if interval_idx < len(intervals) and intervals[interval_idx][0] <= original_idx <= intervals[interval_idx][1]:
            continue
        mapping.append((original_idx, original_idx - deleted_before))
    return mapping


def validate_deletion_perturbation_table(rows, fields):
    errors = []
    warnings = []
    missing = [column for column in DELETION_PERTURBATION_REQUIRED_COLUMNS if column not in fields]
    for column in missing:
        errors.append("missing required column: {}".format(column))
    if missing:
        return errors, warnings

    for idx, row in enumerate(rows, start=2):
        source = "row {}".format(idx)
        try:
            if (row.get("coordinate_system") or "").strip() != COORDINATE_SYSTEM:
                raise ValueError("{}: coordinate_system must be {}".format(source, COORDINATE_SYSTEM))
            wildtype_length = strict_int(row.get("wildtype_length"), source, "wildtype_length")
            if wildtype_length <= 0:
                raise ValueError("{}: wildtype_length must be positive".format(source))
            intervals = parse_deleted_intervals_strict(row, source)
            for start, end in intervals:
                if end >= wildtype_length:
                    raise ValueError("{}: interval end must be < wildtype_length".format(source))
            expected_deleted_length = deleted_length(intervals)
            actual_deleted_length = strict_int(row.get("deleted_length"), source, "deleted_length")
            if actual_deleted_length != expected_deleted_length:
                raise ValueError(
                    "{}: deleted_length {} != closed-inclusive interval sum {}".format(
                        source, actual_deleted_length, expected_deleted_length
                    )
                )
            actual_num_intervals = strict_int(row.get("num_intervals"), source, "num_intervals")
            if actual_num_intervals != len(intervals):
                raise ValueError(
                    "{}: num_intervals {} != parsed interval count {}".format(
                        source, actual_num_intervals, len(intervals)
                    )
                )
            actual_fraction = strict_float(row.get("deletion_fraction"), source, "deletion_fraction")
            expected_fraction = expected_deleted_length / float(wildtype_length)
            if abs(actual_fraction - expected_fraction) > 1e-5:
                raise ValueError(
                    "{}: deletion_fraction {:.8f} != deleted_length / wildtype_length {:.8f}".format(
                        source, actual_fraction, expected_fraction
                    )
                )
            wildtype_sequence = row.get("wildtype_sequence", "")
            mutant_sequence = row.get("mutant_sequence", "")
            if wildtype_sequence:
                if len(wildtype_sequence) != wildtype_length:
                    raise ValueError("{}: wildtype_sequence length does not match wildtype_length".format(source))
                if mutant_sequence and apply_deletion(wildtype_sequence, intervals) != mutant_sequence:
                    raise ValueError("{}: mutant_sequence does not match closed-inclusive deletion reconstruction".format(source))
        except ValueError as exc:
            errors.append(str(exc))
    return errors, warnings


def model_path_status(model_name_or_path):
    if is_empty(model_name_or_path):
        return {
            "provided": False,
            "value": "",
            "local_path_like": False,
            "exists": False,
            "note": "teacher_model_name_or_path was not provided",
        }
    text = str(model_name_or_path).strip()
    local_path_like = text.startswith("/") or text.startswith(".")
    return {
        "provided": True,
        "value": text,
        "local_path_like": local_path_like,
        "exists": os.path.exists(text) if local_path_like else "not_checked_for_model_id",
        "note": "local path checked shallowly" if local_path_like else "model id string recorded without download",
    }


def resolve_teacher_model_path(model_name_or_path, cache_dir=None):
    if not model_name_or_path:
        return model_name_or_path
    if os.path.isdir(model_name_or_path):
        snapshots_dir = os.path.join(model_name_or_path, "snapshots")
        refs_main = os.path.join(model_name_or_path, "refs", "main")
        if os.path.isdir(snapshots_dir):
            snapshot = None
            if os.path.exists(refs_main):
                with open(refs_main) as handle:
                    ref = handle.read().strip()
                candidate = os.path.join(snapshots_dir, ref)
                if os.path.isdir(candidate):
                    snapshot = candidate
            if snapshot is None:
                candidates = [
                    os.path.join(snapshots_dir, name)
                    for name in os.listdir(snapshots_dir)
                    if os.path.isdir(os.path.join(snapshots_dir, name))
                ]
                if candidates:
                    snapshot = sorted(candidates)[-1]
            if snapshot is not None:
                return snapshot
        return model_name_or_path
    if "/" in model_name_or_path and cache_dir:
        namespace, repo = model_name_or_path.split("/", 1)
        repo_dir = os.path.join(cache_dir, "models--{}--{}".format(namespace, repo))
        if os.path.isdir(repo_dir):
            return resolve_teacher_model_path(repo_dir, cache_dir=None)
    return model_name_or_path


def teacher_device_for_real_scoring(args, torch_module):
    if args.teacher_device == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    return torch_module.device(args.teacher_device)


def normalize_esmc_state_key(key):
    if key.endswith("._extra_state") or key.endswith("_extra_state"):
        return None
    if key.startswith("esmc."):
        key = key[len("esmc.") :]
    if ".attn.layernorm_qkv.layer_norm_weight" in key:
        key = key.replace(".attn.layernorm_qkv.layer_norm_weight", ".attn.layernorm_qkv.0.weight")
    elif ".attn.layernorm_qkv.layer_norm_bias" in key:
        key = key.replace(".attn.layernorm_qkv.layer_norm_bias", ".attn.layernorm_qkv.0.bias")
    elif ".attn.layernorm_qkv.weight" in key:
        key = key.replace(".attn.layernorm_qkv.weight", ".attn.layernorm_qkv.1.weight")
    elif ".ffn.layer_norm_weight" in key:
        key = key.replace(".ffn.layer_norm_weight", ".ffn.0.weight")
    elif ".ffn.layer_norm_bias" in key:
        key = key.replace(".ffn.layer_norm_bias", ".ffn.0.bias")
    elif ".ffn.fc1_weight" in key:
        key = key.replace(".ffn.fc1_weight", ".ffn.1.weight")
    elif ".ffn.fc2_weight" in key:
        key = key.replace(".ffn.fc2_weight", ".ffn.3.weight")
    elif key.startswith("lm_head."):
        key = "sequence_head." + key[len("lm_head.") :]
    return key


def load_esm2_transformers_backend(args):
    from transformers import AutoModel, AutoTokenizer
    import torch

    local_files_only = not bool(args.teacher_allow_download)
    cache_dir = args.teacher_cache_dir or os.environ.get("DELEPREF_HF_CACHE") or os.environ.get("HF_HUB_CACHE")
    model_path = resolve_teacher_model_path(args.teacher_model_name_or_path, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    device = teacher_device_for_real_scoring(args, torch)
    model.to(device)
    model.eval()
    return {
        "backend_type": ESM2_TRANSFORMERS_BACKEND,
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "resolved_model_path": model_path,
        "cache_dir": cache_dir or "",
        "local_files_only": local_files_only,
    }


def load_esmc_official_safetensors_backend(args):
    import inspect
    import torch
    from esm.models.esmc import ESMC
    try:
        from esm.tokenization import get_esmc_model_tokenizers
    except ImportError:
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        def get_esmc_model_tokenizers():
            return EsmSequenceTokenizer()
    from safetensors.torch import load_file

    model_path = resolve_teacher_model_path(args.teacher_model_name_or_path, cache_dir=None)
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError("ESMC safetensors not found: {}".format(safetensors_path))
    device = teacher_device_for_real_scoring(args, torch)
    if device.type != "cuda":
        raise RuntimeError("ESMC teacher scoring requires CUDA; got device {}".format(device))
    esmc_kwargs = {
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
        "tokenizer": get_esmc_model_tokenizers(),
    }
    if "use_flash_attn" in inspect.signature(ESMC).parameters:
        esmc_kwargs["use_flash_attn"] = False
    model = ESMC(**esmc_kwargs).eval()
    state = load_file(safetensors_path, device="cpu")
    mapped = {}
    skipped = []
    for key, tensor in state.items():
        target_key = normalize_esmc_state_key(key)
        if target_key is None:
            skipped.append(key)
            continue
        mapped[target_key] = tensor
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "ESMC state_dict mapping mismatch: missing={} unexpected={} skipped={}".format(
                list(missing)[:20],
                list(unexpected)[:20],
                len(skipped),
            )
        )
    model.to(device)
    model.to(torch.bfloat16)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "backend_type": ESMC_OFFICIAL_SAFETENSORS_BACKEND,
        "torch": torch,
        "tokenizer": model.tokenizer,
        "model": model,
        "device": device,
        "resolved_model_path": model_path,
        "cache_dir": "",
        "local_files_only": True,
        "load_info": {
            "loader_type": "official_esm_models_esmc_manual_safetensors",
            "state_dict_keys": len(state),
            "mapped_state_dict_keys": len(mapped),
            "skipped_state_dict_keys": len(skipped),
            "missing_keys": [],
            "unexpected_keys": [],
            "dtype": str(next(model.parameters()).dtype),
        },
    }


def load_real_plm_teacher_backend(args):
    if not args.teacher_model_name_or_path:
        raise ValueError("--teacher_model_name_or_path is required for real PLM teacher scoring")
    if args.teacher_backend == ESM2_TRANSFORMERS_BACKEND:
        return load_esm2_transformers_backend(args)
    if args.teacher_backend == ESMC_OFFICIAL_SAFETENSORS_BACKEND:
        return load_esmc_official_safetensors_backend(args)
    raise ValueError("unsupported teacher_backend: {}".format(args.teacher_backend))


def residue_embeddings_for_sequences_esm2_transformers(backend, sequences, batch_size=None):
    torch = backend["torch"]
    tokenizer = backend["tokenizer"]
    model = backend["model"]
    device = backend["device"]
    if not sequences:
        return []
    effective_batch = max(1, int(batch_size or len(sequences)))
    out = []
    for start in range(0, len(sequences), effective_batch):
        batch_sequences = list(sequences[start : start + effective_batch])
        encoded = tokenizer(
            batch_sequences,
            return_tensors="pt",
            add_special_tokens=True,
            return_special_tokens_mask=True,
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        special_mask = encoded.pop("special_tokens_mask")
        with torch.no_grad():
            output = model(**encoded)
        hidden = output.last_hidden_state
        for batch_idx, sequence in enumerate(batch_sequences):
            residue_mask = (special_mask[batch_idx] == 0) & (encoded["attention_mask"][batch_idx] == 1)
            residues = hidden[batch_idx][residue_mask]
            if int(residues.shape[0]) != len(sequence):
                raise ValueError(
                    "tokenizer residue count {} does not match sequence length {}".format(
                        int(residues.shape[0]), len(sequence)
                    )
                )
            out.append(residues)
    return out


def residue_embeddings_for_sequences_esmc(backend, sequences, batch_size=None):
    torch = backend["torch"]
    tokenizer = backend["tokenizer"]
    model = backend["model"]
    device = backend["device"]
    if not sequences:
        return []
    effective_batch = max(1, int(batch_size or len(sequences)))
    out = []
    for start in range(0, len(sequences), effective_batch):
        batch_sequences = list(sequences[start : start + effective_batch])
        encoded = tokenizer(batch_sequences, return_tensors="pt", padding=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(sequence_tokens=input_ids)
        embeddings = output.embeddings
        for batch_idx, sequence in enumerate(batch_sequences):
            valid_positions = torch.nonzero(attention_mask[batch_idx].bool(), as_tuple=False).flatten().tolist()
            if len(valid_positions) >= len(sequence) + 2:
                positions = valid_positions[1 : 1 + len(sequence)]
                alignment_note = "removed_special_tokens_cls_eos_then_residue_length_matches"
            else:
                positions = valid_positions[: len(sequence)]
                alignment_note = "used_first_valid_tokens_residue_length_matches"
            residues = embeddings[batch_idx, positions, :]
            if int(residues.shape[0]) != len(sequence):
                raise ValueError(
                    "ESMC tokenizer residue count {} does not match sequence length {}".format(
                        int(residues.shape[0]), len(sequence)
                    )
                )
            backend.setdefault("embedding_alignment_notes", set()).add(alignment_note)
            out.append(residues)
    return out


def residue_embeddings_for_sequences(backend, sequences, batch_size=None):
    if backend.get("backend_type") == ESMC_OFFICIAL_SAFETENSORS_BACKEND:
        return residue_embeddings_for_sequences_esmc(backend, sequences, batch_size=batch_size)
    return residue_embeddings_for_sequences_esm2_transformers(backend, sequences, batch_size=batch_size)


def residue_embeddings_for_sequence(backend, sequence):
    return residue_embeddings_for_sequences(backend, [sequence], batch_size=1)[0]


def real_plm_teacher_impact_from_embeddings(row, args, backend, wt_embeddings, mutant_embeddings):
    source = "perturbation {}".format(row.get("perturbation_id", ""))
    wildtype_length = strict_int(row.get("wildtype_length"), source, "wildtype_length")
    intervals = parse_deleted_intervals_strict(row, source)
    wildtype_sequence = row.get("wildtype_sequence", "")
    if not wildtype_sequence:
        raise ValueError("{}: wildtype_sequence is required for real PLM scoring".format(source))
    mutant_sequence = row.get("mutant_sequence") or apply_deletion(wildtype_sequence, intervals)
    if apply_deletion(wildtype_sequence, intervals) != mutant_sequence:
        raise ValueError("{}: mutant_sequence does not match closed-inclusive deletion reconstruction".format(source))
    if int(wt_embeddings.shape[0]) != wildtype_length:
        raise ValueError("{}: cached wildtype embedding length mismatch".format(source))
    if int(mutant_embeddings.shape[0]) != len(mutant_sequence):
        raise ValueError("{}: mutant embedding length mismatch".format(source))

    torch = backend["torch"]
    mapping = retained_residue_mapping(wildtype_length, intervals)
    global_shift = cosine_distance_unit(torch, wt_embeddings.mean(dim=0), mutant_embeddings.mean(dim=0))
    retained_shift = mean_retained_distance(torch, wt_embeddings, mutant_embeddings, mapping)
    junction, junction_detail = junction_plausibility_proxy(torch, wt_embeddings, mutant_embeddings, wildtype_length, intervals)
    global_similarity = clamp01(1.0 - 2.0 * global_shift)
    retained_similarity = clamp01(1.0 - 2.0 * retained_shift)
    impact_confidence = 0.70
    view_status = {
        "coordinate_system": COORDINATE_SYSTEM,
        "teacher_scoring_mode": REAL_PLM_TEACHER_MODE,
        "teacher_backend": args.teacher_backend,
        "global_shift": {
            "status": "ok",
            "metric": "pooled_residue_embedding_cosine_distance_unit",
            "pooling": "mean_pool_residue_tokens",
            "backend": "real_plm",
            "similarity": round(global_similarity, 8),
        },
        "retained_region_shift": {
            "status": "ok",
            "metric": "mean_retained_residue_cosine_distance_unit",
            "retained_count": len(mapping),
            "mapping": "closed_inclusive_deleted_residue_mapping",
            "backend": "real_plm",
            "similarity": round(retained_similarity, 8),
        },
        "junction_plausibility": {
            "status": "ok",
            "metric": "wt_mutant_boundary_anchor_cosine_similarity_proxy",
            "window_radius": 0,
            "boundary_types": boundary_types(wildtype_length, intervals),
            "direction": "higher_is_more_plausible",
            "backend": "real_plm_fast_proxy",
            **junction_detail,
        },
        "sequence_manifold_plausibility": {
            "status": "missing",
            "reason": "deferred_for_fast_real_plm_mode_no_masked_marginal_or_pseudolikelihood",
        },
        "risk_overlap": {
            "status": "missing",
            "metric": "no_risk_input_zero_overlap_placeholder",
            "reason": "no real risk input supplied",
        },
    }
    risk_summary = {
        "coordinate_system": COORDINATE_SYSTEM,
        "risk_source": "none",
        "risk_overlap": 0.0,
        "risk_confidence": 0.0,
        "note": "No real risk input was supplied; zero values are placeholders and not a biological safety claim.",
    }
    metadata = {
        "source_deletion_perturbations_csv": args.deletion_perturbations_csv,
        "generation_policy": row.get("generation_policy", ""),
        "teacher_backend": args.teacher_backend,
        "teacher_model_name_or_path": args.teacher_model_name_or_path,
        "resolved_model_path": backend.get("resolved_model_path", ""),
        "teacher_cache_dir": backend.get("cache_dir", ""),
        "local_files_only": backend.get("local_files_only", True),
        "load_info": backend.get("load_info", {}),
        "teacher_allow_download": bool(args.teacher_allow_download),
        "teacher_device": str(backend["device"]),
        "fast_real_plm": True,
        "full_scoring": False,
        "sequence_manifold_plausibility_deferred": True,
        "no_risk_input": True,
        "teacher_group_by_protein": bool(getattr(args, "teacher_group_by_protein", False)),
        "mutant_batch_size": int(getattr(args, "teacher_batch_size", 1)),
        "global_similarity": round(global_similarity, 8),
        "retained_similarity": round(retained_similarity, 8),
        "embedding_alignment_notes": sorted(backend.get("embedding_alignment_notes", [])),
    }
    if backend["device"].type == "cuda" and torch.cuda.is_available():
        metadata["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(backend["device"]) / (1024.0 * 1024.0)
    teacher_model = args.teacher_model or args.teacher_model_name_or_path
    return {
        "protein_id": row.get("protein_id", ""),
        "perturbation_id": row.get("perturbation_id", ""),
        "plan_id": row.get("plan_id", ""),
        "teacher_model": teacher_model,
        "teacher_scoring_mode": REAL_PLM_TEACHER_MODE,
        "global_shift": fmt_score(global_shift),
        "retained_region_shift": fmt_score(retained_shift),
        "junction_plausibility": fmt_score(junction),
        "sequence_manifold_plausibility": "",
        "risk_overlap": fmt_score(0.0),
        "risk_confidence": fmt_score(0.0),
        "impact_confidence": fmt_score(impact_confidence),
        "scoring_status": "ok",
        "global_similarity": fmt_score(global_similarity),
        "retained_similarity": fmt_score(retained_similarity),
        "junction_mean": fmt_score(junction_detail.get("junction_mean", 0.0)),
        "junction_min": fmt_score(junction_detail.get("junction_min", 0.0)),
        "teacher_backend": args.teacher_backend,
        "max_memory_allocated_mb": "{:.3f}".format(float(metadata.get("max_memory_allocated_mb", 0.0))) if "max_memory_allocated_mb" in metadata else "",
        "view_status_json": json.dumps(view_status, sort_keys=True, separators=(",", ":")),
        "risk_summary_json": json.dumps(risk_summary, sort_keys=True, separators=(",", ":")),
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    }


def cosine_distance_unit(torch, a, b):
    cosine = torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
    return clamp01((1.0 - cosine) / 2.0)


def mean_retained_distance(torch, wt_embeddings, mutant_embeddings, mapping):
    if not mapping:
        return 1.0
    wt_stack = torch.stack([wt_embeddings[original_idx] for original_idx, _mutant_idx in mapping]).float()
    mutant_stack = torch.stack([mutant_embeddings[mutant_idx] for _original_idx, mutant_idx in mapping]).float()
    similarities = torch.nn.functional.cosine_similarity(wt_stack, mutant_stack, dim=1)
    distances = ((1.0 - similarities) / 2.0).clamp(0.0, 1.0)
    return distances.mean().item()


def combine_junction_values(values):
    if not values:
        return 0.0
    if len(values) == 1:
        return clamp01(values[0])
    junction_mean = sum(values) / float(len(values))
    junction_min = min(values)
    return clamp01(0.5 * junction_mean + 0.5 * junction_min)


def junction_plausibility_proxy(torch, wt_embeddings, mutant_embeddings, wildtype_length, intervals):
    values = []
    mapping = dict(retained_residue_mapping(wildtype_length, intervals))
    for start, end in intervals:
        left = start - 1
        right = end + 1
        if left in mapping and right in mapping:
            left_shift = cosine_distance_unit(torch, wt_embeddings[left], mutant_embeddings[mapping[left]])
            right_shift = cosine_distance_unit(torch, wt_embeddings[right], mutant_embeddings[mapping[right]])
            values.append(clamp01(1.0 - (left_shift + right_shift)))
        elif left in mapping or right in mapping:
            anchor = left if left in mapping else right
            values.append(clamp01(1.0 - 2.0 * cosine_distance_unit(torch, wt_embeddings[anchor], mutant_embeddings[mapping[anchor]])))
        else:
            values.append(0.0)
    detail = {
        "junction_values": [round(float(value), 8) for value in values],
        "junction_mean": round(mean(values), 8) if values else 0.0,
        "junction_min": round(min(values), 8) if values else 0.0,
        "multi_interval_aggregation": "single_value" if len(values) <= 1 else "0.5*junction_mean+0.5*junction_min",
    }
    return combine_junction_values(values), detail


def real_plm_teacher_impact_row(row, args, backend):
    source = "perturbation {}".format(row.get("perturbation_id", ""))
    wildtype_length = strict_int(row.get("wildtype_length"), source, "wildtype_length")
    intervals = parse_deleted_intervals_strict(row, source)
    wildtype_sequence = row.get("wildtype_sequence", "")
    if not wildtype_sequence:
        raise ValueError("{}: wildtype_sequence is required for real PLM scoring".format(source))
    mutant_sequence = row.get("mutant_sequence") or apply_deletion(wildtype_sequence, intervals)
    if apply_deletion(wildtype_sequence, intervals) != mutant_sequence:
        raise ValueError("{}: mutant_sequence does not match closed-inclusive deletion reconstruction".format(source))
    wt_embeddings = residue_embeddings_for_sequence(backend, wildtype_sequence)
    mutant_embeddings = residue_embeddings_for_sequence(backend, mutant_sequence)
    return real_plm_teacher_impact_from_embeddings(row, args, backend, wt_embeddings, mutant_embeddings)


def read_existing_teacher_impact_rows(path):
    if not path or not os.path.exists(path):
        return [], set()
    rows, _ = read_csv(path)
    completed = set()
    for row in rows:
        status = (row.get("scoring_status") or "").strip()
        if status in ("ok", "partial"):
            completed.add((row.get("protein_id", ""), row.get("perturbation_id", ""), row.get("plan_id", "")))
    return rows, completed


def failed_real_plm_teacher_impact_row(row, args, error_message):
    view_status = {
        "coordinate_system": COORDINATE_SYSTEM,
        "teacher_scoring_mode": REAL_PLM_TEACHER_MODE,
        "teacher_backend": args.teacher_backend,
        "status": "failed",
        "error": str(error_message),
    }
    metadata = {
        "source_deletion_perturbations_csv": args.deletion_perturbations_csv,
        "teacher_model_name_or_path": args.teacher_model_name_or_path,
        "teacher_allow_download": bool(args.teacher_allow_download),
        "no_download_by_default": not bool(args.teacher_allow_download),
        "teacher_group_by_protein": bool(getattr(args, "teacher_group_by_protein", False)),
    }
    return {
        "protein_id": row.get("protein_id", ""),
        "perturbation_id": row.get("perturbation_id", ""),
        "plan_id": row.get("plan_id", ""),
        "teacher_model": args.teacher_model or args.teacher_model_name_or_path,
        "teacher_scoring_mode": REAL_PLM_TEACHER_MODE,
        "global_shift": "",
        "retained_region_shift": "",
        "junction_plausibility": "",
        "sequence_manifold_plausibility": "",
        "risk_overlap": "",
        "risk_confidence": "",
        "impact_confidence": "0.000000",
        "scoring_status": "failed",
        "global_similarity": "",
        "retained_similarity": "",
        "junction_mean": "",
        "junction_min": "",
        "teacher_backend": args.teacher_backend,
        "max_memory_allocated_mb": "",
        "view_status_json": json.dumps(view_status, sort_keys=True, separators=(",", ":")),
        "risk_summary_json": json.dumps({"risk_source": "none"}, sort_keys=True, separators=(",", ":")),
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    }


def build_real_plm_teacher_impact_scores_grouped(args, rows):
    if not args.teacher_model_name_or_path:
        raise ValueError("--teacher_model_name_or_path is required for {}".format(REAL_PLM_TEACHER_MODE))
    if args.teacher_batch_size <= 0:
        raise ValueError("--teacher_batch_size must be positive")
    if args.max_teacher_rows is not None and args.max_teacher_rows <= 0:
        raise ValueError("--max_teacher_rows must be positive when provided")
    selected_rows = rows[: args.max_teacher_rows] if args.max_teacher_rows is not None else rows
    existing_rows, completed = read_existing_teacher_impact_rows(args.teacher_impact_scores_csv) if args.teacher_resume else ([], set())
    backend = load_real_plm_teacher_backend(args)
    out_rows = list(existing_rows)
    stats = Counter()
    grouped = defaultdict(list)
    for row in selected_rows:
        key = (row.get("protein_id", ""), row.get("perturbation_id", ""), row.get("plan_id", ""))
        if key in completed:
            stats["resumed_rows"] += 1
            continue
        grouped[row.get("protein_id", "")].append(row)

    for protein_idx, protein_id in enumerate(sorted(grouped), start=1):
        protein_rows = grouped[protein_id]
        try:
            wildtype_sequences = sorted(set(row.get("wildtype_sequence", "") for row in protein_rows))
            if len(wildtype_sequences) != 1 or not wildtype_sequences[0]:
                raise ValueError("protein {} must have exactly one non-empty wildtype_sequence".format(protein_id))
            wt_embeddings = residue_embeddings_for_sequence(backend, wildtype_sequences[0])
            stats["wildtype_embedding_calls"] += 1
        except Exception as exc:  # pragma: no cover - real model path.
            for row in protein_rows:
                out_rows.append(failed_real_plm_teacher_impact_row(row, args, exc))
                stats["failed_rows"] += 1
            continue

        mutant_payloads = []
        for row in protein_rows:
            try:
                intervals = parse_deleted_intervals_strict(row, "perturbation {}".format(row.get("perturbation_id", "")))
                wildtype_sequence = row.get("wildtype_sequence", "")
                mutant_sequence = row.get("mutant_sequence") or apply_deletion(wildtype_sequence, intervals)
                if apply_deletion(wildtype_sequence, intervals) != mutant_sequence:
                    raise ValueError("mutant_sequence does not match closed-inclusive deletion reconstruction")
                mutant_payloads.append((row, mutant_sequence, None))
            except Exception as exc:
                mutant_payloads.append((row, "", exc))

        for batch_start in range(0, len(mutant_payloads), int(args.teacher_batch_size)):
            batch = mutant_payloads[batch_start : batch_start + int(args.teacher_batch_size)]
            valid = [(row, sequence) for row, sequence, error in batch if error is None]
            embeddings_by_key = {}
            if valid:
                try:
                    mutant_embeddings = residue_embeddings_for_sequences(
                        backend,
                        [sequence for _row, sequence in valid],
                        batch_size=int(args.teacher_batch_size),
                    )
                    stats["mutant_embedding_batches"] += 1
                    for (row, _sequence), embedding in zip(valid, mutant_embeddings):
                        embeddings_by_key[(row.get("perturbation_id", ""), row.get("plan_id", ""))] = embedding
                except Exception as exc:  # pragma: no cover - real model path.
                    for row, _sequence in valid:
                        embeddings_by_key[(row.get("perturbation_id", ""), row.get("plan_id", ""))] = exc
            for row, _sequence, error in batch:
                if error is not None:
                    out_rows.append(failed_real_plm_teacher_impact_row(row, args, error))
                    stats["failed_rows"] += 1
                    continue
                value = embeddings_by_key.get((row.get("perturbation_id", ""), row.get("plan_id", "")))
                if isinstance(value, Exception):
                    out_rows.append(failed_real_plm_teacher_impact_row(row, args, value))
                    stats["failed_rows"] += 1
                    continue
                try:
                    out_rows.append(real_plm_teacher_impact_from_embeddings(row, args, backend, wt_embeddings, value))
                    stats["ok_rows"] += 1
                except Exception as exc:  # pragma: no cover - real model path.
                    out_rows.append(failed_real_plm_teacher_impact_row(row, args, exc))
                    stats["failed_rows"] += 1
        if protein_idx % 100 == 0:
            print("grouped teacher scoring proteins: {}".format(protein_idx))

    if not out_rows:
        raise ValueError("no teacher impact rows generated")
    ensure_parent(args.teacher_impact_scores_csv)
    fieldnames = TEACHER_IMPACT_REQUIRED_COLUMNS + TEACHER_IMPACT_OPTIONAL_COLUMNS
    tmp_path = args.teacher_impact_scores_csv + ".tmp"
    with open(tmp_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    os.replace(tmp_path, args.teacher_impact_scores_csv)
    stats["total_rows"] = len(out_rows)
    stats[args.teacher_scoring_mode] = stats["ok_rows"]
    return stats


def build_real_plm_teacher_impact_scores(args, rows):
    if not args.teacher_model_name_or_path:
        raise ValueError("--teacher_model_name_or_path is required for {}".format(REAL_PLM_TEACHER_MODE))
    if args.teacher_batch_size <= 0:
        raise ValueError("--teacher_batch_size must be positive")
    if args.max_teacher_rows is not None and args.max_teacher_rows <= 0:
        raise ValueError("--max_teacher_rows must be positive when provided")
    selected_rows = rows[: args.max_teacher_rows] if args.max_teacher_rows is not None else rows
    existing_rows, completed = read_existing_teacher_impact_rows(args.teacher_impact_scores_csv) if args.teacher_resume else ([], set())
    backend = load_real_plm_teacher_backend(args)
    out_rows = list(existing_rows)
    stats = Counter()
    for row in selected_rows:
        key = (row.get("protein_id", ""), row.get("perturbation_id", ""), row.get("plan_id", ""))
        if key in completed:
            stats["resumed_rows"] += 1
            continue
        try:
            out_rows.append(real_plm_teacher_impact_row(row, args, backend))
            stats["ok_rows"] += 1
        except Exception as exc:  # pragma: no cover - exercised only by real model runs.
            out_rows.append(failed_real_plm_teacher_impact_row(row, args, exc))
            stats["failed_rows"] += 1
    if not out_rows:
        raise ValueError("no teacher impact rows generated")
    ensure_parent(args.teacher_impact_scores_csv)
    fieldnames = TEACHER_IMPACT_REQUIRED_COLUMNS + TEACHER_IMPACT_OPTIONAL_COLUMNS
    with open(args.teacher_impact_scores_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    stats["total_rows"] = len(out_rows)
    stats[args.teacher_scoring_mode] = stats["ok_rows"]
    return stats


def real_plm_preflight_summary(args, rows, fields):
    errors, warnings = validate_deletion_perturbation_table(rows, fields)
    if not args.teacher_model_name_or_path:
        warnings.append("--teacher_model_name_or_path was not provided; real scoring would fail until set.")
    if args.teacher_allow_download:
        errors.append("--teacher_allow_download must be false for the reproducible local-teacher protocol")
    if args.teacher_scoring_mode != REAL_PLM_TEACHER_MODE:
        errors.append("--teacher_scoring_mode must be {}".format(REAL_PLM_TEACHER_MODE))
    if args.teacher_backend not in SUPPORTED_TEACHER_BACKENDS:
        errors.append("--teacher_backend must be one of {}".format(", ".join(SUPPORTED_TEACHER_BACKENDS)))
    if args.teacher_device not in SUPPORTED_TEACHER_DEVICES:
        errors.append("--teacher_device must be one of {}".format(", ".join(SUPPORTED_TEACHER_DEVICES)))
    if args.teacher_batch_size <= 0:
        errors.append("--teacher_batch_size must be positive")
    if args.max_teacher_rows is not None and args.max_teacher_rows <= 0:
        errors.append("--max_teacher_rows must be positive when provided")

    policy_counts = Counter(row.get("generation_policy", "") for row in rows)
    summary = {
        "schema_version": "real_plm_teacher_scoring_adapter_preflight_v0",
        "status": "fail" if errors else "pass",
        "not_for_experiment": True,
        "preflight_only": True,
        "teacher_scoring_executed": False,
        "teacher_impact_scores_written": False,
        "no_training": True,
        "no_backward": True,
        "no_optimizer_step": True,
        "no_gpu_used": True,
        "no_srun": True,
        "no_torchrun": True,
        "no_model_loaded": True,
        "no_forward": True,
        "no_download": not bool(args.teacher_allow_download),
        "input_deletion_table": {
            "path": args.deletion_perturbations_csv,
            "exists": os.path.exists(args.deletion_perturbations_csv) if args.deletion_perturbations_csv else False,
            "rows": len(rows),
            "coordinate_system": COORDINATE_SYSTEM,
            "validator_like_checks_pass": not bool(errors),
            "row_counts_by_generation_policy": dict(sorted(policy_counts.items())),
        },
        "adapter": {
            "teacher_scoring_mode": args.teacher_scoring_mode,
            "teacher_backend": args.teacher_backend,
            "teacher_model_name_or_path": args.teacher_model_name_or_path or "",
            "teacher_model_path_status": model_path_status(args.teacher_model_name_or_path),
            "teacher_allow_download": bool(args.teacher_allow_download),
            "teacher_device": args.teacher_device,
            "teacher_batch_size": args.teacher_batch_size,
            "max_teacher_rows": args.max_teacher_rows,
            "resume": bool(args.teacher_resume),
            "cache_dir": args.teacher_cache_dir or "",
        },
        "planned_output_contract": {
            "teacher_impact_scores_csv": args.teacher_impact_scores_csv or "",
            "will_include_views": REAL_PLM_VIEW_NAMES,
            "coordinate_system": COORDINATE_SYSTEM,
            "scoring_status_values": ["ok", "partial", "failed"],
        },
        "warnings": warnings,
        "failed_checks": errors,
        "implementation_notes": [
            "No-model preflight reads and validates the deletion table only.",
            "Real PLM imports are lazy and are not reached by --teacher_no_model_preflight.",
            "Real scoring uses local_files_only=True unless --teacher_allow_download is explicitly provided.",
            "retained_region_shift uses closed-inclusive retained residue mapping, not raw-index alignment.",
            "sequence_manifold_plausibility is deferred for fast_real_plm and must be marked missing in view_status_json.",
            "risk_overlap/risk_confidence are no-risk-input placeholders and not biological safety claims.",
        ],
        "recommended_next_step": "Run a small local-teacher scoring dry-run and validate the output schema.",
    }
    return summary


def build_real_plm_preflight(args, rows, fields):
    if not args.teacher_preflight_summary_json:
        raise ValueError("--teacher_preflight_summary_json is required with --teacher_no_model_preflight")
    summary = real_plm_preflight_summary(args, rows, fields)
    ensure_parent(args.teacher_preflight_summary_json)
    with open(args.teacher_preflight_summary_json, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if summary["status"] != "pass":
        raise ValueError("real PLM adapter preflight failed: {}".format("; ".join(summary["failed_checks"])))
    stats = Counter()
    stats["total_rows"] = len(rows)
    stats["preflight_rows"] = len(rows)
    stats[args.teacher_scoring_mode] = 0
    return stats


def mock_teacher_views(row, intervals, wildtype_length):
    fraction = deleted_length(intervals) / float(wildtype_length)
    junction = 0.80 if "two_sided" in boundary_types(wildtype_length, intervals) else 0.65
    return {
        "global_shift": fraction,
        "retained_region_shift": fraction * 0.5,
        "junction_plausibility": junction,
        "sequence_manifold_plausibility": 0.75,
        "risk_overlap": 0.0,
        "risk_confidence": 0.0,
        "impact_confidence": 0.50,
    }


def fast_tiny_teacher_views(row, intervals, wildtype_length):
    fraction = deleted_length(intervals) / float(wildtype_length)
    types = boundary_types(wildtype_length, intervals)
    two_sided_fraction = sum(1 for item in types if item == "two_sided") / float(max(1, len(types)))
    junction = 0.60 + 0.25 * two_sided_fraction
    return {
        "global_shift": fraction,
        "retained_region_shift": fraction * 0.25,
        "junction_plausibility": junction,
        "sequence_manifold_plausibility": "",
        "risk_overlap": 0.0,
        "risk_confidence": 0.0,
        "impact_confidence": 0.60,
    }


def build_view_status(mode, row, intervals, wildtype_length, views):
    types = boundary_types(wildtype_length, intervals)
    sequence_status = "ok" if views["sequence_manifold_plausibility"] != "" else "missing"
    sequence_payload = (
        {
            "status": "ok",
            "metric": "mock_plausibility",
        }
        if sequence_status == "ok"
        else {
            "status": "missing",
            "reason": "deferred_for_fast_mode",
        }
    )
    return {
        "coordinate_system": COORDINATE_SYSTEM,
        "teacher_scoring_mode": mode,
        "global_shift": {
            "status": "ok",
            "metric": "deleted_fraction_proxy",
            "pooling": "none_mock_proxy",
        },
        "retained_region_shift": {
            "status": "ok",
            "metric": "retained_mapping_proxy",
            "retained_count": retained_count(wildtype_length, intervals),
            "mapping": "closed_inclusive_deleted_residue_mapping",
        },
        "junction_plausibility": {
            "status": "ok",
            "metric": "boundary_type_proxy",
            "window_radius": 0,
            "boundary_types": types,
        },
        "sequence_manifold_plausibility": sequence_payload,
        "risk_overlap": {
            "status": "ok",
            "metric": "no_risk_data_zero_overlap",
        },
    }


def teacher_impact_row(row, mode, teacher_model, source_path):
    source = "perturbation {}".format(row.get("perturbation_id", ""))
    wildtype_length = safe_int(row.get("wildtype_length"), 0)
    intervals = parse_deleted_intervals(row, source)
    if wildtype_length <= 0:
        raise ValueError("{}: wildtype_length must be positive".format(source))
    for start, end in intervals:
        if end >= wildtype_length:
            raise ValueError("{}: interval end must be < wildtype_length".format(source))
    if mode == "mock":
        views = mock_teacher_views(row, intervals, wildtype_length)
    elif mode == "fast":
        views = fast_tiny_teacher_views(row, intervals, wildtype_length)
    else:
        raise ValueError("unsupported teacher_scoring_mode for teacher smoke: {}".format(mode))
    view_status = build_view_status(mode, row, intervals, wildtype_length, views)
    risk_summary = {
        "coordinate_system": COORDINATE_SYSTEM,
        "risk_source": "none",
        "risk_overlap": 0.0,
        "risk_confidence": 0.0,
        "note": "teacher smoke mock/fast tiny mode does not generate DelePref risk features.",
    }
    metadata = {
        "source_deletion_perturbations_csv": source_path,
        "generation_policy": row.get("generation_policy", ""),
        "no_model_loaded": True,
        "no_gpu_required": True,
        "full_scoring": False,
    }
    return {
        "protein_id": row.get("protein_id", ""),
        "perturbation_id": row.get("perturbation_id", ""),
        "plan_id": row.get("plan_id", ""),
        "teacher_model": teacher_model,
        "teacher_scoring_mode": mode,
        "global_shift": fmt_score(views["global_shift"]),
        "retained_region_shift": fmt_score(views["retained_region_shift"]),
        "junction_plausibility": fmt_score(views["junction_plausibility"]),
        "sequence_manifold_plausibility": fmt_score(views["sequence_manifold_plausibility"]),
        "risk_overlap": fmt_score(views["risk_overlap"]),
        "risk_confidence": fmt_score(views["risk_confidence"]),
        "impact_confidence": fmt_score(views["impact_confidence"]),
        "scoring_status": "ok",
        "view_status_json": json.dumps(view_status, sort_keys=True, separators=(",", ":")),
        "risk_summary_json": json.dumps(risk_summary, sort_keys=True, separators=(",", ":")),
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    }


def build_teacher_impact_scores(args):
    if not args.deletion_perturbations_csv:
        raise ValueError("--deletion_perturbations_csv is required with --build_teacher_impact_scores")
    if not args.teacher_impact_scores_csv:
        raise ValueError("--teacher_impact_scores_csv is required with --build_teacher_impact_scores")
    rows, fields = read_csv(args.deletion_perturbations_csv)
    if args.teacher_scoring_mode == REAL_PLM_TEACHER_MODE:
        if args.teacher_no_model_preflight:
            return build_real_plm_preflight(args, rows, fields)
        errors, _warnings = validate_deletion_perturbation_table(rows, fields)
        if errors:
            raise ValueError("deletion perturbation table failed real PLM preflight checks: {}".format("; ".join(errors)))
        if args.teacher_group_by_protein:
            return build_real_plm_teacher_impact_scores_grouped(args, rows)
        return build_real_plm_teacher_impact_scores(args, rows)
    if args.limit_perturbations is not None:
        rows = rows[: args.limit_perturbations]
    teacher_model = args.teacher_model
    if not teacher_model:
        teacher_model = "mock_teacher_v0" if args.teacher_scoring_mode == "mock" else "fast_tiny_string_proxy_v0"
    out_rows = [
        teacher_impact_row(row, args.teacher_scoring_mode, teacher_model, args.deletion_perturbations_csv)
        for row in rows
    ]
    if not out_rows:
        raise ValueError("no teacher impact rows generated")
    ensure_parent(args.teacher_impact_scores_csv)
    fieldnames = TEACHER_IMPACT_REQUIRED_COLUMNS + TEACHER_IMPACT_OPTIONAL_COLUMNS
    with open(args.teacher_impact_scores_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    stats = Counter()
    stats["total_rows"] = len(out_rows)
    stats[args.teacher_scoring_mode] = len(out_rows)
    return stats


def parse_float(value, default=0.0):
    if is_empty(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_optional_finite_float(value):
    if is_empty(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return None
    return parsed


PAIR_QUALITY_BUCKETS = [
    "both_high",
    "clean_preference",
    "acceptable_preference",
    "tie_or_uncertain",
    "low_vs_lower",
    "both_low",
    "conflicting",
    "unknown",
]

PAIR_TRAINING_WEIGHTS = {
    "both_high": 1.0,
    "clean_preference": 0.9,
    "acceptable_preference": 0.7,
    "tie_or_uncertain": 0.25,
    "low_vs_lower": 0.15,
    "both_low": 0.10,
    "conflicting": 0.05,
    "unknown": 0.20,
}


def similarity_from_shift(value):
    shift = parse_float(value)
    if shift is None:
        return None
    return clamp01(1.0 - 2.0 * float(shift))


def candidate_quality_bucket(global_similarity, retained_similarity):
    if global_similarity is None or retained_similarity is None:
        return "unknown"
    if global_similarity >= 0.98 and retained_similarity >= 0.95:
        return "High"
    if global_similarity >= 0.95 and retained_similarity >= 0.90:
        return "Acceptable"
    return "Low"


def parse_args():
    parser = argparse.ArgumentParser(description="Score deletion segments or build teacher preference artifacts.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--base_checkpoint", default=None)
    parser.add_argument("--p0", default="p0.pt")
    parser.add_argument("--segments_csv", default=None)
    parser.add_argument("--core_csv", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--summary_txt", default=None)
    parser.add_argument("--inference_strengths", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35")
    parser.add_argument("--score_aggregation", choices=["mean", "max"], default="mean")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=800)
    parser.add_argument("--limit_proteins", type=int, default=None)
    parser.add_argument("--limit_segments", type=int, default=None)
    parser.add_argument("--disable-fa", action="store_true")
    parser.add_argument("--amp_dtype", choices=["none", "bf16", "fp16"], default="none")
    parser.add_argument("--include_debug_scores", action="store_true", help="Append posterior aggregate debug columns without legacy risk fields.")
    parser.add_argument("--build_teacher_impact_scores", action="store_true")
    parser.add_argument("--deletion_perturbations_csv", default=None)
    parser.add_argument("--teacher_impact_scores_csv", default=None)
    parser.add_argument("--teacher_scoring_mode", choices=["mock", "fast", REAL_PLM_TEACHER_MODE], default="mock")
    parser.add_argument("--teacher_model", default=None)
    parser.add_argument("--limit_perturbations", type=int, default=None)
    parser.add_argument("--teacher_model_name_or_path", default=None)
    parser.add_argument("--teacher_backend", choices=SUPPORTED_TEACHER_BACKENDS, default="esm2_transformers")
    parser.add_argument("--teacher_device", choices=SUPPORTED_TEACHER_DEVICES, default="auto")
    parser.add_argument("--teacher_batch_size", type=int, default=1)
    parser.add_argument("--teacher_group_by_protein", action="store_true")
    parser.add_argument("--max_teacher_rows", type=int, default=None)
    parser.add_argument("--teacher_allow_download", action="store_true")
    parser.add_argument("--teacher_no_model_preflight", action="store_true")
    parser.add_argument("--teacher_preflight_summary_json", default=None)
    parser.add_argument("--teacher_resume", action="store_true")
    parser.add_argument("--teacher_cache_dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.build_teacher_impact_scores:
        stats = build_teacher_impact_scores(args)
        if args.teacher_no_model_preflight:
            print("Real PLM adapter preflight rows: {}".format(stats["preflight_rows"]))
            print("Teacher scoring executed: false")
            print("Teacher impact scores written: false")
            print("Wrote {}".format(args.teacher_preflight_summary_json))
        else:
            print("Teacher impact rows: {}".format(stats["total_rows"]))
            print("{} rows: {}".format(args.teacher_scoring_mode, stats[args.teacher_scoring_mode]))
            print("Wrote {}".format(args.teacher_impact_scores_csv))
    else:
        missing = []
        for field, value in [
            ("--checkpoint", args.checkpoint),
            ("--segments_csv", args.segments_csv),
            ("--core_csv", args.core_csv),
            ("--out_csv", args.out_csv),
            ("--summary_txt", args.summary_txt),
        ]:
            if not value:
                missing.append(field)
        if missing:
            raise SystemExit("missing required arguments for Stage1 segment scoring mode: {}".format(", ".join(missing)))
        score_segments(args)
