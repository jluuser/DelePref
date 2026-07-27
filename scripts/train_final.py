#!/usr/bin/env python3
"""Condition-streaming DDP trainer for the final DelePref model."""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, deque
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# Make direct execution from a source checkout independent of installed
# packages that happen to use the same top-level module names.
REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

from delepref.student_model import (
    generate_strict_budget_plans,
    head_only_state_dict,
    joint_objective_normalization_counts,
    normalized_selected_residue_energy,
    proposal_from_decoded_plan,
    proposal_condition_budget_loss,
    strict_budget_dp_decode,
    trainable_gradient_norm,
    validate_strict_plan,
)
from delepref.preference_core import candidate_from_proposal
from delepref.proposal_validation import (
    checkpoint_candidate,
    choose_checkpoint,
    evaluate_proposal_records,
    read_cold_start_references,
    read_frozen_proteins,
)
from delepref.dataset import ConditionBatchLoader
from delepref.streaming import (
    B20_SOURCE,
    BASE_SOURCE,
    DeterministicShardGroupMixtureSchedule,
    DynamicGlobalBatchPlanner,
    count_optimizer_batches,
    deterministic_shard_group_order,
    read_condition_catalog,
)
from scripts.train_student_pilot import build_delepref, proposal_loss_kwargs
from scripts import score_b20_esmc600m as teacher_driver


DEFAULT_CONFIG = REPO / "configs/deletion_preference_student.yaml"
EXPECTED_WORLD_SIZE = 2
PROPOSAL_CANDIDATE_COLUMNS = (
    "protein_id", "perturbation_id", "plan_id", "wildtype_length",
    "deleted_intervals", "num_intervals", "deleted_length",
    "deletion_fraction", "budget_bin", "generation_policy",
    "coordinate_system", "wildtype_sequence", "mutant_sequence",
    "source_stage1_row_id", "source_start", "source_end", "source_score",
    "wildtype_sha256", "mutant_sha256", "interval_plan_sha256",
    "beta", "seed", "candidate_index", "student_policy_score",
    "student_scorer_score", "metadata_json",
)


def absolute(path):
    value = Path(str(path))
    return value if value.is_absolute() else REPO / value


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("training config must be a mapping")
    return payload


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def csv_bytes(columns, rows):
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
    return buffer.getvalue().encode("utf-8")


def freeze_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError("refusing to replace different frozen artifact {}".format(path))
        return
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_esm_runtime_vendor(config):
    """Optionally prepend a vendored ESM runtime to ``sys.path``.

    Public installations normally use the installed ``esm`` package. A vendor
    directory remains supported for exact environment reproduction.
    """

    configured = str(config.get("model", {}).get("esm_runtime_vendor", "")).strip()
    if not configured:
        return None
    vendor = absolute(configured)
    if not vendor.is_dir():
        raise FileNotFoundError(str(vendor))
    value = str(vendor)
    if value not in sys.path:
        sys.path.insert(0, value)
    return vendor


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object: {}".format(path))
    return payload


def prepare_run_directories(output_dir, checkpoint_dir, *, resume):
    """Create a fresh run atomically enough to reject accidental reuse."""

    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)
    if resume:
        if not output_dir.is_dir() or not checkpoint_dir.is_dir():
            raise FileNotFoundError("resume requires existing output and checkpoint run directories")
        return
    existing = [str(path) for path in (output_dir, checkpoint_dir) if path.exists()]
    if existing:
        raise FileExistsError("fresh run directories already exist: {}".format(existing))
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    except BaseException:
        output_dir.rmdir()
        raise


def verify_frozen_asset_manifest(path, *, required_labels=()):
    """Verify every frozen file and the aggregate digest in an asset manifest."""

    path = Path(path)
    payload = read_json(path)
    assets = payload.get("assets")
    if assets is None and isinstance(payload.get("asset"), dict):
        assets = [payload["asset"]]
    if not isinstance(assets, list) or not assets:
        raise ValueError("asset manifest contains no assets: {}".format(path))
    by_label = {str(asset.get("label")): asset for asset in assets}
    missing_labels = sorted(set(required_labels) - set(by_label))
    if missing_labels:
        raise ValueError("asset manifest lacks labels: {}".format(missing_labels))
    selected = [by_label[label] for label in required_labels] if required_labels else assets
    verified = []
    for asset in selected:
        root = absolute(asset["path"]).resolve()
        expected_entries = asset.get("files")
        if not isinstance(expected_entries, list) or not expected_entries:
            raise ValueError("frozen asset has no file entries: {}".format(asset.get("label")))
        if asset.get("kind") == "file":
            actual_files = [root] if root.is_file() else []
            relative_root = root.parent
        elif asset.get("kind") == "file_collection":
            relative_root = root
            actual_files = [absolute(entry["path"]).resolve() for entry in expected_entries]
            if any(not candidate.is_file() for candidate in actual_files):
                actual_files = []
        else:
            actual_files = sorted(
                (candidate for candidate in root.rglob("*") if candidate.is_file()),
                key=lambda candidate: str(candidate),
            ) if root.is_dir() else []
            relative_root = root
        actual_relatives = [str(candidate.relative_to(relative_root)) for candidate in actual_files]
        expected_relatives = [str(entry["relative_path"]) for entry in expected_entries]
        if actual_relatives != expected_relatives:
            raise ValueError("frozen asset file set changed: {}".format(asset.get("label")))
        aggregate = hashlib.sha256()
        total_size = 0
        for file_path, entry in zip(actual_files, expected_entries):
            size = file_path.stat().st_size
            if size != int(entry["size_bytes"]):
                raise ValueError("frozen asset size changed: {}".format(file_path))
            digest = sha256_file(file_path)
            if digest != str(entry["sha256"]):
                raise ValueError("frozen asset digest changed: {}".format(file_path))
            relative = str(entry["relative_path"])
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(str(size).encode("ascii"))
            aggregate.update(b"\0")
            aggregate.update(digest.encode("ascii"))
            aggregate.update(b"\n")
            total_size += size
        if aggregate.hexdigest() != str(asset["sha256"]):
            raise ValueError("frozen asset aggregate digest changed: {}".format(asset.get("label")))
        if total_size != int(asset["size_bytes"]) or len(actual_files) != int(asset["file_count"]):
            raise ValueError("frozen asset count/size summary changed: {}".format(asset.get("label")))
        verified.append(
            {
                "label": str(asset["label"]),
                "sha256": str(asset["sha256"]),
                "file_count": len(actual_files),
                "size_bytes": total_size,
            }
        )
    runtime = payload.get("runtime")
    if runtime is not None:
        actual_runtime = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
        }
        if runtime != actual_runtime:
            raise ValueError(
                "frozen Python runtime changed: expected {}, observed {}".format(
                    runtime, actual_runtime
                )
            )
    return {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "assets": verified,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal-training", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--run-id")
    parser.add_argument("--dist-backend", default="nccl")
    parser.add_argument("--dist-timeout-seconds", type=int, default=7200)
    parser.add_argument("--skip-proposal-validation", action="store_true")
    return parser.parse_args(argv)


def preflight(
    config, *, require_runtime=False, smoke=False,
    formal_training=False, max_steps=None, verify_asset_contents=True,
):
    blockers = []
    data = config.get("data", {})
    integrity = config.get("integrity", {})
    batching = config.get("batching", {})
    training = config.get("training", {})
    policy = config.get("policy", {})
    decoder = config.get("decoder", {})
    resources = config.get("resources", {})
    paths = {
        name: absolute(data[name])
        for name in (
            "base_pair_index", "base_validation_report", "b20_pair_index",
            "b20_validation_report", "b20_supplement_audit", "condition_catalog",
            "condition_catalog_summary", "train_dev_manifest", "confirmation_lock",
            "proposal_validation_manifest", "proposal_validation_manifest_summary",
            "cold_start_reference",
        )
    }
    for name in ("formal_asset_manifest", "base_asset_manifest", "b20_asset_manifest"):
        if integrity.get(name):
            paths[name] = absolute(integrity[name])
    for name, path in paths.items():
        if not path.is_file():
            blockers.append("missing_{}".format(name))
    catalog_summary = read_json(paths["condition_catalog_summary"]) if paths["condition_catalog_summary"].is_file() else {}
    confirmation_lock = read_json(paths["confirmation_lock"]) if paths["confirmation_lock"].is_file() else {}
    proposal_manifest_summary = (
        read_json(paths["proposal_validation_manifest_summary"])
        if paths["proposal_validation_manifest_summary"].is_file()
        else {}
    )
    for name in ("base_validation_report", "b20_validation_report", "b20_supplement_audit"):
        if paths[name].is_file() and read_json(paths[name]).get("status") != "PASS":
            blockers.append("{}_not_pass".format(name))
    if catalog_summary.get("status") != "PASS":
        blockers.append("condition_catalog_not_pass")
    proposal_validation = config.get("proposal_validation", {})
    if (
        proposal_manifest_summary.get("status") != "PASS"
        or int(proposal_manifest_summary.get("protein_count", -1)) != int(proposal_validation.get("fixed_dev_proteins", -2))
        or proposal_manifest_summary.get("confirmation_manifest_opened") is not False
    ):
        blockers.append("proposal_validation_manifest_not_pass")
    if confirmation_lock.get("status") != "LOCKED_UNSEEN" or confirmation_lock.get("metrics_computed_or_viewed") is not False:
        blockers.append("confirmation_lock_not_unseen")
    if int(policy.get("qmax", -1)) != 3 or int(decoder.get("qmax", -1)) != 3:
        blockers.append("qmax_not_unified_at_3")
    if int(batching.get("base_conditions_per_cycle", -1)) != 3 or int(batching.get("b20_conditions_per_cycle", -1)) != 1:
        blockers.append("condition_mixture_not_3_to_1")
    train_counts = (catalog_summary.get("condition_counts") or {})
    base_train_count = int((train_counts.get("base") or {}).get("train", 0))
    b20_train_count = int((train_counts.get("b20") or {}).get("train", 0))
    base_slots_per_epoch = 3 * b20_train_count
    base_coverage_epochs = (
        int(math.ceil(base_train_count / float(base_slots_per_epoch)))
        if base_train_count > 0 and base_slots_per_epoch > 0
        else 0
    )
    if base_coverage_epochs < 1:
        blockers.append("invalid_base_coverage_cycle")
    elif int(training.get("formal_epochs", -1)) != base_coverage_epochs:
        blockers.append("formal_epochs_must_equal_one_complete_base_coverage_cycle")
    for name in (
        "encoder_microbatch_size", "conditions_per_rank", "pairs_per_condition",
        "gradient_accumulation_steps", "length_bucket_width",
    ):
        if int(batching.get(name, 0)) < 1:
            blockers.append("invalid_{}".format(name))
    if int(batching.get("max_tokens_per_gpu", -1)) < 0:
        blockers.append("invalid_max_tokens_per_gpu")
    vendor_value = str(config.get("model", {}).get("esm_runtime_vendor", "")).strip()
    if vendor_value:
        vendor = absolute(vendor_value)
        required_vendor_files = (
            vendor / "esm/models/esmc.py",
            vendor / "esm/layers/transformer_stack.py",
            vendor / "esm/tokenization/sequence_tokenizer.py",
        )
        if any(not path.is_file() for path in required_vendor_files):
            blockers.append("incomplete_esmc_runtime_vendor")
    teacher_snapshot = absolute(proposal_validation.get("teacher_model_snapshot", ""))
    if not (teacher_snapshot / "model.safetensors").is_file():
        blockers.append("missing_proposal_teacher_model")
    if (
        tuple(float(value) for value in proposal_validation.get("budgets", ())) != (0.10, 0.20)
        or tuple(int(value) for value in proposal_validation.get("seeds", ())) != (13, 29)
        or int(proposal_validation.get("k", -1)) != 8
    ):
        blockers.append("proposal_validation_contract_mismatch")
    output_dir = absolute(training.get("output_dir", ""))
    checkpoint_dir = absolute(training.get("checkpoint_dir", ""))
    if output_dir == checkpoint_dir:
        blockers.append("output_and_checkpoint_directories_must_differ")
    if smoke and (max_steps is None or int(max_steps) < 1):
        blockers.append("smoke_requires_positive_max_steps")
    if formal_training and max_steps is not None:
        blockers.append("formal_training_must_use_frozen_epoch_count_not_max_steps")
    if require_runtime and bool(smoke) == bool(formal_training):
        blockers.append("select_exactly_one_runtime_mode")
    if require_runtime:
        if int(os.environ.get("WORLD_SIZE", "1")) != int(resources.get("required_world_size", EXPECTED_WORLD_SIZE)):
            blockers.append("world_size_mismatch")
        required_gpus = int(resources.get("required_gpu_count", resources.get("required_world_size", EXPECTED_WORLD_SIZE)))
        if not torch.cuda.is_available() or torch.cuda.device_count() < required_gpus:
            blockers.append("insufficient_visible_cuda_devices")
    integrity_verification = []
    if verify_asset_contents and (formal_training or smoke):
        verification_specs = []
        if "formal_asset_manifest" in paths:
            verification_specs.append(("formal_asset_manifest", tuple(integrity.get("formal_asset_labels", ()))))
        if "base_asset_manifest" in paths:
            verification_specs.append(("base_asset_manifest", (str(integrity.get("base_asset_label", "")),)))
        if "b20_asset_manifest" in paths:
            verification_specs.append(("b20_asset_manifest", (str(integrity.get("b20_asset_label", "")),)))
        for name, labels in verification_specs:
            labels = tuple(label for label in labels if label)
            try:
                integrity_verification.append(
                    verify_frozen_asset_manifest(paths[name], required_labels=labels)
                )
            except (KeyError, OSError, TypeError, ValueError) as exc:
                blockers.append("{}_verification_failed:{}".format(name, exc))
    data_hashes = {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
    return {
        "schema_version": "delepref_final_streaming_preflight_v1",
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "paths": {name: str(path) for name, path in paths.items()},
        "data_hashes": data_hashes,
        "config_sha256": canonical_sha256(config),
        "catalog_condition_counts": catalog_summary.get("condition_counts", {}),
        "base_coverage": {
            "base_train_conditions": base_train_count,
            "base_slots_per_epoch": base_slots_per_epoch,
            "coverage_cycle_epochs": base_coverage_epochs,
            "formal_epochs": int(training.get("formal_epochs", -1)),
        },
        "confirmation_manifest_opened": False,
        "confirmation_metrics_viewed": False,
        "formal_training_started": False,
        "runtime_mode": "formal_training" if formal_training else "smoke" if smoke else "preflight_only",
        "integrity_verification": integrity_verification,
    }


def distributed_context(backend, timeout_seconds, expected_world_size=EXPECTED_WORLD_SIZE):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != int(expected_world_size):
        raise RuntimeError("WORLD_SIZE does not match the configured DDP world size")
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=int(timeout_seconds)),
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(local_rank)
    uuid_value = str(getattr(properties, "uuid", ""))
    identity = {
        "rank": rank,
        "local_rank": local_rank,
        "device_index": local_rank,
        "device_name": properties.name,
        "device_uuid": uuid_value,
        "total_memory": int(properties.total_memory),
    }
    identities = [None for _ in range(world_size)]
    dist.all_gather_object(identities, identity)
    if len({item["device_uuid"] for item in identities}) != world_size:
        raise RuntimeError("DDP ranks are not bound to distinct GPU UUIDs")
    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
        "identity": identity,
        "all_identities": identities,
    }


def seed_rank(seed, rank):
    value = int(seed) + int(rank) * 100003
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)


def capture_rng_state(local_rank):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(local_rank),
    }


def restore_rng_state(state, local_rank):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], local_rank)


def lr_multiplier(completed_step, total_steps, warmup_steps, min_lr_ratio):
    step = int(completed_step)
    total = max(1, int(total_steps))
    warmup = max(0, int(warmup_steps))
    minimum = float(min_lr_ratio)
    if warmup and step < warmup:
        return max(1.0e-8, float(step + 1) / float(warmup))
    progress = min(1.0, max(0.0, (step - warmup) / float(max(1, total - warmup))))
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(step, total_steps, warmup_steps, min_lr_ratio),
    )


def all_reduce_scalar(value, device, op=dist.ReduceOp.SUM):
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def normalized_joint_components(info, local_counts, global_counts, config, world_size):
    """Reconstruct exact global component means from one rank/microbatch."""

    def scaled(value, name):
        local = int(local_counts[name])
        global_count = int(global_counts[name])
        if local == 0 or global_count == 0:
            return value * 0.0
        return value * (float(world_size) * float(local) / float(global_count))

    proposal = info["proposal"]
    proposal_weights = config["policy"]
    components = {
        "scorer_loss": scaled(info["scorer"]["loss"], "scorer"),
        "rank_loss": scaled(proposal["rank_loss"], "rank"),
        "tie_loss": scaled(proposal["tie_loss"], "tie"),
        "imitation_loss": scaled(proposal["imitation_loss"], "imitation"),
        "budget_loss": scaled(proposal["budget_aux_loss"], "budget"),
        "structure_loss": scaled(proposal["structure_aux_loss"], "structure"),
    }
    proposal_total = (
        float(proposal_weights["rank_weight"])
        * (components["rank_loss"] + components["tie_loss"])
        + float(proposal_weights["imitation_weight"]) * components["imitation_loss"]
        + float(proposal_weights["budget_aux_weight"]) * components["budget_loss"]
        + float(proposal_weights["structure_aux_weight"]) * components["structure_loss"]
    )
    components["proposal_loss"] = proposal_total
    components["loss"] = (
        float(config["training"]["scorer_loss_weight"]) * components["scorer_loss"]
        + float(config["training"]["proposal_loss_weight"]) * proposal_total
    )
    return components


def encoder_padding_counts(encoded_wildtypes, microbatch_size):
    lengths = [len(sequence) for sequence in encoded_wildtypes.sequences.values()]
    actual = sum(lengths)
    padded = 0
    width = max(1, int(microbatch_size))
    for start in range(0, len(lengths), width):
        chunk = lengths[start:start + width]
        padded += max(chunk) * len(chunk)
    return actual, padded


def gpu_utilization(device_identifier):
    identifier = str(device_identifier)
    if identifier and not identifier.isdigit() and not identifier.startswith("GPU-"):
        identifier = "GPU-" + identifier
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi", "-i", identifier,
                "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        return float(output.strip().splitlines()[0])
    except Exception:
        return None


def binary_auc(scores, labels, weights):
    positive = sum(weight for label, weight in zip(labels, weights) if label == 1)
    negative = sum(weight for label, weight in zip(labels, weights) if label == 0)
    if positive <= 0.0 or negative <= 0.0:
        return None
    ordered = sorted(zip(scores, labels, weights), key=lambda item: item[0])
    concordant = 0.0
    negative_before = 0.0
    index = 0
    while index < len(ordered):
        end = index
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_negative = sum(weight for _score, label, weight in ordered[index:end] if label == 0)
        group_positive = sum(weight for _score, label, weight in ordered[index:end] if label == 1)
        concordant += group_positive * (negative_before + 0.5 * group_negative)
        negative_before += group_negative
        index = end
    return concordant / (positive * negative)


def fixed_dev_references(dev_base, dev_b20, config, seed):
    validation = config["cheap_validation"]
    total = int(validation["fixed_dev_conditions"])
    b20_count = max(1, total // 4)
    base_count = total - b20_count
    width = int(config["batching"]["length_bucket_width"])
    ordered_base = deterministic_shard_group_order(
        dev_base, seed=seed, epoch=0, source=BASE_SOURCE, length_bucket_width=width
    )
    ordered_b20 = deterministic_shard_group_order(
        dev_b20, seed=seed, epoch=0, source=B20_SOURCE, length_bucket_width=width
    )
    return tuple(ordered_base[:base_count] + ordered_b20[:b20_count])


def apply_deletion(sequence, intervals):
    pieces = []
    cursor = 0
    for start, end in intervals:
        pieces.append(sequence[cursor:int(start)])
        cursor = int(end) + 1
    pieces.append(sequence[cursor:])
    return "".join(pieces)


def proposal_teacher_scoring_config(config, manifest_sha256):
    proposal = config["proposal_validation"]
    model_path = Path(str(proposal["teacher_model_snapshot"])) / "model.safetensors"
    payload = {
        "schema_version": "delepref_final_proposal_teacher_scoring_config_v1",
        "teacher_backend": teacher_driver.scorer.ESMC_OFFICIAL_SAFETENSORS_BACKEND,
        "teacher_scoring_mode": teacher_driver.scorer.REAL_PLM_TEACHER_MODE,
        "teacher_model_snapshot": str(Path(str(proposal["teacher_model_snapshot"])).resolve()),
        "teacher_model_sha256": sha256_file(model_path),
        "dtype": "torch.bfloat16",
        "mutant_batch_size": int(proposal["teacher_mutant_batch_size"]),
        "global_metric": "pooled_residue_embedding_cosine_similarity",
        "retained_metric": "mean_retained_residue_cosine_similarity",
        "junction_metric": "wt_mutant_boundary_anchor_cosine_similarity_proxy",
        "teacher_utility": "0.35*global_similarity+0.45*retained_similarity+0.20*junction_plausibility",
        "risk_input": "absent",
        "proposal_manifest_sha256": str(manifest_sha256),
        "backend_implementation_sha256": sha256_file(teacher_driver.SCORER_PATH),
        "driver_implementation_sha256": sha256_file(Path(teacher_driver.__file__).resolve()),
        "cache_reuse_requires_exact_identity": True,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def generate_proposal_candidate_rows(model, proteins, config):
    proposal_config = config["proposal_validation"]
    policy_config = config["policy"]
    decoder_config = config["decoder"]
    budgets = tuple(float(value) for value in proposal_config["budgets"])
    seeds = tuple(int(value) for value in proposal_config["seeds"])
    requested_k = int(proposal_config["k"])
    qmax = int(policy_config["qmax"])
    min_span = int(policy_config["min_span_len"])
    max_fraction = float(policy_config["max_span_len_fraction"])
    was_training = model.training
    model.eval()
    rows = []
    with torch.no_grad():
        encoded = model.encode_unique_wildtypes(
            [(item["protein_id"], item["wildtype_sequence"]) for item in proteins],
            microbatch_size=int(config["batching"]["encoder_microbatch_size"]),
        )
        for protein in proteins:
            protein_id = protein["protein_id"]
            sequence = protein["wildtype_sequence"]
            sequence_hash = protein["sequence_sha256"]
            embedding = encoded.embeddings[encoded.key_for(protein_id, sequence)]
            policy_outputs = model.propose_from_embeddings(
                [embedding for _ in budgets], list(budgets)
            )
            for beta, policy_output in zip(budgets, policy_outputs):
                max_span = max(min_span, min(len(sequence) - 1, int(math.floor(max_fraction * len(sequence)))))
                for seed in seeds:
                    generated = generate_strict_budget_plans(
                        policy_output.residue_logits,
                        beta,
                        num_candidates=requested_k,
                        qmax=qmax,
                        min_span_len=min_span,
                        max_span_len=max_span,
                        seed=seed,
                        perturbation_scale=float(decoder_config["perturbation_scale"]),
                        max_attempts=requested_k * int(decoder_config["max_attempts_multiplier"]),
                    )
                    if len(generated.plans) != requested_k:
                        raise RuntimeError(
                            "proposal validation generated {}/{} unique plans for protein={} beta={} seed={} reason={}".format(
                                len(generated.plans), requested_k, protein_id, beta, seed, generated.reason
                            )
                        )
                    proposals = []
                    for candidate_index, decoded in enumerate(generated.plans, start=1):
                        validate_strict_plan(
                            len(sequence), beta, decoded.deleted_intervals,
                            qmax=qmax, min_span_len=min_span, max_span_len=max_span,
                        )
                        proposals.append(
                            proposal_from_decoded_plan(
                                protein_id=protein_id,
                                wildtype_sequence=sequence,
                                beta=beta,
                                decoded=decoded,
                                rank=candidate_index,
                                generation_reason=generated.reason,
                            )
                        )
                    candidates = [candidate_from_proposal(item) for item in proposals]
                    scorer_scores = model.score_plans_from_embeddings(candidates, encoded)[
                        "preference_score"
                    ].detach().float().cpu().tolist()
                    for candidate_index, (proposal, scorer_score) in enumerate(
                        zip(proposals, scorer_scores), start=1
                    ):
                        candidate_id = "{}__proposalval__b{:02d}__s{:02d}__k{:02d}".format(
                            protein_id, int(round(beta * 100)), seed, candidate_index
                        )
                        intervals_json = json.dumps(
                            [list(interval) for interval in proposal.deleted_intervals],
                            separators=(",", ":"),
                        )
                        mutant = apply_deletion(sequence, proposal.deleted_intervals)
                        rows.append(
                            {
                                "protein_id": protein_id,
                                "perturbation_id": candidate_id,
                                "plan_id": candidate_id,
                                "wildtype_length": len(sequence),
                                "deleted_intervals": intervals_json,
                                "num_intervals": len(proposal.deleted_intervals),
                                "deleted_length": proposal.deleted_length,
                                "deletion_fraction": "{:.10g}".format(proposal.deletion_fraction),
                                "budget_bin": "proposal_validation_b{:02d}".format(int(round(beta * 100))),
                                "generation_policy": "delepref_seeded_perturbation_strict_dp",
                                "coordinate_system": "0_based_closed_inclusive",
                                "wildtype_sequence": sequence,
                                "mutant_sequence": mutant,
                                "source_stage1_row_id": "",
                                "source_start": "",
                                "source_end": "",
                                "source_score": "",
                                "wildtype_sha256": sequence_hash,
                                "mutant_sha256": hashlib.sha256(mutant.encode("utf-8")).hexdigest(),
                                "interval_plan_sha256": hashlib.sha256(intervals_json.encode("utf-8")).hexdigest(),
                                "beta": "{:.2f}".format(beta),
                                "seed": seed,
                                "candidate_index": candidate_index,
                                "student_policy_score": "{:.10g}".format(proposal.metadata["raw_policy_score"] if "raw_policy_score" in proposal.metadata else proposal.proposal_logprob),
                                "student_scorer_score": "{:.10g}".format(float(scorer_score)),
                                "metadata_json": json.dumps(
                                    {
                                        "requested_k": requested_k,
                                        "returned_k": len(generated.plans),
                                        "generation_attempts": generated.attempts,
                                        "duplicate_decodes": generated.duplicate_count,
                                        "generation_reason": generated.reason,
                                        "student_confidence_available": False,
                                        "student_confidence_used_for_ranking": False,
                                    },
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ),
                            }
                        )
    if was_training:
        model.train()
    return rows


def score_proposal_candidates_real_teacher(rows, config, scoring_config, output_path):
    model_sha256 = scoring_config["teacher_model_sha256"]
    if Path(output_path).exists():
        teacher_driver.validate_cached_output(
            output_path, rows, model_sha256, scoring_config["sha256"]
        )
        return teacher_driver.read_csv(output_path)[0], "resumed_strict_cache_match"
    proposal = config["proposal_validation"]
    args = SimpleNamespace(
        model_snapshot=Path(str(proposal["teacher_model_snapshot"])),
        protein_batch_size=1,
    )
    score_args = teacher_driver.backend_args(args)
    backend = teacher_driver.scorer.load_real_plm_teacher_backend(score_args)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["protein_id"], []).append(row)
    impacts = []
    try:
        for protein_id in sorted(grouped):
            protein_rows = grouped[protein_id]
            wildtypes = {row["wildtype_sequence"] for row in protein_rows}
            if len(wildtypes) != 1:
                raise ValueError("proposal validation has inconsistent wild type for {}".format(protein_id))
            wildtype = next(iter(wildtypes))
            wt_embedding = teacher_driver.scorer.residue_embeddings_for_sequences(
                backend, [wildtype], batch_size=1
            )[0]
            by_condition = {}
            for row in protein_rows:
                by_condition.setdefault((float(row["beta"]), int(row["seed"])), []).append(row)
            for condition in sorted(by_condition):
                condition_rows = sorted(by_condition[condition], key=lambda item: int(item["candidate_index"]))
                mutant_embeddings = teacher_driver.scorer.residue_embeddings_for_sequences(
                    backend,
                    [row["mutant_sequence"] for row in condition_rows],
                    batch_size=int(proposal["teacher_mutant_batch_size"]),
                )
                for row, mutant_embedding in zip(condition_rows, mutant_embeddings):
                    impact = teacher_driver.scorer.real_plm_teacher_impact_from_embeddings(
                        row, score_args, backend, wt_embedding, mutant_embedding
                    )
                    metadata = json.loads(impact["metadata_json"])
                    metadata.update(
                        {
                            "strict_cache_identity": teacher_driver.strict_identity(
                                row, model_sha256, scoring_config["sha256"]
                            ),
                            "teacher_model_sha256": model_sha256,
                            "teacher_scoring_config_sha256": scoring_config["sha256"],
                            "cache_reuse_requires_exact_identity": True,
                            "validation_role": "checkpoint_selection_real_teacher_rescore",
                        }
                    )
                    impact["metadata_json"] = json.dumps(
                        metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                    )
                    impacts.append(impact)
            del wt_embedding
        if len(impacts) != len(rows) or any(row.get("scoring_status") != "ok" for row in impacts):
            raise RuntimeError("real proposal teacher did not return one successful impact per candidate")
        freeze_bytes(output_path, csv_bytes(teacher_driver.IMPACT_COLUMNS, impacts))
        teacher_driver.validate_cached_output(
            output_path, rows, model_sha256, scoring_config["sha256"]
        )
        return impacts, "written_real_esmc600m"
    finally:
        backend.clear()
        del backend
        gc.collect()
        torch.cuda.empty_cache()


def run_real_proposal_validation(model, config, preflight_report, output_dir, global_step):
    data = config["data"]
    proposal = config["proposal_validation"]
    manifest_path = absolute(data["proposal_validation_manifest"])
    cold_path = absolute(data["cold_start_reference"])
    proteins = read_frozen_proteins(
        manifest_path, expected_count=int(proposal["fixed_dev_proteins"])
    )
    cold_references = read_cold_start_references(
        cold_path,
        expected_proteins=int(proposal["fixed_dev_proteins"]),
        budgets=proposal["budgets"],
    )
    step_dir = Path(output_dir) / "proposal_validation/step_{:08d}".format(global_step)
    candidates_path = step_dir / "student_candidates.csv"
    impacts_path = step_dir / "teacher_impacts.csv"
    scoring_config_path = step_dir / "teacher_scoring_config.json"
    report_path = step_dir / "report.json"
    rows = generate_proposal_candidate_rows(model, proteins, config)
    freeze_bytes(candidates_path, csv_bytes(PROPOSAL_CANDIDATE_COLUMNS, rows))
    scoring_config = proposal_teacher_scoring_config(
        config, preflight_report["data_hashes"]["proposal_validation_manifest"]
    )
    freeze_bytes(
        scoring_config_path,
        (json.dumps(scoring_config, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    impacts, cache_status = score_proposal_candidates_real_teacher(
        rows, config, scoring_config, impacts_path
    )
    impact_by_id = {
        (row["protein_id"], row["perturbation_id"], row["plan_id"]): row for row in impacts
    }
    evaluation_records = []
    for row in rows:
        impact = impact_by_id[(row["protein_id"], row["perturbation_id"], row["plan_id"])]
        evaluation_records.append(
            {
                "protein_id": row["protein_id"],
                "sequence_sha256": row["wildtype_sha256"],
                "wildtype_length": int(row["wildtype_length"]),
                "beta": float(row["beta"]),
                "seed": int(row["seed"]),
                "candidate_id": row["perturbation_id"],
                "deleted_intervals": row["deleted_intervals"],
                "student_scorer_score": float(row["student_scorer_score"]),
                "teacher_scoring_status": impact["scoring_status"],
                "global_similarity": float(impact["global_similarity"]),
                "retained_similarity": float(impact["retained_similarity"]),
                "junction_plausibility": float(impact["junction_plausibility"]),
            }
        )
    report = evaluate_proposal_records(
        evaluation_records,
        protein_ids=[row["protein_id"] for row in proteins],
        budgets=proposal["budgets"],
        seeds=proposal["seeds"],
        requested_k=int(proposal["k"]),
        qmax=int(config["policy"]["qmax"]),
        cold_start_references=cold_references,
    )
    report.update(
        {
            "step": int(global_step),
            "teacher_model_sha256": scoring_config["teacher_model_sha256"],
            "teacher_scoring_config_sha256": scoring_config["sha256"],
            "teacher_cache_status": cache_status,
            "artifacts": {
                "student_candidates": {"path": str(candidates_path), "sha256": sha256_file(candidates_path)},
                "teacher_impacts": {"path": str(impacts_path), "sha256": sha256_file(impacts_path)},
                "teacher_scoring_config": {"path": str(scoring_config_path), "sha256": sha256_file(scoring_config_path)},
            },
            "confirmation_manifest_opened": False,
            "confirmation_metrics_viewed": False,
        }
    )
    atomic_json(report_path, report)
    return report


def update_checkpoint_selection(output_dir, candidate, tie_tolerance):
    path = Path(output_dir) / "checkpoint_selection.json"
    if path.exists():
        payload = read_json(path)
    else:
        payload = {
            "schema_version": "delepref_checkpoint_selection_v1",
            "selection_rule": candidate["selection_rule"],
            "candidates": [],
        }
    existing = next(
        (item for item in payload["candidates"] if int(item["step"]) == int(candidate["step"])),
        None,
    )
    if existing is not None and existing != candidate:
        raise RuntimeError("checkpoint selection candidate changed for an existing step")
    if existing is None:
        payload["candidates"].append(candidate)
    selected = choose_checkpoint(payload["candidates"], tie_tolerance=float(tie_tolerance))
    if selected is None:
        raise RuntimeError("no checkpoint passes proposal legality gates")
    payload["selected"] = selected
    payload["status"] = "PASS"
    payload["confirmation_metrics_used"] = False
    atomic_json(path, payload)
    return payload


def complete_proposal_validation(
    model,
    config,
    preflight_report,
    output_dir,
    global_step,
    checkpoint_path,
    latest_cheap_validation,
    *,
    rank,
):
    """Run or replay one checkpoint-selection validation on all DDP ranks."""

    if int(latest_cheap_validation.get("step", -1)) != int(global_step):
        raise RuntimeError("proposal validation requires same-step cheap validation")
    if latest_cheap_validation.get("status") != "PASS":
        raise RuntimeError("proposal validation requires passing cheap validation")
    dist.barrier()
    proposal_report = None
    if rank == 0:
        proposal_report = run_real_proposal_validation(
            model, config, preflight_report, output_dir, global_step
        )
        selection_candidate = checkpoint_candidate(
            checkpoint_path=str(checkpoint_path),
            step=global_step,
            proposal_report=proposal_report,
            strict_pair_accuracy=float(latest_cheap_validation["strict_scorer_accuracy"]),
        )
        selection = update_checkpoint_selection(
            output_dir,
            selection_candidate,
            tie_tolerance=float(config["checkpoint_selection"]["tie_tolerance"]),
        )
        atomic_json(
            Path(output_dir)
            / "proposal_validation/step_{:08d}/completion.json".format(global_step),
            {
                "schema_version": "proposal_validation_completion_v1",
                "status": proposal_report["status"],
                "step": int(global_step),
                "checkpoint_path": str(checkpoint_path),
                "report_sha256": sha256_file(
                    Path(output_dir)
                    / "proposal_validation/step_{:08d}/report.json".format(global_step)
                ),
                "checkpoint_selection_sha256": sha256_file(
                    Path(output_dir) / "checkpoint_selection.json"
                ),
                "selected_step": int(selection["selected"]["step"]),
            },
        )
    status_box = [proposal_report["status"] if rank == 0 else None]
    dist.broadcast_object_list(status_box, src=0)
    if status_box[0] != "PASS":
        raise RuntimeError("real-teacher proposal validation gate failed")
    dist.barrier()
    return proposal_report


def cheap_validation(model, loader, references, config, device, training_gradient_norms):
    was_training = model.training
    model.eval()
    loaded = loader.load(references, epoch=0)
    pairs = list(loaded.pairs)
    with torch.no_grad():
        encoded = model.encode_unique_wildtypes(
            [candidate for pair in pairs for candidate in (pair.a, pair.b)],
            microbatch_size=int(config["batching"]["encoder_microbatch_size"]),
        )
        scorer_scores = model.score_plans_from_embeddings(
            [candidate for pair in pairs for candidate in (pair.a, pair.b)], encoded
        )["preference_score"]
    policy_outputs = {}
    scorer_correct = policy_correct = weight_total = 0.0
    auc_scores = []
    auc_labels = []
    auc_weights = []
    scorer_ties = []
    policy_ties = []
    budget_errors = []
    count_errors = []
    with torch.no_grad():
        for index, pair in enumerate(pairs):
            condition_id = pair.metadata["condition_id"]
            output = policy_outputs.get(condition_id)
            if output is None:
                output = model.propose_from_embeddings(
                    [encoded.for_candidate(pair.a)], [pair.a.target_delete_fraction]
                )[0]
                policy_outputs[condition_id] = output
                budget_errors.append(float(proposal_condition_budget_loss(output).sqrt().cpu()))
                try:
                    intervals, _score = strict_budget_dp_decode(
                        output.residue_logits,
                        pair.a.target_delete_fraction,
                        qmax=3,
                    )
                    decoded_count = sum(end - start + 1 for start, end in intervals)
                    count_errors.append(abs(decoded_count - pair.a.deleted_length))
                except Exception:
                    count_errors.append(float("inf"))
            energy_a = normalized_selected_residue_energy(model.policy, pair.a, output)[0]
            energy_b = normalized_selected_residue_energy(model.policy, pair.b, output)[0]
            scorer_delta = float((scorer_scores[2 * index] - scorer_scores[2 * index + 1]).cpu())
            policy_delta = float((energy_a - energy_b).cpu())
            preference = str(pair.preference).lower()
            weight = float(pair.loss_weight)
            if preference == "a_better":
                label = 1
                scorer_correct += weight * float(scorer_delta > 0.0)
                policy_correct += weight * float(policy_delta > 0.0)
            elif preference == "b_better":
                label = 0
                scorer_correct += weight * float(scorer_delta < 0.0)
                policy_correct += weight * float(policy_delta < 0.0)
            elif preference == "tie":
                scorer_ties.append(abs(scorer_delta))
                policy_ties.append(abs(policy_delta))
                continue
            else:
                continue
            weight_total += weight
            auc_scores.append(scorer_delta)
            auc_labels.append(label)
            auc_weights.append(weight)
    gradient_pairs = pairs[: min(len(pairs), 32)]
    with torch.no_grad():
        gradient_info = model.joint_objective(
            gradient_pairs,
            encoder_microbatch_size=int(config["batching"]["encoder_microbatch_size"]),
            scorer_tie_weight_scale=float(config["training"]["scorer_tie_weight_scale"]),
            proposal_kwargs=proposal_loss_kwargs(config),
            scorer_loss_weight=float(config["training"]["scorer_loss_weight"]),
            proposal_loss_weight=float(config["training"]["proposal_loss_weight"]),
        )
    scorer_gradient = float(training_gradient_norms["scorer"])
    policy_gradient = float(training_gradient_norms["policy"])
    if was_training:
        model.train()
    return {
        "schema_version": "delepref_cheap_validation_v1",
        "status": "PASS" if all(math.isfinite(value) for value in (scorer_gradient, policy_gradient)) and not any(math.isinf(value) for value in count_errors) else "FAIL",
        "conditions": len(references),
        "pairs": len(pairs),
        "strict_scorer_accuracy": scorer_correct / max(1.0e-12, weight_total),
        "strict_scorer_auc": binary_auc(auc_scores, auc_labels, auc_weights),
        "policy_energy_pair_accuracy": policy_correct / max(1.0e-12, weight_total),
        "mean_scorer_tie_delta": sum(scorer_ties) / max(1, len(scorer_ties)),
        "mean_policy_tie_delta": sum(policy_ties) / max(1, len(policy_ties)),
        "mean_budget_fraction_error": sum(budget_errors) / max(1, len(budget_errors)),
        "max_decoded_count_error": max(count_errors) if count_errors else 0,
        "losses": {
            "joint": float(gradient_info["loss"].detach().cpu()),
            "scorer": float(gradient_info["scorer"]["loss"].detach().cpu()),
            "proposal": float(gradient_info["proposal"]["loss"].detach().cpu()),
        },
        "gradient_norms": {"scorer": scorer_gradient, "policy": policy_gradient},
    }


def capture_rank_state(context, planner):
    return {
        "rank": context["rank"],
        "device_uuid": context["identity"]["device_uuid"],
        "planner": planner.state_dict(),
        "rng": capture_rng_state(context["local_rank"]),
    }


def save_checkpoint(
    path,
    *,
    ddp_model,
    optimizer,
    scheduler,
    context,
    planner,
    global_step,
    total_steps,
    epoch,
    config,
    preflight_report,
    run_id,
    run_mode,
    recent_metrics,
    interrupted,
    pending_validation_step=None,
):
    local_state = capture_rank_state(context, planner)
    rank_states = [None for _ in range(context["world_size"])]
    dist.all_gather_object(rank_states, local_state)
    cursors = {state["planner"]["global_cursor"] for state in rank_states}
    if len(cursors) != 1:
        raise RuntimeError("DDP ranks disagree on sampler cursor at checkpoint")
    if context["rank"] == 0:
        model = ddp_model.module
        payload = {
            "schema_version": "delepref_final_streaming_checkpoint_v1",
            "student_version": "delepref",
            "run_id": run_id,
            "run_mode": str(run_mode),
            "global_step": int(global_step),
            "total_steps": int(total_steps),
            "epoch": int(epoch),
            "student_head_state_dict": head_only_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rank_states": rank_states,
            "config": config,
            "config_sha256": preflight_report["config_sha256"],
            "data_hashes": preflight_report["data_hashes"],
            "recent_metrics": list(recent_metrics),
            "interrupted": bool(interrupted),
            "pending_validation_step": (
                None if pending_validation_step is None else int(pending_validation_step)
            ),
            "encoder_weights_saved": False,
            "confirmation_manifest_opened": False,
            "confirmation_metrics_viewed": False,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError("refusing to replace existing checkpoint {}".format(path))
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(str(temporary), str(path))
        latest = path.parent / "checkpoint_latest.pt"
        latest_tmp = latest.with_name(latest.name + ".tmp")
        torch.save(payload, latest_tmp)
        os.replace(str(latest_tmp), str(latest))
    dist.barrier()


def load_checkpoint(
    path, *, model, optimizer, scheduler, context, config_sha256,
    data_hashes, run_mode, total_steps,
):
    # PyTorch 2.6 defaults weights_only=True, but this trainer checkpoint also
    # contains Python/NumPy RNG and sampler state.  Callers restrict --resume
    # to this run's corrected checkpoint root before this trusted full load.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with PyTorch releases before this kwarg.
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != "delepref_final_streaming_checkpoint_v1":
        raise ValueError("not a DelePref final streaming checkpoint")
    if payload.get("config_sha256") != config_sha256 or payload.get("data_hashes") != data_hashes:
        raise ValueError("checkpoint config/data hash mismatch")
    if payload.get("run_mode") != str(run_mode):
        raise ValueError("checkpoint runtime mode mismatch")
    if int(payload.get("total_steps", -1)) != int(total_steps):
        raise ValueError("checkpoint total-step schedule mismatch")
    missing, unexpected = model.load_state_dict(payload["student_head_state_dict"], strict=False)
    non_encoder_missing = [name for name in missing if not name.startswith("encoder.")]
    if non_encoder_missing or unexpected:
        raise RuntimeError("checkpoint head mismatch")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    rank_states = payload.get("rank_states") or []
    state = next((item for item in rank_states if int(item["rank"]) == context["rank"]), None)
    if state is None:
        raise ValueError("checkpoint lacks this rank state")
    saved_ranks = {int(item["rank"]) for item in rank_states}
    if saved_ranks != set(range(context["world_size"])):
        raise ValueError("checkpoint rank-state world size mismatch")
    saved_uuids = [str(item.get("device_uuid") or "") for item in rank_states]
    if len(set(saved_uuids)) != context["world_size"]:
        raise ValueError("checkpoint did not record distinct rank GPU UUIDs")
    restore_rng_state(state["rng"], context["local_rank"])
    return payload, state["planner"]


def model_parameter_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(head_only_state_dict(model).items()):
        digest.update(name.encode("utf-8"))
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def write_failure_artifact(exc, argv=None):
    """Best-effort per-rank failure evidence; never retries or shrinks a batch."""

    try:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", default=str(DEFAULT_CONFIG))
        parser.add_argument("--formal-training", action="store_true")
        parsed, _unknown = parser.parse_known_args(argv)
        config = load_config(parsed.config)
        output_dir = absolute(config.get("training", {}).get("output_dir", ""))
        if not str(output_dir):
            return None
        rank = int(os.environ.get("RANK", "0"))
        path = output_dir / "failures/failure_rank_{:02d}_pid_{}.json".format(rank, os.getpid())
        message = str(exc)
        payload = {
            "schema_version": "delepref_final_streaming_failure_v1",
            "status": "FAIL",
            "exception_type": type(exc).__name__,
            "message": message,
            "traceback": traceback.format_exc(),
            "rank": rank,
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "node": os.uname().nodename,
            "cuda_oom": isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message.lower(),
            "automatic_batch_shrink_attempted": False,
            "automatic_retry_attempted": False,
            "formal_training": bool(parsed.formal_training),
            "confirmation_manifest_opened": False,
            "confirmation_metrics_viewed": False,
        }
        atomic_json(path, payload)
        return path
    except Exception:
        return None


def main(argv=None):
    args = parse_args(argv)
    config = load_config(args.config)
    preflight_report = preflight(
        config,
        require_runtime=not args.preflight_only,
        smoke=args.smoke,
        formal_training=args.formal_training,
        max_steps=args.max_steps,
        verify_asset_contents=args.preflight_only,
    )
    if args.preflight_only:
        print(json.dumps(preflight_report, indent=2, sort_keys=True))
        return 0 if preflight_report["status"] == "PASS" else 2
    if preflight_report["status"] != "PASS":
        raise RuntimeError("preflight blocked: {}".format(preflight_report["blockers"]))
    if bool(args.smoke) == bool(args.formal_training):
        raise RuntimeError("select exactly one of --smoke or --formal-training")
    if args.formal_training and args.skip_proposal_validation:
        raise RuntimeError("formal training cannot skip real-teacher proposal validation")
    run_mode = "formal_training" if args.formal_training else "smoke"

    ensure_esm_runtime_vendor(config)
    expected_world_size = int(config.get("resources", {}).get("required_world_size", EXPECTED_WORLD_SIZE))
    context = distributed_context(
        args.dist_backend,
        args.dist_timeout_seconds,
        expected_world_size=expected_world_size,
    )
    rank = context["rank"]
    device = context["device"]
    verified_preflight_box = [None]
    if rank == 0:
        verified_preflight_box[0] = preflight(
            config,
            require_runtime=True,
            smoke=args.smoke,
            formal_training=args.formal_training,
            max_steps=args.max_steps,
            verify_asset_contents=True,
        )
    dist.broadcast_object_list(verified_preflight_box, src=0)
    preflight_report = verified_preflight_box[0]
    if preflight_report["status"] != "PASS":
        raise RuntimeError(
            "verified preflight blocked: {}".format(preflight_report["blockers"])
        )
    training = config["training"]
    batching = config["batching"]
    data = config["data"]
    seed_rank(int(training["seed"]), rank)
    requested_run_id = None
    if rank == 0:
        if args.resume:
            inferred_run_id = Path(args.resume).resolve().parent.name
            if args.run_id and str(args.run_id) != inferred_run_id:
                raise ValueError("--run-id does not match the resume checkpoint directory")
            requested_run_id = inferred_run_id
        else:
            requested_run_id = args.run_id or str(uuid.uuid4())
    run_id_box = [requested_run_id]
    dist.broadcast_object_list(run_id_box, src=0)
    run_id = str(run_id_box[0])
    output_root = absolute(training["output_dir"])
    checkpoint_root = absolute(training["checkpoint_dir"])
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if resume_path.parent.parent != checkpoint_root.resolve():
            raise ValueError("--resume must point inside the corrected checkpoint root")
        if resume_path.suffix != ".pt" or not resume_path.name.startswith("checkpoint_"):
            raise ValueError("--resume must point to a DelePref checkpoint_*.pt file")
    output_dir = output_root / run_id
    checkpoint_dir = checkpoint_root / run_id
    directory_error_box = [None]
    if rank == 0:
        try:
            prepare_run_directories(
                output_dir, checkpoint_dir, resume=bool(args.resume)
            )
        except BaseException as exc:
            directory_error_box[0] = "{}: {}".format(type(exc).__name__, exc)
    dist.broadcast_object_list(directory_error_box, src=0)
    if directory_error_box[0] is not None:
        raise RuntimeError("run directory safety gate failed: {}".format(directory_error_box[0]))
    if rank == 0:
        atomic_json(output_dir / "preflight.json", preflight_report)
        atomic_json(
            output_dir / "gpu_runtime_audit.json",
            {
                "schema_version": "delepref_gpu_runtime_audit_v1",
                "status": "PASS",
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.uname().nodename.split(".")[0],
                "world_size": context["world_size"],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "rank_gpu_identities": context["all_identities"],
                "distinct_gpu_uuids": len(
                    {item["device_uuid"] for item in context["all_identities"]}
                ) == context["world_size"],
            },
        )
    dist.barrier()

    catalog_path = absolute(data["condition_catalog"])
    catalog_sha = sha256_file(catalog_path)
    train_refs = read_condition_catalog(catalog_path, split="train")
    dev_refs = read_condition_catalog(catalog_path, split="dev")
    base_train = [item for item in train_refs if item.source == BASE_SOURCE]
    b20_train = [item for item in train_refs if item.source == B20_SOURCE]
    base_dev = [item for item in dev_refs if item.source == BASE_SOURCE]
    b20_dev = [item for item in dev_refs if item.source == B20_SOURCE]
    del train_refs, dev_refs
    loader = ConditionBatchLoader(
        base_pair_index=absolute(data["base_pair_index"]),
        b20_pair_index=absolute(data["b20_pair_index"]),
        pairs_per_condition=int(batching["pairs_per_condition"]),
        seed=int(training["seed"]),
        cache_shards=2,
    )
    validation_loader = ConditionBatchLoader(
        base_pair_index=absolute(data["base_pair_index"]),
        b20_pair_index=absolute(data["b20_pair_index"]),
        pairs_per_condition=int(config["cheap_validation"]["pairs_per_condition"]),
        seed=int(config["cheap_validation"]["selection_seed"]),
        cache_shards=2,
    )
    dev_fixed = fixed_dev_references(base_dev, b20_dev, config, int(config["cheap_validation"]["selection_seed"]))

    epoch_step_counts = None
    if args.formal_training:
        counts_box = [None]
        if rank == 0:
            counts = []
            accumulation = int(batching["gradient_accumulation_steps"])
            for schedule_epoch in range(int(training["formal_epochs"])):
                counting_schedule = DeterministicShardGroupMixtureSchedule(
                    base_train,
                    b20_train,
                    seed=int(training["seed"]),
                    epoch=schedule_epoch,
                    length_bucket_width=int(batching["length_bucket_width"]),
                    catalog_sha256=catalog_sha,
                )
                microbatches = count_optimizer_batches(
                    counting_schedule,
                    world_size=context["world_size"],
                    conditions_per_rank=int(batching["conditions_per_rank"]),
                    max_tokens_per_gpu=int(batching["max_tokens_per_gpu"]),
                )
                counts.append(int(math.ceil(microbatches / float(accumulation))))
                del counting_schedule
                gc.collect()
            counts_box[0] = counts
        dist.broadcast_object_list(counts_box, src=0)
        epoch_step_counts = tuple(int(value) for value in counts_box[0])
        total_steps = sum(epoch_step_counts)
        max_steps = total_steps
    else:
        max_steps = int(args.max_steps)
        total_steps = max_steps
    if total_steps < 1:
        raise RuntimeError("training schedule has no optimizer steps")
    schedule_report = {
        "schema_version": "delepref_final_training_schedule_v1",
        "run_mode": run_mode,
        "total_optimizer_steps": total_steps,
        "formal_epochs": int(training["formal_epochs"]) if args.formal_training else None,
        "optimizer_steps_per_epoch": list(epoch_step_counts) if epoch_step_counts is not None else None,
        "gradient_accumulation_steps": int(batching["gradient_accumulation_steps"]),
        "conditions_per_rank": int(batching["conditions_per_rank"]),
        "world_size": context["world_size"],
        "base_catalog_conditions": len(base_train),
        "base_conditions_per_epoch": 3 * len(b20_train),
        "base_coverage_cycle_epochs": int(
            math.ceil(len(base_train) / float(3 * len(b20_train)))
        ),
    }
    if rank == 0:
        atomic_json(output_dir / "training_schedule.json", schedule_report)
    dist.barrier()
    warmup_steps = int(round(float(training["warmup_ratio"]) * total_steps))
    model = build_delepref(config, device)
    ddp_model = DistributedDataParallel(
        model,
        device_ids=[context["local_rank"]],
        output_device=context["local_rank"],
        broadcast_buffers=False,
        find_unused_parameters=bool(training["ddp_find_unused_parameters"]),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=float(training["min_lr_ratio"]),
    )
    epoch = 0
    global_step = 0
    resume_planner_state = None
    recent_metrics = deque(maxlen=20)
    last_checkpoint_step = -1
    latest_cheap_validation = None
    pending_validation_step = None
    if args.resume:
        payload, resume_planner_state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            context=context,
            config_sha256=preflight_report["config_sha256"],
            data_hashes=preflight_report["data_hashes"],
            run_mode=run_mode,
            total_steps=total_steps,
        )
        if str(payload["run_id"]) != run_id:
            raise ValueError("checkpoint run_id does not match its run directory")
        epoch = int(payload["epoch"])
        global_step = int(payload["global_step"])
        run_id = str(payload["run_id"])
        recent_metrics.extend(payload.get("recent_metrics") or [])
        last_checkpoint_step = global_step
        pending_validation_step = payload.get("pending_validation_step")
    interrupted = {"requested": False, "signal": None}

    def request_stop(signum, _frame):
        interrupted["requested"] = True
        interrupted["signal"] = int(signum)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    log_path = output_dir / "train_log.jsonl"
    if rank == 0 and global_step == 0:
        with log_path.open("w", encoding="utf-8"):
            pass
    dist.barrier()

    if pending_validation_step is not None:
        if int(pending_validation_step) != global_step:
            raise RuntimeError("checkpoint pending-validation step does not match global step")
        validation_path = output_dir / "cheap_validation_step_{:08d}.json".format(global_step)
        if not validation_path.is_file():
            raise FileNotFoundError(
                "pending proposal validation lacks same-step cheap validation: {}".format(
                    validation_path
                )
            )
        latest_cheap_validation = read_json(validation_path)
        complete_proposal_validation(
            model,
            config,
            preflight_report,
            output_dir,
            global_step,
            checkpoint_dir / "checkpoint_step_{:08d}.pt".format(global_step),
            latest_cheap_validation,
            rank=rank,
        )
        pending_validation_step = None

    planner = None
    start_wall = time.time()
    proposal_kwargs = proposal_loss_kwargs(config)
    while global_step < max_steps and not interrupted["requested"]:
        if planner is None:
            schedule = DeterministicShardGroupMixtureSchedule(
                base_train,
                b20_train,
                seed=int(training["seed"]),
                epoch=epoch,
                length_bucket_width=int(batching["length_bucket_width"]),
                catalog_sha256=catalog_sha,
            )
            planner = DynamicGlobalBatchPlanner(
                schedule,
                rank=rank,
                world_size=context["world_size"],
                conditions_per_rank=int(batching["conditions_per_rank"]),
                max_tokens_per_gpu=int(batching["max_tokens_per_gpu"]),
            )
            if resume_planner_state is not None:
                planner.load_state_dict(resume_planner_state)
                resume_planner_state = None
        plans = []
        for _ in range(int(batching["gradient_accumulation_steps"])):
            plan = planner.next_batch()
            if plan is None:
                break
            plans.append(plan)
        if not plans:
            epoch += 1
            planner = None
            continue
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        step_start = time.time()
        step_pairs = step_conditions = step_tokens = 0
        step_encoder_tokens = step_encoder_padded_tokens = 0
        step_source_counts = Counter()
        step_global_source_counts = Counter()
        step_padding = []
        loss_sums = Counter()
        last_info = None
        try:
            loaded_batches = []
            micro_counts = []
            all_pairs = []
            for plan in plans:
                loaded = loader.load(plan.local_conditions, epoch=epoch)
                loaded_batches.append(loaded)
                pairs = list(loaded.pairs)
                all_pairs.extend(pairs)
                micro_counts.append(joint_objective_normalization_counts(pairs))
                step_pairs += len(loaded.pairs)
                step_conditions += len(loaded.references)
                step_tokens += loaded.sequence_tokens
                step_source_counts.update(loaded.source_condition_counts)
                step_global_source_counts.update(plan.global_source_counts)
                step_padding.append(plan.padding_fraction)
            local_counts = joint_objective_normalization_counts(all_pairs)
            for name in local_counts:
                if sum(item[name] for item in micro_counts) != local_counts[name]:
                    raise RuntimeError(
                        "condition identity crossed a gradient-accumulation boundary for {}".format(name)
                    )
            global_counts = {
                name: int(all_reduce_scalar(value, device)) for name, value in local_counts.items()
            }
            for micro_index, (loaded, normalization) in enumerate(zip(loaded_batches, micro_counts)):
                sync_context = (
                    ddp_model.no_sync()
                    if micro_index + 1 < len(plans)
                    else contextlib.nullcontext()
                )
                with sync_context:
                    info = ddp_model(
                        list(loaded.pairs),
                        encoder_microbatch_size=int(batching["encoder_microbatch_size"]),
                        scorer_tie_weight_scale=float(training["scorer_tie_weight_scale"]),
                        proposal_kwargs=proposal_kwargs,
                        scorer_loss_weight=float(training["scorer_loss_weight"]),
                        proposal_loss_weight=float(training["proposal_loss_weight"]),
                    )
                    components = normalized_joint_components(
                        info,
                        normalization,
                        global_counts,
                        config,
                        context["world_size"],
                    )
                    loss = components["loss"]
                    if not bool(torch.isfinite(loss.detach()).item()):
                        raise RuntimeError("non-finite joint loss")
                    loss.backward()
                encoder_tokens, encoder_padded = encoder_padding_counts(
                    info["encoded_wildtypes"], int(batching["encoder_microbatch_size"])
                )
                step_encoder_tokens += encoder_tokens
                step_encoder_padded_tokens += encoder_padded
                last_info = info
                for name, value in components.items():
                    loss_sums[name] += float(value.detach().cpu())
            scorer_grad = trainable_gradient_norm(model.scorer.parameters())
            policy_grad = trainable_gradient_norm(model.policy.parameters())
            if not math.isfinite(scorer_grad) or not math.isfinite(policy_grad):
                raise RuntimeError("non-finite gradient norm")
            clip = float(training["gradient_clip_norm"])
            total_grad = float(torch.nn.utils.clip_grad_norm_(parameters, clip).detach().cpu()) if clip > 0.0 else 0.0
            if not math.isfinite(total_grad):
                raise RuntimeError("non-finite total gradient norm")
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            # Deliberately no automatic batch shrink or retry: Phase 7 uses
            # this failure to select an explicit safe configuration.
            raise
        global_step += 1
        elapsed = max(1.0e-9, time.time() - step_start)
        global_pairs = all_reduce_scalar(step_pairs, device)
        global_conditions = all_reduce_scalar(step_conditions, device)
        global_tokens = all_reduce_scalar(step_tokens, device)
        global_encoder_tokens = all_reduce_scalar(step_encoder_tokens, device)
        global_encoder_padded_tokens = all_reduce_scalar(step_encoder_padded_tokens, device)
        global_losses = {
            name: all_reduce_scalar(value, device) / float(context["world_size"])
            for name, value in loss_sums.items()
        }
        max_elapsed = all_reduce_scalar(elapsed, device, op=dist.ReduceOp.MAX)
        max_allocated = all_reduce_scalar(torch.cuda.max_memory_allocated(device), device, op=dist.ReduceOp.MAX)
        max_reserved = all_reduce_scalar(torch.cuda.max_memory_reserved(device), device, op=dist.ReduceOp.MAX)
        scorer_grad_global = all_reduce_scalar(scorer_grad, device, op=dist.ReduceOp.MAX)
        policy_grad_global = all_reduce_scalar(policy_grad, device, op=dist.ReduceOp.MAX)
        total_grad_global = all_reduce_scalar(total_grad, device, op=dist.ReduceOp.MAX)
        planner_padding_global = all_reduce_scalar(
            sum(step_padding) / len(step_padding), device
        ) / float(context["world_size"])
        padding_global = (
            (global_encoder_padded_tokens - global_encoder_tokens) / global_encoder_padded_tokens
            if global_encoder_padded_tokens > 0.0
            else 0.0
        )
        local_utilization = gpu_utilization(context["identity"]["device_uuid"])
        utilization_by_rank = [None for _ in range(context["world_size"])]
        dist.all_gather_object(utilization_by_rank, local_utilization)
        record = {
            "step": global_step,
            "epoch": epoch,
            "global_cursor": planner.cursor,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "loss": global_losses["loss"],
            "scorer_loss": global_losses["scorer_loss"],
            "proposal_loss": global_losses["proposal_loss"],
            "rank_loss": global_losses["rank_loss"],
            "tie_loss": global_losses["tie_loss"],
            "imitation_loss": global_losses["imitation_loss"],
            "budget_loss": global_losses["budget_loss"],
            "structure_loss": global_losses["structure_loss"],
            "scorer_gradient_norm": scorer_grad_global,
            "policy_gradient_norm": policy_grad_global,
            "total_gradient_norm_pre_clip": total_grad_global,
            "global_pairs": int(global_pairs),
            "global_conditions": int(global_conditions),
            "global_tokens": int(global_tokens),
            "global_unique_encoder_tokens": int(global_encoder_tokens),
            "global_padded_encoder_tokens": int(global_encoder_padded_tokens),
            "local_source_conditions": dict(step_source_counts),
            "global_source_conditions": dict(step_global_source_counts),
            "global_base_to_b20_running_pattern": "3:1_condition_level",
            "step_time_seconds": max_elapsed,
            "tokens_per_second": global_tokens / max_elapsed,
            "effective_tokens_per_second": global_tokens / max_elapsed,
            "unique_encoder_tokens_per_second": global_encoder_tokens / max_elapsed,
            "conditions_per_second": global_conditions / max_elapsed,
            "pairs_per_second": global_pairs / max_elapsed,
            "padding_fraction": padding_global,
            "planner_conservative_padding_fraction": planner_padding_global,
            "max_memory_allocated_bytes": int(max_allocated),
            "max_memory_reserved_bytes": int(max_reserved),
            "gpu_utilization_percent_by_rank": utilization_by_rank,
            "objective_normalization_counts_global": global_counts,
            "world_size": context["world_size"],
            "conditions_per_rank_limit": int(batching["conditions_per_rank"]),
            "pairs_per_condition_limit": int(batching["pairs_per_condition"]),
            "encoder_microbatch_size": int(batching["encoder_microbatch_size"]),
            "gradient_accumulation_steps": int(batching["gradient_accumulation_steps"]),
            "max_tokens_per_gpu": int(batching["max_tokens_per_gpu"]),
            "num_budget_conditions_last_microbatch": int(last_info["proposal"]["num_budget_conditions"]),
            "num_imitation_used_last_microbatch": int(last_info["proposal"]["num_imitation_used"]),
            "num_negative_supervision_used_last_microbatch": int(last_info["proposal"]["num_negative_supervision_used"]),
        }
        recent_metrics.append(record)
        if rank == 0:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            if global_step == 1 or global_step % int(training["log_every"]) == 0:
                print(json.dumps({"event": "delepref_final_train_step", **record}, sort_keys=True), flush=True)

        stop_requested = all_reduce_scalar(
            1.0 if interrupted["requested"] else 0.0, device, op=dist.ReduceOp.MAX
        ) > 0.0
        interrupted["requested"] = bool(stop_requested)
        final_step = global_step >= max_steps
        need_validation = (
            global_step % int(training["cheap_validation_every"]) == 0 or final_step
        )
        if need_validation:
            dist.barrier()
            validation_payload = cheap_validation(
                model,
                validation_loader,
                dev_fixed,
                config,
                device,
                {"scorer": scorer_grad, "policy": policy_grad},
            ) if rank == 0 else None
            validation_box = [validation_payload]
            dist.broadcast_object_list(validation_box, src=0)
            validation_payload = validation_box[0]
            validation_payload["step"] = global_step
            latest_cheap_validation = validation_payload
            if rank == 0:
                atomic_json(output_dir / "cheap_validation_step_{:08d}.json".format(global_step), validation_payload)
            dist.barrier()
            if validation_payload["status"] != "PASS":
                raise RuntimeError("cheap validation gate failed")
        need_proposal_validation = (
            not args.skip_proposal_validation
            and not interrupted["requested"]
            and (
                global_step % int(training["proposal_validation_every"]) == 0
                or final_step
            )
        )
        need_checkpoint = (
            global_step % int(training["checkpoint_every"]) == 0
            or need_proposal_validation
            or final_step
            or interrupted["requested"]
        )
        checkpoint_path = checkpoint_dir / "checkpoint_step_{:08d}.pt".format(global_step)
        if need_checkpoint:
            save_checkpoint(
                checkpoint_path,
                ddp_model=ddp_model,
                optimizer=optimizer,
                scheduler=scheduler,
                context=context,
                planner=planner,
                global_step=global_step,
                total_steps=total_steps,
                epoch=epoch,
                config=config,
                preflight_report=preflight_report,
                run_id=run_id,
                run_mode=run_mode,
                recent_metrics=recent_metrics,
                interrupted=interrupted["requested"],
                pending_validation_step=(
                    global_step if need_proposal_validation else None
                ),
            )
            last_checkpoint_step = global_step
        if need_proposal_validation:
            if latest_cheap_validation is None:
                raise RuntimeError("proposal validation requires a cheap-validation result")
            complete_proposal_validation(
                model,
                config,
                preflight_report,
                output_dir,
                global_step,
                checkpoint_path,
                latest_cheap_validation,
                rank=rank,
            )
        if interrupted["requested"]:
            break

    if planner is not None and global_step > last_checkpoint_step:
        checkpoint_path = checkpoint_dir / "checkpoint_step_{:08d}.pt".format(global_step)
        save_checkpoint(
            checkpoint_path,
            ddp_model=ddp_model,
            optimizer=optimizer,
            scheduler=scheduler,
            context=context,
            planner=planner,
            global_step=global_step,
            total_steps=total_steps,
            epoch=epoch,
            config=config,
            preflight_report=preflight_report,
            run_id=run_id,
            run_mode=run_mode,
            recent_metrics=recent_metrics,
            interrupted=interrupted["requested"],
            pending_validation_step=None,
        )
        last_checkpoint_step = global_step

    final_digest = model_parameter_digest(model)
    digests = [None for _ in range(context["world_size"])]
    dist.all_gather_object(digests, final_digest)
    if len(set(digests)) != 1:
        raise RuntimeError("DDP head parameters are not synchronized")
    final_sampler_cursor = (
        planner.cursor
        if planner is not None
        else int((resume_planner_state or {}).get("global_cursor", 0))
    )
    final_summary = {
        "schema_version": "delepref_final_streaming_run_summary_v1",
        "status": "INTERRUPTED_CHECKPOINTED" if interrupted["requested"] else "BOUNDED_COMPLETE",
        "run_id": run_id,
        "global_step": global_step,
        "total_steps": total_steps,
        "epoch": epoch,
        "sampler_global_cursor": final_sampler_cursor,
        "final_checkpoint_step": last_checkpoint_step,
        "proposal_validation_skipped": bool(args.skip_proposal_validation),
        "parameter_sha256": final_digest,
        "ddp_parameter_sync": True,
        "gpu_identities": context["all_identities"],
        "elapsed_seconds": time.time() - start_wall,
        "formal_training": bool(args.formal_training),
        "smoke": bool(args.smoke),
        "interrupted_signal": interrupted["signal"],
        "confirmation_manifest_opened": False,
        "confirmation_metrics_viewed": False,
    }
    if rank == 0:
        atomic_json(output_dir / "run_summary.json", final_summary)
        print(json.dumps(final_summary, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as error:
        if not isinstance(error, SystemExit) or error.code not in (0, None):
            write_failure_artifact(error)
        raise
