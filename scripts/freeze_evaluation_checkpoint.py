#!/usr/bin/env python3
"""Freeze one completed formal Student DelePref selection for evaluation.

This tool is deliberately separate from the protected training entry point.  It
accepts only a completed formal run, strips every resumability-only field, and
emits a head-only checkpoint that can be read with ``weights_only=True``.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
FINAL_CONFIG = REPO_ROOT / "configs/deletion_preference_student.yaml"
FORMAL_RUN_ID = "delepref_final_training"
PENDING_STATUS = "PENDING_FINAL_SELECTED_CHECKPOINT"
EXPORT_SCHEMA = "delepref_evaluation_checkpoint_v1"
STREAMING_SCHEMA = "delepref_final_streaming_checkpoint_v1"
STUDENT_VERSION = "strict_budget_delepref_v1"
COORDINATE_SYSTEM = "0_based_closed_inclusive"
ALLOWED_BUDGETS = (0.10, 0.20)
FINAL_QMAX = 3
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class FreezeValidationError(RuntimeError):
    """Raised when provenance is incomplete or inconsistent."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze a completed Student DelePref selection for evaluation")
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--training-run-summary", required=True)
    parser.add_argument("--config", default=str(FINAL_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Match the protected trainer's config digest exactly."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise FreezeValidationError("missing required JSON: {}".format(path)) from exc
    except (OSError, ValueError) as exc:
        raise FreezeValidationError("invalid required JSON {}: {}".format(path, exc)) from exc
    if not isinstance(payload, Mapping):
        raise FreezeValidationError("JSON root must be an object: {}".format(path))
    return payload


def load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise FreezeValidationError("config root must be a mapping")
    return payload


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _config_path(value: object) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise FreezeValidationError("config training.checkpoint_dir is empty")
    path = Path(raw)
    return _resolved(path if path.is_absolute() else REPO_ROOT / path)


def _is_complete_summary(summary: Mapping[str, Any]) -> Tuple[bool, str]:
    status = str(summary.get("status") or "").strip()
    try:
        global_step = int(summary.get("global_step", -1))
        total_steps = int(summary.get("total_steps", -2))
    except (TypeError, ValueError):
        return False, "training summary has non-integer step fields"
    interrupted = summary.get("interrupted")
    if interrupted is None:
        interrupted = summary.get("interrupted_signal") not in (None, "", False)
    if status not in {"BOUNDED_COMPLETE", "FORMAL_TRAINING_COMPLETE", "PASS"}:
        return False, "training status is not complete"
    if global_step < 1 or global_step != total_steps:
        return False, "global_step does not equal total_steps"
    if bool(interrupted):
        return False, "training summary is interrupted"
    if bool(summary.get("proposal_validation_skipped", False)):
        return False, "proposal validation was skipped"
    if bool(summary.get("confirmation_metrics_viewed", False)):
        return False, "confirmation metrics were viewed"
    return True, "complete"


def inspect_training_state(selection_path: Path, summary_path: Path) -> Dict[str, Any]:
    """Return a non-mutating release result without opening a checkpoint."""

    result: Dict[str, Any] = {
        "status": PENDING_STATUS,
        "selection_json": str(_resolved(selection_path)),
        "training_run_summary": str(_resolved(summary_path)),
        "exported": False,
    }
    if not selection_path.is_file() or not summary_path.is_file():
        result["reason"] = "selection_or_completed_training_summary_missing"
        return result
    selection = read_json(selection_path)
    summary = read_json(summary_path)
    complete, reason = _is_complete_summary(summary)
    if not complete:
        result["reason"] = reason
        return result
    if selection.get("status") != "PASS":
        result["reason"] = "checkpoint_selection_status_not_PASS"
        return result
    result.update(
        {
            "status": "READY_FOR_EVALUATION_CHECKPOINT_EXPORT",
            "reason": "completed_training_and_passing_selection_present",
            "selected_step": (selection.get("selected") or {}).get("step"),
        }
    )
    return result


def _validate_hash_mapping(value: object, field: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise FreezeValidationError("{} must be a non-empty mapping".format(field))
    normalized = {str(key): str(item).lower() for key, item in value.items()}
    bad = [key for key, item in normalized.items() if not HEX64.fullmatch(item)]
    if bad:
        raise FreezeValidationError("{} contains invalid SHA-256 values: {}".format(field, bad[:5]))
    return normalized


def _load_trusted_streaming_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch

    # Formal streaming checkpoints are trusted local training products.  They
    # intentionally contain Python/NumPy RNG state, so weights_only=False is
    # explicit and is used only after the root/path provenance gates.
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise FreezeValidationError("source checkpoint root must be a mapping")
    return payload


def _validate_selection(selection: Mapping[str, Any]) -> Mapping[str, Any]:
    if selection.get("status") != "PASS":
        raise FreezeValidationError("checkpoint_selection.json.status must be PASS")
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise FreezeValidationError("checkpoint selection has no selected object")
    if selected.get("legality_gates_pass") is not True:
        raise FreezeValidationError("selected checkpoint failed legality gates")
    if selection.get("confirmation_metrics_used") is not False:
        raise FreezeValidationError("confirmation metrics were used by checkpoint selection")
    if selected.get("confirmation_metrics_used") is not False:
        raise FreezeValidationError("selected candidate used confirmation metrics")
    if selected.get("forbidden_metrics_used") not in ([], ()):
        raise FreezeValidationError("selected candidate used forbidden metrics")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise FreezeValidationError("checkpoint selection has no candidates")
    if any(item.get("confirmation_metrics_used") is not False for item in candidates if isinstance(item, Mapping)):
        raise FreezeValidationError("a checkpoint-selection candidate used confirmation metrics")
    return selected


def _completion_for(selection_path: Path, step: int) -> Tuple[Path, Mapping[str, Any]]:
    path = selection_path.parent / "proposal_validation" / "step_{:08d}".format(step) / "completion.json"
    payload = read_json(path)
    if payload.get("status") != "PASS" or int(payload.get("step", -1)) != int(step):
        raise FreezeValidationError("proposal validation completion is not PASS for step {}".format(step))
    return path, payload


def validate_export_inputs(
    selection_path: Path,
    summary_path: Path,
    config_path: Path,
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Path]:
    selection = read_json(selection_path)
    summary = read_json(summary_path)
    config = load_yaml(config_path)
    complete, reason = _is_complete_summary(summary)
    if not complete:
        raise FreezeValidationError("training is not complete: {}".format(reason))
    selected = _validate_selection(selection)
    total_steps = int(summary["total_steps"])
    final_completion_path, final_completion = _completion_for(selection_path, total_steps)
    if int(final_completion.get("selected_step", -1)) != int(selected.get("step", -2)):
        raise FreezeValidationError("final proposal validation and selection disagree on selected step")
    if final_completion.get("checkpoint_selection_sha256") != sha256_file(selection_path):
        raise FreezeValidationError("final validation completion has stale checkpoint-selection SHA-256")

    selected_step = int(selected.get("step", -1))
    checkpoint_raw = str(selected.get("checkpoint_path") or "").strip()
    if selected_step < 1 or not checkpoint_raw:
        raise FreezeValidationError("selected checkpoint step/path is empty")
    checkpoint_path = _resolved(Path(checkpoint_raw))
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise FreezeValidationError("selected checkpoint is missing or empty: {}".format(checkpoint_path))
    run_id = str(summary.get("run_id") or FORMAL_RUN_ID)
    checkpoint_base = _config_path((config.get("training") or {}).get("checkpoint_dir"))
    expected_root = _resolved(checkpoint_base / run_id)
    try:
        checkpoint_path.relative_to(expected_root)
    except ValueError as exc:
        raise FreezeValidationError("selected checkpoint is outside formal DelePref checkpoint root {}".format(expected_root)) from exc
    if checkpoint_path.parent != expected_root:
        raise FreezeValidationError("selected checkpoint must be directly inside the formal run root")
    if checkpoint_path.name != "checkpoint_step_{:08d}.pt".format(selected_step):
        raise FreezeValidationError("selected checkpoint filename/step mismatch")
    expected_source_hash = selected.get("checkpoint_sha256") or selected.get("source_checkpoint_sha256")
    if expected_source_hash not in (None, "") and str(expected_source_hash) != sha256_file(checkpoint_path):
        raise FreezeValidationError("selected checkpoint/source hash mismatch")

    checkpoint = _load_trusted_streaming_checkpoint(checkpoint_path)
    if checkpoint.get("schema_version") != STREAMING_SCHEMA:
        raise FreezeValidationError("unexpected source checkpoint schema")
    if checkpoint.get("student_version") != STUDENT_VERSION:
        raise FreezeValidationError("unexpected source checkpoint student_version")
    if int(checkpoint.get("global_step", -1)) != selected_step:
        raise FreezeValidationError("source checkpoint step differs from selection JSON")
    if int(checkpoint.get("total_steps", -1)) != total_steps:
        raise FreezeValidationError("source checkpoint total_steps differs from training summary")
    if bool(checkpoint.get("interrupted", False)):
        raise FreezeValidationError("selected checkpoint is interrupted")
    pending = checkpoint.get("pending_validation_step")
    if pending is not None:
        # The protected trainer persists the pre-validation checkpoint before
        # writing completion.json.  A same-step immutable PASS completion is
        # the only accepted evidence that this marker has been discharged.
        selected_completion_path, selected_completion = _completion_for(selection_path, selected_step)
        if int(pending) != selected_step:
            raise FreezeValidationError("selected checkpoint has an unresolved pending_validation_step")
        if _resolved(Path(str(selected_completion.get("checkpoint_path") or ""))) != checkpoint_path:
            raise FreezeValidationError("selected-step validation completion points to another checkpoint")
        if not selected_completion_path.is_file():
            raise FreezeValidationError("selected checkpoint pending validation is unresolved")
    if checkpoint.get("confirmation_metrics_viewed") is not False:
        raise FreezeValidationError("source checkpoint viewed confirmation metrics")
    if checkpoint.get("confirmation_manifest_opened") is not False:
        raise FreezeValidationError("source checkpoint opened the confirmation manifest")
    if checkpoint.get("encoder_weights_saved") is not False:
        raise FreezeValidationError("source checkpoint unexpectedly stores encoder weights")
    config_sha = canonical_sha256(config)
    if checkpoint.get("config_sha256") != config_sha:
        raise FreezeValidationError("source checkpoint config SHA-256 mismatch")
    _validate_hash_mapping(checkpoint.get("data_hashes"), "data_hashes")
    state = checkpoint.get("student_head_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise FreezeValidationError("source checkpoint has no student head state")
    if any(str(key).startswith("encoder.") for key in state):
        raise FreezeValidationError("source checkpoint head state contains encoder weights")
    if int((config.get("policy") or {}).get("qmax", -1)) != FINAL_QMAX:
        raise FreezeValidationError("final config Qmax must equal 3")
    budgets = tuple(float(value) for value in (config.get("proposal_validation") or {}).get("budgets", ()))
    if budgets != ALLOWED_BUDGETS:
        raise FreezeValidationError("final config proposal budgets must be exactly 0.10 and 0.20")
    return selection, summary, config, checkpoint, checkpoint_path


def _metadata_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_metadata_safe(item) for item in value)
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _metadata_safe(item) for key, item in value.items())
    return False


def _atomic_torch_save_no_replace(payload: Mapping[str, Any], destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError("refusing to overwrite existing evaluation checkpoint {}".format(destination))
    handle = tempfile.NamedTemporaryFile(prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent), delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(dict(payload), str(temporary))
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.link(str(temporary), str(destination))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _verify_export(path: Path, expected: Mapping[str, Any]) -> None:
    import torch

    loaded = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise FreezeValidationError("evaluation checkpoint did not reload as a mapping")
    if set(loaded) != set(expected):
        raise FreezeValidationError("evaluation checkpoint top-level fields changed after serialization")
    for key, value in expected.items():
        actual = loaded[key]
        if key == "student_head_state_dict":
            if set(actual) != set(value):
                raise FreezeValidationError("evaluation checkpoint tensor names changed")
            for name, tensor in value.items():
                if not bool(actual[name].equal(tensor)):
                    raise FreezeValidationError("evaluation checkpoint tensor changed: {}".format(name))
        elif actual != value:
            raise FreezeValidationError("evaluation checkpoint metadata changed: {}".format(key))


def export_evaluation_checkpoint(
    selection_path: Path,
    summary_path: Path,
    config_path: Path,
    output_dir: Path,
) -> Tuple[Path, Dict[str, Any]]:
    selection, _summary, config, checkpoint, checkpoint_path = validate_export_inputs(
        selection_path, summary_path, config_path
    )
    selected = selection["selected"]
    selected_step = int(selected["step"])
    source_hash = sha256_file(checkpoint_path)
    config_sha = canonical_sha256(config)
    state = {str(key): value.detach().cpu() for key, value in checkpoint["student_head_state_dict"].items()}
    payload: Dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA,
        "student_version": STUDENT_VERSION,
        "selected_step": selected_step,
        "student_head_state_dict": state,
        "model_config": dict(config.get("model") or {}),
        "policy_config": dict(config.get("policy") or {}),
        "decoder_config": dict(config.get("decoder") or {}),
        "batching_config": {"encoder_microbatch_size": int((config.get("batching") or {}).get("encoder_microbatch_size", 1))},
        "config_sha256": config_sha,
        "data_hashes": dict(checkpoint["data_hashes"]),
        "source_checkpoint_path": str(checkpoint_path),
        "source_checkpoint_sha256": source_hash,
        "checkpoint_selection_sha256": sha256_file(selection_path),
        "coordinate_system": str(checkpoint.get("coordinate_system") or COORDINATE_SYSTEM),
        "qmax": FINAL_QMAX,
        "max_length": int((config.get("model") or {}).get("max_length", 1022)),
        "budgets": list(ALLOWED_BUDGETS),
        "created_at": _datetime.datetime.now(_datetime.timezone.utc).replace(microsecond=0).isoformat(),
    }
    if not _metadata_safe({key: value for key, value in payload.items() if key != "student_head_state_dict"}):
        raise FreezeValidationError("evaluation checkpoint contains metadata unsafe for weights_only=True")
    forbidden = {
        "optimizer_state_dict", "scheduler_state_dict", "rank_states", "rng", "python_rng",
        "numpy_rng", "cuda_rng", "sampler_cursor", "pending_validation_step",
    }
    if forbidden.intersection(payload):
        raise FreezeValidationError("evaluation checkpoint contains forbidden resumability state")
    destination = _resolved(output_dir) / "delepref_evaluation_step_{:08d}.pt".format(selected_step)
    _atomic_torch_save_no_replace(payload, destination)
    _verify_export(destination, payload)
    report = {
        "status": "PASS_EVALUATION_CHECKPOINT_EXPORTED",
        "exported": True,
        "evaluation_checkpoint": str(destination),
        "evaluation_checkpoint_sha256": sha256_file(destination),
        "selected_step": selected_step,
        "source_checkpoint_sha256": source_hash,
        "checkpoint_selection_sha256": payload["checkpoint_selection_sha256"],
        "weights_only_reload": "PASS",
    }
    return destination, report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    selection_path = _resolved(Path(args.selection_json))
    summary_path = _resolved(Path(args.training_run_summary))
    config_path = _resolved(Path(args.config))
    if args.inspect_only:
        print(json.dumps(inspect_training_state(selection_path, summary_path), sort_keys=True))
        return 0
    _path, report = export_evaluation_checkpoint(
        selection_path, summary_path, config_path, _resolved(Path(args.output_dir))
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FreezeValidationError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
