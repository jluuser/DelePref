#!/usr/bin/env python3
"""Streaming evaluation runtime for a frozen Student DelePref evaluation checkpoint.

The runtime uses the repository's existing ``DeletionPreferenceStudent``,
strict dynamic-programming decoder, proposal converter, and legality validator.
It contains no alternate model, heuristic decoder, or teacher-scoring path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for entry in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from delepref.preference_core import (  # noqa: E402
    COORDINATE_SYSTEM,
    DeletionCandidate,
    candidate_from_proposal,
    parse_deleted_intervals,
)
from delepref.student_model import (  # noqa: E402
    DeletionPreferenceStudent,
    EncodedWildTypes,
    STUDENT_VERSION,
    generate_strict_budget_plans,
    proposal_from_decoded_plan,
    strict_delete_count,
    validate_strict_plan,
)


EVALUATION_SCHEMA = "delepref_evaluation_checkpoint_v1"
FINAL_CONFIG = REPO_ROOT / "configs/deletion_preference_student.yaml"
SUPPORTED_BUDGETS = (0.10, 0.20)
QMAX = 3
MAX_LENGTH = 1022
DEFAULT_K = 8
DECODER_NAME = "strict_dynamic_programming"
CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion "
    "perturbation preference and optional auxiliary risk evidence."
)


CANONICAL_COLUMNS = [
    "student_version",
    "source_checkpoint_sha256",
    "selected_checkpoint_step",
    "protein_id",
    "sequence_sha256",
    "sequence_length",
    "budget",
    "generation_seed",
    "condition_seed",
    "raw_candidate_rank",
    "deleted_intervals",
    "deleted_length",
    "deletion_mask_sha256",
    "span_count",
    "policy_energy",
    "scorer_score",
    "scorer_rank",
    "policy_native_rank",
    "decoder",
    "qmax",
    "generation_attempts",
    "duplicate_decodes",
    "generation_reason",
    "strict_plan_valid",
    "student_confidence_available",
    "student_confidence_used_for_ranking",
    "preference_confidence",
    "coordinate_system",
    "wildtype_sequence",
]


@dataclass(frozen=True)
class EvaluationRuntime:
    model: DeletionPreferenceStudent
    config: Mapping[str, Any]
    checkpoint: Mapping[str, Any]
    device: Any
    source_checkpoint_sha256: str
    selected_step: int
    encoder_microbatch_size: int
    k: int = DEFAULT_K


@dataclass(frozen=True)
class EncodedChunk:
    encoded: EncodedWildTypes
    records: Tuple[Mapping[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def normalize_budget(value: object) -> float:
    try:
        budget = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("budget must be 0.10 or 0.20") from exc
    if not math.isfinite(budget) or not any(math.isclose(budget, item, rel_tol=0.0, abs_tol=1.0e-12) for item in SUPPORTED_BUDGETS):
        raise ValueError("unsupported Student DelePref evaluation budget {}; expected 0.10 or 0.20".format(value))
    return next(item for item in SUPPORTED_BUDGETS if math.isclose(budget, item, rel_tol=0.0, abs_tol=1.0e-12))


def derive_condition_seed(
    global_generation_seed: int,
    protein_id: str,
    sequence_hash: str,
    budget: object,
) -> int:
    """Derive an order-independent seed from stable condition identity."""

    protein = str(protein_id).strip()
    digest_text = str(sequence_hash).strip().lower()
    normalized = normalize_budget(budget)
    if not protein:
        raise ValueError("protein_id must be non-empty")
    if len(digest_text) != 64 or any(character not in "0123456789abcdef" for character in digest_text):
        raise ValueError("sequence_hash must be a lowercase SHA-256 digest")
    material = "{}\0{}\0{}\0{:.2f}".format(int(global_generation_seed), protein, digest_text, normalized)
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16) & ((1 << 63) - 1)


def _load_config(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required") from exc
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation config root must be a mapping")
    return payload


def _load_evaluation_payload(path: Path) -> Mapping[str, Any]:
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation checkpoint root must be a mapping")
    return payload


def _validate_evaluation_payload(payload: Mapping[str, Any], config_path: Path) -> None:
    if payload.get("schema_version") != EVALUATION_SCHEMA:
        raise ValueError("not a Student DelePref evaluation checkpoint")
    if payload.get("student_version") != STUDENT_VERSION:
        raise ValueError("evaluation checkpoint student_version mismatch")
    config = _load_config(config_path)
    if payload.get("config_sha256") != canonical_sha256(config):
        raise ValueError("evaluation checkpoint config SHA-256 mismatch")
    if int(payload.get("qmax", -1)) != QMAX:
        raise ValueError("evaluation checkpoint Qmax must equal 3")
    if int(payload.get("max_length", -1)) != MAX_LENGTH:
        raise ValueError("evaluation checkpoint max_length must equal 1022")
    if tuple(float(item) for item in payload.get("budgets", ())) != SUPPORTED_BUDGETS:
        raise ValueError("evaluation checkpoint budgets must equal [0.10, 0.20]")
    if payload.get("coordinate_system") != COORDINATE_SYSTEM:
        raise ValueError("evaluation checkpoint coordinate system mismatch")
    state = payload.get("student_head_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("evaluation checkpoint has no student head state")
    if any(str(key).startswith("encoder.") for key in state):
        raise ValueError("evaluation checkpoint contains frozen encoder weights")
    forbidden = {
        "optimizer_state_dict", "scheduler_state_dict", "rank_states", "python_rng",
        "numpy_rng", "cuda_rng", "sampler_cursor", "pending_validation_step",
    }
    present = sorted(forbidden.intersection(payload))
    if present:
        raise ValueError("evaluation checkpoint contains forbidden fields: {}".format(present))
    source_path = Path(str(payload.get("source_checkpoint_path") or ""))
    expected_hash = str(payload.get("source_checkpoint_sha256") or "")
    if not source_path.is_file():
        raise ValueError("evaluation checkpoint source checkpoint is missing")
    if sha256_file(source_path) != expected_hash:
        raise ValueError("evaluation checkpoint/source hash mismatch")


def load_runtime(
    evaluation_checkpoint: os.PathLike[str] | str,
    *,
    config_path: os.PathLike[str] | str = FINAL_CONFIG,
    device: str = "cuda",
    k: int = DEFAULT_K,
) -> EvaluationRuntime:
    """Load the official DelePref model with one shared frozen encoder and head state."""

    import torch
    from train_student_pilot import build_delepref

    if int(k) < 1:
        raise ValueError("K must be positive")
    if int(k) != DEFAULT_K:
        raise ValueError("formal Student DelePref evaluation requires K=8")
    config_file = Path(config_path).expanduser().resolve()
    checkpoint_file = Path(evaluation_checkpoint).expanduser().resolve()
    config = _load_config(config_file)
    payload = _load_evaluation_payload(checkpoint_file)
    _validate_evaluation_payload(payload, config_file)
    if int((config.get("policy") or {}).get("qmax", -1)) != QMAX:
        raise ValueError("formal config Qmax must equal 3")
    if int((config.get("model") or {}).get("max_length", -1)) != MAX_LENGTH:
        raise ValueError("formal config max_length must equal 1022")
    if device != "cuda":
        raise ValueError("formal ESMC Student DelePref evaluation runtime requires the configured CUDA encoder")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; evaluation runtime does not substitute another encoder")
    torch_device = torch.device(device)
    model = build_delepref(config, torch_device)
    missing, unexpected = model.load_state_dict(payload["student_head_state_dict"], strict=False)
    non_encoder_missing = [name for name in missing if not str(name).startswith("encoder.")]
    if non_encoder_missing or unexpected:
        raise RuntimeError("evaluation checkpoint head mismatch: missing={} unexpected={}".format(non_encoder_missing[:20], list(unexpected)[:20]))
    if model.scorer.encoder is not model.encoder or model.policy.encoder is not model.encoder:
        raise RuntimeError("DelePref scorer and policy do not share the same frozen encoder")
    model.eval()
    return EvaluationRuntime(
        model=model,
        config=config,
        checkpoint=payload,
        device=torch_device,
        source_checkpoint_sha256=str(payload["source_checkpoint_sha256"]),
        selected_step=int(payload["selected_step"]),
        encoder_microbatch_size=int((config.get("batching") or {}).get("encoder_microbatch_size", 1)),
        k=DEFAULT_K,
    )


def _normalize_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    protein_id = str(record.get("protein_id") or "").strip()
    sequence = str(record.get("wildtype_sequence") or record.get("sequence") or "").strip().upper()
    if not protein_id:
        raise ValueError("protein_id must be non-empty")
    if len(sequence) < 2 or len(sequence) > MAX_LENGTH:
        raise ValueError("protein {} length {} is outside the supported range 2..1022; no truncation is permitted".format(protein_id, len(sequence)))
    if not all(character in "ACDEFGHIKLMNPQRSTVWYX" for character in sequence):
        raise ValueError("protein {} contains unsupported sequence symbols".format(protein_id))
    budget = normalize_budget(record.get("budget", record.get("beta", record.get("target_delete_fraction"))))
    return {
        "protein_id": protein_id,
        "wildtype_sequence": sequence,
        "sequence_sha256": sequence_sha256(sequence),
        "budget": budget,
    }


def encode_chunk(runtime: EvaluationRuntime, records: Sequence[Mapping[str, Any]]) -> EncodedChunk:
    """Validate and encode only one bounded chunk of wild types."""

    normalized = tuple(_normalize_record(record) for record in records)
    if not normalized:
        raise ValueError("encode_chunk requires at least one input record")
    seen = set()
    for row in normalized:
        key = (row["protein_id"], row["sequence_sha256"], row["budget"])
        if key in seen:
            raise ValueError("duplicate protein/sequence/budget condition in chunk: {}".format(key))
        seen.add(key)
    encoded = runtime.model.encode_unique_wildtypes(
        [(row["protein_id"], row["wildtype_sequence"]) for row in normalized],
        microbatch_size=runtime.encoder_microbatch_size,
    )
    return EncodedChunk(encoded=encoded, records=normalized)


def _condition_key(row: Mapping[str, Any]) -> Tuple[str, str, float, int]:
    return (
        str(row["protein_id"]),
        str(row["sequence_sha256"]),
        float(row["budget"]),
        int(row["condition_seed"]),
    )


def _intervals_json(intervals: Sequence[Tuple[int, int]]) -> str:
    return json.dumps([[int(start), int(end)] for start, end in intervals], separators=(",", ":"))


def _mask_sha256(length: int, intervals: Sequence[Tuple[int, int]]) -> str:
    mask = bytearray(length)
    for start, end in intervals:
        mask[int(start) : int(end) + 1] = b"\x01" * (int(end) - int(start) + 1)
    return hashlib.sha256(bytes(mask)).hexdigest()


def generate_raw_candidates(
    runtime: EvaluationRuntime,
    chunk: EncodedChunk,
    *,
    global_generation_seed: int,
    k: int = DEFAULT_K,
) -> List[Dict[str, Any]]:
    """Generate exactly K unique strict-DP candidates per condition."""

    import torch

    if int(k) != runtime.k or int(k) != DEFAULT_K:
        raise ValueError("formal Student DelePref evaluation requires K=8")
    policy_config = runtime.config.get("policy") or {}
    decoder_config = runtime.config.get("decoder") or {}
    qmax = int(policy_config.get("qmax", -1))
    if qmax != QMAX:
        raise ValueError("formal Student DelePref evaluation requires Qmax=3")
    min_span = int(policy_config.get("min_span_len", 1))
    max_fraction = float(policy_config.get("max_span_len_fraction", 0.5))
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for record in chunk.records:
            protein_id = str(record["protein_id"])
            sequence = str(record["wildtype_sequence"])
            sequence_hash = str(record["sequence_sha256"])
            budget = float(record["budget"])
            condition_seed = derive_condition_seed(global_generation_seed, protein_id, sequence_hash, budget)
            embedding = chunk.encoded.embeddings[chunk.encoded.key_for(protein_id, sequence)]
            policy_output = runtime.model.propose_from_embeddings([embedding], [budget])[0]
            max_span = max(min_span, min(len(sequence) - 1, int(math.floor(max_fraction * len(sequence)))))
            generated = generate_strict_budget_plans(
                policy_output.residue_logits,
                budget,
                num_candidates=int(k),
                qmax=qmax,
                min_span_len=min_span,
                max_span_len=max_span,
                seed=condition_seed,
                perturbation_scale=float(decoder_config.get("perturbation_scale", 1.0)),
                max_attempts=int(k) * int(decoder_config.get("max_attempts_multiplier", 16)),
            )
            if len(generated.plans) != int(k):
                raise RuntimeError(
                    "incomplete K for protein={} budget={:.2f}: returned {}/{} reason={}; no fill or fallback is permitted".format(
                        protein_id, budget, len(generated.plans), k, generated.reason
                    )
                )
            condition_rows: List[Dict[str, Any]] = []
            for raw_rank, decoded in enumerate(generated.plans, start=1):
                validate_strict_plan(
                    len(sequence), budget, decoded.deleted_intervals,
                    qmax=qmax, min_span_len=min_span, max_span_len=max_span,
                )
                proposal = proposal_from_decoded_plan(
                    protein_id=protein_id,
                    wildtype_sequence=sequence,
                    beta=budget,
                    decoded=decoded,
                    rank=raw_rank,
                    generation_reason=generated.reason,
                )
                intervals = tuple(proposal.deleted_intervals)
                condition_rows.append(
                    {
                        "student_version": STUDENT_VERSION,
                        "source_checkpoint_sha256": runtime.source_checkpoint_sha256,
                        "selected_checkpoint_step": runtime.selected_step,
                        "protein_id": protein_id,
                        "sequence_sha256": sequence_hash,
                        "sequence_length": len(sequence),
                        "budget": budget,
                        "generation_seed": int(global_generation_seed),
                        "condition_seed": condition_seed,
                        "raw_candidate_rank": raw_rank,
                        "deleted_intervals": _intervals_json(intervals),
                        "deleted_length": int(proposal.deleted_length),
                        "deletion_mask_sha256": _mask_sha256(len(sequence), intervals),
                        "span_count": len(intervals),
                        "policy_energy": float(decoded.score) / float(decoded.deleted_length),
                        "scorer_score": "",
                        "scorer_rank": "",
                        "policy_native_rank": 0,
                        "decoder": DECODER_NAME,
                        "qmax": qmax,
                        "generation_attempts": generated.attempts,
                        "duplicate_decodes": generated.duplicate_count,
                        "generation_reason": generated.reason,
                        "strict_plan_valid": True,
                        "student_confidence_available": False,
                        "student_confidence_used_for_ranking": False,
                        "preference_confidence": "",
                        "coordinate_system": COORDINATE_SYSTEM,
                        "wildtype_sequence": sequence,
                    }
                )
            native_order = sorted(
                range(len(condition_rows)),
                key=lambda index: (
                    -float(condition_rows[index]["policy_energy"]),
                    parse_deleted_intervals(condition_rows[index]["deleted_intervals"], len(sequence)),
                ),
            )
            for native_rank, index in enumerate(native_order, start=1):
                condition_rows[index]["policy_native_rank"] = native_rank
            rows.extend(condition_rows)
    return rows


def _candidate_from_row(row: Mapping[str, Any]) -> DeletionCandidate:
    sequence = str(row.get("wildtype_sequence") or "")
    intervals = parse_deleted_intervals(row.get("deleted_intervals"), len(sequence))
    deleted = sum(end - start + 1 for start, end in intervals)
    return DeletionCandidate(
        protein_id=str(row["protein_id"]),
        perturbation_id="{}__deleprefeval__{}".format(row["protein_id"], row["deletion_mask_sha256"][:16]),
        plan_id="{}__deleprefeval__{}".format(row["protein_id"], row["deletion_mask_sha256"][:16]),
        wildtype_sequence=sequence,
        deleted_intervals=intervals,
        deleted_length=deleted,
        deletion_fraction=float(deleted) / float(len(sequence)),
        target_delete_fraction=float(row["budget"]),
        budget_bin="delepref_eval_b{:02d}".format(int(round(float(row["budget"]) * 100))),
        generation_policy="delepref_seeded_perturbation_strict_dp",
    )


def score_candidates(
    runtime: EvaluationRuntime,
    chunk: EncodedChunk,
    raw_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Score persisted raw candidates and assign deterministic scorer ranks."""

    import torch

    candidates = [_candidate_from_row(row) for row in raw_rows]
    with torch.no_grad():
        values = runtime.model.score_plans_from_embeddings(candidates, chunk.encoded)["preference_score"]
    scores = values.detach().float().cpu().tolist()
    scored: List[Dict[str, Any]] = []
    for raw, score in zip(raw_rows, scores):
        item = dict(raw)
        item["scorer_score"] = float(score)
        item["scorer_rank"] = 0
        scored.append(item)
    groups: MutableMapping[Tuple[str, str, float, int], List[int]] = defaultdict(list)
    for index, row in enumerate(scored):
        groups[_condition_key(row)].append(index)
    for indices in groups.values():
        ordered = sorted(
            indices,
            key=lambda index: (
                -float(scored[index]["scorer_score"]),
                parse_deleted_intervals(scored[index]["deleted_intervals"], int(scored[index]["sequence_length"])),
            ),
        )
        for rank, index in enumerate(ordered, start=1):
            scored[index]["scorer_rank"] = rank
    return scored


def _select_ranked(rows: Sequence[Mapping[str, Any]], field: str) -> List[Dict[str, Any]]:
    grouped: MutableMapping[Tuple[str, str, float, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_condition_key(row)].append(row)
    selected = []
    for key in sorted(grouped):
        matches = [row for row in grouped[key] if int(row.get(field, 0)) == 1]
        if len(matches) != 1:
            raise ValueError("condition {} has {} top-1 rows for {}".format(key, len(matches), field))
        selected.append(dict(matches[0]))
    return selected


def select_policy_native_top1(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return _select_ranked(rows, "policy_native_rank")


def select_scorer_top1(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return _select_ranked(rows, "scorer_rank")


def emit_canonical_rows(
    rows: Sequence[Mapping[str, Any]],
    output_path: Optional[os.PathLike[str] | str] = None,
    *,
    append: bool = False,
) -> List[Dict[str, Any]]:
    """Normalize rows and optionally stream them to CSV or JSONL."""

    canonical: List[Dict[str, Any]] = []
    for row in rows:
        item = {column: row.get(column, "") for column in CANONICAL_COLUMNS}
        item["student_confidence_available"] = False
        item["student_confidence_used_for_ranking"] = False
        item["preference_confidence"] = ""
        canonical.append(item)
    if output_path is None:
        return canonical
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    if destination.suffix.lower() == ".csv":
        write_header = not append or not destination.exists() or destination.stat().st_size == 0
        with destination.open(mode, encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(canonical)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        with destination.open(mode, encoding="utf-8") as handle:
            for item in canonical:
                handle.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return canonical


def _read_inputs(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            base = dict(row)
            explicit = row.get("budget") or row.get("beta") or row.get("target_delete_fraction")
            budgets: Iterable[object] = SUPPORTED_BUDGETS if explicit in (None, "") else (explicit,)
            for budget in budgets:
                expanded = dict(base)
                expanded["budget"] = budget
                try:
                    rows.append(_normalize_record(expanded))
                except ValueError as exc:
                    raise ValueError("input row {}: {}".format(row_number, exc)) from exc
    if not rows:
        raise ValueError("input CSV is empty")
    return rows


def _chunks(items: Sequence[Mapping[str, Any]], size: int) -> Iterable[Sequence[Mapping[str, Any]]]:
    if int(size) < 1:
        raise ValueError("chunk_size must be positive")
    for start in range(0, len(items), int(size)):
        yield items[start : start + int(size)]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run streaming Student DelePref evaluation candidate generation/reranking")
    parser.add_argument("--evaluation-checkpoint", required=True)
    parser.add_argument("--config", default=str(FINAL_CONFIG))
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--raw-candidates", required=True, help="JSONL/CSV written and fsynced before scorer execution")
    parser.add_argument("--scored-candidates", required=True)
    parser.add_argument("--canonical-top1", required=True)
    parser.add_argument("--generation-seed", type=int, default=13)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    runtime = load_runtime(args.evaluation_checkpoint, config_path=args.config, device=args.device, k=args.k)
    inputs = _read_inputs(Path(args.input_csv))
    raw_path = Path(args.raw_candidates)
    scored_path = Path(args.scored_candidates)
    top_path = Path(args.canonical_top1)
    for path in (raw_path, scored_path, top_path):
        if path.exists():
            raise FileExistsError("refusing to overwrite evaluation output {}".format(path))
    first = True
    conditions = candidates = 0
    for records in _chunks(inputs, args.chunk_size):
        chunk = encode_chunk(runtime, records)
        raw = generate_raw_candidates(runtime, chunk, global_generation_seed=args.generation_seed, k=args.k)
        emit_canonical_rows(raw, raw_path, append=not first)
        scored = score_candidates(runtime, chunk, raw)
        emit_canonical_rows(scored, scored_path, append=not first)
        emit_canonical_rows(select_scorer_top1(scored), top_path, append=not first)
        first = False
        conditions += len(records)
        candidates += len(raw)
        del chunk, raw, scored
    print(
        json.dumps(
            {
                "status": "PASS",
                "conditions": conditions,
                "candidates": candidates,
                "k": args.k,
                "claim_boundary": CLAIM_BOUNDARY,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
