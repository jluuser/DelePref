#!/usr/bin/env python3
"""Freeze the held-out dev proteins used by DelePref proposal validation.

The confirmation manifest is deliberately not an input.  Selection is a
deterministic hash sample from the already-frozen dev split.  Existing cold
start queries are attached only as a diagnostic reference; they are never a
checkpoint-selection signal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import defaultdict
from io import StringIO
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = REPO / "data/processed/preference_training/delepref/splits/delepref_sequence_hash_v1/train_dev_manifest.csv"
DEFAULT_BASE_RAW_INDEX = REPO / "data/processed/preference_training/delepref/raw_teacher_assets/indexes/deletion_perturbations_shard_index.csv"
DEFAULT_BASE_IMPACT_INDEX = REPO / "data/processed/preference_training/delepref/raw_teacher_assets/indexes/teacher_impact_shard_index.csv"
DEFAULT_B20_RAW_INDEX = REPO / "data/processed/preference_training/delepref/b20_supplement/raw_candidates/indexes/deletion_perturbations_shard_index.csv"
DEFAULT_B20_IMPACT_INDEX = REPO / "data/processed/preference_training/delepref/b20_supplement/teacher_scoring/indexes/teacher_impact_shard_index.csv"
DEFAULT_OUTPUT = REPO / "data/processed/preference_training/delepref/proposal_validation/v1"

SELECTION_NAMESPACE = "delepref-final-proposal-validation-dev-v1"
PROTEIN_COLUMNS = (
    "selection_rank", "protein_id", "sequence_sha256", "wildtype_length",
    "split", "base_shard_index", "b20_shard_index", "selection_digest",
    "wildtype_sequence",
)
REFERENCE_COLUMNS = (
    "protein_id", "sequence_sha256", "wildtype_length", "beta",
    "deleted_length", "reference_role", "candidate_pool_size",
    "best_perturbation_id", "best_plan_id", "best_deleted_intervals",
    "best_generation_policy", "best_teacher_utility", "global_similarity",
    "retained_similarity", "junction_plausibility", "absolute_gates_pass",
    "source_raw_csv", "source_impact_csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def resolve_path(value: str, index_path: Path) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [REPO / path, index_path.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(str(candidates[0]))


def read_index(path: Path, path_fields) -> dict:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for ordinal, row in enumerate(csv.DictReader(handle)):
            shard = int(row.get("shard_index") or ordinal)
            value = next((row.get(field) for field in path_fields if row.get(field)), None)
            if value is None:
                raise ValueError("index row lacks a path field: {}".format(path))
            if shard in result:
                raise ValueError("duplicate shard {} in {}".format(shard, path))
            result[shard] = resolve_path(value, path)
    return result


def strict_delete_count(length: int, beta: float) -> int:
    return max(1, min(int(length) - 1, int(round(float(beta) * int(length)))))


def utility(impact: dict) -> float:
    values = (
        float(impact["global_similarity"]),
        float(impact["retained_similarity"]),
        float(impact["junction_plausibility"]),
    )
    if impact.get("scoring_status") != "ok" or not all(math.isfinite(item) for item in values):
        raise ValueError("cold-start impact is not a successful finite real-teacher score")
    return 0.35 * values[0] + 0.45 * values[1] + 0.20 * values[2]


def read_split_candidates(path: Path, *, minimum_length: int, maximum_length: int):
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            length = int(row["wildtype_length"])
            if row.get("split") != "dev" or not minimum_length <= length <= maximum_length:
                continue
            digest = sha256_text(
                "{}\0{}\0{}".format(SELECTION_NAMESPACE, row["protein_id"], row["sequence_sha256"])
            )
            rows.append({
                **row,
                "wildtype_length": length,
                "base_shard_index": int(row["base_shard_index"]),
                "b20_shard_index": int(row["b20_shard_index"]),
                "selection_digest": digest,
            })
    rows.sort(key=lambda item: (item["selection_digest"], item["protein_id"]))
    return rows


def filtered_csv_rows(path: Path, protein_ids):
    """Read only rows for selected proteins, avoiding huge JSON CSV fields."""

    wanted = set(protein_ids)
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
    if not header:
        raise ValueError("empty CSV {}".format(path))
    if shutil.which("rg") and wanted:
        pattern = "^(?:{})\\,".format("|".join(re.escape(item) for item in sorted(wanted)))
        completed = subprocess.run(
            ["rg", "--no-line-number", "--color=never", "--regexp", pattern, str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.returncode not in (0, 1):
            raise RuntimeError("rg failed for {}: {}".format(path, completed.stderr.strip()))
        return list(csv.DictReader(StringIO(header + completed.stdout)))
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("protein_id") in wanted:
                rows.append(row)
    return rows


def collect_source(candidate_rows, raw_paths, impact_paths, *, beta: float):
    """Collect one diagnostic cold-start reference for each candidate protein."""

    by_shard = defaultdict(dict)
    for candidate in candidate_rows:
        shard_field = "b20_shard_index" if math.isclose(beta, 0.20) else "base_shard_index"
        by_shard[int(candidate[shard_field])][candidate["protein_id"]] = candidate
    sequences = {}
    references = {}
    for shard in sorted(by_shard):
        if shard not in raw_paths or shard not in impact_paths:
            raise ValueError("source indexes omit shard {}".format(shard))
        wanted = by_shard[shard]
        raw_by_id = {}
        for row in filtered_csv_rows(raw_paths[shard], wanted):
            protein_id = row.get("protein_id")
            if protein_id not in wanted:
                continue
            sequence = row.get("wildtype_sequence", "")
            expected = wanted[protein_id]
            if len(sequence) != expected["wildtype_length"] or sha256_text(sequence) != expected["sequence_sha256"]:
                raise ValueError("wild-type identity mismatch for {}".format(protein_id))
            previous = sequences.setdefault(protein_id, sequence)
            if previous != sequence:
                raise ValueError("source rows disagree on wild type for {}".format(protein_id))
            target_d = strict_delete_count(len(sequence), beta)
            if int(row["deleted_length"]) == target_d:
                key = (protein_id, row["perturbation_id"], row["plan_id"])
                if key in raw_by_id:
                    raise ValueError("duplicate cold-start candidate {}".format(key))
                raw_by_id[key] = row
        impacts = {}
        for row in filtered_csv_rows(impact_paths[shard], wanted):
            key = (row.get("protein_id"), row.get("perturbation_id"), row.get("plan_id"))
            if key in raw_by_id:
                if key in impacts:
                    raise ValueError("duplicate cold-start impact {}".format(key))
                impacts[key] = row
        grouped = defaultdict(list)
        for key, raw in raw_by_id.items():
            impact = impacts.get(key)
            if impact is None:
                continue
            try:
                score = utility(impact)
            except (KeyError, TypeError, ValueError):
                continue
            grouped[key[0]].append((score, raw, impact))
        for protein_id in wanted:
            group = grouped.get(protein_id, [])
            if not group:
                continue
            group.sort(key=lambda item: (-item[0], item[1]["perturbation_id"], item[1]["plan_id"]))
            score, raw, impact = group[0]
            references[(protein_id, beta)] = {
                "protein_id": protein_id,
                "sequence_sha256": wanted[protein_id]["sequence_sha256"],
                "wildtype_length": wanted[protein_id]["wildtype_length"],
                "beta": "{:.2f}".format(beta),
                "deleted_length": strict_delete_count(wanted[protein_id]["wildtype_length"], beta),
                "reference_role": "diagnostic_only_not_checkpoint_selection",
                "candidate_pool_size": len(group),
                "best_perturbation_id": raw["perturbation_id"],
                "best_plan_id": raw["plan_id"],
                "best_deleted_intervals": raw["deleted_intervals"],
                "best_generation_policy": raw.get("generation_policy", ""),
                "best_teacher_utility": "{:.10g}".format(score),
                "global_similarity": impact["global_similarity"],
                "retained_similarity": impact["retained_similarity"],
                "junction_plausibility": impact["junction_plausibility"],
                "absolute_gates_pass": str(
                    float(impact["global_similarity"]) >= 0.95
                    and float(impact["retained_similarity"]) >= 0.90
                ).lower(),
                "source_raw_csv": str(raw_paths[shard]),
                "source_impact_csv": str(impact_paths[shard]),
            }
    return sequences, references


def csv_bytes(columns, rows) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
    return buffer.getvalue().encode("utf-8")


def freeze_bytes(path: Path, content: bytes) -> None:
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--base-raw-index", type=Path, default=DEFAULT_BASE_RAW_INDEX)
    parser.add_argument("--base-impact-index", type=Path, default=DEFAULT_BASE_IMPACT_INDEX)
    parser.add_argument("--b20-raw-index", type=Path, default=DEFAULT_B20_RAW_INDEX)
    parser.add_argument("--b20-impact-index", type=Path, default=DEFAULT_B20_IMPACT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--candidate-pool", type=int, default=512)
    parser.add_argument("--minimum-length", type=int, default=64)
    parser.add_argument("--maximum-length", type=int, default=1022)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.count < 1 or args.candidate_pool < args.count:
        raise ValueError("candidate pool must be at least the requested count")
    split_rows = read_split_candidates(
        args.split_manifest, minimum_length=args.minimum_length, maximum_length=args.maximum_length
    )
    candidates = split_rows[: args.candidate_pool]
    if len(candidates) < args.count:
        raise ValueError("dev split has too few eligible proteins")
    base_raw = read_index(args.base_raw_index, ("shard_path", "source_deletion_perturbations_csv"))
    base_impacts = read_index(args.base_impact_index, ("output_csv",))
    b20_raw = read_index(args.b20_raw_index, ("shard_path", "source_deletion_perturbations_csv"))
    b20_impacts = read_index(args.b20_impact_index, ("output_csv",))
    base_sequences, base_refs = collect_source(candidates, base_raw, base_impacts, beta=0.10)
    b20_sequences, b20_refs = collect_source(candidates, b20_raw, b20_impacts, beta=0.20)

    complete = []
    for candidate in candidates:
        protein_id = candidate["protein_id"]
        if (protein_id, 0.10) not in base_refs or (protein_id, 0.20) not in b20_refs:
            continue
        sequence = base_sequences.get(protein_id) or b20_sequences.get(protein_id)
        if not sequence or base_sequences.get(protein_id) != b20_sequences.get(protein_id):
            raise ValueError("base and b20 wild types disagree for {}".format(protein_id))
        complete.append((candidate, sequence))
        if len(complete) == args.count:
            break
    if len(complete) != args.count:
        raise RuntimeError("only {} deterministic dev proteins have both diagnostic references".format(len(complete)))

    protein_rows = []
    reference_rows = []
    for rank, (candidate, sequence) in enumerate(complete, start=1):
        protein_id = candidate["protein_id"]
        protein_rows.append({
            "selection_rank": rank,
            "protein_id": protein_id,
            "sequence_sha256": candidate["sequence_sha256"],
            "wildtype_length": candidate["wildtype_length"],
            "split": "dev",
            "base_shard_index": candidate["base_shard_index"],
            "b20_shard_index": candidate["b20_shard_index"],
            "selection_digest": candidate["selection_digest"],
            "wildtype_sequence": sequence,
        })
        reference_rows.extend((base_refs[(protein_id, 0.10)], b20_refs[(protein_id, 0.20)]))

    proteins_path = args.output_dir / "proteins.csv"
    references_path = args.output_dir / "cold_start_reference.csv"
    freeze_bytes(proteins_path, csv_bytes(PROTEIN_COLUMNS, protein_rows))
    freeze_bytes(references_path, csv_bytes(REFERENCE_COLUMNS, reference_rows))
    summary = {
        "schema_version": "delepref_final_proposal_validation_manifest_v1",
        "status": "PASS",
        "selection_namespace": SELECTION_NAMESPACE,
        "selection_source": "frozen_dev_split_only",
        "protein_count": len(protein_rows),
        "budgets": [0.10, 0.20],
        "seeds": [13, 29],
        "k": 8,
        "minimum_length": args.minimum_length,
        "maximum_length": args.maximum_length,
        "cold_start_reference_role": "diagnostic_only_not_checkpoint_selection",
        "cold_start_reference_rows": len(reference_rows),
        "confirmation_manifest_opened": False,
        "confirmation_metrics_computed_or_viewed": False,
        "inputs": {
            "split_manifest": {"path": str(args.split_manifest), "sha256": sha256_file(args.split_manifest)},
            "base_raw_index": {"path": str(args.base_raw_index), "sha256": sha256_file(args.base_raw_index)},
            "base_impact_index": {"path": str(args.base_impact_index), "sha256": sha256_file(args.base_impact_index)},
            "b20_raw_index": {"path": str(args.b20_raw_index), "sha256": sha256_file(args.b20_raw_index)},
            "b20_impact_index": {"path": str(args.b20_impact_index), "sha256": sha256_file(args.b20_impact_index)},
        },
        "outputs": {
            "proteins": {"path": str(proteins_path), "sha256": sha256_file(proteins_path)},
            "cold_start_reference": {"path": str(references_path), "sha256": sha256_file(references_path)},
        },
        "teacher_utility": "0.35*global_similarity+0.45*retained_similarity+0.20*junction_plausibility",
        "teacher_preference_is_ground_truth": False,
        "claim_boundary": "The method ranks candidate deletion plans by teacher-derived deletion perturbation preference and optional auxiliary risk evidence.",
    }
    summary_path = args.output_dir / "summary.json"
    freeze_bytes(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
