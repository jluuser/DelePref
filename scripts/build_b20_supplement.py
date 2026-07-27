#!/usr/bin/env python3
"""Deterministically build the eight-candidate DelePref beta=0.20 supplement.

The raw candidate shards are written before any teacher scoring.  Exactly two
terminal, two position-separated internal, two boundary-oriented, one two-span,
and one three-span plan are generated for every input protein.  Every plan uses
D=min(L-1, round(0.2*L)) and Q<=3.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_INDEX = REPO / "data/processed/preference_training/delepref/raw_teacher_assets/indexes/deletion_perturbations_shard_index.csv"
DEFAULT_OUTPUT = REPO / "data/processed/preference_training/delepref/b20_supplement/raw_candidates"
COORDINATE_SYSTEM = "0_based_closed_inclusive"

COLUMNS = (
    "protein_id", "perturbation_id", "plan_id", "wildtype_length",
    "deleted_intervals", "num_intervals", "deleted_length",
    "deletion_fraction", "budget_bin", "generation_policy",
    "coordinate_system", "wildtype_sequence", "mutant_sequence",
    "source_stage1_row_id", "source_start", "source_end", "source_score",
    "wildtype_sha256", "mutant_sha256", "interval_plan_sha256",
    "metadata_json",
)

INDEX_COLUMNS = (
    "shard_id", "shard_index", "shard_path", "source_shard_path",
    "num_proteins", "num_candidates", "candidates_per_protein_target",
    "status", "metadata_json",
)

AA_CLASS = {
    **{aa: 0 for aa in "AVILMFWY"},
    **{aa: 1 for aa in "STNQCGP"},
    **{aa: 2 for aa in "DE"},
    **{aa: 3 for aa in "KRH"},
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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


def atomic_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_index(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for ordinal, row in enumerate(csv.DictReader(handle)):
            rows.append((int(row.get("shard_index") or ordinal), Path(row["shard_path"]), row))
    if not rows:
        raise ValueError("empty source index")
    return rows


def source_split(row):
    try:
        metadata = json.loads(row.get("metadata_json") or "{}")
    except Exception:
        metadata = {}
    return str(metadata.get("source_split") or "train")


def first_rows_by_protein(path):
    seen = set()
    result = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            protein_id = row.get("protein_id", "")
            if not protein_id:
                raise ValueError("source candidate has empty protein_id")
            if protein_id in seen:
                continue
            seen.add(protein_id)
            sequence = row.get("wildtype_sequence", "")
            if not sequence or int(row.get("wildtype_length") or 0) != len(sequence):
                raise ValueError("invalid source wild type for {}".format(protein_id))
            result.append(row)
    return result


def single_interval(start, count, length):
    start = max(0, min(int(start), length - count))
    return ((start, start + count - 1),)


def internal_start(length, count, fraction):
    available = length - count
    start = int(round(float(fraction) * available))
    return max(1, min(start, length - count - 1))


def boundary_strength(sequence, boundary):
    left = AA_CLASS.get(sequence[boundary - 1], 4)
    right = AA_CLASS.get(sequence[boundary], 4)
    class_change = 1.0 if left != right else 0.0
    left_window = sequence[max(0, boundary - 4):boundary]
    right_window = sequence[boundary:min(len(sequence), boundary + 4)]
    composition_change = len(set(left_window) ^ set(right_window)) / 8.0
    return class_change + composition_change


def boundary_intervals(sequence, count, excluded):
    length = len(sequence)
    candidates = []
    for boundary in range(2, length - 1):
        start = max(1, min(boundary - count // 2, length - count - 1))
        intervals = single_interval(start, count, length)
        tie = sha256_text(sequence + "|boundary|" + str(boundary))
        candidates.append((-boundary_strength(sequence, boundary), tie, intervals))
    chosen = []
    for _negative_score, _tie, intervals in sorted(candidates):
        if intervals in excluded or intervals in chosen:
            continue
        chosen.append(intervals)
        if len(chosen) == 2:
            return chosen
    raise ValueError("unable to create two unique boundary-oriented plans")


def split_lengths(total, count):
    base, remainder = divmod(total, count)
    return [base + (1 if index < remainder else 0) for index in range(count)]


def multi_intervals(length, deleted_count, span_count):
    widths = split_lengths(deleted_count, span_count)
    retained = length - deleted_count
    gap_count = span_count + 1
    base_gap, gap_remainder = divmod(retained, gap_count)
    gaps = [base_gap + (1 if index < gap_remainder else 0) for index in range(gap_count)]
    intervals = []
    cursor = gaps[0]
    for index, width in enumerate(widths):
        intervals.append((cursor, cursor + width - 1))
        cursor += width + gaps[index + 1]
    if any(width <= 0 for width in widths):
        raise ValueError("deleted count is too small for requested multi-span topology")
    if any(intervals[index][0] <= intervals[index - 1][1] + 1 for index in range(1, len(intervals))):
        raise ValueError("multi-span intervals do not have retained gaps")
    if intervals[-1][1] >= length:
        raise ValueError("multi-span interval out of bounds")
    return tuple(intervals)


def apply_deletion(sequence, intervals):
    pieces = []
    cursor = 0
    for start, end in intervals:
        pieces.append(sequence[cursor:start])
        cursor = end + 1
    pieces.append(sequence[cursor:])
    return "".join(pieces)


def plans_for(sequence):
    length = len(sequence)
    if length < 16:
        raise ValueError("b20 eight-plan construction requires L>=16")
    deleted_count = min(length - 1, int(round(0.20 * length)))
    if deleted_count < 3:
        raise ValueError("b20 deleted count must support three spans")
    plans = [
        ("terminal_n", single_interval(0, deleted_count, length)),
        ("terminal_c", single_interval(length - deleted_count, deleted_count, length)),
        ("internal_20", single_interval(internal_start(length, deleted_count, 0.20), deleted_count, length)),
        ("internal_70", single_interval(internal_start(length, deleted_count, 0.70), deleted_count, length)),
    ]
    excluded = {intervals for _policy, intervals in plans}
    for index, intervals in enumerate(boundary_intervals(sequence, deleted_count, excluded), start=1):
        plans.append(("boundary_{:d}".format(index), intervals))
    plans.append(("multi_2", multi_intervals(length, deleted_count, 2)))
    plans.append(("multi_3", multi_intervals(length, deleted_count, 3)))
    interval_sets = [intervals for _policy, intervals in plans]
    if len(plans) != 8 or len(set(interval_sets)) != 8:
        raise ValueError("b20 plans are not exactly eight unique interval sets")
    for policy, intervals in plans:
        actual = sum(end - start + 1 for start, end in intervals)
        if actual != deleted_count or len(intervals) > 3:
            raise ValueError("invalid b20 plan {}".format(policy))
    return deleted_count, plans


def candidate_rows(source, shard_index):
    sequence = source["wildtype_sequence"]
    protein_id = source["protein_id"]
    split = source_split(source)
    deleted_count, plans = plans_for(sequence)
    wt_hash = sha256_text(sequence)
    rows = []
    for ordinal, (policy, intervals) in enumerate(plans, start=1):
        intervals_json = canonical_json([list(interval) for interval in intervals])
        mutant = apply_deletion(sequence, intervals)
        plan_id = "{}__b20__{:02d}__{}".format(protein_id, ordinal, policy)
        metadata = {
            "schema_version": "delepref_b20_raw_candidate_v1",
            "source_split": split,
            "base_source_split": split,
            "source_candidate_shard_index": shard_index,
            "source_candidate_path": str(source.get("_source_path", "")),
            "beta": 0.20,
            "strict_deleted_count_rule": "min(L-1,round(0.2*L))",
            "teacher_scored": False,
            "candidate_ordinal": ordinal,
            "qmax": 3,
        }
        rows.append({
            "protein_id": protein_id,
            "perturbation_id": plan_id,
            "plan_id": plan_id,
            "wildtype_length": len(sequence),
            "deleted_intervals": intervals_json,
            "num_intervals": len(intervals),
            "deleted_length": deleted_count,
            "deletion_fraction": "{:.10f}".format(deleted_count / float(len(sequence))),
            "budget_bin": "b20",
            "generation_policy": "b20_" + policy,
            "coordinate_system": COORDINATE_SYSTEM,
            "wildtype_sequence": sequence,
            "mutant_sequence": mutant,
            "source_stage1_row_id": "",
            "source_start": "",
            "source_end": "",
            "source_score": "",
            "wildtype_sha256": wt_hash,
            "mutant_sha256": sha256_text(mutant),
            "interval_plan_sha256": sha256_text(intervals_json),
            "metadata_json": canonical_json(metadata),
        })
    return rows


def build(args):
    output = args.output_dir.resolve()
    source_rows = parse_index(args.source_index)
    index_rows = []
    policy_counts = Counter()
    total_proteins = 0
    total_candidates = 0
    for shard_index, source_path, _index_row in source_rows:
        proteins = first_rows_by_protein(source_path)
        rows = []
        for source in proteins:
            source["_source_path"] = str(source_path)
            generated = candidate_rows(source, shard_index)
            rows.extend(generated)
            policy_counts.update(row["generation_policy"] for row in generated)
        if len(rows) != 8 * len(proteins):
            raise RuntimeError("shard {} did not produce exactly 8 candidates per protein".format(shard_index))
        shard_path = output / "shards/deletion_perturbations_shard_{:06d}.csv".format(shard_index)
        if shard_path.exists() and not args.force:
            raise FileExistsError("refusing to overwrite existing raw shard {}".format(shard_path))
        atomic_csv(shard_path, COLUMNS, rows)
        index_rows.append({
            "shard_id": "b20_deletion_perturbations_shard_{:06d}".format(shard_index),
            "shard_index": shard_index,
            "shard_path": str(shard_path.relative_to(REPO)),
            "source_shard_path": str(source_path),
            "num_proteins": len(proteins),
            "num_candidates": len(rows),
            "candidates_per_protein_target": 8,
            "status": "written",
            "metadata_json": canonical_json({"beta": 0.20, "qmax": 3, "teacher_scored": False}),
        })
        total_proteins += len(proteins)
        total_candidates += len(rows)
        print(canonical_json({"event": "b20_raw_shard_written", "shard_index": shard_index, "proteins": len(proteins), "candidates": len(rows)}), flush=True)
    index_path = output / "indexes/deletion_perturbations_shard_index.csv"
    atomic_csv(index_path, INDEX_COLUMNS, index_rows)
    summary = {
        "schema_version": "delepref_b20_raw_candidate_summary_v1",
        "status": "PASS",
        "source_index": str(args.source_index),
        "source_index_sha256": sha256_file(args.source_index),
        "shards": len(index_rows),
        "proteins": total_proteins,
        "candidates": total_candidates,
        "candidates_per_protein": 8,
        "beta": 0.20,
        "deleted_count_rule": "D=min(L-1,round(0.2*L))",
        "qmax": 3,
        "policy_counts": dict(policy_counts),
        "teacher_scored": False,
        "raw_saved_before_teacher_scoring": True,
        "supports_beta_030": False,
        "claim_boundary": "Candidates are teacher queries, not functional labels or guarantees.",
    }
    atomic_json(output / "raw_candidate_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build(args)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
