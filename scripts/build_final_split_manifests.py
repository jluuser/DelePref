#!/usr/bin/env python3
"""Build and audit immutable DelePref train/dev/confirmation split manifests.

The split unit is the exact wild-type sequence SHA256.  A protein ID and all
duplicate sequence identities therefore receive one assignment across both the
corrected base and beta=0.20 sources.  If an explicit cluster column is ever
added to an input schema this script fails closed instead of silently ignoring
it; the current inputs do not provide such a column.

The confirmation manifest is written separately and is never consumed for
training, tuning, validation, or checkpoint selection.  This script audits
membership and leakage only; it does not compute confirmation metrics.
"""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data/processed/preference_training/delepref"
BASE_PAIR_DIR = DATA_ROOT / "base_pairs"
B20_ROOT = DATA_ROOT / "b20_supplement"
DEFAULT_BASE_INDEX = BASE_PAIR_DIR / "indexes/preference_pair_shard_index.csv"
DEFAULT_B20_RAW_INDEX = B20_ROOT / "raw_candidates/indexes/deletion_perturbations_shard_index.csv"
DEFAULT_B20_PAIR_INDEX = B20_ROOT / "pairs/indexes/preference_pair_shard_index.csv"
DEFAULT_OUTPUT = DATA_ROOT / "splits/delepref_sequence_hash_v1"
DEFAULT_AUDIT = REPO / "results/delepref_esmc600m_same_count/delepref/phase_3/split_audit.json"
DEFAULT_STATE = REPO / "results/delepref_esmc600m_same_count/delepref/phase_3_state.json"

SPLIT_SALT = "delepref-delepref-final-release-sequence-split-v1"
MANIFEST_COLUMNS = (
    "protein_id", "sequence_sha256", "wildtype_length", "split",
    "assignment_bucket", "assignment_digest", "base_shard_index",
    "b20_shard_index",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def read_index(path, path_column):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for ordinal, row in enumerate(csv.DictReader(handle)):
            index = int(row.get("shard_index") or ordinal)
            if index in result:
                raise ValueError("duplicate shard index {} in {}".format(index, path))
            result[index] = resolve_path(row[path_column])
    if not result:
        raise ValueError("empty index: {}".format(path))
    return result


def inspect_header(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        fields = tuple(csv.reader(handle).__next__())
    cluster_fields = tuple(field for field in fields if "cluster" in field.lower())
    return fields, cluster_fields


def scan_candidate_shard(task):
    source, shard_index, path, expected_per_protein = task
    by_protein = {}
    rows = 0
    fields, cluster_fields = inspect_header(path)
    required = {"protein_id", "wildtype_sequence", "wildtype_length"}
    if not required.issubset(fields):
        raise ValueError("candidate schema missing {} in {}".format(sorted(required - set(fields)), path))
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            protein = str(row["protein_id"]).strip()
            sequence = str(row["wildtype_sequence"]).strip()
            length = int(row["wildtype_length"])
            if not protein or not sequence or length != len(sequence):
                raise ValueError("invalid protein/sequence/length in {}".format(path))
            record = by_protein.get(protein)
            if record is None:
                sequence_hash = sha256_text(sequence)
                stored_hash = str(row.get("wildtype_sha256") or "").strip()
                if stored_hash and stored_hash != sequence_hash:
                    raise ValueError("stored sequence hash mismatch for {}".format(protein))
                by_protein[protein] = [sequence, sequence_hash, length, 1]
            else:
                if sequence != record[0] or length != record[2]:
                    raise ValueError("protein has inconsistent WT sequence within shard: {}".format(protein))
                stored_hash = str(row.get("wildtype_sha256") or "").strip()
                if stored_hash and stored_hash != record[1]:
                    raise ValueError("stored sequence hash changed within protein: {}".format(protein))
                record[3] += 1
    bad_counts = [protein for protein, record in by_protein.items() if record[3] != expected_per_protein]
    if bad_counts:
        raise ValueError("{} proteins do not have {} {} candidates in shard {}".format(
            len(bad_counts), expected_per_protein, source, shard_index
        ))
    records = [
        {
            "protein_id": protein,
            "sequence_sha256": record[1],
            "wildtype_length": record[2],
            "candidate_rows": record[3],
            "shard_index": shard_index,
        }
        for protein, record in sorted(by_protein.items())
    ]
    return {
        "source": source,
        "shard_index": shard_index,
        "path": str(path),
        "rows": rows,
        "proteins": len(records),
        "records": records,
        "fields": fields,
        "cluster_fields": cluster_fields,
    }


def scan_pair_shard(task):
    source, shard_index, path = task
    protein_counts = Counter()
    source_splits = Counter()
    fields, cluster_fields = inspect_header(path)
    if len(fields) < 2 or fields[0] != "protein_id" or fields[1] != "source_split":
        raise ValueError("pair CSV must begin with protein_id,source_split: {}".format(path))
    # protein_id and source_split are the first two simple fields.  Limiting
    # split to two separators avoids materializing the other ~47 CSV fields
    # while still scanning every pair row.
    with Path(path).open(encoding="utf-8") as handle:
        next(handle)
        for line_number, line in enumerate(handle, 2):
            pieces = line.split(",", 2)
            if len(pieces) != 3:
                raise ValueError("malformed pair row {} in {}".format(line_number, path))
            protein, source_split = pieces[0].strip(), pieces[1].strip()
            if not protein or not source_split:
                raise ValueError("blank pair protein/split at row {} in {}".format(line_number, path))
            protein_counts[protein] += 1
            source_splits[source_split] += 1
    return {
        "source": source,
        "shard_index": shard_index,
        "path": str(path),
        "rows": sum(protein_counts.values()),
        "protein_counts": dict(protein_counts),
        "source_split_counts": dict(source_splits),
        "fields": fields,
        "cluster_fields": cluster_fields,
    }


def run_tasks(function, tasks, workers, event):
    results = []
    if workers == 1:
        for ordinal, task in enumerate(tasks, 1):
            result = function(task)
            results.append(result)
            print(json.dumps({"event": event, "ordinal": ordinal, "total": len(tasks), "source": result["source"], "shard_index": result["shard_index"]}, sort_keys=True), flush=True)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, task): task for task in tasks}
        for ordinal, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(json.dumps({"event": event, "ordinal": ordinal, "total": len(tasks), "source": result["source"], "shard_index": result["shard_index"]}, sort_keys=True), flush=True)
    return results


def merge_candidate_results(results, source):
    merged = {}
    rows = 0
    cluster_fields = set()
    schema_fields = set()
    shard_proteins = {}
    for result in results:
        if result["source"] != source:
            continue
        rows += int(result["rows"])
        cluster_fields.update(result["cluster_fields"])
        schema_fields.update(result["fields"])
        proteins = set()
        for record in result["records"]:
            protein = record["protein_id"]
            if protein in merged:
                raise ValueError("{} protein occurs in multiple shards: {}".format(source, protein))
            merged[protein] = record
            proteins.add(protein)
        shard_proteins[int(result["shard_index"])] = proteins
    return merged, rows, cluster_fields, schema_fields, shard_proteins


def assignment(sequence_hash, dev_basis_points, confirmation_basis_points):
    digest = sha256_text(SPLIT_SALT + "\0" + sequence_hash)
    bucket = int(digest[:16], 16) % 10000
    if bucket < confirmation_basis_points:
        split = "confirmation"
    elif bucket < confirmation_basis_points + dev_basis_points:
        split = "dev"
    else:
        split = "train"
    return split, bucket, digest


def pairwise_overlap(sets):
    names = sorted(sets)
    return {
        "{}_vs_{}".format(left, right): len(sets[left] & sets[right])
        for index, left in enumerate(names)
        for right in names[index + 1:]
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pair-index", type=Path, default=DEFAULT_BASE_INDEX)
    parser.add_argument("--b20-raw-index", type=Path, default=DEFAULT_B20_RAW_INDEX)
    parser.add_argument("--b20-pair-index", type=Path, default=DEFAULT_B20_PAIR_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--state-output", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dev-basis-points", type=int, default=500)
    parser.add_argument("--confirmation-basis-points", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >=1")
    if args.dev_basis_points <= 0 or args.confirmation_basis_points <= 0:
        raise ValueError("dev and confirmation fractions must be positive")
    if args.dev_basis_points + args.confirmation_basis_points >= 10000:
        raise ValueError("dev+confirmation must leave a non-empty train fraction")

    base_candidates = read_index(args.base_pair_index, "source_candidate_csv")
    base_pairs = read_index(args.base_pair_index, "output_csv")
    b20_candidates = read_index(args.b20_raw_index, "shard_path")
    b20_pairs = read_index(args.b20_pair_index, "output_csv")
    expected_shards = set(range(300))
    for name, mapping in (
        ("base_candidates", base_candidates), ("base_pairs", base_pairs),
        ("b20_candidates", b20_candidates), ("b20_pairs", b20_pairs),
    ):
        if set(mapping) != expected_shards:
            raise ValueError("{} does not contain exactly shards 0..299".format(name))

    candidate_tasks = []
    candidate_tasks.extend(("base", index, base_candidates[index], 64) for index in sorted(base_candidates))
    candidate_tasks.extend(("b20", index, b20_candidates[index], 8) for index in sorted(b20_candidates))
    candidate_results = run_tasks(scan_candidate_shard, candidate_tasks, args.workers, "split_candidate_shard_audited")
    base_records, base_rows, base_clusters, base_fields, base_shard_proteins = merge_candidate_results(candidate_results, "base")
    b20_records, b20_rows, b20_clusters, b20_fields, b20_shard_proteins = merge_candidate_results(candidate_results, "b20")

    if len(base_records) != 300000 or len(b20_records) != 300000:
        raise ValueError("candidate sources do not each contain 300000 proteins")
    if base_rows != 19200000 or b20_rows != 2400000:
        raise ValueError("candidate row totals differ from 19.2M base / 2.4M b20")
    if set(base_records) != set(b20_records):
        raise ValueError("base and b20 protein sets differ")
    for protein in base_records:
        base_record = base_records[protein]
        b20_record = b20_records[protein]
        if (base_record["sequence_sha256"], base_record["wildtype_length"]) != (
            b20_record["sequence_sha256"], b20_record["wildtype_length"]
        ):
            raise ValueError("base/b20 WT identity mismatch: {}".format(protein))

    pair_tasks = []
    pair_tasks.extend(("base", index, base_pairs[index]) for index in sorted(base_pairs))
    pair_tasks.extend(("b20", index, b20_pairs[index]) for index in sorted(b20_pairs))
    pair_results = run_tasks(scan_pair_shard, pair_tasks, args.workers, "split_pair_shard_audited")
    pair_rows = Counter()
    pair_protein_counts = {"base": Counter(), "b20": Counter()}
    inherited_source_splits = {"base": Counter(), "b20": Counter()}
    pair_cluster_fields = set()
    pair_fields = set()
    for result in pair_results:
        source = result["source"]
        index = int(result["shard_index"])
        candidate_proteins = base_shard_proteins[index] if source == "base" else b20_shard_proteins[index]
        unknown = set(result["protein_counts"]) - candidate_proteins
        if unknown:
            raise ValueError("{} pair shard {} has proteins outside its candidate shard".format(source, index))
        pair_rows[source] += int(result["rows"])
        pair_protein_counts[source].update(result["protein_counts"])
        inherited_source_splits[source].update(result["source_split_counts"])
        pair_cluster_fields.update(result["cluster_fields"])
        pair_fields.update(result["fields"])
    if pair_rows["base"] != 19194195 or pair_rows["b20"] != 5191822:
        raise ValueError("pair row totals differ from frozen Phase 1/2 totals")
    if not set(pair_protein_counts["base"]).issubset(base_records):
        raise ValueError("base pair proteins are not a subset of base candidates")
    if not pair_protein_counts["base"]:
        raise ValueError("base pair dataset has no proteins")
    if not set(pair_protein_counts["b20"]).issubset(b20_records):
        raise ValueError("b20 pair proteins are not a subset of b20 candidates")
    if len(pair_protein_counts["b20"]) != 288505:
        raise ValueError("b20 legal-pair protein count differs from Phase 2 coverage result")

    all_cluster_fields = sorted(base_clusters | b20_clusters | pair_cluster_fields)
    if all_cluster_fields:
        raise ValueError("explicit cluster fields now exist and require cluster-aware splitting: {}".format(all_cluster_fields))

    sequence_to_proteins = defaultdict(list)
    for protein, record in base_records.items():
        sequence_to_proteins[record["sequence_sha256"]].append(protein)
    rows_by_split = {"train": [], "dev": [], "confirmation": []}
    split_by_protein = {}
    for sequence_hash in sorted(sequence_to_proteins):
        split, bucket, digest = assignment(sequence_hash, args.dev_basis_points, args.confirmation_basis_points)
        for protein in sorted(sequence_to_proteins[sequence_hash]):
            base_record = base_records[protein]
            b20_record = b20_records[protein]
            row = {
                "protein_id": protein,
                "sequence_sha256": sequence_hash,
                "wildtype_length": base_record["wildtype_length"],
                "split": split,
                "assignment_bucket": bucket,
                "assignment_digest": digest,
                "base_shard_index": base_record["shard_index"],
                "b20_shard_index": b20_record["shard_index"],
            }
            rows_by_split[split].append(row)
            split_by_protein[protein] = split

    protein_sets = {name: {row["protein_id"] for row in rows} for name, rows in rows_by_split.items()}
    sequence_sets = {name: {row["sequence_sha256"] for row in rows} for name, rows in rows_by_split.items()}
    protein_overlap = pairwise_overlap(protein_sets)
    sequence_overlap = pairwise_overlap(sequence_sets)
    if any(protein_overlap.values()) or any(sequence_overlap.values()):
        raise ValueError("split leakage detected")
    if set().union(*protein_sets.values()) != set(base_records):
        raise ValueError("split manifests do not partition all proteins")

    train_dev_rows = sorted(rows_by_split["train"] + rows_by_split["dev"], key=lambda row: row["protein_id"])
    confirmation_rows = sorted(rows_by_split["confirmation"], key=lambda row: row["protein_id"])
    train_dev_path = args.output_dir / "train_dev_manifest.csv"
    confirmation_path = args.output_dir / "confirmation_manifest.csv"
    atomic_csv(train_dev_path, MANIFEST_COLUMNS, train_dev_rows)
    atomic_csv(confirmation_path, MANIFEST_COLUMNS, confirmation_rows)
    train_dev_sha = sha256_file(train_dev_path)
    confirmation_sha = sha256_file(confirmation_path)

    confirmation_lock = {
        "schema_version": "delepref_confirmation_manifest_lock_v1",
        "created_at_utc": utc_now(),
        "status": "LOCKED_UNSEEN",
        "manifest_path": str(confirmation_path.relative_to(REPO)),
        "manifest_sha256": confirmation_sha,
        "proteins": len(confirmation_rows),
        "sequence_groups": len(sequence_sets["confirmation"]),
        "allowed_during_release": ["membership_count", "identity_hash", "leakage_audit"],
        "forbidden_uses": [
            "training", "hyperparameter_tuning", "checkpoint_selection",
            "validation_metric_reporting", "confirmation_metric_inspection",
        ],
        "metrics_computed_or_viewed": False,
    }
    confirmation_lock_path = args.output_dir / "confirmation_manifest_lock.json"
    atomic_json(confirmation_lock_path, confirmation_lock)

    source_row_counts_by_split = {source: Counter() for source in ("base_candidates", "b20_candidates", "base_pairs", "b20_pairs")}
    for protein, record in base_records.items():
        source_row_counts_by_split["base_candidates"][split_by_protein[protein]] += int(record["candidate_rows"])
    for protein, record in b20_records.items():
        source_row_counts_by_split["b20_candidates"][split_by_protein[protein]] += int(record["candidate_rows"])
    for source in ("base", "b20"):
        for protein, count in pair_protein_counts[source].items():
            source_row_counts_by_split[source + "_pairs"][split_by_protein[protein]] += int(count)

    duplicate_sequence_groups = [proteins for proteins in sequence_to_proteins.values() if len(proteins) > 1]
    audit = {
        "schema_version": "delepref_split_audit_v1",
        "created_at_utc": utc_now(),
        "status": "PASS",
        "split_method": {
            "unit": "sequence_sha256",
            "salt": SPLIT_SALT,
            "digest": "sha256(salt + NUL + sequence_sha256)",
            "bucket": "first_64_digest_bits_mod_10000",
            "train_basis_points": 10000 - args.dev_basis_points - args.confirmation_basis_points,
            "dev_basis_points": args.dev_basis_points,
            "confirmation_basis_points": args.confirmation_basis_points,
        },
        "cluster_labels": {
            "available": False,
            "mode": "none_no_explicit_auditable_cluster_column",
            "searched_candidate_fields": sorted(base_fields | b20_fields),
            "searched_pair_fields": sorted(pair_fields),
            "matching_cluster_fields": all_cluster_fields,
            "protein_id_prefix_treated_as_cluster": False,
            "cluster_overlap": 0,
            "cluster_overlap_applicable": False,
            "cluster_units_compared": 0,
        },
        "source_counts": {
            "base_candidate_shards": len(base_candidates),
            "base_candidate_rows": base_rows,
            "b20_candidate_shards": len(b20_candidates),
            "b20_candidate_rows": b20_rows,
            "base_pair_shards": len(base_pairs),
            "base_pair_rows": pair_rows["base"],
            "b20_pair_shards": len(b20_pairs),
            "b20_pair_rows": pair_rows["b20"],
            "proteins": len(base_records),
            "sequence_groups": len(sequence_to_proteins),
            "duplicate_sequence_groups": len(duplicate_sequence_groups),
            "proteins_in_duplicate_sequence_groups": sum(len(group) for group in duplicate_sequence_groups),
            "base_proteins_with_legal_pairs": len(pair_protein_counts["base"]),
            "base_proteins_without_legal_pairs": len(base_records) - len(pair_protein_counts["base"]),
            "b20_proteins_with_legal_pairs": len(pair_protein_counts["b20"]),
            "b20_proteins_without_legal_pairs": len(b20_records) - len(pair_protein_counts["b20"]),
        },
        "split_counts": {name: len(rows_by_split[name]) for name in ("train", "dev", "confirmation")},
        "sequence_group_counts": {name: len(sequence_sets[name]) for name in ("train", "dev", "confirmation")},
        "source_row_counts_by_split": {name: dict(counter) for name, counter in source_row_counts_by_split.items()},
        "legacy_source_split_counts": {name: dict(counter) for name, counter in inherited_source_splits.items()},
        "leakage": {
            "protein_overlap": protein_overlap,
            "protein_overlap_total": sum(protein_overlap.values()),
            "sequence_sha256_overlap": sequence_overlap,
            "sequence_sha256_overlap_total": sum(sequence_overlap.values()),
            "cluster_overlap": 0,
            "cluster_overlap_applicable": False,
            "all_candidates_and_pairs_joined_to_one_manifest_assignment": True,
            "base_b20_sequence_identity_match": True,
        },
        "manifests": {
            "train_dev": {
                "path": str(train_dev_path.relative_to(REPO)),
                "sha256": train_dev_sha,
                "rows": len(train_dev_rows),
            },
            "confirmation": {
                "path": str(confirmation_path.relative_to(REPO)),
                "sha256": confirmation_sha,
                "rows": len(confirmation_rows),
                "status": "LOCKED_UNSEEN",
                "metrics_computed_or_viewed": False,
            },
            "confirmation_lock": {
                "path": str(confirmation_lock_path.relative_to(REPO)),
                "sha256": sha256_file(confirmation_lock_path),
            },
        },
        "confirmation_metrics_computed_or_viewed": False,
        "student_training_started": False,
    }
    atomic_json(args.audit_output, audit)

    summary = {
        "schema_version": "delepref_final_split_manifest_summary_v1",
        "created_at_utc": utc_now(),
        "status": "PASS",
        "audit_path": str(args.audit_output.relative_to(REPO)),
        "audit_sha256": sha256_file(args.audit_output),
        "train_dev_manifest": audit["manifests"]["train_dev"],
        "confirmation_manifest": audit["manifests"]["confirmation"],
        "confirmation_lock": audit["manifests"]["confirmation_lock"],
        "split_counts": audit["split_counts"],
        "sequence_group_counts": audit["sequence_group_counts"],
        "protein_overlap_total": 0,
        "sequence_sha256_overlap_total": 0,
        "cluster_overlap": 0,
        "cluster_labels_available": False,
        "confirmation_metrics_computed_or_viewed": False,
    }
    summary_path = args.output_dir / "split_manifest_summary.json"
    atomic_json(summary_path, summary)

    state = {
        "schema_version": "delepref_phase_state_v1",
        "goal": "DelePref_FINAL_TRAINING_READINESS",
        "phase": 3,
        "status": "PASS",
        "updated_at_utc": utc_now(),
        "details": {
            "gate": "PASS",
            "next_phase_authorized": True,
            "split_unit": "sequence_sha256",
            "split_counts": audit["split_counts"],
            "sequence_group_counts": audit["sequence_group_counts"],
            "protein_overlap": 0,
            "sequence_sha256_overlap": 0,
            "cluster_overlap": 0,
            "cluster_labels_available": False,
            "cluster_mode": "none_no_explicit_auditable_cluster_column",
            "train_dev_manifest": summary["train_dev_manifest"],
            "confirmation_manifest": summary["confirmation_manifest"],
            "confirmation_lock": summary["confirmation_lock"],
            "confirmation_status": "LOCKED_UNSEEN",
            "confirmation_metrics_computed_or_viewed": False,
            "all_base_b20_candidates_pairs_assigned": True,
            "student_training_started": False,
            "audit": {"path": str(args.audit_output.relative_to(REPO)), "sha256": sha256_file(args.audit_output)},
            "summary": {"path": str(summary_path.relative_to(REPO)), "sha256": sha256_file(summary_path)},
        },
    }
    atomic_json(args.state_output, state)
    print(json.dumps({
        "status": "PASS",
        "phase": 3,
        "split_counts": audit["split_counts"],
        "sequence_group_counts": audit["sequence_group_counts"],
        "protein_overlap": 0,
        "sequence_sha256_overlap": 0,
        "cluster_overlap": 0,
        "train_dev_sha256": train_dev_sha,
        "confirmation_sha256": confirmation_sha,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
