#!/usr/bin/env python3
"""Build the audited base+b20 condition catalog for formal DelePref streaming."""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CORRECTED = REPO / "data/processed/preference_training/delepref"
DEFAULT_BASE_INDEX = CORRECTED / "base_pairs/indexes/preference_pair_shard_index.csv"
DEFAULT_B20_INDEX = CORRECTED / "b20_supplement/pairs/indexes/preference_pair_shard_index.csv"
DEFAULT_SPLIT = CORRECTED / "splits/delepref_sequence_hash_v1/train_dev_manifest.csv"
DEFAULT_OUTPUT = CORRECTED / "condition_catalog/v1"
CATALOG_COLUMNS = (
    "source", "shard_index", "protein_id", "sequence_sha256",
    "wildtype_length", "deleted_length", "beta", "split", "pair_count",
    "condition_id",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def read_pair_index(path, source):
    entries = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for ordinal, row in enumerate(csv.DictReader(handle)):
            index = int(row.get("shard_index") or ordinal)
            entries[index] = {
                "source": source,
                "shard_index": index,
                "pair_path": resolve_path(row["output_csv"]),
                "candidate_path": resolve_path(row["source_candidate_csv"]),
                "expected_pairs": int(row["num_pairs"]),
            }
    if set(entries) != set(range(300)):
        raise ValueError("{} pair index must contain shards 0..299".format(source))
    return entries


def read_split_manifest(path):
    records = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            protein = row["protein_id"]
            if protein in records:
                raise ValueError("duplicate protein in train/dev manifest: {}".format(protein))
            split = row["split"]
            if split not in ("train", "dev"):
                raise ValueError("train/dev manifest contains a forbidden split")
            records[protein] = {
                "sequence_sha256": row["sequence_sha256"],
                "wildtype_length": int(row["wildtype_length"]),
                "split": split,
            }
    if not records:
        raise ValueError("empty train/dev manifest")
    return records


def scan_pair_shard(entry):
    counts = Counter()
    rows = 0
    with Path(entry["pair_path"]).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        protein_column = header.index("protein_id")
        deleted_column = header.index("deleted_length")
        for row in reader:
            rows += 1
            protein = row[protein_column]
            deleted = int(row[deleted_column])
            if not protein or deleted < 1:
                raise ValueError("invalid condition row in {}".format(entry["pair_path"]))
            counts[(protein, deleted)] += 1
    if rows != int(entry["expected_pairs"]):
        raise ValueError("pair index row-count mismatch for {} shard {}".format(entry["source"], entry["shard_index"]))
    return {
        "source": entry["source"],
        "shard_index": entry["shard_index"],
        "pair_path": str(entry["pair_path"]),
        "candidate_path": str(entry["candidate_path"]),
        "pair_rows": rows,
        "conditions": [(protein, deleted, count) for (protein, deleted), count in sorted(counts.items())],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pair-index", type=Path, default=DEFAULT_BASE_INDEX)
    parser.add_argument("--b20-pair-index", type=Path, default=DEFAULT_B20_INDEX)
    parser.add_argument("--train-dev-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    split_records = read_split_manifest(args.train_dev_manifest)
    indexes = {
        "base": read_pair_index(args.base_pair_index, "base"),
        "b20": read_pair_index(args.b20_pair_index, "b20"),
    }
    entries = [indexes[source][index] for source in ("base", "b20") for index in range(300)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan_pair_shard, entry): entry for entry in entries}
        for ordinal, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(json.dumps({
                "event": "condition_catalog_shard_scanned",
                "ordinal": ordinal,
                "total": len(entries),
                "source": result["source"],
                "shard_index": result["shard_index"],
                "conditions": len(result["conditions"]),
            }, sort_keys=True), flush=True)

    catalog_rows = []
    condition_counts = Counter()
    pair_counts = Counter()
    excluded_confirmation_pair_rows = Counter()
    excluded_confirmation_conditions = Counter()
    seen_ids = set()
    shard_condition_counts = []
    for result in sorted(results, key=lambda item: (item["source"], item["shard_index"])):
        included = 0
        for protein, deleted, pair_count in result["conditions"]:
            split = split_records.get(protein)
            if split is None:
                excluded_confirmation_pair_rows[result["source"]] += pair_count
                excluded_confirmation_conditions[result["source"]] += 1
                continue
            length = int(split["wildtype_length"])
            if not 0 < deleted < length:
                raise ValueError("condition deletion count is invalid")
            beta = deleted / float(length)
            identity = "{}\0{}\0{}\0{}".format(
                result["source"], protein, split["sequence_sha256"], deleted
            )
            condition_id = sha256_text(identity)
            if condition_id in seen_ids:
                raise ValueError("duplicate condition identity")
            seen_ids.add(condition_id)
            row = {
                "source": result["source"],
                "shard_index": result["shard_index"],
                "protein_id": protein,
                "sequence_sha256": split["sequence_sha256"],
                "wildtype_length": length,
                "deleted_length": deleted,
                "beta": "{:.17g}".format(beta),
                "split": split["split"],
                "pair_count": pair_count,
                "condition_id": condition_id,
            }
            catalog_rows.append(row)
            condition_counts[(result["source"], split["split"])] += 1
            pair_counts[(result["source"], split["split"])] += pair_count
            included += 1
        shard_condition_counts.append({
            "source": result["source"],
            "shard_index": result["shard_index"],
            "pair_rows": result["pair_rows"],
            "included_conditions": included,
        })
    catalog_rows.sort(key=lambda row: (row["source"], int(row["shard_index"]), row["protein_id"], int(row["deleted_length"])))
    train_base = condition_counts[("base", "train")]
    train_b20 = condition_counts[("b20", "train")]
    if train_b20 < 1 or train_base < 3 * train_b20:
        raise ValueError("train catalog cannot supply exact condition-level 3:1 mixture")
    catalog_path = args.output_dir / "conditions.csv"
    atomic_csv(catalog_path, CATALOG_COLUMNS, catalog_rows)
    payload = {
        "schema_version": "delepref_formal_condition_catalog_summary_v1",
        "created_at_utc": utc_now(),
        "status": "PASS",
        "catalog": {
            "path": str(catalog_path.relative_to(REPO)),
            "sha256": sha256_file(catalog_path),
            "rows": len(catalog_rows),
        },
        "inputs": {
            "base_pair_index": {"path": str(args.base_pair_index.relative_to(REPO)), "sha256": sha256_file(args.base_pair_index)},
            "b20_pair_index": {"path": str(args.b20_pair_index.relative_to(REPO)), "sha256": sha256_file(args.b20_pair_index)},
            "train_dev_manifest": {"path": str(args.train_dev_manifest.relative_to(REPO)), "sha256": sha256_file(args.train_dev_manifest)},
            "confirmation_manifest_opened": False,
            "confirmation_metrics_viewed": False,
        },
        "condition_counts": {
            source: {split: condition_counts[(source, split)] for split in ("train", "dev")}
            for source in ("base", "b20")
        },
        "pair_counts": {
            source: {split: pair_counts[(source, split)] for split in ("train", "dev")}
            for source in ("base", "b20")
        },
        "excluded_confirmation_conditions": dict(excluded_confirmation_conditions),
        "excluded_confirmation_pair_rows": dict(excluded_confirmation_pair_rows),
        "train_base_to_b20_capacity_ratio": train_base / float(train_b20),
        "formal_mixture_ratio": {"base": 3, "b20": 1, "unit": "condition"},
        "shards": {"base": 300, "b20": 300},
        "shard_condition_counts": shard_condition_counts,
    }
    atomic_json(args.output_dir / "summary.json", payload)
    print(json.dumps({
        "status": "PASS",
        "catalog_rows": len(catalog_rows),
        "catalog_sha256": payload["catalog"]["sha256"],
        "condition_counts": payload["condition_counts"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
