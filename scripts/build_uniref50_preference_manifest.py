#!/usr/bin/env python3
"""Build filtered-valid UniRef50 manifests for preference training.

This script prepares natural-protein manifests only. It does not construct
deletion perturbations, run teacher scoring, or train a model.
"""

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import os
import sys
import time
from collections import Counter, defaultdict


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
LENGTH_BINS = [
    ("080_127", 80, 127),
    ("128_191", 128, 191),
    ("192_255", 192, 255),
    ("256_383", 256, 383),
    ("384_511", 384, 511),
    ("512_800", 512, 800),
]
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = [
    "protein_id",
    "fasta_header",
    "sequence",
    "sequence_length",
    "sequence_sha1",
    "split",
    "length_bin",
    "source_fasta",
    "source_record_index",
    "manifest_rank",
]
SHARD_FIELDS = [
    "shard_id",
    "split",
    "length_bin",
    "num_records",
    "manifest_path",
    "shard_key",
    "notes",
]


def parse_bool(value):
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected true/false, got {!r}".format(value))


def sha1_hex(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sequence_sha1(sequence):
    return hashlib.sha1(sequence.encode("ascii")).hexdigest()


def stable_split(seq_sha1):
    value = int(seq_sha1[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "val"
    return "test"


def length_bin_for(length):
    for name, low, high in LENGTH_BINS:
        if low <= length <= high:
            return name
    return None


def parse_protein_id(header):
    return header.split(None, 1)[0]


def iter_fasta(path, max_records=None):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = None
        chunks = []
        record_index = -1
        for line in handle:
            line = line.rstrip("\n\r")
            if line.startswith(">"):
                if header is not None:
                    record_index += 1
                    yield record_index, header, "".join(chunks)
                    if max_records is not None and record_index + 1 >= max_records:
                        return
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            record_index += 1
            if max_records is None or record_index < max_records:
                yield record_index, header, "".join(chunks)


def atomic_writer(path, fieldnames):
    tmp_path = path + ".tmp"
    handle = open(tmp_path, "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    return tmp_path, handle, writer


def replace_tmp(tmp_path, path):
    os.replace(tmp_path, path)


def fail_if_outputs_exist(paths, overwrite):
    if overwrite:
        return
    existing = [path for path in paths if path and os.path.exists(path)]
    if existing:
        raise SystemExit(
            "output exists and --overwrite was not provided: {}".format(
                ", ".join(existing)
            )
        )


def empty_split_bin_counts():
    return {split: {name: 0 for name, _low, _high in LENGTH_BINS} for split in SPLITS}


def nested_counter_to_dict(counter):
    return {str(key): int(value) for key, value in sorted(counter.items())}


def split_length_counts_to_dict(counts):
    return {
        split: {length_bin: int(counts[split].get(length_bin, 0)) for length_bin, _low, _high in LENGTH_BINS}
        for split in SPLITS
    }


def allocate_balanced_quota(counts_by_bin, requested):
    requested = int(requested)
    total_available = sum(int(counts_by_bin.get(name, 0)) for name, _low, _high in LENGTH_BINS)
    requested = min(requested, total_available)
    quotas = {name: 0 for name, _low, _high in LENGTH_BINS}
    if requested <= 0:
        return quotas

    active = [name for name, _low, _high in LENGTH_BINS if int(counts_by_bin.get(name, 0)) > 0]
    remaining = requested
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for name in list(active):
            capacity = int(counts_by_bin.get(name, 0)) - quotas[name]
            if capacity <= 0:
                active.remove(name)
                continue
            add = min(capacity, share, remaining)
            if add > 0:
                quotas[name] += add
                remaining -= add
                progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
    return quotas


def sample_key_int(row, seed_tag, subset_name):
    text = "{}\t{}\t{}\t{}".format(
        row["protein_id"],
        row["sequence_sha1"],
        seed_tag,
        subset_name,
    )
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)


def push_bounded(heap, capacity, key, row):
    if capacity <= 0:
        return
    item = (-key, row["sequence_sha1"], dict(row))
    if len(heap) < capacity:
        heapq.heappush(heap, item)
        return
    current_max = -heap[0][0]
    if key < current_max:
        heapq.heapreplace(heap, item)


def selected_rows_from_heaps(heaps, seed_tag, subset_name):
    rows = []
    for _length_bin, heap in heaps.items():
        rows.extend(item[2] for item in heap)
    rows.sort(key=lambda row: (row["length_bin"], sample_key_int(row, seed_tag, subset_name), row["sequence_sha1"]))
    return rows


def global_top_rows(rows, size, seed_tag, subset_name):
    ranked = list(rows)
    ranked.sort(key=lambda row: (sample_key_int(row, seed_tag, subset_name), row["sequence_sha1"]))
    return ranked[: min(int(size), len(ranked))]


def write_manifest_rows(path, rows):
    tmp_path, handle, writer = atomic_writer(path, MANIFEST_FIELDS)
    try:
        for rank, row in enumerate(rows):
            out = {field: row[field] for field in MANIFEST_FIELDS}
            out["manifest_rank"] = str(rank)
            writer.writerow(out)
    finally:
        handle.close()
    replace_tmp(tmp_path, path)


def validate_manifest_file(path, expected_rows=None, check_duplicate_sha1=False):
    errors = []
    seen_sha1 = set()
    row_count = 0
    split_counts = Counter()
    length_bin_counts = Counter()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [field for field in MANIFEST_FIELDS if field not in fieldnames]
        if missing:
            errors.append("{} missing fields {}".format(path, ",".join(missing)))
            return errors, {}
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            sequence = row["sequence"]
            try:
                length = int(row["sequence_length"])
                rank = int(row["manifest_rank"])
            except ValueError:
                errors.append("{} row {} has non-integer length or rank".format(path, row_number))
                continue
            if rank != row_count - 1:
                errors.append("{} row {} manifest_rank is not zero-based contiguous".format(path, row_number))
            if length != len(sequence):
                errors.append("{} row {} sequence_length mismatch".format(path, row_number))
            if not (80 <= length <= 800):
                errors.append("{} row {} length out of [80,800]".format(path, row_number))
            if not set(sequence).issubset(STANDARD_AA):
                errors.append("{} row {} has non-standard AA".format(path, row_number))
            if row["sequence_sha1"] != sequence_sha1(sequence):
                errors.append("{} row {} sequence_sha1 mismatch".format(path, row_number))
            if row["split"] not in SPLITS:
                errors.append("{} row {} invalid split".format(path, row_number))
            expected_bin = length_bin_for(length)
            if row["length_bin"] != expected_bin:
                errors.append("{} row {} length_bin mismatch".format(path, row_number))
            if check_duplicate_sha1:
                if row["sequence_sha1"] in seen_sha1:
                    errors.append("{} row {} duplicate sequence_sha1".format(path, row_number))
                seen_sha1.add(row["sequence_sha1"])
            split_counts[row["split"]] += 1
            length_bin_counts[row["length_bin"]] += 1
    if expected_rows is not None and row_count != int(expected_rows):
        errors.append("{} row count {} != expected {}".format(path, row_count, expected_rows))
    return errors, {
        "rows": row_count,
        "split_counts": nested_counter_to_dict(split_counts),
        "length_bin_counts": nested_counter_to_dict(length_bin_counts),
        "duplicate_sequence_sha1_checked": bool(check_duplicate_sha1),
    }


def write_shard_index(path, shard_counts, manifest_path, hash_shards):
    tmp_path, handle, writer = atomic_writer(path, SHARD_FIELDS)
    try:
        for split in SPLITS:
            for length_bin, _low, _high in LENGTH_BINS:
                for shard in range(hash_shards):
                    key = (split, length_bin, shard)
                    count = int(shard_counts.get(key, 0))
                    writer.writerow(
                        {
                            "shard_id": "{}__{}__h{:04d}".format(split, length_bin, shard),
                            "split": split,
                            "length_bin": length_bin,
                            "num_records": str(count),
                            "manifest_path": manifest_path,
                            "shard_key": "int(sequence_sha1[:8],16)%{}={}".format(hash_shards, shard),
                            "notes": "hash_shard_count_for_filtered_valid_manifest",
                        }
                    )
    finally:
        handle.close()
    replace_tmp(tmp_path, path)


def scan_and_write_full(args, paths):
    stats = {
        "scanned_records": 0,
        "filtered_valid_records": 0,
        "skipped_empty_header": 0,
        "skipped_empty_sequence": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_nonstandard": 0,
    }
    split_counts = Counter()
    length_bin_counts = Counter()
    split_length_bin_counts = empty_split_bin_counts()
    shard_counts = Counter()

    writer_handle = None
    tmp_path = None
    writer = None
    if args.write_full_manifest:
        tmp_path, writer_handle, writer = atomic_writer(paths["full_manifest"], MANIFEST_FIELDS)

    source_fasta = os.path.abspath(args.input_fasta)
    start_time = time.time()
    try:
        for record_index, header, raw_sequence in iter_fasta(args.input_fasta, max_records=args.max_records):
            stats["scanned_records"] += 1
            if args.progress_every and stats["scanned_records"] % args.progress_every == 0:
                elapsed = time.time() - start_time
                print(
                    "scanned_records={} filtered_valid_records={} elapsed_sec={:.1f}".format(
                        stats["scanned_records"],
                        stats["filtered_valid_records"],
                        elapsed,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            fasta_header = header.strip()
            sequence = "".join(raw_sequence.split()).upper()
            if not fasta_header:
                stats["skipped_empty_header"] += 1
                continue
            if not sequence:
                stats["skipped_empty_sequence"] += 1
                continue
            length = len(sequence)
            if length < args.min_len:
                stats["skipped_too_short"] += 1
                continue
            if length > args.max_len:
                stats["skipped_too_long"] += 1
                continue
            if not set(sequence).issubset(STANDARD_AA):
                stats["skipped_nonstandard"] += 1
                continue
            seq_sha1 = sequence_sha1(sequence)
            split = stable_split(seq_sha1)
            length_bin = length_bin_for(length)
            if length_bin is None:
                if length < args.min_len:
                    stats["skipped_too_short"] += 1
                elif length > args.max_len:
                    stats["skipped_too_long"] += 1
                continue
            shard = int(seq_sha1[:8], 16) % args.hash_shards
            manifest_rank = stats["filtered_valid_records"]
            row = {
                "protein_id": parse_protein_id(fasta_header),
                "fasta_header": fasta_header,
                "sequence": sequence,
                "sequence_length": str(length),
                "sequence_sha1": seq_sha1,
                "split": split,
                "length_bin": length_bin,
                "source_fasta": source_fasta,
                "source_record_index": str(record_index),
                "manifest_rank": str(manifest_rank),
            }
            if writer is not None:
                writer.writerow(row)
                if args.flush_every and (manifest_rank + 1) % args.flush_every == 0:
                    writer_handle.flush()
            stats["filtered_valid_records"] += 1
            split_counts[split] += 1
            length_bin_counts[length_bin] += 1
            split_length_bin_counts[split][length_bin] += 1
            shard_counts[(split, length_bin, shard)] += 1
    except Exception:
        if writer_handle is not None:
            writer_handle.close()
        raise
    finally:
        if writer_handle is not None and not writer_handle.closed:
            writer_handle.close()

    if args.write_full_manifest:
        replace_tmp(tmp_path, paths["full_manifest"])

    return stats, split_counts, length_bin_counts, split_length_bin_counts, shard_counts


def select_subsets_from_full_manifest(args, paths, train_quotas, val_quotas):
    train_heaps = {name: [] for name, _low, _high in LENGTH_BINS}
    val_heaps = {name: [] for name, _low, _high in LENGTH_BINS}
    with open(paths["full_manifest"], newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row["split"]
            length_bin = row["length_bin"]
            if split == "train":
                key = sample_key_int(row, args.seed_tag, "formal_train")
                push_bounded(train_heaps[length_bin], int(train_quotas.get(length_bin, 0)), key, row)
            elif split == "val":
                key = sample_key_int(row, args.seed_tag, "formal_val")
                push_bounded(val_heaps[length_bin], int(val_quotas.get(length_bin, 0)), key, row)
    train_rows = selected_rows_from_heaps(train_heaps, args.seed_tag, "formal_train")
    val_rows = selected_rows_from_heaps(val_heaps, args.seed_tag, "formal_val")
    smoke_rows = global_top_rows(train_rows, args.smoke_size, args.seed_tag, "smoke_train")
    write_manifest_rows(paths["formal_train"], train_rows)
    write_manifest_rows(paths["formal_val"], val_rows)
    write_manifest_rows(paths["smoke"], smoke_rows)
    return train_rows, val_rows, smoke_rows


def build_summary(args, paths, stats, split_counts, length_bin_counts, split_length_bin_counts, shard_counts, train_quotas, val_quotas, validation_summary):
    output_paths = {
        "full_manifest": paths["full_manifest"],
        "formal_train_manifest": paths["formal_train"],
        "formal_val_manifest": paths["formal_val"],
        "smoke_manifest": paths["smoke"],
        "shard_index": paths["shards"],
        "summary_json": paths["summary"],
    }
    shard_totals = Counter()
    for (split, _length_bin, shard), count in shard_counts.items():
        shard_totals["{:04d}".format(shard)] += count
    return {
        "input_fasta": os.path.abspath(args.input_fasta),
        "min_len": args.min_len,
        "max_len": args.max_len,
        "standard_aa_only": True,
        "scanned_records": int(stats["scanned_records"]),
        "filtered_valid_records": int(stats["filtered_valid_records"]),
        "skipped_empty_header": int(stats["skipped_empty_header"]),
        "skipped_empty_sequence": int(stats["skipped_empty_sequence"]),
        "skipped_too_short": int(stats["skipped_too_short"]),
        "skipped_too_long": int(stats["skipped_too_long"]),
        "skipped_nonstandard": int(stats["skipped_nonstandard"]),
        "split_counts": nested_counter_to_dict(split_counts),
        "length_bin_counts": nested_counter_to_dict(length_bin_counts),
        "split_length_bin_counts": split_length_counts_to_dict(split_length_bin_counts),
        "formal_train_requested": int(args.formal_train_size),
        "formal_train_written": int(validation_summary["formal_train"]["rows"]),
        "formal_val_requested": int(args.formal_val_size),
        "formal_val_written": int(validation_summary["formal_val"]["rows"]),
        "smoke_requested": int(args.smoke_size),
        "smoke_written": int(validation_summary["smoke"]["rows"]),
        "hash_shards": int(args.hash_shards),
        "hash_shard_nonempty_count": sum(1 for value in shard_totals.values() if value > 0),
        "formal_train_length_bin_distribution": validation_summary["formal_train"]["length_bin_counts"],
        "formal_val_length_bin_distribution": validation_summary["formal_val"]["length_bin_counts"],
        "smoke_length_bin_distribution": validation_summary["smoke"]["length_bin_counts"],
        "formal_train_quotas": {name: int(train_quotas.get(name, 0)) for name, _low, _high in LENGTH_BINS},
        "formal_val_quotas": {name: int(val_quotas.get(name, 0)) for name, _low, _high in LENGTH_BINS},
        "output_paths": output_paths,
        "deterministic_rules": {
            "split": "int(sequence_sha1[:8],16)%100: 0-79 train, 80-89 val, 90-99 test",
            "length_bins": ["{}:{}-{}".format(name, low, high) for name, low, high in LENGTH_BINS],
            "subset_sampling": "per-split length-bin balanced quotas with smallest sha1(protein_id + sequence_sha1 + seed_tag + subset_name)",
            "seed_tag": args.seed_tag,
            "hash_shard": "int(sequence_sha1[:8],16)%hash_shards",
        },
        "validation": validation_summary,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_fasta", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--formal_train_size", type=int, default=300000)
    parser.add_argument("--formal_val_size", type=int, default=20000)
    parser.add_argument("--smoke_size", type=int, default=1000)
    parser.add_argument("--min_len", type=int, default=80)
    parser.add_argument("--max_len", type=int, default=800)
    parser.add_argument("--hash_shards", type=int, default=1024)
    parser.add_argument("--seed_tag", default="formal_300k_v1")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--write_full_manifest", type=parse_bool, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--progress_every", type=int, default=1000000)
    parser.add_argument("--flush_every", type=int, default=100000)
    args = parser.parse_args(argv)

    if args.min_len < 1 or args.max_len < args.min_len:
        raise SystemExit("invalid length bounds")
    if args.hash_shards <= 0:
        raise SystemExit("--hash_shards must be positive")
    if not os.path.exists(args.input_fasta):
        raise SystemExit("input FASTA does not exist: {}".format(args.input_fasta))
    if not args.write_full_manifest:
        raise SystemExit("--write_full_manifest false is not supported for subset selection in this version")

    os.makedirs(args.output_dir, exist_ok=True)
    paths = {
        "full_manifest": os.path.join(args.output_dir, "uniref50_filtered_valid_manifest.csv"),
        "formal_train": os.path.join(args.output_dir, "uniref50_formal_300k_manifest.csv"),
        "formal_val": os.path.join(args.output_dir, "uniref50_formal_val_20k_manifest.csv"),
        "smoke": os.path.join(args.output_dir, "uniref50_smoke_1k_manifest.csv"),
        "shards": os.path.join(args.output_dir, "uniref50_filtered_valid_shards.csv"),
        "summary": args.summary_json or os.path.join(args.output_dir, "uniref50_manifest_summary.json"),
    }
    fail_if_outputs_exist(list(paths.values()), args.overwrite)

    started = time.time()
    stats, split_counts, length_bin_counts, split_length_bin_counts, shard_counts = scan_and_write_full(args, paths)
    train_quotas = allocate_balanced_quota(split_length_bin_counts["train"], args.formal_train_size)
    val_quotas = allocate_balanced_quota(split_length_bin_counts["val"], args.formal_val_size)
    select_subsets_from_full_manifest(args, paths, train_quotas, val_quotas)
    write_shard_index(paths["shards"], shard_counts, paths["full_manifest"], args.hash_shards)

    validation_summary = {}
    full_errors = []
    if os.path.exists(paths["full_manifest"]):
        with open(paths["full_manifest"], newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
        missing = [field for field in MANIFEST_FIELDS if field not in header]
        if missing:
            full_errors.append("full manifest missing fields {}".format(",".join(missing)))
    validation_summary["full_manifest"] = {
        "rows": int(stats["filtered_valid_records"]),
        "header_checked": True,
        "errors": full_errors,
    }
    for label, path, expected, duplicate_check in [
        ("formal_train", paths["formal_train"], sum(train_quotas.values()), True),
        ("formal_val", paths["formal_val"], sum(val_quotas.values()), False),
        ("smoke", paths["smoke"], min(args.smoke_size, sum(train_quotas.values())), True),
    ]:
        errors, summary = validate_manifest_file(path, expected_rows=expected, check_duplicate_sha1=duplicate_check)
        summary["errors"] = errors
        validation_summary[label] = summary

    validation_errors = []
    for label, summary in validation_summary.items():
        validation_errors.extend("{}: {}".format(label, error) for error in summary.get("errors", []))
    if validation_errors:
        for error in validation_errors:
            print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)

    summary = build_summary(
        args,
        paths,
        stats,
        split_counts,
        length_bin_counts,
        split_length_bin_counts,
        shard_counts,
        train_quotas,
        val_quotas,
        validation_summary,
    )
    summary["runtime_sec"] = round(time.time() - started, 3)
    tmp_summary = paths["summary"] + ".tmp"
    with open(tmp_summary, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    replace_tmp(tmp_summary, paths["summary"])

    print("UNIREF50_PREFERENCE_MANIFEST_BUILD_PASS")
    print("scanned_records: {}".format(summary["scanned_records"]))
    print("filtered_valid_records: {}".format(summary["filtered_valid_records"]))
    print("formal_train_written: {}".format(summary["formal_train_written"]))
    print("formal_val_written: {}".format(summary["formal_val_written"]))
    print("smoke_written: {}".format(summary["smoke_written"]))
    print("output_dir: {}".format(os.path.abspath(args.output_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
