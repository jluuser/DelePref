#!/usr/bin/env python3
"""Prepare EGFR binder deletion-retention benchmark inputs.

This script intentionally keeps the EGFR workflow in one place:

1. Convert the local Adaptyv EGFR parquet file into a standard binder CSV.
2. Write binder-only FASTA input for Stage-1 interval proposal.
3. Write original binder:EGFR complex FASTA files for complex prediction.
4. Optionally apply selected deletion intervals and write deleted complex inputs.

Stage 1 remains a no-budget proposer.  Any target deletion ratio belongs to the
Stage-2 planner that writes the selected interval CSV consumed here.
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict


ACCEPTED_STATUSES = set(["certified", "certified_fixed_ratio", "optimal", "optimal_candidate_capped"])


def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path)


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def clean_sequence(sequence):
    sequence = re.sub(r"[^A-Za-z]", "", sequence or "").upper()
    return sequence


def safe_id(text, fallback):
    text = str(text or "").strip()
    if not text:
        text = str(fallback)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("_")
    return text or str(fallback)


def read_parquet_rows(path):
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        sys.stderr.write(
            "pandas/pyarrow are required to read parquet. Use the project conda python, "
            "for example the Python executable from a Boltz environment.\n"
        )
        raise
    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def read_csv_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    ensure_parent(path)
    fields = list(fields or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_fasta(path, records):
    ensure_parent(path)
    with open(path, "w") as handle:
        for header, sequence in records:
            handle.write(">{}\n{}\n".format(header, sequence))


def write_complex_fastas(rows, out_dir, mode):
    ensure_dir(out_dir)
    manifest = []
    for row in rows:
        binder_id = row["binder_id"]
        complex_id = "{}_{}".format(binder_id, mode)
        fasta_path = os.path.join(out_dir, complex_id + ".fasta")
        binder_sequence = row["deleted_binder_sequence"] if mode == "deleted" else row["binder_sequence"]
        target_sequence = row["target_sequence"]
        with open(fasta_path, "w") as handle:
            handle.write(">{}\n{}:{}\n".format(complex_id, binder_sequence, target_sequence))
        manifest.append(
            {
                "complex_id": complex_id,
                "binder_id": binder_id,
                "mode": mode,
                "complex_fasta": fasta_path,
                "binder_length": len(binder_sequence),
                "target_length": len(target_sequence),
                "pkd": row.get("pkd", ""),
                "kd": row.get("kd", ""),
                "name": row.get("name", ""),
            }
        )
    return manifest


def standardize_egfr_rows(raw_rows, min_pkd=0.0, max_records=None):
    rows = []
    seen = set()
    for idx, raw in enumerate(raw_rows):
        binder = clean_sequence(raw.get("SeqA"))
        target = clean_sequence(raw.get("SeqB"))
        if not binder or not target:
            continue
        name = raw.get("name") or "egfr_binder_{}".format(idx + 1)
        binder_id = safe_id(name, "egfr_binder_{}".format(idx + 1))
        if binder_id in seen:
            binder_id = "{}_{}".format(binder_id, idx + 1)
        seen.add(binder_id)
        try:
            pkd = float(raw.get("pkd"))
        except (TypeError, ValueError):
            pkd = 0.0
        try:
            kd = float(raw.get("kd"))
        except (TypeError, ValueError):
            kd = 0.0
        rows.append(
            {
                "accession": binder_id,
                "binder_id": binder_id,
                "name": name,
                "source_dataset": "adaptyv_egfr_round2",
                "pkd": "{:.9g}".format(pkd),
                "kd": "{:.9g}".format(kd),
                "is_binder": int(pkd >= float(min_pkd) and kd > 0.0),
                "binder_sequence": binder,
                "target_sequence": target,
                "binder_length": len(binder),
                "target_length": len(target),
            }
        )
        if max_records is not None and len(rows) >= int(max_records):
            break
    return rows


def read_selected_intervals(path, profile=None):
    if not path:
        return {}
    grouped = defaultdict(list)
    for row in read_csv_rows(path):
        status = row.get("selection_status") or row.get("solver_status")
        if status and status not in ACCEPTED_STATUSES:
            continue
        if profile and row.get("auto_profile") and row.get("auto_profile") != profile:
            continue
        accession = row.get("accession") or row.get("binder_id") or row.get("protein_id")
        if not accession:
            continue
        try:
            start = int(float(row.get("seg_start") or row.get("start")))
            end = int(float(row.get("seg_end") or row.get("end")))
        except (TypeError, ValueError):
            continue
        grouped[accession].append((start, end))
    return dict(grouped)


def apply_deletions(sequence, intervals):
    intervals = sorted(intervals, key=lambda item: (item[0], item[1]))
    if not intervals:
        return sequence, [], "no_selected_intervals"
    cleaned = []
    last_end = -1
    for start, end in intervals:
        if start < 0 or end < start or end >= len(sequence):
            return sequence, [], "interval_out_of_bounds"
        if start <= last_end:
            return sequence, [], "overlapping_intervals"
        cleaned.append((start, end))
        last_end = end
    keep = []
    cursor = 0
    for start, end in cleaned:
        keep.append(sequence[cursor:start])
        cursor = end + 1
    keep.append(sequence[cursor:])
    return "".join(keep), cleaned, "success"


def build_deleted_rows(rows, selected_intervals):
    deleted_rows = []
    for row in rows:
        binder_id = row["binder_id"]
        deleted, intervals, status = apply_deletions(row["binder_sequence"], selected_intervals.get(binder_id, []))
        deleted_len = len(row["binder_sequence"]) - len(deleted)
        out = dict(row)
        out.update(
            {
                "deletion_status": status,
                "deleted_binder_sequence": deleted,
                "deleted_binder_length": len(deleted),
                "deleted_len": deleted_len,
                "deleted_ratio": "{:.6f}".format(deleted_len / float(max(1, len(row["binder_sequence"])))),
                "deleted_intervals": ";".join("{}-{}".format(start, end) for start, end in intervals),
            }
        )
        deleted_rows.append(out)
    return deleted_rows


def to_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def metrics_by_binder(path):
    grouped = {}
    if not path:
        return grouped
    for row in read_csv_rows(path):
        binder_id = row.get("binder_id")
        if not binder_id:
            complex_id = row.get("complex_id", "")
            for suffix in ("_original", "_deleted"):
                if complex_id.endswith(suffix):
                    binder_id = complex_id[: -len(suffix)]
                    break
        if binder_id:
            grouped[binder_id] = row
    return grouped


def evaluate_complex_retention(args):
    original = metrics_by_binder(args.original_complex_metrics_csv)
    deleted = metrics_by_binder(args.deleted_complex_metrics_csv)
    rows = []
    for binder_id in sorted(set(original) & set(deleted)):
        orig = original[binder_id]
        dele = deleted[binder_id]
        original_contacts = to_float(orig.get("interface_contact_count"), 0.0)
        deleted_contacts = to_float(dele.get("interface_contact_count"), 0.0)
        contact_retention = deleted_contacts / float(max(1.0, original_contacts))
        original_iptm = to_float(orig.get("iptm"), to_float(orig.get("ipTM"), 0.0))
        deleted_iptm = to_float(dele.get("iptm"), to_float(dele.get("ipTM"), 0.0))
        original_pdockq = to_float(orig.get("pdockq"), to_float(orig.get("pDockQ"), 0.0))
        deleted_pdockq = to_float(dele.get("pdockq"), to_float(dele.get("pDockQ"), 0.0))
        original_pae = to_float(orig.get("interface_pae"), to_float(orig.get("mean_interface_pae"), 0.0))
        deleted_pae = to_float(dele.get("interface_pae"), to_float(dele.get("mean_interface_pae"), 0.0))
        delta_iptm = deleted_iptm - original_iptm
        delta_pdockq = deleted_pdockq - original_pdockq
        delta_pae = deleted_pae - original_pae
        retention_pass = (
            contact_retention >= float(args.min_contact_retention)
            and delta_iptm >= -float(args.max_iptm_drop)
            and delta_pdockq >= -float(args.max_pdockq_drop)
            and delta_pae <= float(args.max_interface_pae_increase)
        )
        rows.append(
            {
                "binder_id": binder_id,
                "original_complex_id": orig.get("complex_id", ""),
                "deleted_complex_id": dele.get("complex_id", ""),
                "original_iptm": "{:.6f}".format(original_iptm),
                "deleted_iptm": "{:.6f}".format(deleted_iptm),
                "delta_iptm": "{:.6f}".format(delta_iptm),
                "original_pdockq": "{:.6f}".format(original_pdockq),
                "deleted_pdockq": "{:.6f}".format(deleted_pdockq),
                "delta_pdockq": "{:.6f}".format(delta_pdockq),
                "original_interface_contacts": "{:.6f}".format(original_contacts),
                "deleted_interface_contacts": "{:.6f}".format(deleted_contacts),
                "contact_retention": "{:.6f}".format(contact_retention),
                "original_interface_pae": "{:.6f}".format(original_pae),
                "deleted_interface_pae": "{:.6f}".format(deleted_pae),
                "delta_interface_pae": "{:.6f}".format(delta_pae),
                "complex_retention_pass": int(retention_pass),
            }
        )
    write_csv(args.out_complex_retention_csv, rows)
    ensure_parent(args.out_complex_retention_report)
    passed = sum(int(row["complex_retention_pass"]) for row in rows)
    with open(args.out_complex_retention_report, "w") as handle:
        handle.write("EGFR binder complex-retention evaluation\n\n")
        handle.write("original_complex_metrics_csv: {}\n".format(args.original_complex_metrics_csv))
        handle.write("deleted_complex_metrics_csv: {}\n".format(args.deleted_complex_metrics_csv))
        handle.write("paired_complexes: {}\n".format(len(rows)))
        handle.write("retention_pass: {}\n".format(passed))
        handle.write("retention_pass_rate: {:.6f}\n".format(passed / float(max(1, len(rows)))))
        handle.write("min_contact_retention: {}\n".format(args.min_contact_retention))
        handle.write("max_iptm_drop: {}\n".format(args.max_iptm_drop))
        handle.write("max_pdockq_drop: {}\n".format(args.max_pdockq_drop))
        handle.write("max_interface_pae_increase: {}\n".format(args.max_interface_pae_increase))
        handle.write("\nEGFR_COMPLEX_RETENTION_EVAL_PASS\n")
    print("Complex-retention rows: {}".format(len(rows)))
    print("Wrote {}".format(args.out_complex_retention_csv))
    print("Wrote {}".format(args.out_complex_retention_report))


def write_summary(path, args, rows, deleted_rows=None):
    ensure_parent(path)
    deleted_rows = deleted_rows or []
    with open(path, "w") as handle:
        handle.write("EGFR binder benchmark preparation summary\n\n")
        handle.write("input_parquet: {}\n".format(args.input_parquet))
        handle.write("binder_rows: {}\n".format(len(rows)))
        handle.write("binder_fasta: {}\n".format(args.out_binder_fasta or ""))
        handle.write("original_complex_dir: {}\n".format(args.out_original_complex_dir or ""))
        handle.write("selected_segments_csv: {}\n".format(args.selected_segments_csv or ""))
        handle.write("deleted_rows: {}\n".format(len(deleted_rows)))
        if deleted_rows:
            success = sum(1 for row in deleted_rows if row.get("deletion_status") == "success")
            handle.write("deleted_success_rows: {}\n".format(success))
        handle.write("\nEGFR_BINDER_BENCHMARK_PREP_PASS\n")


def run(args):
    raw_rows = read_parquet_rows(args.input_parquet)
    rows = standardize_egfr_rows(raw_rows, min_pkd=args.min_pkd, max_records=args.max_records)
    binder_fields = [
        "accession",
        "binder_id",
        "name",
        "source_dataset",
        "pkd",
        "kd",
        "is_binder",
        "binder_length",
        "target_length",
        "binder_sequence",
        "target_sequence",
    ]
    write_csv(args.out_binders_csv, rows, binder_fields)
    if args.out_binder_fasta:
        write_fasta(args.out_binder_fasta, [(row["binder_id"], row["binder_sequence"]) for row in rows])
    if args.out_original_complex_dir:
        manifest = write_complex_fastas(rows, args.out_original_complex_dir, mode="original")
        write_csv(args.out_original_complex_manifest_csv, manifest)

    deleted_rows = []
    if args.selected_segments_csv:
        selected = read_selected_intervals(args.selected_segments_csv, profile=args.selected_profile)
        deleted_rows = build_deleted_rows(rows, selected)
        write_csv(args.out_deleted_binders_csv, deleted_rows)
        if args.out_deleted_binder_fasta:
            success_records = [
                (row["binder_id"], row["deleted_binder_sequence"])
                for row in deleted_rows
                if row.get("deletion_status") == "success"
            ]
            write_fasta(args.out_deleted_binder_fasta, success_records)
        if args.out_deleted_complex_dir:
            success_rows = [row for row in deleted_rows if row.get("deletion_status") == "success"]
            deleted_manifest = write_complex_fastas(success_rows, args.out_deleted_complex_dir, mode="deleted")
            write_csv(args.out_deleted_complex_manifest_csv, deleted_manifest)

    write_summary(args.summary_txt, args, rows, deleted_rows=deleted_rows)
    if args.original_complex_metrics_csv and args.deleted_complex_metrics_csv:
        evaluate_complex_retention(args)
    print("Binder rows: {}".format(len(rows)))
    print("Wrote {}".format(args.out_binders_csv))
    if args.out_binder_fasta:
        print("Wrote {}".format(args.out_binder_fasta))
    if args.out_original_complex_dir:
        print("Wrote {}".format(args.out_original_complex_manifest_csv))
    if args.selected_segments_csv:
        print("Deleted rows: {}".format(len(deleted_rows)))
        print("Wrote {}".format(args.out_deleted_binders_csv))
    print("Wrote {}".format(args.summary_txt))


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare EGFR binder deletion-retention benchmark inputs.")
    parser.add_argument("--input_parquet", default="data/external/adaptyv_egfr_round2/train-00000-of-00001.parquet")
    parser.add_argument("--out_binders_csv", default="data/processed/egfr_binder/egfr_round2_binders.csv")
    parser.add_argument("--out_binder_fasta", default="data/processed/egfr_binder/egfr_round2_binders.fasta")
    parser.add_argument("--out_original_complex_dir", default="data/processed/egfr_binder/original_complex_fastas")
    parser.add_argument("--out_original_complex_manifest_csv", default="data/processed/egfr_binder/original_complex_manifest.csv")
    parser.add_argument("--selected_segments_csv", default=None)
    parser.add_argument("--selected_profile", default=None)
    parser.add_argument("--out_deleted_binders_csv", default="data/processed/egfr_binder/egfr_round2_deleted_binders.csv")
    parser.add_argument("--out_deleted_binder_fasta", default="data/processed/egfr_binder/egfr_round2_deleted_binders.fasta")
    parser.add_argument("--out_deleted_complex_dir", default="data/processed/egfr_binder/deleted_complex_fastas")
    parser.add_argument("--out_deleted_complex_manifest_csv", default="data/processed/egfr_binder/deleted_complex_manifest.csv")
    parser.add_argument("--summary_txt", default="data/processed/egfr_binder/egfr_binder_benchmark_summary.txt")
    parser.add_argument("--original_complex_metrics_csv", default=None)
    parser.add_argument("--deleted_complex_metrics_csv", default=None)
    parser.add_argument("--out_complex_retention_csv", default="results/egfr_binder/egfr_complex_retention_metrics.csv")
    parser.add_argument("--out_complex_retention_report", default="results/egfr_binder/EGFR_COMPLEX_RETENTION_REPORT.md")
    parser.add_argument("--min_contact_retention", type=float, default=0.70)
    parser.add_argument("--max_iptm_drop", type=float, default=0.05)
    parser.add_argument("--max_pdockq_drop", type=float, default=0.05)
    parser.add_argument("--max_interface_pae_increase", type=float, default=2.0)
    parser.add_argument("--min_pkd", type=float, default=0.0)
    parser.add_argument("--max_records", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
