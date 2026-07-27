#!/usr/bin/env python3
"""Annotate formal teacher preference pair shards with quality-aware weights."""

import argparse
import csv
import glob
import json
import os
import statistics


APPENDED_FIELDS = [
    "global_shift_delta",
    "retained_region_shift_delta",
    "junction_plausibility_delta",
    "num_views_support_a",
    "num_views_support_b",
    "num_views_tie",
    "view_agreement",
    "dominance_type",
    "quality_weight",
    "student_loss_weight",
    "quality_status",
    "quality_reason",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate formal preference-pair shards with view support and loss weights."
    )
    parser.add_argument("--input_pair_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--start_shard", type=int, required=True)
    parser.add_argument("--end_shard", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only rebuild index/summary from existing quality shards; do not read inputs or write shards.",
    )
    parser.add_argument("--view_delta_epsilon", type=float, default=0.005)
    parser.add_argument("--clear_min_support", type=int, default=2)
    parser.add_argument("--tie_margin_threshold", type=float, default=0.05)
    return parser.parse_args()


def shard_name(prefix, shard_id):
    return "%s_%06d.csv" % (prefix, shard_id)


def input_shard_path(input_pair_dir, shard_id):
    return os.path.join(
        input_pair_dir,
        shard_name("teacher_preference_pairs_shard", shard_id),
    )


def output_shard_path(output_dir, shard_id):
    return os.path.join(
        output_dir,
        shard_name("teacher_preference_pairs_quality_shard", shard_id),
    )


def count_data_rows(path):
    with open(path, "r", newline="") as handle:
        count = -1
        for count, _line in enumerate(handle):
            pass
    return max(count, 0)


def count_csv_data_rows(path):
    with open(path, "r", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _row in reader)


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_float(value):
    if value is None:
        raise ValueError("missing float value")
    return float(str(value).strip())


def fmt_float(value):
    return "%.6f" % value


def normalize_preference(value):
    text = str(value).strip().lower()
    if text in {
        "a",
        "a_better",
        "better_a",
        "prefer_a",
        "preferred_a",
        "perturbation_a",
        "perturbation_id_a",
    }:
        return "a"
    if text in {
        "b",
        "b_better",
        "better_b",
        "prefer_b",
        "preferred_b",
        "perturbation_b",
        "perturbation_id_b",
    }:
        return "b"
    if text in {"tie", "equal", "none", "ambiguous"}:
        return "tie"
    return text


def support_from_delta(delta, epsilon):
    if delta > epsilon:
        return "a"
    if delta < -epsilon:
        return "b"
    return "tie"


def view_agreement(num_support_a, num_support_b):
    if num_support_a and num_support_b:
        return "mixed"
    if num_support_a:
        return "supports_a"
    if num_support_b:
        return "supports_b"
    return "all_tie"


def build_invalid_annotation(reason):
    return {
        "global_shift_delta": "",
        "retained_region_shift_delta": "",
        "junction_plausibility_delta": "",
        "num_views_support_a": "0",
        "num_views_support_b": "0",
        "num_views_tie": "0",
        "view_agreement": "invalid",
        "dominance_type": "invalid",
        "quality_weight": "0.000000",
        "student_loss_weight": "0.000000",
        "quality_status": "invalid",
        "quality_reason": reason,
    }


def annotate_row(row, args):
    if not as_bool(row.get("is_valid_pair", "")):
        return build_invalid_annotation("input_pair_invalid")

    try:
        basis = json.loads(row.get("comparison_basis_json", ""))
    except Exception:
        return build_invalid_annotation("comparison_basis_json_parse_error")

    try:
        a = basis["a"]
        b = basis["b"]
        global_shift_delta = as_float(b.get("global_shift")) - as_float(a.get("global_shift"))
        retained_region_shift_delta = as_float(b.get("retained_region_shift")) - as_float(
            a.get("retained_region_shift")
        )
        junction_plausibility_delta = as_float(a.get("junction_plausibility")) - as_float(
            b.get("junction_plausibility")
        )
    except Exception:
        return build_invalid_annotation("missing_or_invalid_teacher_view_value")

    supports = [
        support_from_delta(global_shift_delta, args.view_delta_epsilon),
        support_from_delta(retained_region_shift_delta, args.view_delta_epsilon),
        support_from_delta(junction_plausibility_delta, args.view_delta_epsilon),
    ]
    num_support_a = supports.count("a")
    num_support_b = supports.count("b")
    num_tie = supports.count("tie")
    agreement = view_agreement(num_support_a, num_support_b)

    preference = normalize_preference(row.get("preference", ""))
    try:
        margin = as_float(row.get("preference_margin", "0"))
    except Exception:
        margin = 0.0

    dominant_side = ""
    if num_support_a >= args.clear_min_support and num_support_b == 0:
        dominant_side = "a"
    elif num_support_b >= args.clear_min_support and num_support_a == 0:
        dominant_side = "b"

    if preference == "tie":
        dominance_type = "tie"
        loss_weight = 0.2
        quality_reason = "original_preference_tie"
    elif margin <= args.tie_margin_threshold:
        dominance_type = "tie"
        loss_weight = 0.2
        quality_reason = "preference_margin_within_tie_threshold"
    elif num_support_a and num_support_b:
        dominance_type = "tradeoff"
        loss_weight = 0.35
        quality_reason = "mixed_view_support"
    elif dominant_side and dominant_side == preference:
        dominance_type = "clear_dominance"
        loss_weight = 1.0
        quality_reason = "clear_views_agree_with_preference"
    elif dominant_side and dominant_side != preference:
        dominance_type = "tradeoff"
        loss_weight = 0.25
        quality_reason = "preference_view_conflict"
    else:
        dominance_type = "tradeoff"
        loss_weight = 0.35
        quality_reason = "weak_or_ambiguous_view_support"

    return {
        "global_shift_delta": fmt_float(global_shift_delta),
        "retained_region_shift_delta": fmt_float(retained_region_shift_delta),
        "junction_plausibility_delta": fmt_float(junction_plausibility_delta),
        "num_views_support_a": str(num_support_a),
        "num_views_support_b": str(num_support_b),
        "num_views_tie": str(num_tie),
        "view_agreement": agreement,
        "dominance_type": dominance_type,
        "quality_weight": fmt_float(loss_weight),
        "student_loss_weight": fmt_float(loss_weight),
        "quality_status": "ok",
        "quality_reason": quality_reason,
    }


def add_count(mapping, key, amount=1):
    mapping[key] = mapping.get(key, 0) + amount


def update_summary(summary, row):
    add_count(summary["dominance_type_counts"], row.get("dominance_type", ""))
    add_count(summary["preference_counts"], normalize_preference(row.get("preference", "")))
    if row.get("quality_status") == "invalid":
        summary["invalid_count"] += 1
    try:
        summary["student_loss_weight_sum"] += as_float(row.get("student_loss_weight"))
        summary["student_loss_weight_count"] += 1
    except Exception:
        pass


def process_shard(shard_id, args):
    in_path = input_shard_path(args.input_pair_dir, shard_id)
    out_path = output_shard_path(args.output_dir, shard_id)
    tmp_path = out_path + ".tmp"

    if not os.path.exists(in_path):
        raise FileNotFoundError(in_path)

    if args.resume and os.path.exists(out_path):
        shard_summary = summarize_existing_output(shard_id, in_path, out_path)
        if shard_summary["rows"] == count_csv_data_rows(in_path):
            shard_summary["skipped"] = True
            return {
                **shard_summary,
            }

    input_rows = count_csv_data_rows(in_path)

    shard_summary = {
        "shard_id": shard_id,
        "input_path": in_path,
        "output_path": out_path,
        "rows": 0,
        "skipped": False,
        "dominance_type_counts": {},
        "preference_counts": {},
        "student_loss_weight_sum": 0.0,
        "student_loss_weight_count": 0,
        "invalid_count": 0,
    }

    with open(in_path, "r", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError("missing header in %s" % in_path)
        fieldnames = list(reader.fieldnames)
        for appended in APPENDED_FIELDS:
            if appended not in fieldnames:
                fieldnames.append(appended)

        with open(tmp_path, "w", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                annotation = annotate_row(row, args)
                row.update(annotation)
                writer.writerow(row)
                shard_summary["rows"] += 1
                update_summary(shard_summary, row)

    os.replace(tmp_path, out_path)

    if shard_summary["rows"] != input_rows:
        raise ValueError(
            "row count mismatch for shard %06d: input=%d output=%d"
            % (shard_id, input_rows, shard_summary["rows"])
        )

    return shard_summary


def summarize_existing_output(shard_id, in_path, out_path):
    shard_summary = {
        "shard_id": shard_id,
        "input_path": in_path,
        "output_path": out_path,
        "rows": 0,
        "skipped": False,
        "dominance_type_counts": {},
        "preference_counts": {},
        "student_loss_weight_sum": 0.0,
        "student_loss_weight_count": 0,
        "invalid_count": 0,
    }
    with open(out_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            shard_summary["rows"] += 1
            update_summary(shard_summary, row)
    return shard_summary


def merge_summary(total, shard_summary):
    total["total_shards"] += 1
    total["total_rows"] += shard_summary["rows"]
    total["invalid_count"] += shard_summary["invalid_count"]
    for key, value in shard_summary["dominance_type_counts"].items():
        add_count(total["dominance_type_counts"], key, value)
    for key, value in shard_summary["preference_counts"].items():
        add_count(total["preference_counts"], key, value)
    total["student_loss_weight_sum"] += shard_summary["student_loss_weight_sum"]
    total["student_loss_weight_count"] += shard_summary["student_loss_weight_count"]


def write_index(output_dir, shard_records):
    path = os.path.join(output_dir, "preference_pair_quality_shard_index.csv")
    tmp_path = path + ".tmp"
    fieldnames = ["shard_id", "path", "rows", "skipped"]
    with open(tmp_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in shard_records:
            writer.writerow(
                {
                    "shard_id": "%06d" % record["shard_id"],
                    "path": record["output_path"],
                    "rows": record["rows"],
                    "skipped": str(bool(record["skipped"])).lower(),
                }
            )
    os.replace(tmp_path, path)


def write_summary(output_dir, summary):
    weight_sum = summary.pop("student_loss_weight_sum")
    weight_count = summary.pop("student_loss_weight_count")
    if weight_count:
        summary["student_loss_weight_mean"] = weight_sum / weight_count
    else:
        summary["student_loss_weight_mean"] = 0.0
    path = os.path.join(output_dir, "preference_pair_quality_summary.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def existing_output_glob(output_dir):
    return glob.glob(os.path.join(output_dir, "teacher_preference_pairs_quality_shard_*.csv"))


def main():
    args = parse_args()
    if args.start_shard > args.end_shard:
        raise SystemExit("--start_shard must be <= --end_shard")

    os.makedirs(args.output_dir, exist_ok=True)

    summary = {
        "total_shards": 0,
        "total_rows": 0,
        "dominance_type_counts": {},
        "preference_counts": {},
        "student_loss_weight_sum": 0.0,
        "student_loss_weight_count": 0,
        "invalid_count": 0,
        "failed_shards": [],
    }
    shard_records = []

    for shard_id in range(args.start_shard, args.end_shard + 1):
        try:
            if args.summary_only:
                out_path = output_shard_path(args.output_dir, shard_id)
                if not os.path.exists(out_path):
                    raise FileNotFoundError(out_path)
                shard_summary = summarize_existing_output(
                    shard_id,
                    input_shard_path(args.input_pair_dir, shard_id),
                    out_path,
                )
                shard_summary["skipped"] = True
            else:
                shard_summary = process_shard(shard_id, args)
            shard_records.append(shard_summary)
            merge_summary(summary, shard_summary)
        except Exception as exc:
            summary["failed_shards"].append(
                {"shard_id": "%06d" % shard_id, "error": str(exc)}
            )
            print("FAILED shard %06d: %s" % (shard_id, exc))
            continue

        if shard_id == args.start_shard or shard_id == args.end_shard or (shard_id + 1) % 10 == 0:
            print(
                "progress shard=%06d total_shards=%d total_rows=%d skipped=%s"
                % (
                    shard_id,
                    summary["total_shards"],
                    summary["total_rows"],
                    str(bool(shard_summary["skipped"])).lower(),
                )
            )

    write_index(args.output_dir, shard_records)
    write_summary(args.output_dir, summary)

    print("quality_shards_present:", len(existing_output_glob(args.output_dir)))
    print("failed_shards:", len(summary["failed_shards"]))
    if summary["failed_shards"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
