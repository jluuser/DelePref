#!/usr/bin/env python3
"""Run frozen-teacher impact scoring shard by shard.

This runner writes counterfactual teacher-impact tables only. Strict same-count
preference pairs are constructed separately by ``build_pair_same_count.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
SCORER = SCRIPT_DIR / "teacher_scoring.py"

INDEX_COLUMNS = (
    "shard_id",
    "shard_index",
    "source_deletion_perturbations_csv",
    "output_csv",
    "num_rows",
    "expected_rows",
    "status",
    "runtime_seconds",
    "metadata_json",
)


def ensure_dir(path):
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def count_rows(path):
    path = Path(path)
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def shard_number_from_id(shard_id):
    text = str(shard_id)
    digits = "".join(character for character in text.split("_")[-1] if character.isdigit())
    if not digits:
        raise ValueError("cannot parse shard index from {!r}".format(shard_id))
    return int(digits)


def impact_path(output_dir, shard_index):
    return str(
        Path(output_dir)
        / "teacher_impacts"
        / "teacher_impact_scores_shard_{:06d}.csv".format(int(shard_index))
    )


def run_command(command, environment, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ {}\n".format(" ".join(command)))
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        runtime = time.time() - started
        handle.write("EXIT_CODE={} RUNTIME_SECONDS={:.3f}\n".format(completed.returncode, runtime))
    return completed.returncode, runtime


def valid_existing_outputs(impact_csv, _pair_csv, expected_impacts, _expected_pairs, impact_only=True):
    if not impact_only:
        return False
    return count_rows(impact_csv) == int(expected_impacts)


def write_index(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in INDEX_COLUMNS} for row in records)
    os.replace(temporary, path)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_summary(path, args, impact_records, failed_shards, runtimes):
    summary = {
        "schema_version": "delepref_teacher_impact_summary_v1",
        "status": "PASS" if not failed_shards else "FAIL",
        "deletion_query_shard_index": str(args.deletion_query_shard_index),
        "output_dir": str(args.output_dir),
        "teacher_model_name_or_path": str(args.teacher_model_name_or_path),
        "teacher_backend": str(args.teacher_backend),
        "teacher_scoring_mode": str(args.teacher_scoring_mode),
        "teacher_batch_size": int(args.teacher_batch_size),
        "impact_only": True,
        "impact_shards_done": sum(
            row["status"] in {"written", "resumed_existing"} for row in impact_records
        ),
        "total_impact_rows": sum(int(row["num_rows"]) for row in impact_records),
        "failed_shards": failed_shards,
        "device_ids": str(args.device_ids),
        "max_workers": int(args.max_workers),
        "runtimes": runtimes,
        "teacher_preference_is_ground_truth": False,
        "training": False,
        "pair_construction": False,
    }
    atomic_json(path, summary)
    return summary


def write_policy_config(path, args):
    atomic_json(
        path,
        {
            "schema_version": "delepref_teacher_scoring_config_v1",
            "teacher_scoring_mode": str(args.teacher_scoring_mode),
            "teacher_backend": str(args.teacher_backend),
            "teacher_model_name_or_path": str(args.teacher_model_name_or_path),
            "local_files_only": True,
            "teacher_group_by_protein": True,
            "teacher_batch_size": int(args.teacher_batch_size),
            "impact_only": True,
            "teacher_preference_is_ground_truth": False,
        },
    )


def build_shard_task(row, _args):
    raw_index = row.get("shard_index")
    shard_index = int(raw_index) if str(raw_index or "").strip() else shard_number_from_id(row.get("shard_id"))
    source_csv = row.get("shard_path") or row.get("source_deletion_perturbations_csv")
    if not source_csv:
        raise ValueError("shard {} has no source CSV".format(shard_index))
    expected_impacts = int(row.get("num_candidates") or count_rows(source_csv))
    return {
        "shard_index": shard_index,
        "shard_id": row.get("shard_id") or "deletion_perturbations_shard_{:06d}".format(shard_index),
        "source_csv": str(source_csv),
        "expected_impacts": expected_impacts,
    }


def run_shard_task(task, args, device_id):
    shard_index = int(task["shard_index"])
    output_csv = impact_path(args.output_dir, shard_index)
    log_path = Path(args.output_dir) / "logs" / "shard_{:06d}.log".format(shard_index)
    expected_rows = int(task["expected_impacts"])
    existing = count_rows(output_csv) == expected_rows
    if args.resume and existing:
        status = "resumed_existing"
        runtime = 0.0
    else:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(device_id)
        command = [
            sys.executable,
            str(SCORER),
            "--build_teacher_impact_scores",
            "--deletion_perturbations_csv",
            str(task["source_csv"]),
            "--teacher_impact_scores_csv",
            output_csv,
            "--teacher_scoring_mode",
            str(args.teacher_scoring_mode),
            "--teacher_model_name_or_path",
            str(args.teacher_model_name_or_path),
            "--teacher_backend",
            str(args.teacher_backend),
            "--teacher_device",
            str(args.teacher_device),
            "--teacher_batch_size",
            str(args.teacher_batch_size),
            "--teacher_group_by_protein",
            "--teacher_resume",
        ]
        code, runtime = run_command(command, environment, log_path)
        if code != 0 or count_rows(output_csv) != expected_rows:
            return {
                "ok": False,
                "failed_shard": {"shard_index": shard_index, "stage": "impact", "exit_code": code},
                "message": "teacher impact scoring failed for shard {:06d}".format(shard_index),
            }
        status = "written"
    metadata = {
        "device_id": str(device_id),
        "teacher_group_by_protein": True,
        "teacher_batch_size": int(args.teacher_batch_size),
    }
    record = {
        "shard_id": str(task["shard_id"]),
        "shard_index": shard_index,
        "source_deletion_perturbations_csv": str(task["source_csv"]),
        "output_csv": output_csv,
        "num_rows": count_rows(output_csv),
        "expected_rows": expected_rows,
        "status": status,
        "runtime_seconds": "{:.3f}".format(runtime),
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    }
    return {
        "ok": True,
        "shard_index": shard_index,
        "message": "teacher impact shard {:06d}: {} rows".format(shard_index, expected_rows),
        "impact_record": record,
        "runtime": {"impact_seconds": runtime},
    }


def write_progress(args, impact_records, _pair_records, failed_shards, runtimes):
    ordered = sorted(impact_records, key=lambda row: int(row["shard_index"]))
    root = Path(args.output_dir)
    write_index(root / "indexes" / "teacher_impact_shard_index.csv", ordered)
    summary = write_summary(root / "run_summary.json", args, ordered, failed_shards, runtimes)
    log_path = root / "logs" / "progress.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")


def run_all(tasks, args, device_ids):
    impact_records = []
    failed_shards = []
    runtimes = {}
    pending = list(tasks)
    active = {}
    with ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
        while pending or active:
            while pending and len(active) < int(args.max_workers):
                task = pending.pop(0)
                used = set(active.values())
                device_id = next(device for device in device_ids if device not in used)
                active[executor.submit(run_shard_task, task, args, device_id)] = device_id
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                result = future.result()
                if not result["ok"]:
                    failed_shards.append(result["failed_shard"])
                    write_progress(args, impact_records, [], failed_shards, runtimes)
                    raise RuntimeError(result["message"])
                impact_records.append(result["impact_record"])
                runtimes[str(result["shard_index"])] = result["runtime"]
                print(result["message"], flush=True)
                write_progress(args, impact_records, [], failed_shards, runtimes)
    return impact_records, failed_shards, runtimes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deletion_query_shard_index", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teacher_model_name_or_path", required=True)
    parser.add_argument("--teacher_backend", default="esmc_official_safetensors")
    parser.add_argument("--teacher_scoring_mode", default="fast_real_plm")
    parser.add_argument("--teacher_device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--teacher_batch_size", type=int, default=16)
    parser.add_argument("--device_ids", default="0")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--impact_only", "--impact-only", action="store_true", default=True)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--only_shard_index", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_workers <= 0:
        raise ValueError("--max_workers must be positive")
    device_ids = [item.strip() for item in str(args.device_ids).split(",") if item.strip()]
    if len(device_ids) < int(args.max_workers):
        raise ValueError("--device_ids must provide at least one unique device per worker")
    rows, _ = read_csv(args.deletion_query_shard_index)
    tasks = [build_shard_task(row, args) for row in rows]
    tasks.sort(key=lambda item: item["shard_index"])
    if args.only_shard_index is not None:
        tasks = [task for task in tasks if task["shard_index"] == int(args.only_shard_index)]
    if args.max_shards is not None:
        tasks = tasks[: int(args.max_shards)]
    if not tasks:
        raise ValueError("no shard index rows selected")

    root = Path(args.output_dir)
    for directory in (root / "teacher_impacts", root / "logs", root / "indexes"):
        directory.mkdir(parents=True, exist_ok=True)
    write_policy_config(root / "teacher_scoring_config.json", args)
    impact_records, failed_shards, runtimes = run_all(tasks, args, device_ids)
    summary = write_summary(root / "run_summary.json", args, impact_records, failed_shards, runtimes)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
