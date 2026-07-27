#!/usr/bin/env python3
"""Strict-cache real ESMC-600M scoring for DelePref beta=0.20 raw shards."""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import socket
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
SCORER_PATH = REPO / "scripts/teacher_scoring.py"
SPEC = importlib.util.spec_from_file_location("delepref_b20_real_teacher_backend", str(SCORER_PATH))
scorer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)

DEFAULT_RAW_INDEX = REPO / "data/processed/preference_training/delepref/b20_supplement/raw_candidates/indexes/deletion_perturbations_shard_index.csv"
DEFAULT_OUTPUT = REPO / "data/processed/preference_training/delepref/b20_supplement/teacher_scoring"
DEFAULT_MODEL = Path("models/esmc_600m")
DEFAULT_ESM_VENDOR = None

IMPACT_COLUMNS = tuple(scorer.TEACHER_IMPACT_REQUIRED_COLUMNS + scorer.TEACHER_IMPACT_OPTIONAL_COLUMNS)
INDEX_COLUMNS = (
    "shard_id", "shard_index", "source_deletion_perturbations_csv", "output_csv",
    "num_rows", "expected_rows", "status", "runtime_seconds", "metadata_json",
)


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


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), tuple(reader.fieldnames or ())


def parse_raw_index(path):
    rows, _fields = read_csv(path)
    result = []
    for ordinal, row in enumerate(rows):
        shard_index = int(row.get("shard_index") or ordinal)
        shard_path = Path(row["shard_path"])
        if not shard_path.is_absolute():
            shard_path = REPO / shard_path
        result.append({
            "shard_index": shard_index,
            "shard_id": row.get("shard_id") or "b20_{:06d}".format(shard_index),
            "path": shard_path,
            "expected_rows": int(row["num_candidates"]),
            "expected_proteins": int(row["num_proteins"]),
        })
    return result


def impact_path(output_dir, shard_index):
    return Path(output_dir) / "teacher_impacts/teacher_impact_scores_shard_{:06d}.csv".format(shard_index)


def report_path(output_dir, shard_index):
    return Path(output_dir) / "shard_reports/shard_{:06d}.json".format(shard_index)


def canonical_intervals(row):
    intervals = scorer.parse_deleted_intervals_strict(row, "cache identity")
    return canonical_json([list(interval) for interval in intervals])


def strict_identity(row, model_sha256, scoring_config_sha256):
    wildtype = row.get("wildtype_sequence", "")
    mutant = row.get("mutant_sequence", "")
    intervals_json = canonical_intervals(row)
    expected_mutant = scorer.apply_deletion(wildtype, json.loads(intervals_json))
    if not wildtype or mutant != expected_mutant:
        raise ValueError("candidate sequence/cache identity is invalid for {}".format(row.get("perturbation_id")))
    identity = {
        "protein_id": row.get("protein_id", ""),
        "wildtype_sequence_sha256": sha256_text(wildtype),
        "interval_plan": json.loads(intervals_json),
        "interval_plan_sha256": sha256_text(intervals_json),
        "mutant_sequence_sha256": sha256_text(mutant),
        "teacher_model_sha256": model_sha256,
        "teacher_scoring_config_sha256": scoring_config_sha256,
    }
    identity["cache_key_sha256"] = sha256_text(canonical_json(identity))
    return identity


def scoring_config(args, model_sha256):
    if args.esm_runtime_vendor is None:
        raise ValueError("--esm-runtime-vendor must point to the local official ESMC runtime")
    vendor = args.esm_runtime_vendor.resolve()
    vendor_files = (
        vendor / "esm/models/esmc.py",
        vendor / "esm/layers/transformer_stack.py",
        vendor / "esm/layers/attention.py",
        vendor / "esm/tokenization/sequence_tokenizer.py",
    )
    if any(not path.is_file() for path in vendor_files):
        raise FileNotFoundError("ESMC runtime vendor is incomplete: {}".format(vendor))
    payload = {
        "schema_version": "delepref_b20_esmc600m_scoring_config_v1",
        "teacher_backend": scorer.ESMC_OFFICIAL_SAFETENSORS_BACKEND,
        "teacher_scoring_mode": scorer.REAL_PLM_TEACHER_MODE,
        "teacher_model_snapshot": str(args.model_snapshot.resolve()),
        "teacher_model_sha256": model_sha256,
        "dtype": "torch.bfloat16",
        "wt_batch_size": int(args.protein_batch_size),
        "mutant_batch_size": int(args.protein_batch_size) * 8,
        "candidate_count_per_protein": 8,
        "global_metric": "pooled_residue_embedding_cosine_similarity",
        "retained_metric": "mean_retained_residue_cosine_similarity",
        "junction_metric": "wt_mutant_boundary_anchor_cosine_similarity_proxy",
        "risk_input": "absent",
        "sequence_manifold_plausibility": "deferred",
        "raw_candidate_index_sha256": sha256_file(args.raw_index),
        "backend_implementation_sha256": sha256_file(SCORER_PATH),
        "driver_implementation_sha256": sha256_file(Path(__file__).resolve()),
        "esm_runtime_vendor": str(vendor),
        "esm_runtime_file_sha256": {
            str(path.relative_to(vendor)): sha256_file(path) for path in vendor_files
        },
    }
    payload["sha256"] = sha256_text(canonical_json(payload))
    return payload


def validate_raw_rows(rows, expected_rows, expected_proteins):
    if len(rows) != expected_rows:
        raise ValueError("raw candidate row count {} != {}".format(len(rows), expected_rows))
    grouped = defaultdict(list)
    keys = set()
    for row in rows:
        key = (row.get("protein_id", ""), row.get("perturbation_id", ""), row.get("plan_id", ""))
        if key in keys:
            raise ValueError("duplicate raw candidate key {}".format(key))
        keys.add(key)
        grouped[row["protein_id"]].append(row)
        length = int(row["wildtype_length"])
        expected_d = min(length - 1, int(round(0.20 * length)))
        if row.get("budget_bin") != "b20" or int(row["deleted_length"]) != expected_d:
            raise ValueError("candidate is not strict b20")
        if int(row["num_intervals"]) > 3:
            raise ValueError("candidate exceeds Qmax=3")
    if len(grouped) != expected_proteins or any(len(group) != 8 for group in grouped.values()):
        raise ValueError("raw shard does not contain exactly eight candidates per protein")
    return grouped


def validate_cached_output(path, raw_rows, model_sha256, scoring_config_sha256):
    if not Path(path).exists():
        return False
    impacts, fields = read_csv(path)
    if fields != IMPACT_COLUMNS or len(impacts) != len(raw_rows):
        raise ValueError("existing cache schema/row-count mismatch: {}".format(path))
    raw_by_key = {
        (row["protein_id"], row["perturbation_id"], row["plan_id"]): row for row in raw_rows
    }
    seen = set()
    for impact in impacts:
        key = (impact.get("protein_id", ""), impact.get("perturbation_id", ""), impact.get("plan_id", ""))
        if key in seen or key not in raw_by_key or impact.get("scoring_status") != "ok":
            raise ValueError("existing cache key/status mismatch: {}".format(key))
        seen.add(key)
        metadata = json.loads(impact.get("metadata_json") or "{}")
        actual = metadata.get("strict_cache_identity")
        expected = strict_identity(raw_by_key[key], model_sha256, scoring_config_sha256)
        if actual != expected:
            raise ValueError("strict cache identity mismatch for {}".format(key))
    if seen != set(raw_by_key):
        raise ValueError("existing cache omits raw candidates")
    return True


def backend_args(args):
    return SimpleNamespace(
        teacher_model_name_or_path=str(args.model_snapshot.resolve()),
        teacher_backend=scorer.ESMC_OFFICIAL_SAFETENSORS_BACKEND,
        teacher_device="cuda",
        teacher_cache_dir=None,
        teacher_allow_download=False,
        teacher_model="frozen_esmc_600m_delepref",
        teacher_group_by_protein=True,
        teacher_batch_size=int(args.protein_batch_size) * 8,
        deletion_perturbations_csv="",
    )


def score_group_chunk(chunk, score_args, backend, model_sha256, config_sha256):
    protein_ids = [protein_id for protein_id, _rows in chunk]
    wildtypes = []
    mutant_rows = []
    for protein_id, rows in chunk:
        sequences = {row["wildtype_sequence"] for row in rows}
        if len(sequences) != 1:
            raise ValueError("protein {} has inconsistent wild types".format(protein_id))
        wildtypes.append(next(iter(sequences)))
        for row in rows:
            mutant_rows.append(row)
    wt_embeddings = scorer.residue_embeddings_for_sequences(backend, wildtypes, batch_size=len(wildtypes))
    mutant_embeddings = scorer.residue_embeddings_for_sequences(
        backend,
        [row["mutant_sequence"] for row in mutant_rows],
        batch_size=len(mutant_rows),
    )
    wt_by_protein = dict(zip(protein_ids, wt_embeddings))
    results = []
    for row, mutant_embedding in zip(mutant_rows, mutant_embeddings):
        impact = scorer.real_plm_teacher_impact_from_embeddings(
            row, score_args, backend, wt_by_protein[row["protein_id"]], mutant_embedding
        )
        metadata = json.loads(impact["metadata_json"])
        metadata["strict_cache_identity"] = strict_identity(row, model_sha256, config_sha256)
        metadata["teacher_model_sha256"] = model_sha256
        metadata["teacher_scoring_config_sha256"] = config_sha256
        metadata["cache_reuse_requires_exact_identity"] = True
        impact["metadata_json"] = canonical_json(metadata)
        results.append(impact)
    return results


def score_shard(shard, args, score_args, backend, model_sha256, config):
    started = time.time()
    rows, _fields = read_csv(shard["path"])
    grouped = validate_raw_rows(rows, shard["expected_rows"], shard["expected_proteins"])
    output = impact_path(args.output_dir, shard["shard_index"])
    if output.exists():
        if not args.resume:
            raise FileExistsError("refusing to overwrite existing teacher output {}".format(output))
        validate_cached_output(output, rows, model_sha256, config["sha256"])
        status = "resumed_strict_cache_match"
    else:
        ordered = sorted(grouped.items(), key=lambda item: (len(item[1][0]["wildtype_sequence"]), item[0]))
        impacts = []
        for start in range(0, len(ordered), int(args.protein_batch_size)):
            chunk = ordered[start:start + int(args.protein_batch_size)]
            impacts.extend(score_group_chunk(chunk, score_args, backend, model_sha256, config["sha256"]))
            completed = min(start + len(chunk), len(ordered))
            if completed % 100 == 0:
                print(canonical_json({
                    "event": "b20_teacher_progress", "shard_index": shard["shard_index"],
                    "proteins_complete": completed, "proteins_total": len(ordered),
                }), flush=True)
        if len(impacts) != len(rows) or any(row.get("scoring_status") != "ok" for row in impacts):
            raise RuntimeError("teacher did not produce one successful impact per raw candidate")
        atomic_csv(output, IMPACT_COLUMNS, impacts)
        validate_cached_output(output, rows, model_sha256, config["sha256"])
        status = "written_real_esmc600m"
    report = {
        "schema_version": "delepref_b20_teacher_shard_report_v1",
        "status": "PASS",
        "shard_index": shard["shard_index"],
        "raw_candidate_csv": str(shard["path"]),
        "raw_candidate_sha256": sha256_file(shard["path"]),
        "teacher_impact_csv": str(output),
        "teacher_impact_sha256": sha256_file(output),
        "rows": len(rows),
        "proteins": len(grouped),
        "teacher_model_sha256": model_sha256,
        "teacher_scoring_config_sha256": config["sha256"],
        "cache_status": status,
        "runtime_seconds": time.time() - started,
        "real_teacher_scoring": True,
        "teacher_backend": scorer.ESMC_OFFICIAL_SAFETENSORS_BACKEND,
    }
    atomic_json(report_path(args.output_dir, shard["shard_index"]), report)
    return report


def score(args):
    if args.shard_modulus < 1 or not 0 <= args.shard_remainder < args.shard_modulus:
        raise ValueError("invalid shard modulus/remainder")
    model_file = args.model_snapshot / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(str(model_file))
    model_sha256 = args.model_sha256 or sha256_file(model_file)
    config = scoring_config(args, model_sha256)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_config_path = args.output_dir / "teacher_scoring_config.json"
    if existing_config_path.exists():
        existing = json.loads(existing_config_path.read_text(encoding="utf-8"))
        if existing != config:
            existing_impacts = list((args.output_dir / "teacher_impacts").glob("*.csv"))
            if existing_impacts:
                raise ValueError("teacher scoring config is already locked to different content")
            correction = {
                "schema_version": "delepref_b20_teacher_config_correction_v1",
                "reason": "prior_probe_failed_before_any_impact_shard_was_written",
                "prior_config_sha256": sha256_text(canonical_json(existing)),
                "corrected_config_sha256": config["sha256"],
                "impact_shards_present_at_correction": 0,
            }
            atomic_json(args.output_dir / "teacher_scoring_config_correction.json", correction)
            atomic_json(existing_config_path, config)
    else:
        atomic_json(existing_config_path, config)
    shards = [
        shard for shard in parse_raw_index(args.raw_index)
        if shard["shard_index"] % args.shard_modulus == args.shard_remainder
    ]
    if args.max_shards is not None:
        shards = shards[:args.max_shards]
    if not shards:
        raise ValueError("worker has no assigned shards")
    score_args = backend_args(args)
    backend = scorer.load_real_plm_teacher_backend(score_args)
    torch = backend["torch"]
    device = backend["device"]
    gpu = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_uuid": str(torch.cuda.get_device_properties(device).uuid),
    }
    reports = []
    for ordinal, shard in enumerate(shards, start=1):
        report = score_shard(shard, args, score_args, backend, model_sha256, config)
        reports.append(report)
        print(canonical_json({
            "event": "b20_teacher_shard_complete", "ordinal": ordinal,
            "assigned": len(shards), "shard_index": shard["shard_index"],
            "cache_status": report["cache_status"], "runtime_seconds": report["runtime_seconds"],
        }), flush=True)
    worker = {
        "schema_version": "delepref_b20_teacher_worker_report_v1",
        "status": "PASS",
        "worker": {"modulus": args.shard_modulus, "remainder": args.shard_remainder},
        "gpu": gpu,
        "shards": [report["shard_index"] for report in reports],
        "rows": sum(report["rows"] for report in reports),
        "real_teacher_scoring": True,
        "teacher_model_sha256": model_sha256,
        "teacher_scoring_config_sha256": config["sha256"],
    }
    atomic_json(args.output_dir / "worker_reports/worker_m{:02d}_r{:02d}.json".format(args.shard_modulus, args.shard_remainder), worker)
    print(json.dumps(worker, indent=2, sort_keys=True))
    return 0


def finalize(args):
    shards = parse_raw_index(args.raw_index)
    config_path = args.output_dir / "teacher_scoring_config.json"
    if not config_path.exists():
        raise FileNotFoundError(str(config_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_sha256 = config["teacher_model_sha256"]
    index_rows = []
    total_rows = 0
    total_runtime = 0.0
    statuses = Counter()
    for shard in shards:
        raw_rows, _ = read_csv(shard["path"])
        output = impact_path(args.output_dir, shard["shard_index"])
        validate_cached_output(output, raw_rows, model_sha256, config["sha256"])
        report = json.loads(report_path(args.output_dir, shard["shard_index"]).read_text(encoding="utf-8"))
        if report.get("status") != "PASS" or report.get("rows") != shard["expected_rows"]:
            raise ValueError("invalid shard report {}".format(shard["shard_index"]))
        statuses[report["cache_status"]] += 1
        total_runtime += float(report["runtime_seconds"])
        total_rows += len(raw_rows)
        index_rows.append({
            "shard_id": shard["shard_id"],
            "shard_index": shard["shard_index"],
            "source_deletion_perturbations_csv": str(shard["path"]),
            "output_csv": str(output),
            "num_rows": len(raw_rows),
            "expected_rows": shard["expected_rows"],
            "status": "strict_cache_validated",
            "runtime_seconds": "{:.6f}".format(float(report["runtime_seconds"])),
            "metadata_json": canonical_json({
                "real_teacher_scoring": True,
                "teacher_model_sha256": model_sha256,
                "teacher_scoring_config_sha256": config["sha256"],
            }),
        })
    index_path = args.output_dir / "indexes/teacher_impact_shard_index.csv"
    atomic_csv(index_path, INDEX_COLUMNS, index_rows)
    workers = []
    for path in sorted((args.output_dir / "worker_reports").glob("*.json")):
        workers.append(json.loads(path.read_text(encoding="utf-8")))
    gpu_uuids = sorted({worker["gpu"]["device_uuid"] for worker in workers if worker.get("status") == "PASS"})
    summary = {
        "schema_version": "delepref_b20_teacher_scoring_summary_v1",
        "status": "PASS" if len(index_rows) == 300 and total_rows == 2400000 and len(gpu_uuids) == 2 else "FAIL",
        "shards": len(index_rows),
        "proteins": total_rows // 8,
        "rows": total_rows,
        "teacher_model_sha256": model_sha256,
        "teacher_scoring_config_sha256": config["sha256"],
        "strict_cache_identity_fields": [
            "protein_id", "wildtype_sequence_sha256", "interval_plan",
            "interval_plan_sha256", "mutant_sequence_sha256",
            "teacher_model_sha256", "teacher_scoring_config_sha256",
        ],
        "cache_status_counts": dict(statuses),
        "worker_gpu_uuids": gpu_uuids,
        "distinct_gpu_count": len(gpu_uuids),
        "runtime_seconds_sum": total_runtime,
        "real_teacher_scoring": True,
        "teacher_preference_is_ground_truth": False,
    }
    atomic_json(args.output_dir / "teacher_scoring_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-index", type=Path, default=DEFAULT_RAW_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command")
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_MODEL)
    score_parser.add_argument("--model-sha256", default="")
    score_parser.add_argument("--esm-runtime-vendor", type=Path, required=True)
    score_parser.add_argument("--protein-batch-size", type=int, default=4)
    score_parser.add_argument("--shard-modulus", type=int, default=2)
    score_parser.add_argument("--shard-remainder", type=int, required=True)
    score_parser.add_argument("--max-shards", type=int, default=None)
    score_parser.add_argument("--resume", action="store_true")
    subparsers.add_parser("finalize")
    args = parser.parse_args()
    if args.command is None:
        raise SystemExit("a command is required: score or finalize")
    return args


def main():
    args = parse_args()
    return score(args) if args.command == "score" else finalize(args)


if __name__ == "__main__":
    raise SystemExit(main())
