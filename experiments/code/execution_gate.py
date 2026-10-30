"""One-shot, fail-closed Slurm dependencies through the first development comparison.

This dispatcher never submits structure predictions or confirmation evaluation.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from common import AUTHORIZED_ROOT, confined_path, identity, read_jsonl, sha_file, write_json
from cache_plm import candidates, pilot_rows, verified_cache


def read(path):
    return json.loads(Path(path).read_text())


def stage_prerequisites(protocol_path):
    root = AUTHORIZED_ROOT
    protocol = read(protocol_path)
    if Path(protocol["root"]).resolve() != root.resolve():
        raise ValueError("unauthorized protocol root")
    seal = read(root / "configs/execution_freeze.json")
    for name, expected_hash in seal["input_hashes"].items():
        if sha_file(root / name) != expected_hash:
            raise ValueError("frozen execution input changed: " + name)
    asset = read(root / "reports/asset_audit.json")
    manifest = read(root / "reports/manifest_audit.json")
    labels = read(root / "reports/label_audit.json")
    splits = read(root / "manifests/splits.json")
    confirmation = read(root / "reports/confirmation_wt_audit.json")
    if not asset["readiness"]["response_data_work_may_continue"]:
        raise ValueError("core assets not ready")
    if manifest["status"] != "MANIFEST_FROZEN" or labels["status"] != "LABELS_VERIFIED" or splits["status"] != "SPLIT_FROZEN":
        raise ValueError("data stages not verified")
    full_hash = sha_file(root / "manifests/development.jsonl")
    if manifest["candidates"] != seal["counts"]["development_candidates"] or manifest["primary_candidates"] != seal["counts"]["primary_candidates"]:
        raise ValueError("frozen cohort counts changed")
    if splits["train_count"] != seal["counts"]["train_wt"] or splits["validation_count"] != seal["counts"]["validation_wt"]:
        raise ValueError("frozen split counts changed")
    if any(x != full_hash for x in [manifest["manifest_sha256"], labels["manifest_sha256"], splits["manifest_sha256"], confirmation["development_manifest_sha256"]]):
        raise ValueError("development manifest changed")
    if manifest["primary_manifest_sha256"] != sha_file(root / "manifests/development_primary.jsonl"):
        raise ValueError("primary manifest changed")
    if labels["protocol_sha256"] != sha_file(protocol_path) or splits["protocol_sha256"] != sha_file(protocol_path):
        raise ValueError("data protocol changed")
    if confirmation["status"] != "CONFIRMATION_WT_FROZEN" or confirmation["cohort_count"] != 32 or confirmation["independent_families"] != 32:
        raise ValueError("confirmation WT audit incomplete")
    if confirmation["manifest_sha256"] != sha_file(root / "manifests/confirmation_wt_frozen.jsonl"):
        raise ValueError("confirmation cohort changed")
    return protocol


def frozen_code():
    source = AUTHORIZED_ROOT / "code"
    hashes = {str(p.relative_to(source)): sha_file(p) for p in sorted(source.rglob("*.py"))}
    snapshot = confined_path(AUTHORIZED_ROOT / "provenance/frozen_code" / identity(hashes) / "code")
    if not snapshot.exists():
        shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    if any(sha_file(snapshot / name) != expected for name, expected in hashes.items()):
        raise ValueError("source snapshot identity failed")
    return snapshot, hashes


def submit(protocol, script, *arguments):
    command = [str(Path(protocol["environments"]["slurm_bin"]) / "sbatch"), "--parsable", *arguments, str(AUTHORIZED_ROOT / "slurm" / script)]
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if result.returncode:
        raise RuntimeError("Slurm submission failed: " + result.stderr)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise ValueError("unexpected sbatch job ID")
    return {"job_id": job_id, "command": command}


def pilot_evidence(root, expected_rows, protocol_hash, expected_profile):
    reports = [read(root / f"caches/embeddings/pilot_shard{s}_report.json") for s in (0, 1)]
    rows = [r for s in (0, 1) for r in read_jsonl(root / f"caches/embeddings/pilot_shard{s}_index.jsonl")]
    if len(rows) != 12 or len({r["sequence_sha256"] for r in rows}) != 12:
        raise ValueError("pilot did not complete 12 distinct required sequences")
    expected_identities = {(r["sequence_sha256"], r["length"]) for r in expected_rows}
    if {(r["sequence_sha256"], r["length"]) for r in rows} != expected_identities:
        raise ValueError("pilot sequences differ from the frozen authorized cohort")
    timings = [t for r in reports for t in r["timings"]]
    if any(r["failures"] for r in reports) or not timings or max(t["batch"] for t in timings) != 4:
        raise ValueError("pilot failed or full microbatch was not measured")
    peak = max(t["peak_bytes"] for t in timings)
    if peak > 40 * 1024 ** 3:
        raise ValueError("pilot exceeds reserved A40 memory margin")
    profiles = {identity(r["profile"]) for r in reports}
    if len(profiles) != 1:
        raise ValueError("pilot profiles differ")
    profile = reports[0]["profile"]
    if any(profile.get(k) != v for k, v in expected_profile.items()):
        raise ValueError("pilot profile differs from authorized current computation")
    for row in rows:
        if row.get("protocol_sha256") != protocol_hash or any(row.get(k) != v for k, v in profile.items()):
            raise ValueError("pilot cache protocol/profile mismatch")
        expected = {k: v for k, v in row.items() if k not in ("key", "path", "reused")}
        if not verified_cache(Path(row["path"]), Path(row["path"]).with_suffix(".json"), expected):
            raise ValueError("pilot cache integrity failed")
    return reports, timings, peak


def after_pilot(protocol, protocol_path):
    root = AUTHORIZED_ROOT
    sequences = candidates(read_jsonl(root / "manifests/development.jsonl"))
    import importlib.metadata
    import inspect
    import torch
    from adapters.esmc_runtime import tokenizer_profile
    source = read(root / "provenance/source_files.json")["files"]["preference_core"]
    if sha_file(source["path"]) != source["sha256"]:
        raise ValueError("frozen PLM loader changed")
    weights = Path(protocol["weights"]["esmc"])
    expected_profile = {"weight_sha256": identity({p.name: sha_file(p) for p in sorted(weights.glob("*.safetensors"))}),
                        **tokenizer_profile(), "implementation_sha256": sha_file(Path(__file__).parent / "cache_plm.py"),
                        "torch_version": str(torch.__version__), "esm_version": importlib.metadata.version("esm"),
                        "hardware": "NVIDIA A40", "compute_precision": "bfloat16", "cache_precision": "float32",
                        "output_layer": "ESMC.embeddings", "special_token_policy": "assert_L_plus_2_remove_1_minus_1",
                        "batch_profile": {"equal_lengths": True, "batch_size": 4}}
    expected_pilot = [row for _, row in pilot_rows(sequences, 4)]
    reports, timings, peak = pilot_evidence(root, expected_pilot, sha_file(protocol_path), expected_profile)
    counts = {}
    for row in sequences:
        counts[row["length"]] = counts.get(row["length"], 0) + 1
    estimated_gpu_seconds = 0.
    for length, count in counts.items():
        nearest = min(timings, key=lambda t: abs(t["length"] - length))
        scale = max(1., (length / nearest["length"]) ** 2)
        estimated_gpu_seconds += math.ceil(count / 4) * nearest["seconds"] * scale
    wall_hours = max(1, math.ceil(estimated_gpu_seconds / 2 / 3600 * 2 + .25))
    if wall_hours > 12:
        raise ValueError("measured PLM cost exceeds automatic 12-hour job envelope")
    bytes_required = sum(r["length"] * 1152 * 4 for r in sequences)
    if shutil.disk_usage(root).free < bytes_required * 1.2:
        raise ValueError("insufficient cache storage")
    snapshot, hashes = frozen_code()
    full = submit(protocol, "cache_plm.sbatch", f"--time={wall_hours:02d}:00:00", f"--export=ALL,CACHE_MODE=full,CODE_DIR={snapshot}")
    gate = submit(protocol, "execution_gate.sbatch", f"--dependency=afterany:{full['job_id']}", "--export=ALL,EXECUTION_STAGE=after_cache")
    return {"status": "FULL_CACHE_SUBMITTED", "pilot_peak_bytes": peak, "pilot_timings": timings,
            "sequence_count": len(sequences), "estimated_total_gpu_hours": estimated_gpu_seconds / 3600,
            "estimated_wall_hours_with_margin": wall_hours, "full_cache": full, "next_gate": gate,
            "code_snapshot": str(snapshot), "code_hashes": hashes, "structure_calls": 0}


def after_cache(protocol, protocol_path):
    root = AUTHORIZED_ROOT; dispatch = read(root / "reports/dispatch_after_pilot.json")
    snapshot = Path(dispatch["code_snapshot"])
    if any(sha_file(snapshot / name) != expected for name, expected in dispatch["code_hashes"].items()):
        raise ValueError("frozen computational code changed")
    indices = [root / f"caches/embeddings/full_shard{s}_index.jsonl" for s in (0, 1)]
    rows = [row for path in indices for row in read_jsonl(path)]
    expected = {r["sequence_sha256"] for r in candidates(read_jsonl(root / "manifests/development.jsonl"))}
    if len(expected) != read(root / "configs/execution_freeze.json")["counts"]["unique_plm_sequences"]:
        raise ValueError("frozen PLM sequence count changed")
    if len(rows) != len(expected) or {r["sequence_sha256"] for r in rows} != expected:
        raise ValueError("full embedding cache is incomplete")
    for s in (0, 1):
        if read(root / f"caches/embeddings/full_shard{s}_report.json")["failures"]:
            raise ValueError("full cache has failures")
    for row in rows:
        expected_meta = {k: v for k, v in row.items() if k not in ("key", "path", "reused")}
        if not verified_cache(Path(row["path"]), Path(row["path"]).with_suffix(".json"), expected_meta):
            raise ValueError("full cache integrity failed")
    subprocess.run([sys.executable, "-m", "pytest", str(snapshot / "tests"), "-q"], cwd=snapshot, check=True)
    for arm in "SABCD":
        command = [sys.executable, str(snapshot / "train.py"), "--protocol", str(protocol_path),
                   "--manifest", str(root / "manifests/development_primary.jsonl"), "--splits", str(root / "manifests/splits.json"),
                   "--labels", str(root / "labels/pairs"), "--label-summary", str(root / "reports/label_candidates.jsonl"),
                   "--cache-index", *(str(p) for p in indices), "--arm", arm, "--lr", "0.0003", "--seed", "20261013",
                   "--output", str(root / "checkpoints/technical" / arm), "--technical", "--steps", "5", "--device", "cpu"]
        subprocess.run(command, check=True)
    train = submit(protocol, "train.sbatch", f"--export=ALL,CODE_DIR={snapshot}")
    gate = submit(protocol, "execution_gate.sbatch", f"--dependency=afterany:{train['job_id']}", "--export=ALL,EXECUTION_STAGE=after_train")
    return {"status": "FIRST_SEED_DEVELOPMENT_SUBMITTED", "sequences_verified": len(rows), "technical_arms_passed": list("SABCD"),
            "training": train, "next_gate": gate, "code_snapshot": str(snapshot), "structure_calls": 0}


def after_train(protocol, protocol_path):
    root = AUTHORIZED_ROOT; models = {}
    for arm in "SABCD":
        runs = []
        for lr in ("0.0003", "0.001"):
            directory = root / f"checkpoints/{arm}_s20261013_lr{lr}"
            status = read(directory / "status.json")
            if status["status"] != "DEVELOPMENT_RUN_COMPLETE":
                raise ValueError("one or more development runs incomplete")
            trace = read_jsonl(directory / "validation_trace.jsonl")
            best = min(trace, key=lambda row: row["ce"])
            report_path = directory / f"validation_{best['step']}/report.json"
            runs.append({"lr": float(lr), "best_ce": status["best_ce"], "steps": status["steps"],
                         "report": read(report_path), "report_path": str(report_path), "checkpoint_path": str(directory / "best.pt"),
                         "checkpoint_sha256": sha_file(directory / "best.pt")})
        models[arm] = {"selected": min(runs, key=lambda row: row["best_ce"]), "runs": runs}
    c = models["C"]["selected"]["best_ce"]
    comparisons = {f"C-{arm}": {"validation_ce_difference": c - models[arm]["selected"]["best_ce"]} for arm in "SABD"}
    return {"status": "FIRST_SEED_DEVELOPMENT_READY_FOR_EVIDENCE_REVIEW", "models": models, "comparisons": comparisons,
            "confirmation_folding_authorized_by_this_dispatcher": False,
            "next_step": "Review convergence, per-protein G/T decisions and paired intervals; apply development evidence gate before sensitivity, three seeds and any confirmation folds.",
            "structure_calls": 0}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True)
    parser.add_argument("--stage", choices=("after_pilot", "after_cache", "after_train"), required=True)
    args = parser.parse_args(); root = AUTHORIZED_ROOT
    lock_path = confined_path(root / "ledger/development_dispatch.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output = root / f"reports/dispatch_{args.stage}.json"
        if output.exists():
            print("stage already attempted; no automatic duplicate submission", flush=True)
            return
        write_json(output, {"status": "EXECUTION_GATE_RUNNING", "stage": args.stage, "structure_calls": 0})
        try:
            protocol = stage_prerequisites(args.protocol)
            report = globals()[args.stage](protocol, Path(args.protocol))
            write_json(output, report)
            print(json.dumps(report), flush=True)
        except Exception as error:
            write_json(output, {"status": "EXECUTION_GATE_FAILED", "stage": args.stage, "error": repr(error), "structure_calls": 0})
            raise


if __name__ == "__main__":
    main()
