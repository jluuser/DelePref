#!/usr/bin/env python3
"""Pre-scientific bootstrap for the Student DelePref-21 ProtT5 audit.

This program is deliberately unable to read the DelePref-20 cohort.  Its two modes
only (1) materialize and hash the pinned encoder snapshot plus a task-local
SentencePiece runtime and (2) prove that those assets load offline on the
currently assigned single A40.  A DelePref-21 protocol must not exist while either
mode is running.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/experiments/delepref_step004000_delepref_21_prott5_crossfamily_audit"
FAILURE_DIR = OUT / "audit/failure_or_recovery_history"
MODEL_ROOT = Path("models/prott5_xl_half_uniref50_enc")
HF_CACHE = MODEL_ROOT / "hf_cache"
VENDOR_ROOT = MODEL_ROOT / "runtime_vendor"
VENDOR_SITE = VENDOR_ROOT / "site-packages"
WHEEL_DIR = VENDOR_ROOT / "wheels"

REPO_ID = "Rostlab/prot_t5_xl_half_uniref50-enc"
REVISION = "94a6abc029ae13029317b140b7424e012bf8dfbf"
HF_ENDPOINT = "https://hf-mirror.com"
# The reachable task index exposes a CPython 3.10 wheel for 0.2.0.  This exact
# pin is compatible with Transformers 4.40.2 and is recorded byte-for-byte.
SENTENCEPIECE_VERSION = "0.2.0"
EXPECTED_FILES = (
    ".gitattributes",
    "README.md",
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer_config.json",
)
EXPECTED_ARCHITECTURE = "T5EncoderModel"
BLOCKED_STATUS = "BLOCKED_DelePref_21_PROTT5_BOOTSTRAP_UNAVAILABLE"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path: Path, payload: bytes) -> None:
    ensure_dir(path.parent)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_sha256(files: Sequence[Mapping[str, Any]]) -> str:
    stable = [
        {"relative_path": row["relative_path"], "size": int(row["size"]), "sha256": row["sha256"]}
        for row in files
    ]
    return hashlib.sha256(json.dumps(stable, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()


def check_protocol_absent() -> None:
    forbidden = [OUT / "protocol.json", OUT / "protocol.sha256", OUT / "formal_scientific_started.json"]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(f"bootstrap refuses to run after protocol/science creation: {present}")


def disk_evidence() -> Dict[str, Any]:
    usage = shutil.disk_usage("/public")
    return {
        "path": "/public",
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "free_gib": usage.free / (1024**3),
        "minimum_free_gib": 40,
        "pass": usage.free >= 40 * 1024**3,
    }


def run_logged(command: Sequence[str], log_path: Path, environment: Mapping[str, str] | None = None) -> None:
    ensure_dir(log_path.parent)
    started = utc_now()
    completed = subprocess.run(
        list(command),
        cwd=REPO,
        env=dict(environment) if environment is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    text = "\n".join(
        [
            f"started_at_utc={started}",
            "command=" + json.dumps(list(command)),
            completed.stdout,
            f"exit_code={completed.returncode}",
            f"finished_at_utc={utc_now()}",
            "",
        ]
    )
    atomic_text(log_path, text)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}; see {log_path}")


def vendor_static_inventory() -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    if VENDOR_ROOT.is_dir():
        for path in sorted(VENDOR_ROOT.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            files.append(
                {
                    "relative_path": str(path.relative_to(VENDOR_ROOT)),
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "root": str(VENDOR_ROOT),
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": aggregate_sha256(files),
    }


def install_sentencepiece() -> Dict[str, Any]:
    ensure_dir(WHEEL_DIR)
    ensure_dir(VENDOR_SITE)
    version_tag = SENTENCEPIECE_VERSION.replace(".", "_")
    wheels = sorted(WHEEL_DIR.glob(f"sentencepiece-{SENTENCEPIECE_VERSION}-*.whl"))
    download_performed = False
    if not wheels:
        run_logged(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--no-deps",
                "--retries",
                "10",
                "--timeout",
                "60",
                "--dest",
                str(WHEEL_DIR),
                f"sentencepiece=={SENTENCEPIECE_VERSION}",
            ],
            FAILURE_DIR / f"sentencepiece_{version_tag}_pip_download.log",
        )
        wheels = sorted(WHEEL_DIR.glob(f"sentencepiece-{SENTENCEPIECE_VERSION}-*.whl"))
        download_performed = True
    if len(wheels) != 1:
        raise RuntimeError(f"expected one SentencePiece wheel, found {wheels}")
    wheel = wheels[0]
    marker = VENDOR_ROOT / "sentencepiece_install_complete.json"
    if not marker.is_file():
        run_logged(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--target",
                str(VENDOR_SITE),
                str(wheel),
            ],
            FAILURE_DIR / f"sentencepiece_{version_tag}_pip_install.log",
        )
        atomic_json(
            marker,
            {
                "version": SENTENCEPIECE_VERSION,
                "wheel": str(wheel),
                "wheel_sha256": sha256_file(wheel),
                "installed_at_utc": utc_now(),
                "base_environment_modified": False,
            },
        )
    sys.path.insert(0, str(VENDOR_SITE))
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sentencepiece = importlib.import_module("sentencepiece")
    imported_version = importlib.metadata.version("sentencepiece")
    provenance = {
        "schema_version": "delepref_21_runtime_vendor_provenance_v1",
        "status": "PASS",
        "package": "sentencepiece",
        "requested_version": SENTENCEPIECE_VERSION,
        "imported_version": imported_version,
        "wheel": {
            "path": str(wheel),
            "filename": wheel.name,
            "size": wheel.stat().st_size,
            "sha256": sha256_file(wheel),
            "download_performed_this_invocation": download_performed,
            "source_index": "https://pypi.org/simple/sentencepiece/",
        },
        "install_root": str(VENDOR_SITE),
        "import_file": str(Path(sentencepiece.__file__).resolve()),
        "base_environment_modified": False,
        "base_environment_sentencepiece_present_before_vendor": False,
        "inventory": vendor_static_inventory(),
        "created_at_utc": utc_now(),
    }
    if imported_version != SENTENCEPIECE_VERSION or not str(Path(sentencepiece.__file__).resolve()).startswith(str(VENDOR_SITE)):
        raise RuntimeError("SentencePiece import provenance/version mismatch")
    atomic_json(OUT / "runtime_vendor_provenance.json", provenance)
    return provenance


def model_inventory(snapshot: Path) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for name in EXPECTED_FILES:
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError(f"required ProtT5 snapshot file missing: {path}")
        files.append(
            {
                "relative_path": name,
                "path": str(path),
                "resolved_path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_url": f"{HF_ENDPOINT}/{REPO_ID}/resolve/{REVISION}/{name}",
            }
        )
    extras = sorted(path.name for path in snapshot.iterdir() if path.is_file() and path.name not in EXPECTED_FILES)
    if extras:
        raise RuntimeError(f"unexpected files in pinned snapshot: {extras}")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    architectures = list(config.get("architectures", []))
    if EXPECTED_ARCHITECTURE not in architectures:
        raise RuntimeError(f"snapshot is not the required encoder architecture: {architectures}")
    return {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "snapshot_path": str(snapshot),
        "architecture": EXPECTED_ARCHITECTURE,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(row["size"]) for row in files),
        "aggregate_sha256": aggregate_sha256(files),
    }


def query_model_metadata() -> Dict[str, Any]:
    from huggingface_hub import HfApi  # type: ignore

    errors: List[str] = []
    for attempt in range(1, 9):
        try:
            info = HfApi(endpoint=HF_ENDPOINT).model_info(REPO_ID, revision=REVISION, files_metadata=True, timeout=90)
            if info.sha != REVISION:
                raise RuntimeError(f"resolved revision {info.sha} != pinned {REVISION}")
            siblings = []
            for sibling in info.siblings:
                if sibling.rfilename not in EXPECTED_FILES:
                    continue
                lfs = getattr(sibling, "lfs", None)
                siblings.append(
                    {
                        "relative_path": sibling.rfilename,
                        "api_size": getattr(sibling, "size", None),
                        "lfs": dict(lfs) if lfs else None,
                        "source_url": f"{HF_ENDPOINT}/{REPO_ID}/resolve/{REVISION}/{sibling.rfilename}",
                    }
                )
            if {row["relative_path"] for row in siblings} != set(EXPECTED_FILES):
                raise RuntimeError("model API file inventory does not match the seven-file encoder snapshot")
            return {
                "repo_id": REPO_ID,
                "revision": REVISION,
                "endpoint": HF_ENDPOINT,
                "last_modified": str(info.lastModified),
                "files": siblings,
                "attempt": attempt,
            }
        except Exception as exc:  # preserve all bounded failures
            errors.append(f"attempt={attempt} error={type(exc).__name__}: {exc!r}")
            time.sleep(min(15, attempt * 2))
    atomic_text(FAILURE_DIR / "model_metadata_query_failures.log", "\n".join(errors) + "\n")
    raise RuntimeError("unable to resolve pinned model metadata from configured mirror")


def download_snapshot() -> Dict[str, Any]:
    from huggingface_hub import snapshot_download  # type: ignore

    metadata = query_model_metadata()
    started_wall = time.time()
    started_at = utc_now()
    snapshot_path = Path(
        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            cache_dir=str(HF_CACHE),
            endpoint=HF_ENDPOINT,
            allow_patterns=list(EXPECTED_FILES),
            max_workers=1,
            local_files_only=False,
        )
    )
    finished_at = utc_now()
    inventory = model_inventory(snapshot_path)
    provenance = {
        "schema_version": "delepref_21_model_provenance_v1",
        "status": "BOOTSTRAP_DOWNLOADED_AWAITING_OFFLINE_LOAD_CHECK",
        "model_role": "frozen_cross_family_sequence_representation_proxy",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "endpoint": HF_ENDPOINT,
        "download_method": "huggingface_hub.snapshot_download; pinned revision; one worker; exact seven-file encoder allowlist",
        "download_started_at_utc": started_at,
        "download_finished_at_utc": finished_at,
        "download_elapsed_seconds": time.time() - started_wall,
        "cache_root": str(MODEL_ROOT),
        "hf_cache": str(HF_CACHE),
        "snapshot": inventory,
        "remote_metadata": metadata,
        "offline_required_after_bootstrap": True,
        "quantized": False,
        "lora": False,
        "adapter": False,
        "fine_tuned_for_this_task": False,
        "decoder_checkpoint": False,
        "training": False,
        "created_at_utc": utc_now(),
    }
    atomic_json(OUT / "model_provenance.json", provenance)
    return provenance


def parse_nvidia_smi() -> Dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if query.returncode != 0:
        raise RuntimeError(f"nvidia-smi query failed: {query.stderr}")
    rows = []
    for line in query.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) != 8:
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        rows.append(dict(zip(("index", "name", "uuid", "driver_version", "memory_total_mib", "memory_free_mib", "memory_used_mib", "utilization_gpu_percent"), values)))
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    process_rows = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    return {"gpus": rows, "compute_processes": process_rows}


def offline_gpu_check() -> Dict[str, Any]:
    check_protocol_absent()
    disk = disk_evidence()
    if not disk["pass"]:
        raise RuntimeError("/public has less than 40 GiB free")
    model_provenance_path = OUT / "model_provenance.json"
    runtime_path = OUT / "runtime_vendor_provenance.json"
    if not model_provenance_path.is_file() or not runtime_path.is_file():
        raise RuntimeError("download/vendor provenance missing before offline check")
    model_provenance = json.loads(model_provenance_path.read_text(encoding="utf-8"))
    snapshot = Path(model_provenance["snapshot"]["snapshot_path"])
    before_inventory = model_inventory(snapshot)
    vendor_before = vendor_static_inventory()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(VENDOR_SITE))

    gpu = parse_nvidia_smi()
    one_gpu = len(gpu["gpus"]) == 1
    row = gpu["gpus"][0] if one_gpu else {}
    environment = {
        key: os.environ.get(key, "")
        for key in ("SLURM_JOB_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "CONDA_DEFAULT_ENV", "CONDA_PREFIX")
    }
    checks = {
        "protocol_absent": True,
        "public_free_at_least_40_gib": bool(disk["pass"]),
        "one_visible_gpu_smi": one_gpu,
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
    }
    if not all(checks.values()):
        raise RuntimeError("offline GPU resource gates failed: " + json.dumps(checks, sort_keys=True))

    import sentencepiece  # type: ignore
    import torch  # type: ignore
    import transformers  # type: ignore
    from transformers import T5EncoderModel, T5Tokenizer  # type: ignore

    checks.update(
        {
            "sentencepiece_vendor_import": str(Path(sentencepiece.__file__).resolve()).startswith(str(VENDOR_SITE)),
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "torch_one_visible_gpu": int(torch.cuda.device_count()) == 1,
            "torch_gpu_detected": bool(torch.cuda.is_available()) and bool(torch.cuda.get_device_name(0)),
        }
    )
    if not all(checks.values()):
        raise RuntimeError("runtime import/CUDA gates failed: " + json.dumps(checks, sort_keys=True))

    tokenizer = T5Tokenizer.from_pretrained(str(snapshot), local_files_only=True, do_lower_case=False)
    sample = "A C D E F G H I K L M N P Q R S T V W Y"
    tokenized = tokenizer(sample, add_special_tokens=True, truncation=False, return_attention_mask=True)
    sample_tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
    checks["sample_residue_token_count"] = len(tokenized["input_ids"]) - 1 == 20
    checks["sample_eos_present"] = sample_tokens[-1] == "</s>"
    started_load = time.time()
    model = T5EncoderModel.from_pretrained(str(snapshot), local_files_only=True, torch_dtype=torch.float16)
    model.eval()
    model.to(torch.device("cuda:0"))
    checks["model_is_t5_encoder"] = model.__class__.__name__ == EXPECTED_ARCHITECTURE
    checks["model_eval"] = not model.training
    checks["model_fp16"] = next(model.parameters()).dtype == torch.float16
    model_load_seconds = time.time() - started_load
    loaded_memory = torch.cuda.memory_allocated(0)
    del model, tokenizer
    torch.cuda.empty_cache()
    after_inventory = model_inventory(snapshot)
    vendor_after = vendor_static_inventory()
    checks["model_snapshot_immutable"] = before_inventory["aggregate_sha256"] == after_inventory["aggregate_sha256"]
    checks["runtime_vendor_immutable"] = vendor_before["aggregate_sha256"] == vendor_after["aggregate_sha256"]
    checks["all_offline_load_checks"] = all(checks.values())
    if not checks["all_offline_load_checks"]:
        raise RuntimeError("offline load check failed: " + json.dumps(checks, sort_keys=True))

    result = {
        "schema_version": "delepref_21_bootstrap_offline_load_check_v1",
        "status": "PASS",
        "created_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "environment": environment,
        "disk": disk,
        "nvidia_smi": gpu,
        "checks": checks,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "snapshot_sha256": after_inventory["aggregate_sha256"],
        "runtime_vendor_sha256": vendor_after["aggregate_sha256"],
        "offline_environment": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"},
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "sentencepiece_version": importlib.metadata.version("sentencepiece"),
        "sentencepiece_import_file": str(Path(sentencepiece.__file__).resolve()),
        "model_load_seconds": model_load_seconds,
        "model_loaded_memory_bytes": loaded_memory,
        "sample_preprocessing": {"raw": "ACDEFGHIKLMNPQRSTVWY", "spaced": sample, "tokens": sample_tokens},
        "candidate_cohort_read": False,
        "candidate_forward_performed": False,
        "protocol_created": False,
        "network_used_during_check": False,
    }
    atomic_json(OUT / "audit/bootstrap_offline_load_check.json", result)
    atomic_json(OUT / "gpu_resource_provenance.json", {**result, "schema_version": "delepref_21_gpu_resource_provenance_v1", "phase": "pre_scientific_bootstrap"})
    model_provenance.update(
        {
            "status": "PASS_BOOTSTRAP_COMPLETE",
            "offline_load_check": {
                "path": str(OUT / "audit/bootstrap_offline_load_check.json"),
                "sha256": sha256_file(OUT / "audit/bootstrap_offline_load_check.json"),
            },
            "offline_load_model_class": EXPECTED_ARCHITECTURE,
            "offline_load_dtype": "torch.float16",
            "offline_load_transformers_version": transformers.__version__,
            "offline_load_torch_version": torch.__version__,
            "snapshot_after_offline_load": after_inventory,
            "snapshot_immutable_during_offline_load": True,
        }
    )
    atomic_json(model_provenance_path, model_provenance)
    atomic_json(
        OUT / "bootstrap_status.json",
        {
            "schema_version": "delepref_21_bootstrap_status_v1",
            "status": "PASS_BOOTSTRAP_COMPLETE",
            "repo_id": REPO_ID,
            "revision": REVISION,
            "snapshot_sha256": after_inventory["aggregate_sha256"],
            "runtime_vendor_sha256": vendor_after["aggregate_sha256"],
            "offline": True,
            "protocol_created": False,
            "candidate_cohort_read": False,
            "created_at_utc": utc_now(),
        },
    )
    return result


def record_blocked(phase: str, exc: BaseException) -> None:
    ensure_dir(FAILURE_DIR)
    payload = {
        "schema_version": "delepref_21_bootstrap_failure_v1",
        "status": BLOCKED_STATUS,
        "phase": phase,
        "error_type": type(exc).__name__,
        "error": repr(exc),
        "created_at_utc": utc_now(),
        "protocol_created": (OUT / "protocol.json").exists(),
        "scientific_run_started": (OUT / "formal_scientific_started.json").exists(),
        "pass_sentinel_written": (OUT / "PASS_DelePref_21_PROTT5_CROSSFAMILY_AUDIT").exists(),
    }
    failure_path = FAILURE_DIR / f"bootstrap_{phase}_failure.json"
    if failure_path.exists():
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        failure_path = FAILURE_DIR / f"bootstrap_{phase}_failure_{stamp}.json"
    atomic_json(failure_path, payload)
    atomic_json(OUT / "bootstrap_status.json", payload)


def bootstrap_download() -> Dict[str, Any]:
    check_protocol_absent()
    disk = disk_evidence()
    if not disk["pass"]:
        raise RuntimeError("/public has less than 40 GiB free")
    ensure_dir(FAILURE_DIR)
    prior_status_path = OUT / "bootstrap_status.json"
    if prior_status_path.is_file():
        prior = json.loads(prior_status_path.read_text(encoding="utf-8"))
        if prior.get("status") == BLOCKED_STATUS:
            atomic_json(
                FAILURE_DIR / "bootstrap_recovery_attempt_001.json",
                {
                    "schema_version": "delepref_21_bootstrap_recovery_v1",
                    "status": "RECOVERY_STARTED_BEFORE_PROTOCOL_OR_SCIENCE",
                    "prior_status_sha256": sha256_file(prior_status_path),
                    "prior_failure": prior,
                    "recovery_scope": "replace an unavailable agent-chosen SentencePiece 0.2.1 pin with the available exact 0.2.0 wheel",
                    "scientific_retry": False,
                    "candidate_cohort_read_by_bootstrap": False,
                    "protocol_created": False,
                    "created_at_utc": utc_now(),
                },
            )
    vendor = install_sentencepiece()
    model = download_snapshot()
    status = {
        "schema_version": "delepref_21_bootstrap_status_v1",
        "status": "DOWNLOADED_AWAITING_OFFLINE_LOAD_CHECK",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "snapshot_sha256": model["snapshot"]["aggregate_sha256"],
        "runtime_vendor_sha256": vendor["inventory"]["aggregate_sha256"],
        "disk": disk,
        "protocol_created": False,
        "candidate_cohort_read": False,
        "created_at_utc": utc_now(),
    }
    atomic_json(OUT / "bootstrap_status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--download", action="store_true", help="download/vendor only; never reads cohort or creates protocol")
    modes.add_argument("--offline-gpu-check", action="store_true", help="offline model-load/resource check only; never reads cohort")
    args = parser.parse_args()
    phase = "download" if args.download else "offline_gpu_check"
    try:
        result = bootstrap_download() if args.download else offline_gpu_check()
        print(json.dumps({"event": "DelePref_21_PROTT5_BOOTSTRAP", "phase": phase, "status": result["status"]}, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        record_blocked(phase, exc)
        print(json.dumps({"event": BLOCKED_STATUS, "phase": phase, "error": repr(exc)}, sort_keys=True), flush=True)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
