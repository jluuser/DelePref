"""Portable runtime provenance helpers used by DelePref experiments."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def collect_runtime_provenance(
    *,
    require_cuda: bool = True,
    expected_gpu_count: int | None = None,
    minimum_gpu_memory_gib: float | None = None,
) -> dict[str, Any]:
    import torch

    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )

    executable = shutil.which("nvidia-smi")
    command = [
        executable,
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader",
    ] if executable else []
    result = subprocess.run(command, capture_output=True, text=True, check=False) if command else None

    checks = {"cuda_available": bool(torch.cuda.is_available()) or not require_cuda}
    if expected_gpu_count is not None:
        checks["gpu_count"] = len(devices) == int(expected_gpu_count)
    if minimum_gpu_memory_gib is not None:
        minimum_bytes = float(minimum_gpu_memory_gib) * 1024**3
        checks["gpu_memory"] = bool(devices) and all(
            device["total_memory_bytes"] >= minimum_bytes for device in devices
        )

    return {
        "schema_version": "delepref_runtime_provenance_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "hostname": platform.node(),
        "cwd": str(Path.cwd()),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "scheduler_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "devices": devices,
        "nvidia_smi_returncode": None if result is None else result.returncode,
        "nvidia_smi_stdout": "" if result is None else result.stdout,
        "nvidia_smi_stderr": "" if result is None else result.stderr,
    }
