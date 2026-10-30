"""Prepare or execute provenance-bound, length-grouped frozen ESMC caches."""
import argparse
import importlib.util
import importlib.metadata
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np
from common import AUTHORIZED_ROOT, confined_path, identity, read_jsonl, sha_file, sha_text, write_json, write_jsonl
from adapters.esmc_runtime import tokenizer_profile, verify_loader
from adapters.esmc_loader import load_esmc_official_safetensors_model


def candidates(rows):
    sequences = {}
    for row in rows:
        for field in ("wt_sequence", "sequence"):
            seq = row[field]
            key = sha_text(seq)
            item = sequences.setdefault(key, {"sequence": seq, "sequence_sha256": key, "length": len(seq), "uses": []})
            item["uses"].append({"candidate_id": row["candidate_id"], "field": field, "primary_eligible": row.get("primary_eligible", False)})
    return sorted(sequences.values(), key=lambda r: (r["length"], r["sequence_sha256"]))


def verified_cache(path, meta, expected):
    if not path.exists() or not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
        array = np.load(path, mmap_mode="r")
        return all(data.get(k) == v for k, v in expected.items()) and data["output_sha256"] == sha_file(path) and array.shape == (expected["length"], 1152) and array.dtype == np.float32 and np.isfinite(array).all()
    except (ValueError, KeyError, OSError):
        return False


def pilot_rows(rows, batch_size):
    buckets = {}
    for row in rows:
        buckets.setdefault(row["length"], []).append(row)
    lengths = sorted(k for k, v in buckets.items() if len(v) >= batch_size)
    if len(lengths) < 3:
        raise ValueError("three length strata with full pilot microbatches are required")
    indices = [0, len(lengths) // 2, len(lengths) - 1]
    return [(group % 2, row) for group, index in enumerate(indices)
            for row in buckets[lengths[index]][:batch_size]]


def main():
    total_begin = time.perf_counter()
    p = argparse.ArgumentParser()
    for key in ("protocol", "manifest", "provenance", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--shard", type=int, choices=(0, 1), default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--index-prefix", help="Separate immutable index namespace for later candidate cohorts")
    args = p.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    if Path(protocol["root"]).resolve() != AUTHORIZED_ROOT.resolve() or args.batch_size < 1:
        raise ValueError("invalid root/batch profile")
    root = confined_path(args.output)
    if root.resolve() != (AUTHORIZED_ROOT / "caches/embeddings").resolve():
        raise ValueError("cache output must be caches/embeddings")
    if args.execute and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("execution requires a Slurm allocation")
    root.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(args.provenance).read_text())["files"]["preference_core"]
    runtime_profile = tokenizer_profile()
    if args.execute:
        preflight = json.loads((AUTHORIZED_ROOT/'reports/plm_cpu_preflight.json').read_text())
        if preflight.get('status') != 'PLM_CPU_PREFLIGHT_PASSED' or preflight['runtime_profile'] != runtime_profile:
            raise ValueError('matching successful CPU runtime preflight is required')
        if preflight['protocol_sha256'] != sha_file(args.protocol):
            raise ValueError('CPU preflight protocol differs')
    weights = Path(protocol["weights"]["esmc"])
    weight_files = sorted(weights.glob("*.safetensors"))
    if not weight_files:
        raise ValueError("no frozen safetensors")
    weight_hashes = {f.name:sha_file(f) for f in weight_files}
    audited = json.loads((AUTHORIZED_ROOT/'reports/asset_audit.json').read_text())['weights']['esmc']['files']
    expected_weights = {Path(f['path']).name:f['sha256'] for f in audited if f['path'].endswith('.safetensors')}
    if weight_hashes != expected_weights:
        raise ValueError('ESMC weights differ from the initial asset audit')
    profile = {"weight_sha256": identity(weight_hashes),
               "snapshot_config_hashes": {str(f.relative_to(weights)): sha_file(f) for f in sorted(weights.rglob("*")) if f.is_file() and f.suffix in (".json", ".txt", ".model")},
               "batch_source_sha256": "388cd58117280ec183ff2a9307f4192cbb145e9198ca135d0f837758574b7d3e",
               "output_layer": "ESMC.embeddings", "special_token_policy": "assert_L_plus_2_remove_1_minus_1",
               "compute_precision": "bfloat16", "cache_precision": "float32",
               "batch_profile": {"equal_lengths": True, "batch_size": args.batch_size},
               **runtime_profile, "implementation_sha256": sha_file(__file__)}
    rows = candidates(read_jsonl(args.manifest))
    if args.pilot:
        assigned = pilot_rows(rows, args.batch_size)
        rows = [row for _, row in assigned]
        selected = [row for shard, row in assigned if shard == args.shard]
    else:
        selected = [r for n, r in enumerate(rows) if n % 2 == args.shard]
    if args.index_prefix and not re.fullmatch(r"[A-Za-z0-9_-]+", args.index_prefix):
        raise ValueError("invalid cache index namespace")
    name = (args.index_prefix or ("pilot" if args.pilot else "full")) + f"_shard{args.shard}"
    write_json(root / (name + "_plan.json"), {"profile": profile, "manifest_sha256": sha_file(args.manifest),
               "protocol_sha256": sha_file(args.protocol), "sequences": selected, "execute": args.execute,
               "all_sequence_count": len(rows), "shards": 2, "max_concurrency": 2})
    if not args.execute:
        return
    print(json.dumps({"stage": "CACHE_START", "shard": args.shard, "sequences": len(selected), "pilot": args.pilot}), flush=True)
    import torch
    if not torch.cuda.is_available() or sha_file(source["path"]) != source["sha256"] or sha_file(protocol["sources"]["encoder"]) != profile["batch_source_sha256"]:
        raise RuntimeError("CUDA/source identity prerequisite failed")
    gpu = torch.cuda.get_device_name()
    if "A40" not in gpu:
        raise RuntimeError("cache profile requires NVIDIA A40")
    load_begin = time.perf_counter()
    try:
        wrapper = load_esmc_official_safetensors_model(str(weights), dtype="bfloat16", device="cuda", require_cuda=True)
    except Exception as error:
        write_json(root/(name+'_report.json'),{'status':'INITIALIZATION_FAILED','timings':[],
                   'failures':[{'stage':'model_initialization','error':repr(error)}],
                   'profile':profile,'slurm_job_id':os.environ['SLURM_JOB_ID']})
        raise
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_begin
    print(json.dumps({"stage": "ESMC_LOADED", "seconds": model_load_seconds, "gpu": gpu}), flush=True)
    profile.update({"hardware": gpu, "torch_version": str(torch.__version__),
                    "esm_version": importlib.metadata.version("esm"),
                    **tokenizer_profile(wrapper.tokenizer)})
    write_json(root / (name + "_plan.json"), {"profile": profile, "manifest_sha256": sha_file(args.manifest),
               "protocol_sha256": sha_file(args.protocol), "sequences": selected, "execute": True,
               "all_sequence_count": len(rows), "shards": 2, "max_concurrency": 2})
    wrapper.esmc_model.eval()
    for param in wrapper.esmc_model.parameters():
        param.requires_grad_(False)
    index, timings, failures = [], [], []
    consumed = set()
    for start in range(0, len(selected)):
        # A deterministic equal-length batch is consumed once, without padding across lengths.
        if start in consumed:
            continue
        row = selected[start]
        batch = [row]
        for n, other in enumerate(selected[start+1:], start+1):
            if other["length"] != row["length"] or len(batch) == args.batch_size:
                break
            batch.append(other); consumed.add(n)
        pending = []
        for item in batch:
            expected = {**profile, "sequence_sha256": item["sequence_sha256"], "length": item["length"], "protocol_sha256": sha_file(args.protocol)}
            key = identity(expected); path = confined_path(root / (key + ".npy")); meta = path.with_suffix(".json")
            if verified_cache(path, meta, expected):
                index.append({"key": key, "path": str(path), **expected, "reused": True})
            else:
                pending.append((item, expected, key, path, meta))
        if not pending:
            continue
        try:
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begin = time.perf_counter()
            with torch.inference_mode():
                encoded = wrapper.tokenizer([x[0]["sequence"] for x in pending], padding=True, return_tensors="pt")
                output = wrapper.esmc_model(sequence_tokens=encoded['input_ids'].cuda()).embeddings
                if output.shape != (len(pending), pending[0][0]["length"] + 2, 1152):
                    raise ValueError("token/embedding shape mismatch; truncation prohibited")
                arrays = output[:, 1:-1].float().cpu().numpy()
                if arrays.dtype != np.float32 or arrays.shape != (len(pending), pending[0][0]["length"], 1152) or not np.isfinite(arrays).all():
                    raise ValueError("invalid shape, dtype, or nonfinite cache output")
            torch.cuda.synchronize(); elapsed = time.perf_counter()-begin
            timings.append({"seconds": elapsed, "batch": len(pending), "length": pending[0][0]["length"], "peak_bytes": torch.cuda.max_memory_allocated()})
            for values, (_, expected, key, path, meta) in zip(arrays, pending):
                tmp = confined_path(path.with_name(path.name + f".{os.getpid()}.tmp"))
                with tmp.open("wb") as handle:
                    np.save(handle, values)
                os.replace(tmp, path)
                write_json(meta, {**expected, "output_sha256": sha_file(path), "gpu": torch.cuda.get_device_name(), "batch_seconds": elapsed})
                index.append({"key": key, "path": str(path), **expected, "reused": False})
        except Exception as exc:
            failures.append({"sequences": [x[0]["sequence_sha256"] for x in pending], "error": repr(exc)})
        if len(timings) % 25 == 0 or failures or args.pilot:
            write_jsonl(root / (name + "_index.jsonl"), index)
            print(json.dumps({"stage": "CACHE_PROGRESS", "shard": args.shard, "cached": len(index), "total": len(selected), "failures": len(failures)}), flush=True)
    write_jsonl(root / (name + "_index.jsonl"), index)
    seconds = [t["seconds"] for t in timings]
    write_json(root / (name + "_report.json"), {"timings": timings, "failures": failures,
               "model_load_seconds": model_load_seconds, "elapsed_total_seconds": time.perf_counter()-total_begin,
               "profile": profile, "slurm_job_id": os.environ["SLURM_JOB_ID"],
               "p50_seconds": float(np.percentile(seconds, 50)) if seconds else None,
               "p90_seconds": float(np.percentile(seconds, 90)) if seconds else None})
    if failures:
        raise RuntimeError("cache failures recorded")
    print(json.dumps({"stage": "CACHE_COMPLETE", "shard": args.shard, "sequences": len(index), "seconds": time.perf_counter()-total_begin}), flush=True)


if __name__ == "__main__":
    main()
