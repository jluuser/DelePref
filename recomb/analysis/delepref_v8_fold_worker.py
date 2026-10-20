"""Slurm-only frozen final-output folding with append-only call accounting."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path("/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013")
sys.path.insert(0, str(ROOT / "code"))
from budget_ledger import BudgetLedger


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", choices=["esmfold", "boltz2"], required=True)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("all GPU inference must run through Slurm")
    freeze_path = ROOT / "configs/delepref_v8_confirmation_freeze.json"
    if sha(freeze_path) != args.freeze_sha256:
        raise ValueError("confirmation freeze identity mismatch")
    freeze = json.loads(freeze_path.read_text())
    if not freeze["fold_allowed"] or not freeze["all_selected_outputs_frozen"]:
        raise ValueError("complete confirmation freeze required")
    for record in freeze["files"]:
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("frozen experiment input/source changed: " + record["path"])
    preflight = json.loads((ROOT / freeze["predictor_reports"][args.predictor]).read_text())
    for record in preflight["files"] + preflight["implementation"] + preflight.get("standard_amino_acid_assets", []):
        if sha(record["path"]) != record["sha256"]:
            raise ValueError("frozen predictor changed: " + record["path"])
    profiles = json.loads((ROOT / "reports/delepref_v8_structure_preparation/predictor_profiles.json").read_text())
    profile = profiles[args.predictor]
    requests = json.loads((ROOT / "reports/delepref_v8_structure_preparation/physical_requests.json").read_text())
    requests = [r for r in requests if r["predictor"] == args.predictor]
    if not 0 <= args.shard_index < args.shards:
        raise ValueError("invalid dispatch shard")
    requests = [r for i, r in enumerate(requests) if i % args.shards == args.shard_index]
    ledger = BudgetLedger(ROOT / "reports/delepref_v8_evaluation_preparation/statistical_protocol.json",
                          ROOT / "reports/delepref_v8_structures/budget_ledger.jsonl")
    out = ROOT / "reports/delepref_v8_structures"
    out.mkdir(parents=True, exist_ok=True)
    import torch
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("frozen predictor device profile requires allocated A100")
    torch.backends.cuda.matmul.allow_tf32 = profile.get("tf32_allowed", False)
    if args.predictor == "esmfold":
        from transformers import AutoTokenizer, EsmForProteinFolding
        model_dir = str(Path(next(r["path"] for r in preflight["files"] if r["path"].endswith("pytorch_model.bin"))).parent)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = EsmForProteinFolding.from_pretrained(model_dir, local_files_only=True).eval()
        model.esm = model.esm.half()
        model.trunk.set_chunk_size(profile["chunk_size"])
        model = model.cuda()
    else:
        from boltz.main import predict
        from adapters.structure_adapter import read_coordinates
        cache = Path(preflight["files"][0]["path"]).parent
        assert (cache / "boltz2_aff.ckpt").is_file() and (cache / "mols").is_dir()
    print("FROZEN_PREDICTOR_LOADED " + args.predictor, flush=True)
    consecutive_failures = 0
    for index, request in enumerate(requests):
        task = request["task_id"]
        assert hashlib.sha256(request["sequence"].encode()).hexdigest() == request["sequence_sha256"]
        reservation = ledger.reserve(task_id=task, sequence_sha=request["sequence_sha256"],
            wt_variant=request["wt_variant"], predictor_weights_sha=request["weights_sha256"],
            profile_sha=request["profile_sha256"], seed=request["seed"], sample=request["sample"],
            category=request["category"], job_id=os.environ["SLURM_JOB_ID"])
        if not reservation["new_reservation"]:
            if reservation["state"] not in ("succeeded", "cache_hit", "failed"):
                raise RuntimeError("unfinished dispatch requires audit; do not duplicate " + task)
            continue
        sequence = request["sequence"]
        path = out / "artifacts" / (task + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        random.seed(request["seed"]); np.random.seed(request["seed"]); torch.manual_seed(request["seed"])
        torch.cuda.manual_seed_all(request["seed"])
        torch.cuda.reset_peak_memory_stats()
        ledger.start(task)
        started = time.monotonic()
        print("STRUCTURE_FORWARD_START " + task + " " + str(index + 1) + "/" + str(len(requests)), flush=True)
        try:
            if args.predictor == "esmfold":
                inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False).to("cuda")
                with torch.inference_mode():
                    output = model(**inputs, num_recycles=profile["num_recycles"])
                torch.cuda.synchronize()
                coords = output.positions[-1, 0, :, 1].detach().float().cpu().numpy()
                valid = output.atom14_atom_exists[0, :, 1].detach().bool().cpu().numpy()
                coords[~valid] = np.nan
                confidence = float(output.ptm.detach().float().cpu().reshape(-1)[0])
                del output, inputs
            else:
                directory = out / "boltz_inputs" / task
                directory.mkdir(parents=True, exist_ok=True)
                yaml = directory / "input.yaml"
                write(yaml, {"version": 1, "sequences": [{"protein": {"id": "A", "sequence": sequence, "msa": "empty"}}]})
                predict.callback(data=yaml, out_dir=directory / "output", cache=cache,
                    checkpoint=Path(preflight["files"][0]["path"]), accelerator="gpu", devices=1,
                    model="boltz2", recycling_steps=profile["recycling_steps"], sampling_steps=profile["sampling_steps"],
                    diffusion_samples=1, max_parallel_samples=1, num_workers=0, seed=request["seed"],
                    step_scale=profile["step_scale"], output_format="mmcif", preprocessing_threads=1,
                    use_msa_server=False, no_trifast=True)
                files = list((directory / "output").rglob("*_model_0.cif"))
                if len(files) != 1:
                    raise RuntimeError("expected exactly one Boltz structure")
                coords, mapping = read_coordinates(files[0], sequence, chain_id="A")
                confidence = None
                torch.cuda.synchronize()
            if coords.shape != (len(sequence), 3) or not np.isfinite(coords).all(1).any():
                raise ValueError("invalid predicted CA coordinate output")
            np.savez_compressed(path, coordinates=coords, sequence=np.array(sequence), sequence_sha256=np.array(request["sequence_sha256"]))
            elapsed = time.monotonic() - started
            ledger.finish(task, succeeded=True, seconds=elapsed, max_memory=float(torch.cuda.max_memory_allocated()), output_path=path)
            write(out / "records" / (task + ".json"), dict(request, status="SUCCEEDED", output_path=str(path), output_sha256=sha(path),
                  seconds=elapsed, max_GPU_memory_bytes=torch.cuda.max_memory_allocated(), confidence=confidence,
                  confirmation_freeze_sha256=args.freeze_sha256, job_id=os.environ["SLURM_JOB_ID"]))
            consecutive_failures = 0
            print("STRUCTURE_FORWARD_COMPLETE " + task + " seconds=" + str(round(elapsed, 3)), flush=True)
        except Exception as error:
            ledger.finish(task, succeeded=False, seconds=time.monotonic() - started, max_memory=float(torch.cuda.max_memory_allocated()))
            write(out / "records" / (task + ".json"), dict(request, status="FAILED", error=repr(error),
                  confirmation_freeze_sha256=args.freeze_sha256, job_id=os.environ["SLURM_JOB_ID"]))
            consecutive_failures += 1
            print("STRUCTURE_FORWARD_FAILED " + task + " " + repr(error), flush=True)
            if consecutive_failures >= 2:
                raise RuntimeError("two consecutive failures; stop for technical audit") from error
        finally:
            gc.collect(); torch.cuda.empty_cache()
    write(out / (args.predictor + "_shard_" + str(args.shard_index) + "_completion.json"), {"status": "FROZEN_DISPATCH_FINISHED", "predictor": args.predictor,
        "physical_requests": len(requests), "confirmation_freeze_sha256": args.freeze_sha256,
        "shard_index": args.shard_index, "shards": args.shards,
        "budget": {k: v for k, v in ledger.snapshot().items() if k != "attempts"}})


if __name__ == "__main__":
    main()
