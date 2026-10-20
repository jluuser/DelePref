"""One-shot, label-free WT retrieval for the frozen homolog-risk development pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

D_FREEZE = "configs/d_ranking_trial_freeze.json"
D_SHA = "2075da8f99054354033f83ec58bde39b105d1eed8c3dd7efcdf8cb6ec7511f5d"
MANIFEST, SPLITS = "manifests/development_primary.jsonl", "manifests/splits.json"
PLAN = "docs/superpowers/plans/2026-10-15-recomb-homolog-risk-pilot.md"
FREEZE = "configs/homolog_risk_pilot_freeze.json"
BASE = "retrieval/homolog_risk_pilot"
SOURCES = ["analysis/homolog_pilot_retrieval.py", "analysis/tests/test_homolog_pilot_retrieval.py",
           "slurm/homolog_pilot_retrieval_cpu.sbatch", PLAN]
MMSEQS = "/public/nfs-data/keyuan.zhou/evocompress/tools/mmseqs/bin/mmseqs"
TARGET = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/delepath/results/tomography/evolution_geometry_uniref50_homolog_audit_v1/mmseqs_tmp/12081993509977918528/target"
SOURCE_PROTOCOL = "/public/nfs-data/keyuan.zhou/evocompress/data/cath_v7_mmseqs_msa_v1/protocol.json"
SOURCE_PROTOCOL_SHA = "cde3061b3adb6a60025332959cb5c214323299a230535d2051dc519ab984f9ee"
DB_SUFFIXES = ("", ".index", "_h", "_h.index", ".dbtype", "_h.dbtype", ".source")
FORMAT = "query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qcov,tcov,qaln,taln"
CONTRACT = {
    "scope": "development_only", "train_wt": 50, "validation_wt": 10, "candidate_rows": 2520,
    "D0_seeds": [20261013, 20261014, 20261015], "D0_step": 2000,
    "new_PLM_forwards": 0, "new_structure_predictions": 0, "GPU_jobs": 0,
    "paper_changes": False, "confirmation_access": False, "automatic_retry": False,
    "resources": {"partition": "cpu_cu", "cpus": 32, "memory": "192G", "time": "01:00:00"},
    "retrieval": {"all_60_requeried": True, "existing_exact_MSA_matches": 1,
                  "evalue": .001, "min_seq_id": .20, "query_coverage": .80,
                  "sensitivity": 7.5, "max_seqs": 512, "split_memory_limit": "160G"},
    "profile": {"identity_min": .20, "identity_max_exclusive": .95, "qcov_min": .80,
                "evalue_max": .001, "coverage_min": .80, "gap_fraction_max": .20,
                "cluster_identity": .90, "cluster_common_standard_AA_fraction_min": .50,
                "representatives_max": 64, "depth_min": 8, "observed_min": 8, "nongap_min": 4,
                "deleted_support_min": .80, "junction_support_min": .80,
                "unknown": ".", "internal_gap": "-", "exclude_self_and_duplicate_projection": True,
                "target_dedup": "highest_bits", "order": "-bits,evalue,target,row",
                "qfreq": "WT_matches/non_gap_standard_AA", "gapfreq": "internal_gap/(internal_gap+standard_AA)"},
    "homolog_features": ["deleted_qfreq", "deleted_gapfreq", "junction_qfreq"],
    "static_features": ["C", "budget", "deletion_runs/L", "has_junction"],
    "junction": "both retained endpoints of internal deletion; absent value=0",
    "fit": {"ridge_lambda": 1., "penalize_intercept": False, "objective": "weighted_mean_squared_error+lambda*sum(beta^2)",
            "weights": "equal_WT_budget_source_then_equal_available_candidates", "standardization": "training_available_only; zero_variance_scale=1",
            "D0_training_predictions": "in_sample_frozen_CPU_head_only", "hybrid_training_rows": "same_available_rows",
            "static_direct_training_rows": "all_train", "eligible_train_WT_min": 10, "insufficient_eligible_train_WT": "inconclusive_no_fit",
            "target": "G", "prediction_clip": [0., 1.], "selection": "T=C*G"},
    "selectors": ["D0", "homolog_static_direct", "D0_static_residual", "D0_homolog_static_residual", "20_position_shuffles"],
    "hybrid_missing_fallback": "exact_D0", "direct_missing_fallback": "static_direct",
    "shuffle": {"seeds": list(range(20261015, 20261035)), "joint_qfreq_gapfreq": True,
                "within_WT_supported_positions": True, "preserve_support_depth_availability": True, "refit": True},
    "dropout": {"fraction": .50, "seed": 20261015, "refit": False, "per_WT_fixed_subset": True},
    "primary": {"comparison": "full_hybrid-D0", "endpoint": "mixed_selected_T", "average": "budgets_then_3_seeds_within_WT",
                "bootstrap_unit": "10_WT", "bootstrap_replicates": 10000, "bootstrap_seed": 20261013},
    "gates": {"mean_min": .005, "CI95_lower_strict_min": 0., "positive_seeds_min": 2,
              "each_source_mean_min": 0., "full_minus_static_mean_strict_min": 0.,
              "beat_all_20_shuffle_deltas_strictly": True, "available_validation_WT_min": 5,
              "available_WT": "each_source_x_budget_has_at_least_5_of_10_WT_with_at_least_one_eligible_candidate", "matched_coverage_delta_C_max": .01,
              "matched_coverage_evidence_required": True, "matched_ranking_improvement_strict_min": 0.},
    "matched_ranking": "same_WT_budget_source; exclude_truth_ties; predicted_tie=.5; equal_pool_then_WT",
    "report": ["all_sources_mixed_and_within_source", "budgets", "seeds", "T", "regret", "oracle", "changed_selection_positive_negative",
               "G_T_MAE_MSE", "retrieval_coverage_latency", "availability", "all_favorable_unfavorable_results"],
    "failure_policy": "insufficient_depth_coverage=inconclusive; other_gate_failure=current_no_support; no_extra_search_or_tuning"
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value, atomic=False):
    path = Path(path)
    require(not path.exists(), f"refusing overwrite: {path}")
    temporary = path.with_name(path.name + ".partial") if atomic else path
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if atomic:
        os.replace(temporary, path)


def verify_hashes(root, hashes):
    for path, expected in hashes.items():
        require(sha(root / path) == expected, f"SHA256 mismatch: {path}")


def owned_base(root):
    base = root / BASE
    require(not (root / "retrieval").is_symlink() and not base.is_symlink(), "output namespace must not be a symlink")
    return base


def queries_from_rows(rows, assignments):
    unique = {}
    for row in rows:
        protein, sequence, digest = row["protein_id"], row["wt_sequence"], row["wt_sequence_sha256"]
        require(isinstance(protein, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", protein), "unsafe protein ID")
        require(isinstance(sequence, str) and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", sequence), "invalid exact WT sequence")
        require(hashlib.sha256(sequence.encode("ascii")).hexdigest() == digest, "WT sequence SHA256 mismatch")
        split = assignments.get(protein)
        require(split in ("train", "validation"), "query outside development split")
        value = {"protein_id": protein, "wt_sequence": sequence, "wt_sequence_sha256": digest, "split": split}
        require(protein not in unique or unique[protein] == value, "inconsistent WT rows")
        unique[protein] = value
    result = [dict(query_id=f"recombq_{i:04d}", **unique[p]) for i, p in enumerate(sorted(unique))]
    require(len({q["wt_sequence_sha256"] for q in result}) == len(result), "duplicate WT sequence across IDs")
    return result


def source_queries(root):
    verify_hashes(root, {D_FREEZE: D_SHA})
    frozen = json.loads((root / D_FREEZE).read_text())
    inputs = {p: frozen["input_hashes"][p] for p in (MANIFEST, SPLITS)}
    verify_hashes(root, inputs)
    with (root / MANIFEST).open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    require(len(rows) == 2520, "expected 2520 candidate rows")
    queries = queries_from_rows(rows, json.loads((root / SPLITS).read_text())["assignment"])
    require(len(queries) == 60 and sum(q["split"] == "train" for q in queries) == 50, "expected 50 train/10 validation WT")
    return queries, {D_FREEZE: D_SHA, **inputs}


def database_metadata(target):
    result = {}
    for suffix in DB_SUFFIXES:
        path = Path(str(target) + suffix)
        stat = path.stat()
        require(path.is_file() and stat.st_size > 0, f"missing/empty database file: {path}")
        result[str(path)] = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if stat.st_size <= 1024 * 1024:
            result[str(path)]["sha256"] = sha(path)
    return result


def verify_database(config):
    require(database_metadata(config["target"]) == config["database"], "database metadata changed")


def tool_identity(executable):
    return {"path": executable, "sha256": sha(executable),
            "version": subprocess.check_output([executable, "version"], text=True).strip()}


def commands(executable, target):
    return [[executable, "createdb", "queries.fasta", "query_db"],
            [executable, "search", "query_db", target, "result_db", "tmp", "-a", "1", "--alignment-mode", "3",
             "-e", "0.001", "--min-seq-id", "0.20", "-c", "0.80", "--cov-mode", "2", "--threads", "32",
             "-s", "7.5", "--max-seqs", "512", "--split-mode", "0", "--split-memory-limit", "160G", "--remove-tmp-files", "1"],
            [executable, "convertalis", "query_db", target, "result_db", "hits.tsv", "--threads", "32", "--format-output", FORMAT]]


def prepare(root):
    require(not (root / FREEZE).exists(), "freeze already exists")
    queries, inputs = source_queries(root)
    verify_hashes(root, {SOURCE_PROTOCOL: SOURCE_PROTOCOL_SHA})
    provenance = json.loads(Path(SOURCE_PROTOCOL).read_text())["inputs"]["uniref50_fasta"]
    metadata, identity = database_metadata(TARGET), tool_identity(MMSEQS)
    require(metadata[TARGET]["size"] == 17402685881, "unexpected UniRef50 sequence database size")
    hashes = {p: sha(root / p) for p in SOURCES}
    prepared = owned_base(root) / "prepared"
    prepared.mkdir(parents=True, exist_ok=False)
    with (prepared / "queries.fasta").open("x", encoding="ascii", newline="\n") as handle:
        for query in queries:
            handle.write(f'>{query["query_id"]}\n{query["wt_sequence"]}\n')
    write_json(prepared / "query_map.json", queries)
    query_hashes = {f"{BASE}/prepared/{p}": sha(prepared / p) for p in ("queries.fasta", "query_map.json")}
    config = {"status": "HOMOLOG_RISK_PILOT_FROZEN", "contract": CONTRACT, "input_hashes": inputs,
              "source_hashes": hashes, "query_hashes": query_hashes, "query_count": len(queries),
              "tool": identity, "target": TARGET, "database": metadata, "commands": commands(MMSEQS, TARGET),
              "database_provenance": {"name": "UniRef50", "inherited_source_record": provenance, "source_protocol": SOURCE_PROTOCOL,
                                      "source_protocol_sha256": SOURCE_PROTOCOL_SHA, "original_compressed_FASTA_rehashed_this_run": False}}
    write_json(root / FREEZE, config, atomic=True)
    return {"freeze_sha256": sha(root / FREEZE), "queries": len(queries)}


def load_verified(root, freeze_sha):
    require(bool(re.fullmatch("[0-9a-f]{64}", freeze_sha or "")), "explicit freeze SHA256 required")
    verify_hashes(root, {FREEZE: freeze_sha})
    config = json.loads((root / FREEZE).read_text())
    require(config["contract"] == CONTRACT, "scientific contract changed")
    for key in ("source_hashes", "input_hashes", "query_hashes"):
        verify_hashes(root, config[key])
    provenance = config["database_provenance"]
    verify_hashes(root, {provenance["source_protocol"]: provenance["source_protocol_sha256"]})
    require(config["tool"] == tool_identity(config["tool"]["path"]), "MMseqs binary/version changed")
    require(config["commands"] == commands(config["tool"]["path"], config["target"]), "commands changed")
    verify_database(config)
    return config


def submit(root, freeze_sha):
    load_verified(root, freeze_sha)
    base = owned_base(root)
    write_json(base / "submission_receipt.json", {"freeze_sha256": freeze_sha, "time": time.time(), "status": "attempt_claimed_no_retry"})
    env = {k: v for k, v in os.environ.items() if not k.startswith("SBATCH_")}
    env["HOMOLOG_PILOT_FREEZE_SHA256"] = freeze_sha
    result = subprocess.run(["/opt/gridview/slurm/bin/sbatch", "--parsable", str(root / SOURCES[2])], cwd=root, env=env, capture_output=True, text=True)
    write_json(base / "submission_result.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, atomic=True)
    require(result.returncode == 0 and re.fullmatch(r"[0-9]+(?:;[^\s]+)?\s*", result.stdout), "submission failed/unknown; do not retry")
    return {"job_id": result.stdout.strip(), "freeze_sha256": freeze_sha}


def execute(root, freeze_sha):
    require(os.environ.get("SLURM_JOB_ID", "").isdigit() and os.environ.get("SLURM_CPUS_PER_TASK") == "32", "requires Slurm 32 CPU allocation")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be empty")
    require(all(os.environ.get(k, "") in ("", "N/A", "NoDevFiles") for k in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"))
            and all(os.environ.get(k, "") in ("", "0", "N/A") for k in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE")), "GPU allocation forbidden")
    base = owned_base(root)
    receipt = json.loads((base / "submission_receipt.json").read_text())
    require(receipt["freeze_sha256"] == freeze_sha, "submission receipt mismatch")
    write_json(base / "execution_claim.json", {"job_id": os.environ["SLURM_JOB_ID"], "freeze_sha256": freeze_sha, "time": time.time()})
    start = time.time()
    try:
        config = load_verified(root, freeze_sha)
        search = base / "search"
        search.mkdir(exist_ok=False)
        with (search / "queries.fasta").open("xb") as handle:
            handle.write((base / "prepared/queries.fasta").read_bytes())
        db_hashes = {p: sha(p) for p in config["database"]}
        verify_database(config)
        write_json(search / "database_sha256_before.json", db_hashes, atomic=True)
        timings = []
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="32", MKL_NUM_THREADS="32", OPENBLAS_NUM_THREADS="32")
        for number, command in enumerate(config["commands"]):
            before = time.time()
            with (search / f"{number}_{command[1]}.stdout.log").open("xb") as out, (search / f"{number}_{command[1]}.stderr.log").open("xb") as err:
                subprocess.run(command, cwd=search, env=env, stdout=out, stderr=err, check=True)
            timings.append({"command": command, "seconds": time.time() - before})
        verify_database(config)
        hits = search / "hits.tsv"
        require(hits.is_file(), "convertalis did not create raw hits")
        manifest = {"status": "complete" if hits.stat().st_size else "complete_zero_hits", "hits_sha256": sha(hits), "hits_bytes": hits.stat().st_size,
                    "freeze_sha256": freeze_sha, "tool": config["tool"], "database_sha256": db_hashes, "database_metadata_after": database_metadata(config["target"]),
                    "query_hashes": config["query_hashes"], "source_hashes": config["source_hashes"], "protocol": config["contract"],
                    "job_id": os.environ["SLURM_JOB_ID"], "started": start, "finished": time.time(), "commands": timings}
        write_json(base / "completion.json", manifest, atomic=True)
        return manifest["status"]
    except BaseException as error:
        write_json(base / "failure.json", {"status": "failed_no_retry", "type": type(error).__name__, "error": str(error), "started": start, "finished": time.time()}, atomic=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--freeze-sha256")
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "submit", "execute"):
        modes.add_argument("--" + mode, action="store_true")
    args = parser.parse_args()
    action = prepare if args.prepare else submit if args.submit else execute
    print(json.dumps(action(args.root.resolve()) if args.prepare else action(args.root.resolve(), args.freeze_sha256), sort_keys=True))
