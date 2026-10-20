"""One-shot label-free v6 retrieval for 60 development and 32 confirmation WTs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

MANIFEST, SPLITS = "manifests/development_primary.jsonl", "manifests/splits.json"
PLAN = "docs/2026-10-15-recomb-v6-time-priority-plan.md"
FREEZE = "configs/rag_v6_retrieval_freeze.json"
BASE = "retrieval/rag_v6"
SOURCES = ["analysis/rag_v6_retrieval.py", "analysis/tests/test_rag_v6_retrieval.py",
           "slurm/rag_v6_retrieval_cpu.sbatch", PLAN]
MMSEQS = "/public/nfs-data/keyuan.zhou/evocompress/tools/mmseqs/bin/mmseqs"
TARGET = "/public/home/zhangyangroup/chengshiz/keyuan.zhou/delepath/results/tomography/evolution_geometry_uniref50_homolog_audit_v1/mmseqs_tmp/12081993509977918528/target"
SOURCE_PROTOCOL = "/public/nfs-data/keyuan.zhou/evocompress/data/cath_v7_mmseqs_msa_v1/protocol.json"
SOURCE_PROTOCOL_SHA = "cde3061b3adb6a60025332959cb5c214323299a230535d2051dc519ab984f9ee"
DB_SUFFIXES = ("", ".index", "_h", "_h.index", ".dbtype", "_h.dbtype", ".source")
FORMAT = "query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,evalue,bits,qcov,tcov,qaln,taln"
CONTRACT = {
    "scope": "label_free_retrieval_development_and_confirmation_inputs",
    "train_wt": 50, "validation_wt": 10, "confirmation_wt": 32,
    "new_PLM_forwards": 0, "new_structure_predictions": 0, "GPU_jobs": 0,
    "paper_changes": False, "automatic_retry": False,
    "confirmation_labels_read": False, "confirmation_used_for_fit": False,
    "resources": {"partition": "cpu_cu", "cpus": 32, "memory": "192G", "time": "01:00:00"},
    "retrieval": {"evalue": .001, "min_seq_id": .20, "query_coverage": .60,
                  "sensitivity": 7.5, "max_seqs": 512, "split_memory_limit": "160G",
                  "relaxed_search_coverage_reason": "avoid censoring internal natural gaps; known-coordinate coverage checked after projection"},
    "query_order": "development_sorted_ID_then_confirmation_sorted_ID",
    "raw_hits_only": True,
    "profile_and_ranking_contract": "separate_v6_freeze_before_validation_ranking"
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
        require(split in ("train", "validation", "confirmation"), "query outside frozen cohorts")
        value = {"protein_id": protein, "wt_sequence": sequence, "wt_sequence_sha256": digest, "split": split}
        require(protein not in unique or unique[protein] == value, "inconsistent WT rows")
        unique[protein] = value
    result = [dict(query_id=f"recombq_{i:04d}", **unique[p]) for i, p in enumerate(sorted(unique))]
    require(len({q["wt_sequence_sha256"] for q in result}) == len(result), "duplicate WT sequence across IDs")
    return result


def source_queries(root):
    pins = {
        MANIFEST: "73206ff4cd2534e4045bad623998ab907396d21b91b37d920a4479c35a8039ac",
        SPLITS: "0f814f7011fcd5c6277ce31204ec6aac099c30f15e8e0645c2a980a59c3c23fa",
        "manifests/confirmation_wt_frozen.jsonl": "7b7bfc84859aa4fb9a43390810b3d40e40d8bcdd640887f5442587240e91baa4",
        "reports/confirmation_wt_audit.json": "4b80196f3acda04b57acaf5fba8498e488bc9de82e9202bb7ef15ba3409aefcc",
    }
    verify_hashes(root, pins)
    rows = [json.loads(line) for line in (root / MANIFEST).read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(rows) == 2520, "expected 2520 frozen development rows")
    assignments = json.loads((root / SPLITS).read_text(encoding="utf-8"))["assignment"]
    dev = queries_from_rows(rows, assignments)
    require(len(dev) == 60 and sum(q["split"] == "train" for q in dev) == 50, "expected 50 train/10 validation WT")
    conf = [json.loads(line) for line in (root / "manifests/confirmation_wt_frozen.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    conf = queries_from_rows(conf, {x["protein_id"]: "confirmation" for x in conf})
    require(len(conf) == 32, "expected 32 frozen confirmation WT")
    all_queries = dev + conf
    require(len({q["protein_id"] for q in all_queries}) == 92, "overlapping cohort IDs")
    require(len({q["wt_sequence_sha256"] for q in all_queries}) == 92, "overlapping exact sequences")
    for i, q in enumerate(all_queries): q["query_id"] = f"recombq_{i:04d}"
    return all_queries, pins


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
             "-e", "0.001", "--min-seq-id", "0.20", "-c", "0.60", "--cov-mode", "2", "--threads", "32",
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
    config = {"status": "RAG_V6_RETRIEVAL_FROZEN", "contract": CONTRACT, "input_hashes": inputs,
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
    env["RAG_V6_RETRIEVAL_SHA256"] = freeze_sha
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
