#!/usr/bin/env python3
"""Read-only source, recovery, environment, and scheduler audit for RECOMB v2.

Only ``--output`` is written. Imports run with CUDA hidden, offline mode, and
bytecode disabled; no model is instantiated and no checkpoint is deserialized
without ``weights_only=True``. Confirmation manifests are never opened.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tarfile
import zipfile
from collections import deque

VERIFIED = "VERIFIED"
PENDING = "PATH_EXISTS_PROFILE_PENDING"
MISSING = "MISSING"
SCHEMA = "recomb_asset_audit_v1"
AUTHORIZED_ROOT = Path("/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013")
SKIP_DIRS = {".git", "__pycache__", ".cache", ".pytest_cache", ".ruff_cache",
             "node_modules", "site-packages", "miniconda3", "mambaforge",
             "anaconda3", "mols", "model_cache", "checkpoints-encoder"}
FUNCTIONS = {
    "student_model": ["BoundaryAwarePreferenceScorer", "StrictBudgetProposalPolicy",
                      "strict_budget_dp_decode", "generate_strict_budget_plans",
                      "validate_strict_plan", "FrozenWTEncoder"],
    "preference_core": ["parse_deleted_intervals", "deletion_mask_from_intervals",
                        "intervals_from_deletion_mask", "load_esmc_official_safetensors_model",
                        "residue_embeddings_from_backend"],
    "encoder": ["BatchedESMCBackend"],
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path, *, hash_content=True):
    """Record exact bytes and link identity, without conflating directories/files."""
    path = Path(path)
    result = {"path": str(path), "status": MISSING}
    if path.is_symlink():
        result["symlink_target"] = os.readlink(path)
    try:
        stat = path.stat()
        result.update(resolved_path=str(path.resolve()), size=stat.st_size,
                      mtime_ns=stat.st_mtime_ns,
                      mtime_utc=dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
                      kind="directory" if path.is_dir() else "file", status=PENDING)
        if path.is_file() and hash_content:
            result["sha256"] = sha256_file(path)
            result["status"] = VERIFIED
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_readonly(command, timeout=120):
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        completed = subprocess.run([str(item) for item in command], capture_output=True,
                                   text=True, timeout=timeout, env=env)
        return {"command": [str(item) for item in command], "returncode": completed.returncode,
                "stdout": completed.stdout[-60000:], "stderr": completed.stderr[-12000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": [str(item) for item in command], "returncode": None,
                "error": f"{type(exc).__name__}: {exc}"}


def definitions(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = {node.name: node.lineno for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "STUDENT_VERSION", "EVALUATION_SCHEMA", "SUPPORTED_BUDGETS", "COORDINATE_SYSTEM"}:
                    constants[target.id] = value
    return names, constants


def bounded_walk(root, max_depth=5, max_entries=2000):
    """Deterministic breadth-first walk; never recurse through a symlink."""
    root = Path(root)
    queue = deque([(root, 0)])
    count, hits, errors, cutoffs, entry_limit = 0, [], [], [], False
    while queue and count < max_entries:
        directory, depth = queue.popleft()
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            errors.append({"path": str(directory), "error": str(exc)})
            continue
        for path in entries:
            count += 1
            if count > max_entries:
                entry_limit = True
                break
            hits.append(path)
            if path.is_symlink() or not path.is_dir():
                continue
            if path.name in SKIP_DIRS or path.name.startswith((".venv", ".conda")):
                continue
            if depth < max_depth:
                queue.append((path, depth + 1))
            else:
                cutoffs.append(str(path))
    return hits, {"root": str(root), "max_depth": max_depth, "max_entries": max_entries,
                  "entries_examined": min(count, max_entries), "entry_limit_reached": entry_limit or bool(queue),
                  "depth_cutoffs": cutoffs[:100], "depth_cutoff_count": len(cutoffs), "errors": errors[:100]}


def recovery_search(sources):
    user_home = Path(sources["delepath"]).parent
    nfs_home = Path(sources["general240"]).parents[2]
    roots = [("symlinks_and_backups", Path(sources["delepath"])),
             ("symlinks_and_backups", user_home / "delepath-evoaudit-recovery"),
             ("user_projects", user_home), ("user_projects", nfs_home)]
    all_paths, searches = {}, []
    for phase, root in roots:
        if not root.is_dir():
            searches.append({"phase": phase, "root": str(root), "status": MISSING})
            continue
        paths, info = bounded_walk(root)
        info["phase"] = phase
        searches.append(info)
        for path in paths:
            all_paths[str(path)] = path
    # Follow only relevant link directories whose resolved targets remain in the
    # two authorized user project trees, and record this extra bound separately.
    permitted = [user_home.resolve(), nfs_home.resolve()]
    targets = set()
    for path in list(all_paths.values()):
        if path.is_symlink() and path.is_dir() and any(tag in str(path).lower() for tag in ("delepref", "delepath", "backup", "archive")):
            target = path.resolve()
            if any(target == root or root in target.parents for root in permitted):
                targets.add(target)
    for target in sorted(targets)[:20]:
        paths, info = bounded_walk(target, max_depth=7, max_entries=10000)
        info["phase"] = "bounded_symlink_target"
        searches.append(info)
        for path in paths:
            all_paths[str(path)] = path
    checkpoints, libraries, links, archives, manuscripts, backup_manifests = [], [], [], [], [], []
    for path in all_paths.values():
        lower = str(path).lower()
        name = path.name.lower()
        if path.is_symlink() and any(tag in lower for tag in ("delepref", "delepath", "backup", "archive")):
            links.append(record(path, hash_content=False))
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors"}:
            if ("delepref" in lower and "esmc" not in lower and "esmfold" not in lower
                    and "boltz" not in lower):
                checkpoints.append(str(path))
        if ((path.is_dir() and name == "delepref" and "preference_training" in lower)
                or (path.is_file() and name == "preference_pair_shard_index.csv")):
            libraries.append(str(path))
        if path.is_file() and any(tag in name for tag in ("backup", "archive", "release")) and path.suffix in {".json", ".txt", ".md"}:
            backup_manifests.append(record(path))
        if path.is_file() and any(tag in name for tag in ("delepref", "delepath", "aaai")):
            if name.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                archives.append(path)
        if (path.is_file() and path.suffix.lower() in {".tex", ".pdf", ".docx", ".md", ".txt"}
                and any(tag in lower for tag in ("aaai-protein", "review", "rebuttal", "manuscript"))):
            manuscripts.append(record(path))
    archive_reports = []
    for archive in archives[:30]:
        item = record(archive)
        item["matches"] = []
        try:
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as handle:
                    members = (entry.filename for entry in handle.infolist())
                    names = list(members)[:50000]
            else:
                with tarfile.open(archive, "r|*") as handle:
                    names = []
                    for index, member in enumerate(handle):
                        if index == 50000:
                            break
                        names.append(member.name)
            item["members_examined"] = len(names)
            item["matches"] = [name for name in names if
                re.search(r"(delepref.*\.(pt|pth|ckpt)$|preference_training|review|rebuttal|\.tex$)", name.lower())][:500]
            item["note"] = "Archive index only; no extraction or archived asset verification."
        except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
            item["archive_error"] = str(exc)
        archive_reports.append(item)
    return {"search_order": ["symlinks_and_backups", "user_projects", "release_archive_indices"],
            "searches": searches, "checkpoint_candidates": sorted(set(checkpoints)),
            "preference_library_candidates": sorted(set(libraries)), "symlinks": links[:300],
            "backup_manifests": backup_manifests[:100], "release_archives": archive_reports,
            "manuscript_review_assets": manuscripts[:500],
            "symlink_targets_truncated": len(targets) > 20,
            "limitations": "Bounded project search; relevant symlinks are followed only inside authorized user trees. "
                           "No system-wide search and no confirmation manifest contents read."}


ENV_PROBE = r'''
import importlib, importlib.metadata, json, platform, sys
modules = json.loads(sys.argv[1]); result = {"python": sys.version, "executable": sys.executable,
    "platform": platform.platform(), "imports": {}, "packages": {}}
for name in modules:
    try:
        module = importlib.import_module(name)
        result["imports"][name] = {"ok": True, "file": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None)}
    except Exception as exc:
        result["imports"][name] = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
for name in ["torch", "numpy", "esm", "transformers", "safetensors", "biopython", "boltz", "lightning"]:
    try: result["packages"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: pass
if "torch" in sys.modules: result["torch_compiled_cuda"] = sys.modules["torch"].version.cuda
print(json.dumps(result))
'''


def environment_probe(python, modules):
    item = record(python)
    if item["status"] == MISSING:
        return item
    probe = run_readonly([python, "-B", "-c", ENV_PROBE, json.dumps(modules)], timeout=180)
    item["probe"] = probe
    try:
        payload = json.loads(probe.get("stdout", "").strip().splitlines()[-1])
        item["profile"] = payload
        item["status"] = VERIFIED if all(x["ok"] for x in payload["imports"].values()) else PENDING
    except (ValueError, IndexError):
        item["status"] = PENDING
    return item


CHECKPOINT_PROBE = r'''
import ast, hashlib, importlib.util, json, pathlib, sys, torch
path, config_path, student_path, runtime_path = map(pathlib.Path, sys.argv[1:5])
expected_eval_path, expected_eval_sha = sys.argv[5:7]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
payload = torch.load(str(path), map_location="cpu", weights_only=True)
if not isinstance(payload, dict): raise ValueError("Checkpoint is not a mapping")
state = payload.get("student_head_state_dict", {})
summary = {"keys": sorted(str(k) for k in payload), "state_keys": sorted(str(k) for k in state),
    "tensor_shapes": {str(k): list(v.shape) for k,v in state.items() if hasattr(v,"shape")},
    "metadata": {k:payload.get(k) for k in ["schema_version","student_version","step","selected_checkpoint_step",
        "config_sha256","source_checkpoint_path","source_checkpoint_sha256","qmax","max_length","budgets","coordinate_system"]},
    "checks": {}, "eval_fingerprints": {}}
expected_version = None
for node in ast.parse(student_path.read_text()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=="STUDENT_VERSION" for t in node.targets):
        expected_version=ast.literal_eval(node.value)
summary["checks"]["student_version"] = payload.get("student_version")==expected_version
summary["checks"]["nonempty_head_state"] = isinstance(state,dict) and bool(state)
summary["checks"]["encoder_weights_absent"] = not any(str(k).startswith("encoder.") for k in state)
summary["checks"]["scorer_and_policy_keys"] = any(str(k).startswith("scorer.") for k in state) and any(str(k).startswith("policy.") for k in state)
step=payload.get("selected_checkpoint_step",payload.get("step"))
summary["checks"]["expected_step_4000"] = step==4000
summary["checks"]["known_formal_evaluation_schema"] = payload.get("schema_version")=="delepref_evaluation_checkpoint_v1"
import yaml
config=yaml.safe_load(config_path.read_text())
canonical=hashlib.sha256(json.dumps(config,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
summary["expected_config_canonical_sha256"] = canonical
summary["checks"]["config"] = payload.get("config_sha256")==canonical or payload.get("config")==config
for scope in [payload,payload.get("metadata",{})]:
    if isinstance(scope,dict):
        for k,v in scope.items():
            if any(tag in str(k).lower() for tag in ["fingerprint","eval_manifest_sha","evaluation_data_sha","dev_manifest_sha"]):
                if isinstance(v,(str,int,float,bool,type(None))):summary["eval_fingerprints"][str(k)]=v
formal_keys={"evaluation_manifest_sha256","eval_manifest_sha256","evaluation_data_sha256","formal_evaluation_manifest_sha256"}
fingerprint_values=[]
for scope in [payload,payload.get("metadata",{})]:
    if isinstance(scope,dict):fingerprint_values.extend(v for k,v in scope.items() if k in formal_keys)
summary["checks"]["formal_eval_data_fingerprint_matches_actual_manifest"] = False
if expected_eval_path and expected_eval_sha:
    if "confirmation" in expected_eval_path.lower():raise ValueError("Confirmation manifest must not be opened")
    actual=digest(pathlib.Path(expected_eval_path))
    summary["formal_eval_manifest_actual_sha256"]=actual
    summary["checks"]["formal_eval_data_fingerprint_matches_actual_manifest"] = actual==expected_eval_sha and actual in fingerprint_values
else:
    summary["formal_eval_fingerprint_note"]="No expected formal evaluation manifest and digest supplied; remains pending."
summary["checks"]["original_evaluation_validator"] = False
if payload.get("schema_version")=="delepref_evaluation_checkpoint_v1":
    spec=importlib.util.spec_from_file_location("recomb_readonly_evaluation_runtime",runtime_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    sys.path.insert(0,str(runtime_path.parent));sys.path.insert(0,str(student_path.parents[1]))
    try:
        spec.loader.exec_module(module);module._validate_evaluation_payload(payload,config_path)
        summary["checks"]["original_evaluation_validator"] = True
    except Exception as exc:
        summary["checks"]["original_evaluation_validator"] = False
        summary["validator_error"] = type(exc).__name__+": "+str(exc)
print(json.dumps(summary,default=str))
'''


def audit_checkpoint(path, python, config, student, runtime, expected_eval=None):
    item = record(path)
    if item["status"] == MISSING:
        return item
    item["status"] = PENDING
    expected_eval = expected_eval or {}
    probe = run_readonly([python, "-B", "-c", CHECKPOINT_PROBE, path, config, student, runtime,
                         expected_eval.get("path", ""), expected_eval.get("sha256", "")], timeout=180)
    item["validation_command"] = probe
    try:
        validation = json.loads(probe.get("stdout", "").strip().splitlines()[-1])
        item["validation"] = validation
        if validation["checks"] and all(validation["checks"].values()):
            item["status"] = VERIFIED
    except (ValueError, IndexError):
        item["validation_error"] = "Safe CPU checkpoint inspection did not return JSON."
    return item


def weight_inventory(path, required_patterns):
    item = record(path, hash_content=False)
    if item["status"] == MISSING:
        return item
    files = []
    # Only actual model weights/config/tokenizers, not cached molecular libraries.
    for entry in sorted(Path(path).iterdir()):
        if entry.is_file() and (entry.suffix in {".json", ".txt", ".safetensors", ".bin", ".ckpt"}):
            files.append(record(entry))
    item["files"] = files
    item["required_patterns"] = {pattern: any(Path(x["path"]).match(pattern) and x["status"] == VERIFIED for x in files)
                                 for pattern in required_patterns}
    item["weights_present_and_hashed"] = all(item["required_patterns"].values())
    item["status"] = PENDING
    item["profile_note"] = "Weights and configuration inventoried; GPU forward/profile validation has not run."
    return item


def library_inventory(path):
    item = record(path, hash_content=False)
    if item["status"] == MISSING:
        return item
    root = Path(path)
    if root.is_file():
        root = root.parent
    paths, bounds = bounded_walk(root, max_depth=6, max_entries=40000)
    indexes = [p for p in paths if p.is_file() and p.name in {
        "preference_pair_shard_index.csv", "validation_report.json", "conditions.csv", "train_dev_manifest.csv"}]
    item["non_confirmation_indexes"] = [record(p) for p in indexes]
    item["bounds"] = bounds
    item["status"] = PENDING
    item["note"] = "Candidate library; full shard counts, config agreement, and split fingerprints remain required. Confirmation manifests unopened."
    return item


def write_json(path, payload):
    # Reject pre-existing symlinks in either the destination or its temporary
    # pathname before mkdir/open can traverse into historical/shared roots.
    for candidate in (path, path.with_suffix(path.suffix + ".tmp")):
        resolved = candidate.resolve()
        if AUTHORIZED_ROOT != resolved and AUTHORIZED_ROOT not in resolved.parents:
            raise ValueError(f"Write destination escapes authorized experiment root: {candidate}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit(protocol_path, output):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    expected_output = Path(protocol["root"]).resolve()
    if output.resolve() != AUTHORIZED_ROOT or expected_output != AUTHORIZED_ROOT:
        raise ValueError("--output and protocol.root must both equal the authorized new experiment root")
    sources, envs, weights = protocol["sources"], protocol["environments"], protocol["weights"]
    legacy = Path(sources["delepath"])
    source_paths = {
        "student_model": Path(sources["delepref"]) / "src/delepref/student_model.py",
        "preference_core": Path(sources["delepref"]) / "src/delepref/preference_core.py",
        "legacy_student_model": legacy / "src/delepref/student_model.py",
        "legacy_preference_core": legacy / "src/delepref/preference_core.py",
        "legacy_training_config": legacy / "configs/deletion_preference_student.yaml",
        "legacy_evaluation_runtime": legacy / "scripts/evaluation_runtime.py",
        "encoder": Path(sources["encoder"]), "protocol": protocol_path.resolve(),
        "audit_implementation": Path(__file__).resolve(),
    }
    entries = {name: record(path) for name, path in source_paths.items()}
    function_sources = {}
    for name, requested in FUNCTIONS.items():
        if entries[name]["status"] == MISSING:
            continue
        names, constants = definitions(source_paths[name])
        entries[name]["constants"] = constants
        for symbol in requested:
            function_sources[symbol] = {"source_key": name, "path": str(source_paths[name]),
                "sha256": entries[name]["sha256"], "line": names.get(symbol),
                "status": VERIFIED if symbol in names else MISSING}
    provenance = {"schema_version": SCHEMA, "created_utc": utc_now(), "files": entries,
                  "core_function_sources": function_sources,
                  "duplicate_source_agreement": {name: entries[name].get("sha256") == entries["legacy_" + name].get("sha256")
                                                 for name in ("student_model", "preference_core")}}
    write_json(output / "provenance/source_files.json", provenance)
    recovery = recovery_search(sources)
    expected_checkpoint = legacy / "results/experiments/delepref_evaluation_checkpoint/final_freeze/delepref_evaluation_step_00004000.pt"
    checkpoint_paths = [str(expected_checkpoint)] + sorted(set(recovery["checkpoint_candidates"]) - {str(expected_checkpoint)})
    checkpoints = [audit_checkpoint(path, envs["plm_python"], source_paths["legacy_training_config"],
                                   source_paths["student_model"], source_paths["legacy_evaluation_runtime"],
                                   protocol.get("legacy_formal_evaluation_manifest"))
                   for path in checkpoint_paths[:30]]
    recovery["checkpoint_validation_scope"] = {"candidate_count": len(checkpoint_paths),
        "validation_limit": 30, "validated_count": len(checkpoints),
        "validation_truncated": len(checkpoint_paths) > 30,
        "unvalidated_candidates": [{"path": path, "status": PENDING,
                                    "note": "Candidate validation cap reached."}
                                   for path in checkpoint_paths[30:]]}
    expected_library = legacy / "data/processed/preference_training/delepref"
    libraries = [library_inventory(path) for path in sorted(set([str(expected_library)] + recovery["preference_library_candidates"]))]
    environments = {
        "plm": environment_probe(envs["plm_python"], ["torch", "numpy", "esm.models.esmc", "safetensors", "yaml", "Bio"]),
        "esmfold": environment_probe(envs.get("esmfold_python", envs["plm_python"]), ["torch", "transformers", "esm.esmfold.v1.esmfold"]),
        "boltz": environment_probe(envs["boltz_python"], ["torch", "boltz.main", "rdkit", "lightning"]),
    }
    weight_reports = {
        "esmc": weight_inventory(weights["esmc"], ["model.safetensors"]),
        "esmfold": weight_inventory(weights["esmfold"], ["pytorch_model.bin", "config.json"]),
        "boltz": weight_inventory(weights["boltz_cache"], ["boltz2_conf.ckpt", "boltz2_aff.ckpt"]),
    }
    slurm = Path(envs["slurm_bin"])
    scheduler = {
        "partitions": run_readonly([slurm / "sinfo", "--noheader", "--format=%P|%a|%l|%G|%D"]),
        "personal_queue": run_readonly([slurm / "squeue", "--user", getpass.getuser(), "--noheader",
                                       "--format=%i|%P|%j|%T|%M|%D|%R"]),
        "user": getpass.getuser(), "read_only": True,
    }
    scheduler["status"] = VERIFIED if all(scheduler[key].get("returncode") == 0 for key in ("partitions", "personal_queue")) else PENDING
    cdhit = record(envs["cdhit"])
    cdhit["probe"] = run_readonly([envs["cdhit"], "-h"])
    if cdhit["status"] != MISSING:
        cdhit["status"] = VERIFIED if "CD-HIT" in cdhit["probe"].get("stdout", "") else PENDING
    cath = record(legacy / "data/par_contraction/cath_v4_3_0/cath-dataset-nonredundant-S40.pdb.tgz")
    data = {key: record(sources[key], hash_content=False) for key in ("general240", "e1", "pilot")}
    checkpoint_verified = any(x["status"] == VERIFIED for x in checkpoints)
    missing = [name for name, item in entries.items() if item["status"] == MISSING]
    missing += ["legacy_evaluation_checkpoint"] if not any(x["status"] != MISSING for x in checkpoints) else []
    missing += ["legacy_preference_library"] if not any(x["status"] != MISSING for x in libraries) else []
    pending = ["weights." + name for name, item in weight_reports.items() if item["status"] == PENDING]
    pending += ["environments." + name for name, item in environments.items() if item["status"] == PENDING]
    for group, items in (("environments", environments), ("weights", weight_reports), ("historical_data", data)):
        missing.extend(group + "." + name for name, item in items.items() if item["status"] == MISSING)
    if not checkpoint_verified:
        pending.append("original_delepref_full_baseline_unavailable")
    sources_fixed = len(function_sources) == sum(map(len, FUNCTIONS.values())) and all(x["status"] == VERIFIED for x in function_sources.values())
    core_ready = sources_fixed and environments["plm"]["status"] == VERIFIED and weight_reports["esmc"].get("weights_present_and_hashed", False)
    baseline_ready = checkpoint_verified and any(x["status"] == VERIFIED for x in libraries)
    report = {
        "schema_version": SCHEMA, "created_utc": utc_now(), "host": socket.gethostname(),
        "protocol": entries["protocol"], "stage": "ASSET_AUDIT_COMPLETE" if core_ready else "ASSET_AUDIT_CORE_PENDING",
        "inventory_complete": True, "audit_complete": core_ready, "reusable_baseline_complete": baseline_ready,
        "meaning": "All scoped audit sections recorded; readiness and unavailable assets are explicit below.",
        "source_manifest": {"path": str(output / "provenance/source_files.json"),
                            "sha256": sha256_file(output / "provenance/source_files.json")},
        "recovery": recovery, "legacy_checkpoints": checkpoints, "legacy_preference_libraries": libraries,
        "environments": environments, "weights": weight_reports, "scheduler": scheduler,
        "cdhit": cdhit, "cath_native_archive": cath, "historical_data_roots": data,
        "missing": missing, "profile_pending": pending,
        "readiness": {
            "source_functions_fixed": sources_fixed,
            "response_data_work_may_continue": core_ready and all(item["status"] != MISSING for item in data.values()),
            "original_delepref_full_baseline_verified": baseline_ready,
            "gpu_inference_or_structure_profile_validated": False,
        },
        "profile_constraints": {"requested_esmfold_profile": protocol.get("esmfold_profile"),
            "historical_predictor_profiles": "Not established by directory existence; candidate-level evidence is required by manifest audit.",
            "original_evaluation_budgets": [0.1, 0.2], "new_30_percent_is_original_protocol": False},
        "safety": {"historical_roots_modified": False, "dependencies_upgraded": False,
            "jobs_cancelled_or_submitted": False, "gpu_inference_run": False,
            "confirmation_manifest_contents_opened": False},
    }
    write_json(output / "reports/asset_audit.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit(args.protocol, args.output)
    print(json.dumps({key: report[key] for key in ("stage", "audit_complete", "missing", "profile_pending", "readiness")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
