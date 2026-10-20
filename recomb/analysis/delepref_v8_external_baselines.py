"""Original teacher U and published M=1 SCISOR denoiser on fixed observations."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from delepref_v8_external_baseline_core import (load_original_teacher, original_utility,
                                              residue_deletion_probabilities, scisor_mapping_views, paired_target_difference)

ROOT = Path("/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013")
SOURCE = Path("/public/home/zhangyangroup/chengshiz/keyuan.zhou/SurgEdit")
BASE = "external/delepref_v8_baselines"
FREEZE = "configs/delepref_v8_external_baselines_freeze.json"
DATA = "external/rag_v6/preparation/primary_rows.jsonl"
PROTOCOL = "external/rag_v6/preparation/protocol.json"
CONTRACT = "external/rag_v6/preparation/evaluation_entry_contract.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def write(path, value):
    path = ROOT / path; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    with (ROOT / path).open("x") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def prepare():
    if (ROOT / FREEZE).exists():
        raise FileExistsError("external baseline protocol already frozen")
    inputs = read("external/delepref_v8_scorer/inputs.json")
    assert len(inputs["wildtypes"]) == 5 and len(inputs["edits"]) == 866 and len(inputs["links"]) == 1422
    sequences = dict.fromkeys(inputs["wildtypes"].values())
    for edit in inputs["edits"]:
        wt = edit["wt_sequence"]; pos = edit["deleted_position"]
        assert wt == inputs["wildtypes"][edit["target"]]
        sequences[wt[:pos] + wt[pos + 1:]] = None
    assert len(sequences) == 802
    inventory = []
    old_files = [
        SOURCE / "results/experiments/student_v2_step00295000_exp6_enzyme_fitness_benchmark/enzyme/measured_interval_scoring.csv",
        SOURCE / "results/experiments/student_v3_step004000_v3_20_direct_plm_cross_teacher/direct_plm_candidate_scores.csv"]
    wt_hashes = {hashlib.sha256(s.encode()).hexdigest() for s in inputs["wildtypes"].values()}
    fixed_edits = {(hashlib.sha256(r["wt_sequence"].encode()).hexdigest(), r["deleted_position"]) for r in inputs["edits"]}
    for path in old_files:
        with path.open(newline="") as f:
            reader = csv.DictReader(f); columns = reader.fieldnames; count = 0; exact_WT_rows = 0; reusable = 0
            for row in reader:
                count += 1
                sequence = row.get("wildtype_sequence") or row.get("protein_sequence")
                if sequence and hashlib.sha256(sequence.encode()).hexdigest() in wt_hashes:
                    exact_WT_rows += 1
                    intervals = json.loads(row.get("deleted_intervals") or "[]")
                    if len(intervals) == 1 and intervals[0][0] == intervals[0][1] and (
                        hashlib.sha256(sequence.encode()).hexdigest(), intervals[0][0]) in fixed_edits:
                        reusable += 1
        inventory.append(dict(path=str(path), sha256=sha(path), rows=count, columns=columns,
                              exact_fixed_WT_rows=exact_WT_rows,
                              exact_fixed_single_deletion_rows=reusable,
                              has_teacher_views=all(k in columns for k in ("global_similarity", "retained_similarity", "junction_plausibility"))))
    if any(r["has_teacher_views"] and r["exact_fixed_single_deletion_rows"] for r in inventory):
        raise RuntimeError("potential original teacher cache found; audit its profile before scheduling re-encoding")
    write(BASE + "/legacy_reuse_audit.json", {"tables": inventory,
        "decision": "No verified reusable teacher U matrix for the fixed cohort: legacy enzyme table contains student_v2 only; prior direct-PLM table uses a different generation cohort. Re-encode all802 with the exact original teacher path.",
        "old_original_teacher_git_commit": "239942a",
        "cleaned_teacher_not_used": "Current renamed teacher_scoring.py lacks the old mean helper; use complete unmodified historical script instead.",
        "cache_decision": "Do not mix the5WT Scorer cache with a teacher profile that lacks explicit equivalent lineage; re-encoding5WT is cheap.",
        "cohort_unchanged": True, "new_structure_predictions": 0})
    write(BASE + "/sequence_inventory.json", [{"sequence": s, "sha256": hashlib.sha256(s.encode()).hexdigest(), "length": len(s)}
                                              for s in sorted(sequences, key=lambda s: (len(s), hashlib.sha256(s.encode()).hexdigest()))])
    scisor = read("configs/delepref_v8_scisor_native_freeze_v3.json")
    snapshot = Path("/public/home/zhangyangroup/chengshiz/keyuan.zhou/model_cache/esmc_600m/snapshot")
    vendor = SOURCE / "results/experiments/student_v2_step00295000_exp6_enzyme_fitness_benchmark/audit/esm_runtime_vendor"
    assets = {str(snapshot / "model.safetensors"): sha(snapshot / "model.safetensors")}
    for asset in scisor["assets"].values():
        assert sha(asset["path"]) == asset["sha256"]
        assets[asset["path"]] = asset["sha256"]
    sources = ["analysis/delepref_v8_external_baseline_core.py", "analysis/delepref_v8_external_baselines.py",
        "analysis/delepref_v8_scisor_compat.py", "analysis/delepref_v8_scisor_inference_import_v2.py",
        "analysis/tests/test_delepref_v8_external_baselines.py", "analysis/rag_v6_external_evaluate.py",
        "analysis/rag_v6_retrieval.py", "slurm/delepref_v8_external_baselines.sbatch", "slurm/delepref_v8_external_baseline_metrics.sbatch",
        "provenance/original_teacher_239942a/score_scisor_style_stage1_segments.py",
        "provenance/original_teacher_239942a/score_v3_b20_esmc600m.py"]
    fixed = [DATA, PROTOCOL, CONTRACT, "external/delepref_v8_scorer/inputs.json", "external/delepref_v8_scorer/mapping_scores.jsonl",
        "configs/delepref_v8_external_scorer_freeze.json", "configs/delepref_v8_scisor_native_freeze_v3.json",
        BASE + "/legacy_reuse_audit.json", BASE + "/sequence_inventory.json", BASE + "/semantic_tests.log"]
    files = {str(ROOT / p): sha(ROOT / p) for p in sources + fixed}
    files.update(scisor["server_source_hashes"])
    files.update({str(p): sha(p) for p in vendor.rglob("*.py")})
    write(FREEZE, {"schema": "delepref_v8_fixed_external_baselines_v1", "status": "FROZEN_BEFORE_BASELINE_SCORES",
        "files": files, "assets": assets, "teacher_snapshot": str(snapshot), "teacher_vendor": str(vendor),
        "SCISOR_checkpoint": scisor["assets"]["checkpoint"]["path"], "WT": 5, "sequences_encoded_for_U": 802,
        "physical_observations": 1317, "unique_edit_mappings": 866, "mapping_predictions": 1422,
        "teacher_U": {"original_commit": "239942a", "weights": [.35, .45, .20], "mutant_batch_size": 8,
            "WT_batch_size": 1, "views": "original real_plm_teacher_impact_from_embeddings including original6-decimal field serialization",
            "dtype": "bfloat16 weights and CUDA autocast; original final residue embeddings excluding BOS/EOS",
            "unknown_cache_not_reused": True, "all802_encoded_once": True},
        "SCISOR": {"source": "https://arxiv.org/html/2511.07390v2#S5", "appendix": "https://arxiv.org/html/2511.07390v2#A2.SS5",
            "conditioning": "M=1; t=0 ignored when S provided to the network; original model_predict, temperature1",
            "primary": "SCISOR_U90_S_M1_probability_common_map_mean",
            "ambiguity": "fixed evaluator averages all legal coordinate-event probabilities, matching the existing common mapping protocol",
            "secondary": "SCISOR_U90_S_M1_sequence_marginal_probability sums all equivalent one-deletion event probabilities; descriptive diagnostic",
            "not_claimed": "Exact reproduction of the authors'61-assay aggregate or final ICLR full-size-best-model benchmark",
            "5_WT_forwards": True, "no_generation": True, "no_outcome_based_probability_vs_log_choice": True},
        "evaluation": "same frozen8conditions1317rows, measurement direction and mapping protocol; target macro plus paired differences; all favorable/unfavorable results",
        "seed": 20261017, "tf32_allowed": False, "GPU_only_Slurm": True, "max_task_GPUs": 2,
        "resource_dependency": "Run only after Boltz21116910 completes, while ESMFold array remains throttle1; this job uses1GPU",
        "measured_labels_used_for_fitting": False, "new_training": False, "new_structure_predictions": 0,
        "no_phenotype_ranked_variant_recommendations": True, "paper_paused": True})
    print(json.dumps({"status": "FIXED_EXTERNAL_BASELINES_FROZEN", "sha256": sha(ROOT / FREEZE), "sequences": 802}))


def verified(checksum):
    if sha(ROOT / FREEZE) != checksum:
        raise ValueError("external baseline freeze changed")
    freeze = read(FREEZE)
    for path, expected in {**freeze["files"], **freeze["assets"]}.items():
        if sha(path) != expected:
            raise ValueError("external source/input/weights changed: " + path)
    return freeze


def execute(checksum):
    import torch
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Slurm GPU allocation required")
    freeze = verified(checksum)
    torch.set_num_threads(4); torch.manual_seed(freeze["seed"]); np.random.seed(freeze["seed"])
    torch.backends.cuda.matmul.allow_tf32 = freeze["tf32_allowed"]
    write(BASE + "/execution_claim.json", {"job_id": os.environ["SLURM_JOB_ID"], "freeze_sha256": checksum})
    inputs = read("external/delepref_v8_scorer/inputs.json")
    from delepref_v8_scisor_compat import patch_tokenizer
    from delepref_v8_scisor_inference_import_v2 import load_native_class
    sys.path.insert(0, str(SOURCE))
    started = time.monotonic()
    model = load_native_class().load_from_checkpoint(freeze["SCISOR_checkpoint"], map_location="cpu").cuda().eval()
    patch_tokenizer(model.tokenizer)
    print("SCISOR_EXTERNAL_M1_MODEL_LOADED", flush=True)
    by_target = {}
    with torch.inference_mode():
        for target, sequence in sorted(inputs["wildtypes"].items()):
            ids = model.tokenizer([sequence], return_tensors="pt", padding=True)["input_ids"].cuda()
            assert ids.shape == (1, len(sequence) + 2)
            decoded = model.tokenizer.convert_ids_to_tokens(ids[0, 1:-1].tolist())
            assert "".join(decoded) == sequence
            logits = model.model_predict(ids, torch.zeros(1, device="cuda"), None, torch.ones(1, device="cuda"))
            by_target[target] = residue_deletion_probabilities(logits.detach().float().cpu().numpy(), len(sequence))
    probabilities = {r["edit_key"]: float(by_target[r["target"]][r["deleted_position"]]) for r in inputs["edits"]}
    primary, marginal = scisor_mapping_views(inputs["links"], probabilities)
    write_jsonl(BASE + "/scisor_M1_mapping_scores.jsonl", primary)
    write_jsonl(BASE + "/scisor_M1_sequence_marginal_scores.jsonl", marginal)
    scisor_seconds = time.monotonic() - started
    del model, logits, ids; gc.collect(); torch.cuda.empty_cache()
    print("SCISOR_EXTERNAL_M1_SCORES_COMPLETE_5WT", flush=True)
    sys.path.insert(0, freeze["teacher_vendor"])
    teacher = load_original_teacher(ROOT)
    args = SimpleNamespace(teacher_model_name_or_path=freeze["teacher_snapshot"],
        teacher_backend=teacher.ESMC_OFFICIAL_SAFETENSORS_BACKEND, teacher_device="cuda", teacher_cache_dir=None,
        teacher_allow_download=False, teacher_model="frozen_esmc_600m_delepref", teacher_group_by_protein=True,
        teacher_batch_size=8, deletion_perturbations_csv=str(ROOT / DATA))
    started = time.monotonic()
    backend = teacher.load_real_plm_teacher_backend(args)
    print("ORIGINAL_TEACHER_ESMC_MODEL_LOADED", flush=True)
    embeddings = {}
    with torch.inference_mode():
        for sequence in inputs["wildtypes"].values():
            key = hashlib.sha256(sequence.encode()).hexdigest()
            if key not in embeddings:
                embeddings[key] = teacher.residue_embeddings_for_sequences(backend, [sequence], batch_size=1)[0].detach().cpu()
        inventory = read(BASE + "/sequence_inventory.json")
        remaining = [r for r in inventory if r["sha256"] not in embeddings]
        for start in range(0, len(remaining), 8):
            chunk = remaining[start:start + 8]
            encoded = teacher.residue_embeddings_for_sequences(backend, [r["sequence"] for r in chunk], batch_size=8)
            for row, value in zip(chunk, encoded):
                embeddings[row["sha256"]] = value.detach().cpu()
            del encoded
            print("ORIGINAL_TEACHER_ENCODED " + str(len(embeddings)) + "/802", flush=True)
        assert len(embeddings) == 802
        views = {}; scores = {}
        for edit in inputs["edits"]:
            wt = edit["wt_sequence"]; pos = edit["deleted_position"]; mutant = wt[:pos] + wt[pos + 1:]
            row = dict(protein_id=edit["target"], perturbation_id=edit["edit_key"], plan_id=edit["edit_key"],
                wildtype_sequence=wt, wildtype_length=len(wt), mutant_sequence=mutant,
                coordinate_system="0_based_closed_inclusive", deleted_intervals=json.dumps([[pos, pos]]),
                deleted_length=1, num_intervals=1, deletion_fraction=1 / len(wt))
            impact = teacher.real_plm_teacher_impact_from_embeddings(row, args, backend,
                embeddings[hashlib.sha256(wt.encode()).hexdigest()].cuda(),
                embeddings[hashlib.sha256(mutant.encode()).hexdigest()].cuda())
            scores[edit["edit_key"]] = original_utility(impact)
            views[edit["edit_key"]] = {k: float(impact[k]) for k in ("global_similarity", "retained_similarity", "junction_plausibility")}
    teacher_rows = [dict(row_id=r["row_id"], deleted_position=r["deleted_position"], score=scores[r["edit_key"]],
                         **views[r["edit_key"]]) for r in inputs["links"]]
    write_jsonl(BASE + "/original_U_mapping_scores.jsonl", teacher_rows)
    for view in ("global_similarity", "retained_similarity", "junction_plausibility"):
        write_jsonl(BASE + "/original_U_" + view + "_mapping_scores.jsonl",
                    [dict(row_id=r["row_id"], deleted_position=r["deleted_position"], score=r[view]) for r in teacher_rows])
    write(BASE + "/scoring_completion.json", {"status": "FIXED_EXTERNAL_BASELINE_SCORES_COMPLETE", "job_id": os.environ["SLURM_JOB_ID"],
        "freeze_sha256": checksum, "WT": 5, "unique_edit_mappings": 866, "mapping_predictions": 1422,
        "SCISOR_WT_forwards": 5, "teacher_sequence_encodings": len(embeddings), "teacher_WT_encodings": 5,
        "teacher_variant_sequence_encodings": len(embeddings) - 5, "SCISOR_seconds": scisor_seconds,
        "teacher_seconds": time.monotonic() - started, "GPU_name": torch.cuda.get_device_name(),
        "max_GPU_memory_bytes": torch.cuda.max_memory_allocated(), "new_structure_predictions": 0,
        "scores": {p.name: sha(p) for p in (ROOT / BASE).glob("*_scores.jsonl")},
        "sequence_alignment_notes": sorted(backend.get("embedding_alignment_notes", [])),
        "phenotype_ranked_edit_recommendations": False})
    print("FIXED_EXTERNAL_BASELINE_SCORES_COMPLETE", flush=True)


def metrics(checksum):
    verified(checksum)
    completion = read(BASE + "/scoring_completion.json")
    for name, expected in completion["scores"].items():
        assert sha(ROOT / BASE / name) == expected
    methods = {"original_U": "original_U_mapping_scores.jsonl",
        "SCISOR_U90_S_M1_common_map_mean": "scisor_M1_mapping_scores.jsonl",
        "SCISOR_U90_S_M1_sequence_marginal_diagnostic": "scisor_M1_sequence_marginal_scores.jsonl"}
    methods.update({"original_U_" + v: "original_U_" + v + "_mapping_scores.jsonl"
                    for v in ("global_similarity", "retained_similarity", "junction_plausibility")})
    for method, filename in methods.items():
        subprocess.run([sys.executable, str(ROOT / "analysis/rag_v6_external_evaluate.py"),
            "--dataset", str(ROOT / DATA), "--protocol", str(ROOT / PROTOCOL), "--entry-contract", str(ROOT / CONTRACT),
            "--scores", str(ROOT / BASE / filename), "--level", "mapping", "--method", method,
            "--output", str(ROOT / BASE / (method + "_aggregate_metrics.json"))], check=True)
    subprocess.run([sys.executable, str(ROOT / "analysis/rag_v6_external_evaluate.py"),
        "--dataset", str(ROOT / DATA), "--protocol", str(ROOT / PROTOCOL), "--entry-contract", str(ROOT / CONTRACT),
        "--scores", str(ROOT / "external/delepref_v8_scorer/mapping_scores.jsonl"), "--level", "mapping",
        "--method", "original_Scorer", "--output", str(ROOT / BASE / "original_Scorer_aggregate_metrics.json")], check=True)
    names = ["original_Scorer"] + list(methods)
    aggregates = {name: read(BASE + "/" + name + "_aggregate_metrics.json") for name in names}
    comparisons = {"original_Scorer_minus_" + name: paired_target_difference(
        aggregates["original_Scorer"]["target_mean_Spearman"], aggregates[name]["target_mean_Spearman"])
        for name in ("original_U", "SCISOR_U90_S_M1_common_map_mean")}
    write(BASE + "/paired_baseline_comparisons.json", comparisons)
    write(BASE + "/baseline_matrix.json", {name: {k: value[k] for k in (
        "macro_Spearman", "conditional_protein_bootstrap_CI95", "target_mean_Spearman", "assays", "per_target_cluster_intervals") if k in value}
        for name, value in aggregates.items()})
    write(BASE + "/metrics_completion.json", {"status": "FIXED_EXTERNAL_BASELINE_AGGREGATION_COMPLETE",
        "freeze_sha256": checksum, "physical_observations": 1317, "methods": list(methods),
        "no_outcome_based_mapping_direction_or_method_selection": True, "new_structure_predictions": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "score", "metrics"], required=True)
    parser.add_argument("--freeze-sha256")
    args = parser.parse_args()
    {"prepare": lambda: prepare(), "score": lambda: execute(args.freeze_sha256),
     "metrics": lambda: metrics(args.freeze_sha256)}[args.stage]()
