"""Create the confirmation gate only after inputs, software and profiles pass."""
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys

from delepref_v8_structure_prepare import ROOT, sha, read, write


def main():
    target = ROOT / "configs/delepref_v8_confirmation_freeze.json"
    if target.exists():
        raise FileExistsError("confirmation already frozen; do not overwrite")
    preparation = read("reports/delepref_v8_structure_preparation/completion.json")
    assert preparation["status"] == "REQUESTS_AND_METRIC_PREPARATION_COMPLETE_NO_FOLD"
    assert preparation["logical_requests"] == 732 and preparation["unique_physical_requests"] <= 732
    assert read("reports/delepref_v8_main/semantic_audit.json")["status"] == "MAIN_100WT_OUTPUT_SEMANTICS_VERIFIED"
    files = []
    relative = [
        "analysis/delepref_v8_confirmation_freeze.py", "analysis/delepref_v8_structure_prepare.py",
        "analysis/delepref_v8_structure_metrics.py", "analysis/delepref_v8_structure_evaluate.py", "analysis/delepref_v8_fold_worker.py",
        "analysis/delepref_v8_predictor_preflight_v2.py", "analysis/tests/test_delepref_v8_structure_metrics.py",
        "code/build_pairs.py", "code/common.py", "code/paired_statistics.py", "code/budget_ledger.py", "code/adapters/structure_adapter.py",
        "configs/delepref_v8_main_freeze.json", "configs/delepref_v8_scisor_native_freeze_v3.json",
        "reports/delepref_v8_native_panel/main_panel.json", "reports/delepref_v8_main/semantic_audit.json",
        "reports/delepref_v8_main/primary_secondary_ablation_selections.json", "reports/delepref_v8_main/second_predictor_selections.json",
        "reports/delepref_v8_scisor_native_v3/native_plans.json", "reports/delepref_v8_scisor_native_v3/completion.json",
        "reports/delepref_v8_evaluation_preparation/statistical_protocol.json", "reports/delepref_v8_evaluation_preparation/subsets.json",
        "reports/delepref_v8_structure_preparation/predictor_profiles.json", "reports/delepref_v8_structure_preparation/metric_protocol.json",
        "reports/delepref_v8_structure_preparation/logical_requests.json", "reports/delepref_v8_structure_preparation/physical_requests.json",
        "reports/delepref_v8_structure_preparation/native_metric_checks.json", "reports/delepref_v8_structure_preparation/completion.json",
        "reports/delepref_v8_predictors/esmfold_preflight.json", "reports/delepref_v8_predictors/boltz2_preflight.json",
        "provenance/usalign_20261017/manifest.json", "provenance/usalign_20261017/USalign",
        "slurm/delepref_v8_fold_esmfold.sbatch", "slurm/delepref_v8_fold_boltz2.sbatch", "slurm/delepref_v8_structure_evaluate.sbatch"]
    for rel in relative:
        path = ROOT / rel
        assert path.is_file(), rel
        if path.suffix == ".py":
            compile(path.read_text(), str(path), "exec")
        files.append({"path": str(path), "sha256": sha(path)})
    for wt in read("reports/delepref_v8_native_panel/main_panel.json"):
        assert sha(wt["wt_native_json"]) == wt["wt_native_json_sha256"]
        files.append({"path": wt["wt_native_json"], "sha256": wt["wt_native_json_sha256"]})
    binary = ROOT / "provenance/usalign_20261017/USalign"
    assert sha(binary) == preparation["USalign_binary_sha256"]
    env = dict(os.environ, V8_USALIGN_BINARY=str(binary), CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "analysis/tests"),
                                "-p", "test_delepref_v8_structure_metrics.py", "-v"], env=env, capture_output=True, text=True)
    (ROOT / "reports/delepref_v8_structure_preparation/metric_tests.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError("metric tests failed; no folding gate")
    for predictor in ("esmfold", "boltz2"):
        report = read("reports/delepref_v8_predictors/" + predictor + "_preflight.json")
        assert report["status"] == "IDENTITY_AND_IMPORT_PASS_NO_FOLD"
        assert not report["structure_forward_executed"]
    files.append({"path": str(ROOT / "reports/delepref_v8_structure_preparation/metric_tests.log"),
                  "sha256": sha(ROOT / "reports/delepref_v8_structure_preparation/metric_tests.log")})
    write("configs/delepref_v8_confirmation_freeze.json", {"schema": "delepref_v8_complete_confirmation_v1",
        "fold_allowed": True, "all_selected_outputs_frozen": True, "paper_paused": True,
        "all_deletion_span_counts_allowed": True, "new_structure_predictions_at_freeze": 0,
        "logical_requests": 732, "unique_physical_requests": preparation["unique_physical_requests"],
        "max_new_complete_structure_predictions": 800, "max_simultaneous_task_GPU": 2,
        "predictor_reports": {p: "reports/delepref_v8_predictors/" + p + "_preflight.json" for p in ("esmfold", "boltz2")},
        "files": files, "no_model_or_selector_tuning_after_freeze": True,
        "frozen_seed": 20261017, "SCISOR_K1_separate_reference": True,
        "Scorer_more_than_3_spans_is_original_training_range_extrapolation": True,
        "primary_statistics": "100 paired WT/component bootstrap 10000, seed20261016; fixed native denominator",
        "unknown_coordinate_handling": "retained contributions intervals; no outcome-based row drop",
        "quality_gain_or_changed_selection_not_a_dispatch_requirement": True})
    print(json.dumps({"status": "COMPLETE_CONFIRMATION_FROZEN", "sha256": sha(target),
                      "logical_requests": 732, "physical_requests": preparation["unique_physical_requests"]}))


if __name__ == "__main__":
    main()
