"""Evaluate every frozen output, retaining failures and original residue maps."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from delepref_v8_structure_metrics import relationship_metrics, fixed_full_wt_tm, common_retained_diagnostic
from delepref_v8_structure_prepare import ROOT, sha, read, write
sys.path.insert(0, str(ROOT / "code"))
from budget_ledger import BudgetLedger
from paired_statistics import paired_bootstrap


def main():
    freeze = read("configs/delepref_v8_confirmation_freeze.json")
    for r in freeze["files"]:
        assert sha(r["path"]) == r["sha256"], r["path"]
    panel = read("reports/delepref_v8_native_panel/main_panel.json")
    natives = {}; roster = {}
    for wt in panel:
        assert sha(wt["wt_native_json"]) == wt["wt_native_json_sha256"]
        native = json.loads(Path(wt["wt_native_json"]).read_text())
        coords = np.full((len(wt["sequence"]), 3), np.nan)
        for k, v in native["coordinates"].items():
            coords[int(k)] = v
        natives[wt["protein_id"]] = coords
        roster[wt["protein_id"]] = wt["panel_component"]
    protocol_path = ROOT / "reports/delepref_v8_evaluation_preparation/statistical_protocol.json"
    ledger = BudgetLedger(protocol_path, ROOT / "reports/delepref_v8_structures/budget_ledger.jsonl")
    attempts = ledger.snapshot()["attempts"]
    states = {r["task_id"]: r for r in attempts}
    physical = read("reports/delepref_v8_structure_preparation/physical_requests.json")
    if any(r["task_id"] not in states or states[r["task_id"]]["state"] not in ("succeeded", "cache_hit", "failed") for r in physical):
        raise RuntimeError("frozen requests remain pending; do not aggregate an incomplete cohort")
    predictions = {}
    for request in physical:
        state = states[request["task_id"]]
        if state["state"] in ("succeeded", "cache_hit"):
            assert sha(state["output_path"]) == state["output_sha"]
            with np.load(state["output_path"], allow_pickle=False) as a:
                assert a["sequence"].item() == request["sequence"]
                assert a["sequence_sha256"].item() == request["sequence_sha256"]
                predictions[request["task_id"]] = a["coordinates"].copy()
        else:
            predictions[request["task_id"]] = np.full((len(request["sequence"]), 3), np.nan)
    requests = read("reports/delepref_v8_structure_preparation/logical_requests.json")
    results = []; paired = []; common = []; lookup = {}
    binary = ROOT / "provenance/usalign_20261017/USalign"
    for row in requests:
        coords = predictions[row["physical_id"]]
        metrics = relationship_metrics(natives[row["protein_id"]], coords, row["kept"])
        tm = fixed_full_wt_tm(natives[row["protein_id"]], coords, row["kept"], binary)
        result = dict(row, metrics=metrics, full_WT_fixed_TM=tm, prediction_status=states[row["physical_id"]]["state"])
        results.append(result)
        lookup[(row["predictor"], row["role"], row["protein_id"], row["method"])] = row
        if row["predictor"] == "esmfold" and row["role"] == "primary_20pct":
            paired.append(dict(wt_id=row["protein_id"], group_id=roster[row["protein_id"]], method=row["method"],
                               lower=metrics["T_lower"], upper=metrics["T_upper"]))
    for wt in roster:
        a = lookup[("esmfold", "primary_20pct", wt, "rag_unlimited")]
        b = lookup[("esmfold", "primary_20pct", wt, "delepref_unlimited")]
        common.append(dict(protein_id=wt, **common_retained_diagnostic(natives[wt], predictions[a["physical_id"]], a["kept"],
                                                                  predictions[b["physical_id"]], b["kept"])))
    protocol = json.loads(protocol_path.read_text())
    bootstrap = paired_bootstrap(paired, "rag_unlimited", "delepref_unlimited", roster,
                                 replicates=10000, seed=20261016)
    out = "reports/delepref_v8_structure_evaluation/"
    write(out + "per_output_metrics.json", results)
    write(out + "common_retained_diagnostics.json", common)
    write(out + "primary_paired_statistics.json", bootstrap)
    summaries = []
    conditions = sorted({(r["predictor"], r["role"], r["method"]) for r in results})
    for predictor, role, method in conditions:
        rows = [r for r in results if (r["predictor"], r["role"], r["method"]) == (predictor, role, method)]
        summary = dict(predictor=predictor, role=role, method=method, outputs=len(rows),
                       prediction_status=dict(Counter(r["prediction_status"] for r in rows)))
        for key in ("T_lower", "T_upper", "G_lower", "G_upper", "retained_ca_lddt_lower", "retained_ca_lddt_upper",
                    "contact_recovery_lower", "contact_recovery_upper", "static_relationship_loss", "static_contact_loss"):
            values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
            summary[key] = float(np.mean(values)) if values else None
            summary[key + "_defined_WT"] = len(values)
        for key in ("lower", "upper"):
            summary["full_WT_fixed_TM_" + key] = float(np.mean([r["full_WT_fixed_TM"][key] for r in rows]))
        summaries.append(summary)
    write(out + "descriptive_condition_summaries.json", summaries)
    write(out + "completion.json", {"status": "ALL_FROZEN_OUTPUTS_EVALUATED", "logical_outputs": len(results),
          "physical_requests": len(physical), "physical_status": dict(Counter(states[r["task_id"]]["state"] for r in physical)),
          "WT": len(roster), "confirmation_freeze_sha256": sha(ROOT / "configs/delepref_v8_confirmation_freeze.json"),
          "primary_difference_bounds": bootstrap["difference_bounds"], "primary_95pct_interval": bootstrap["confidence_interval"],
          "biological_function_established": False, "all_positive_and_negative_cases_retained": True})


if __name__ == "__main__":
    main()
