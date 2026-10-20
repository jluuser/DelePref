"""Read-only-input development audit for the authorized confirm-v2 experiments."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def incumbent_audit(original, rag, tolerance=1e-5):
    if len(original) != 8 or len(rag) != 8:
        raise ValueError("Expected the two complete K8 candidate pools")
    if any(not np.isfinite(float(r["scorer_score"])) for r in original + rag):
        raise ValueError("Nonfinite scorer output")
    a = min(original, key=lambda r: r["scorer_rank"])
    b = min(rag, key=lambda r: r["scorer_rank"])
    present = any(r["deletion_mask_sha256"] == a["deletion_mask_sha256"] for r in rag)
    margin = float(a["scorer_score"]) - float(b["scorer_score"])
    return dict(original_winner_in_RAG_pool=present,
                original_winner_mask=a["deletion_mask_sha256"],
                RAG_winner_mask=b["deletion_mask_sha256"],
                winner_changed=a["deletion_mask_sha256"] != b["deletion_mask_sha256"],
                original_winner_score=float(a["scorer_score"]),
                RAG_winner_score=float(b["scorer_score"]),
                incumbent_minus_selected_score=margin,
                reinsertion_strictly_changes_scorer_choice=bool(not present and margin > tolerance),
                reinsertion_tie_unresolved=bool(not present and abs(margin) <= tolerance),
                scorer_numeric_tolerance=tolerance,
                selected_generation_kind=b.get("generation_kind"))


def span_features(row):
    raw = row["deleted_intervals"]
    spans = json.loads(raw) if isinstance(raw, str) else raw
    n = row["sequence_length"]
    return dict(span_count=len(spans), longest_span=max(b - a + 1 for a, b in spans),
                internal_junctions=sum(a > 0 and b < n - 1 for a, b in spans),
                terminal_deleted_residues=sum(b-a+1 for a,b in spans if a == 0 or b == n-1))


def loco(pairs, components, paired_summary):
    summary = paired_summary(pairs, components, 20261017)
    labels = sorted({components[p["protein_id"]] for p in pairs})
    omitted = []
    for label in labels:
        subset = [p for p in pairs if components[p["protein_id"]] != label]
        item = paired_summary(subset, components, 20261017)
        omitted.append(dict(omitted_component=label, **item))
    means = [p["construct_macro_difference_lower"] for p in omitted
             if "construct_macro_difference_lower" in p]
    return dict(full=summary, leave_one_component_out=omitted,
                leave_one_out_lower_mean_range=[min(means), max(means)] if means else None,
                no_case_removed_from_primary=True)


def main(root):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU Slurm allocation required")
    start = time.monotonic()
    out = root / "reports/recomb_confirm_v2_diagnostics"
    if (out / "completion.json").exists():
        raise FileExistsError("Completed audit is immutable")
    freeze = read(root / "configs/recomb_confirm_v2_diagnostics_freeze.json")
    for name, digest in freeze["files"].items():
        if sha(root / name) != digest:
            raise ValueError("Frozen source/input differs: " + name)
    from recomb_benign_structure_evaluate_20261017 import paired_summary
    panel = read(root / "reports/recomb_benign_scope_20261017/general_panel.json")
    apps = read(root / "reports/recomb_benign_application_constructs_20261017/panel.json")
    approved = {r["protein_id"] for r in panel + apps}
    if len(panel) != 44 or len(apps) != 23 or len(approved) != 67:
        raise ValueError("Unexpected positively reviewed development roster")
    route = {(r["protein_id"], r["budget"]): r for r in read(
        root / "reports/recomb_rag_decision_audit_20261017/per_condition.json")
        if r["generation_seed"] == 13}
    pools = defaultdict(list)
    for name in ["reports/recomb_benign_plans_20261017/candidate_scores.json",
                 "reports/recomb_benign_app_model_20261017/candidate_scores.json"]:
        for r in read(root / name):
            if r["protein_id"] not in approved:
                raise ValueError("Unreviewed input in candidate source")
            if r["budget"] == .2 and r["generation_seed"] == 13:
                pools[r["protein_id"], r["method"]].append(r)
    values = read(root / "reports/recomb_benign_structure_evaluation_20261017/per_output_metrics.json")
    metrics = {}
    for r in values:
        if r["protein_id"] not in approved:
            raise ValueError("Unreviewed input in metric source")
        expected = "esmfold" if r["cohort"] == "general" else (
            "boltz_binding" if r["cohort"] == "binding_interface" else "boltz_enzyme")
        if r["budget"] == .2 and r["profile_name"] == expected and r["method"] in (
                "delepref_unlimited", "rag_unlimited"):
            key = r["protein_id"], r["method"]
            if key in metrics:
                raise ValueError("Duplicate primary logical output")
            metrics[key] = r
    records, features, endpoint_pairs = [], [], defaultdict(list)
    for pid in sorted(approved):
        original, rag = pools[pid, "delepref_unlimited"], pools[pid, "rag_unlimited"]
        result = incumbent_audit(original, rag)
        a, b = metrics[pid, "delepref_unlimited"], metrics[pid, "rag_unlimited"]
        cohort = a["cohort"]
        rr = route[pid, .2]
        for key in ("participation", "fallback_reason", "reliable_representatives",
                    "reliable_internal_event_homologs", "retrieved_proposals",
                    "reliable_gap_coverage", "candidate_set_changed"):
            result[key] = rr.get(key)
        result.update(protein_id=pid, cohort=cohort)
        for endpoint in ("retained_ca_lddt", "contact_recovery"):
            lo = b["geometry"][endpoint+"_lower"] - a["geometry"][endpoint+"_upper"]
            hi = b["geometry"][endpoint+"_upper"] - a["geometry"][endpoint+"_lower"]
            result[endpoint+"_difference_lower"] = lo
            result[endpoint+"_difference_upper"] = hi
            endpoint_pairs[cohort, endpoint].append(dict(protein_id=pid, difference_lower=lo,
                                                       difference_upper=hi))
        if cohort == "binding_interface":
            endpoint_pairs[cohort, "interface_contact_recovery"].append(dict(protein_id=pid,
                difference_lower=b["interface"]["contact_recovery_lower"]-a["interface"]["contact_recovery_upper"],
                difference_upper=b["interface"]["contact_recovery_upper"]-a["interface"]["contact_recovery_lower"]))
        result["local_degradation_class"] = (
            "quota_loss_with_scorer_counterfactual" if result["retained_ca_lddt_difference_upper"] < -1e-8
                and result["reinsertion_strictly_changes_scorer_choice"] else
            "incumbent_available_selector_changed" if result["retained_ca_lddt_difference_upper"] < -1e-8
                and result["original_winner_in_RAG_pool"] else
            "incumbent_reinsertion_would_not_strictly_win" if result["retained_ca_lddt_difference_upper"] < -1e-8
                else "not_confirmed_local_degradation")
        records.append(result)
        for method, candidates in [("delepref_unlimited", original), ("rag_unlimited", rag)]:
            selected = min(candidates, key=lambda r: r["scorer_rank"])
            features.append(dict(protein_id=pid, cohort=cohort, method=method,
                                 **span_features(selected)))
    if len(records) != 67 or len(features) != 134:
        raise ValueError("Incomplete audit")
    components = read(root / "reports/recomb_benign_prediction_catalog_20261017/statistical_protocol.json")["component_by_protein_id"]
    influence = [dict(cohort=c, endpoint=e, **loco(p, components, paired_summary))
                 for (c,e),p in sorted(endpoint_pairs.items())]
    audit = read(root / "reports/recomb_benign_scope_20261017/general_audit.json")
    scope = [dict(protein_id=r["protein_id"], WT_sha256=r["WT_sha256"], status=r["status"],
                  reason=r["reason"], individual_decision_timestamp=None,
                  timestamp_note="Not in per-row audit; no timestamp invented",
                  prior_structure_results_existed=True, eligible_as_unseen_confirmation=False)
             for r in audit]
    for filename, data in [("incumbent_and_routing_audit.json", records),
                           ("leave_one_component_out.json", influence),
                           ("fragmentation_features.json", features),
                           ("historical_scope_flow.json", scope)]:
        write(out / filename, data)
    with (out / "incumbent_and_routing_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    report = dict(status="DEVELOPMENT_CPU_AUDIT_COMPLETE", development_WT=67,
                  cohorts=dict(Counter(r["cohort"] for r in records)),
                  incumbent_removed=sum(not r["original_winner_in_RAG_pool"] for r in records),
                  reinsertion_strictly_changes_choice=sum(r["reinsertion_strictly_changes_scorer_choice"] for r in records),
                  degradation_classes=dict(Counter(r["local_degradation_class"] for r in records)),
                  historical_general_disposition=dict(Counter(r["status"] for r in scope)),
                  new_PLM_forwards=0, new_training=False, new_structure_predictions=0,
                  seconds=time.monotonic()-start, job_id=os.environ["SLURM_JOB_ID"],
                  files={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    write(out / "completion.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
