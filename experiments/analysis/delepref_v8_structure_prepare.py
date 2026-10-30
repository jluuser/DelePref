"""Validate metric software and assemble all preselected structure requests.

Preparation has no predictor forward and does not authorize folding. A final
confirmation freeze binds this entry, worker, evaluator, metrics, predictors,
outputs, native references, subsets and the prior statistical protocol.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from delepref_v8_structure_metrics import relationship_metrics, fixed_full_wt_tm

ROOT = Path("/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def write(path, value):
    path = ROOT / path; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def kept_from_row(row, sequence):
    intervals = row.get("deleted_intervals", [])
    if isinstance(intervals, str):
        intervals = json.loads(intervals)
    deleted = []
    for a, b in intervals:
        if not 0 <= a <= b < len(sequence):
            raise ValueError("deletion outside WT")
        deleted.extend(range(a, b + 1))
    if len(set(deleted)) != len(deleted):
        raise ValueError("overlapping intervals")
    kept = [i for i in range(len(sequence)) if i not in set(deleted)]
    edited = "".join(sequence[i] for i in kept)
    if edited != row["edited_sequence"] or hashlib.sha256(edited.encode()).hexdigest() != row["edited_sequence_sha256"]:
        raise ValueError("selected output reconstruction/hash mismatch")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    out = "reports/delepref_v8_structure_preparation/"
    binary = ROOT / "provenance/usalign_20261017/USalign"
    panel = read("reports/delepref_v8_native_panel/main_panel.json")
    roster = {r["protein_id"]: r for r in panel}
    checks = []
    for wt in panel:
        path = Path(wt["wt_native_json"])
        assert sha(path) == wt["wt_native_json_sha256"]
        native = json.loads(path.read_text())
        assert native["wt_sequence_sha256"] == wt["sequence_sha256"]
        coords = np.full((len(wt["sequence"]), 3), np.nan)
        for k, v in native["coordinates"].items():
            coords[int(k)] = v
        identity = relationship_metrics(coords, coords, np.arange(len(coords)))
        assert identity["T_lower"] == identity["T_upper"] == 1.
        assert identity["contact_recovery_lower"] == 1.
        kept = np.arange(round(.2 * len(coords)), len(coords))
        perfect = relationship_metrics(coords, coords[kept], kept)
        assert perfect["T_lower"] == perfect["C"]
        failed = relationship_metrics(coords, np.full((len(kept), 3), np.nan), kept)
        assert failed["T_lower"] == 0. and failed["T_upper"] == perfect["C"]
        checks.append({"protein_id": wt["protein_id"], "native_pairs": identity["reference_pairs"],
                       "native_contacts": identity["native_contacts"], "known_CA": int(np.isfinite(coords).all(1).sum())})
    write(out + "native_metric_checks.json", {"status": "PASS", "WT": len(checks), "checks": checks,
          "synthetic_diagnostic_only": True, "structure_predictions": 0})
    print("NATIVE_METRIC_SOFTWARE_CHECKS_PASS_100WT", flush=True)
    if not args.assemble:
        return
    primary = read("reports/delepref_v8_main/primary_secondary_ablation_selections.json")
    scisor = read("reports/delepref_v8_scisor_native_v3/native_plans.json")
    second = read("reports/delepref_v8_main/second_predictor_selections.json")
    profiles = {}
    predictor_reports = {}
    for predictor in ("esmfold", "boltz2"):
        report_path = "reports/delepref_v8_predictors/" + predictor + "_preflight.json"
        report = read(report_path)
        assert report["status"] == "IDENTITY_AND_IMPORT_PASS_NO_FOLD" and not report["structure_forward_executed"]
        profile = dict(report["planned_profile"], software_version=report["version"],
                       preflight_report_sha256=sha(ROOT / report_path))
        if predictor == "boltz2":
            profile.update(no_trifast=True, step_scale=1.5, output_format="mmcif", max_parallel_samples=1,
                           precision="bf16-mixed", preprocessing_threads=1)
        profiles[predictor] = profile
        predictor_reports[predictor] = report
    write(out + "predictor_profiles.json", profiles)
    requests = []; physical = {}

    def add(row, category, predictor):
        wt = roster[row["protein_id"]]
        kept = kept_from_row(row, wt["sequence"])
        if row["method"] != "WT":
            assert len(wt["sequence"]) - len(kept) == round(float(row["budget"]) * len(wt["sequence"]))
        sequence = row["edited_sequence"]
        sequence_sha = hashlib.sha256(sequence.encode()).hexdigest()
        physical_id = predictor + "_" + sequence_sha
        logical = dict(protein_id=row["protein_id"], method=row["method"],
                       role=row.get("evaluation_role", category), budget=row.get("budget", 0.),
                       sequence_sha256=sequence_sha, kept=kept, predictor=predictor,
                       physical_id=physical_id, category=category)
        logical["request_id"] = digest(logical)
        requests.append(logical)
        if physical_id not in physical:
            report = predictor_reports[predictor]
            weight = next(f for f in report["files"] if f["path"].endswith(("pytorch_model.bin", "boltz2_conf.ckpt")))
            physical[physical_id] = dict(task_id=physical_id, sequence=sequence, sequence_sha256=sequence_sha,
                                        predictor=predictor, category=category,
                                        wt_variant="WT" if row["method"] == "WT" else "variant",
                                        weights_sha256=weight["sha256"], profile_sha256=digest(profiles[predictor]),
                                        seed=profiles[predictor]["seed"], sample=0)
    for wt in panel:
        add(dict(protein_id=wt["protein_id"], method="WT", edited_sequence=wt["sequence"],
                 edited_sequence_sha256=wt["sequence_sha256"]), "WT_controls", "esmfold")
    for row in primary:
        category = {"primary_20pct": "main_outputs", "secondary_10pct": "secondary_10pct",
                    "completion_ablation": "completion_ablation"}.get(row["evaluation_role"])
        if category is None:
            raise ValueError("unexpected evaluation role " + row["evaluation_role"])
        add(row, category, "esmfold")
    for row in scisor:
        row = dict(row, method="SCISOR", evaluation_role="SCISOR_native_20pct")
        add(row, "SCISOR", "esmfold")
    for row in second:
        add(row, "second_predictor", "boltz2")
    assert len(requests) == 732 and len({r["request_id"] for r in requests}) == 732
    protocol = read("reports/delepref_v8_evaluation_preparation/statistical_protocol.json")
    counts = Counter(r["category"] for r in physical.values())
    for key, count in counts.items():
        assert count <= protocol["compute"]["structure_categories"][key]
    assert len(physical) <= 732
    write(out + "logical_requests.json", requests)
    write(out + "physical_requests.json", list(physical.values()))
    metric_protocol = dict(primary=protocol["primary_endpoint"],
        retained_CA_lDDT="Mean over retained WT residues with at least one retained known-native neighbor; native CA distance<15A and WT sequence separation>=3. Per-residue mean of four strict thresholds .5/1/2/4A. Unknown predicted endpoints contribute [0,1]; isolated residues counted separately. Operational CA metric, not all-atom lDDT.",
        full_WT_TM="Official USalign fixed supplied one-to-one known-CA correspondence (-I), no fast search, normalization and protein d0 from full WT sequence length (-u). Deleted positions contribute zero. Unknown retained positions [0,1] conditional on this known-CA transform. No free realignment and no Kabsch-only TM approximation.",
        contacts="Native CA distance<8A and WT separation>=3; full WT native-contact denominator; deleted endpoints zero, retained unknown [0,1]. Recovery means predicted CA distance<8A.",
        common_retained="RAG/original common retained WT positions; compare errors on exactly the same native pairs. Diagnostic only.",
        failures="Keep every frozen WT/method. All retained predicted coordinates unknown. TM no-fit [0,retained/fullWT]; primary T [0,static retained-pair fraction]. No outcome-based deletion or mapping changes.")
    write(out + "metric_protocol.json", metric_protocol)
    write(out + "completion.json", {"status": "REQUESTS_AND_METRIC_PREPARATION_COMPLETE_NO_FOLD",
        "logical_requests": len(requests), "unique_physical_requests": len(physical),
        "physical_counts": dict(counts), "predictor_counts": dict(Counter(r["predictor"] for r in physical.values())),
        "remaining_structure_budget": 800, "structure_predictions": 0, "fold_allowed": False,
        "USalign_binary_sha256": sha(binary), "USalign_commit": read("provenance/usalign_20261017/manifest.json")["commit"],
        "USalign_compile_command": "g++ -std=c++11 -include cmath -O3 -o USalign USalign.cpp",
        "inputs": [{"path": p, "sha256": sha(ROOT / p)} for p in [
            "reports/delepref_v8_native_panel/main_panel.json", "reports/delepref_v8_main/semantic_audit.json",
            "reports/delepref_v8_main/primary_secondary_ablation_selections.json",
            "reports/delepref_v8_main/second_predictor_selections.json",
            "reports/delepref_v8_scisor_native_v3/native_plans.json",
            "reports/delepref_v8_evaluation_preparation/statistical_protocol.json"]]})
    print(json.dumps({"logical": len(requests), "physical": len(physical), "counts": dict(counts)}))


if __name__ == "__main__":
    main()
