#!/usr/bin/env python3
"""Compare canonical DelePref and SCISOR contact retention at 10% and 20%."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/delepref_step004000_delepref_24_binder_budget_contact_comparison"
B10 = ROOT / "results/experiments/delepref_step004000_delepref_23_binder_b10_contact_metrics"
B20 = ROOT / "results/experiments/delepref_step004000_delepref_22_binder_budget_case_search"
RUNTIME_PATH = ROOT / "scripts/run_delepref_22_binder_budget_case_search.py"
EXPECTED_CASES = 22
EXPECTED_NEW_PREDICTIONS = {"0.10": 10, "0.20": 14}
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260725
METRICS = ("contact_recovery", "contact_jaccard")
METHODS = ("delepref", "scisor")
BUDGETS = ("0.10", "0.20")
STATUS = "PASS_DelePref_24_BINDER_BUDGET_CONTACT_COMPARISON"
CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion "
    "perturbation preference and optional auxiliary risk evidence."
)
PROXY_BOUNDARY = (
    "Contact recovery and Jaccard are post-deletion Boltz-2 complex-prediction "
    "proxies, not evidence of experimental binding, Kd/Ki, or functional preservation."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, value: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def load_runtime() -> Any:
    spec = importlib.util.spec_from_file_location("delepref_22_runtime_for_delepref_24", RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Binder runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def source_by_budget() -> dict[str, Path]:
    return {"0.10": B10, "0.20": B20}


def require_source_packages() -> None:
    expected = {
        B10: "PASS_DelePref_23_BINDER_B10_CONTACT_METRICS",
        B20: "PASS_DelePref_22_BINDER_B20_CASE_FOUND",
    }
    for root, status in expected.items():
        observed = load_json(root / "run_summary.json").get("status")
        validation = load_json(root / "independent_validation.json").get("status")
        if observed != status or validation != "PASS":
            raise RuntimeError(f"source package is not validated: {root} ({observed}, {validation})")


def prepare() -> None:
    require_source_packages()
    if (OUT / "protocol.json").exists():
        raise FileExistsError(OUT / "protocol.json")
    runtime = load_runtime()
    source_hashes: dict[str, dict[str, str]] = {}
    case_sets: dict[str, list[str]] = {}
    plans: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    input_dir = OUT / "boltz_inputs"

    for budget, source in source_by_budget().items():
        source_hashes[budget] = {
            name: sha256_file(source / name)
            for name in (
                "protocol.json",
                "run_summary.json",
                "independent_validation.json",
                "cohort.csv",
                "delepref_canonical_scorer_top1.csv",
                "selected_plans.csv",
                "case_metrics.csv",
            )
        }
        cohort = {row["protein_id"]: row for row in read_csv(source / "cohort.csv")}
        canonical = {row["protein_id"]: row for row in read_csv(source / "delepref_canonical_scorer_top1.csv")}
        selected = {
            row["protein_id"]: row
            for row in read_csv(source / "selected_plans.csv")
            if row["method"] == "ours"
        }
        metrics = {
            row["protein_id"]: row
            for row in read_csv(source / "case_metrics.csv")
            if row["method"] == "ours"
        }
        if not (set(cohort) == set(canonical) == set(selected) == set(metrics)):
            raise RuntimeError(f"source protein identities differ at budget {budget}")
        case_sets[budget] = sorted((row["case_id"] for row in cohort.values()), key=int)

        for protein_id in sorted(cohort, key=lambda key: int(cohort[key]["case_id"])):
            item = cohort[protein_id]
            candidate = canonical[protein_id]
            selected_plan = selected[protein_id]
            sequence = item["wildtype_sequence"]
            intervals = runtime.parse_intervals(candidate["deleted_intervals"])
            edited = runtime.mutant(sequence, intervals)
            edited_sha = sha256_text(edited)
            deleted_length = sum(end - start + 1 for start, end in intervals)
            expected_deleted = max(1, min(len(sequence) - 1, int(round(float(budget) * len(sequence)))))
            if deleted_length != expected_deleted or len(sequence) - len(edited) != expected_deleted:
                raise RuntimeError(f"canonical exact-budget failure for {protein_id} at {budget}")
            reused = candidate["deletion_mask_sha256"] == selected_plan["deletion_mask_sha256"]
            if reused:
                record_id = metrics[protein_id]["record_id"]
                yaml_path = Path(selected_plan["yaml_path"])
                confidence_path = Path(metrics[protein_id]["confidence_json"])
                structure_path = Path(metrics[protein_id]["structure_cif"])
                prediction_source = "REUSED_IDENTICAL_CANONICAL_PLAN"
            else:
                record_id = f"case{int(item['case_id']):03d}_delepref24_b{int(100 * float(budget)):02d}_delepref_{edited_sha[:12]}"
                yaml_path = input_dir / f"{record_id}.yaml"
                runtime.yaml_with_edited_binder(
                    Path(selected_plan["wt_yaml_path"]), yaml_path, sequence, edited
                )
                confidence_path = Path("")
                structure_path = Path("")
                prediction_source = "NEW_CANONICAL_PLAN_PREDICTION"
                jobs.append(
                    {
                        "record_id": record_id,
                        "budget": budget,
                        "case_id": item["case_id"],
                        "protein_id": protein_id,
                        "yaml_path": str(yaml_path),
                        "yaml_sha256": sha256_file(yaml_path),
                    }
                )
            plans.append(
                {
                    "case_id": item["case_id"],
                    "protein_id": protein_id,
                    "budget": budget,
                    "wildtype_sequence": sequence,
                    "deleted_intervals": candidate["deleted_intervals"],
                    "deleted_length": deleted_length,
                    "deletion_mask_sha256": candidate["deletion_mask_sha256"],
                    "scorer_score": candidate["scorer_score"],
                    "edited_sequence": edited,
                    "edited_sequence_sha256": edited_sha,
                    "prediction_source": prediction_source,
                    "record_id": record_id,
                    "yaml_path": str(yaml_path),
                    "confidence_json": str(confidence_path) if reused else "",
                    "structure_cif": str(structure_path) if reused else "",
                    "wt_yaml_path": selected_plan["wt_yaml_path"],
                    "wt_confidence_json": selected_plan["wt_confidence_json"],
                }
            )

    if case_sets["0.10"] != case_sets["0.20"] or len(case_sets["0.10"]) != EXPECTED_CASES:
        raise RuntimeError("10% and 20% cohorts are not the same 22 cases")
    new_by_budget = defaultdict(int)
    for row in jobs:
        new_by_budget[row["budget"]] += 1
    if dict(new_by_budget) != EXPECTED_NEW_PREDICTIONS:
        raise RuntimeError(f"unexpected new prediction counts: {dict(new_by_budget)}")

    protocol = {
        "schema_version": "delepref_24_binder_budget_contact_comparison_protocol_v1",
        "analysis_id": "DelePref-24",
        "cohort": {
            "source": "identical 22-case single-binder subset from the frozen 50-case Binder manifest",
            "case_count": EXPECTED_CASES,
            "case_ids": [int(value) for value in case_sets["0.10"]],
            "budgets": [0.10, 0.20],
        },
        "methods": {
            "delepref": "frozen canonical scorer top-1; no WT-interface preselection",
            "scisor": "frozen explicit-index SCISOR plans from the validated budget packages",
        },
        "prediction_reuse": {
            "rule": "reuse only when canonical deletion_mask_sha256 exactly equals the previously predicted interface-protected plan",
            "new_predictions_by_budget": dict(new_by_budget),
            "total_new_predictions": len(jobs),
        },
        "boltz": {
            "model": "boltz2",
            "sampling_steps": 50,
            "recycling_steps": 1,
            "diffusion_samples": 1,
            "empty_msa_for_edited_binder": True,
            "same_config_for_delepref_and_scisor": True,
            "zero_fill": False,
        },
        "metrics": {
            "contact_recovery": "|C_WT intersect C_edited| / |C_WT|",
            "contact_jaccard": "|C_WT intersect C_edited| / |C_WT union C_edited|",
            "contact_cutoff_angstrom": 5.0,
        },
        "statistics": {
            "unit": "Binder case",
            "paired_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ci": "percentile 95%",
        },
        "source_hashes": source_hashes,
        "claim_boundary": CLAIM_BOUNDARY,
        "proxy_boundary": PROXY_BOUNDARY,
    }
    write_json(OUT / "protocol.json", protocol)
    write_text(OUT / "protocol.sha256", sha256_file(OUT / "protocol.json") + "\n", encoding="ascii")
    write_csv(OUT / "canonical_delepref_plans.csv", plans, list(plans[0]))
    write_csv(OUT / "boltz_jobs.csv", jobs, list(jobs[0]))
    print(json.dumps({"status": "PASS_PREPARE", "plans": len(plans), "new_predictions": len(jobs)}, sort_keys=True))


def locate_confidence(record_id: str) -> Path:
    matches = sorted((OUT / "boltz_outputs").rglob(f"confidence_{record_id}_model_*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one confidence file for {record_id}, observed {len(matches)}")
    return matches[0]


def bootstrap_ci(values: Sequence[float], key: str) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    key_seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(BOOTSTRAP_SEED + key_seed)
    indices = rng.integers(0, vector.size, size=(BOOTSTRAP_REPLICATES, vector.size))
    means = vector[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(vector.mean()), float(low), float(high)


def analyze() -> None:
    runtime = load_runtime()
    v12 = runtime.load_v12_module()
    plans = read_csv(OUT / "canonical_delepref_plans.csv")
    output: list[dict[str, Any]] = []

    for plan in plans:
        case_id = plan["case_id"]
        wt_spec = runtime.make_spec(
            v12,
            case_id,
            plan["protein_id"],
            "wildtype_shared",
            Path(plan["wt_yaml_path"]),
            Path(plan["wt_confidence_json"]),
        )
        wt = v12.load_prediction(wt_spec, canonical_target_sequences=None)
        targets = [entry["sequence"] for entry in wt.entities if entry["role"] == "target"]
        confidence_path = (
            Path(plan["confidence_json"])
            if plan["prediction_source"] == "REUSED_IDENTICAL_CANONICAL_PLAN"
            else locate_confidence(plan["record_id"])
        )
        edited_spec = runtime.make_spec(
            v12,
            case_id,
            plan["protein_id"],
            "delepref",
            Path(plan["yaml_path"]),
            confidence_path,
        )
        edited = v12.load_prediction(edited_spec, canonical_target_sequences=targets)
        if wt.geometry is None or edited.geometry is None:
            raise RuntimeError(f"missing geometry for {plan['protein_id']} at {plan['budget']}")
        mapping, deleted = v12.delepref_residue_mapping(
            plan["wildtype_sequence"],
            plan["edited_sequence"],
            runtime.parse_intervals(plan["deleted_intervals"]),
            int(plan["deleted_length"]),
        )
        remapped = v12.remap_geometry(edited.geometry, mapping)
        intersection = wt.geometry.contacts & remapped.contacts
        union = wt.geometry.contacts | remapped.contacts
        output.append(
            {
                "case_id": case_id,
                "protein_id": plan["protein_id"],
                "budget": plan["budget"],
                "method": "delepref",
                "contact_recovery": len(intersection) / len(wt.geometry.contacts),
                "contact_jaccard": len(intersection) / len(union) if union else 0.0,
                "wt_contact_count": len(wt.geometry.contacts),
                "edited_contact_count": len(remapped.contacts),
                "deleted_length": plan["deleted_length"],
                "deleted_intervals": plan["deleted_intervals"],
                "prediction_source": plan["prediction_source"],
                "record_id": plan["record_id"],
                "confidence_json": str(confidence_path),
                "structure_cif": str(runtime.confidence_to_structure(confidence_path)),
            }
        )

    for budget, source in source_by_budget().items():
        for row in read_csv(source / "case_metrics.csv"):
            if row["method"] != "scisor":
                continue
            output.append(
                {
                    "case_id": row["case_id"],
                    "protein_id": row["protein_id"],
                    "budget": budget,
                    "method": "scisor",
                    "contact_recovery": row["contact_recovery"],
                    "contact_jaccard": row["contact_jaccard"],
                    "wt_contact_count": row["wt_contact_count"],
                    "edited_contact_count": row["edited_contact_count"],
                    "deleted_length": row["deleted_length"],
                    "deleted_intervals": row["deleted_intervals"],
                    "prediction_source": "VALIDATED_SOURCE_PACKAGE",
                    "record_id": row["record_id"],
                    "confidence_json": row["confidence_json"],
                    "structure_cif": row["structure_cif"],
                }
            )
    output.sort(key=lambda row: (float(row["budget"]), int(row["case_id"]), row["method"]))
    write_csv(OUT / "binder_budget_case_metrics.csv", output, list(output[0]))

    summary_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for method in METHODS:
            rows = [row for row in output if row["budget"] == budget and row["method"] == method]
            for metric in METRICS:
                values = [float(row[metric]) for row in rows]
                summary_rows.append(
                    {
                        "budget": budget,
                        "method": method,
                        "metric": metric,
                        "direction": "higher_is_better",
                        "n": len(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "sample_sd": statistics.stdev(values),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                )
                for threshold in (0.50, 0.60, 0.80):
                    count = sum(value >= threshold for value in values)
                    threshold_rows.append(
                        {
                            "budget": budget,
                            "method": method,
                            "metric": metric,
                            "threshold": threshold,
                            "count": count,
                            "n": len(values),
                            "fraction": count / len(values),
                        }
                    )
    write_csv(OUT / "binder_budget_method_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(OUT / "binder_budget_threshold_summary.csv", threshold_rows, list(threshold_rows[0]))

    by_key = {
        (row["budget"], row["method"], row["case_id"]): row
        for row in output
    }
    case_ids = sorted({row["case_id"] for row in output}, key=int)
    ci_rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for metric in METRICS:
            differences = [
                float(by_key[(budget, "delepref", case_id)][metric])
                - float(by_key[(budget, "scisor", case_id)][metric])
                for case_id in case_ids
            ]
            key = f"delepref_minus_scisor|{budget}|{metric}"
            point, low, high = bootstrap_ci(differences, key)
            ci_rows.append(
                {
                    "contrast": "delepref_minus_scisor",
                    "budget": budget,
                    "method": "",
                    "metric": metric,
                    "n_paired_cases": len(case_ids),
                    "point_estimate": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
    for method in METHODS:
        for metric in METRICS:
            differences = [
                float(by_key[("0.20", method, case_id)][metric])
                - float(by_key[("0.10", method, case_id)][metric])
                for case_id in case_ids
            ]
            key = f"b20_minus_b10|{method}|{metric}"
            point, low, high = bootstrap_ci(differences, key)
            ci_rows.append(
                {
                    "contrast": "budget_0.20_minus_0.10",
                    "budget": "",
                    "method": method,
                    "metric": metric,
                    "n_paired_cases": len(case_ids),
                    "point_estimate": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
    for metric in METRICS:
        differences = [
            (
                float(by_key[("0.20", "delepref", case_id)][metric])
                - float(by_key[("0.20", "scisor", case_id)][metric])
            )
            - (
                float(by_key[("0.10", "delepref", case_id)][metric])
                - float(by_key[("0.10", "scisor", case_id)][metric])
            )
            for case_id in case_ids
        ]
        key = f"interaction|{metric}"
        point, low, high = bootstrap_ci(differences, key)
        ci_rows.append(
            {
                "contrast": "interaction_(delepref-scisor)_b20_minus_b10",
                "budget": "",
                "method": "",
                "metric": metric,
                "n_paired_cases": len(case_ids),
                "point_estimate": point,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    write_csv(OUT / "binder_budget_paired_bootstrap_ci.csv", ci_rows, list(ci_rows[0]))
    print(json.dumps({"status": "PASS_ANALYZE", "case_metric_rows": len(output), "ci_rows": len(ci_rows)}, sort_keys=True))


def validate() -> None:
    protocol = load_json(OUT / "protocol.json")
    checks: dict[str, bool] = {}
    checks["protocol_sha256"] = sha256_file(OUT / "protocol.json") == (OUT / "protocol.sha256").read_text(encoding="ascii").strip()
    for budget, source in source_by_budget().items():
        for name, expected in protocol["source_hashes"][budget].items():
            checks[f"source_{budget}_{name}"] = sha256_file(source / name) == expected

    plans = read_csv(OUT / "canonical_delepref_plans.csv")
    jobs = read_csv(OUT / "boltz_jobs.csv")
    rows = read_csv(OUT / "binder_budget_case_metrics.csv")
    summaries = read_csv(OUT / "binder_budget_method_summary.csv")
    cis = read_csv(OUT / "binder_budget_paired_bootstrap_ci.csv")
    checks["canonical_plan_count"] = len(plans) == EXPECTED_CASES * len(BUDGETS)
    checks["new_prediction_count"] = len(jobs) == sum(EXPECTED_NEW_PREDICTIONS.values())
    checks["case_metric_count"] = len(rows) == EXPECTED_CASES * len(BUDGETS) * len(METHODS)
    checks["summary_count"] = len(summaries) == len(BUDGETS) * len(METHODS) * len(METRICS)
    checks["ci_count"] = len(cis) == 10

    expected_cases = [str(value) for value in protocol["cohort"]["case_ids"]]
    assignments_ok = True
    finite_ok = True
    for budget in BUDGETS:
        for method in METHODS:
            subset = [row for row in rows if row["budget"] == budget and row["method"] == method]
            assignments_ok &= sorted((row["case_id"] for row in subset), key=int) == expected_cases
            for row in subset:
                for metric in METRICS:
                    value = float(row[metric])
                    finite_ok &= math.isfinite(value) and 0.0 <= value <= 1.0
    checks["complete_paired_assignments"] = bool(assignments_ok)
    checks["finite_metric_ranges"] = bool(finite_ok)

    prediction_ok = True
    for job in jobs:
        confidence = locate_confidence(job["record_id"])
        structure = load_runtime().confidence_to_structure(confidence)
        prediction_ok &= confidence.is_file() and structure.is_file()
    checks["new_prediction_sidecars"] = bool(prediction_ok)

    recomputed_summary = {}
    for budget in BUDGETS:
        for method in METHODS:
            subset = [row for row in rows if row["budget"] == budget and row["method"] == method]
            for metric in METRICS:
                recomputed_summary[(budget, method, metric)] = statistics.mean(float(row[metric]) for row in subset)
    summary_ok = all(
        math.isclose(float(row["mean"]), recomputed_summary[(row["budget"], row["method"], row["metric"])], rel_tol=0.0, abs_tol=1e-15)
        for row in summaries
    )
    checks["summary_means_recomputed"] = summary_ok

    failed = sorted(name for name, passed in checks.items() if not passed)
    validation = {
        "schema_version": "delepref_24_binder_budget_contact_comparison_validation_v1",
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": failed,
        "validated_cases": EXPECTED_CASES,
        "validated_method_budget_rows": len(rows),
    }
    write_json(OUT / "independent_validation.json", validation)
    if failed:
        raise RuntimeError(f"validation failed: {failed}")

    summary_rows = read_csv(OUT / "binder_budget_method_summary.csv")
    ci_rows = read_csv(OUT / "binder_budget_paired_bootstrap_ci.csv")
    summary_by = {(row["budget"], row["method"], row["metric"]): row for row in summary_rows}
    ci_by = {(row["contrast"], row["budget"], row["metric"]): row for row in ci_rows}
    lines = [
        "# DelePref-24 Binder budget contact comparison",
        "",
        f"Status: `{STATUS}`",
        "",
        "The same 22 single-binder cases are evaluated at exact 10% and 20% deletion budgets. DelePref is the frozen canonical scorer top-1 without WT-interface preselection; SCISOR uses explicit deleted indices. Both methods use the same Boltz-2 configuration.",
        "",
        "| Budget | Method | Contact recovery | Contact Jaccard |",
        "|---:|---|---:|---:|",
    ]
    for budget in BUDGETS:
        for method in METHODS:
            recovery = float(summary_by[(budget, method, "contact_recovery")]["mean"])
            jaccard = float(summary_by[(budget, method, "contact_jaccard")]["mean"])
            lines.append(f"| {budget} | {method} | {recovery:.6f} | {jaccard:.6f} |")
    lines.extend(["", "Paired DelePref minus SCISOR effects (20,000 case-level bootstrap replicates):", ""])
    for budget in BUDGETS:
        for metric in METRICS:
            row = ci_by[("delepref_minus_scisor", budget, metric)]
            lines.append(
                f"- {budget} {metric}: {float(row['point_estimate']):+.6f} "
                f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}]"
            )
    lines.extend(["", PROXY_BOUNDARY, "", CLAIM_BOUNDARY, ""])
    write_text(OUT / "SUMMARY.md", "\n".join(lines))
    run_summary = {
        "schema_version": "delepref_24_binder_budget_contact_comparison_summary_v1",
        "status": STATUS,
        "case_count": EXPECTED_CASES,
        "budgets": [0.10, 0.20],
        "methods": list(METHODS),
        "new_predictions": len(jobs),
        "reused_delepref_predictions": len(plans) - len(jobs),
        "case_metric_rows": len(rows),
        "protocol_sha256": sha256_file(OUT / "protocol.json"),
        "independent_validation": "PASS",
        "claim_boundary": CLAIM_BOUNDARY,
        "proxy_boundary": PROXY_BOUNDARY,
    }
    write_json(OUT / "run_summary.json", run_summary)

    excluded = {"audit_manifest.json", STATUS}
    artifacts = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file()):
        relative = path.relative_to(OUT).as_posix()
        if relative in excluded or relative.startswith("audit/"):
            continue
        artifacts.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(OUT / "audit_manifest.json", {"schema_version": "delepref_24_artifact_manifest_v1", "artifacts": artifacts})
    write_text(OUT / STATUS, STATUS + "\n", encoding="ascii")
    print(json.dumps({"status": STATUS, "validation": "PASS", "artifacts": len(artifacts)}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "analyze", "validate"))
    args = parser.parse_args()
    {"prepare": prepare, "analyze": analyze, "validate": validate}[args.phase]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
