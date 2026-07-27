#!/usr/bin/env python3
"""Derive the complete Binder-50 metric table from frozen DelePref-25 outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import py_compile
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/experiments/delepref_step004000_delepref_25_binder50_fair_contact_comparison"
OUT = ROOT / "results/experiments/delepref_step004000_delepref_26_binder50_complete_metrics"
SCRIPT = ROOT / "scripts/run_delepref_26_binder50_complete_metrics.py"
SOURCE_STATUS = "PASS_DelePref_25_BINDER50_FAIR_CONTACT_COMPARISON"
STATUS = "PASS_DelePref_26_BINDER50_COMPLETE_METRICS"
EXPECTED_CASES = 50
BUDGETS = ("0.10", "0.20")
METHODS = ("delepref", "scisor")
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20260726
STRICT_CONFIDENCE = 0.70
STRICT_MAX_DELTA = 0.05
RELAXED_CONFIDENCE = 0.50
RELAXED_MAX_DELTA = 0.10
CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion "
    "perturbation preference and optional auxiliary risk evidence."
)
PROXY_BOUNDARY = (
    "All Binder results are post-deletion Boltz-2 complex-prediction proxies; "
    "they do not establish experimental binding, Kd/Ki, or functional preservation."
)

METRICS = {
    "wt_confidence": "descriptive_baseline",
    "edited_confidence": "higher_is_better",
    "delta_confidence": "lower_is_better",
    "relative_confidence_retention": "higher_is_better",
    "strict_retention": "higher_is_better",
    "relaxed_retention": "higher_is_better",
    "interface_confidence": "higher_is_better",
    "contact_recovery": "higher_is_better",
    "contact_jaccard": "higher_is_better",
    "interface_residue_recovery": "higher_is_better",
    "interchain_pae": "lower_is_better",
    "deleted_wt_interface_overlap_fraction": "lower_is_better",
    "steric_clash_count": "lower_is_better",
    "contact_count_retention": "descriptive",
    "span_count": "descriptive",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, value: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def as_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return number


def bootstrap_ci(values: Sequence[float], key: str) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    key_seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(BOOTSTRAP_SEED + key_seed)
    indices = rng.integers(0, vector.size, size=(BOOTSTRAP_REPLICATES, vector.size))
    distribution = vector[indices].mean(axis=1)
    low, high = np.quantile(distribution, [0.025, 0.975])
    return float(vector.mean()), float(low), float(high)


def source_manifest_hash(path: Path) -> str:
    manifest = load_json(SOURCE / "audit_manifest.json")
    matches = [row for row in manifest["artifacts"] if row["path"] == path.name]
    if len(matches) != 1:
        raise RuntimeError(f"source manifest entry count for {path.name}: {len(matches)}")
    return str(matches[0]["sha256"])


def build_protocol() -> dict[str, Any]:
    source_metrics = SOURCE / "binder50_case_metrics.csv"
    source_protocol = SOURCE / "protocol.json"
    source_summary = SOURCE / "run_summary.json"
    protocol = {
        "schema_version": "delepref_26_binder50_complete_metrics_protocol_v1",
        "analysis_id": "DelePref-26",
        "analysis_type": "CPU-only deterministic derivation from frozen DelePref-25 predictions",
        "source": {
            "package": str(SOURCE),
            "required_status": SOURCE_STATUS,
            "case_metrics": str(source_metrics),
            "case_metrics_sha256": sha256_file(source_metrics),
            "case_metrics_manifest_sha256": source_manifest_hash(source_metrics),
            "protocol": str(source_protocol),
            "protocol_sha256": sha256_file(source_protocol),
            "run_summary": str(source_summary),
            "run_summary_sha256": sha256_file(source_summary),
        },
        "cohort": {
            "cases": EXPECTED_CASES,
            "budgets": [0.10, 0.20],
            "methods": list(METHODS),
            "expected_edited_rows": EXPECTED_CASES * len(BUDGETS) * len(METHODS),
        },
        "retention_definitions": {
            "delta_confidence": "WT confidence - edited confidence; lower is better",
            "relative_confidence_retention": "edited confidence / WT confidence; higher is better",
            "strict_retention": (
                f"edited confidence >= {STRICT_CONFIDENCE:.2f} and delta confidence <= {STRICT_MAX_DELTA:.2f}"
            ),
            "relaxed_retention": (
                f"edited confidence >= {RELAXED_CONFIDENCE:.2f} and delta confidence <= {RELAXED_MAX_DELTA:.2f}"
            ),
        },
        "metrics": METRICS,
        "statistics": {
            "unit": "Binder case",
            "contrast": "DelePref minus SCISOR",
            "paired_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ci": "percentile 95%",
            "budgets_analyzed_separately": True,
        },
        "comparison_scope": "native end-to-end; DelePref K=8 plus frozen scorer versus one native SCISOR plan; not equal-K/equal-compute",
        "no_new_scientific_compute": {
            "gpu": False,
            "boltz": False,
            "candidate_generation": False,
            "model_or_checkpoint_load": False,
        },
        "script": str(SCRIPT),
        "script_sha256": sha256_file(SCRIPT),
        "claim_boundary": CLAIM_BOUNDARY,
        "proxy_boundary": PROXY_BOUNDARY,
    }
    write_json(OUT / "protocol.json", protocol)
    write_text(OUT / "protocol.sha256", sha256_file(OUT / "protocol.json") + "\n", encoding="ascii")
    return protocol


def derive_rows(source_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in source_rows:
        wt = as_float(row["wt_confidence"])
        edited = as_float(row["confidence"])
        delta = wt - edited
        if abs(delta - as_float(row["delta_confidence"])) > 1e-12:
            raise RuntimeError(f"source delta mismatch: case {row['case_id']} {row['budget']} {row['method']}")
        wt_contacts = int(row["wt_contact_count"])
        edited_contacts = int(row["edited_contact_count"])
        if wt <= 0 or wt_contacts <= 0:
            raise RuntimeError(f"invalid WT denominator: case {row['case_id']}")
        derived = dict(row)
        derived["edited_confidence"] = edited
        derived["relative_confidence_retention"] = edited / wt
        derived["strict_retention"] = int(edited >= STRICT_CONFIDENCE and delta <= STRICT_MAX_DELTA)
        derived["relaxed_retention"] = int(edited >= RELAXED_CONFIDENCE and delta <= RELAXED_MAX_DELTA)
        derived["contact_count_retention"] = edited_contacts / wt_contacts
        output.append(derived)
    output.sort(key=lambda row: (float(row["budget"]), int(row["case_id"]), str(row["method"])))
    return output


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for method in METHODS:
            subset = [row for row in rows if f"{float(row['budget']):.2f}" == budget and row["method"] == method]
            if len(subset) != EXPECTED_CASES:
                raise RuntimeError(f"incomplete subset: {budget} {method} has {len(subset)} rows")
            for metric, direction in METRICS.items():
                values = [as_float(row[metric]) for row in subset]
                output.append(
                    {
                        "budget": budget,
                        "method": method,
                        "metric": metric,
                        "direction": direction,
                        "n_expected": EXPECTED_CASES,
                        "n_available": len(values),
                        "coverage": len(values) / EXPECTED_CASES,
                        "mean": statistics.fmean(values),
                        "median": statistics.median(values),
                        "sample_sd": statistics.stdev(values),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                )
    return output


def paired_cis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keyed = {
        (f"{float(row['budget']):.2f}", str(row["method"]), str(row["case_id"])): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for metric, direction in METRICS.items():
            differences = [
                as_float(keyed[(budget, "delepref", str(case_id))][metric])
                - as_float(keyed[(budget, "scisor", str(case_id))][metric])
                for case_id in range(1, EXPECTED_CASES + 1)
            ]
            point, low, high = bootstrap_ci(differences, f"delepref_minus_scisor|{budget}|{metric}")
            sign = 1.0 if direction == "higher_is_better" else (-1.0 if direction == "lower_is_better" else None)
            output.append(
                {
                    "contrast": "delepref_minus_scisor",
                    "budget": budget,
                    "metric": metric,
                    "direction": direction,
                    "n_pairs_expected": EXPECTED_CASES,
                    "n_pairs_available": EXPECTED_CASES,
                    "point_estimate": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "oriented_effect_positive_favors_delepref": "" if sign is None else sign * point,
                    "oriented_ci95_low": "" if sign is None else min(sign * low, sign * high),
                    "oriented_ci95_high": "" if sign is None else max(sign * low, sign * high),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "bootstrap_unit": "Binder case",
                }
            )
    return output


def paper_rows(summaries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["budget"]), str(row["method"]), str(row["metric"])): row["mean"]
        for row in summaries
    }
    output = []
    for budget in BUDGETS:
        for method in METHODS:
            row: dict[str, Any] = {
                "budget": budget,
                "method": "DelePref" if method == "delepref" else "SCISOR",
                "n": EXPECTED_CASES,
            }
            for metric in METRICS:
                row[metric] = lookup[(budget, method, metric)]
            output.append(row)
    return output


def compare_float(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    return abs(as_float(left) - as_float(right)) <= tolerance


def validate(
    protocol: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, str]],
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    cis: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    source_summary = load_json(SOURCE / "run_summary.json")
    checks["source_status"] = source_summary.get("status") == SOURCE_STATUS
    checks["source_hash"] = sha256_file(SOURCE / "binder50_case_metrics.csv") == protocol["source"]["case_metrics_sha256"]
    checks["source_manifest_hash"] = protocol["source"]["case_metrics_sha256"] == protocol["source"]["case_metrics_manifest_sha256"]
    checks["source_row_count"] = len(source_rows) == EXPECTED_CASES * len(BUDGETS) * len(METHODS)
    checks["derived_row_count"] = len(rows) == len(source_rows)
    checks["unique_keys"] = len({(row["case_id"], row["budget"], row["method"]) for row in rows}) == len(rows)
    checks["complete_groups"] = all(
        sum(f"{float(row['budget']):.2f}" == budget and row["method"] == method for row in rows) == EXPECTED_CASES
        for budget in BUDGETS
        for method in METHODS
    )
    checks["retention_formulas"] = all(
        compare_float(row["relative_confidence_retention"], as_float(row["confidence"]) / as_float(row["wt_confidence"]))
        and int(row["strict_retention"])
        == int(as_float(row["confidence"]) >= STRICT_CONFIDENCE and as_float(row["delta_confidence"]) <= STRICT_MAX_DELTA)
        and int(row["relaxed_retention"])
        == int(as_float(row["confidence"]) >= RELAXED_CONFIDENCE and as_float(row["delta_confidence"]) <= RELAXED_MAX_DELTA)
        for row in rows
    )
    checks["summary_row_count"] = len(summaries) == len(BUDGETS) * len(METHODS) * len(METRICS)
    checks["summary_recomputed"] = all(
        compare_float(
            row["mean"],
            statistics.fmean(
                as_float(item[row["metric"]])
                for item in rows
                if f"{float(item['budget']):.2f}" == row["budget"] and item["method"] == row["method"]
            ),
        )
        for row in summaries
    )
    checks["bootstrap_row_count"] = len(cis) == len(BUDGETS) * len(METRICS)
    ci_lookup = {(row["budget"], row["metric"]): row for row in cis}
    independent_ci_checks = []
    keyed = {
        (f"{float(row['budget']):.2f}", str(row["method"]), int(row["case_id"])): row
        for row in rows
    }
    for budget in BUDGETS:
        for metric in METRICS:
            differences = [
                as_float(keyed[(budget, "delepref", case_id)][metric])
                - as_float(keyed[(budget, "scisor", case_id)][metric])
                for case_id in range(1, EXPECTED_CASES + 1)
            ]
            point, low, high = bootstrap_ci(differences, f"delepref_minus_scisor|{budget}|{metric}")
            observed = ci_lookup[(budget, metric)]
            independent_ci_checks.append(
                compare_float(point, observed["point_estimate"])
                and compare_float(low, observed["ci95_low"])
                and compare_float(high, observed["ci95_high"])
            )
    checks["bootstrap_recomputed"] = all(independent_ci_checks)
    old_summary = read_csv(SOURCE / "binder50_method_summary.csv")
    old_cis = read_csv(SOURCE / "binder50_paired_bootstrap_ci.csv")
    summary_lookup = {(row["budget"], row["method"], row["metric"]): row for row in summaries}
    checks["delepref_25_means_preserved"] = all(
        compare_float(row["mean"], summary_lookup[(row["budget"], row["method"], row["metric"])]["mean"])
        for row in old_summary
    )
    new_ci_lookup = {(row["budget"], row["metric"]): row for row in cis}
    checks["delepref_25_cis_preserved"] = all(
        compare_float(row["point_estimate"], new_ci_lookup[(row["budget"], row["metric"])]["point_estimate"])
        and compare_float(row["ci95_low"], new_ci_lookup[(row["budget"], row["metric"])]["ci95_low"])
        and compare_float(row["ci95_high"], new_ci_lookup[(row["budget"], row["metric"])]["ci95_high"])
        for row in old_cis
    )
    checks["all_numeric_finite"] = all(
        math.isfinite(as_float(row[metric])) for row in rows for metric in METRICS
    )
    checks["retention_binary"] = all(
        int(row[metric]) in (0, 1) for row in rows for metric in ("strict_retention", "relaxed_retention")
    )
    checks["protocol_sha256"] = sha256_file(OUT / "protocol.json") == (OUT / "protocol.sha256").read_text(encoding="ascii").strip()
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "delepref_26_binder50_complete_metrics_validation_v1",
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": failed,
        "validated_source_rows": len(source_rows),
        "validated_derived_rows": len(rows),
        "validated_summary_rows": len(summaries),
        "validated_bootstrap_rows": len(cis),
    }


def fmt(value: Any, digits: int = 4) -> str:
    return f"{as_float(value):.{digits}f}"


def build_summary(paper: Sequence[Mapping[str, Any]], cis: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# DelePref-26 Binder-50 Complete Metrics",
        "",
        f"Status: `{STATUS}`",
        "",
        "CPU-only derivation from the frozen DelePref-25 250-prediction Boltz-2 run. No structure prediction, model inference, candidate generation, or score selection was rerun.",
        "",
        "Retention definitions:",
        "",
        f"- Strict: edited confidence >= {STRICT_CONFIDENCE:.2f} and WT-minus-edited confidence <= {STRICT_MAX_DELTA:.2f}.",
        f"- Relaxed: edited confidence >= {RELAXED_CONFIDENCE:.2f} and WT-minus-edited confidence <= {RELAXED_MAX_DELTA:.2f}.",
        "- Relative confidence retention: edited confidence divided by WT confidence.",
        "",
        "## Paper-facing complete table",
        "",
        "| Budget | Method | Delta confidence down | Strict up | Relaxed up | Interface confidence up | Contact recovery up | Contact Jaccard up | Interface-residue recovery up | Inter-chain PAE down |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paper:
        lines.append(
            f"| {row['budget']} | {row['method']} | {fmt(row['delta_confidence'])} | "
            f"{fmt(row['strict_retention'], 2)} | {fmt(row['relaxed_retention'], 2)} | "
            f"{fmt(row['interface_confidence'])} | {fmt(row['contact_recovery'])} | "
            f"{fmt(row['contact_jaccard'])} | {fmt(row['interface_residue_recovery'])} | "
            f"{fmt(row['interchain_pae'])} |"
        )
    lines.extend(
        [
            "",
            "## Additional diagnostics",
            "",
            "| Budget | Method | Edited confidence up | Relative confidence retention up | Deleted-interface overlap down | Clashes down | Contact-count retention descriptive | Mean spans descriptive |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paper:
        lines.append(
            f"| {row['budget']} | {row['method']} | {fmt(row['edited_confidence'])} | "
            f"{fmt(row['relative_confidence_retention'])} | {fmt(row['deleted_wt_interface_overlap_fraction'])} | "
            f"{fmt(row['steric_clash_count'], 2)} | {fmt(row['contact_count_retention'])} | "
            f"{fmt(row['span_count'], 2)} |"
        )
    lines.extend(
        [
            "",
            "## Paired retention effects",
            "",
            "DelePref minus SCISOR; 20,000 paired case-bootstrap replicates:",
            "",
            "| Budget | Metric | Difference | 95% CI |",
            "|---:|---|---:|---:|",
        ]
    )
    for budget in BUDGETS:
        for metric in ("delta_confidence", "strict_retention", "relaxed_retention"):
            row = next(item for item in cis if item["budget"] == budget and item["metric"] == metric)
            lines.append(
                f"| {budget} | {metric} | {fmt(row['point_estimate'])} | "
                f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] |"
            )
    lines.extend(
        [
            "",
            f"Comparison scope: native end-to-end, not equal-K/equal-compute. {PROXY_BOUNDARY}",
            "",
        ]
    )
    return "\n".join(lines)


def artifact_manifest() -> dict[str, Any]:
    artifacts = []
    for path in sorted(candidate for candidate in OUT.rglob("*") if candidate.is_file() and candidate.name != "audit_manifest.json"):
        artifacts.append(
            {
                "path": str(path.relative_to(OUT)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": "delepref_26_binder50_complete_metrics_manifest_v1",
        "artifacts": artifacts,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUT}")
    OUT.mkdir(parents=True)
    protocol = build_protocol()
    source_rows = read_csv(SOURCE / "binder50_case_metrics.csv")
    rows = derive_rows(source_rows)
    summaries = summarize(rows)
    cis = paired_cis(rows)
    paper = paper_rows(summaries)

    write_csv(OUT / "binder50_complete_case_metrics.csv", rows, list(rows[0]))
    write_csv(OUT / "binder50_complete_method_summary.csv", summaries, list(summaries[0]))
    write_csv(OUT / "binder50_complete_paired_bootstrap_ci.csv", cis, list(cis[0]))
    write_csv(OUT / "binder50_complete_paper_table.csv", paper, list(paper[0]))

    validation = validate(protocol, source_rows, rows, summaries, cis)
    write_json(OUT / "independent_validation.json", validation)
    if validation["status"] != "PASS":
        raise RuntimeError(f"validation failed: {validation['failed']}")

    write_text(OUT / "SUMMARY.md", build_summary(paper, cis))
    run_summary = {
        "schema_version": "delepref_26_binder50_complete_metrics_summary_v1",
        "status": STATUS,
        "source_status": SOURCE_STATUS,
        "source_protocol_sha256": protocol["source"]["protocol_sha256"],
        "protocol_sha256": sha256_file(OUT / "protocol.json"),
        "case_count": EXPECTED_CASES,
        "budgets": [0.10, 0.20],
        "methods": list(METHODS),
        "derived_case_rows": len(rows),
        "summary_rows": len(summaries),
        "paired_bootstrap_rows": len(cis),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "cpu_only": True,
        "host": platform.node(),
        "python": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "comparison_scope": protocol["comparison_scope"],
        "claim_boundary": CLAIM_BOUNDARY,
        "proxy_boundary": PROXY_BOUNDARY,
    }
    write_json(OUT / "run_summary.json", run_summary)
    write_text(OUT / STATUS, STATUS + "\n", encoding="ascii")

    py_compile.compile(str(SCRIPT), doraise=True)
    diff_check = subprocess.run(
        ["git", "diff", "--check"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    if diff_check.returncode != 0:
        raise RuntimeError(f"git diff --check failed: {diff_check.stdout}{diff_check.stderr}")
    write_json(OUT / "audit_manifest.json", artifact_manifest())
    print(json.dumps(run_summary, sort_keys=True))


if __name__ == "__main__":
    main()
