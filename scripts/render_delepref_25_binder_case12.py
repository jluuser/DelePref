#!/usr/bin/env python3
"""Render the DelePref-25 Binder case 12 from the fresh common-protocol outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import render_delepref_22_binder_case12 as base  # noqa: E402


V25 = ROOT / "results/experiments/delepref_step004000_delepref_25_binder50_fair_contact_comparison"
OUT = V25 / "case_study_case012"
CASE_ID = "12"
PROTEIN_ID = "AB-Bind:2NY7:G_HL"
BUDGET = 0.20
EXPECTED_PROTOCOL_SHA256 = "de91968990041db9f5e65459bea1d6984ea0b9b71836230bcf1d6ad664622a08"
EXPECTED_VALUES = {
    "ours": {"contact_recovery": 0.9074074074074074, "contact_jaccard": 0.8166666666666667},
    "scisor": {"contact_recovery": 0.09259259259259259, "contact_jaccard": 0.04504504504504504},
}

base.V22 = V25
base.OUT = OUT


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_intervals(value: str) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for start, end in json.loads(value)]


def mutant(sequence: str, intervals: list[tuple[int, int]]) -> str:
    deleted = {position for start, end in intervals for position in range(start, end + 1)}
    return "".join(residue for index, residue in enumerate(sequence) if index not in deleted)


def load_case() -> dict[str, Any]:
    summary = read_json(V25 / "run_summary.json")
    validation = read_json(V25 / "independent_validation.json")
    protocol_sha = (V25 / "protocol.sha256").read_text(encoding="ascii").strip()
    if summary.get("status") != "PASS_DelePref_25_BINDER50_FAIR_CONTACT_COMPARISON":
        raise RuntimeError("DelePref-25 package status is not PASS")
    if validation.get("status") != "PASS" or validation.get("failed"):
        raise RuntimeError("DelePref-25 independent validation is not PASS")
    if protocol_sha != EXPECTED_PROTOCOL_SHA256 or sha256_file(V25 / "protocol.json") != protocol_sha:
        raise RuntimeError("DelePref-25 protocol SHA256 mismatch")

    metrics = [
        row
        for row in read_csv(V25 / "binder50_case_metrics.csv")
        if row["case_id"] == CASE_ID
        and row["protein_id"] == PROTEIN_ID
        and abs(float(row["budget"]) - BUDGET) < 1e-12
    ]
    if len(metrics) != 2 or {row["method"] for row in metrics} != {"delepref", "scisor"}:
        raise RuntimeError("expected paired DelePref-25 case 12 rows")
    metric_by_method = {row["method"]: row for row in metrics}
    wt_rows = [
        row
        for row in read_csv(V25 / "binder50_wt_reference_audit.csv")
        if row["case_id"] == CASE_ID and row["protein_id"] == PROTEIN_ID
    ]
    if len(wt_rows) != 1:
        raise RuntimeError("expected one DelePref-25 WT audit row")
    wt = wt_rows[0]

    cohort_rows = [
        row
        for row in read_csv(V25 / "cohort.csv")
        if row["case_id"] == CASE_ID and row["protein_id"] == PROTEIN_ID
    ]
    if len(cohort_rows) != 1:
        raise RuntimeError("expected one DelePref-25 cohort row")
    cohort = cohort_rows[0]
    sequence = cohort["wildtype_sequence"]
    if len(sequence) != base.WT_LENGTH:
        raise RuntimeError("unexpected case 12 binder length")

    records: list[dict[str, Any]] = [
        {
            "id": "full_length",
            "label": "Full-length",
            "source_method": "wt",
            "structure": Path(wt["structure_cif"]),
            "confidence_path": Path(wt["confidence_json"]),
            "confidence": float(wt["wt_confidence"]),
            "wt_contact_count": int(wt["wt_contact_count"]),
            "length": base.WT_LENGTH,
            "intervals": [],
        }
    ]
    for source_method, output_id, label in (
        ("delepref", "ours", "Ours"),
        ("scisor", "scisor", "SCISOR"),
    ):
        row = metric_by_method[source_method]
        parsed = parse_intervals(row["deleted_intervals"])
        deleted_length = sum(end - start + 1 for start, end in parsed)
        edited_sequence = mutant(sequence, parsed)
        if deleted_length != base.DELETED_LENGTH or len(edited_sequence) != base.EDITED_LENGTH:
            raise RuntimeError(f"invalid exact-budget plan for {source_method}")
        expected = EXPECTED_VALUES[output_id]
        if abs(float(row["contact_recovery"]) - expected["contact_recovery"]) > 1e-15:
            raise RuntimeError(f"contact recovery changed for {source_method}")
        if abs(float(row["contact_jaccard"]) - expected["contact_jaccard"]) > 1e-15:
            raise RuntimeError(f"contact Jaccard changed for {source_method}")
        records.append(
            {
                "id": output_id,
                "label": label,
                "source_method": source_method,
                "record_id": row["record_id"],
                "structure": Path(row["structure_cif"]),
                "confidence_path": Path(row["confidence_json"]),
                "confidence": float(row["confidence"]),
                "delta_confidence": float(row["delta_confidence"]),
                "contact_recovery": float(row["contact_recovery"]),
                "contact_jaccard": float(row["contact_jaccard"]),
                "interface_residue_recovery": float(row["interface_residue_recovery"]),
                "iptm": float(row["interface_confidence"]),
                "pae": float(row["interchain_pae"]),
                "clashes": int(row["steric_clash_count"]),
                "length": len(edited_sequence),
                "intervals": base.one_based(parsed),
                "span_count": int(row["span_count"]),
            }
        )
    for record in records:
        for field in ("structure", "confidence_path"):
            path = record[field]
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
    return {"records": records, "metrics": metric_by_method, "wt": wt, "cohort": cohort}


def write_manifest(case: dict[str, Any], script: Path, figures: list[Path]) -> None:
    source_paths = [
        V25 / "protocol.json",
        V25 / "protocol.sha256",
        V25 / "run_summary.json",
        V25 / "independent_validation.json",
        V25 / "binder50_case_metrics.csv",
        V25 / "binder50_wt_reference_audit.csv",
        V25 / "cohort.csv",
    ]
    for record in case["records"]:
        source_paths.extend([record["structure"], record["confidence_path"]])
        if record["id"] == "full_length":
            source_paths.append(V25 / "boltz_inputs/case012_00_wt.yaml")
        else:
            source_paths.append(V25 / "boltz_inputs" / f"{record['record_id']}.yaml")
    source_rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_paths
    ]
    generated_paths = [
        script,
        OUT / "pymol_alignment_and_render_audit.json",
        OUT / "pymol_render.log",
        *figures,
    ]
    generated_rows = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in generated_paths
    ]
    png = next(path for path in figures if path.suffix == ".png")
    image_audit = base.image_audit(png)
    if not image_audit["nonblank"] or image_audit["width"] < 4000 or image_audit["height"] < 1700:
        raise RuntimeError(f"figure image audit failed: {image_audit}")
    render_audit = read_json(OUT / "pymol_alignment_and_render_audit.json")
    if render_audit.get("ours_deleted_interface_ca") != 0:
        raise RuntimeError("render found an Ours deletion/interface overlap")

    caption = (
        "Wild-type complex (A) and the post-deletion complexes obtained using the plans returned by "
        "DelePref (B) and SCISOR (C) for AB-Bind:2NY7:G_HL at the 20% deletion budget. Both methods "
        "delete exactly 58 of 290 binder residues, and the complete complexes are aligned on the unchanged "
        "target. The bottom tracks show the deleted positions and the predicted wild-type interface. All "
        "displayed structures are generated using the same frozen DelePref-25 Boltz-2 prediction protocol. These "
        "structures are complex-prediction proxies rather than experimental binding evidence."
    )
    base.write_text(OUT / "paper_caption.txt", caption + "\n")
    manifest = {
        "schema_version": "delepref_25_binder_case12_figure_v1",
        "status": "PASS_DelePref_25_BINDER_CASE12_FIGURE",
        "case_id": 12,
        "protein_id": PROTEIN_ID,
        "budget": BUDGET,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "metrics": EXPECTED_VALUES,
        "image_audit": image_audit,
        "source_artifacts": source_rows,
        "generated_artifacts": generated_rows,
        "caption": caption,
        "proxy_boundary": base.PROXY_BOUNDARY,
        "claim_boundary": base.CLAIM_BOUNDARY,
    }
    base.write_json(OUT / "figure_manifest.json", manifest)
    base.write_text(OUT / "PASS_DelePref_25_BINDER_CASE12_FIGURE", manifest["status"] + "\n")
    print(json.dumps({"status": manifest["status"], "figure": str(png), "image": image_audit}, sort_keys=True))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    case = load_case()
    script = base.render_pymol(case)
    figures = base.make_figure(case)
    write_manifest(case, script, figures)


if __name__ == "__main__":
    main()
