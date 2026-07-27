#!/usr/bin/env python3
"""Validate experiment preparation tables and write the single summary file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_SCHEMAS = {
    "dataset_registry.csv": [
        "resource_id",
        "group",
        "path",
        "size_bytes",
        "file_count",
        "status",
        "role",
        "notes",
    ],
    "proteingym_indel_variants.csv": [
        "assay_id",
        "dms_id",
        "source_file",
        "variant_id",
        "wildtype_sequence",
        "mutant_sequence",
        "edit_type",
        "deleted_intervals",
        "inserted_sequence",
        "deletion_length",
        "deletion_fraction",
        "dms_score",
        "dms_score_bin",
        "target_length",
        "uniprot_id",
        "metadata_json",
    ],
    "proteingym_assay_registry.csv": [
        "assay_id",
        "dms_id",
        "uniprot_id",
        "target_seq",
        "seq_len",
        "assay_type_guess",
        "source_file",
        "total_variants",
        "total_single_segment_deletions",
        "notes",
    ],
    "binder_complex_registry.csv": [
        "source",
        "complex_id",
        "pdb_id",
        "structure_path",
        "binder_chains",
        "target_chains",
        "antigen_name",
        "affinity_value",
        "affinity_unit",
        "delta_g",
        "resolution",
        "metadata_json",
    ],
    "binder_mutation_scores.csv": [
        "source",
        "complex_id",
        "pdb_id",
        "chain_id",
        "mutation",
        "residue_index_raw",
        "wildtype_aa",
        "mutant_aa",
        "ddg",
        "score",
        "score_direction",
        "assay_info",
        "metadata_json",
    ],
    "cas9_asset_index.csv": [
        "asset_type",
        "path",
        "size_bytes",
        "status",
        "description",
        "sheet_name",
        "columns_json",
        "row_count",
    ],
}


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
    except OSError as exc:
        raise SystemExit(f"failed to read processed table {path}: {exc}") from exc


def validate_json_cell(value: str, file_name: str, row_num: int, column: str, errors: list[str]) -> None:
    try:
        json.loads(value if value != "" else "{}")
    except json.JSONDecodeError as exc:
        errors.append(f"{file_name} row {row_num}: invalid JSON in {column}: {exc}")


def validate_deleted_intervals(value: str, file_name: str, row_num: int, errors: list[str]) -> None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        errors.append(f"{file_name} row {row_num}: invalid deleted_intervals JSON: {exc}")
        return
    if not isinstance(parsed, list):
        errors.append(f"{file_name} row {row_num}: deleted_intervals must be a JSON list")
        return
    for interval in parsed:
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(isinstance(item, int) for item in interval)
            or interval[0] > interval[1]
        ):
            errors.append(f"{file_name} row {row_num}: invalid deleted interval {interval!r}")


def validate_table(processed_dir: Path, file_name: str, errors: list[str]) -> dict[str, object]:
    path = processed_dir / file_name
    schema = REQUIRED_SCHEMAS[file_name]
    if not path.exists():
        errors.append(f"missing required table: {path}")
        return {"rows": 0, "columns_ok": False, "sources": []}

    columns, rows = read_rows(path)
    columns_ok = columns == schema
    if not columns_ok:
        errors.append(f"{file_name}: columns differ from required schema. observed={columns!r} expected={schema!r}")

    source_values = sorted({row.get("source", "") for row in rows if row.get("source", "")})
    asset_types = sorted({row.get("asset_type", "") for row in rows if row.get("asset_type", "")})
    edit_types = sorted({row.get("edit_type", "") for row in rows if row.get("edit_type", "")})

    for row_num, row in enumerate(rows, start=2):
        if "metadata_json" in schema:
            validate_json_cell(row.get("metadata_json", ""), file_name, row_num, "metadata_json", errors)
        if file_name == "proteingym_indel_variants.csv":
            validate_deleted_intervals(row.get("deleted_intervals", ""), file_name, row_num, errors)
            if row.get("edit_type") == "single_segment_deletion":
                intervals = json.loads(row.get("deleted_intervals", "[]"))
                if not intervals:
                    errors.append(f"{file_name} row {row_num}: single_segment_deletion lacks deleted_intervals")
        if file_name == "cas9_asset_index.csv":
            validate_json_cell(row.get("columns_json", ""), file_name, row_num, "columns_json", errors)

    return {
        "rows": len(rows),
        "columns_ok": columns_ok,
        "sources": source_values,
        "asset_types": asset_types,
        "edit_types": edit_types,
    }


def write_summary(summary_path: Path, stats: dict[str, dict[str, object]], errors: list[str]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment Processing Summary",
        "",
        "## Validation Status",
        "",
        f"- status: {'PASS' if not errors else 'FAIL'}",
        f"- tables_checked: {len(REQUIRED_SCHEMAS)}",
        "",
        "## Processed Tables",
        "",
        "| table | rows | required columns |",
        "| --- | ---: | --- |",
    ]
    for file_name in REQUIRED_SCHEMAS:
        table_stats = stats.get(file_name, {})
        lines.append(
            f"| {file_name} | {table_stats.get('rows', 0)} | {'yes' if table_stats.get('columns_ok') else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Parsed Sources",
            "",
            f"- binder_complex_registry sources: {', '.join(stats.get('binder_complex_registry.csv', {}).get('sources', [])) or 'none'}",
            f"- binder_mutation_scores sources: {', '.join(stats.get('binder_mutation_scores.csv', {}).get('sources', [])) or 'none'}",
            f"- cas9 asset types: {', '.join(stats.get('cas9_asset_index.csv', {}).get('asset_types', [])) or 'none'}",
            f"- ProteinGym edit types: {', '.join(stats.get('proteingym_indel_variants.csv', {}).get('edit_types', [])) or 'none'}",
            "",
            "## Scope And Limitations",
            "",
            "- ProteinGym DMS labels are external experimental labels, not teacher labels.",
            "- Cas9 FASTQs are indexed by path and size only; no read parsing, mapping, or activity landscape is produced.",
            "- BindingGYM is registered in dataset_registry.csv only for this preparation layer; no score rows are fabricated from input.zip.",
            "- SAbDab contributes complex metadata only; per-mutation binding scores are not inferred.",
            "- PPB-Affinity mutation rows are emitted only when the source workbook already contains mutation text; row-level KD is recorded without inferring ddG.",
            "- AB-Bind and SKEMPI chain groups use source-provided chain notation; ambiguous semantic binder/target roles are documented in metadata_json rather than guessed.",
            "- MaveDB score rows do not provide PDB or chain fields in this local export, so those fields are left blank.",
            "- No training, GPU job, srun, torchrun, download, teacher scoring, or Cas9 FASTQ downstream mapping is part of this run.",
            "",
        ]
    )

    if errors:
        lines.extend(["## Validation Errors", ""])
        lines.extend(f"- {error}" for error in errors[:200])
        if len(errors) > 200:
            lines.append(f"- ... {len(errors) - 200} additional errors omitted")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate required experiment processed tables and rewrite PROCESSING_SUMMARY.md."
    )
    parser.add_argument("--processed_dir", default="data/processed/experiments", help="Directory containing processed tables.")
    parser.add_argument(
        "--summary_md",
        default=None,
        help="Summary Markdown path. Defaults to PROCESSING_SUMMARY.md inside --processed_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    summary_path = Path(args.summary_md) if args.summary_md else processed_dir / "PROCESSING_SUMMARY.md"
    errors: list[str] = []
    stats: dict[str, dict[str, object]] = {}

    for file_name in REQUIRED_SCHEMAS:
        stats[file_name] = validate_table(processed_dir, file_name, errors)

    write_summary(summary_path, stats, errors)

    if errors:
        for error in errors[:20]:
            print(f"ERROR: {error}")
        if len(errors) > 20:
            print(f"ERROR: {len(errors) - 20} additional validation errors")
        raise SystemExit(1)

    for file_name in REQUIRED_SCHEMAS:
        print(f"{file_name}: rows={stats[file_name]['rows']} columns_ok=yes")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
