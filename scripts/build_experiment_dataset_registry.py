#!/usr/bin/env python3
"""Build a lightweight registry of raw experiment data assets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = [
    "resource_id",
    "group",
    "path",
    "size_bytes",
    "file_count",
    "status",
    "role",
    "notes",
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def tree_stats(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1

    total_size = 0
    total_files = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total_size += child.stat().st_size
                total_files += 1
            except OSError:
                continue
    return total_size, total_files


def resource_specs(repo_root: Path) -> list[dict[str, str | Path]]:
    parent_raw = repo_root.parent / "data" / "raw"
    return [
        {
            "resource_id": "proteingym",
            "group": "protein_effect",
            "path": repo_root / "data/raw/proteingym",
            "role": "ProteinGym deletion/indel experimental labels",
            "notes": "Raw metadata CSVs and DMS ZIP bundles; labels are external experiments, not teacher labels.",
        },
        {
            "resource_id": "cas9_miser",
            "group": "cas9",
            "path": repo_root / "data/raw/cas9_miser",
            "role": "Cas9 MISER supplementary assets and raw FASTQs",
            "notes": "FASTQs are indexed only; no read processing or mapping is performed.",
        },
        {
            "resource_id": "cas9_structures",
            "group": "cas9",
            "path": repo_root / "data/raw/cas9_structures",
            "role": "Cas9 structures and UniProt domain table",
            "notes": "Structure/domain assets for later Cas9 analysis.",
        },
        {
            "resource_id": "ab_bind",
            "group": "binder",
            "path": repo_root / "data/raw/ab_bind",
            "role": "AB-Bind mutation ddG table and structures",
            "notes": "Partner-chain order follows source metadata when available.",
        },
        {
            "resource_id": "sabdab",
            "group": "binder",
            "path": repo_root / "data/raw/sabdab",
            "role": "SAbDab antibody-antigen complex metadata and structures",
            "notes": "Parsed for complex registry; no per-mutation scores are inferred.",
        },
        {
            "resource_id": "ppb_affinity",
            "group": "binder",
            "path": repo_root / "data/raw/ppb_affinity",
            "role": "PPB-Affinity complex affinity workbook and structures",
            "notes": "Parsed for complex registry; mutation rows are row-level affinities when source mutations are present.",
        },
        {
            "resource_id": "mavedb",
            "group": "binder",
            "path": repo_root / "data/raw/mavedb",
            "role": "Selected MaveDB score sets",
            "notes": "Score-set rows are source-defined assay scores without structural chain assignment.",
        },
        {
            "resource_id": "bindinggym",
            "group": "binder",
            "path": repo_root / "data/raw/bindinggym",
            "role": "BindingGYM input archive",
            "notes": "Current preparation layer records this source but does not fabricate processed binder labels.",
        },
        {
            "resource_id": "skempi_v2",
            "group": "binder",
            "path": parent_raw / "skempi_v2",
            "role": "SKEMPI v2 mutation/affinity benchmark",
            "notes": "External raw path; parsed with source chain groups and affinity ratios only.",
        },
        {
            "resource_id": "cath_s40",
            "group": "proposal_registry",
            "path": parent_raw / "cath_s40",
            "role": "External structural-domain registry input",
            "notes": "Registered for later proposal benchmarks; not parsed in this preparation layer.",
        },
        {
            "resource_id": "disprot",
            "group": "proposal_registry",
            "path": parent_raw / "disprot",
            "role": "External disorder annotation registry input",
            "notes": "Registered for later proposal benchmarks; not parsed in this preparation layer.",
        },
    ]


def build_rows(repo_root: Path, max_records: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    specs = resource_specs(repo_root)
    if max_records is not None:
        specs = specs[:max_records]

    for spec in specs:
        path = Path(spec["path"])
        size_bytes, file_count = tree_stats(path)
        rows.append(
            {
                "resource_id": str(spec["resource_id"]),
                "group": str(spec["group"]),
                "path": display_path(path, repo_root),
                "size_bytes": str(size_bytes),
                "file_count": str(file_count),
                "status": "present" if path.exists() else "missing",
                "role": str(spec["role"]),
                "notes": str(spec["notes"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    present = sum(1 for row in rows if row["status"] == "present")
    missing = len(rows) - present
    lines = [
        "# Experiment Processing Summary",
        "",
        "This summary is initialized by the dataset registry builder. The validation script rewrites it after all processed tables are generated.",
        "",
        "## Dataset Registry",
        "",
        f"- resources: {len(rows)}",
        f"- present: {present}",
        f"- missing: {missing}",
        "",
        "## Scope Notes",
        "",
        "- This preparation layer indexes and normalizes raw experimental data tables only.",
        "- It does not run training, GPU jobs, srun, torchrun, downloads, teacher scoring, or Cas9 FASTQ mapping.",
        "- ProteinGym labels are external experimental labels, not teacher labels.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/processed/experiments/dataset_registry.csv from local raw experiment assets."
    )
    parser.add_argument("--repo_root", default=str(repo_root_from_script()), help="Repository root. Defaults to this script's repo.")
    parser.add_argument(
        "--out_csv",
        default="data/processed/experiments/dataset_registry.csv",
        help="Output dataset registry CSV.",
    )
    parser.add_argument(
        "--summary_md",
        default="data/processed/experiments/PROCESSING_SUMMARY.md",
        help="Initial processing summary Markdown path.",
    )
    parser.add_argument("--max_records", type=int, default=None, help="Optional cap on registry rows for smoke runs.")
    parser.add_argument("--smoke", action="store_true", help="Alias for --max_records 3 when no explicit cap is supplied.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    max_records = args.max_records
    if args.smoke and max_records is None:
        max_records = 3

    rows = build_rows(repo_root, max_records=max_records)
    write_csv(repo_root / args.out_csv if not Path(args.out_csv).is_absolute() else Path(args.out_csv), rows)
    write_summary(repo_root / args.summary_md if not Path(args.summary_md).is_absolute() else Path(args.summary_md), rows)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
