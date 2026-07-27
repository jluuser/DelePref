#!/usr/bin/env python3
"""Index Cas9 MISER raw assets without processing FASTQ reads."""

from __future__ import annotations

import argparse
import csv
import json
import posixpath
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


COLUMNS = [
    "asset_type",
    "path",
    "size_bytes",
    "status",
    "description",
    "sheet_name",
    "columns_json",
    "row_count",
]

MAIN_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_size(path: Path) -> str:
    return str(path.stat().st_size) if path.exists() and path.is_file() else "0"


def col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return max(index - 1, 0)


def shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return ["".join(item.itertext()) for item in root.findall("m:si", MAIN_NS)]


def workbook_sheet_paths(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map: dict[str, str] = {}
    for rel in rels_root.findall("r:Relationship", REL_NS):
        target = rel.attrib.get("Target", "")
        if target.startswith("/"):
            target_path = target.lstrip("/")
        else:
            target_path = posixpath.normpath(posixpath.join("xl", target))
        rel_map[rel.attrib.get("Id", "")] = target_path

    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.findall(".//m:sheet", MAIN_NS):
        rid = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id", "")
        path = rel_map.get(rid)
        if path:
            sheets.append((sheet.attrib.get("name", ""), path))
    return sheets


def cell_value(cell: ET.Element, shared: list[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        inline = cell.find("m:is", MAIN_NS)
        return "".join(inline.itertext()).strip() if inline is not None else ""
    value = cell.find("m:v", MAIN_NS)
    raw = value.text if value is not None else ""
    if cell_type == "s" and raw not in ("", None):
        try:
            return shared[int(raw)].strip()
        except (ValueError, IndexError):
            return str(raw).strip()
    return str(raw).strip()


def sheet_header_and_rows(zf: zipfile.ZipFile, sheet_path: str, shared: list[str]) -> tuple[list[str], int]:
    root = ET.fromstring(zf.read(sheet_path))
    non_empty_rows: list[list[str]] = []
    for row in root.findall(".//m:sheetData/m:row", MAIN_NS):
        values: list[str] = []
        for cell in row.findall("m:c", MAIN_NS):
            idx = col_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            values[idx] = cell_value(cell, shared)
        if any(value != "" for value in values):
            non_empty_rows.append(values)
    if not non_empty_rows:
        return [], 0
    header = non_empty_rows[0]
    columns = [value if value else f"column_{idx + 1}" for idx, value in enumerate(header)]
    return columns, max(len(non_empty_rows) - 1, 0)


def xlsx_rows(path: Path, repo_root: Path, description: str) -> list[dict[str, str]]:
    try:
        with zipfile.ZipFile(path) as zf:
            shared = shared_strings(zf)
            sheets = workbook_sheet_paths(zf)
            rows: list[dict[str, str]] = []
            for sheet_name, sheet_path in sheets:
                columns, row_count = sheet_header_and_rows(zf, sheet_path, shared)
                rows.append(
                    {
                        "asset_type": "supplementary_xlsx",
                        "path": display_path(path, repo_root),
                        "size_bytes": file_size(path),
                        "status": "present",
                        "description": description,
                        "sheet_name": sheet_name,
                        "columns_json": json_text(columns),
                        "row_count": str(row_count),
                    }
                )
            return rows
    except (OSError, zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise SystemExit(f"failed to read XLSX metadata from {path}: {exc}") from exc


def delimited_asset_row(asset_type: str, path: Path, repo_root: Path, description: str, delimiter: str = "\t") -> dict[str, str]:
    columns: list[str] = []
    row_count = 0
    if path.exists():
        try:
            with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                columns = next(reader, [])
                row_count = sum(1 for _ in reader)
        except OSError as exc:
            raise SystemExit(f"failed to read delimited asset {path}: {exc}") from exc
    return {
        "asset_type": asset_type,
        "path": display_path(path, repo_root),
        "size_bytes": file_size(path),
        "status": "present" if path.exists() else "missing",
        "description": description,
        "sheet_name": "",
        "columns_json": json_text(columns),
        "row_count": str(row_count) if path.exists() else "",
    }


def simple_asset_row(asset_type: str, path: Path, repo_root: Path, description: str) -> dict[str, str]:
    return {
        "asset_type": asset_type,
        "path": display_path(path, repo_root),
        "size_bytes": file_size(path),
        "status": "present" if path.exists() else "missing",
        "description": description,
        "sheet_name": "",
        "columns_json": json_text([]),
        "row_count": "",
    }


def build_rows(raw_dir: Path, structure_dir: Path, repo_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for fastq in sorted((raw_dir / "raw_fastq").glob("*.fastq.gz")):
        rows.append(simple_asset_row("raw_fastq", fastq, repo_root, "Indexed only; FASTQ reads are not processed or mapped."))

    rows.append(delimited_asset_row("run_table_tsv", raw_dir / "PRJNA746606_runs.tsv", repo_root, "SRA/PRJNA run table metadata."))

    for pdf in sorted(raw_dir.glob("*.pdf")):
        rows.append(simple_asset_row("supplementary_pdf", pdf, repo_root, "Cas9 MISER supplementary PDF."))

    for workbook in sorted(raw_dir.glob("*.xlsx")):
        rows.extend(xlsx_rows(workbook, repo_root, "Cas9 MISER supplementary workbook metadata; sheet names, columns, and row counts only."))

    for pdb_id in ["4UN3", "5Y36"]:
        for suffix in [".cif", ".pdb"]:
            rows.append(simple_asset_row("structure", structure_dir / f"{pdb_id}{suffix}", repo_root, f"Cas9 structure asset {pdb_id}{suffix}."))

    rows.append(
        delimited_asset_row(
            "domain_tsv",
            structure_dir / "uniprot_Q99ZW2_domains.tsv",
            repo_root,
            "UniProt Q99ZW2 domain annotation table.",
            delimiter="\t",
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index Cas9 MISER raw assets without FASTQ read processing.")
    parser.add_argument("--repo_root", default=str(repo_root_from_script()), help="Repository root.")
    parser.add_argument("--raw_dir", default="data/raw/cas9_miser", help="Cas9 MISER raw directory.")
    parser.add_argument("--structure_dir", default="data/raw/cas9_structures", help="Cas9 structure/domain raw directory.")
    parser.add_argument(
        "--out_csv",
        default="data/processed/experiments/cas9_asset_index.csv",
        help="Output Cas9 asset index CSV.",
    )
    parser.add_argument("--max_records", type=int, default=None, help="Optional cap on output rows for smoke checks.")
    parser.add_argument("--smoke", action="store_true", help="Alias for --max_records 10 when no explicit cap is supplied.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    rows = build_rows(resolve_path(repo_root, args.raw_dir), resolve_path(repo_root, args.structure_dir), repo_root)
    max_records = args.max_records
    if args.smoke and max_records is None:
        max_records = 10
    if max_records is not None:
        rows = rows[:max_records]
    write_csv(resolve_path(repo_root, args.out_csv), rows)
    print(f"wrote {len(rows)} Cas9 asset rows")


if __name__ == "__main__":
    main()
