#!/usr/bin/env python3
"""Build lightweight binder complex and mutation-score preparation tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


COMPLEX_COLUMNS = [
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
]

MUTATION_COLUMNS = [
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
]

MAIN_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {
    "r": "http://schemas.openxmlformats.org/package/2006/relationships",
    "office": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

GENERIC_MUT_RE = re.compile(r"^(?:(?P<chain>[A-Za-z0-9]+):)?(?P<wt>[A-Z])(?P<idx>-?\d+[A-Za-z]?)(?P<mut>[A-Z])$")
SKEMPI_MUT_RE = re.compile(r"^(?P<wt>[A-Z])(?P<chain>[A-Za-z0-9])(?P<idx>-?\d+[A-Za-z]?)(?P<mut>[A-Z])$")
HGVS_PRO_RE = re.compile(r"([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})")

AA3_TO_1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Ter": "*",
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "na", "n/a", "none", "nan", "null"}:
        return ""
    return text


def read_delimited(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    except OSError as exc:
        raise SystemExit(f"failed to read table {path}: {exc}") from exc


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
    values = []
    for si in root.findall("m:si", MAIN_NS):
        values.append("".join(si.itertext()))
    return values


def workbook_sheet_paths(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    rel_map: dict[str, str] = {}
    rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    for rel in rels_root.findall("r:Relationship", REL_NS):
        target = rel.attrib.get("Target", "")
        if target.startswith("/"):
            target_path = target.lstrip("/")
        else:
            target_path = posixpath.normpath(posixpath.join("xl", target))
        rel_map[rel.attrib.get("Id", "")] = target_path

    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    sheets = []
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
        return clean("".join(inline.itertext()) if inline is not None else "")
    value = cell.find("m:v", MAIN_NS)
    raw = value.text if value is not None else ""
    if cell_type == "s" and raw not in ("", None):
        try:
            return clean(shared[int(raw)])
        except (ValueError, IndexError):
            return clean(raw)
    return clean(raw)


def read_xlsx_records(path: Path, sheet_index: int = 0) -> tuple[str, list[dict[str, str]]]:
    try:
        with zipfile.ZipFile(path) as zf:
            shared = shared_strings(zf)
            sheets = workbook_sheet_paths(zf)
            if not sheets:
                return "", []
            sheet_name, sheet_path = sheets[sheet_index]
            root = ET.fromstring(zf.read(sheet_path))
    except (OSError, zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise SystemExit(f"failed to read XLSX workbook {path}: {exc}") from exc

    rows: list[list[str]] = []
    for row in root.findall(".//m:sheetData/m:row", MAIN_NS):
        values: list[str] = []
        for cell in row.findall("m:c", MAIN_NS):
            idx = col_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")
            values[idx] = cell_value(cell, shared)
        if any(values):
            rows.append(values)

    if not rows:
        return sheet_name, []

    header = rows[0]
    columns: list[str] = []
    seen: dict[str, int] = {}
    for idx, name in enumerate(header):
        column = clean(name) or f"column_{idx + 1}"
        seen[column] = seen.get(column, 0) + 1
        if seen[column] > 1:
            column = f"{column}_{seen[column]}"
        columns.append(column)

    records: list[dict[str, str]] = []
    for values in rows[1:]:
        if len(values) < len(columns):
            values = values + [""] * (len(columns) - len(values))
        records.append({column: values[idx] if idx < len(values) else "" for idx, column in enumerate(columns)})
    return sheet_name, records


def find_structure(structure_dir: Path, pdb_id: str, repo_root: Path) -> str:
    if not pdb_id:
        return ""
    candidates = []
    for stem in {pdb_id, pdb_id.lower(), pdb_id.upper()}:
        for suffix in [".cif", ".cif.gz", ".pdb", ".pdb.gz"]:
            candidates.append(structure_dir / f"{stem}{suffix}")
    for candidate in candidates:
        if candidate.exists():
            return display_path(candidate, repo_root)
    return ""


def split_partners(text: str) -> tuple[str, str]:
    parts = [clean(part) for part in clean(text).split("_")]
    parts = [part for part in parts if part]
    if len(parts) >= 2:
        return parts[0], ",".join(parts[1:])
    if len(parts) == 1:
        return parts[0], ""
    return "", ""


def parse_skempi_complex_field(text: str) -> tuple[str, list[str]]:
    parts = [part for part in clean(text).split("_") if part]
    if not parts:
        return "", []
    return parts[0].upper(), parts[1:]


def parse_source_mutation(token: str, style: str = "generic") -> dict[str, str]:
    token = clean(token)
    if not token:
        return {"mutation": "", "chain_id": "", "residue_index_raw": "", "wildtype_aa": "", "mutant_aa": ""}
    match = GENERIC_MUT_RE.match(token)
    if not match and style == "skempi":
        match = SKEMPI_MUT_RE.match(token)
    if not match:
        match = SKEMPI_MUT_RE.match(token)
    if not match:
        return {"mutation": token, "chain_id": "", "residue_index_raw": "", "wildtype_aa": "", "mutant_aa": ""}
    groups = match.groupdict()
    return {
        "mutation": token,
        "chain_id": clean(groups.get("chain", "")),
        "residue_index_raw": clean(groups.get("idx", "")),
        "wildtype_aa": clean(groups.get("wt", "")),
        "mutant_aa": clean(groups.get("mut", "")),
    }


def mutation_tokens(text: str) -> list[str]:
    value = clean(text)
    if not value:
        return []
    return [token.strip() for token in re.split(r"[,;]", value) if token.strip()]


def parse_hgvs_pro(text: str) -> list[dict[str, str]]:
    value = clean(text)
    if not value:
        return []
    compact = value.replace("p.", "").replace("[", "").replace("]", "")
    parsed: list[dict[str, str]] = []
    for token in [item.strip() for item in compact.split(";") if item.strip()]:
        match = HGVS_PRO_RE.search(token)
        if not match:
            parsed.append(
                {"mutation": f"p.{token}", "chain_id": "", "residue_index_raw": "", "wildtype_aa": "", "mutant_aa": ""}
            )
            continue
        wt3, idx, mut3 = match.groups()
        parsed.append(
            {
                "mutation": f"p.{token}",
                "chain_id": "",
                "residue_index_raw": idx,
                "wildtype_aa": AA3_TO_1.get(wt3, ""),
                "mutant_aa": AA3_TO_1.get(mut3, ""),
            }
        )
    return parsed


def log10_ratio(numerator: str, denominator: str) -> str:
    try:
        top = float(clean(numerator))
        bottom = float(clean(denominator))
    except ValueError:
        return ""
    if top <= 0 or bottom <= 0:
        return ""
    return f"{math.log10(top / bottom):.8g}"


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def resolve_out(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def under_cap(counts: dict[str, int], source: str, max_records: int | None) -> bool:
    return max_records is None or counts.get(source, 0) < max_records


def bump(counts: dict[str, int], source: str) -> None:
    counts[source] = counts.get(source, 0) + 1


def add_ab_bind(
    rows_complex: list[dict[str, str]],
    rows_mutation: list[dict[str, str]],
    repo_root: Path,
    raw_dir: Path,
    max_records: int | None,
    complex_counts: dict[str, int],
    mutation_counts: dict[str, int],
) -> None:
    source = "AB-Bind"
    table = raw_dir / "AB-Bind_experimental_data.csv"
    records = read_delimited(table)
    seen_complexes: set[str] = set()
    structure_dir = raw_dir / "structures"
    for record in records:
        pdb_id = clean(record.get("#PDB", "")).upper()
        partners = clean(record.get("Partners(A_B)", ""))
        complex_id = f"{source}:{pdb_id}:{partners}" if partners else f"{source}:{pdb_id}"
        binder_chains, target_chains = split_partners(partners)
        if complex_id not in seen_complexes and under_cap(complex_counts, source, max_records):
            seen_complexes.add(complex_id)
            rows_complex.append(
                {
                    "source": source,
                    "complex_id": complex_id,
                    "pdb_id": pdb_id,
                    "structure_path": find_structure(structure_dir, pdb_id, repo_root),
                    "binder_chains": binder_chains,
                    "target_chains": target_chains,
                    "antigen_name": "",
                    "affinity_value": "",
                    "affinity_unit": "",
                    "delta_g": "",
                    "resolution": clean(record.get("PDB Res. (Angstroms)", "")),
                    "metadata_json": json_text(
                        {
                            "source_file": display_path(table, repo_root),
                            "source_partner_order": "Partners(A_B) mapped to binder_chains,target_chains; semantic binder/target role is source-order only.",
                            "protein_1": clean(record.get("Protein-1", "")),
                            "protein_2": clean(record.get("Protein-2", "")),
                            "assay_name": clean(record.get("Assay Name", "")),
                        }
                    ),
                }
            )
            bump(complex_counts, source)

        for token in mutation_tokens(record.get("Mutation", "")):
            if not under_cap(mutation_counts, source, max_records):
                break
            parsed = parse_source_mutation(token)
            rows_mutation.append(
                {
                    "source": source,
                    "complex_id": complex_id,
                    "pdb_id": pdb_id,
                    "chain_id": parsed["chain_id"],
                    "mutation": parsed["mutation"],
                    "residue_index_raw": parsed["residue_index_raw"],
                    "wildtype_aa": parsed["wildtype_aa"],
                    "mutant_aa": parsed["mutant_aa"],
                    "ddg": clean(record.get("ddG(kcal/mol)", "")),
                    "score": clean(record.get("ddG(kcal/mol)", "")),
                    "score_direction": "higher_ddg_weaker_binding",
                    "assay_info": clean(record.get("Assay Name", "")),
                    "metadata_json": json_text(
                        {
                            "source_file": display_path(table, repo_root),
                            "source_mutation_set": clean(record.get("Mutation", "")),
                            "partners": partners,
                            "assay_pH": clean(record.get("Assay pH", "")),
                            "assay_temp_celsius": clean(record.get("Assay Temp (Celcius)", "")),
                        }
                    ),
                }
            )
            bump(mutation_counts, source)


def add_sabdab(
    rows_complex: list[dict[str, str]],
    repo_root: Path,
    raw_dir: Path,
    max_records: int | None,
    complex_counts: dict[str, int],
) -> None:
    source = "SAbDab"
    table = raw_dir / "sabdab_summary_all.tsv"
    records = read_delimited(table, delimiter="\t")
    structure_dir = raw_dir / "structures"
    seen: set[str] = set()
    for record in records:
        if not under_cap(complex_counts, source, max_records):
            break
        pdb_id = clean(record.get("pdb", "")).upper()
        hchain = clean(record.get("Hchain", ""))
        lchain = clean(record.get("Lchain", ""))
        antigen_chain = clean(record.get("antigen_chain", ""))
        complex_id = f"{source}:{pdb_id}:{hchain}{lchain}:{antigen_chain}"
        if complex_id in seen:
            continue
        seen.add(complex_id)
        binder = ",".join(part for part in [hchain, lchain] if part)
        rows_complex.append(
            {
                "source": source,
                "complex_id": complex_id,
                "pdb_id": pdb_id,
                "structure_path": find_structure(structure_dir, pdb_id, repo_root),
                "binder_chains": binder,
                "target_chains": antigen_chain,
                "antigen_name": clean(record.get("antigen_name", "")),
                "affinity_value": clean(record.get("affinity", "")),
                "affinity_unit": "",
                "delta_g": clean(record.get("delta_g", "")),
                "resolution": clean(record.get("resolution", "")),
                "metadata_json": json_text(
                    {
                        "source_file": display_path(table, repo_root),
                        "affinity_method": clean(record.get("affinity_method", "")),
                        "method": clean(record.get("method", "")),
                        "antigen_type": clean(record.get("antigen_type", "")),
                        "pmid": clean(record.get("pmid", "")),
                    }
                ),
            }
        )
        bump(complex_counts, source)


def add_ppb_affinity(
    rows_complex: list[dict[str, str]],
    rows_mutation: list[dict[str, str]],
    repo_root: Path,
    raw_dir: Path,
    max_records: int | None,
    complex_counts: dict[str, int],
    mutation_counts: dict[str, int],
) -> None:
    source = "PPB-Affinity"
    workbook = raw_dir / "PPB-Affinity.xlsx"
    sheet_name, records = read_xlsx_records(workbook)
    structure_dir = raw_dir / "structures"
    seen: set[str] = set()
    for record in records:
        pdb_id = clean(record.get("PDB", "")).upper()
        complex_id = clean(record.get("Complex ID", "")) or f"{source}:{pdb_id}"
        if complex_id not in seen and under_cap(complex_counts, source, max_records):
            seen.add(complex_id)
            rows_complex.append(
                {
                    "source": source,
                    "complex_id": complex_id,
                    "pdb_id": pdb_id,
                    "structure_path": find_structure(structure_dir, pdb_id, repo_root),
                    "binder_chains": clean(record.get("Ligand Chains", "")),
                    "target_chains": clean(record.get("Receptor Chains", "")),
                    "antigen_name": clean(record.get("Receptor Name", "")),
                    "affinity_value": clean(record.get("KD(M)", "")),
                    "affinity_unit": "M" if clean(record.get("KD(M)", "")) else "",
                    "delta_g": "",
                    "resolution": clean(record.get("Resolution", "")),
                    "metadata_json": json_text(
                        {
                            "source_file": display_path(workbook, repo_root),
                            "sheet_name": sheet_name,
                            "source_data_set": clean(record.get("Source Data Set", "")),
                            "ligand_name": clean(record.get("Ligand Name", "")),
                            "affinity_method": clean(record.get("Affinity Method", "")),
                            "structure_method": clean(record.get("Structure Method", "")),
                        }
                    ),
                }
            )
            bump(complex_counts, source)

        mutations = mutation_tokens(record.get("Mutations", ""))
        if not mutations:
            continue
        for token in mutations:
            if not under_cap(mutation_counts, source, max_records):
                break
            parsed = parse_source_mutation(token)
            rows_mutation.append(
                {
                    "source": source,
                    "complex_id": complex_id,
                    "pdb_id": pdb_id,
                    "chain_id": parsed["chain_id"],
                    "mutation": parsed["mutation"],
                    "residue_index_raw": parsed["residue_index_raw"],
                    "wildtype_aa": parsed["wildtype_aa"],
                    "mutant_aa": parsed["mutant_aa"],
                    "ddg": "",
                    "score": clean(record.get("KD(M)", "")),
                    "score_direction": "higher_kd_weaker_binding",
                    "assay_info": clean(record.get("Affinity Method", "")),
                    "metadata_json": json_text(
                        {
                            "source_file": display_path(workbook, repo_root),
                            "sheet_name": sheet_name,
                            "source_mutation_set": clean(record.get("Mutations", "")),
                            "note": "PPB row-level affinity is recorded for mutation-bearing rows; no ddG is inferred.",
                        }
                    ),
                }
            )
            bump(mutation_counts, source)


def add_skempi(
    rows_complex: list[dict[str, str]],
    rows_mutation: list[dict[str, str]],
    repo_root: Path,
    csv_path: Path,
    structure_dir: Path,
    max_records: int | None,
    complex_counts: dict[str, int],
    mutation_counts: dict[str, int],
) -> None:
    source = "SKEMPI_v2"
    records = read_delimited(csv_path, delimiter=";")
    seen: set[str] = set()
    for record in records:
        pdb_id, chain_groups = parse_skempi_complex_field(record.get("#Pdb", ""))
        complex_id = f"{source}:{clean(record.get('#Pdb', ''))}"
        binder_chains = chain_groups[0] if chain_groups else ""
        target_chains = ",".join(chain_groups[1:]) if len(chain_groups) > 1 else ""
        if complex_id not in seen and under_cap(complex_counts, source, max_records):
            seen.add(complex_id)
            rows_complex.append(
                {
                    "source": source,
                    "complex_id": complex_id,
                    "pdb_id": pdb_id,
                    "structure_path": find_structure(structure_dir, pdb_id, repo_root),
                    "binder_chains": binder_chains,
                    "target_chains": target_chains,
                    "antigen_name": clean(record.get("Protein 2", "")),
                    "affinity_value": clean(record.get("Affinity_wt_parsed", "")),
                    "affinity_unit": "M" if clean(record.get("Affinity_wt_parsed", "")) else "",
                    "delta_g": "",
                    "resolution": "",
                    "metadata_json": json_text(
                        {
                            "source_file": display_path(csv_path, repo_root),
                            "source_chain_groups": chain_groups,
                            "chain_role_note": "SKEMPI #Pdb chain groups are mapped in source order; semantic binder/target roles are not guaranteed.",
                            "protein_1": clean(record.get("Protein 1", "")),
                            "protein_2": clean(record.get("Protein 2", "")),
                            "method": clean(record.get("Method", "")),
                        }
                    ),
                }
            )
            bump(complex_counts, source)

        for token in mutation_tokens(record.get("Mutation(s)_cleaned", "")):
            if not under_cap(mutation_counts, source, max_records):
                break
            parsed = parse_source_mutation(token, style="skempi")
            rows_mutation.append(
                {
                    "source": source,
                    "complex_id": complex_id,
                    "pdb_id": pdb_id,
                    "chain_id": parsed["chain_id"],
                    "mutation": parsed["mutation"],
                    "residue_index_raw": parsed["residue_index_raw"],
                    "wildtype_aa": parsed["wildtype_aa"],
                    "mutant_aa": parsed["mutant_aa"],
                    "ddg": "",
                    "score": log10_ratio(record.get("Affinity_mut_parsed", ""), record.get("Affinity_wt_parsed", "")),
                    "score_direction": "higher_log10_kd_ratio_weaker_binding",
                    "assay_info": clean(record.get("Method", "")),
                    "metadata_json": json_text(
                        {
                            "source_file": display_path(csv_path, repo_root),
                            "mutation_pdb": clean(record.get("Mutation(s)_PDB", "")),
                            "affinity_mut_m": clean(record.get("Affinity_mut_parsed", "")),
                            "affinity_wt_m": clean(record.get("Affinity_wt_parsed", "")),
                            "reference": clean(record.get("Reference", "")),
                            "note": "score is log10(Affinity_mut_parsed/Affinity_wt_parsed), using source parsed affinity values.",
                        }
                    ),
                }
            )
            bump(mutation_counts, source)


def load_mavedb_metadata(raw_dir: Path, repo_root: Path) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for path in sorted(raw_dir.glob("*metadata.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in payload.get("scoreSets", []):
            urn = clean(item.get("urn", ""))
            if urn:
                mapping[urn] = {
                    "title": clean(item.get("title", "")),
                    "shortDescription": clean(item.get("shortDescription", "")),
                    "publishedDate": clean(item.get("publishedDate", "")),
                    "numVariants": clean(item.get("numVariants", "")),
                    "metadata_file": display_path(path, repo_root),
                }
    return mapping


def add_mavedb(
    rows_mutation: list[dict[str, str]],
    repo_root: Path,
    raw_dir: Path,
    max_records: int | None,
    mutation_counts: dict[str, int],
) -> None:
    source = "MaveDB"
    metadata = load_mavedb_metadata(raw_dir, repo_root)
    for score_file in sorted((raw_dir / "scores").glob("*.csv")):
        if not under_cap(mutation_counts, source, max_records):
            break
        records = read_delimited(score_file)
        for record in records:
            if not under_cap(mutation_counts, source, max_records):
                break
            score = clean(record.get("score", ""))
            if not score:
                continue
            accession = clean(record.get("accession", ""))
            urn = accession.split("#", 1)[0] if accession else score_file.stem.replace("_scores", "").replace("_", ":")
            parsed_mutations = parse_hgvs_pro(record.get("hgvs_pro", ""))
            if not parsed_mutations:
                parsed_mutations = [
                    {
                        "mutation": clean(record.get("hgvs_pro", "")) or clean(record.get("hgvs_nt", "")),
                        "chain_id": "",
                        "residue_index_raw": "",
                        "wildtype_aa": "",
                        "mutant_aa": "",
                    }
                ]
            meta = metadata.get(urn, {})
            for parsed in parsed_mutations:
                if not under_cap(mutation_counts, source, max_records):
                    break
                rows_mutation.append(
                    {
                        "source": source,
                        "complex_id": urn,
                        "pdb_id": "",
                        "chain_id": "",
                        "mutation": parsed["mutation"],
                        "residue_index_raw": parsed["residue_index_raw"],
                        "wildtype_aa": parsed["wildtype_aa"],
                        "mutant_aa": parsed["mutant_aa"],
                        "ddg": "",
                        "score": score,
                        "score_direction": "source_defined",
                        "assay_info": clean(meta.get("title", "")),
                        "metadata_json": json_text(
                            {
                                "source_file": display_path(score_file, repo_root),
                                "accession": accession,
                                "hgvs_nt": clean(record.get("hgvs_nt", "")),
                                "hgvs_pro": clean(record.get("hgvs_pro", "")),
                                "metadata": meta,
                                "note": "MaveDB rows do not provide PDB chain fields in this local score export.",
                            }
                        ),
                    }
                )
                bump(mutation_counts, source)


def build_tables(args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    repo_root = Path(args.repo_root).resolve()
    max_records = args.max_records_per_source
    if args.max_records is not None:
        max_records = args.max_records
    if args.smoke and max_records is None:
        max_records = 20

    complex_rows: list[dict[str, str]] = []
    mutation_rows: list[dict[str, str]] = []
    complex_counts: dict[str, int] = {}
    mutation_counts: dict[str, int] = {}

    add_ab_bind(
        complex_rows,
        mutation_rows,
        repo_root,
        resolve_path(repo_root, args.ab_bind_dir),
        max_records,
        complex_counts,
        mutation_counts,
    )
    add_sabdab(complex_rows, repo_root, resolve_path(repo_root, args.sabdab_dir), max_records, complex_counts)
    add_ppb_affinity(
        complex_rows,
        mutation_rows,
        repo_root,
        resolve_path(repo_root, args.ppb_affinity_dir),
        max_records,
        complex_counts,
        mutation_counts,
    )
    add_skempi(
        complex_rows,
        mutation_rows,
        repo_root,
        resolve_path(repo_root, args.skempi_csv),
        resolve_path(repo_root, args.skempi_dir),
        max_records,
        complex_counts,
        mutation_counts,
    )
    add_mavedb(mutation_rows, repo_root, resolve_path(repo_root, args.mavedb_dir), max_records, mutation_counts)
    return complex_rows, mutation_rows


def resolve_path(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build binder complex registry and mutation score tables from local raw experimental sources."
    )
    parser.add_argument("--repo_root", default=str(repo_root_from_script()), help="Repository root.")
    parser.add_argument("--ab_bind_dir", default="data/raw/ab_bind", help="AB-Bind raw directory.")
    parser.add_argument("--sabdab_dir", default="data/raw/sabdab", help="SAbDab raw directory.")
    parser.add_argument("--ppb_affinity_dir", default="data/raw/ppb_affinity", help="PPB-Affinity raw directory.")
    parser.add_argument("--mavedb_dir", default="data/raw/mavedb", help="MaveDB raw directory.")
    parser.add_argument(
        "--skempi_dir",
        default="data/raw/skempi_v2",
        help="SKEMPI v2 structure directory.",
    )
    parser.add_argument(
        "--skempi_csv",
        default="data/raw/skempi_v2/skempi_v2.csv",
        help="SKEMPI v2 semicolon-delimited CSV.",
    )
    parser.add_argument(
        "--out_complex_csv",
        default="data/processed/experiments/binder_complex_registry.csv",
        help="Output binder complex registry CSV.",
    )
    parser.add_argument(
        "--out_mutation_csv",
        default="data/processed/experiments/binder_mutation_scores.csv",
        help="Output binder mutation score CSV.",
    )
    parser.add_argument("--max_records_per_source", type=int, default=None, help="Cap emitted rows per source and table.")
    parser.add_argument("--max_records", type=int, default=None, help="Alias cap for smoke runs.")
    parser.add_argument("--smoke", action="store_true", help="Alias for --max_records_per_source 20 when no cap is supplied.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    complex_rows, mutation_rows = build_tables(args)
    write_csv(resolve_out(repo_root, args.out_complex_csv), COMPLEX_COLUMNS, complex_rows)
    write_csv(resolve_out(repo_root, args.out_mutation_csv), MUTATION_COLUMNS, mutation_rows)
    print(f"wrote {len(complex_rows)} complexes and {len(mutation_rows)} mutation-score rows")


if __name__ == "__main__":
    main()
