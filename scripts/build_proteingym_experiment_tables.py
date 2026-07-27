#!/usr/bin/env python3
"""Normalize ProteinGym indel assays into experiment preparation tables."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Iterable


VARIANT_COLUMNS = [
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
]

ASSAY_COLUMNS = [
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
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_metadata(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise SystemExit(f"failed to read ProteinGym metadata CSV {path}: {exc}") from exc


def zip_csv_member_map(path: Path) -> dict[str, str]:
    try:
        with zipfile.ZipFile(path) as zf:
            return {Path(name).name: name for name in zf.namelist() if name.lower().endswith(".csv")}
    except (OSError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"failed to inspect ProteinGym ZIP {path}: {exc}") from exc


def iter_zip_csv_rows(path: Path, member: str) -> Iterable[dict[str, str]]:
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
                yield from csv.DictReader(text)
    except KeyError as exc:
        raise SystemExit(f"missing ProteinGym ZIP member {member} in {path}") from exc
    except (OSError, zipfile.BadZipFile, csv.Error) as exc:
        raise SystemExit(f"failed to read ProteinGym ZIP member {member} in {path}: {exc}") from exc


def common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def common_suffix_len(left: str, right: str, prefix_len: int) -> int:
    max_suffix = min(len(left), len(right)) - prefix_len
    idx = 0
    while idx < max_suffix and left[len(left) - 1 - idx] == right[len(right) - 1 - idx]:
        idx += 1
    return idx


def classify_edit(wildtype: str, mutant: str) -> dict[str, object]:
    if wildtype == mutant:
        return {
            "edit_type": "wildtype",
            "deleted_intervals": [],
            "inserted_sequence": "",
            "deletion_length": 0,
        }

    prefix = common_prefix_len(wildtype, mutant)
    suffix = common_suffix_len(wildtype, mutant, prefix)
    wt_mid_end = len(wildtype) - suffix
    mut_mid_end = len(mutant) - suffix
    deleted = wildtype[prefix:wt_mid_end]
    inserted = mutant[prefix:mut_mid_end]

    if deleted and not inserted and wildtype[:prefix] + wildtype[wt_mid_end:] == mutant:
        return {
            "edit_type": "single_segment_deletion",
            "deleted_intervals": [[prefix, wt_mid_end - 1]],
            "inserted_sequence": "",
            "deletion_length": len(deleted),
        }

    intervals = [[prefix, wt_mid_end - 1]] if deleted else []
    return {
        "edit_type": "complex_indel",
        "deleted_intervals": intervals,
        "inserted_sequence": inserted,
        "deletion_length": len(deleted),
    }


def assay_type_guess(meta: dict[str, str]) -> str:
    candidates = [
        meta.get("selection_assay", ""),
        meta.get("selection_type", ""),
        meta.get("coarse_selection_type", ""),
        meta.get("raw_DMS_phenotype_name", ""),
        meta.get("title", ""),
    ]
    text = " | ".join(item for item in candidates if item)
    return text or "unknown"


def metadata_for_variant(row: dict[str, str], meta: dict[str, str], zip_member: str) -> str:
    promoted = {"mutated_sequence", "DMS_score", "DMS_score_bin"}
    extra_row = {key: value for key, value in row.items() if key not in promoted and value not in ("", None)}
    payload = {
        "protein_gym_label_source": "external_experimental_label",
        "zip_member": zip_member,
        "metadata_fields": {
            key: meta.get(key, "")
            for key in [
                "taxon",
                "source_organism",
                "first_author",
                "title",
                "year",
                "molecule_name",
                "selection_assay",
                "selection_type",
                "raw_DMS_directionality",
                "ProteinGym_version",
                "coarse_selection_type",
            ]
        },
    }
    if extra_row:
        payload["source_row_extra"] = extra_row
    return json_text(payload)


def build_tables(
    raw_dir: Path,
    repo_root: Path,
    max_assays: int | None = None,
    max_records: int | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metadata_path = raw_dir / "DMS_indels.csv"
    zip_path = raw_dir / "DMS_ProteinGym_indels.zip"
    metadata_rows = read_metadata(metadata_path)
    member_by_file = zip_csv_member_map(zip_path)

    if max_assays is not None:
        metadata_rows = metadata_rows[:max_assays]

    variant_rows: list[dict[str, str]] = []
    assay_rows: list[dict[str, str]] = []
    source_zip = display_path(zip_path, repo_root)

    for assay_index, meta in enumerate(metadata_rows):
        dms_id = meta.get("DMS_id", "")
        dms_filename = meta.get("DMS_filename", "")
        zip_member = member_by_file.get(dms_filename)
        if not zip_member:
            raise SystemExit(f"ProteinGym ZIP lacks assay file listed in metadata: {dms_filename}")

        wildtype = meta.get("target_seq", "")
        target_len = len(wildtype) if wildtype else int(meta.get("seq_len") or 0)
        assay_id = meta.get("DMS_index") or f"proteingym_indel_{assay_index:04d}"
        total_variants = 0
        total_single_deletions = 0

        for row_index, row in enumerate(iter_zip_csv_rows(zip_path, zip_member)):
            if max_records is not None and total_variants >= max_records:
                break
            mutant = row.get("mutated_sequence", "")
            edit = classify_edit(wildtype, mutant)
            deletion_length = int(edit["deletion_length"])
            if edit["edit_type"] == "single_segment_deletion":
                total_single_deletions += 1
            total_variants += 1
            deletion_fraction = deletion_length / float(target_len) if target_len else 0.0
            variant_rows.append(
                {
                    "assay_id": assay_id,
                    "dms_id": dms_id,
                    "source_file": f"{source_zip}::{zip_member}",
                    "variant_id": f"{dms_id}:{row_index + 1}",
                    "wildtype_sequence": wildtype,
                    "mutant_sequence": mutant,
                    "edit_type": str(edit["edit_type"]),
                    "deleted_intervals": json_text(edit["deleted_intervals"]),
                    "inserted_sequence": str(edit["inserted_sequence"]),
                    "deletion_length": str(deletion_length),
                    "deletion_fraction": f"{deletion_fraction:.8g}",
                    "dms_score": row.get("DMS_score", ""),
                    "dms_score_bin": row.get("DMS_score_bin", ""),
                    "target_length": str(target_len),
                    "uniprot_id": meta.get("UniProt_ID", ""),
                    "metadata_json": metadata_for_variant(row, meta, zip_member),
                }
            )

        assay_rows.append(
            {
                "assay_id": assay_id,
                "dms_id": dms_id,
                "uniprot_id": meta.get("UniProt_ID", ""),
                "target_seq": wildtype,
                "seq_len": str(target_len),
                "assay_type_guess": assay_type_guess(meta),
                "source_file": f"{source_zip}::{zip_member}",
                "total_variants": str(total_variants),
                "total_single_segment_deletions": str(total_single_deletions),
                "notes": "ProteinGym indel labels are external experimental labels; no teacher scoring applied.",
            }
        )

    return variant_rows, assay_rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def resolve_out(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ProteinGym indel variant and assay registry tables for experiment preparation."
    )
    parser.add_argument("--repo_root", default=str(repo_root_from_script()), help="Repository root.")
    parser.add_argument("--raw_dir", default="data/raw/proteingym", help="ProteinGym raw directory.")
    parser.add_argument(
        "--out_variants_csv",
        default="data/processed/experiments/proteingym_indel_variants.csv",
        help="Output normalized ProteinGym variant CSV.",
    )
    parser.add_argument(
        "--out_assays_csv",
        default="data/processed/experiments/proteingym_assay_registry.csv",
        help="Output ProteinGym assay registry CSV.",
    )
    parser.add_argument("--max_assays", type=int, default=None, help="Optional cap on assays for smoke runs.")
    parser.add_argument("--max_records", type=int, default=None, help="Optional cap on variant rows per assay.")
    parser.add_argument("--smoke", action="store_true", help="Alias for --max_assays 5 when no explicit cap is supplied.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = repo_root / raw_dir
    max_assays = args.max_assays
    if args.smoke and max_assays is None:
        max_assays = 5

    variant_rows, assay_rows = build_tables(
        raw_dir=raw_dir,
        repo_root=repo_root,
        max_assays=max_assays,
        max_records=args.max_records,
    )
    write_csv(resolve_out(repo_root, args.out_variants_csv), VARIANT_COLUMNS, variant_rows)
    write_csv(resolve_out(repo_root, args.out_assays_csv), ASSAY_COLUMNS, assay_rows)
    print(f"wrote {len(variant_rows)} variants and {len(assay_rows)} assays")


if __name__ == "__main__":
    main()
