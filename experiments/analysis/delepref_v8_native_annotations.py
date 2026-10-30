"""Descriptive native-structure annotations, never a selector or quality gate.

Deposited HELIX/SHEET records are positive annotations only. Their absence is
unknown, not coil/disorder. Native CA contact degree is a geometric descriptor,
not an experimentally established core, active site, or deletion tolerance.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import gemmi
import numpy as np

from delepref_v8_structure_prepare import ROOT, sha, read, write


def native_features(wt):
    assert sha(wt["wt_native_json"]) == wt["wt_native_json_sha256"]
    assert sha(wt["native_reference_path"]) == wt["native_reference_sha256"]
    native = json.loads(Path(wt["wt_native_json"]).read_text())
    length = len(wt["sequence"])
    coords = np.full((length, 3), np.nan)
    for k, value in native["coordinates"].items():
        coords[int(k)] = value
    distance = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    separation = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :])
    contacts = np.isfinite(distance) & (distance < 8.) & (separation >= 3)
    degree = contacts.sum(1)
    known = np.isfinite(coords).all(1)
    threshold = max(1., float(np.quantile(degree[known], .75)))
    high = known & (degree >= threshold)
    structure = gemmi.read_structure(wt["native_reference_path"])
    residues = native["mapping"]["residue_map"]
    helix = np.zeros(length, dtype=bool)
    sheet = np.zeros(length, dtype=bool)

    def add(start, end, mask):
        if start.chain_name != end.chain_name:
            return
        a = (start.res_id.seqid.num, start.res_id.seqid.icode.strip("\x00 "))
        b = (end.res_id.seqid.num, end.res_id.seqid.icode.strip("\x00 "))
        if a > b:
            raise ValueError("reversed deposited secondary-structure annotation")
        for residue in residues:
            key = (residue["auth_number"], residue["insertion_code"].strip("\x00 "))
            if residue["chain"] == start.chain_name and a <= key <= b:
                mask[residue["sequence_index"]] = True
    for h in structure.helices:
        add(h.start, h.end, helix)
    for s in structure.sheets:
        for strand in s.strands:
            add(strand.start, strand.end, sheet)
    return coords, degree, known, high, helix, sheet, threshold


def main():
    out = "reports/delepref_v8_native_annotations/"
    (ROOT / out).mkdir(exist_ok=False)
    panel = read("reports/delepref_v8_native_panel/main_panel.json")
    evidence = {(r["protein_id"], r["budget"]): r for r in read("reports/delepref_v8_evaluation_preparation/retrieval_budget_evidence.json")}
    logical = read("reports/delepref_v8_structure_preparation/logical_requests.json")
    features = {}; protein_rows = []
    for wt in panel:
        coords, degree, known, high, helix, sheet, threshold = native_features(wt)
        features[wt["protein_id"]] = (wt, degree, known, high, helix, sheet)
        protein_rows.append(dict(protein_id=wt["protein_id"], source=wt["stratum"], WT_length=len(wt["sequence"]),
            native_reference_sha256=wt["native_reference_sha256"], native_json_sha256=wt["wt_native_json_sha256"],
            observed_CA=int(known.sum()), unknown_CA=int((~known).sum()),
            native_contact_degree=degree.tolist(), high_contact_threshold=threshold,
            high_contact_positions=np.flatnonzero(high).tolist(),
            deposited_helix_positions=np.flatnonzero(helix).tolist(), deposited_sheet_positions=np.flatnonzero(sheet).tolist(),
            unannotated_observed_CA=int((known & ~(helix | sheet)).sum()),
            secondary_structure_source="deposited native HELIX/SHEET records mapped by original chain/auth number/insertion code; no DSSP or positional proxy"))
    output_rows = []
    for row in logical:
        wt, degree, known, high, helix, sheet = features[row["protein_id"]]
        length = len(wt["sequence"])
        deleted = np.ones(length, dtype=bool); deleted[row["kept"]] = False
        count = int(deleted.sum())
        known_deleted = deleted & known
        support = evidence.get((row["protein_id"], row["budget"]))
        terminal_width = int(np.ceil(.1 * length))
        terminal = np.zeros(length, dtype=bool); terminal[:terminal_width] = True; terminal[-terminal_width:] = True
        output_rows.append(dict(request_id=row["request_id"], protein_id=row["protein_id"], predictor=row["predictor"],
            role=row["role"], method=row["method"], budget=row["budget"], source=wt["stratum"],
            length_bin="<=250" if length <= 250 else "251-400" if length <= 400 else ">400",
            retrieval_support_class=support["support_class"] if support else "WT_not_applicable",
            retrieval_representatives=support["representatives"] if support else None,
            deleted_total=count, deleted_observed_CA=int(known_deleted.sum()), deleted_unknown_CA=int((deleted & ~known).sum()),
            deleted_annotated_helix=int((deleted & helix).sum()), deleted_annotated_sheet=int((deleted & sheet).sum()),
            annotated_helix_retention=1. - int((deleted & helix).sum()) / int(helix.sum()) if helix.any() else None,
            annotated_sheet_retention=1. - int((deleted & sheet).sum()) / int(sheet.sum()) if sheet.any() else None,
            high_contact_fraction_deleted=int((deleted & high).sum()) / int(high.sum()) if high.any() else None,
            mean_native_contact_degree_deleted=float(degree[known_deleted].mean()) if known_deleted.any() else None,
            deletion_fraction_at_outer_10pct=int((deleted & terminal).sum()) / count if count else None))
    write(out + "per_WT_native_features.json", protein_rows)
    write(out + "per_output_deletion_annotations.json", output_rows)
    report = dict(status="AUXILIARY_NATIVE_ANNOTATIONS_COMPLETE", source_sha256=sha(__file__),
        created_at_UTC=datetime.now(timezone.utc).isoformat(), gemmi_version=gemmi.__version__, WT=len(panel), outputs=len(output_rows),
        WTs_with_deposited_helix=sum(bool(r["deposited_helix_positions"]) for r in protein_rows),
        WTs_with_deposited_sheet=sum(bool(r["deposited_sheet_positions"]) for r in protein_rows),
        full_cohort_source_counts=dict(Counter(r["stratum"] for r in panel)),
        primary_20pct_support_counts=dict(Counter(evidence[(r["protein_id"], .2)]["support_class"] for r in panel)),
        definitions_set_after_prediction_started_but_before_reading_quality_results=True,
        primary_confirmation_protocol_changed=False, frozen_generation_or_selection_changed=False,
        scope="auxiliary descriptive source/length/support/observed-secondary-structure/contact-degree diagnostics on all frozen outputs; no confirmatory subgroup claim",
        limitations=["Absence of deposited HELIX/SHEET records is unknown, not disorder or a linker annotation",
                     "CA contact-degree upper quartile (ties retained; minimum degree1) is not a proven functional or all-atom structural core",
                     "No catalytic/binding residue or biological-interface preservation claim without external annotations"],
        outcome_based_filtering=False, new_prediction_calls=0, paper_changes=False,
        files={name: sha(ROOT / out / name) for name in ("per_WT_native_features.json", "per_output_deletion_annotations.json")})
    write(out + "completion.json", report)
    print(json.dumps({k: report[k] for k in ("status", "WT", "outputs", "WTs_with_deposited_helix", "WTs_with_deposited_sheet")}))


if __name__ == "__main__":
    main()
