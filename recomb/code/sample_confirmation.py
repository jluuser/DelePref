"""Freeze independent WT references using WT-only metadata and explicit similarity checks."""
import argparse
import csv
import json
from pathlib import Path
import gemmi
import numpy as np
from common import AUTHORIZED_ROOT, confined_path, identity, read_jsonl, sha_file, sha_text, write_json, write_jsonl
from adapters.structure_adapter import read_coordinates, read_cath_reference
from split_families import make_aligner, similarity, threshold_hit, unique_proteins


def freeze_wt(protocol_path, manifest, output):
    protocol = json.loads(Path(protocol_path).read_text()); config = protocol["confirmation"]
    if Path(protocol["root"]).resolve() != AUTHORIZED_ROOT.resolve():
        raise ValueError("unauthorized protocol root")
    output = confined_path(output)
    if output.resolve() != (AUTHORIZED_ROOT / "manifests/confirmation_wt_frozen.jsonl").resolve():
        raise ValueError("WT output must be experiment manifests/confirmation_wt_frozen.jsonl")
    if output.exists():
        raise ValueError("confirmation WT cohort already frozen; refusing overwrite")
    dev = unique_proteins(read_jsonl(manifest))
    base = Path(protocol["sources"]["delepath"]) / "data"
    pool_path = base / "processed/evocompress/v1/candidates/structure_candidate_pool.csv"
    cache = base / "raw/evocompress/rcsb_selected/entries"
    families = {x["cath_h_superfamily"] for x in dev.values()}
    sources = {(x["pdb_id"].upper(), x["reference_auth_chain"]) for x in dev.values()}
    pilot_path = Path(protocol["sources"]["pilot"]) / "enzyme_manifest.json"
    pilot = json.loads(pilot_path.read_text())["proteins"]
    enzyme_pdbs = set()
    for protein in pilot:
        if sha_file(protein["native_mapping"]) != protein["native_mapping_sha256"]:
            raise ValueError("enzyme WT reference manifest identity changed")
        native = json.loads(Path(protein["native_mapping"]).read_text())
        enzyme_pdbs.add(Path(native["reference_structure_path"]).name.split(".")[0].lower())
    cath_base = base / "par_contraction/cath_v4_3_0"
    cath_list = cath_base / "cath-domain-list.txt"
    enzyme_families = set(); enzyme_annotation_pdbs = set()
    for line in cath_list.open():
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if fields[0][:4].lower() in enzyme_pdbs:
            enzyme_families.add(".".join(fields[1:5])); enzyme_annotation_pdbs.add(fields[0][:4].lower())
    if enzyme_pdbs != enzyme_annotation_pdbs:
        raise ValueError("some enzyme20 reference PDBs lack CATH family annotations")
    families.update(enzyme_families)
    controls = [(p, r["wt_sequence"]) for p, r in dev.items()] + [(r["protein_id"], r["wt_sequence"]) for r in pilot]
    pool = list(csv.DictReader(pool_path.open()))
    pool.sort(key=lambda r: identity([protocol["seed"], r["domain_id"]]))
    aligner = make_aligner(protocol["split"]["alignment"])
    selected = []; rejected = []; comparisons = []; selected_families = set(); selected_sources = set()
    target = config["wt_count"] + 16
    for row in pool:
        pid = row["domain_id"]; sequence = row["sequence"]; family = row["cath_h_superfamily"]
        source = row["pdb_id"].upper(), row["auth_chain_id"]
        if family in families or family in selected_families or source in sources or source in selected_sources:
            continue
        if not config["length_min"] <= len(sequence) <= config["length_max"] or set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
            continue
        reference = cache / (row["pdb_id"].upper() + ".cif.gz")
        cath_reference = cath_base / "cath_s40_domains/dompdb" / pid
        if not reference.exists() and not cath_reference.exists():
            continue
        try:
            if sha_text(sequence) != row["sequence_sha256"]:
                raise ValueError("WT metadata sequence SHA mismatch")
            try:
                xyz, mapping = read_coordinates(reference, sequence, row["auth_chain_id"], native=True)
                reference_kind = "rcsb_mmcif_polymer_mapping"
            except Exception:
                xyz, mapping = read_cath_reference(cath_reference, sequence, row["auth_chain_id"])
                reference = cath_reference
                reference_kind = "CATH_atom_only_unique_exact_subsequence"
            coverage = float(np.isfinite(xyz).all(1).mean())
            if coverage < .9:
                raise ValueError("WT reference CA coverage below predeclared 0.90")
            current = []; hits = []
            for other_pid, other_seq in controls + [(x["protein_id"], x["wt_sequence"]) for x in selected]:
                stats = similarity(sequence, other_seq, aligner)
                item = {"confirmation_candidate": pid, "other": other_pid, **stats}
                current.append(item)
                if threshold_hit(stats, protocol["split"]):
                    hits.append(item)
            comparisons.extend(current)
            if hits:
                raise ValueError("cross-set or within-cohort sequence threshold hit")
            native_path = AUTHORIZED_ROOT / "caches/wt_references" / (pid + ".json")
            write_json(native_path, {"protein_id": pid, "wt_sequence_sha256": row["sequence_sha256"],
                       "reference_path": str(reference), "reference_sha256": mapping["sha256"],
                       "coordinates": {str(i): xyz[i].tolist() for i in range(len(sequence)) if np.isfinite(xyz[i]).all()},
                       "coordinate_scope": row["coordinate_scope"], "mapping": mapping})
            selected.append({"protein_id": pid, "wt_sequence": sequence, "wt_sequence_sha256": sha_text(sequence),
                             "cath_h_superfamily": family, "coordinate_scope": row["coordinate_scope"],
                             "reference_path": str(reference), "reference_sha256": mapping["sha256"],
                             "reference_auth_chain": row["auth_chain_id"], "pdb_id": row["pdb_id"],
                             "wt_native_json": str(native_path), "wt_native_json_sha256": sha_file(native_path),
                             "wt_ca_coverage": coverage, "resolution_metadata": row.get("resolution"),
                             "source_header": row["source_header"], "selection_rank": len(selected),
                             "reference_kind": reference_kind,
                             "role": "independent_confirmation", "variant_labels_generated": False})
            selected_families.add(family); selected_sources.add(source)
            if len(selected) >= target:
                break
        except Exception as error:
            rejected.append({"protein_id": pid, "reason": str(error)})
    write_jsonl(AUTHORIZED_ROOT / "reports/confirmation_wt_candidates.jsonl", selected)
    write_jsonl(AUTHORIZED_ROOT / "reports/confirmation_wt_rejected.jsonl", rejected)
    write_jsonl(AUTHORIZED_ROOT / "manifests/similarity/development_confirmation_pairs.jsonl", comparisons)
    if len(selected) < config["wt_count"]:
        write_json(AUTHORIZED_ROOT / "reports/confirmation_wt_audit.json", {"status": "INSUFFICIENT_WT_CANDIDATES", "eligible_count": len(selected), "required": config["wt_count"], "candidate_pool_sha256": sha_file(pool_path)})
        raise ValueError(f"only {len(selected)} eligible independent WT references; no cohort frozen")
    write_jsonl(output, selected[:config["wt_count"]])
    write_jsonl(AUTHORIZED_ROOT / "manifests/confirmation_wt_reserves_frozen.jsonl", selected[config["wt_count"]:])
    report = {"status": "CONFIRMATION_WT_FROZEN", "cohort_count": config["wt_count"],
              "independent_families": len({x["cath_h_superfamily"] for x in selected[:config["wt_count"]]}),
              "reserve_count": max(0, len(selected) - config["wt_count"]), "manifest_sha256": sha_file(output),
              "development_manifest_sha256": sha_file(manifest), "protocol_sha256": sha_file(protocol_path),
              "candidate_pool_path": str(pool_path), "candidate_pool_sha256": sha_file(pool_path),
              "enzyme_pilot_manifest_sha256": sha_file(pilot_path), "minimum_ca_coverage": .9,
              "cath_family_list_path": str(cath_list), "cath_family_list_sha256": sha_file(cath_list),
              "excluded_enzyme_families": sorted(enzyme_families), "enzyme_reference_pdbs": sorted(enzyme_pdbs),
              "software": {"gemmi": gemmi.__version__}, "pairwise_comparisons": len(comparisons),
              "highest_pairs": sorted(comparisons, key=lambda x: -x["identity"])[:20],
              "eligibility": "cached RCSB or CATH WT reference, standard sequence 150–550, unique exact polymer/subsequence mapping, >=90% CA; no CATH family/source/sequence overlap with full development or enzyme20; one WT per independent family. Missing WT coordinates remain NaN in full sequence positions.",
              "selection_reads_variant_labels_or_scores": False, "legacy_policy_training_overlap": "optional_baseline_omitted_per_user"}
    write_json(AUTHORIZED_ROOT / "reports/confirmation_wt_audit.json", report)
    print(json.dumps({k: report[k] for k in ["status", "cohort_count", "independent_families", "reserve_count", "pairwise_comparisons"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    freeze_wt(args.protocol, args.manifest, args.output)
