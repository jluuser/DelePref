"""Audit historical exact-deletion records into explicitly non-independent development data."""
import argparse
import collections
import csv
import json
from pathlib import Path
import numpy as np
from common import AUTHORIZED_ROOT, confined_path, identity, read_jsonl, sha_file, sha_text, write_json, write_jsonl
from adapters.delepref_adapter import retained_mapping, morphology, strict_delete_count
from adapters.structure_adapter import audited_wt_reference, read_coordinates
from build_pairs import reference_pairs

AA = set("ACDEFGHIKLMNPQRSTVWY")


def record_key(row):
    return (row["protein_id"], row["method"], round(float(row.get("budget_requested", row.get("budget"))), 6), int(row.get("candidate_index", 0)))


def source_records(e1):
    records = {}; references = {}; source_hashes = {}
    for folder, field in [("Existing_core", "rows"), ("K8", "existing_records")]:
        for path in sorted((e1 / folder / "groups").glob("E1_*.json")):
            group = json.loads(path.read_text()); source_hashes[str(path)] = sha_file(path)
            references[group["protein_id"]] = group
            for row in group[field]:
                if row["method"] not in ("evocompress", "scisor"):
                    continue
                key = record_key(row)
                if key in records and records[key][0]["sequence"] != row["sequence"]:
                    raise ValueError("conflicting frozen candidate identity: " + str(key))
                records[key] = (row, str(path))
    for path in sorted((e1 / "K8/scisor").glob("E1_*/candidates.jsonl")):
        source_hashes[str(path)] = sha_file(path)
        for row in read_jsonl(path):
            key = record_key(row)
            if key in records and records[key][0]["sequence"] != row["sequence"]:
                raise ValueError("conflicting SCISOR candidate")
            records[key] = (row, str(path))
    return records, references, source_hashes


def forward_evidence(prediction_path, row, e1):
    path = Path(prediction_path)
    own = path.with_suffix(".json")
    if own.exists():
        data = json.loads(own.read_text())
        completion = path.parent / "completion.json"
        if data.get("status") != "complete" or data.get("sequence_sha256") != sha_text(row["sequence"]):
            raise ValueError("candidate prediction metadata identity/status mismatch")
        if not data.get("cache_hit"):
            if not completion.exists() or not (data.get("inference_seconds") or 0) > 0:
                raise ValueError("missing actual forward evidence")
            group = json.loads(completion.read_text())
            if group.get("status") != "complete" or not group.get("job_id") or not group.get("gpu"):
                raise ValueError("missing group forward completion")
            return {"profile": data["predictor_profile"], "gpu": group["gpu"], "seed": 20260908,
                    "evidence_path": str(own), "evidence_sha256": sha_file(own),
                    "group_evidence_path": str(completion), "group_evidence_sha256": sha_file(completion),
                    "runtime_weights_hash_recorded": False, "weights_identity": "historical_path_and_code_only",
                    "code_sha256": group.get("code_sha256"), "actual_forward": True}
        # Cached K8 candidate zero must trace to its original forward report.
    report = path.parent / "report.json"
    if not report.exists():
        raise ValueError("original forward report absent")
    data = json.loads(report.read_text())
    matches = [x for x in data["records"] if x["protein_id"] == row["protein_id"] and x["method"] == row["method"] and float(x["budget"]) == float(row.get("budget_requested", row.get("budget"))) and x.get("status") == "success"]
    if len(matches) != 1 or not (matches[0].get("runtime_sec") or 0) > 0 or not data.get("gpu"):
        raise ValueError("original forward not uniquely verified")
    code = e1.parents[1] / "formal_v9/revision_20260906_abc/D/code/e1_array.py"
    return {"profile": "ESMFold-v1|recycles3|chunk64|ESMhalf|eval|seed_unrecorded",
            "gpu": data["gpu"], "seed": None, "actual_forward": True,
            "evidence_path": str(report), "evidence_sha256": sha_file(report),
            "runtime_weights_hash_recorded": False, "weights_identity": "historical_path_and_code_only",
            "code_path": str(code), "code_sha256": sha_file(code)}


def audit(protocol, output):
    root = Path(protocol["root"])
    if root.resolve() != AUTHORIZED_ROOT.resolve():
        raise ValueError("protocol.root must be the authorized new experiment root")
    confined_path(output)
    if output.resolve() != (root / "manifests/development.jsonl").resolve():
        raise ValueError("manifest output must be new experiment root/manifests/development.jsonl")
    e1 = Path(protocol["sources"]["e1"]); general = Path(protocol["sources"]["general240"])
    records, groups, source_hashes = source_records(e1)
    final_path = e1 / "FINAL/per_candidate.jsonl"
    source_hashes[str(final_path)] = sha_file(final_path)
    source_hashes[str(general / "selected_with_predictions.csv")] = sha_file(general / "selected_with_predictions.csv")
    refs = {}; audit_rows = []; accepted = {}; exclusions = []; native_hashes = {}
    for pid, group in sorted(groups.items()):
        wt = group["wt_sequence"]; ref = group["reference"]; native = general / "native" / (pid + ".json")
        try:
            coords, mapping = audited_wt_reference(native, wt, ref["reference_coordinate_path"], ref["auth_chain_id"])
            if set(wt) - AA:
                raise ValueError("non-standard WT amino acid")
            i, j = reference_pairs(coords)
            contact_i, contact_j = reference_pairs(coords, 8.)
            mapping["sequence_sha256"] = sha_text(wt)
            write_json(root / "caches/wt_references" / (pid + ".mapping.json"), mapping)
            refs[pid] = (wt, coords, i, j, contact_i, contact_j, ref, native)
            native_hashes[pid] = sha_file(native)
        except Exception as error:
            exclusions.append({"protein_id": pid, "scope": "WT_reference", "reason": str(error)})

    def add(row, source, pred, expected_hash, evidence, source_path, legacy=None):
        pid = row["protein_id"]
        if pid not in refs:
            raise ValueError("WT reference failed audit")
        wt, coords, i, j, ci, cj, ref, native = refs[pid]
        seq = row["sequence"]; mask = row.get("deletion_mask", row.get("mask"))
        if isinstance(mask, str):
            mask = json.loads(mask)
        kept = retained_mapping(wt, seq, mask)
        if set(seq) - AA:
            raise ValueError("non-standard variant amino acid")
        budget = float(row.get("budget_requested", row.get("budget")))
        if len(wt) - len(seq) != strict_delete_count(len(wt), budget):
            raise ValueError("candidate does not satisfy exact budget")
        pred_sha = sha_file(pred)
        if pred_sha != expected_hash:
            raise ValueError("variant PDB hash mismatch")
        mut, mut_map = read_coordinates(pred, seq)
        mask = np.asarray(mask, dtype=bool)
        covered = (~mask[i]) & (~mask[j])
        contact_covered = (~mask[ci]) & (~mask[cj])
        logical_key = (sha_text(wt), sha_text("".join("1" if d else "0" for d in mask)), sha_text(seq))
        candidate_id = identity(logical_key)
        morph = morphology(mask)
        profile_id = identity({k: evidence.get(k) for k in ["profile", "gpu", "seed", "weights_identity"]})
        item = {"candidate_id": candidate_id, "protein_id": pid, "wt_sequence": wt, "wt_sequence_sha256": logical_key[0],
                "sequence": seq, "sequence_sha256": logical_key[2], "mask": mask.astype(int).tolist(), "mask_sha256": logical_key[1],
                "budget": budget, "source": source, "generator": row.get("method", "historical_scored_pool"),
                "candidate_index": int(row.get("candidate_index", 0)), "source_path": source_path,
                "source_sha256": source_hashes[source_path],
                "source_selection": "historical_score_selected" if source == "general240_selected" else "frozen_generator_candidates",
                "role": "historical_development", "original_external_test_state": ref.get("external_test_state"),
                "original_future_training_policy": ref.get("future_training_policy"),
                "coordinate_scope": ref["coordinate_scope"], "cath_h_superfamily": ref["cath_h_superfamily"],
                "pdb_id": ref["pdb_id"], "reference_auth_chain": ref["auth_chain_id"],
                "wt_native_json": str(native), "wt_native_json_sha256": native_hashes[pid],
                "reference_path": ref["reference_coordinate_path"], "reference_sha256": ref["reference_coordinate_sha256"],
                "reference_experimental_methods": ref.get("experimental_methods"),
                "wt_ca_coverage": float(np.isfinite(coords).all(1).mean()),
                "prediction_path": str(pred), "prediction_sha256": pred_sha,
                "prediction_profile": evidence, "profile_id": profile_id,
                "prediction_missing_positions": mut_map["missing_positions"],
                "direct_relationship_coverage": float(covered.mean()) if len(i) else None,
                "direct_contact_loss": 1 - float(contact_covered.mean()) if len(ci) else None,
                "morphology": morph, "legacy_metrics": legacy or {},
                "audit_status": "VERIFIED_MAPPING_AND_FORWARD", "runtime_weight_identity_limitation": not evidence.get("runtime_weights_hash_recorded", False)}
        occurrence = {k: item[k] for k in ["source", "generator", "source_path", "source_sha256", "source_selection", "role", "original_external_test_state", "original_future_training_policy", "prediction_path", "prediction_sha256", "prediction_profile", "profile_id", "reference_path", "reference_sha256", "legacy_metrics"]}
        if candidate_id in accepted:
            accepted[candidate_id]["occurrences"].append(occurrence)
        else:
            item["occurrences"] = [occurrence]
            accepted[candidate_id] = item
        audit_rows.append({"candidate_id": candidate_id, "protein_id": pid, "source": source, "budget": budget, "status": "accepted", "profile_id": profile_id, **morph})

    for final in read_jsonl(final_path):
        if final["experiment"] != "E1":
            continue
        if final.get("mapping_type") != "exactgenerationdeletionmap":
            exclusions.append({"protein_id": final["protein_id"], "method": final["method"], "candidate_index": final["candidate_index"], "budget": final["budget"], "reason": "inferred_mapping_excluded"})
            continue
        try:
            row, source_path = records[record_key(final)]
            evidence = forward_evidence(final["prediction_path"], row, e1)
            add(row, "e1_exact", final["prediction_path"], final["prediction_sha256"], evidence, source_path, {k: final[k] for k in ["retained_ca_lddt", "native_contact_recovery_lower", "native_contact_recovery_upper", "native_contact_n"] if k in final})
        except Exception as error:
            exclusions.append({"protein_id": final["protein_id"], "method": final["method"], "candidate_index": final["candidate_index"], "budget": final["budget"], "reason": str(error)})
    with (general / "selected_with_predictions.csv").open() as handle:
        for row in csv.DictReader(handle):
            try:
                done = general / "preds" / row["prediction_key"] / "completion.json"
                data = json.loads(done.read_text())
                if data.get("status") != "complete" or data["sequence_sha256"] != sha_text(row["sequence"]) or data.get("cache_hit") or not (data.get("inference_seconds") or 0) > 0:
                    raise ValueError("general forward evidence incomplete")
                evidence = {"profile": data["predictor_profile"], "gpu": data["gpu"], "seed": data["predictor_profile"]["seed"],
                            "runtime_weights_hash_recorded": True, "weights_identity": data["weights_sha256"], "actual_forward": True,
                            "evidence_path": str(done), "evidence_sha256": sha_file(done)}
                add(row, "general240_selected", data["prediction_path"], data["prediction_sha256"], evidence, str(general / "selected_with_predictions.csv"))
            except Exception as error:
                exclusions.append({"protein_id": row["protein_id"], "candidate_index": row["candidate_index"], "source": "general240_selected", "reason": str(error)})
    rows = sorted(accepted.values(), key=lambda x: x["candidate_id"])
    # Freeze one documented hardware/settings/seed stratum for the first main run.
    # Other audited E1 strata remain available for explicitly separate sensitivity runs.
    primary_profile = "ESMFold-v1|recycles3|chunk64|ESMhalf|eval|seed20260908"
    for row in rows:
        profile = row["prediction_profile"]
        row["primary_eligible"] = row["source"] == "e1_exact" and profile["profile"] == primary_profile and profile["gpu"] == "NVIDIA A40"
        row["primary_exclusion_reason"] = None if row["primary_eligible"] else "separate_predictor_profile_or_historical_selected_sensitivity"
    write_jsonl(output, rows); write_jsonl(root / "manifests/candidate_audit.jsonl", audit_rows)
    primary_path = root / "manifests/development_primary.jsonl"
    primary = [r for r in rows if r["primary_eligible"]]
    write_jsonl(primary_path, primary)
    write_jsonl(root / "reports/manifest_exclusions.jsonl", exclusions)
    matrix = collections.Counter((r["source"], str(r["budget"]), str(r["morphology"]["num_spans"]), str(r["morphology"]["terminal"]), r["profile_id"]) for r in rows)
    report = {"status": "MANIFEST_FROZEN" if rows else "MANIFEST_FAILED", "input_hashes": source_hashes,
              "manifest_sha256": sha_file(output), "candidates": len(rows), "unique_variant_sequences": len({r["sequence_sha256"] for r in rows}),
              "proteins": len({r["protein_id"] for r in rows}), "source_counts": dict(collections.Counter(r["source"] for r in rows)),
              "primary_manifest_sha256": sha_file(primary_path), "primary_candidates": len(primary), "primary_proteins": len({r["protein_id"] for r in primary}),
              "primary_profile": {"settings": primary_profile, "hardware": "NVIDIA A40", "selection_reason": "Only the explicitly recorded seeded profile is combined in first main training; older unrecorded-seed E1 profile retained for separate sensitivity. Thus any reduction in WT count is an audit consequence, not a predicted-WT prerequisite."},
              "exclusion_reasons": dict(collections.Counter(x["reason"] for x in exclusions)),
              "audit_matrix": [{"source": k[0], "budget": k[1], "spans": k[2], "terminal": k[3], "profile_id": k[4], "count": v} for k, v in matrix.items()],
              "historical_weight_hash_limitation": "E1 forward reports do not record runtime weights SHA; code and path traced, current assets hashed separately. This is disclosed, not silently treated as complete runtime identity."}
    write_json(root / "reports/manifest_audit.json", report)
    print(json.dumps({k: report[k] for k in ["status", "candidates", "proteins", "source_counts", "exclusion_reasons"]}))
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol", required=True)
    parser.add_argument("--sources", default="general240,e1_exact"); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.sources != "general240,e1_exact":
        raise ValueError("v2 manifest audit requires both historical sources")
    audit(json.loads(Path(args.protocol).read_text()), Path(args.output))


if __name__ == "__main__":
    main()
