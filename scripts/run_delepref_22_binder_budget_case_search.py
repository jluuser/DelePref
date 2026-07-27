#!/usr/bin/env python3
"""Run a frozen single-chain Binder budget case-study search.

The GPU workflow has separate phases so DelePref, the optional external SCISOR
baseline, and Boltz prediction can use their maintained environments.
Full-length predictions are reused; only edited complexes are predicted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/delepref_step004000_delepref_22_binder_budget_case_search"
BINDER = ROOT / "results/experiments/delepref_step004000_binder"
V12 = ROOT / "results/experiments/delepref_step004000_delepref_12_binder_interface_analysis"
CHECKPOINT = ROOT / "results/experiments/delepref_evaluation_checkpoint/final_freeze/delepref_evaluation_step_00004000.pt"
CONFIG = ROOT / "configs/deletion_preference_student.yaml"
SCISOR_CHECKPOINT = Path(
    os.environ.get("DELEPREF_SCISOR_CHECKPOINT", "models/scisor/SCISOR_U90_S.ckpt")
).expanduser()
SCISOR_P0 = Path(os.environ.get("DELEPREF_SCISOR_P0", str(ROOT / "p0.pt"))).expanduser()
BUDGET = 0.20
ANALYSIS_ID = "DelePref-22"
RUN_TAG = "delepref22_b20"
PROTOCOL_SCHEMA_VERSION = "delepref_22_binder_budget_case_search_protocol_v1"
SUMMARY_SCHEMA_VERSION = "delepref_22_binder_budget_case_search_summary_v1"
VALIDATION_SCHEMA_VERSION = "delepref_22_binder_budget_case_search_validation_v1"
MANIFEST_SCHEMA_VERSION = "delepref_22_artifact_manifest_v1"
STATUS_FOUND = "PASS_DelePref_22_BINDER_B20_CASE_FOUND"
STATUS_NO_ELIGIBLE = "PASS_DelePref_22_BINDER_B20_SEARCH_NO_ELIGIBLE_CASE"
K = 8
GENERATION_SEED = 13
SCISOR_SEED = 20260724
MAX_BINDER_LENGTH = 300
EXPECTED_COHORT_CASES = 22
CONTACT_THRESHOLD = 0.85
CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion "
    "perturbation preference and optional auxiliary risk evidence."
)
PROXY_BOUNDARY = (
    "Post-deletion complex-prediction proxy only; not evidence of experimental "
    "binding, Kd/Ki, or functional preservation."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_text(path: Path, value: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require_status(path: Path, expected: str) -> None:
    actual = load_json(path / "run_summary.json").get("status")
    if actual != expected:
        raise RuntimeError(f"status mismatch for {path}: {actual!r}")


def strict_count(length: int) -> int:
    return max(1, min(length - 1, int(round(BUDGET * length))))


def parse_intervals(value: Any) -> list[tuple[int, int]]:
    raw = json.loads(value) if isinstance(value, str) else value
    return [(int(start), int(end)) for start, end in raw]


def intervals_json(intervals: Sequence[tuple[int, int]]) -> str:
    return json.dumps([[int(a), int(b)] for a, b in intervals], separators=(",", ":"))


def mutant(sequence: str, intervals: Sequence[tuple[int, int]]) -> str:
    deleted = {position for start, end in intervals for position in range(start, end + 1)}
    return "".join(residue for index, residue in enumerate(sequence) if index not in deleted)


def deleted_to_intervals(indices: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(int(index) for index in indices))
    if not ordered:
        return []
    result: list[list[int]] = [[ordered[0], ordered[0]]]
    for index in ordered[1:]:
        if index == result[-1][1] + 1:
            result[-1][1] = index
        else:
            result.append([index, index])
    return [(start, end) for start, end in result]


def protocol_payload(cohort: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "analysis_id": ANALYSIS_ID,
        "budget": BUDGET,
        "cohort_rule": {
            "source": "frozen DelePref Binder 50-case manifest",
            "binder_chain_count": 1,
            "maximum_binder_length": MAX_BINDER_LENGTH,
            "case_count": len(cohort),
            "case_ids": [int(row["case_id"]) for row in cohort],
            "expected_case_count": EXPECTED_COHORT_CASES,
        },
        "frozen_inputs": {
            "binder_manifest": str(BINDER / "binder_manifest.csv"),
            "binder_manifest_sha256": sha256_file(BINDER / "binder_manifest.csv"),
            "binder_prediction_details": str(BINDER / "binder_prediction_details.csv"),
            "binder_prediction_details_sha256": sha256_file(BINDER / "binder_prediction_details.csv"),
            "binder_boltz_cache_audit": str(BINDER / "binder_boltz_cache_audit.csv"),
            "binder_boltz_cache_audit_sha256": sha256_file(BINDER / "binder_boltz_cache_audit.csv"),
            "delepref_12_protocol": str(V12 / "protocol.json"),
            "delepref_12_protocol_sha256": sha256_file(V12 / "protocol.json"),
        },
        "ours": {
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "config": str(CONFIG),
            "config_sha256": sha256_file(CONFIG),
            "k": K,
            "generation_seed": GENERATION_SEED,
            "selection": "minimum deleted WT-interface overlap, then frozen scorer score, then mask SHA256",
            "risk_evidence_role": "optional auxiliary planner evidence",
        },
        "scisor": {
            "checkpoint": str(SCISOR_CHECKPOINT),
            "checkpoint_sha256": sha256_file(SCISOR_CHECKPOINT),
            "p0": str(SCISOR_P0),
            "p0_sha256": sha256_file(SCISOR_P0),
            "seed": SCISOR_SEED,
            "exact_deleted_count": f"round({BUDGET:.2f} * binder_length)",
            "explicit_deleted_indices_required": True,
        },
        "boltz": {
            "model": "boltz2",
            "sampling_steps": 50,
            "recycling_steps": 1,
            "diffusion_samples": 1,
            "same_config_for_ours_and_scisor": True,
            "reuse_full_length_predictions": True,
            "zero_fill": False,
        },
        "case_study_gate": {
            "ours_contact_recovery_minimum": CONTACT_THRESHOLD,
            "all_cases_and_negative_results_retained": True,
            "ranking_after_gate": ["contact_recovery_desc", "jaccard_desc", "iptm_desc", "pae_asc", "case_id_asc"],
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "proxy_boundary": PROXY_BOUNDARY,
    }


def prepare() -> None:
    require_status(BINDER, "PASS_DelePref_BINDER")
    require_status(V12, "PASS_DelePref_12_BINDER_INTERFACE_ANALYSIS")
    for path in (CHECKPOINT, CONFIG, SCISOR_CHECKPOINT, SCISOR_P0):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("protocol already exists")
    manifest = read_csv(BINDER / "binder_manifest.csv")
    cohort = []
    for row in manifest:
        chains = json.loads(row["binder_chain_ids_json"])
        length = int(row["sequence_length"])
        if len(chains) == 1 and length <= MAX_BINDER_LENGTH:
            cohort.append(row)
    cohort.sort(key=lambda row: int(row["case_id"]))
    if len(cohort) != EXPECTED_COHORT_CASES:
        raise RuntimeError(f"expected {EXPECTED_COHORT_CASES} cohort cases, observed {len(cohort)}")
    protocol = protocol_payload(cohort)
    write_json(OUT / "protocol.json", protocol)
    write_text(OUT / "protocol.sha256", sha256_file(OUT / "protocol.json") + "\n", encoding="ascii")
    cohort_rows = [
        {
            "case_id": row["case_id"],
            "protein_id": row["protein_id"],
            "wildtype_sequence": row["wildtype_sequence"],
            "sequence_length": row["sequence_length"],
            "binder_chain_ids_json": row["binder_chain_ids_json"],
            "target_chain_ids_json": row["target_chain_ids_json"],
            "budget": f"{BUDGET:.2f}",
        }
        for row in cohort
    ]
    fields = list(cohort_rows[0])
    write_csv(OUT / "cohort.csv", cohort_rows, fields)
    write_csv(
        OUT / "delepref_inputs.csv",
        [{"protein_id": row["protein_id"], "wildtype_sequence": row["wildtype_sequence"], "budget": f"{BUDGET:.2f}"} for row in cohort],
        ["protein_id", "wildtype_sequence", "budget"],
    )
    print(json.dumps({"status": "PASS_PREPARE", "cohort": len(cohort), "protocol_sha256": sha256_file(OUT / "protocol.json")}, sort_keys=True))


def patch_tokenizer(tokenizer: Any) -> None:
    values = {
        "_unk_token": "<unk>",
        "_cls_token": "<cls>",
        "_bos_token": "<cls>",
        "_eos_token": "<eos>",
        "_sep_token": "<eos>",
        "_pad_token": "<pad>",
        "_mask_token": "<mask>",
        "_additional_special_tokens": [],
    }
    for name, value in values.items():
        if not hasattr(tokenizer, name):
            setattr(tokenizer, name, value)


def untokenize(tokens: Any, tokenizer: Any) -> str:
    return tokenizer.decode(tokens).replace(" ", "").replace("<cls>", "").replace("<eos>", "").replace("<pad>", "")


def generate_scisor() -> None:
    import numpy as np
    import torch

    from external_scisor import load_shortening_scud

    ShorteningSCUD = load_shortening_scud()

    destination = OUT / "scisor_plans.csv"
    if destination.exists():
        raise FileExistsError(destination)
    random.seed(SCISOR_SEED)
    np.random.seed(SCISOR_SEED)
    torch.manual_seed(SCISOR_SEED)
    torch.cuda.manual_seed_all(SCISOR_SEED)
    device = torch.device("cuda")
    model = ShorteningSCUD.load_from_checkpoint(str(SCISOR_CHECKPOINT), map_location=device)
    model.to(device).eval()
    patch_tokenizer(model.tokenizer)
    model.p0 = torch.load(str(SCISOR_P0), map_location=device)
    rate = 1 / 1.1
    model.alpha = lambda t: (1 - t) ** rate
    model.beta = lambda t: rate / (1 - t)
    model.log_alpha = lambda t: -torch.log(model.alpha(t))
    rows = []
    with torch.no_grad():
        for item in read_csv(OUT / "cohort.csv"):
            sequence = item["wildtype_sequence"]
            ids = model.tokenizer(sequence).input_ids
            x = torch.tensor(ids, device=device).unsqueeze(0)
            count = strict_count(len(sequence))
            output_tokens, preserved = model.shrink_sequence(x, torch.tensor([count], device=device), temperature=1)
            preserved_wt = sorted(set(int(index) - 1 for index in preserved[0] if 0 < int(index) <= len(sequence)))
            deleted = sorted(set(range(len(sequence))) - set(preserved_wt))
            intervals = deleted_to_intervals(deleted)
            edited = untokenize(output_tokens[0], model.tokenizer)
            reconstructed = mutant(sequence, intervals)
            if len(deleted) != count or reconstructed != edited:
                raise RuntimeError(f"SCISOR reconstruction failed for {item['protein_id']}")
            rows.append(
                {
                    "case_id": item["case_id"],
                    "protein_id": item["protein_id"],
                    "budget": f"{BUDGET:.2f}",
                    "deleted_intervals": intervals_json(intervals),
                    "deleted_length": count,
                    "edited_sequence": edited,
                    "edited_sequence_sha256": sha256_text(edited),
                    "seed": SCISOR_SEED,
                }
            )
    write_csv(destination, rows, list(rows[0]))
    if len(rows) != EXPECTED_COHORT_CASES:
        raise RuntimeError(f"expected {EXPECTED_COHORT_CASES} SCISOR plans, observed {len(rows)}")
    print(json.dumps({"status": "PASS_SCISOR", "rows": len(rows)}, sort_keys=True))


def load_v12_module() -> Any:
    path = ROOT / "scripts/run_delepref_12_binder_interface_analysis.py"
    spec = importlib.util.spec_from_file_location("delepref_12_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load DelePref-12 module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def confidence_to_structure(path: Path) -> Path:
    name = path.name
    return path.with_name(name[len("confidence_") : -5] + ".cif")


def make_spec(v12: Any, case_id: str, protein_id: str, method: str, yaml_path: Path, confidence_path: Path) -> Any:
    return v12.PredictionSpec(
        case_id=case_id,
        protein_id=protein_id,
        method=method,
        comparison_class="SAME_PREDICTOR_CONFIG" if method != "wildtype_shared" else "WT_BASELINE",
        yaml_path=yaml_path,
        confidence_path=confidence_path,
        plan_hash="",
        edited_sequence_sha256="",
        prediction_status="ok",
    )


def yaml_with_edited_binder(source: Path, destination: Path, wildtype: str, edited: str) -> None:
    import yaml

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    sequences = payload.get("sequences") or []
    matching = [entry for entry in sequences if "protein" in entry and entry["protein"].get("sequence") == wildtype]
    if len(matching) != 1:
        raise ValueError(f"expected one exact binder-sequence match in WT YAML {source}, found {len(matching)}")
    matching[0]["protein"]["sequence"] = edited
    matching[0]["protein"]["msa"] = "empty"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())


def select_and_build() -> None:
    v12 = load_v12_module()
    cohort = {row["protein_id"]: row for row in read_csv(OUT / "cohort.csv")}
    scored = read_csv(OUT / "delepref_scored_candidates.csv")
    scisor = {row["protein_id"]: row for row in read_csv(OUT / "scisor_plans.csv")}
    details = read_csv(BINDER / "binder_prediction_details.csv")
    wt_details = {row["case_id"]: row for row in details if row["method"] == "delepref_full_control"}
    cache = read_csv(BINDER / "binder_boltz_cache_audit.csv")
    wt_cache = {row["case_id"]: row for row in cache if row["method"] == "wildtype_shared"}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scored:
        grouped[row["protein_id"]].append(row)
    if set(grouped) != set(cohort):
        raise RuntimeError("DelePref scored protein set does not equal frozen cohort")
    for protein_id, rows in grouped.items():
        if len(rows) != K:
            raise RuntimeError(f"expected K={K} for {protein_id}, observed {len(rows)}")
        if len({row["deletion_mask_sha256"] for row in rows}) != K:
            raise RuntimeError(f"duplicate DelePref masks for {protein_id}")
        for row in rows:
            intervals = parse_intervals(row["deleted_intervals"])
            if int(row["deleted_length"]) != strict_count(len(cohort[protein_id]["wildtype_sequence"])):
                raise RuntimeError(f"non-exact DelePref budget for {protein_id}")
            if mutant(cohort[protein_id]["wildtype_sequence"], intervals) == cohort[protein_id]["wildtype_sequence"]:
                raise RuntimeError(f"DelePref plan deletes no residues for {protein_id}")
    if set(scisor) != set(cohort):
        raise RuntimeError("SCISOR protein set does not equal frozen cohort")
    selected_rows = []
    jobs = []
    input_dir = OUT / "boltz_inputs"
    if input_dir.exists():
        raise FileExistsError(input_dir)
    for protein_id in sorted(cohort, key=lambda key: int(cohort[key]["case_id"])):
        item = cohort[protein_id]
        case_id = item["case_id"]
        wt_yaml = Path(wt_cache[case_id]["source_yaml_path"])
        wt_confidence = Path(wt_details[case_id]["wt_confidence_json"])
        wt_spec = make_spec(v12, case_id, protein_id, "wildtype_shared", wt_yaml, wt_confidence)
        wt = v12.load_prediction(wt_spec, canonical_target_sequences=None)
        if wt.geometry is None:
            raise RuntimeError(f"WT geometry missing for {protein_id}")
        candidates = []
        for row in grouped[protein_id]:
            intervals = parse_intervals(row["deleted_intervals"])
            deleted = {position for start, end in intervals for position in range(start, end + 1)}
            overlap = len(deleted & wt.geometry.binder_interface_residues)
            candidates.append((overlap, -float(row["scorer_score"]), row["deletion_mask_sha256"], row))
        overlap, _negative_score, _mask, ours = sorted(candidates)[0]
        ours_intervals = parse_intervals(ours["deleted_intervals"])
        ours_edited = mutant(item["wildtype_sequence"], ours_intervals)
        methods = [
            ("ours", ours_intervals, ours_edited, overlap, float(ours["scorer_score"]), ours["deletion_mask_sha256"]),
            ("scisor", parse_intervals(scisor[protein_id]["deleted_intervals"]), scisor[protein_id]["edited_sequence"], "", "", ""),
        ]
        for method, intervals, edited, interface_overlap, scorer_score, mask_sha in methods:
            deleted_length = sum(end - start + 1 for start, end in intervals)
            if deleted_length != strict_count(len(item["wildtype_sequence"])):
                raise RuntimeError(f"non-exact {method} budget for {protein_id}")
            if mutant(item["wildtype_sequence"], intervals) != edited:
                raise RuntimeError(f"{method} intervals do not reconstruct edited sequence for {protein_id}")
            edited_sha = sha256_text(edited)
            record_id = f"case{int(case_id):03d}_{RUN_TAG}_{method}_{edited_sha[:12]}"
            yaml_path = input_dir / f"{record_id}.yaml"
            yaml_with_edited_binder(wt_yaml, yaml_path, item["wildtype_sequence"], edited)
            row = {
                "case_id": case_id,
                "protein_id": protein_id,
                "method": method,
                "budget": f"{BUDGET:.2f}",
                "deleted_intervals": intervals_json(intervals),
                "deleted_length": deleted_length,
                "edited_sequence": edited,
                "edited_sequence_sha256": edited_sha,
                "wt_interface_deleted_count_preselection": interface_overlap,
                "scorer_score": scorer_score,
                "deletion_mask_sha256": mask_sha,
                "record_id": record_id,
                "yaml_path": str(yaml_path),
                "yaml_sha256": sha256_file(yaml_path),
                "wt_yaml_path": str(wt_yaml),
                "wt_confidence_json": str(wt_confidence),
            }
            selected_rows.append(row)
            jobs.append({"record_id": record_id, "yaml_path": str(yaml_path), "method": method, "case_id": case_id})
    write_csv(OUT / "selected_plans.csv", selected_rows, list(selected_rows[0]))
    write_csv(OUT / "boltz_jobs.csv", jobs, list(jobs[0]))
    if len(selected_rows) != 2 * EXPECTED_COHORT_CASES or len(jobs) != len(selected_rows):
        raise RuntimeError("selected plan/job count mismatch")
    print(json.dumps({"status": "PASS_SELECT_BUILD", "plans": len(selected_rows), "jobs": len(jobs)}, sort_keys=True))


def locate_confidence(record_id: str) -> Path:
    matches = sorted((OUT / "boltz_outputs").rglob(f"confidence_{record_id}_model_*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one confidence file for {record_id}, found {len(matches)}")
    return matches[0]


def analyze() -> None:
    v12 = load_v12_module()
    cohort = {row["protein_id"]: row for row in read_csv(OUT / "cohort.csv")}
    selected = read_csv(OUT / "selected_plans.csv")
    rows = []
    for plan in selected:
        item = cohort[plan["protein_id"]]
        case_id = plan["case_id"]
        wt_yaml = Path(plan["wt_yaml_path"])
        wt_confidence = Path(plan["wt_confidence_json"])
        wt_spec = make_spec(v12, case_id, plan["protein_id"], "wildtype_shared", wt_yaml, wt_confidence)
        wt = v12.load_prediction(wt_spec, canonical_target_sequences=None)
        targets = [entry["sequence"] for entry in wt.entities if entry["role"] == "target"]
        edited_confidence = locate_confidence(plan["record_id"])
        edited_spec = make_spec(v12, case_id, plan["protein_id"], plan["method"], Path(plan["yaml_path"]), edited_confidence)
        edited = v12.load_prediction(edited_spec, canonical_target_sequences=targets)
        if wt.geometry is None or edited.geometry is None:
            raise RuntimeError(f"geometry unavailable for {plan['record_id']}")
        intervals = parse_intervals(plan["deleted_intervals"])
        mapping, deleted = v12.delepref_residue_mapping(
            item["wildtype_sequence"], plan["edited_sequence"], intervals, int(plan["deleted_length"])
        )
        remapped = v12.remap_geometry(edited.geometry, mapping)
        intersection = wt.geometry.contacts & remapped.contacts
        union = wt.geometry.contacts | remapped.contacts
        confidence = load_json(edited_confidence)
        wt_conf = load_json(wt_confidence)
        overlap = deleted & wt.geometry.binder_interface_residues
        rows.append(
            {
                "case_id": case_id,
                "protein_id": plan["protein_id"],
                "method": plan["method"],
                "budget": f"{BUDGET:.2f}",
                "sequence_length": len(item["wildtype_sequence"]),
                "deleted_length": int(plan["deleted_length"]),
                "deleted_intervals": plan["deleted_intervals"],
                "contact_recovery": len(intersection) / len(wt.geometry.contacts) if wt.geometry.contacts else "",
                "contact_jaccard": len(intersection) / len(union) if union else "",
                "interface_residue_recovery": len(wt.geometry.binder_interface_residues & remapped.binder_interface_residues) / len(wt.geometry.binder_interface_residues) if wt.geometry.binder_interface_residues else "",
                "deleted_wt_interface_overlap_count": len(overlap),
                "wt_contact_count": len(wt.geometry.contacts),
                "edited_contact_count": len(remapped.contacts),
                "confidence": float(confidence["confidence_score"]),
                "wt_confidence": float(wt_conf["confidence_score"]),
                "delta_confidence": float(wt_conf["confidence_score"]) - float(confidence["confidence_score"]),
                "iptm": edited.confidence_metrics["interface_confidence"] if edited.confidence_metrics["interface_confidence"] is not None else "",
                "interchain_pae": edited.pae_metric if edited.pae_metric is not None else "",
                "steric_clash_count": edited.geometry.clash_count,
                "record_id": plan["record_id"],
                "confidence_json": str(edited_confidence),
                "structure_cif": str(confidence_to_structure(edited_confidence)),
            }
        )
    write_csv(OUT / "case_metrics.csv", rows, list(rows[0]))
    ours = [row for row in rows if row["method"] == "ours"]
    eligible = [row for row in ours if float(row["contact_recovery"]) >= CONTACT_THRESHOLD]
    eligible.sort(
        key=lambda row: (
            -float(row["contact_recovery"]),
            -float(row["contact_jaccard"]),
            -float(row["iptm"]) if row["iptm"] != "" else math.inf,
            float(row["interchain_pae"]) if row["interchain_pae"] != "" else math.inf,
            int(row["case_id"]),
        )
    )
    status = STATUS_FOUND if eligible else STATUS_NO_ELIGIBLE
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "cohort_cases": len(cohort),
        "edited_predictions": len(rows),
        "eligible_case_count": len(eligible),
        "eligible_case_ids": [int(row["case_id"]) for row in eligible],
        "best_case": eligible[0] if eligible else None,
        "protocol_sha256": sha256_file(OUT / "protocol.json"),
        "claim_boundary": CLAIM_BOUNDARY,
        "proxy_boundary": PROXY_BOUNDARY,
    }
    write_json(OUT / "analysis_summary.json", summary)
    lines = [
        f"# {ANALYSIS_ID} Binder {100 * BUDGET:.0f}% single-chain case search",
        "",
        f"Status: `{status}`",
        "",
        f"- Cohort cases: `{len(cohort)}`",
        f"- Edited predictions: `{len(rows)}`",
        f"- Ours contact-recovery >= 0.85: `{len(eligible)}`",
        f"- Eligible case IDs: `{[int(row['case_id']) for row in eligible]}`",
        "",
        PROXY_BOUNDARY,
        "",
        CLAIM_BOUNDARY,
        "",
    ]
    write_text(OUT / "SUMMARY.preview.md", "\n".join(lines))
    print(json.dumps({"status": status, "eligible": len(eligible), "best_case": eligible[0]["case_id"] if eligible else None}, sort_keys=True))


def validate() -> None:
    protocol = load_json(OUT / "protocol.json")
    expected_protocol_sha = (OUT / "protocol.sha256").read_text(encoding="ascii").strip()
    checks: dict[str, Any] = {}
    checks["protocol_sha256"] = sha256_file(OUT / "protocol.json") == expected_protocol_sha
    for key, path_key in (
        ("binder_manifest_sha256", "binder_manifest"),
        ("binder_prediction_details_sha256", "binder_prediction_details"),
        ("binder_boltz_cache_audit_sha256", "binder_boltz_cache_audit"),
        ("delepref_12_protocol_sha256", "delepref_12_protocol"),
    ):
        checks[key] = sha256_file(Path(protocol["frozen_inputs"][path_key])) == protocol["frozen_inputs"][key]

    cohort = {row["protein_id"]: row for row in read_csv(OUT / "cohort.csv")}
    raw = read_csv(OUT / "delepref_raw_candidates.csv")
    scored = read_csv(OUT / "delepref_scored_candidates.csv")
    scisor = read_csv(OUT / "scisor_plans.csv")
    selected = read_csv(OUT / "selected_plans.csv")
    metrics = read_csv(OUT / "case_metrics.csv")
    checks["cohort_count"] = len(cohort) == EXPECTED_COHORT_CASES
    checks["delepref_row_counts"] = len(raw) == len(scored) == EXPECTED_COHORT_CASES * K
    checks["scisor_count"] = len(scisor) == EXPECTED_COHORT_CASES
    checks["selected_metric_counts"] = len(selected) == len(metrics) == EXPECTED_COHORT_CASES * 2

    candidate_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scored:
        candidate_groups[row["protein_id"]].append(row)
    candidate_ok = set(candidate_groups) == set(cohort)
    for protein_id, group in candidate_groups.items():
        sequence = cohort[protein_id]["wildtype_sequence"]
        expected_deleted = strict_count(len(sequence))
        candidate_ok &= len(group) == K and len({row["deletion_mask_sha256"] for row in group}) == K
        for row in group:
            intervals = parse_intervals(row["deleted_intervals"])
            candidate_ok &= int(row["deleted_length"]) == expected_deleted
            candidate_ok &= len(sequence) - len(mutant(sequence, intervals)) == expected_deleted
    checks["delepref_k8_unique_exact_budget"] = bool(candidate_ok)

    selected_by_key = {(row["protein_id"], row["method"]): row for row in selected}
    metrics_by_key = {(row["protein_id"], row["method"]): row for row in metrics}
    plan_ok = len(selected_by_key) == len(selected) and len(metrics_by_key) == len(metrics)
    sidecar_ok = True
    metric_ok = True
    for key, plan in selected_by_key.items():
        sequence = cohort[plan["protein_id"]]["wildtype_sequence"]
        intervals = parse_intervals(plan["deleted_intervals"])
        plan_ok &= mutant(sequence, intervals) == plan["edited_sequence"]
        plan_ok &= int(plan["deleted_length"]) == strict_count(len(sequence))
        confidence_path = locate_confidence(plan["record_id"])
        for path in (confidence_path, confidence_to_structure(confidence_path)):
            sidecar_ok &= path.is_file() and path.stat().st_size > 0
        row = metrics_by_key[key]
        for field in ("contact_recovery", "contact_jaccard", "interface_residue_recovery"):
            value = float(row[field])
            metric_ok &= math.isfinite(value) and 0.0 <= value <= 1.0
        for field in ("confidence", "wt_confidence", "delta_confidence"):
            metric_ok &= math.isfinite(float(row[field]))
    checks["plan_identity_and_exact_budget"] = bool(plan_ok)
    checks["prediction_sidecars_present"] = bool(sidecar_ok)
    checks["metric_ranges_finite"] = bool(metric_ok)

    analysis = load_json(OUT / "analysis_summary.json")
    ours = [row for row in metrics if row["method"] == "ours"]
    eligible = sorted(
        [row for row in ours if float(row["contact_recovery"]) >= CONTACT_THRESHOLD],
        key=lambda row: (
            -float(row["contact_recovery"]),
            -float(row["contact_jaccard"]),
            -float(row["iptm"]) if row["iptm"] else math.inf,
            float(row["interchain_pae"]) if row["interchain_pae"] else math.inf,
            int(row["case_id"]),
        ),
    )
    checks["eligible_selection_recomputed"] = (
        analysis["eligible_case_count"] == len(eligible)
        and analysis["eligible_case_ids"] == [int(row["case_id"]) for row in eligible]
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        write_json(OUT / "independent_validation.json", {"status": "FAIL", "checks": checks, "failed": failed})
        raise RuntimeError(f"final validation failed: {failed}")

    final_status = analysis["status"]
    validation = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "PASS",
        "checks": checks,
        "failed": [],
        "validated_case_count": len(cohort),
        "validated_prediction_count": len(metrics),
    }
    write_json(OUT / "independent_validation.json", validation)
    write_json(OUT / "run_summary.json", {**analysis, "independent_validation": "PASS"})
    write_text(OUT / "SUMMARY.md", (OUT / "SUMMARY.preview.md").read_text(encoding="utf-8"))

    manifest_rows = []
    excluded = {"audit_manifest.json", STATUS_FOUND, STATUS_NO_ELIGIBLE}
    for path in sorted(item for item in OUT.rglob("*") if item.is_file()):
        relative = path.relative_to(OUT).as_posix()
        if relative in excluded or relative.startswith("audit/"):
            continue
        manifest_rows.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(OUT / "audit_manifest.json", {"schema_version": MANIFEST_SCHEMA_VERSION, "artifacts": manifest_rows})
    sentinel = STATUS_FOUND if eligible else STATUS_NO_ELIGIBLE
    write_text(OUT / sentinel, final_status + "\n", encoding="ascii")
    print(json.dumps({"status": final_status, "validation": "PASS", "artifacts": len(manifest_rows)}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "generate-scisor", "select-build", "analyze", "validate"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--budget", type=float, default=None)
    parser.add_argument("--analysis-id", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--protocol-schema-version", default=None)
    parser.add_argument("--summary-schema-version", default=None)
    parser.add_argument("--validation-schema-version", default=None)
    parser.add_argument("--manifest-schema-version", default=None)
    parser.add_argument("--status-found", default=None)
    parser.add_argument("--status-no-eligible", default=None)
    args = parser.parse_args()
    global OUT, BUDGET, ANALYSIS_ID, RUN_TAG
    global PROTOCOL_SCHEMA_VERSION, SUMMARY_SCHEMA_VERSION
    global VALIDATION_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION
    global STATUS_FOUND, STATUS_NO_ELIGIBLE
    if args.output_dir is not None:
        OUT = args.output_dir
    if args.budget is not None:
        if not 0 < args.budget < 1:
            parser.error("--budget must be between 0 and 1")
        BUDGET = args.budget
    if args.analysis_id is not None:
        ANALYSIS_ID = args.analysis_id
    if args.run_tag is not None:
        RUN_TAG = args.run_tag
    if args.protocol_schema_version is not None:
        PROTOCOL_SCHEMA_VERSION = args.protocol_schema_version
    if args.summary_schema_version is not None:
        SUMMARY_SCHEMA_VERSION = args.summary_schema_version
    if args.validation_schema_version is not None:
        VALIDATION_SCHEMA_VERSION = args.validation_schema_version
    if args.manifest_schema_version is not None:
        MANIFEST_SCHEMA_VERSION = args.manifest_schema_version
    if args.status_found is not None:
        STATUS_FOUND = args.status_found
    if args.status_no_eligible is not None:
        STATUS_NO_ELIGIBLE = args.status_no_eligible
    {"prepare": prepare, "generate-scisor": generate_scisor, "select-build": select_and_build, "analyze": analyze, "validate": validate}[args.phase]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
