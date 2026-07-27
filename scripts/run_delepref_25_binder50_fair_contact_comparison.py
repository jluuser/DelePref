#!/usr/bin/env python3
"""Run the fresh 50-case DelePref-versus-external-SCISOR Binder comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/experiments/delepref_step004000_delepref_25_binder50_fair_contact_comparison"
BINDER = ROOT / "results/experiments/delepref_step004000_binder"
V12 = ROOT / "results/experiments/delepref_step004000_delepref_12_binder_interface_analysis"
RUNTIME_PATH = ROOT / "scripts/run_delepref_22_binder_budget_case_search.py"
CHECKPOINT = ROOT / "results/experiments/delepref_evaluation_checkpoint/final_freeze/delepref_evaluation_step_00004000.pt"
CONFIG = ROOT / "configs/deletion_preference_student.yaml"
SCISOR_CHECKPOINT = Path(
    os.environ.get("DELEPREF_SCISOR_CHECKPOINT", "models/scisor/SCISOR_U90_S.ckpt")
).expanduser()
SCISOR_P0 = Path(os.environ.get("DELEPREF_SCISOR_P0", str(ROOT / "p0.pt"))).expanduser()
BUDGETS = (0.10, 0.20)
BUDGET_LABELS = ("0.10", "0.20")
METHODS = ("delepref", "scisor")
EXPECTED_CASES = 50
K = 8
DELEPREF_GENERATION_SEED = 13
SCISOR_BASE_SEED = 20260726
BOLTZ_SEED = 20260726
BOOTSTRAP_SEED = 20260726
BOOTSTRAP_REPLICATES = 20_000
CONTACT_CUTOFF_ANGSTROM = 5.0
STATUS = "PASS_DelePref_25_BINDER50_FAIR_CONTACT_COMPARISON"
PRIMARY_METRICS = ("contact_recovery", "contact_jaccard")
SUMMARY_METRICS = {
    "contact_recovery": "higher_is_better",
    "contact_jaccard": "higher_is_better",
    "interface_residue_recovery": "higher_is_better",
    "interface_confidence": "higher_is_better",
    "interchain_pae": "lower_is_better",
    "delta_confidence": "lower_is_better",
    "steric_clash_count": "lower_is_better",
}
CLAIM_BOUNDARY = (
    "The method ranks candidate deletion plans by teacher-derived deletion "
    "perturbation preference and optional auxiliary risk evidence."
)
PROXY_BOUNDARY = (
    "All Binder results are post-deletion Boltz-2 complex-prediction proxies; "
    "they do not establish experimental binding, Kd/Ki, or functional preservation."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def load_runtime() -> Any:
    spec = importlib.util.spec_from_file_location("delepref_22_runtime_for_delepref_25", RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Binder runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_status(root: Path, expected: str) -> None:
    observed = load_json(root / "run_summary.json").get("status")
    if observed != expected:
        raise RuntimeError(f"status mismatch for {root}: {observed!r}")


def budget_label(value: Any) -> str:
    return f"{float(value):.2f}"


def strict_count(length: int, budget: float) -> int:
    return max(1, min(length - 1, int(round(budget * length))))


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


def intervals_json(intervals: Sequence[tuple[int, int]]) -> str:
    return json.dumps([[int(start), int(end)] for start, end in intervals], separators=(",", ":"))


def condition_seed(case_id: str, budget: str) -> int:
    payload = f"{SCISOR_BASE_SEED}|{case_id}|{budget}".encode("ascii")
    return SCISOR_BASE_SEED + int(hashlib.sha256(payload).hexdigest()[:8], 16)


def prepare() -> None:
    require_status(BINDER, "PASS_DelePref_BINDER")
    require_status(V12, "PASS_DelePref_12_BINDER_INTERFACE_ANALYSIS")
    for path in (CHECKPOINT, CONFIG, SCISOR_CHECKPOINT, SCISOR_P0):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (OUT / "protocol.json").exists():
        raise FileExistsError(OUT / "protocol.json")

    import yaml

    manifest = sorted(read_csv(BINDER / "binder_manifest.csv"), key=lambda row: int(row["case_id"]))
    cache = read_csv(BINDER / "binder_boltz_cache_audit.csv")
    wt_cache = {row["case_id"]: row for row in cache if row["method"] == "wildtype_shared"}
    if len(manifest) != EXPECTED_CASES or len(wt_cache) != EXPECTED_CASES:
        raise RuntimeError("formal Binder-50 inputs are incomplete")

    cohort_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    yaml_audit: list[dict[str, Any]] = []
    for row in manifest:
        source_yaml = Path(wt_cache[row["case_id"]]["source_yaml_path"])
        payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
        entities = [entry["protein"] for entry in payload.get("sequences") or []]
        binder_matches = [entity for entity in entities if entity.get("sequence") == row["wildtype_sequence"]]
        if len(binder_matches) != 1:
            raise RuntimeError(f"WT YAML has {len(binder_matches)} binder matches for case {row['case_id']}")
        if any(entity.get("msa") != "empty" for entity in entities):
            raise RuntimeError(f"WT YAML does not use empty MSA for case {row['case_id']}")
        cohort_rows.append(
            {
                "case_id": row["case_id"],
                "protein_id": row["protein_id"],
                "source": row["source"],
                "pdb_id": row["pdb_id"],
                "wildtype_sequence": row["wildtype_sequence"],
                "wildtype_sequence_sha256": row["wildtype_sequence_sha256"],
                "sequence_length": row["sequence_length"],
                "binder_chain_ids_json": row["binder_chain_ids_json"],
                "target_chain_ids_json": row["target_chain_ids_json"],
                "wt_source_yaml": str(source_yaml),
                "wt_source_yaml_sha256": sha256_file(source_yaml),
            }
        )
        yaml_audit.append(
            {
                "case_id": row["case_id"],
                "protein_id": row["protein_id"],
                "source_yaml": str(source_yaml),
                "source_yaml_sha256": sha256_file(source_yaml),
                "entity_count": len(entities),
                "binder_match_count": len(binder_matches),
                "binder_input_entity_id": binder_matches[0]["id"],
                "all_entities_empty_msa": True,
            }
        )
        for budget in BUDGETS:
            input_rows.append(
                {
                    "protein_id": row["protein_id"],
                    "wildtype_sequence": row["wildtype_sequence"],
                    "budget": f"{budget:.2f}",
                }
            )

    protocol = {
        "schema_version": "delepref_25_binder50_fair_contact_protocol_v1",
        "analysis_id": "DelePref-25",
        "cohort": {
            "name": "formal Binder-50 benchmark",
            "case_count": EXPECTED_CASES,
            "case_ids": [int(row["case_id"]) for row in cohort_rows],
            "composition": {"AB-Bind": 19, "SAbDab": 31},
            "selection": "all formal cases; no length, chain-count, or outcome filtering",
            "benchmark_binder_representation": "one concatenated binder entity per frozen WT YAML",
        },
        "budgets": [0.10, 0.20],
        "methods": {
            "delepref": {
                "checkpoint": str(CHECKPOINT),
                "checkpoint_sha256": sha256_file(CHECKPOINT),
                "config": str(CONFIG),
                "config_sha256": sha256_file(CONFIG),
                "generation_seed": DELEPREF_GENERATION_SEED,
                "candidate_count": K,
                "selection": "frozen canonical scorer top-1; no WT-interface or Boltz-metric preselection",
            },
            "scisor": {
                "checkpoint": str(SCISOR_CHECKPOINT),
                "checkpoint_sha256": sha256_file(SCISOR_CHECKPOINT),
                "p0": str(SCISOR_P0),
                "p0_sha256": sha256_file(SCISOR_P0),
                "base_seed": SCISOR_BASE_SEED,
                "candidate_count": 1,
                "selection": "single native plan with explicit preserved/deleted indices",
            },
            "comparison_scope": "native end-to-end plan comparison; not equal-compute or equal-K",
        },
        "prediction_protocol": {
            "fresh_predictions": 250,
            "wildtype_predictions": 50,
            "edited_predictions": 200,
            "reuse_old_predictions": False,
            "model": "boltz2",
            "seed": BOLTZ_SEED,
            "sampling_steps": 50,
            "recycling_steps": 1,
            "diffusion_samples": 1,
            "num_workers": 0,
            "empty_msa_for_all_proteins": True,
            "input_order": "case_id, then WT, B10 DelePref, B10 SCISOR, B20 DelePref, B20 SCISOR",
            "zero_fill": False,
        },
        "metrics": {
            "contact_cutoff_angstrom": CONTACT_CUTOFF_ANGSTROM,
            "contact_recovery": "|C_WT intersect C_edited| / |C_WT|",
            "contact_jaccard": "|C_WT intersect C_edited| / |C_WT union C_edited|",
            "edited_contacts_are_remapped_to_wt_binder_coordinates": True,
        },
        "statistics": {
            "unit": "Binder case",
            "paired_bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ci": "percentile 95%",
            "budgets_analyzed_separately": True,
        },
        "frozen_inputs": {
            "binder_manifest": str(BINDER / "binder_manifest.csv"),
            "binder_manifest_sha256": sha256_file(BINDER / "binder_manifest.csv"),
            "binder_boltz_cache_audit": str(BINDER / "binder_boltz_cache_audit.csv"),
            "binder_boltz_cache_audit_sha256": sha256_file(BINDER / "binder_boltz_cache_audit.csv"),
            "delepref_12_protocol": str(V12 / "protocol.json"),
            "delepref_12_protocol_sha256": sha256_file(V12 / "protocol.json"),
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "proxy_boundary": PROXY_BOUNDARY,
    }
    write_json(OUT / "protocol.json", protocol)
    write_text(OUT / "protocol.sha256", sha256_file(OUT / "protocol.json") + "\n", encoding="ascii")
    write_csv(OUT / "cohort.csv", cohort_rows, list(cohort_rows[0]))
    write_csv(OUT / "wt_yaml_semantic_audit.csv", yaml_audit, list(yaml_audit[0]))
    write_csv(OUT / "delepref_inputs.csv", input_rows, ["protein_id", "wildtype_sequence", "budget"])
    print(json.dumps({"status": "PASS_PREPARE", "cases": len(cohort_rows), "conditions": len(input_rows), "protocol_sha256": sha256_file(OUT / "protocol.json")}, sort_keys=True))


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
    import torch

    from external_scisor import load_shortening_scud

    ShorteningSCUD = load_shortening_scud()

    destination = OUT / "scisor_plans.csv"
    if destination.exists():
        raise FileExistsError(destination)
    device = torch.device("cuda")
    model = ShorteningSCUD.load_from_checkpoint(str(SCISOR_CHECKPOINT), map_location=device)
    model.to(device).eval()
    patch_tokenizer(model.tokenizer)
    model.p0 = torch.load(str(SCISOR_P0), map_location=device)
    rate = 1 / 1.1
    model.alpha = lambda t: (1 - t) ** rate
    model.beta = lambda t: rate / (1 - t)
    model.log_alpha = lambda t: -torch.log(model.alpha(t))
    runtime = load_runtime()

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in read_csv(OUT / "cohort.csv"):
            sequence = item["wildtype_sequence"]
            ids = model.tokenizer(sequence).input_ids
            x = torch.tensor(ids, device=device).unsqueeze(0)
            for budget in BUDGETS:
                label = f"{budget:.2f}"
                seed = condition_seed(item["case_id"], label)
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                count = strict_count(len(sequence), budget)
                output_tokens, preserved = model.shrink_sequence(
                    x, torch.tensor([count], device=device), temperature=1
                )
                preserved_wt = sorted(
                    set(int(index) - 1 for index in preserved[0] if 0 < int(index) <= len(sequence))
                )
                deleted = sorted(set(range(len(sequence))) - set(preserved_wt))
                intervals = deleted_to_intervals(deleted)
                edited = untokenize(output_tokens[0], model.tokenizer)
                reconstructed = runtime.mutant(sequence, intervals)
                if len(deleted) != count or reconstructed != edited:
                    raise RuntimeError(f"SCISOR reconstruction failed for {item['protein_id']} at {label}")
                rows.append(
                    {
                        "case_id": item["case_id"],
                        "protein_id": item["protein_id"],
                        "budget": label,
                        "condition_seed": seed,
                        "deleted_intervals": intervals_json(intervals),
                        "deleted_length": count,
                        "span_count": len(intervals),
                        "edited_sequence": edited,
                        "edited_sequence_sha256": sha256_text(edited),
                    }
                )
    if len(rows) != EXPECTED_CASES * len(BUDGETS):
        raise RuntimeError(f"expected 100 SCISOR plans, observed {len(rows)}")
    write_csv(destination, rows, list(rows[0]))
    print(json.dumps({"status": "PASS_SCISOR", "plans": len(rows)}, sort_keys=True))


def write_wt_yaml(source: Path, destination: Path) -> None:
    import yaml

    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
        handle.flush()
        os.fsync(handle.fileno())


def select_and_build() -> None:
    runtime = load_runtime()
    cohort = {row["protein_id"]: row for row in read_csv(OUT / "cohort.csv")}
    canonical_rows = read_csv(OUT / "delepref_canonical_scorer_top1.csv")
    canonical = {(row["protein_id"], budget_label(row["budget"])): row for row in canonical_rows}
    scisor_rows = read_csv(OUT / "scisor_plans.csv")
    scisor = {(row["protein_id"], row["budget"]): row for row in scisor_rows}
    expected_keys = {(protein_id, label) for protein_id in cohort for label in BUDGET_LABELS}
    if set(canonical) != expected_keys or set(scisor) != expected_keys:
        raise RuntimeError("canonical DelePref or SCISOR plan keys do not match the 50x2 design")

    input_dir = OUT / "boltz_inputs"
    if input_dir.exists():
        raise FileExistsError(input_dir)
    rows: list[dict[str, Any]] = []
    for protein_id in sorted(cohort, key=lambda key: int(cohort[key]["case_id"])):
        item = cohort[protein_id]
        case_id = int(item["case_id"])
        source_yaml = Path(item["wt_source_yaml"])
        wt_record_id = f"case{case_id:03d}_00_wt"
        wt_yaml = input_dir / f"{wt_record_id}.yaml"
        write_wt_yaml(source_yaml, wt_yaml)
        rows.append(
            {
                "case_id": case_id,
                "protein_id": protein_id,
                "budget": "",
                "method": "wt",
                "record_order": 0,
                "record_id": wt_record_id,
                "wildtype_sequence": item["wildtype_sequence"],
                "edited_sequence": item["wildtype_sequence"],
                "edited_sequence_sha256": item["wildtype_sequence_sha256"],
                "deleted_intervals": "[]",
                "deleted_length": 0,
                "span_count": 0,
                "deletion_mask_sha256": "",
                "selector_score": "",
                "generation_seed": "",
                "yaml_path": str(wt_yaml),
                "yaml_sha256": sha256_file(wt_yaml),
            }
        )
        for budget_index, label in enumerate(BUDGET_LABELS, start=1):
            expected_deleted = strict_count(len(item["wildtype_sequence"]), float(label))
            delepref = canonical[(protein_id, label)]
            scisor_plan = scisor[(protein_id, label)]
            method_plans = (
                (
                    "delepref",
                    runtime.parse_intervals(delepref["deleted_intervals"]),
                    delepref["deletion_mask_sha256"],
                    delepref["scorer_score"],
                    delepref["generation_seed"],
                ),
                (
                    "scisor",
                    runtime.parse_intervals(scisor_plan["deleted_intervals"]),
                    "",
                    "",
                    scisor_plan["condition_seed"],
                ),
            )
            for method_index, (method, intervals, mask_sha, selector_score, generation_seed) in enumerate(method_plans, start=1):
                edited = runtime.mutant(item["wildtype_sequence"], intervals)
                deleted_length = sum(end - start + 1 for start, end in intervals)
                if deleted_length != expected_deleted:
                    raise RuntimeError(f"non-exact {method} plan for {protein_id} at {label}")
                if method == "scisor" and edited != scisor_plan["edited_sequence"]:
                    raise RuntimeError(f"SCISOR edited-sequence mismatch for {protein_id} at {label}")
                edited_sha = sha256_text(edited)
                order = budget_index * 10 + method_index
                record_id = f"case{case_id:03d}_{order:02d}_b{int(100 * float(label)):02d}_{method}_{edited_sha[:12]}"
                yaml_path = input_dir / f"{record_id}.yaml"
                runtime.yaml_with_edited_binder(source_yaml, yaml_path, item["wildtype_sequence"], edited)
                rows.append(
                    {
                        "case_id": case_id,
                        "protein_id": protein_id,
                        "budget": label,
                        "method": method,
                        "record_order": order,
                        "record_id": record_id,
                        "wildtype_sequence": item["wildtype_sequence"],
                        "edited_sequence": edited,
                        "edited_sequence_sha256": edited_sha,
                        "deleted_intervals": intervals_json(intervals),
                        "deleted_length": deleted_length,
                        "span_count": len(intervals),
                        "deletion_mask_sha256": mask_sha,
                        "selector_score": selector_score,
                        "generation_seed": generation_seed,
                        "yaml_path": str(yaml_path),
                        "yaml_sha256": sha256_file(yaml_path),
                    }
                )
    rows.sort(key=lambda row: (int(row["case_id"]), int(row["record_order"])))
    if len(rows) != EXPECTED_CASES * 5:
        raise RuntimeError(f"expected 250 prediction inputs, observed {len(rows)}")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate prediction record IDs")
    write_csv(OUT / "prediction_plan_manifest.csv", rows, list(rows[0]))
    print(json.dumps({"status": "PASS_SELECT_BUILD", "prediction_inputs": len(rows), "edited_inputs": len(rows) - EXPECTED_CASES}, sort_keys=True))


def locate_confidence(record_id: str) -> Path:
    matches = sorted((OUT / "boltz_outputs").rglob(f"confidence_{record_id}_model_*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one confidence file for {record_id}, observed {len(matches)}")
    return matches[0]


def bootstrap_ci(values: Sequence[float], key: str) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    key_seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(BOOTSTRAP_SEED + key_seed)
    indices = rng.integers(0, vector.size, size=(BOOTSTRAP_REPLICATES, vector.size))
    means = vector[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(vector.mean()), float(low), float(high)


def analyze() -> None:
    runtime = load_runtime()
    v12 = runtime.load_v12_module()
    plans = read_csv(OUT / "prediction_plan_manifest.csv")
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in plans:
        by_case[row["case_id"]].append(row)
    output: list[dict[str, Any]] = []
    wt_audit: list[dict[str, Any]] = []

    for case_id in sorted(by_case, key=int):
        case_rows = by_case[case_id]
        wt_plan = next(row for row in case_rows if row["method"] == "wt")
        wt_confidence = locate_confidence(wt_plan["record_id"])
        wt_spec = runtime.make_spec(
            v12, case_id, wt_plan["protein_id"], "wildtype_shared", Path(wt_plan["yaml_path"]), wt_confidence
        )
        wt = v12.load_prediction(wt_spec, canonical_target_sequences=None)
        if wt.geometry is None:
            raise RuntimeError(f"WT geometry unavailable for case {case_id}")
        targets = [entry["sequence"] for entry in wt.entities if entry["role"] == "target"]
        wt_conf = load_json(wt_confidence)
        wt_audit.append(
            {
                "case_id": case_id,
                "protein_id": wt_plan["protein_id"],
                "wt_contact_count": len(wt.geometry.contacts),
                "wt_binder_interface_residue_count": len(wt.geometry.binder_interface_residues),
                "wt_target_interface_residue_count": len(wt.geometry.target_interface_residues),
                "wt_confidence": float(wt_conf["confidence_score"]),
                "confidence_json": str(wt_confidence),
                "structure_cif": str(runtime.confidence_to_structure(wt_confidence)),
            }
        )
        if not wt.geometry.contacts:
            raise RuntimeError(f"WT prediction has zero binder-target contacts for case {case_id}")
        for plan in sorted((row for row in case_rows if row["method"] != "wt"), key=lambda row: int(row["record_order"])):
            edited_confidence = locate_confidence(plan["record_id"])
            edited_spec = runtime.make_spec(
                v12,
                case_id,
                plan["protein_id"],
                plan["method"],
                Path(plan["yaml_path"]),
                edited_confidence,
            )
            edited = v12.load_prediction(edited_spec, canonical_target_sequences=targets)
            if edited.geometry is None:
                raise RuntimeError(f"edited geometry unavailable for {plan['record_id']}")
            mapping, deleted = v12.delepref_residue_mapping(
                plan["wildtype_sequence"],
                plan["edited_sequence"],
                runtime.parse_intervals(plan["deleted_intervals"]),
                int(plan["deleted_length"]),
            )
            remapped = v12.remap_geometry(edited.geometry, mapping)
            intersection = wt.geometry.contacts & remapped.contacts
            union = wt.geometry.contacts | remapped.contacts
            edited_conf = load_json(edited_confidence)
            overlap = deleted & wt.geometry.binder_interface_residues
            output.append(
                {
                    "case_id": case_id,
                    "protein_id": plan["protein_id"],
                    "budget": plan["budget"],
                    "method": plan["method"],
                    "deleted_length": plan["deleted_length"],
                    "span_count": plan["span_count"],
                    "deleted_intervals": plan["deleted_intervals"],
                    "wt_contact_count": len(wt.geometry.contacts),
                    "edited_contact_count": len(remapped.contacts),
                    "contact_recovery": len(intersection) / len(wt.geometry.contacts),
                    "contact_jaccard": len(intersection) / len(union) if union else 0.0,
                    "interface_residue_recovery": len(wt.geometry.binder_interface_residues & remapped.binder_interface_residues) / len(wt.geometry.binder_interface_residues) if wt.geometry.binder_interface_residues else "",
                    "deleted_wt_interface_overlap_count": len(overlap),
                    "deleted_wt_interface_overlap_fraction": len(overlap) / len(wt.geometry.binder_interface_residues) if wt.geometry.binder_interface_residues else "",
                    "confidence": float(edited_conf["confidence_score"]),
                    "wt_confidence": float(wt_conf["confidence_score"]),
                    "delta_confidence": float(wt_conf["confidence_score"]) - float(edited_conf["confidence_score"]),
                    "interface_confidence": edited.confidence_metrics["interface_confidence"] if edited.confidence_metrics["interface_confidence"] is not None else "",
                    "interchain_pae": edited.pae_metric if edited.pae_metric is not None else "",
                    "steric_clash_count": edited.geometry.clash_count,
                    "record_id": plan["record_id"],
                    "confidence_json": str(edited_confidence),
                    "structure_cif": str(runtime.confidence_to_structure(edited_confidence)),
                }
            )
    output.sort(key=lambda row: (float(row["budget"]), int(row["case_id"]), row["method"]))
    if len(output) != EXPECTED_CASES * len(BUDGETS) * len(METHODS):
        raise RuntimeError(f"expected 200 metric rows, observed {len(output)}")
    write_csv(OUT / "binder50_case_metrics.csv", output, list(output[0]))
    write_csv(OUT / "binder50_wt_reference_audit.csv", wt_audit, list(wt_audit[0]))

    summaries: list[dict[str, Any]] = []
    thresholds: list[dict[str, Any]] = []
    for budget in BUDGET_LABELS:
        for method in METHODS:
            subset = [row for row in output if row["budget"] == budget and row["method"] == method]
            for metric, direction in SUMMARY_METRICS.items():
                values = [float(row[metric]) for row in subset if row[metric] != ""]
                summaries.append(
                    {
                        "budget": budget,
                        "method": method,
                        "metric": metric,
                        "direction": direction,
                        "n_expected": EXPECTED_CASES,
                        "n_available": len(values),
                        "coverage": len(values) / EXPECTED_CASES,
                        "mean": statistics.mean(values) if values else "",
                        "median": statistics.median(values) if values else "",
                        "sample_sd": statistics.stdev(values) if len(values) > 1 else "",
                        "minimum": min(values) if values else "",
                        "maximum": max(values) if values else "",
                    }
                )
            for metric in PRIMARY_METRICS:
                values = [float(row[metric]) for row in subset]
                for threshold in (0.25, 0.50, 0.60, 0.80):
                    count = sum(value >= threshold for value in values)
                    thresholds.append(
                        {
                            "budget": budget,
                            "method": method,
                            "metric": metric,
                            "threshold": threshold,
                            "count": count,
                            "n": len(values),
                            "fraction": count / len(values),
                        }
                    )
    write_csv(OUT / "binder50_method_summary.csv", summaries, list(summaries[0]))
    write_csv(OUT / "binder50_threshold_summary.csv", thresholds, list(thresholds[0]))

    by_key = {(row["budget"], row["method"], row["case_id"]): row for row in output}
    case_ids = sorted({row["case_id"] for row in output}, key=int)
    ci_rows: list[dict[str, Any]] = []
    for budget in BUDGET_LABELS:
        for metric in SUMMARY_METRICS:
            paired_ids = [
                case_id
                for case_id in case_ids
                if by_key[(budget, "delepref", case_id)][metric] != ""
                and by_key[(budget, "scisor", case_id)][metric] != ""
            ]
            differences = [
                float(by_key[(budget, "delepref", case_id)][metric])
                - float(by_key[(budget, "scisor", case_id)][metric])
                for case_id in paired_ids
            ]
            point, low, high = bootstrap_ci(differences, f"delepref_minus_scisor|{budget}|{metric}")
            direction = SUMMARY_METRICS[metric]
            oriented = -point if direction == "lower_is_better" else point
            oriented_low = -high if direction == "lower_is_better" else low
            oriented_high = -low if direction == "lower_is_better" else high
            ci_rows.append(
                {
                    "contrast": "delepref_minus_scisor",
                    "budget": budget,
                    "metric": metric,
                    "direction": direction,
                    "n_pairs_expected": EXPECTED_CASES,
                    "n_pairs_available": len(paired_ids),
                    "point_estimate": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "oriented_effect_positive_favors_delepref": oriented,
                    "oriented_ci95_low": oriented_low,
                    "oriented_ci95_high": oriented_high,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                }
            )
    write_csv(OUT / "binder50_paired_bootstrap_ci.csv", ci_rows, list(ci_rows[0]))
    print(json.dumps({"status": "PASS_ANALYZE", "case_metric_rows": len(output), "bootstrap_rows": len(ci_rows)}, sort_keys=True))


def validate() -> None:
    runtime = load_runtime()
    protocol = load_json(OUT / "protocol.json")
    checks: dict[str, bool] = {}
    checks["protocol_sha256"] = sha256_file(OUT / "protocol.json") == (OUT / "protocol.sha256").read_text(encoding="ascii").strip()
    for key in ("binder_manifest", "binder_boltz_cache_audit", "delepref_12_protocol"):
        path = Path(protocol["frozen_inputs"][key])
        checks[f"source_{key}"] = sha256_file(path) == protocol["frozen_inputs"][f"{key}_sha256"]
    checks["checkpoint_sha256"] = sha256_file(CHECKPOINT) == protocol["methods"]["delepref"]["checkpoint_sha256"]
    checks["config_sha256"] = sha256_file(CONFIG) == protocol["methods"]["delepref"]["config_sha256"]
    checks["scisor_checkpoint_sha256"] = sha256_file(SCISOR_CHECKPOINT) == protocol["methods"]["scisor"]["checkpoint_sha256"]
    checks["scisor_p0_sha256"] = sha256_file(SCISOR_P0) == protocol["methods"]["scisor"]["p0_sha256"]

    cohort = read_csv(OUT / "cohort.csv")
    raw = read_csv(OUT / "delepref_raw_candidates.csv")
    scored = read_csv(OUT / "delepref_scored_candidates.csv")
    canonical = read_csv(OUT / "delepref_canonical_scorer_top1.csv")
    scisor = read_csv(OUT / "scisor_plans.csv")
    plans = read_csv(OUT / "prediction_plan_manifest.csv")
    metrics = read_csv(OUT / "binder50_case_metrics.csv")
    summaries = read_csv(OUT / "binder50_method_summary.csv")
    cis = read_csv(OUT / "binder50_paired_bootstrap_ci.csv")
    checks["cohort_count"] = len(cohort) == EXPECTED_CASES
    checks["delepref_candidate_counts"] = len(raw) == len(scored) == EXPECTED_CASES * len(BUDGETS) * K
    checks["canonical_count"] = len(canonical) == EXPECTED_CASES * len(BUDGETS)
    checks["scisor_count"] = len(scisor) == EXPECTED_CASES * len(BUDGETS)
    checks["prediction_plan_count"] = len(plans) == EXPECTED_CASES * 5
    checks["metric_count"] = len(metrics) == EXPECTED_CASES * len(BUDGETS) * len(METHODS)
    checks["summary_count"] = len(summaries) == len(BUDGETS) * len(METHODS) * len(SUMMARY_METRICS)
    checks["bootstrap_count"] = len(cis) == len(BUDGETS) * len(SUMMARY_METRICS)

    cohort_by = {row["protein_id"]: row for row in cohort}
    candidate_groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in scored:
        candidate_groups[(row["protein_id"], budget_label(row["budget"]))].append(row)
    candidate_ok = len(candidate_groups) == EXPECTED_CASES * len(BUDGETS)
    for (protein_id, label), group in candidate_groups.items():
        sequence = cohort_by[protein_id]["wildtype_sequence"]
        expected_deleted = strict_count(len(sequence), float(label))
        candidate_ok &= len(group) == K and len({row["deletion_mask_sha256"] for row in group}) == K
        for row in group:
            intervals = runtime.parse_intervals(row["deleted_intervals"])
            candidate_ok &= int(row["deleted_length"]) == expected_deleted
            candidate_ok &= len(sequence) - len(runtime.mutant(sequence, intervals)) == expected_deleted
    checks["delepref_k8_unique_exact_budget"] = bool(candidate_ok)

    plan_ok = True
    for row in plans:
        confidence = locate_confidence(row["record_id"])
        structure = runtime.confidence_to_structure(confidence)
        plan_ok &= confidence.is_file() and structure.is_file()
        if row["method"] != "wt":
            sequence = row["wildtype_sequence"]
            intervals = runtime.parse_intervals(row["deleted_intervals"])
            plan_ok &= runtime.mutant(sequence, intervals) == row["edited_sequence"]
            plan_ok &= int(row["deleted_length"]) == strict_count(len(sequence), float(row["budget"]))
    checks["fresh_prediction_sidecars_and_plan_reconstruction"] = bool(plan_ok)

    expected_case_ids = [str(value) for value in protocol["cohort"]["case_ids"]]
    assignments_ok = True
    finite_ok = True
    for budget in BUDGET_LABELS:
        for method in METHODS:
            subset = [row for row in metrics if row["budget"] == budget and row["method"] == method]
            assignments_ok &= sorted((row["case_id"] for row in subset), key=int) == expected_case_ids
            for row in subset:
                for metric in PRIMARY_METRICS:
                    value = float(row[metric])
                    finite_ok &= math.isfinite(value) and 0.0 <= value <= 1.0
    checks["complete_50_case_assignments"] = bool(assignments_ok)
    checks["primary_metric_ranges_finite"] = bool(finite_ok)
    checks["primary_metric_coverage_50_of_50"] = all(
        int(row["n_available"]) == EXPECTED_CASES
        for row in summaries
        if row["metric"] in PRIMARY_METRICS
    )

    recomputed_means = {}
    for budget in BUDGET_LABELS:
        for method in METHODS:
            subset = [row for row in metrics if row["budget"] == budget and row["method"] == method]
            for metric in SUMMARY_METRICS:
                values = [float(row[metric]) for row in subset if row[metric] != ""]
                recomputed_means[(budget, method, metric)] = statistics.mean(values)
    checks["summary_means_recomputed"] = all(
        math.isclose(float(row["mean"]), recomputed_means[(row["budget"], row["method"], row["metric"])], rel_tol=0.0, abs_tol=1e-15)
        for row in summaries
    )

    failed = sorted(name for name, passed in checks.items() if not passed)
    validation = {
        "schema_version": "delepref_25_binder50_fair_contact_validation_v1",
        "status": "PASS" if not failed else "FAIL",
        "checks": checks,
        "failed": failed,
        "validated_cases": EXPECTED_CASES,
        "validated_fresh_predictions": len(plans),
        "validated_edited_metric_rows": len(metrics),
    }
    write_json(OUT / "independent_validation.json", validation)
    if failed:
        raise RuntimeError(f"validation failed: {failed}")

    summary_by = {
        (row["budget"], row["method"], row["metric"]): row
        for row in summaries
    }
    ci_by = {
        (row["budget"], row["metric"]): row
        for row in cis
    }
    lines = [
        "# DelePref-25 Binder-50 fair contact comparison",
        "",
        f"Status: `{STATUS}`",
        "",
        "All 50 formal Binder cases were evaluated at exact 10% and 20% deletion budgets. DelePref uses frozen canonical K=8 scorer-top1 inference without interface/Boltz preselection; SCISOR uses one native explicit-index plan. All 50 WT and 200 edited complexes were freshly predicted in one fixed-seed Boltz-2 run.",
        "",
        "| Budget | Method | Contact recovery | Contact Jaccard |",
        "|---:|---|---:|---:|",
    ]
    for budget in BUDGET_LABELS:
        for method in METHODS:
            recovery = float(summary_by[(budget, method, "contact_recovery")]["mean"])
            jaccard = float(summary_by[(budget, method, "contact_jaccard")]["mean"])
            lines.append(f"| {budget} | {method} | {recovery:.6f} | {jaccard:.6f} |")
    lines.extend(["", "Paired DelePref minus SCISOR effects (20,000 case-level bootstrap replicates):", ""])
    for budget in BUDGET_LABELS:
        for metric in PRIMARY_METRICS:
            row = ci_by[(budget, metric)]
            lines.append(
                f"- {budget} {metric}: {float(row['point_estimate']):+.6f} "
                f"[{float(row['ci95_low']):+.6f}, {float(row['ci95_high']):+.6f}]"
            )
    lines.extend(
        [
            "",
            "This is a native end-to-end comparison, not an equal-compute/equal-K ablation: DelePref generates K=8 and applies its frozen scorer, whereas SCISOR emits one native plan.",
            "",
            PROXY_BOUNDARY,
            "",
            CLAIM_BOUNDARY,
            "",
        ]
    )
    write_text(OUT / "SUMMARY.md", "\n".join(lines))
    run_summary = {
        "schema_version": "delepref_25_binder50_fair_contact_summary_v1",
        "status": STATUS,
        "case_count": EXPECTED_CASES,
        "budgets": [0.10, 0.20],
        "methods": list(METHODS),
        "fresh_wt_predictions": EXPECTED_CASES,
        "fresh_edited_predictions": len(metrics),
        "total_fresh_predictions": len(plans),
        "boltz_seed": BOLTZ_SEED,
        "protocol_sha256": sha256_file(OUT / "protocol.json"),
        "independent_validation": "PASS",
        "comparison_scope": "native end-to-end; not equal-compute/equal-K",
        "claim_boundary": CLAIM_BOUNDARY,
        "proxy_boundary": PROXY_BOUNDARY,
    }
    write_json(OUT / "run_summary.json", run_summary)

    excluded = {"audit_manifest.json", STATUS}
    artifacts = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file()):
        relative = path.relative_to(OUT).as_posix()
        if relative in excluded or relative.startswith("audit/"):
            continue
        artifacts.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(OUT / "audit_manifest.json", {"schema_version": "delepref_25_artifact_manifest_v1", "artifacts": artifacts})
    write_text(OUT / STATUS, STATUS + "\n", encoding="ascii")
    print(json.dumps({"status": STATUS, "validation": "PASS", "artifacts": len(artifacts)}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "generate-scisor", "select-build", "analyze", "validate"))
    args = parser.parse_args()
    {
        "prepare": prepare,
        "generate-scisor": generate_scisor,
        "select-build": select_and_build,
        "analyze": analyze,
        "validate": validate,
    }[args.phase]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
