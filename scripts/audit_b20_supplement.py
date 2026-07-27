#!/usr/bin/env python3
"""Full audit of strict same-count beta=0.20 supplement pairs."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO / "scripts/build_pair_same_count.py"
SCORER_PATH = REPO / "scripts/score_b20_esmc600m.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("delepref_b20_pair_builder_for_audit", BUILDER_PATH)
b20_scorer = load_module("delepref_b20_scorer_for_audit", SCORER_PATH)

DEFAULT_RAW_INDEX = REPO / "data/processed/preference_training/delepref/b20_supplement/raw_candidates/indexes/deletion_perturbations_shard_index.csv"
DEFAULT_IMPACT_INDEX = REPO / "data/processed/preference_training/delepref/b20_supplement/teacher_scoring/indexes/teacher_impact_shard_index.csv"
DEFAULT_PAIR_INDEX = REPO / "data/processed/preference_training/delepref/b20_supplement/pairs/indexes/preference_pair_shard_index.csv"
DEFAULT_CONFIG = REPO / "data/processed/preference_training/delepref/b20_supplement/teacher_scoring/teacher_scoring_config.json"
DEFAULT_OUTPUT = REPO / "results/delepref_esmc600m_same_count/delepref/phase_2/b20_supplement_audit.json"


def sha256_file(path, block_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%d" % os.getpid())
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def read_index(path, path_column):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for ordinal, row in enumerate(csv.DictReader(handle)):
            shard_index = int(row.get("shard_index") or ordinal)
            item_path = Path(row[path_column])
            if not item_path.is_absolute():
                item_path = REPO / item_path
            if shard_index in result:
                raise ValueError("duplicate shard index {}".format(shard_index))
            result[shard_index] = {"path": item_path, "row": row}
    return result


def finite(value, name):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("{} is non-finite".format(name))
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-index", type=Path, default=DEFAULT_RAW_INDEX)
    parser.add_argument("--impact-index", type=Path, default=DEFAULT_IMPACT_INDEX)
    parser.add_argument("--pair-index", type=Path, default=DEFAULT_PAIR_INDEX)
    parser.add_argument("--teacher-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = read_index(args.raw_index, "shard_path")
    impacts = read_index(args.impact_index, "output_csv")
    pairs = read_index(args.pair_index, "output_csv")
    if not raw or set(raw) != set(impacts) or set(raw) != set(pairs):
        raise ValueError("raw, impact, and pair indexes do not have identical shard sets")
    teacher_config = json.loads(args.teacher_config.read_text(encoding="utf-8"))
    model_sha = teacher_config["teacher_model_sha256"]
    config_sha = teacher_config["sha256"]

    failures = Counter()
    examples = defaultdict(list)
    policy_counts = Counter()
    gate_histogram = Counter()
    pair_histogram = Counter()
    input_proteins = 0
    input_candidates = 0
    pair_rows = 0
    coverage_proteins = 0
    cache_rows_validated = 0
    split_inheritance_checks = 0
    for shard_index in sorted(raw):
        raw_rows, _ = b20_scorer.read_csv(raw[shard_index]["path"])
        impact_rows, _ = b20_scorer.read_csv(impacts[shard_index]["path"])
        pair_rows_local, fields = b20_scorer.read_csv(pairs[shard_index]["path"])
        if tuple(fields) != builder.PAIR_COLUMNS:
            failures["pair_schema_mismatch"] += 1
        raw_by_key = {}
        raw_by_protein = defaultdict(list)
        base_split = {}
        for row in raw_rows:
            key = (row["protein_id"], row["perturbation_id"], row["plan_id"])
            if key in raw_by_key:
                failures["duplicate_raw_key"] += 1
                continue
            raw_by_key[key] = row
            raw_by_protein[row["protein_id"]].append(row)
            metadata = json.loads(row.get("metadata_json") or "{}")
            split = str(metadata.get("base_source_split") or "")
            if row["protein_id"] in base_split and base_split[row["protein_id"]] != split:
                failures["inconsistent_base_split"] += 1
            base_split[row["protein_id"]] = split
            policy_counts[row["generation_policy"]] += 1
            length = int(row["wildtype_length"])
            deleted = int(row["deleted_length"])
            expected = min(length - 1, int(round(0.20 * length)))
            intervals = json.loads(row["deleted_intervals"])
            if deleted != expected or deleted != sum(end - start + 1 for start, end in intervals):
                failures["strict_b20_deleted_count_mismatch"] += 1
            if not 1 <= len(intervals) <= 3:
                failures["qmax_violation"] += 1
        for protein_id, rows in raw_by_protein.items():
            if len(rows) != 8 or len({row["deleted_intervals"] for row in rows}) != 8:
                failures["not_eight_unique_raw_candidates"] += 1

        impact_by_key = {}
        gate_passing = Counter()
        for row in impact_rows:
            key = (row["protein_id"], row["perturbation_id"], row["plan_id"])
            raw_row = raw_by_key.get(key)
            if raw_row is None or key in impact_by_key:
                failures["impact_key_linkage"] += 1
                continue
            impact_by_key[key] = row
            if row.get("scoring_status") != "ok":
                failures["teacher_scoring_not_ok"] += 1
                continue
            try:
                metadata = json.loads(row.get("metadata_json") or "{}")
                expected_identity = b20_scorer.strict_identity(raw_row, model_sha, config_sha)
                if metadata.get("strict_cache_identity") != expected_identity:
                    failures["strict_cache_identity_mismatch"] += 1
                else:
                    cache_rows_validated += 1
                global_similarity = finite(row["global_similarity"], "global_similarity")
                retained_similarity = finite(row["retained_similarity"], "retained_similarity")
                finite(row["junction_plausibility"], "junction_plausibility")
            except Exception as exc:
                failures["invalid_impact"] += 1
                if len(examples["invalid_impact"]) < 5:
                    examples["invalid_impact"].append(str(exc))
                continue
            if global_similarity >= builder.GLOBAL_GATE and retained_similarity >= builder.RETAINED_GATE:
                gate_passing[row["protein_id"]] += 1
        if set(raw_by_key) != set(impact_by_key):
            failures["raw_impact_key_set_mismatch"] += len(set(raw_by_key) ^ set(impact_by_key))

        legal_pairs = Counter()
        for row in pair_rows_local:
            pair_rows += 1
            protein_id = row["protein_id"]
            a_key = (protein_id, row["perturbation_id_a"], row["plan_id_a"])
            b_key = (protein_id, row["perturbation_id_b"], row["plan_id_b"])
            if a_key not in raw_by_key or b_key not in raw_by_key:
                failures["pair_raw_linkage"] += 1
                continue
            a = raw_by_key[a_key]
            b = raw_by_key[b_key]
            if int(a["deleted_length"]) != int(b["deleted_length"]):
                failures["unequal_count_pair"] += 1
            expected_d = min(int(a["wildtype_length"]) - 1, int(round(0.20 * int(a["wildtype_length"]))))
            if int(row["deleted_length"]) != expected_d:
                failures["pair_not_b20"] += 1
            if row["source_split"] != base_split.get(protein_id):
                failures["b20_split_not_inherited"] += 1
            split_inheritance_checks += 1
            legal_pairs[protein_id] += 1

        for protein_id in raw_by_protein:
            gate_count = gate_passing[protein_id]
            pair_count = legal_pairs[protein_id]
            gate_histogram[gate_count] += 1
            pair_histogram[pair_count] += 1
            if gate_count >= 2 and pair_count >= 1:
                coverage_proteins += 1
        input_proteins += len(raw_by_protein)
        input_candidates += len(raw_rows)
        print(builder.canonical_json({
            "event": "b20_audit_shard_complete", "shard_index": shard_index,
            "proteins": len(raw_by_protein), "pairs": len(pair_rows_local),
            "failures_so_far": sum(failures.values()),
        }), flush=True)

    coverage_rate = coverage_proteins / float(input_proteins) if input_proteins else 0.0
    if len(raw) != 300:
        failures["shard_count_not_300"] += 1
    if input_proteins != 300000:
        failures["protein_count_not_300000"] += 1
    if input_candidates != 2400000:
        failures["candidate_count_not_2400000"] += 1
    if coverage_rate < 0.95:
        failures["coverage_below_0.95"] += 1
    payload = {
        "schema_version": "delepref_b20_supplement_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "hard_gates": {"global_similarity_min": builder.GLOBAL_GATE, "retained_similarity_min": builder.RETAINED_GATE, "lowered": False},
        "teacher_utility": "0.35*global_similarity+0.45*retained_similarity+0.20*junction_plausibility",
        "shards": len(raw),
        "input_proteins": input_proteins,
        "input_candidates": input_candidates,
        "pair_rows": pair_rows,
        "coverage_definition": "gate_passing_candidates>=2 AND legal_strict_same_count_pair_count>=1",
        "coverage_proteins": coverage_proteins,
        "coverage_rate": coverage_rate,
        "coverage_threshold": 0.95,
        "gate_passing_candidate_count_histogram": dict(gate_histogram),
        "legal_pair_count_histogram": dict(pair_histogram),
        "policy_counts": dict(policy_counts),
        "cache_rows_validated": cache_rows_validated,
        "split_inheritance_checks": split_inheritance_checks,
        "failure_count": sum(failures.values()),
        "failure_counts": dict(failures),
        "failure_examples": dict(examples),
        "indexes": {
            "raw": {"path": str(args.raw_index), "sha256": sha256_file(args.raw_index)},
            "impact": {"path": str(args.impact_index), "sha256": sha256_file(args.impact_index)},
            "pair": {"path": str(args.pair_index), "sha256": sha256_file(args.pair_index)},
        },
        "teacher_model_sha256": model_sha,
        "teacher_scoring_config_sha256": config_sha,
        "supports_beta_030": False,
        "teacher_preference_is_ground_truth": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
