#!/usr/bin/env python3
"""Build deletion perturbation tables and legacy corruption data.

Current DelePref uses this script primarily to construct finite, diverse
counterfactual deletion perturbation sets for teacher queries. The legacy JSONL
mode still builds synthetic insertion-corruption examples for old proposer
training utilities.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict


DEFAULT_CONFIG = {
    "budgets": [0.1, 0.2, 0.3],
    "strength_sampling": "continuous",
    "corruption_strength_min": 0.05,
    "corruption_strength_max": 0.35,
    "views_per_sequence": 3,
    "view_seed_stride": 10000019,
    "corruption_mix": "balanced",
    "min_insert_segment_len": 1,
    "max_insert_segment_len": 40,
    "corruption_types": [
        "random_segment_insertion",
        "terminal_tail_insertion",
        "linker_like_insertion",
        "low_complexity_insertion",
        "local_duplication_insertion",
    ],
    "amino_acid_alphabet": "ACDEFGHIKLMNPQRSTVWY",
    "linker_like_tokens": ["G", "S", "P", "N", "Q"],
    "terminal_tail_bias": 0.2,
    "linker_like_bias": 0.25,
    "low_complexity_bias": 0.2,
    "local_duplication_bias": 0.25,
    "random_segment_bias": 0.1,
    "max_repeat_fraction": 0.6,
    "seed": 42,
}


TYPE_TO_BIAS = {
    "random_segment_insertion": "random_segment_bias",
    "terminal_tail_insertion": "terminal_tail_bias",
    "linker_like_insertion": "linker_like_bias",
    "low_complexity_insertion": "low_complexity_bias",
    "local_duplication_insertion": "local_duplication_bias",
}

COORDINATE_SYSTEM = "0_based_closed_inclusive"

DELETION_PERTURBATION_REQUIRED_COLUMNS = [
    "protein_id",
    "perturbation_id",
    "plan_id",
    "wildtype_length",
    "deleted_intervals",
    "num_intervals",
    "deleted_length",
    "deletion_fraction",
    "budget_bin",
    "generation_policy",
    "coordinate_system",
]

DELETION_PERTURBATION_OPTIONAL_COLUMNS = [
    "wildtype_sequence",
    "mutant_sequence",
    "source_stage1_row_id",
    "source_start",
    "source_end",
    "source_score",
    "metadata_json",
]

STAGE1_PROPOSAL_REQUIRED_COLUMNS = [
    "protein_id",
    "sequence_length",
    "start",
    "end",
    "length",
    "stage1_proposal_confidence",
    "stage1_interval_rank",
    "sequence",
]

DELETION_GENERATION_POLICIES = [
    "stage1_proposal",
    "random_interval",
    "uniform_random_interval",
    "single_residue",
    "short_span",
    "budget_matched_random",
    "boundary_challenge",
    "low_complexity_or_boundary",
    "terminal_baseline",
    "multi_interval_combination",
    "high_risk_negative",
]

METHOD_V1_DEFAULT_BUDGET_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.30]
METHOD_V1_DEFAULT_POLICY_QUOTAS = {
    "stage1_proposal": 4,
    "budget_matched_random": 2,
    "random_interval": 4,
    "uniform_random_interval": 4,
    "boundary_challenge": 3,
    "terminal_baseline": 1,
    "multi_interval_combination": 1,
    "high_risk_negative": 2,
}
METHOD_V1_POLICY_PRIORITY = {
    "stage1_proposal": 95.0,
    "high_risk_negative": 82.0,
    "boundary_challenge": 70.0,
    "budget_matched_random": 62.0,
    "multi_interval_combination": 55.0,
    "uniform_random_interval": 45.0,
    "random_interval": 44.0,
    "terminal_baseline": 35.0,
}

FORMAL_QUERY_PROFILE = "formal_cold_start_v1"
FORMAL_QUERY_BUDGET_FRACTIONS = {
    "b001_single": None,
    "b02": 0.02,
    "b05": 0.05,
    "b10": 0.10,
    "b15": 0.15,
    "b20": 0.20,
    "b30": 0.30,
}
FORMAL_QUERY_POLICY_PLAN = [
    ("b001_single", "single_residue", 2),
    ("b02", "short_span", 2),
    ("b02", "uniform_random_interval", 1),
    ("b02", "boundary_challenge", 1),
    ("b05", "uniform_random_interval", 2),
    ("b05", "boundary_challenge", 1),
    ("b05", "low_complexity_or_boundary", 1),
    ("b05", "terminal_baseline", 1),
    ("b05", "multi_interval_combination", 1),
    ("b10", "uniform_random_interval", 2),
    ("b10", "boundary_challenge", 1),
    ("b10", "low_complexity_or_boundary", 1),
    ("b10", "terminal_baseline", 1),
    ("b10", "multi_interval_combination", 1),
    ("b15", "uniform_random_interval", 2),
    ("b15", "boundary_challenge", 1),
    ("b15", "low_complexity_or_boundary", 1),
    ("b15", "terminal_baseline", 1),
    ("b15", "multi_interval_combination", 1),
    ("b20", "uniform_random_interval", 1),
    ("b20", "boundary_challenge", 1),
    ("b20", "terminal_baseline", 1),
    ("b20", "multi_interval_combination", 1),
    ("b30", "uniform_random_interval", 1),
    ("b30", "boundary_challenge", 1),
    ("b30", "terminal_baseline", 1),
    ("b30", "multi_interval_combination", 1),
]
FORMAL_QUERY_FIELDNAMES = DELETION_PERTURBATION_REQUIRED_COLUMNS + DELETION_PERTURBATION_OPTIONAL_COLUMNS
FORMAL_QUERY_SHARD_INDEX_FIELDS = [
    "shard_id",
    "shard_index",
    "shard_path",
    "input_manifest_path",
    "first_manifest_rank",
    "last_manifest_rank",
    "num_proteins",
    "num_candidates",
    "candidates_per_protein_target",
    "query_profile",
    "status",
    "summary_json",
    "metadata_json",
]


def parse_scalar(value):
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("\"'")


def load_config(path):
    try:
        import yaml  # type: ignore

        with open(path) as handle:
            loaded = yaml.safe_load(handle) or {}
        config = dict(DEFAULT_CONFIG)
        config.update(loaded)
        return config
    except ImportError:
        config = dict(DEFAULT_CONFIG)
        current_key = None
        with open(path) as handle:
            for raw_line in handle:
                line = raw_line.rstrip()
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("- ") and current_key:
                    config.setdefault(current_key, []).append(parse_scalar(stripped[2:]))
                    continue
                if ":" not in stripped:
                    continue
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip()
                current_key = key
                if value:
                    config[key] = parse_scalar(value)
                else:
                    config[key] = []
        return config


def read_fasta(path):
    records = []
    header = None
    chunks = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            records.append((header, "".join(chunks)))
    return records


def seq_id_from_header(header):
    return header.split(None, 1)[0]


def weighted_choice(corruption_types, config, rng):
    if str(config.get("corruption_mix", "weighted")).lower() == "balanced":
        return rng.choice(corruption_types)
    weights = [float(config.get(TYPE_TO_BIAS.get(name, ""), 0.0)) for name in corruption_types]
    if sum(weights) <= 0:
        weights = [1.0 for _ in corruption_types]
    return rng.choices(corruption_types, weights=weights, k=1)[0]


def sample_corruption_strength(config, rng):
    """Sample an internal corruption strength for Stage-1 pretraining.

    This replaces the public budget-conditioned view with a latent training
    variable.  Older configs can still force fixed values by setting
    strength_sampling: fixed_budgets.
    """

    mode = str(config.get("strength_sampling", "continuous")).lower()
    fixed = [float(value) for value in config.get("budgets", [])]
    if mode in ("fixed", "fixed_budget", "fixed_budgets") and fixed:
        return rng.choice(fixed)
    lo = float(config.get("corruption_strength_min", 0.05))
    hi = float(config.get("corruption_strength_max", 0.35))
    lo = max(1e-6, min(lo, hi))
    hi = max(lo, hi)
    return rng.uniform(lo, hi)


def iter_corruption_views(seq_id, sequence, config, base_seed=None):
    """Yield multiple independently sampled corruption views for one sequence."""

    views = max(1, int(config.get("views_per_sequence", 1)))
    stride = int(config.get("view_seed_stride", 10000019))
    if base_seed is None:
        base_seed = int(config.get("seed", 42))
    for view_idx in range(views):
        rng = random.Random(int(base_seed) + view_idx * stride)
        strength = sample_corruption_strength(config, rng)
        sample, recovery_ok = corrupt_sequence(seq_id, sequence, strength, config, rng)
        sample["view_idx"] = view_idx
        sample["corruption_strength"] = strength
        sample["budget_ratio"] = strength
        sample.setdefault("metadata", {})["latent_corruption_strength"] = strength
        sample["metadata"]["view_idx"] = view_idx
        sample["metadata"]["strength_sampling"] = str(config.get("strength_sampling", "continuous"))
        yield sample, recovery_ok


def random_segment(length, alphabet, rng):
    return "".join(rng.choice(alphabet) for _ in range(length))


def terminal_tail_segment(length, alphabet, rng):
    tail_tokens = list("GPSTNQDEKRS")
    segment = []
    for _ in range(length):
        if rng.random() < 0.65:
            segment.append(rng.choice(tail_tokens))
        else:
            segment.append(rng.choice(alphabet))
    return "".join(segment)


def linker_like_segment(length, linker_tokens, alphabet, rng):
    segment = []
    for _ in range(length):
        if rng.random() < 0.85:
            segment.append(rng.choice(linker_tokens))
        else:
            segment.append(rng.choice(alphabet))
    return "".join(segment)


def low_complexity_segment(length, alphabet, max_repeat_fraction, rng):
    if length <= 1:
        return rng.choice(alphabet)
    max_repeat = max(1, int(math.ceil(length * max_repeat_fraction)))
    for _ in range(100):
        token_count = rng.randint(2, min(4, len(alphabet)))
        tokens = rng.sample(alphabet, token_count)
        weights = [rng.random() + 0.2 for _ in tokens]
        segment = "".join(rng.choices(tokens, weights=weights, k=length))
        if max(Counter(segment).values()) <= max_repeat:
            return segment
    # Deterministic fallback that respects the repeat cap better than a homopolymer.
    tokens = rng.sample(alphabet, min(3, len(alphabet)))
    return "".join(tokens[i % len(tokens)] for i in range(length))


def local_duplication_segment(sequence, length, rng):
    if length <= len(sequence):
        source_start = rng.randint(0, len(sequence) - length)
        return sequence[source_start : source_start + length], source_start
    repeats = []
    while sum(len(part) for part in repeats) < length:
        source_len = min(len(sequence), length - sum(len(part) for part in repeats))
        source_start = rng.randint(0, len(sequence) - source_len)
        repeats.append(sequence[source_start : source_start + source_len])
    return "".join(repeats)[:length], None


def choose_insert_position(corruption_type, sequence, segment_len, source_start, rng):
    original_len = len(sequence)
    if corruption_type == "terminal_tail_insertion":
        return 0 if rng.random() < 0.5 else original_len
    if corruption_type == "local_duplication_insertion" and source_start is not None:
        center = source_start + segment_len
        low = max(0, center - 20)
        high = min(original_len, center + 20)
        return rng.randint(low, high)
    return rng.randint(0, original_len)


def build_inserted_segment(corruption_type, sequence, segment_len, config, rng):
    alphabet = list(str(config["amino_acid_alphabet"]))
    linker_tokens = list(config["linker_like_tokens"])
    source_start = None
    if corruption_type == "random_segment_insertion":
        inserted = random_segment(segment_len, alphabet, rng)
    elif corruption_type == "terminal_tail_insertion":
        inserted = terminal_tail_segment(segment_len, alphabet, rng)
    elif corruption_type == "linker_like_insertion":
        inserted = linker_like_segment(segment_len, linker_tokens, alphabet, rng)
    elif corruption_type == "low_complexity_insertion":
        inserted = low_complexity_segment(segment_len, alphabet, float(config["max_repeat_fraction"]), rng)
    elif corruption_type == "local_duplication_insertion":
        inserted, source_start = local_duplication_segment(sequence, segment_len, rng)
    else:
        raise ValueError("Unknown corruption type: {}".format(corruption_type))
    return inserted, source_start


def segment_lengths_for_budget(target_len, config, rng):
    min_len = int(config["min_insert_segment_len"])
    max_len = int(config["max_insert_segment_len"])
    remaining = target_len
    lengths = []
    while remaining > 0:
        if remaining <= max_len:
            seg_len = remaining
        else:
            seg_len = rng.randint(min_len, min(max_len, remaining))
        lengths.append(seg_len)
        remaining -= seg_len
    return lengths


def corrupt_sequence(seq_id, sequence, budget_ratio, config, rng):
    target_insert_len = int(math.floor(len(sequence) * budget_ratio))
    target_insert_len = max(1, target_insert_len)
    corruption_types = list(config["corruption_types"])
    original_pieces = [{"kind": "original", "text": aa, "orig_index": idx} for idx, aa in enumerate(sequence)]
    insertion_segments = []
    type_counts = Counter()

    for segment_idx, seg_len in enumerate(segment_lengths_for_budget(target_insert_len, config, rng)):
        corruption_type = weighted_choice(corruption_types, config, rng)
        inserted, source_start = build_inserted_segment(corruption_type, sequence, seg_len, config, rng)
        insert_pos_original = choose_insert_position(corruption_type, sequence, seg_len, source_start, rng)

        # Map original-coordinate insert position to current corrupted-coordinate slot.
        if insert_pos_original == len(sequence):
            current_insert_pos = len(original_pieces)
        else:
            current_insert_pos = next(
                i for i, piece in enumerate(original_pieces)
                if piece.get("kind") == "original" and piece.get("orig_index") == insert_pos_original
            )
        inserted_pieces = [
            {
                "kind": "inserted",
                "text": aa,
                "segment_idx": segment_idx,
                "corruption_type": corruption_type,
            }
            for aa in inserted
        ]
        original_pieces[current_insert_pos:current_insert_pos] = inserted_pieces
        corrupted_start = current_insert_pos
        corrupted_end_exclusive = current_insert_pos + len(inserted)
        insertion_segments.append(
            {
                "segment_idx": segment_idx,
                "corruption_type": corruption_type,
                "inserted_sequence": inserted,
                "inserted_length": len(inserted),
                "original_insert_pos": insert_pos_original,
                "corrupted_start": corrupted_start,
                "corrupted_end_exclusive": corrupted_end_exclusive,
                "source_start": source_start,
                "source_end_exclusive": None if source_start is None else source_start + len(inserted),
            }
        )
        type_counts[corruption_type] += 1

    corrupted_sequence = "".join(piece["text"] for piece in original_pieces)
    delete_labels = [1 if piece["kind"] == "inserted" else 0 for piece in original_pieces]
    keep_labels = [1 - value for value in delete_labels]
    recovered = "".join(aa for aa, label in zip(corrupted_sequence, delete_labels) if label == 0)
    inserted_total_length = sum(delete_labels)
    recovery_ok = (
        recovered == sequence
        and len(delete_labels) == len(corrupted_sequence)
        and len(corrupted_sequence) == len(sequence) + inserted_total_length
        and inserted_total_length == target_insert_len
    )
    return {
        "seq_id": seq_id,
        "original_sequence": sequence,
        "corrupted_sequence": corrupted_sequence,
        "budget_ratio": budget_ratio,
        "original_length": len(sequence),
        "corrupted_length": len(corrupted_sequence),
        "inserted_total_length": inserted_total_length,
        "insertion_segments": insertion_segments,
        "delete_labels": delete_labels,
        "keep_labels": keep_labels,
        "corruption_type_summary": dict(type_counts),
        "metadata": {
            "target_insert_len": target_insert_len,
            "recovery_ok": recovery_ok,
            "generator": "stage1_sequence_prior_segment_corruption",
        },
    }, recovery_ok


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)


def safe_id(value):
    text = str(value).strip()
    out = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            out.append(char)
        else:
            out.append("_")
    safe = "".join(out).strip("_")
    return safe or "unknown"


def parse_required_int(value, field, source):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError("{}: {} must be an integer, got {!r}".format(source, field, value))


def normalize_deletion_intervals(intervals, wildtype_length, source):
    if wildtype_length <= 0:
        raise ValueError("{}: wildtype_length must be positive".format(source))
    normalized = []
    for interval_idx, interval in enumerate(intervals):
        if len(interval) != 2:
            raise ValueError("{}: interval {} must contain start and end".format(source, interval_idx))
        start = int(interval[0])
        end = int(interval[1])
        if start < 0:
            raise ValueError("{}: interval {} start must be nonnegative".format(source, interval_idx))
        if end < start:
            raise ValueError("{}: interval {} end must be >= start".format(source, interval_idx))
        if end >= wildtype_length:
            raise ValueError("{}: interval {} end must be < wildtype_length".format(source, interval_idx))
        normalized.append((start, end))
    normalized.sort(key=lambda item: item[0])
    previous_end = None
    for start, end in normalized:
        if previous_end is not None and start <= previous_end:
            raise ValueError("{}: deletion intervals overlap under closed-inclusive coordinates".format(source))
        previous_end = end
    return normalized


def mutant_from_deletion_intervals(sequence, intervals):
    pieces = []
    cursor = 0
    for start, end in intervals:
        pieces.append(sequence[cursor:start])
        cursor = end + 1
    pieces.append(sequence[cursor:])
    return "".join(pieces)


def deletion_fraction(deleted_length, wildtype_length):
    return deleted_length / float(wildtype_length)


def format_fraction(value):
    return "{:.10f}".format(float(value))


def budget_bin_for_fraction(value):
    percent = max(1, int(round(float(value) * 100.0)))
    return "b{:02d}".format(percent)


def deletion_row(
    protein_id,
    sequence,
    intervals,
    perturbation_id,
    plan_id,
    generation_policy,
    metadata,
    source_stage1_row_id="",
    source_start="",
    source_end="",
    source_score="",
):
    wildtype_length = len(sequence)
    intervals = normalize_deletion_intervals(intervals, wildtype_length, perturbation_id)
    deleted_length = sum(end - start + 1 for start, end in intervals)
    fraction = deletion_fraction(deleted_length, wildtype_length)
    return {
        "protein_id": protein_id,
        "perturbation_id": perturbation_id,
        "plan_id": plan_id,
        "wildtype_length": wildtype_length,
        "deleted_intervals": json.dumps([[start, end] for start, end in intervals], separators=(",", ":")),
        "num_intervals": len(intervals),
        "deleted_length": deleted_length,
        "deletion_fraction": format_fraction(fraction),
        "budget_bin": budget_bin_for_fraction(fraction),
        "generation_policy": generation_policy,
        "coordinate_system": COORDINATE_SYSTEM,
        "wildtype_sequence": sequence,
        "mutant_sequence": mutant_from_deletion_intervals(sequence, intervals),
        "source_stage1_row_id": source_stage1_row_id,
        "source_start": source_start,
        "source_end": source_end,
        "source_score": source_score,
        "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
    }


def read_stage1_proposal_rows(path):
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in STAGE1_PROPOSAL_REQUIRED_COLUMNS if field not in fieldnames]
        if missing:
            raise ValueError("{}: missing required Stage1 proposal columns: {}".format(path, ", ".join(missing)))
        return list(reader)


def parse_stage1_proposal_row(path, row_idx, row):
    protein_id = (row.get("protein_id") or "").strip()
    sequence = (row.get("sequence") or "").strip().upper()
    source = "{} row {}".format(path, row_idx)
    if not protein_id:
        raise ValueError("{}: protein_id is empty".format(source))
    if not sequence:
        raise ValueError("{}: sequence is required for deletion perturbation generation".format(source))
    wildtype_length = parse_required_int(row.get("sequence_length"), "sequence_length", source)
    start = parse_required_int(row.get("start"), "start", source)
    end = parse_required_int(row.get("end"), "end", source)
    length = parse_required_int(row.get("length"), "length", source)
    if len(sequence) != wildtype_length:
        raise ValueError("{}: sequence length does not match sequence_length".format(source))
    if length != end - start + 1:
        raise ValueError("{}: length must equal end - start + 1".format(source))
    return {
        "row_idx": row_idx,
        "protein_id": protein_id,
        "sequence": sequence,
        "wildtype_length": wildtype_length,
        "start": start,
        "end": end,
        "length": length,
        "stage1_proposal_confidence": (row.get("stage1_proposal_confidence") or "").strip(),
        "stage1_interval_rank": (row.get("stage1_interval_rank") or "").strip(),
        "stage1_score_mode": (row.get("stage1_score_mode") or "").strip(),
        "source_fasta": (row.get("source_fasta") or "").strip(),
        "checkpoint": (row.get("checkpoint") or "").strip(),
    }


def read_stage1_deletion_sources(path, max_rows=None):
    parsed_rows = []
    for row_idx, row in enumerate(read_stage1_proposal_rows(path), start=2):
        if max_rows is not None and len(parsed_rows) >= max_rows:
            break
        parsed_rows.append(parse_stage1_proposal_row(path, row_idx, row))
    source_records = []
    seen = set()
    for item in parsed_rows:
        key = item["protein_id"]
        if key in seen:
            continue
        seen.add(key)
        source_records.append({
            "protein_id": item["protein_id"],
            "sequence": item["sequence"],
            "source": "stage1_proposal_csv",
            "source_row_id": item["row_idx"],
        })
    return parsed_rows, source_records


def fasta_source_records(path, max_records=None):
    records = []
    for header, raw_sequence in read_fasta(path):
        if max_records is not None and len(records) >= max_records:
            break
        sequence = "".join(raw_sequence.split()).upper()
        if not sequence:
            continue
        records.append({
            "protein_id": seq_id_from_header(header),
            "sequence": sequence,
            "source": "input_fasta",
            "source_row_id": "",
        })
    return records


def iter_stage1_deletion_perturbations(path, max_rows=None):
    emitted = 0
    for row_idx, row in enumerate(read_stage1_proposal_rows(path), start=2):
        if max_rows is not None and emitted >= max_rows:
            break
        parsed = parse_stage1_proposal_row(path, row_idx, row)
        protein_id = parsed["protein_id"]
        sequence = parsed["sequence"]
        start = parsed["start"]
        end = parsed["end"]
        source_score = parsed["stage1_proposal_confidence"]
        rank = parsed["stage1_interval_rank"]
        perturbation_id = "{}__stage1__row{}__{}_{}".format(safe_id(protein_id), row_idx, start, end)
        metadata = {
            "source": "stage1_proposal_csv",
            "stage1_interval_rank": rank,
            "stage1_score_mode": parsed["stage1_score_mode"],
            "source_fasta": parsed["source_fasta"],
            "checkpoint": parsed["checkpoint"],
        }
        yield deletion_row(
            protein_id=protein_id,
            sequence=sequence,
            intervals=[(start, end)],
            perturbation_id=perturbation_id,
            plan_id=perturbation_id,
            generation_policy="stage1_proposal",
            metadata=metadata,
            source_stage1_row_id=str(row_idx),
            source_start=str(start),
            source_end=str(end),
            source_score=source_score,
        )
        emitted += 1


def choose_random_deletion_interval(sequence_length, fraction, min_len, max_len, rng):
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    target_len = int(round(sequence_length * float(fraction)))
    target_len = max(int(min_len), target_len)
    target_len = min(int(max_len), target_len)
    target_len = min(sequence_length, max(1, target_len))
    start = rng.randint(0, sequence_length - target_len)
    end = start + target_len - 1
    return start, end


def bounded_deletion_length(sequence_length, fraction, min_len, max_len):
    target_len = int(round(sequence_length * float(fraction)))
    target_len = max(int(min_len), target_len)
    target_len = min(int(max_len), target_len)
    return min(sequence_length, max(1, target_len))


def choose_random_interval_with_length(sequence_length, target_len, rng, avoid_interval=None):
    target_len = min(sequence_length, max(1, int(target_len)))
    if sequence_length <= target_len:
        return 0, sequence_length - 1
    for _attempt in range(100):
        start = rng.randint(0, sequence_length - target_len)
        end = start + target_len - 1
        if avoid_interval is None or (start, end) != avoid_interval:
            return start, end
    start = 0
    end = target_len - 1
    if avoid_interval == (start, end) and sequence_length > target_len:
        start = 1
        end = start + target_len - 1
    return start, end


def parse_deletion_generation_policies(value):
    if value is None or str(value).strip() == "":
        return None
    policies = []
    seen = set()
    allowed = set(DELETION_GENERATION_POLICIES)
    for raw in str(value).split(","):
        policy = raw.strip()
        if not policy:
            continue
        if policy not in allowed:
            raise ValueError("unknown deletion generation policy {!r}; allowed: {}".format(
                policy,
                ",".join(DELETION_GENERATION_POLICIES),
            ))
        if policy not in seen:
            policies.append(policy)
            seen.add(policy)
    if not policies:
        raise ValueError("--deletion_generation_policies did not contain any policies")
    return policies


def parse_float_csv(value, default_values):
    if value is None or str(value).strip() == "":
        return list(default_values)
    out = []
    for raw in str(value).split(","):
        text = raw.strip()
        if text:
            out.append(float(text))
    if not out:
        raise ValueError("expected at least one float value")
    return out


def parse_policy_quota_map(value):
    quotas = dict(METHOD_V1_DEFAULT_POLICY_QUOTAS)
    if value is None or str(value).strip() == "":
        return quotas
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("policy quota entries must be policy=count, got {!r}".format(item))
        policy, count = item.split("=", 1)
        policy = policy.strip()
        if policy not in METHOD_V1_DEFAULT_POLICY_QUOTAS:
            raise ValueError("unknown method_v1 policy quota {!r}".format(policy))
        quotas[policy] = max(0, int(count.strip()))
    return quotas


def candidate_bank_enabled(args):
    return bool(str(getattr(args, "candidate_bank_profile", "") or "").strip())


def effective_seed(args):
    if candidate_bank_enabled(args) and getattr(args, "candidate_bank_seed", None) is not None:
        return int(args.candidate_bank_seed)
    return int(args.seed if args.seed is not None else DEFAULT_CONFIG["seed"])


def effective_max_candidates_per_protein(args):
    if candidate_bank_enabled(args):
        return max(1, int(args.candidate_bank_max_candidates_per_protein))
    return max(1, int(args.method_max_teacher_candidates_per_protein))


def effective_max_candidates_per_budget(args):
    if candidate_bank_enabled(args):
        return max(1, int(args.candidate_bank_max_candidates_per_budget_bin))
    return max(1, int(args.method_max_teacher_candidates_per_budget))


def effective_overlap_jaccard_max(args):
    if candidate_bank_enabled(args):
        return float(args.candidate_bank_overlap_jaccard_max)
    return float(args.method_nms_iou_threshold)


def effective_policy_quotas(args):
    return parse_policy_quota_map(args.method_policy_quotas)


def metadata_payload(row):
    try:
        return json.loads(row.get("metadata_json") or "{}")
    except (TypeError, ValueError):
        return {}


def intervals_from_deletion_row(row):
    return [tuple(item) for item in json.loads(row["deleted_intervals"])]


def interval_overlap_length(a, b):
    total = 0
    for a_start, a_end in a:
        for b_start, b_end in b:
            start = max(int(a_start), int(b_start))
            end = min(int(a_end), int(b_end))
            if end >= start:
                total += end - start + 1
    return total


def intervals_total_length(intervals):
    return sum(end - start + 1 for start, end in intervals)


def interval_iou(a, b):
    overlap = interval_overlap_length(a, b)
    if overlap <= 0:
        return 0.0
    union = intervals_total_length(a) + intervals_total_length(b) - overlap
    return overlap / float(max(1, union))


def update_metadata_json(row, updates):
    metadata = metadata_payload(row)
    metadata.update(updates)
    row["metadata_json"] = json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def compact_intervals(intervals):
    return "_".join("{}-{}".format(start, end) for start, end in intervals)


def stable_int(value):
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def formal_rng(seed_tag, protein_id, sequence_sha1, budget_bin, policy, ordinal):
    seed_text = "|".join([
        str(seed_tag),
        str(protein_id),
        str(sequence_sha1),
        str(budget_bin),
        str(policy),
        str(ordinal),
    ])
    return random.Random(stable_int(seed_text))


def is_terminal_deletion(row):
    length = int(row["wildtype_length"])
    intervals = intervals_from_deletion_row(row)
    return any(start == 0 or end == length - 1 for start, end in intervals)


def row_priority(row):
    metadata = metadata_payload(row)
    try:
        return float(metadata.get("selection_priority", 0.0))
    except (TypeError, ValueError):
        return 0.0


def row_deleted_length(row):
    return int(row["deleted_length"])


def row_target_budget_distance(row):
    metadata = metadata_payload(row)
    target = metadata.get("target_budget_fraction")
    try:
        return abs(float(row["deletion_fraction"]) - float(target))
    except (TypeError, ValueError):
        return 1.0


def make_method_candidate_row(
    protein_id,
    sequence,
    intervals,
    policy,
    seed,
    candidate_idx,
    budget_fraction,
    source,
    metadata_extra=None,
    source_stage1_row_id="",
    source_start="",
    source_end="",
    source_score="",
):
    intervals = normalize_deletion_intervals(intervals, len(sequence), "{} candidate {}".format(policy, candidate_idx))
    compact = compact_intervals(intervals)
    budget_tag = budget_bin_for_fraction(deletion_fraction(intervals_total_length(intervals), len(sequence)))
    perturbation_id = "{}__methodv1__{}__{}__{}__{}".format(
        safe_id(protein_id),
        safe_id(policy),
        budget_tag,
        candidate_idx,
        safe_id(compact),
    )
    metadata = {
        "candidate_construction_mode": "method_v1",
        "candidate_pool_stage": "cheap_pool",
        "source": source,
        "seed": seed,
        "target_budget_fraction": budget_fraction,
        "selection_priority": METHOD_V1_POLICY_PRIORITY.get(policy, 0.0),
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return deletion_row(
        protein_id=protein_id,
        sequence=sequence,
        intervals=intervals,
        perturbation_id=perturbation_id,
        plan_id=perturbation_id,
        generation_policy=policy,
        metadata=metadata,
        source_stage1_row_id=source_stage1_row_id,
        source_start=source_start,
        source_end=source_end,
        source_score=source_score,
    )


def bounded_interval_around_center(sequence_length, center, target_len):
    target_len = min(int(sequence_length), max(1, int(target_len)))
    start = int(round(float(center) - (target_len - 1) / 2.0))
    start = max(0, min(start, int(sequence_length) - target_len))
    return start, start + target_len - 1


def low_complexity_boundary_intervals(sequence, target_len, max_count):
    if not sequence or target_len <= 0:
        return []
    target_len = min(len(sequence), max(1, int(target_len)))
    step = max(1, target_len // 2)
    linker_tokens = set("GPSNQ")
    candidates = []
    for start in range(0, len(sequence) - target_len + 1, step):
        end = start + target_len - 1
        window = sequence[start : end + 1]
        linker_fraction = sum(1 for aa in window if aa in linker_tokens) / float(len(window))
        repeat_fraction = max(Counter(window).values()) / float(len(window))
        center_distance = abs((start + end) * 0.5 - (len(sequence) - 1) * 0.5) / float(max(1, len(sequence)))
        score = linker_fraction + 0.5 * repeat_fraction - 0.15 * center_distance
        candidates.append((score, start, end, linker_fraction, repeat_fraction))
    candidates.sort(reverse=True)
    out = []
    for score, start, end, linker_fraction, repeat_fraction in candidates:
        if any(interval_iou([(start, end)], [(old_start, old_end)]) > 0.5 for old_start, old_end, _meta in out):
            continue
        out.append(
            (
                start,
                end,
                {
                    "boundary_type": "low_complexity_or_linker_proxy",
                    "boundary_proxy_score": round(score, 6),
                    "linker_token_fraction": round(linker_fraction, 6),
                    "repeat_fraction": round(repeat_fraction, 6),
                },
            )
        )
        if len(out) >= int(max_count):
            break
    return out


def grid_boundary_intervals(sequence_length, target_len):
    if sequence_length <= 0:
        return []
    centers = [
        int(round(sequence_length * 0.25)),
        int(round(sequence_length * 0.50)),
        int(round(sequence_length * 0.75)),
    ]
    out = []
    for idx, center in enumerate(centers, start=1):
        start, end = bounded_interval_around_center(sequence_length, center, target_len)
        out.append((start, end, {"boundary_type": "internal_grid_boundary", "grid_boundary_idx": idx}))
    return out


def choose_random_multi_intervals(sequence_length, total_len, interval_count, rng):
    total_len = min(sequence_length, max(interval_count, int(total_len)))
    interval_count = max(2, min(int(interval_count), 3, total_len))
    lengths = [max(1, total_len // interval_count) for _ in range(interval_count)]
    for idx in range(total_len - sum(lengths)):
        lengths[idx % interval_count] += 1
    for _attempt in range(200):
        intervals = []
        occupied = []
        ok = True
        for length in lengths:
            if sequence_length <= length:
                start = 0
            else:
                start = rng.randint(0, sequence_length - length)
            end = start + length - 1
            if any(interval_iou([(start, end)], [old]) > 0.0 for old in occupied):
                ok = False
                break
            occupied.append((start, end))
            intervals.append((start, end))
        if ok:
            return sorted(intervals)
    return choose_multi_intervals(sequence_length, total_len)


def read_risk_interval_rows(path):
    if not path:
        return {}
    rows_by_protein = defaultdict(list)
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"protein_id", "start", "end"}
        missing = sorted(required - fields)
        if missing:
            raise ValueError("{} missing risk interval columns: {}".format(path, ", ".join(missing)))
        for row in reader:
            protein_id = (row.get("protein_id") or "").strip()
            if not protein_id:
                continue
            try:
                start = int(float(row.get("start")))
                end = int(float(row.get("end")))
            except (TypeError, ValueError):
                continue
            rows_by_protein[protein_id].append(
                {
                    "start": start,
                    "end": end,
                    "risk_confidence": row.get("risk_confidence") or row.get("confidence") or "",
                    "risk_score": row.get("risk_score") or "",
                    "risk_source": row.get("risk_source") or row.get("feature_type") or row.get("source") or "",
                }
            )
    return rows_by_protein


def build_method_v1_candidate_pool(args, source_records, stage1_rows, risk_rows_by_protein):
    seed = effective_seed(args)
    rng = random.Random(seed)
    budget_fractions = parse_float_csv(args.method_budget_fractions, METHOD_V1_DEFAULT_BUDGET_FRACTIONS)
    rows = []
    stats = Counter()
    stage1_by_protein = defaultdict(list)
    for item in stage1_rows:
        stage1_by_protein[item["protein_id"]].append(item)

    requested = parse_deletion_generation_policies(args.deletion_generation_policies)
    if requested is None and not candidate_bank_enabled(args):
        allowed_policies = set(policy for policy in DELETION_GENERATION_POLICIES if policy != "uniform_random_interval")
    else:
        allowed_policies = set(requested or DELETION_GENERATION_POLICIES)

    for source in source_records:
        protein_id = source["protein_id"]
        sequence = source["sequence"]
        if not sequence:
            continue
        sequence_length = len(sequence)
        candidate_idx = 0

        if "stage1_proposal" in allowed_policies:
            for item in stage1_by_protein.get(protein_id, []):
                candidate_idx += 1
                priority_bonus = 0.0
                try:
                    priority_bonus = 10.0 * float(item.get("stage1_proposal_confidence") or 0.0)
                except (TypeError, ValueError):
                    priority_bonus = 0.0
                rows.append(
                    make_method_candidate_row(
                        protein_id=protein_id,
                        sequence=sequence,
                        intervals=[(item["start"], item["end"])],
                        policy="stage1_proposal",
                        seed=seed,
                        candidate_idx=candidate_idx,
                        budget_fraction=item["length"] / float(max(1, sequence_length)),
                        source="stage1_proposal_csv",
                        metadata_extra={
                            "stage1_interval_rank": item["stage1_interval_rank"],
                            "stage1_score_mode": item["stage1_score_mode"],
                            "selection_priority": METHOD_V1_POLICY_PRIORITY["stage1_proposal"] + priority_bonus,
                        },
                        source_stage1_row_id=str(item["row_idx"]),
                        source_start=str(item["start"]),
                        source_end=str(item["end"]),
                        source_score=item["stage1_proposal_confidence"],
                    )
                )
                stats["pool_stage1_proposal"] += 1

        if "budget_matched_random" in allowed_policies:
            for item in stage1_by_protein.get(protein_id, []):
                for _ in range(max(1, int(args.method_budget_matched_random_per_stage1))):
                    candidate_idx += 1
                    start, end = choose_random_interval_with_length(
                        sequence_length,
                        item["length"],
                        rng,
                        avoid_interval=(item["start"], item["end"]),
                    )
                    rows.append(
                        make_method_candidate_row(
                            protein_id=protein_id,
                            sequence=sequence,
                            intervals=[(start, end)],
                            policy="budget_matched_random",
                            seed=seed,
                            candidate_idx=candidate_idx,
                            budget_fraction=item["length"] / float(max(1, sequence_length)),
                            source="stage1_proposal_csv",
                            metadata_extra={
                                "matched_stage1_row_id": item["row_idx"],
                                "matched_stage1_start": item["start"],
                                "matched_stage1_end": item["end"],
                            },
                            source_stage1_row_id=str(item["row_idx"]),
                            source_start=str(item["start"]),
                            source_end=str(item["end"]),
                            source_score=item["stage1_proposal_confidence"],
                        )
                    )
                    stats["pool_budget_matched_random"] += 1

        for fraction in budget_fractions:
            target_len = bounded_deletion_length(
                sequence_length,
                fraction,
                args.min_random_deletion_len,
                args.method_max_single_interval_len,
            )
            for random_policy in ("uniform_random_interval", "random_interval"):
                if random_policy in allowed_policies:
                    used = set()
                    for random_idx in range(max(0, int(args.method_random_per_budget))):
                        for _attempt in range(100):
                            start, end = choose_random_deletion_interval(
                                sequence_length,
                                fraction,
                                args.min_random_deletion_len,
                                args.method_max_single_interval_len,
                                rng,
                            )
                            if (start, end) not in used:
                                used.add((start, end))
                                break
                        candidate_idx += 1
                        rows.append(
                            make_method_candidate_row(
                                protein_id=protein_id,
                                sequence=sequence,
                                intervals=[(start, end)],
                                policy=random_policy,
                                seed=seed,
                                candidate_idx=candidate_idx,
                                budget_fraction=fraction,
                                source=source["source"],
                                metadata_extra={
                                    "random_perturbation_index": random_idx + 1,
                                    "random_policy_semantics": "uniform_single_interval",
                                },
                            )
                        )
                        stats["pool_{}".format(random_policy)] += 1

            if "terminal_baseline" in allowed_policies:
                terminal_specs = [
                    ("n_terminus", 0, target_len - 1),
                    ("c_terminus", sequence_length - target_len, sequence_length - 1),
                ]
                for terminal_name, start, end in terminal_specs:
                    candidate_idx += 1
                    rows.append(
                        make_method_candidate_row(
                            protein_id=protein_id,
                            sequence=sequence,
                            intervals=[(start, end)],
                            policy="terminal_baseline",
                            seed=seed,
                            candidate_idx=candidate_idx,
                            budget_fraction=fraction,
                            source=source["source"],
                            metadata_extra={"terminal_type": terminal_name},
                        )
                    )
                    stats["pool_terminal_baseline"] += 1

            if "boundary_challenge" in allowed_policies:
                boundary_candidates = grid_boundary_intervals(sequence_length, target_len)
                boundary_candidates.extend(
                    low_complexity_boundary_intervals(
                        sequence,
                        target_len,
                        max(0, int(args.method_boundary_per_budget)),
                    )
                )
                for start, end, metadata in boundary_candidates[: max(0, int(args.method_boundary_per_budget))]:
                    candidate_idx += 1
                    rows.append(
                        make_method_candidate_row(
                            protein_id=protein_id,
                            sequence=sequence,
                            intervals=[(start, end)],
                            policy="boundary_challenge",
                            seed=seed,
                            candidate_idx=candidate_idx,
                            budget_fraction=fraction,
                            source=source["source"],
                            metadata_extra=metadata,
                        )
                    )
                    stats["pool_boundary_challenge"] += 1

            if "multi_interval_combination" in allowed_policies:
                total_len = bounded_deletion_length(
                    sequence_length,
                    fraction,
                    max(2, args.min_random_deletion_len),
                    args.method_max_multi_total_len,
                )
                for multi_idx in range(max(0, int(args.method_multi_per_budget))):
                    interval_count = 2 + (multi_idx % 2)
                    intervals = choose_random_multi_intervals(sequence_length, total_len, interval_count, rng)
                    candidate_idx += 1
                    rows.append(
                        make_method_candidate_row(
                            protein_id=protein_id,
                            sequence=sequence,
                            intervals=intervals,
                            policy="multi_interval_combination",
                            seed=seed,
                            candidate_idx=candidate_idx,
                            budget_fraction=fraction,
                            source=source["source"],
                            metadata_extra={
                                "multi_interval_count": len(intervals),
                                "multi_interval_index": multi_idx + 1,
                            },
                        )
                    )
                    stats["pool_multi_interval_combination"] += 1

            if "high_risk_negative" in allowed_policies:
                risk_rows = risk_rows_by_protein.get(protein_id, [])
                for risk_idx, risk_row in enumerate(risk_rows[: max(0, int(args.method_high_risk_per_budget))], start=1):
                    risk_start = max(0, min(sequence_length - 1, int(risk_row["start"])))
                    risk_end = max(risk_start, min(sequence_length - 1, int(risk_row["end"])))
                    center = 0.5 * (risk_start + risk_end)
                    start, end = bounded_interval_around_center(sequence_length, center, target_len)
                    candidate_idx += 1
                    rows.append(
                        make_method_candidate_row(
                            protein_id=protein_id,
                            sequence=sequence,
                            intervals=[(start, end)],
                            policy="high_risk_negative",
                            seed=seed,
                            candidate_idx=candidate_idx,
                            budget_fraction=fraction,
                            source="risk_interval_csv",
                            metadata_extra={
                                "risk_start": risk_start,
                                "risk_end": risk_end,
                                "risk_confidence": risk_row.get("risk_confidence", ""),
                                "risk_score": risk_row.get("risk_score", ""),
                                "risk_source": risk_row.get("risk_source", ""),
                                "hard_negative_index": risk_idx,
                            },
                        )
                    )
                    stats["pool_high_risk_negative"] += 1
    stats["candidate_pool_rows"] = len(rows)
    return rows, stats, allowed_policies


def dedupe_exact_intervals(rows):
    best = {}
    for row in rows:
        key = (
            row["protein_id"],
            row["budget_bin"],
            json.dumps(json.loads(row["deleted_intervals"]), separators=(",", ":")),
        )
        if key not in best or row_priority(row) > row_priority(best[key]):
            best[key] = row
    return list(best.values())


def count_overlap_duplicate_candidates(deduped_rows, selected_rows, nms_iou):
    selected_ids = set(row["perturbation_id"] for row in selected_rows)
    count = 0
    for row in deduped_rows:
        if row["perturbation_id"] in selected_ids:
            continue
        row_intervals = intervals_from_deletion_row(row)
        for selected in selected_rows:
            if row["protein_id"] != selected["protein_id"]:
                continue
            if row["budget_bin"] != selected["budget_bin"]:
                continue
            if row["num_intervals"] != selected["num_intervals"]:
                continue
            if interval_iou(row_intervals, intervals_from_deletion_row(selected)) > nms_iou:
                count += 1
                break
    return count


def select_with_nms(candidates, selected, nms_iou, terminal_limit, terminal_count):
    for row in candidates:
        if is_terminal_deletion(row) and terminal_count >= terminal_limit:
            continue
        intervals = intervals_from_deletion_row(row)
        if any(interval_iou(intervals, intervals_from_deletion_row(old)) > nms_iou for old in selected):
            continue
        selected.append(row)
        if is_terminal_deletion(row):
            terminal_count += 1
        return row, terminal_count
    return None, terminal_count


def select_method_v1_teacher_queries(args, pool_rows):
    quotas = effective_policy_quotas(args)
    nms_iou = effective_overlap_jaccard_max(args)
    max_per_protein = effective_max_candidates_per_protein(args)
    max_per_budget = effective_max_candidates_per_budget(args)
    terminal_fraction = max(0.0, min(1.0, float(args.method_terminal_max_fraction)))
    rows_by_protein_budget = defaultdict(lambda: defaultdict(list))
    deduped_pool_rows = dedupe_exact_intervals(pool_rows)
    for row in deduped_pool_rows:
        rows_by_protein_budget[row["protein_id"]][row["budget_bin"]].append(row)

    selected_rows = []
    stats = Counter()
    stats["raw_candidates"] = len(pool_rows)
    stats["exact_duplicates_dropped"] = max(0, len(pool_rows) - len(deduped_pool_rows))
    for protein_id in sorted(rows_by_protein_budget):
        protein_selected = []
        terminal_limit = max(1, int(math.floor(max_per_protein * terminal_fraction)))
        terminal_count = 0
        budget_bins = sorted(rows_by_protein_budget[protein_id])
        for budget_idx, budget_bin in enumerate(budget_bins):
            if len(protein_selected) >= max_per_protein:
                break
            budget_selected = []
            remaining_protein_slots = max_per_protein - len(protein_selected)
            remaining_budget_bins = max(1, len(budget_bins) - budget_idx)
            budget_cap = min(
                max_per_budget,
                max(1, int(math.ceil(remaining_protein_slots / float(remaining_budget_bins)))),
            )
            budget_rows = rows_by_protein_budget[protein_id][budget_bin]
            by_policy = defaultdict(list)
            for row in budget_rows:
                by_policy[row["generation_policy"]].append(row)
            for policy_rows in by_policy.values():
                policy_rows.sort(
                    key=lambda row: (
                        row_priority(row),
                        -row_target_budget_distance(row),
                        -row_deleted_length(row),
                        row["perturbation_id"],
                    ),
                    reverse=True,
                )

            if candidate_bank_enabled(args):
                for policy in [
                    "stage1_proposal",
                    "budget_matched_random",
                    "uniform_random_interval",
                    "terminal_baseline",
                    "multi_interval_combination",
                    "boundary_challenge",
                    "random_interval",
                    "high_risk_negative",
                ]:
                    if len(budget_selected) >= budget_cap or len(protein_selected) >= max_per_protein:
                        break
                    if int(quotas.get(policy, 0)) <= 0:
                        continue
                    picked, terminal_count = select_with_nms(
                        by_policy.get(policy, []),
                        protein_selected + budget_selected,
                        nms_iou,
                        terminal_limit,
                        terminal_count,
                    )
                    if picked is not None:
                        budget_selected.append(picked)
                        by_policy[policy].remove(picked)

            for policy in [
                "stage1_proposal",
                "high_risk_negative",
                "boundary_challenge",
                "budget_matched_random",
                "uniform_random_interval",
                "random_interval",
                "multi_interval_combination",
                "terminal_baseline",
            ]:
                quota = min(int(quotas.get(policy, 0)), budget_cap - len(budget_selected))
                for _ in range(max(0, quota)):
                    if len(budget_selected) >= budget_cap or len(protein_selected) >= max_per_protein:
                        break
                    picked, terminal_count = select_with_nms(
                        by_policy.get(policy, []),
                        protein_selected + budget_selected,
                        nms_iou,
                        terminal_limit,
                        terminal_count,
                    )
                    if picked is None:
                        break
                    budget_selected.append(picked)
                    by_policy[policy].remove(picked)

            leftovers = []
            for policy_rows in by_policy.values():
                leftovers.extend(policy_rows)
            leftovers.sort(key=lambda row: (row_priority(row), row["perturbation_id"]), reverse=True)
            while (
                leftovers
                and len(budget_selected) < budget_cap
                and len(protein_selected) + len(budget_selected) < max_per_protein
            ):
                picked, terminal_count = select_with_nms(
                    leftovers,
                    protein_selected + budget_selected,
                    nms_iou,
                    terminal_limit,
                    terminal_count,
                )
                if picked is None:
                    break
                budget_selected.append(picked)
                leftovers.remove(picked)

            protein_selected.extend(budget_selected)
            stats["selected_budget_bins"] += 1
        selected_rows.extend(protein_selected[:max_per_protein])
        stats["selected_proteins"] += 1
    stats["selected_rows"] = len(selected_rows)
    stats["overlap_duplicates_dropped"] = count_overlap_duplicate_candidates(
        deduped_pool_rows,
        selected_rows,
        nms_iou,
    )
    stats["cap_or_quota_dropped"] = max(
        0,
        len(deduped_pool_rows) - len(selected_rows) - int(stats["overlap_duplicates_dropped"]),
    )
    stats["selected_terminal_rows"] = sum(1 for row in selected_rows if is_terminal_deletion(row))
    for row in selected_rows:
        stats["selected_{}".format(row["generation_policy"])] += 1
    return selected_rows, stats


def iter_random_deletion_perturbations(
    fasta_path,
    seed,
    intervals_per_sequence,
    fraction,
    min_len,
    max_len,
    max_records=None,
):
    return iter_random_deletion_perturbations_from_sources(
        source_records=fasta_source_records(fasta_path, max_records=max_records),
        seed=seed,
        intervals_per_sequence=intervals_per_sequence,
        fraction=fraction,
        min_len=min_len,
        max_len=max_len,
        source_label="input_fasta",
    )


def iter_random_deletion_perturbations_from_sources(
    source_records,
    seed,
    intervals_per_sequence,
    fraction,
    min_len,
    max_len,
    source_label,
):
    rng = random.Random(seed)
    for source in source_records:
        protein_id = source["protein_id"]
        sequence = source["sequence"]
        if not sequence:
            continue
        used_intervals = set()
        for perturbation_idx in range(max(0, int(intervals_per_sequence))):
            for _attempt in range(100):
                start, end = choose_random_deletion_interval(len(sequence), fraction, min_len, max_len, rng)
                if (start, end) not in used_intervals:
                    used_intervals.add((start, end))
                    break
            perturbation_id = "{}__random__{}__{}_{}".format(
                safe_id(protein_id),
                perturbation_idx + 1,
                start,
                end,
            )
            metadata = {
                "source": source_label,
                "seed": seed,
                "random_perturbation_index": perturbation_idx + 1,
                "target_deletion_fraction": fraction,
            }
            if source.get("source_row_id") != "":
                metadata["source_row_id"] = source.get("source_row_id")
            yield deletion_row(
                protein_id=protein_id,
                sequence=sequence,
                intervals=[(start, end)],
                perturbation_id=perturbation_id,
                plan_id=perturbation_id,
                generation_policy="random_interval",
                metadata=metadata,
            )


def iter_budget_matched_random_perturbations(stage1_rows, seed):
    rng = random.Random(int(seed) + 1009)
    for item in stage1_rows:
        protein_id = item["protein_id"]
        sequence = item["sequence"]
        start, end = choose_random_interval_with_length(
            len(sequence),
            item["length"],
            rng,
            avoid_interval=(item["start"], item["end"]),
        )
        perturbation_id = "{}__budget_matched_random__row{}__{}_{}".format(
            safe_id(protein_id),
            item["row_idx"],
            start,
            end,
        )
        metadata = {
            "source": "stage1_proposal_csv",
            "matched_stage1_row_id": item["row_idx"],
            "matched_stage1_start": item["start"],
            "matched_stage1_end": item["end"],
            "matched_deleted_length": item["length"],
            "seed": seed,
        }
        yield deletion_row(
            protein_id=protein_id,
            sequence=sequence,
            intervals=[(start, end)],
            perturbation_id=perturbation_id,
            plan_id=perturbation_id,
            generation_policy="budget_matched_random",
            metadata=metadata,
            source_stage1_row_id=str(item["row_idx"]),
            source_start=str(item["start"]),
            source_end=str(item["end"]),
            source_score=item["stage1_proposal_confidence"],
        )


def iter_boundary_challenge_perturbations(source_records, fraction, min_len, max_len):
    for source in source_records:
        protein_id = source["protein_id"]
        sequence = source["sequence"]
        target_len = bounded_deletion_length(len(sequence), fraction, min_len, max_len)
        start = 1 if len(sequence) > target_len + 1 else 0
        end = start + target_len - 1
        perturbation_id = "{}__boundary_challenge__near_n__{}_{}".format(
            safe_id(protein_id),
            start,
            end,
        )
        metadata = {
            "source": source["source"],
            "boundary_type": "near_n_terminus",
            "target_deletion_fraction": fraction,
        }
        yield deletion_row(
            protein_id=protein_id,
            sequence=sequence,
            intervals=[(start, end)],
            perturbation_id=perturbation_id,
            plan_id=perturbation_id,
            generation_policy="boundary_challenge",
            metadata=metadata,
        )


def iter_terminal_baseline_perturbations(source_records, fraction, min_len, max_len):
    for source in source_records:
        protein_id = source["protein_id"]
        sequence = source["sequence"]
        target_len = bounded_deletion_length(len(sequence), fraction, min_len, max_len)
        terminal_specs = [
            ("n_terminus", 0, target_len - 1),
            ("c_terminus", len(sequence) - target_len, len(sequence) - 1),
        ]
        for terminal_name, start, end in terminal_specs:
            perturbation_id = "{}__terminal_baseline__{}__{}_{}".format(
                safe_id(protein_id),
                terminal_name,
                start,
                end,
            )
            metadata = {
                "source": source["source"],
                "terminal_type": terminal_name,
                "target_deletion_fraction": fraction,
            }
            yield deletion_row(
                protein_id=protein_id,
                sequence=sequence,
                intervals=[(start, end)],
                perturbation_id=perturbation_id,
                plan_id=perturbation_id,
                generation_policy="terminal_baseline",
                metadata=metadata,
            )


def choose_multi_intervals(sequence_length, total_len):
    total_len = min(sequence_length, max(2, int(total_len)))
    interval_count = 3 if total_len >= 6 and sequence_length >= total_len + 2 else 2
    interval_count = min(interval_count, total_len)
    base_len = max(1, total_len // interval_count)
    lengths = [base_len for _ in range(interval_count)]
    for idx in range(total_len - sum(lengths)):
        lengths[idx % interval_count] += 1
    available_gap = max(0, sequence_length - total_len)
    gap = available_gap // (interval_count + 1) if interval_count > 0 else 0
    intervals = []
    cursor = gap
    for length in lengths:
        start = min(cursor, sequence_length - length)
        if intervals and start <= intervals[-1][1]:
            start = intervals[-1][1] + 1
        end = start + length - 1
        intervals.append((start, end))
        cursor = end + 1 + gap
    return intervals


def iter_multi_interval_combination_perturbations(source_records, fraction, min_len, max_len):
    for source in source_records:
        protein_id = source["protein_id"]
        sequence = source["sequence"]
        total_len = bounded_deletion_length(len(sequence), fraction, max(2, min_len), max_len)
        intervals = choose_multi_intervals(len(sequence), total_len)
        compact = "_".join("{}-{}".format(start, end) for start, end in intervals)
        perturbation_id = "{}__multi_interval_combination__{}".format(
            safe_id(protein_id),
            compact,
        )
        metadata = {
            "source": source["source"],
            "target_deletion_fraction": fraction,
            "multi_interval_count": len(intervals),
        }
        yield deletion_row(
            protein_id=protein_id,
            sequence=sequence,
            intervals=intervals,
            perturbation_id=perturbation_id,
            plan_id=perturbation_id,
            generation_policy="multi_interval_combination",
            metadata=metadata,
        )


def write_deletion_perturbation_summary(
    path,
    args,
    rows,
    stats,
    requested_policies,
    skipped_policies,
    policy_warnings,
):
    if not path:
        return
    ensure_parent(path)
    row_counts = Counter(row["generation_policy"] for row in rows)
    construction_mode = str(getattr(args, "candidate_construction_mode", "legacy"))
    summary = {
        "schema_version": "deletion_perturbation_generation_summary_v1",
        "candidate_construction_mode": construction_mode,
        "total_rows": len(rows),
        "row_counts_by_generation_policy": dict(sorted(row_counts.items())),
        "coordinate_system": COORDINATE_SYSTEM,
        "unique_proteins": sorted(set(row["protein_id"] for row in rows)),
        "unique_budget_bins": sorted(set(row["budget_bin"] for row in rows)),
        "requested_generation_policies": list(requested_policies or []),
        "any_skipped_policies": list(skipped_policies),
        "any_policy_level_warnings": list(policy_warnings),
        "smoke": True,
        "proxy": True,
        "not_for_experiment": True,
        "no_training": True,
        "no_gpu": True,
        "no_srun": True,
        "teacher_scoring": False,
        "planner_run": False,
        "downloads": False,
        "high_risk_negative_requires_risk_annotations": True,
    }
    if construction_mode == "method_v1":
        summary.update({
            "candidate_pool_rows": int(stats.get("candidate_pool_rows", 0)),
            "selected_rows": int(stats.get("selected_rows", len(rows))),
            "selected_proteins": int(stats.get("selected_proteins", 0)),
            "selected_budget_bins": int(stats.get("selected_budget_bins", 0)),
            "selected_terminal_rows": int(stats.get("selected_terminal_rows", 0)),
            "pool_counts_by_generation_policy": {
                policy: int(stats.get("pool_{}".format(policy), 0))
                for policy in DELETION_GENERATION_POLICIES
            },
            "selected_counts_by_generation_policy": {
                policy: int(stats.get("selected_{}".format(policy), 0))
                for policy in DELETION_GENERATION_POLICIES
            },
            "method_budget_fractions": parse_float_csv(
                args.method_budget_fractions,
                METHOD_V1_DEFAULT_BUDGET_FRACTIONS,
            ),
            "method_policy_quotas": parse_policy_quota_map(args.method_policy_quotas),
            "method_max_teacher_candidates_per_protein": int(args.method_max_teacher_candidates_per_protein),
            "method_max_teacher_candidates_per_budget": int(args.method_max_teacher_candidates_per_budget),
            "method_nms_iou_threshold": float(args.method_nms_iou_threshold),
            "method_terminal_max_fraction": float(args.method_terminal_max_fraction),
            "risk_interval_csv": args.risk_interval_csv or "",
        })
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def candidate_bank_pairability(rows, min_pairable):
    grouped = defaultdict(list)
    policies_by_group = defaultdict(set)
    for row in rows:
        key = (row["protein_id"], row["budget_bin"])
        grouped[key].append(row)
        policies_by_group[key].add(row["generation_policy"])
    proteins_with_pairable = set()
    bins_with_min_candidates = 0
    bins_with_min_policies = 0
    pairable_pairs = 0
    for (protein_id, _budget_bin), group_rows in grouped.items():
        n = len(group_rows)
        if n >= int(min_pairable):
            bins_with_min_candidates += 1
            pairable_pairs += n * (n - 1) // 2
            proteins_with_pairable.add(protein_id)
        if len(policies_by_group[(protein_id, _budget_bin)]) >= 2:
            bins_with_min_policies += 1
    return {
        "proteins_with_pairable_budget_bins": sorted(proteins_with_pairable),
        "budget_bins_with_at_least_2_candidates": bins_with_min_candidates,
        "budget_bins_with_at_least_2_policies": bins_with_min_policies,
        "pairable_candidate_pairs_estimate": pairable_pairs,
    }


def candidate_bank_counts(rows):
    row_counts_by_policy = Counter(row["generation_policy"] for row in rows)
    row_counts_by_budget = Counter(row["budget_bin"] for row in rows)
    per_protein_counts = Counter(row["protein_id"] for row in rows)
    per_protein_budget_counts = defaultdict(dict)
    for row in rows:
        protein_id = row["protein_id"]
        budget_bin = row["budget_bin"]
        per_protein_budget_counts[protein_id][budget_bin] = per_protein_budget_counts[protein_id].get(budget_bin, 0) + 1
    return {
        "row_counts_by_generation_policy": dict(sorted(row_counts_by_policy.items())),
        "row_counts_by_budget_bin": dict(sorted(row_counts_by_budget.items())),
        "per_protein_counts": dict(sorted(per_protein_counts.items())),
        "per_protein_budget_bin_counts": {
            protein_id: dict(sorted(counts.items()))
            for protein_id, counts in sorted(per_protein_budget_counts.items())
        },
    }


def annotate_candidate_bank_rows(rows, args, risk_input_present):
    profile = str(args.candidate_bank_profile)
    risk_input = "present" if risk_input_present else "absent"
    for row in rows:
        update_metadata_json(
            row,
            {
                "candidate_bank_profile": profile,
                "generation_policy": row["generation_policy"],
                "raw_generation_policy": row["generation_policy"],
                "dedup_kept": True,
                "budget_bin": row["budget_bin"],
                "num_intervals": int(row["num_intervals"]),
                "risk_input": risk_input,
                "not_for_experiment": True,
                "smoke": True,
            },
        )


def filter_candidate_bank_rows(rows, args):
    kept = []
    stats = Counter()
    max_fraction = float(args.candidate_bank_max_deletion_fraction)
    for row in rows:
        try:
            intervals = intervals_from_deletion_row(row)
            wildtype_length = int(row["wildtype_length"])
            deleted_length = int(row["deleted_length"])
            deletion_frac = float(row["deletion_fraction"])
            mutant_sequence = row.get("mutant_sequence", "")
            normalize_deletion_intervals(intervals, wildtype_length, row["perturbation_id"])
            if deleted_length <= 0:
                raise ValueError("deleted_length <= 0")
            if deleted_length >= wildtype_length:
                raise ValueError("deleted_length >= wildtype_length")
            if deletion_frac > max_fraction:
                raise ValueError("deletion_fraction exceeds candidate bank max")
            if not mutant_sequence:
                raise ValueError("mutant_sequence is empty")
        except Exception:
            stats["invalid_candidates_dropped"] += 1
            continue
        kept.append(row)
    stats["valid_candidates_kept"] = len(kept)
    return kept, stats


def build_candidate_bank_audit(rows, summary, args, requested_policies, skipped_policies):
    errors = []
    warnings = []
    checks = {}
    required = set(DELETION_PERTURBATION_REQUIRED_COLUMNS)
    optional = set(DELETION_PERTURBATION_OPTIONAL_COLUMNS)
    checks["required_columns_present"] = all(required.issubset(set(row.keys())) for row in rows)
    if not checks["required_columns_present"]:
        errors.append("required deletion perturbation columns missing from one or more rows")

    checks["coordinate_system_closed_inclusive"] = all(row.get("coordinate_system") == COORDINATE_SYSTEM for row in rows)
    checks["intervals_valid"] = True
    checks["intervals_sorted_non_overlapping"] = True
    checks["deleted_length_consistent"] = True
    checks["deletion_fraction_consistent"] = True
    checks["mutant_sequence_reconstructs"] = True
    seen_ids = set()
    duplicate_ids = []
    seen_intervals = set()
    duplicate_intervals = []
    for row in rows:
        perturbation_id = row.get("perturbation_id", "")
        if perturbation_id in seen_ids:
            duplicate_ids.append(perturbation_id)
        seen_ids.add(perturbation_id)
        try:
            intervals = intervals_from_deletion_row(row)
            wildtype_length = int(row["wildtype_length"])
            normalized = normalize_deletion_intervals(intervals, wildtype_length, perturbation_id)
            if normalized != intervals:
                checks["intervals_sorted_non_overlapping"] = False
            computed_length = intervals_total_length(intervals)
            if computed_length != int(row["deleted_length"]):
                checks["deleted_length_consistent"] = False
            expected_fraction = computed_length / float(wildtype_length)
            if abs(float(row["deletion_fraction"]) - expected_fraction) > 1e-6:
                checks["deletion_fraction_consistent"] = False
            if row.get("wildtype_sequence") and row.get("mutant_sequence"):
                expected_mutant = mutant_from_deletion_intervals(row["wildtype_sequence"], intervals)
                if expected_mutant != row["mutant_sequence"]:
                    checks["mutant_sequence_reconstructs"] = False
            interval_key = (
                row["protein_id"],
                json.dumps([[start, end] for start, end in intervals], separators=(",", ":")),
            )
            if interval_key in seen_intervals:
                duplicate_intervals.append(interval_key)
            seen_intervals.add(interval_key)
        except Exception as exc:
            checks["intervals_valid"] = False
            errors.append("{} interval audit failed: {}".format(perturbation_id, exc))

    checks["perturbation_ids_unique"] = len(duplicate_ids) == 0
    if duplicate_ids:
        errors.append("duplicate perturbation_ids: {}".format(", ".join(sorted(duplicate_ids))))
    checks["plan_ids_nonempty"] = all(str(row.get("plan_id") or "").strip() for row in rows)
    checks["exact_duplicate_intervals_absent_per_protein"] = len(duplicate_intervals) == 0
    if duplicate_intervals:
        errors.append("exact duplicate deleted_intervals found within protein")

    max_per_protein = effective_max_candidates_per_protein(args)
    max_per_budget = effective_max_candidates_per_budget(args)
    caps_ok = True
    for protein_id, count in summary["per_protein_counts"].items():
        if int(count) > max_per_protein:
            caps_ok = False
            errors.append("{} exceeds max candidates per protein".format(protein_id))
    for protein_id, counts in summary["per_protein_budget_bin_counts"].items():
        for budget_bin, count in counts.items():
            if int(count) > max_per_budget:
                caps_ok = False
                errors.append("{} {} exceeds max candidates per budget bin".format(protein_id, budget_bin))
    checks["caps_respected"] = caps_ok

    expected_policies = set(requested_policies or []) - set(skipped_policies or [])
    expected_policies.discard("high_risk_negative")
    observed_policies = set(summary["row_counts_by_generation_policy"])
    missing_policies = sorted(policy for policy in expected_policies if policy not in observed_policies)
    checks["policy_coverage_nonempty"] = not missing_policies and bool(observed_policies)
    if missing_policies:
        errors.append("requested non-risk policies missing from final bank: {}".format(", ".join(missing_policies)))

    pairability = summary["pairability"]
    checks["pairability_nonempty"] = bool(pairability.get("proteins_with_pairable_budget_bins"))
    if not checks["pairability_nonempty"]:
        errors.append("no protein has a pairable budget bin")

    risk_input_present = bool(summary.get("input_sources", {}).get("risk_interval_csv"))
    checks["high_risk_negative_skipped_without_risk_input"] = (
        risk_input_present
        or "high_risk_negative" not in (requested_policies or [])
        or "high_risk_negative" in (skipped_policies or [])
    )
    if not checks["high_risk_negative_skipped_without_risk_input"]:
        errors.append("high_risk_negative was requested without risk input but was not recorded as skipped")
    if "high_risk_negative" in (skipped_policies or []) and not risk_input_present:
        warnings.append("high_risk_negative skipped because no risk annotations input was provided")

    for name, ok in list(checks.items()):
        if not ok and name not in ("policy_coverage_nonempty", "pairability_nonempty"):
            errors.append("audit check failed: {}".format(name))

    return {
        "schema_version": "candidate_bank_audit_v0",
        "status": "fail" if errors else "pass",
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def iter_formal_manifest_records(path, max_proteins=None):
    required = [
        "protein_id",
        "fasta_header",
        "sequence",
        "sequence_length",
        "sequence_sha1",
        "split",
        "length_bin",
        "source_fasta",
        "source_record_index",
        "manifest_rank",
    ]
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [field for field in required if field not in fieldnames]
        if missing:
            raise ValueError("{} missing manifest columns: {}".format(path, ", ".join(missing)))
        for row_idx, row in enumerate(reader, start=2):
            if max_proteins is not None and row_idx - 2 >= int(max_proteins):
                break
            protein_id = (row.get("protein_id") or "").strip()
            sequence = "".join((row.get("sequence") or "").split()).upper()
            if not protein_id:
                raise ValueError("{} row {}: protein_id is empty".format(path, row_idx))
            if not sequence:
                raise ValueError("{} row {}: sequence is empty".format(path, row_idx))
            length = parse_required_int(row.get("sequence_length"), "sequence_length", "{} row {}".format(path, row_idx))
            if length != len(sequence):
                raise ValueError("{} row {}: sequence_length does not match sequence".format(path, row_idx))
            yield {
                "protein_id": protein_id,
                "fasta_header": (row.get("fasta_header") or "").strip(),
                "sequence": sequence,
                "sequence_length": length,
                "sequence_sha1": (row.get("sequence_sha1") or "").strip(),
                "split": (row.get("split") or "").strip(),
                "length_bin": (row.get("length_bin") or "").strip(),
                "source_fasta": (row.get("source_fasta") or "").strip(),
                "source_record_index": (row.get("source_record_index") or "").strip(),
                "manifest_rank": parse_required_int(
                    row.get("manifest_rank"),
                    "manifest_rank",
                    "{} row {}".format(path, row_idx),
                ),
            }


def formal_target_len(sequence_length, budget_bin, policy=None):
    if sequence_length <= 1:
        return 1
    if budget_bin == "b001_single" or policy == "single_residue":
        return 1
    if policy == "short_span":
        target = max(2, min(int(round(0.02 * sequence_length)), 8))
    else:
        fraction = FORMAL_QUERY_BUDGET_FRACTIONS.get(budget_bin)
        if fraction is None:
            target = 1
        else:
            target = max(1, int(round(float(fraction) * sequence_length)))
    return min(sequence_length - 1, max(1, target))


def canonical_interval_key(intervals):
    normalized = sorted((int(start), int(end)) for start, end in intervals)
    return json.dumps([[start, end] for start, end in normalized], separators=(",", ":"))


def intervals_overlap_any(a_start, a_end, intervals):
    for old_start, old_end in intervals:
        if max(a_start, old_start) <= min(a_end, old_end):
            return True
    return False


def formal_random_single_interval(sequence_length, target_len, rng):
    target_len = min(sequence_length - 1, max(1, int(target_len)))
    start = rng.randint(0, sequence_length - target_len)
    return [(start, start + target_len - 1)], {}


def formal_single_residue_interval(sequence_length, rng):
    position = rng.randint(0, sequence_length - 1)
    return [(position, position)], {"single_residue_position": position}


def formal_short_span_interval(sequence_length, rng):
    target_len = formal_target_len(sequence_length, "b02", policy="short_span")
    intervals, metadata = formal_random_single_interval(sequence_length, target_len, rng)
    metadata["short_span_target_len"] = target_len
    return intervals, metadata


def formal_boundary_interval(sequence, target_len, ordinal):
    candidates = low_complexity_boundary_intervals(sequence, target_len, max_count=4)
    candidates.extend(grid_boundary_intervals(len(sequence), target_len))
    if candidates:
        start, end, metadata = candidates[int(ordinal) % len(candidates)]
        out_metadata = dict(metadata)
        out_metadata["boundary_proxy_semantics"] = "sequence_only_proxy_not_true_domain_boundary"
        return [(start, end)], out_metadata
    start, end = bounded_interval_around_center(len(sequence), len(sequence) * 0.5, target_len)
    return [(start, end)], {
        "boundary_type": "deterministic_center_fallback",
        "boundary_proxy_semantics": "sequence_only_proxy_not_true_domain_boundary",
        "fallback_reason": "no_boundary_proxy_candidate",
    }


def formal_low_complexity_or_boundary_interval(sequence, target_len, ordinal):
    candidates = low_complexity_boundary_intervals(sequence, target_len, max_count=8)
    if candidates:
        start, end, metadata = candidates[int(ordinal) % len(candidates)]
        out_metadata = dict(metadata)
        out_metadata["low_complexity_proxy_semantics"] = "sequence_only_linker_or_repeat_proxy"
        return [(start, end)], out_metadata
    grid = grid_boundary_intervals(len(sequence), target_len)
    if grid:
        start, end, metadata = grid[int(ordinal) % len(grid)]
        out_metadata = dict(metadata)
        out_metadata["fallback_reason"] = "no_low_complexity_window"
        out_metadata["low_complexity_proxy_semantics"] = "sequence_only_linker_or_repeat_proxy"
        return [(start, end)], out_metadata
    intervals, metadata = formal_random_single_interval(
        len(sequence),
        target_len,
        random.Random(stable_int("{}|{}|{}".format(sequence[:32], target_len, ordinal))),
    )
    metadata["fallback_reason"] = "no_low_complexity_or_grid_candidate"
    metadata["low_complexity_proxy_semantics"] = "sequence_only_linker_or_repeat_proxy"
    return intervals, metadata


def formal_terminal_interval(sequence_length, target_len, rng, ordinal):
    use_n = (int(ordinal) + rng.randint(0, 1)) % 2 == 0
    target_len = min(sequence_length - 1, max(1, int(target_len)))
    if use_n:
        return [(0, target_len - 1)], {"terminal_type": "n_terminus"}
    return [(sequence_length - target_len, sequence_length - 1)], {"terminal_type": "c_terminus"}


def formal_multi_intervals(sequence_length, target_len, rng):
    if target_len < 2 or sequence_length < 4:
        intervals, metadata = formal_random_single_interval(sequence_length, target_len, rng)
        metadata["fallback_reason"] = "sequence_too_short_for_multi_interval"
        return intervals, metadata
    interval_count = 3 if target_len >= 6 and sequence_length >= 12 and rng.random() < 0.5 else 2
    interval_count = min(interval_count, target_len, max(2, sequence_length // 2))
    intervals = choose_random_multi_intervals(sequence_length, target_len, interval_count, rng)
    if len(intervals) < 2:
        intervals = choose_multi_intervals(sequence_length, target_len)
    metadata = {"multi_interval_count": len(intervals)}
    if len(intervals) < 2:
        metadata["fallback_reason"] = "multi_interval_constructor_returned_single_interval"
    return intervals, metadata


def formal_candidate_intervals(sequence, budget_bin, policy, ordinal, rng):
    length = len(sequence)
    target_len = formal_target_len(length, budget_bin, policy=policy)
    if policy == "single_residue":
        return formal_single_residue_interval(length, rng)
    if policy == "short_span":
        return formal_short_span_interval(length, rng)
    if policy == "uniform_random_interval":
        return formal_random_single_interval(length, target_len, rng)
    if policy == "boundary_challenge":
        return formal_boundary_interval(sequence, target_len, ordinal)
    if policy == "low_complexity_or_boundary":
        return formal_low_complexity_or_boundary_interval(sequence, target_len, ordinal)
    if policy == "terminal_baseline":
        return formal_terminal_interval(length, target_len, rng, ordinal)
    if policy == "multi_interval_combination":
        return formal_multi_intervals(length, target_len, rng)
    raise ValueError("unknown formal generation policy: {}".format(policy))


def formal_fallback_intervals(sequence_length, target_len, used_keys):
    target_len = min(sequence_length - 1, max(1, int(target_len)))
    candidate_lengths = [target_len]
    for delta in range(1, min(sequence_length - 1, 8) + 1):
        if target_len - delta >= 1:
            candidate_lengths.append(target_len - delta)
        if target_len + delta <= sequence_length - 1:
            candidate_lengths.append(target_len + delta)
    for length in candidate_lengths:
        for start in range(0, sequence_length - length + 1):
            intervals = [(start, start + length - 1)]
            key = canonical_interval_key(intervals)
            if key not in used_keys:
                return intervals, {"fallback_reason": "dedupe_deterministic_scan"}
    for total_len in range(2, sequence_length):
        intervals = choose_multi_intervals(sequence_length, total_len)
        key = canonical_interval_key(intervals)
        if key not in used_keys:
            return intervals, {"fallback_reason": "dedupe_multi_interval_scan"}
    return None, {"fallback_reason": "no_unique_interval_available"}


def formal_perturbation_id(protein_id, generation_policy, budget_bin, candidate_index, intervals):
    return "{}__formalqv1__{}__{}__{:03d}__{}".format(
        safe_id(protein_id),
        safe_id(generation_policy),
        safe_id(budget_bin),
        int(candidate_index),
        safe_id(compact_intervals(intervals)),
    )


def make_formal_query_row(manifest_record, intervals, budget_bin, generation_policy, candidate_index, args, metadata_extra):
    protein_id = manifest_record["protein_id"]
    sequence = manifest_record["sequence"]
    intervals = normalize_deletion_intervals(
        intervals,
        len(sequence),
        "{} {} {}".format(protein_id, budget_bin, generation_policy),
    )
    perturbation_id = formal_perturbation_id(
        protein_id,
        generation_policy,
        budget_bin,
        candidate_index,
        intervals,
    )
    metadata = {
        "query_profile": str(args.query_profile),
        "teacher_query_bootstrap": True,
        "final_search_space": False,
        "formal_training_set": True,
        "source_manifest_path": str(args.input_manifest_csv),
        "source_manifest_rank": int(manifest_record["manifest_rank"]),
        "source_sequence_sha1": manifest_record["sequence_sha1"],
        "source_length_bin": manifest_record["length_bin"],
        "source_split": manifest_record["split"],
        "source_fasta": manifest_record["source_fasta"],
        "source_record_index": manifest_record["source_record_index"],
        "seed_tag": str(args.seed_tag),
        "target_delete_fraction": FORMAL_QUERY_BUDGET_FRACTIONS.get(budget_bin),
        "budget_bin": budget_bin,
        "no_risk_input": True,
        "risk_input": "absent",
        "candidate_index_within_protein": int(candidate_index),
        "policy_quota_name": "{}:{}".format(budget_bin, generation_policy),
        "coordinate_system": COORDINATE_SYSTEM,
    }
    metadata.update(metadata_extra or {})
    row = deletion_row(
        protein_id=protein_id,
        sequence=sequence,
        intervals=intervals,
        perturbation_id=perturbation_id,
        plan_id=perturbation_id,
        generation_policy=generation_policy,
        metadata=metadata,
    )
    row["budget_bin"] = budget_bin
    update_metadata_json(row, {"actual_deletion_fraction": float(row["deletion_fraction"])})
    return row


def generate_formal_rows_for_protein(manifest_record, args, stats):
    sequence = manifest_record["sequence"]
    if len(sequence) < 2:
        stats["underfilled_sequence_too_short"] += 1
        return [], [{
            "protein_id": manifest_record["protein_id"],
            "reason": "sequence_too_short",
            "sequence_length": len(sequence),
        }]

    rows = []
    used_keys = set()
    underfilled = []
    candidate_index = 0
    ordinal = 0
    for budget_bin, policy, count in FORMAL_QUERY_POLICY_PLAN:
        for quota_idx in range(int(count)):
            if len(rows) >= int(args.candidates_per_protein):
                break
            ordinal += 1
            rng = formal_rng(
                args.seed_tag,
                manifest_record["protein_id"],
                manifest_record["sequence_sha1"],
                budget_bin,
                policy,
                quota_idx,
            )
            metadata_extra = {
                "profile_ordinal": ordinal,
                "quota_index": quota_idx + 1,
                "quota_target_count": int(count),
            }
            intervals = None
            for attempt in range(80):
                attempt_rng = formal_rng(
                    args.seed_tag,
                    manifest_record["protein_id"],
                    manifest_record["sequence_sha1"],
                    budget_bin,
                    policy,
                    "{}:{}".format(quota_idx, attempt),
                )
                candidate_intervals, generated_metadata = formal_candidate_intervals(
                    sequence,
                    budget_bin,
                    policy,
                    ordinal + attempt,
                    attempt_rng if attempt else rng,
                )
                candidate_intervals = normalize_deletion_intervals(
                    candidate_intervals,
                    len(sequence),
                    "formal candidate attempt",
                )
                if intervals_total_length(candidate_intervals) >= len(sequence):
                    continue
                key = canonical_interval_key(candidate_intervals)
                if key not in used_keys:
                    intervals = candidate_intervals
                    metadata_extra.update(generated_metadata)
                    if attempt > 0:
                        metadata_extra["dedupe_resample_attempts"] = attempt
                    break
                stats["duplicate_candidates_skipped"] += 1
            if intervals is None:
                target_len = formal_target_len(len(sequence), budget_bin, policy=policy)
                intervals, fallback_metadata = formal_fallback_intervals(len(sequence), target_len, used_keys)
                metadata_extra.update(fallback_metadata)
                stats["fallback_{}".format(fallback_metadata.get("fallback_reason", "unknown"))] += 1
            if intervals is None:
                underfilled.append({
                    "protein_id": manifest_record["protein_id"],
                    "budget_bin": budget_bin,
                    "generation_policy": policy,
                    "reason": "no_unique_interval_available",
                })
                continue
            used_keys.add(canonical_interval_key(intervals))
            candidate_index += 1
            row = make_formal_query_row(
                manifest_record,
                intervals,
                budget_bin,
                policy,
                candidate_index,
                args,
                metadata_extra,
            )
            rows.append(row)
        if len(rows) >= int(args.candidates_per_protein):
            break
    if len(rows) < int(args.candidates_per_protein):
        underfilled.append({
            "protein_id": manifest_record["protein_id"],
            "reason": "unique_candidate_count_below_target",
            "sequence_length": len(sequence),
            "rows_generated": len(rows),
            "target": int(args.candidates_per_protein),
        })
    return rows, underfilled


def formal_shard_path(output_query_dir, shard_index):
    return os.path.join(
        output_query_dir,
        "shards",
        "deletion_perturbations_shard_{:06d}.csv".format(int(shard_index)),
    )


def safe_remove_formal_outputs(output_query_dir):
    if not output_query_dir:
        raise ValueError("output_query_dir is required")
    os.makedirs(output_query_dir, exist_ok=True)
    patterns = [
        os.path.join(output_query_dir, "deletion_perturbations_shard_index.csv"),
        os.path.join(output_query_dir, "deletion_query_summary.json"),
        os.path.join(output_query_dir, "deletion_query_policy_config.json"),
        os.path.join(output_query_dir, "*.tmp"),
        os.path.join(output_query_dir, "shards", "deletion_perturbations_shard_*.csv"),
        os.path.join(output_query_dir, "shards", "*.tmp"),
    ]
    removed = 0
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.isfile(path):
                os.remove(path)
                removed += 1
    return removed


def validate_existing_formal_shard(path, expected_proteins, candidates_per_protein):
    if not os.path.isfile(path):
        return False, {"reason": "missing"}
    try:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if set(FORMAL_QUERY_FIELDNAMES) - set(reader.fieldnames or []):
                return False, {"reason": "missing_columns"}
            rows = 0
            proteins = set()
            for row in reader:
                rows += 1
                proteins.add((row.get("protein_id") or "").strip())
                if (row.get("coordinate_system") or "").strip() != COORDINATE_SYSTEM:
                    return False, {"reason": "bad_coordinate_system"}
                if is_terminal_deletion(row) and not row.get("deleted_intervals"):
                    return False, {"reason": "bad_deleted_intervals"}
            expected_rows = int(expected_proteins) * int(candidates_per_protein)
            if rows != expected_rows:
                return False, {"reason": "row_count_mismatch", "rows": rows, "expected_rows": expected_rows}
            if len(proteins) != int(expected_proteins):
                return False, {"reason": "protein_count_mismatch", "proteins": len(proteins), "expected_proteins": expected_proteins}
            return True, {"rows": rows, "proteins": len(proteins)}
    except Exception as exc:
        return False, {"reason": "exception", "error": str(exc)}


def write_formal_shard(path, rows):
    ensure_parent(path)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORMAL_QUERY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def formal_summary_json_from_counts(args, stats, shard_records, underfilled_examples):
    total = int(stats.get("total_candidates_written", 0))
    terminal_count = int(stats.get("terminal_candidate_count", 0))
    multi_count = int(stats.get("multi_interval_candidate_count", 0))
    single_count = int(stats.get("single_residue_candidate_count", 0))
    return {
        "schema_version": "formal_deletion_query_summary_v1",
        "input_manifest_path": str(args.input_manifest_csv),
        "output_query_dir": str(args.output_query_dir),
        "query_profile": str(args.query_profile),
        "seed_tag": str(args.seed_tag),
        "coordinate_system": COORDINATE_SYSTEM,
        "num_manifest_rows_read": int(stats.get("num_manifest_rows_read", 0)),
        "num_proteins_processed": int(stats.get("num_proteins_processed", 0)),
        "candidates_per_protein_target": int(args.candidates_per_protein),
        "total_candidates_written": total,
        "num_shards_written": len(shard_records),
        "proteins_per_shard": int(args.proteins_per_shard),
        "underfilled_proteins": int(stats.get("underfilled_proteins", 0)),
        "underfilled_examples": list(underfilled_examples)[:20],
        "budget_bin_counts": dict(sorted(stats.get("budget_bin_counts", Counter()).items())),
        "generation_policy_counts": dict(sorted(stats.get("generation_policy_counts", Counter()).items())),
        "length_bin_counts": dict(sorted(stats.get("length_bin_counts", Counter()).items())),
        "split_counts": dict(sorted(stats.get("split_counts", Counter()).items())),
        "terminal_candidate_fraction": terminal_count / float(total) if total else 0.0,
        "multi_interval_candidate_fraction": multi_count / float(total) if total else 0.0,
        "single_residue_candidate_fraction": single_count / float(total) if total else 0.0,
        "duplicate_candidates_skipped": int(stats.get("duplicate_candidates_skipped", 0)),
        "fallback_counts": {
            key[len("fallback_"):]: int(value)
            for key, value in sorted(stats.items())
            if str(key).startswith("fallback_")
        },
        "validation_errors": [],
        "deterministic_rules": {
            "rng_seed": "sha1(seed_tag|protein_id|sequence_sha1|budget_bin|policy|ordinal)",
            "coordinate_system": COORDINATE_SYSTEM,
            "budget_plan": FORMAL_QUERY_POLICY_PLAN,
            "no_risk_input": True,
            "teacher_scoring": False,
            "training": False,
            "planner_run": False,
        },
        "output_paths": {
            "shard_index_csv": os.path.join(args.output_query_dir, "deletion_perturbations_shard_index.csv"),
            "summary_json": os.path.join(args.output_query_dir, "deletion_query_summary.json"),
            "policy_config_json": os.path.join(args.output_query_dir, "deletion_query_policy_config.json"),
            "shards_dir": os.path.join(args.output_query_dir, "shards"),
        },
        "not_for_experiment_result": True,
        "teacher_query_bootstrap": True,
        "final_search_space": False,
        "no_training": True,
        "no_teacher_scoring": True,
        "no_gpu": True,
        "no_srun": True,
        "downloads": False,
    }


def write_formal_policy_config(args):
    path = os.path.join(args.output_query_dir, "deletion_query_policy_config.json")
    config = {
        "schema_version": "formal_deletion_query_policy_config_v1",
        "query_profile": str(args.query_profile),
        "seed_tag": str(args.seed_tag),
        "coordinate_system": COORDINATE_SYSTEM,
        "candidates_per_protein": int(args.candidates_per_protein),
        "proteins_per_shard": int(args.proteins_per_shard),
        "budget_fractions": FORMAL_QUERY_BUDGET_FRACTIONS,
        "policy_plan": [
            {
                "budget_bin": budget_bin,
                "generation_policy": policy,
                "count": count,
            }
            for budget_bin, policy, count in FORMAL_QUERY_POLICY_PLAN
        ],
        "risk_input": "absent",
        "high_risk_negative_enabled": False,
        "stage1_proposal_enabled": False,
        "teacher_scoring": False,
        "training": False,
        "planner_run": False,
    }
    ensure_parent(path)
    with open(path, "w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_formal_shard_index(args, shard_records, summary_path):
    path = os.path.join(args.output_query_dir, "deletion_perturbations_shard_index.csv")
    ensure_parent(path)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORMAL_QUERY_SHARD_INDEX_FIELDS)
        writer.writeheader()
        for record in shard_records:
            out = dict(record)
            out["summary_json"] = summary_path
            writer.writerow(out)
    os.replace(tmp_path, path)
    return path


def update_formal_stats(stats, manifest_record, rows, underfilled):
    stats["num_manifest_rows_read"] += 1
    stats["num_proteins_processed"] += 1
    stats["length_bin_counts"][manifest_record["length_bin"]] += 1
    stats["split_counts"][manifest_record["split"]] += 1
    if underfilled:
        stats["underfilled_proteins"] += 1
    for row in rows:
        stats["total_candidates_written"] += 1
        stats["budget_bin_counts"][row["budget_bin"]] += 1
        stats["generation_policy_counts"][row["generation_policy"]] += 1
        if is_terminal_deletion(row):
            stats["terminal_candidate_count"] += 1
        if int(row["num_intervals"]) > 1:
            stats["multi_interval_candidate_count"] += 1
        if row["generation_policy"] == "single_residue":
            stats["single_residue_candidate_count"] += 1


def build_formal_deletion_queries(args):
    if not args.input_manifest_csv:
        raise ValueError("--input_manifest_csv is required with --build_formal_deletion_queries")
    if not args.output_query_dir:
        raise ValueError("--output_query_dir is required with --build_formal_deletion_queries")
    if args.query_profile != FORMAL_QUERY_PROFILE:
        raise ValueError("unsupported --query_profile {!r}; expected {}".format(args.query_profile, FORMAL_QUERY_PROFILE))
    if int(args.candidates_per_protein) != sum(count for _budget, _policy, count in FORMAL_QUERY_POLICY_PLAN):
        raise ValueError("formal_cold_start_v1 requires --candidates_per_protein 32")
    os.makedirs(args.output_query_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_query_dir, "shards"), exist_ok=True)
    if args.overwrite:
        removed = safe_remove_formal_outputs(args.output_query_dir)
    else:
        removed = 0
    policy_config_path = write_formal_policy_config(args)

    stats = Counter()
    stats["budget_bin_counts"] = Counter()
    stats["generation_policy_counts"] = Counter()
    stats["length_bin_counts"] = Counter()
    stats["split_counts"] = Counter()
    shard_records = []
    underfilled_examples = []
    current_rows = []
    current_manifest_records = []
    skipped_existing_shards = 0
    shard_index = 0

    def flush_shard():
        nonlocal current_rows, current_manifest_records, shard_index, skipped_existing_shards
        if not current_manifest_records:
            return
        path = formal_shard_path(args.output_query_dir, shard_index)
        num_proteins = len(current_manifest_records)
        num_candidates = len(current_rows)
        first_rank = current_manifest_records[0]["manifest_rank"]
        last_rank = current_manifest_records[-1]["manifest_rank"]
        status = "written"
        if args.resume:
            valid, _info = validate_existing_formal_shard(path, num_proteins, int(args.candidates_per_protein))
            if valid:
                skipped_existing_shards += 1
                status = "resumed_existing"
            else:
                write_formal_shard(path, current_rows)
        else:
            write_formal_shard(path, current_rows)
        metadata = {
            "coordinate_system": COORDINATE_SYSTEM,
            "first_protein_id": current_manifest_records[0]["protein_id"],
            "last_protein_id": current_manifest_records[-1]["protein_id"],
            "teacher_scoring": False,
            "training": False,
            "no_risk_input": True,
        }
        shard_records.append({
            "shard_id": "deletion_perturbations_shard_{:06d}".format(shard_index),
            "shard_index": shard_index,
            "shard_path": path,
            "input_manifest_path": str(args.input_manifest_csv),
            "first_manifest_rank": first_rank,
            "last_manifest_rank": last_rank,
            "num_proteins": num_proteins,
            "num_candidates": num_candidates,
            "candidates_per_protein_target": int(args.candidates_per_protein),
            "query_profile": str(args.query_profile),
            "status": status,
            "summary_json": "",
            "metadata_json": json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        })
        if shard_index % 10 == 0:
            print("formal shard {:06d}: {} proteins, {} candidates, {}".format(
                shard_index,
                num_proteins,
                num_candidates,
                status,
            ))
        current_rows = []
        current_manifest_records = []
        shard_index += 1

    for manifest_record in iter_formal_manifest_records(args.input_manifest_csv, max_proteins=args.max_proteins):
        rows, underfilled = generate_formal_rows_for_protein(manifest_record, args, stats)
        if underfilled and len(underfilled_examples) < 20:
            underfilled_examples.extend(underfilled[: 20 - len(underfilled_examples)])
        update_formal_stats(stats, manifest_record, rows, underfilled)
        current_rows.extend(rows)
        current_manifest_records.append(manifest_record)
        if len(current_manifest_records) >= int(args.proteins_per_shard):
            flush_shard()
    flush_shard()

    summary = formal_summary_json_from_counts(args, stats, shard_records, underfilled_examples)
    summary["overwrite_removed_files"] = removed
    summary["resume_skipped_existing_shards"] = skipped_existing_shards
    summary_path = os.path.join(args.output_query_dir, "deletion_query_summary.json")
    ensure_parent(summary_path)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    shard_index_path = write_formal_shard_index(args, shard_records, summary_path)
    return {
        "num_manifest_rows_read": int(stats.get("num_manifest_rows_read", 0)),
        "num_proteins_processed": int(stats.get("num_proteins_processed", 0)),
        "total_candidates_written": int(stats.get("total_candidates_written", 0)),
        "num_shards_written": len(shard_records),
        "underfilled_proteins": int(stats.get("underfilled_proteins", 0)),
        "output_query_dir": args.output_query_dir,
        "summary_json": summary_path,
        "shard_index_csv": shard_index_path,
        "policy_config_json": policy_config_path,
        "resume_skipped_existing_shards": skipped_existing_shards,
    }


def write_candidate_bank_outputs(
    args,
    rows,
    stats,
    requested_policies,
    skipped_policies,
    policy_warnings,
):
    if not candidate_bank_enabled(args):
        return
    counts = candidate_bank_counts(rows)
    pairability = candidate_bank_pairability(rows, args.candidate_bank_min_pairable_per_budget_bin)
    warnings = list(policy_warnings or [])
    skipped = list(skipped_policies or [])
    if "high_risk_negative" in skipped and not args.risk_interval_csv:
        warnings.append("high_risk_negative skipped because no risk annotations input was provided")
    summary = {
        "schema_version": "candidate_bank_summary_v0",
        "status": "pass",
        "not_for_experiment": True,
        "smoke": True,
        "no_training": True,
        "no_teacher_scoring": True,
        "no_gpu": True,
        "no_srun": True,
        "coordinate_system": COORDINATE_SYSTEM,
        "input_sources": {
            "stage1_proposal_csv": args.stage1_proposal_csv or "",
            "source_fasta": args.input_fasta or "",
            "risk_interval_csv": args.risk_interval_csv or "",
        },
        "profile": str(args.candidate_bank_profile),
        "candidate_bank_controls": {
            "max_candidates_per_protein": effective_max_candidates_per_protein(args),
            "max_candidates_per_budget_bin": effective_max_candidates_per_budget(args),
            "max_candidates_per_policy_per_protein": int(args.candidate_bank_max_candidates_per_policy_per_protein),
            "overlap_jaccard_max": effective_overlap_jaccard_max(args),
            "min_pairable_per_budget_bin": int(args.candidate_bank_min_pairable_per_budget_bin),
            "max_deletion_fraction": float(args.candidate_bank_max_deletion_fraction),
            "seed": effective_seed(args),
            "policy_quotas_per_budget": effective_policy_quotas(args),
        },
        "total_rows": len(rows),
        "unique_proteins": sorted(set(row["protein_id"] for row in rows)),
        "pairability": pairability,
        "dedup": {
            "raw_candidates": int(stats.get("raw_candidates", stats.get("candidate_pool_rows", 0))),
            "exact_duplicates_dropped": int(stats.get("exact_duplicates_dropped", 0)),
            "overlap_duplicates_dropped": int(stats.get("overlap_duplicates_dropped", 0)),
            "invalid_candidates_dropped": int(stats.get("invalid_candidates_dropped", 0)),
            "cap_or_quota_dropped": int(stats.get("cap_or_quota_dropped", 0)),
        },
        "skipped_policies": skipped,
        "warnings": warnings,
    }
    summary.update(counts)
    audit = build_candidate_bank_audit(rows, summary, args, requested_policies, skipped)
    summary["status"] = audit["status"]
    if audit["errors"]:
        summary["warnings"] = warnings + ["candidate_bank_audit_failed"]
    if args.candidate_bank_summary_json:
        ensure_parent(args.candidate_bank_summary_json)
        with open(args.candidate_bank_summary_json, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
    if args.candidate_bank_audit_json:
        ensure_parent(args.candidate_bank_audit_json)
        with open(args.candidate_bank_audit_json, "w") as handle:
            json.dump(audit, handle, indent=2, sort_keys=True)
            handle.write("\n")


def build_deletion_perturbations(args):
    if not args.output_deletion_perturbations_csv:
        raise ValueError("--output_deletion_perturbations_csv is required with --build_deletion_perturbations")
    rows = []
    stats = Counter()
    requested_policies = parse_deletion_generation_policies(args.deletion_generation_policies)
    skipped_policies = []
    policy_warnings = []
    stage1_rows = []
    stage1_source_records = []
    if args.stage1_proposal_csv:
        stage1_rows, stage1_source_records = read_stage1_deletion_sources(
            args.stage1_proposal_csv,
            max_rows=args.max_stage1_rows,
        )
    fasta_records = []
    if args.input_fasta:
        fasta_records = fasta_source_records(
            args.input_fasta,
            max_records=args.max_deletion_source_records,
        )
    stage1_source_protein_ids = set(item["protein_id"] for item in stage1_source_records)
    source_records = stage1_source_records + [
        record
        for record in fasta_records
        if record["protein_id"] not in stage1_source_protein_ids
    ]

    construction_mode = str(getattr(args, "candidate_construction_mode", "legacy")).strip().lower()
    if candidate_bank_enabled(args):
        construction_mode = "method_v1"
    if construction_mode == "method_v1":
        if not source_records:
            raise ValueError(
                "no source sequences available for method_v1 candidate construction; "
                "provide --stage1_proposal_csv or --input_fasta"
            )
        risk_rows_by_protein = read_risk_interval_rows(args.risk_interval_csv)
        pool_rows, pool_stats, allowed_policies = build_method_v1_candidate_pool(
            args,
            source_records,
            stage1_rows,
            risk_rows_by_protein,
        )
        if candidate_bank_enabled(args):
            pool_rows, filter_stats = filter_candidate_bank_rows(pool_rows, args)
            pool_stats.update(filter_stats)
        rows, selection_stats = select_method_v1_teacher_queries(args, pool_rows)
        stats.update(pool_stats)
        stats.update(selection_stats)
        if candidate_bank_enabled(args):
            annotate_candidate_bank_rows(rows, args, bool(risk_rows_by_protein))
        for row in rows:
            stats[row["generation_policy"]] += 1
        requested_policies = sorted(allowed_policies)
        if "high_risk_negative" in allowed_policies and not risk_rows_by_protein:
            skipped_policies.append("high_risk_negative")
            policy_warnings.append(
                "high_risk_negative produced no rows because --risk_interval_csv was not provided"
            )
        explicitly_requested = set(parse_deletion_generation_policies(args.deletion_generation_policies) or [])
        for policy in sorted(explicitly_requested):
            if stats.get("pool_{}".format(policy), 0) == 0 and policy not in skipped_policies:
                skipped_policies.append(policy)
                policy_warnings.append("{} produced no candidate-pool rows".format(policy))
        if pool_stats.get("candidate_pool_rows", 0) == 0:
            raise ValueError("method_v1 candidate construction produced an empty candidate pool")
    elif construction_mode != "legacy":
        raise ValueError("unknown candidate_construction_mode {!r}".format(construction_mode))

    if construction_mode == "legacy" and requested_policies is None and args.stage1_proposal_csv:
        for row in iter_stage1_deletion_perturbations(args.stage1_proposal_csv, max_rows=args.max_stage1_rows):
            rows.append(row)
            stats["stage1_proposal"] += 1
    if (
        construction_mode == "legacy"
        and requested_policies is None
        and args.input_fasta
        and int(args.random_intervals_per_sequence) > 0
    ):
        for row in iter_random_deletion_perturbations(
            fasta_path=args.input_fasta,
            seed=int(args.seed if args.seed is not None else DEFAULT_CONFIG["seed"]),
            intervals_per_sequence=args.random_intervals_per_sequence,
            fraction=args.random_deletion_fraction,
            min_len=args.min_random_deletion_len,
            max_len=args.max_random_deletion_len,
            max_records=args.max_deletion_source_records,
        ):
            rows.append(row)
            stats["random_interval"] += 1
    if construction_mode == "legacy" and requested_policies is not None:
        seed = effective_seed(args)
        for policy in requested_policies:
            before = len(rows)
            if policy == "stage1_proposal":
                if not args.stage1_proposal_csv:
                    skipped_policies.append(policy)
                    policy_warnings.append("stage1_proposal skipped because --stage1_proposal_csv was not provided")
                else:
                    for row in iter_stage1_deletion_perturbations(args.stage1_proposal_csv, max_rows=args.max_stage1_rows):
                        rows.append(row)
                        stats[policy] += 1
            elif policy == "random_interval":
                if not source_records:
                    skipped_policies.append(policy)
                    policy_warnings.append("random_interval skipped because no input_fasta or Stage1 proposal sequences were available")
                elif int(args.random_intervals_per_sequence) <= 0:
                    skipped_policies.append(policy)
                    policy_warnings.append("random_interval skipped because --random_intervals_per_sequence <= 0")
                else:
                    for row in iter_random_deletion_perturbations_from_sources(
                        source_records=source_records,
                        seed=seed,
                        intervals_per_sequence=args.random_intervals_per_sequence,
                        fraction=args.random_deletion_fraction,
                        min_len=args.min_random_deletion_len,
                        max_len=args.max_random_deletion_len,
                        source_label="stage1_or_fasta_sequence",
                    ):
                        rows.append(row)
                        stats[policy] += 1
            elif policy == "budget_matched_random":
                if not stage1_rows:
                    skipped_policies.append(policy)
                    policy_warnings.append("budget_matched_random skipped because Stage1 proposal rows were not available")
                else:
                    for row in iter_budget_matched_random_perturbations(stage1_rows, seed=seed):
                        rows.append(row)
                        stats[policy] += 1
            elif policy == "boundary_challenge":
                if not source_records:
                    skipped_policies.append(policy)
                    policy_warnings.append("boundary_challenge skipped because no source sequences were available")
                else:
                    for row in iter_boundary_challenge_perturbations(
                        source_records,
                        fraction=args.random_deletion_fraction,
                        min_len=args.min_random_deletion_len,
                        max_len=args.max_random_deletion_len,
                    ):
                        rows.append(row)
                        stats[policy] += 1
            elif policy == "terminal_baseline":
                if not source_records:
                    skipped_policies.append(policy)
                    policy_warnings.append("terminal_baseline skipped because no source sequences were available")
                else:
                    for row in iter_terminal_baseline_perturbations(
                        source_records,
                        fraction=args.random_deletion_fraction,
                        min_len=args.min_random_deletion_len,
                        max_len=args.max_random_deletion_len,
                    ):
                        rows.append(row)
                        stats[policy] += 1
            elif policy == "multi_interval_combination":
                if not source_records:
                    skipped_policies.append(policy)
                    policy_warnings.append("multi_interval_combination skipped because no source sequences were available")
                else:
                    for row in iter_multi_interval_combination_perturbations(
                        source_records,
                        fraction=args.random_deletion_fraction,
                        min_len=args.min_random_deletion_len,
                        max_len=args.max_random_deletion_len,
                    ):
                        rows.append(row)
                        stats[policy] += 1
            elif policy == "high_risk_negative":
                skipped_policies.append(policy)
                policy_warnings.append("high_risk_negative skipped because no risk annotations input was provided")
            if len(rows) == before and policy not in skipped_policies:
                skipped_policies.append(policy)
                policy_warnings.append("{} produced no rows".format(policy))
    if not rows:
        raise ValueError(
            "no deletion perturbations generated; provide --stage1_proposal_csv "
            "or --input_fasta with --random_intervals_per_sequence > 0"
        )
    perturbation_ids = set()
    for row in rows:
        perturbation_id = row["perturbation_id"]
        if perturbation_id in perturbation_ids:
            raise ValueError("duplicate perturbation_id generated: {}".format(perturbation_id))
        perturbation_ids.add(perturbation_id)
    ensure_parent(args.output_deletion_perturbations_csv)
    fieldnames = DELETION_PERTURBATION_REQUIRED_COLUMNS + DELETION_PERTURBATION_OPTIONAL_COLUMNS
    with open(args.output_deletion_perturbations_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    stats["total_rows"] = len(rows)
    write_deletion_perturbation_summary(
        args.deletion_perturbation_summary_json,
        args,
        rows,
        stats,
        requested_policies or [],
        skipped_policies,
        policy_warnings,
    )
    write_candidate_bank_outputs(
        args,
        rows,
        stats,
        requested_policies or [],
        skipped_policies,
        policy_warnings,
    )
    return stats


def build_dataset(args):
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    records = read_fasta(args.input_fasta)
    views_per_sequence = max(1, int(config.get("views_per_sequence", 1)))

    ensure_parent(args.out_jsonl)
    ensure_parent(args.summary_txt)
    debug_path = args.out_jsonl + ".debug_failed.jsonl"
    stats = Counter()
    strength_bins = Counter()
    type_counts = Counter()
    segment_length_counts = Counter()
    original_lengths = []
    corrupted_lengths = []
    inserted_fractions = []
    corruption_strengths = []
    examples_by_type = {}
    failed_samples = []

    with open(args.out_jsonl, "w") as out_handle:
        for header, sequence in records:
            seq_id = seq_id_from_header(header)
            base_seed = int(config["seed"]) + sum(ord(ch) for ch in seq_id)
            for sample, recovery_ok in iter_corruption_views(seq_id, sequence, config, base_seed=base_seed):
                out_handle.write(json.dumps(sample, sort_keys=True) + "\n")
                stats["total_samples"] += 1
                if recovery_ok:
                    stats["recovery_pass"] += 1
                else:
                    stats["recovery_fail"] += 1
                    failed_samples.append(sample)
                strength_bin = "{:.2f}".format(float(sample["corruption_strength"]))
                strength_bins[strength_bin] += 1
                corruption_strengths.append(float(sample["corruption_strength"]))
                original_lengths.append(sample["original_length"])
                corrupted_lengths.append(sample["corrupted_length"])
                inserted_fractions.append(sample["inserted_total_length"] / float(sample["original_length"]))
                for segment in sample["insertion_segments"]:
                    corruption_type = segment["corruption_type"]
                    type_counts[corruption_type] += 1
                    segment_length_counts[segment["inserted_length"]] += 1
                    examples_by_type.setdefault(
                        corruption_type,
                        {
                            "seq_id": seq_id,
                            "corruption_strength": float(sample["corruption_strength"]),
                            "inserted_length": segment["inserted_length"],
                            "inserted_sequence": segment["inserted_sequence"],
                            "original_insert_pos": segment["original_insert_pos"],
                        },
                    )

    if failed_samples:
        with open(debug_path, "w") as handle:
            for sample in failed_samples:
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
    elif os.path.exists(debug_path):
        os.remove(debug_path)

    write_summary(
        args.summary_txt,
        args,
        config,
        records,
        stats,
        type_counts,
        segment_length_counts,
        original_lengths,
        corrupted_lengths,
        corruption_strengths,
        inserted_fractions,
        examples_by_type,
        debug_path if failed_samples else None,
        views_per_sequence,
        strength_bins,
    )
    return stats


def mean(values):
    return sum(values) / float(len(values)) if values else 0.0


def write_summary(
    path,
    args,
    config,
    records,
    stats,
    type_counts,
    segment_length_counts,
    original_lengths,
    corrupted_lengths,
    corruption_strengths,
    inserted_fractions,
    examples_by_type,
    debug_path,
    views_per_sequence,
    strength_bins,
):
    expected_types = list(config["corruption_types"])
    recovery_rate = stats["recovery_pass"] / float(stats["total_samples"]) if stats["total_samples"] else 0.0
    total_expected_samples = len(records) * max(1, int(views_per_sequence))
    all_strengths_covered = stats["total_samples"] == total_expected_samples
    all_types_covered = all(type_counts[name] > 0 for name in expected_types)
    all_pass = (
        stats["total_samples"] == total_expected_samples
        and recovery_rate == 1.0
        and all_strengths_covered
        and all_types_covered
        and stats["recovery_fail"] == 0
    )
    with open(path, "w") as handle:
        handle.write("Stage-1 deletion pretraining corruption summary\n\n")
        handle.write("input_fasta: {}\n".format(args.input_fasta))
        handle.write("config: {}\n".format(args.config))
        handle.write("out_jsonl: {}\n".format(args.out_jsonl))
        handle.write("seed: {}\n".format(config["seed"]))
        handle.write("num_input_sequences: {}\n".format(len(records)))
        handle.write("views_per_sequence: {}\n".format(views_per_sequence))
        handle.write("total_samples: {}\n".format(stats["total_samples"]))
        handle.write("recovery_check_pass_rate: {:.6f}\n".format(recovery_rate))
        handle.write("recovery_failures: {}\n".format(stats["recovery_fail"]))
        if debug_path:
            handle.write("debug_failed_jsonl: {}\n".format(debug_path))
        handle.write("\nSamples per latent corruption strength bin:\n")
        for strength_bin in sorted(strength_bins):
            handle.write("- {}: {}\n".format(strength_bin, strength_bins[strength_bin]))
        handle.write("\nCorruption type counts:\n")
        for corruption_type in expected_types:
            handle.write("- {}: {}\n".format(corruption_type, type_counts[corruption_type]))
        handle.write("\nInserted segment length distribution:\n")
        for length, count in sorted(segment_length_counts.items()):
            handle.write("- {}: {}\n".format(length, count))
        handle.write("\nAverages:\n")
        handle.write("average_original_length: {:.3f}\n".format(mean(original_lengths)))
        handle.write("average_corrupted_length: {:.3f}\n".format(mean(corrupted_lengths)))
        handle.write("average_inserted_fraction: {:.6f}\n".format(mean(inserted_fractions)))
        handle.write("average_corruption_strength: {:.6f}\n".format(mean(corruption_strengths)))
        handle.write("\nExamples of each corruption type:\n")
        for corruption_type in expected_types:
            handle.write("- {}: {}\n".format(corruption_type, json.dumps(examples_by_type.get(corruption_type, {}), sort_keys=True)))
        handle.write("\nQuality checks:\n")
        handle.write("delete_labels_match_corrupted_sequence: enforced_per_sample\n")
        handle.write("corrupted_length_matches_original_plus_inserted: enforced_per_sample\n")
        handle.write("all_strength_bins_covered: {}\n".format(all_strengths_covered))
        handle.write("all_corruption_types_covered: {}\n".format(all_types_covered))
        handle.write("\n{}\n".format("ALL_PASS" if all_pass else "WARN_CHECK_FAILED"))


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stage-1 segment corruption JSONL or deletion perturbation CSV.")
    parser.add_argument("--input_fasta", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output_jsonl", dest="out_jsonl", default=None)
    parser.add_argument("--summary_txt", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--build_formal_deletion_queries", action="store_true")
    parser.add_argument("--input_manifest_csv", default=None)
    parser.add_argument("--output_query_dir", default=None)
    parser.add_argument("--candidates_per_protein", type=int, default=32)
    parser.add_argument("--proteins_per_shard", type=int, default=1000)
    parser.add_argument("--query_profile", default=FORMAL_QUERY_PROFILE)
    parser.add_argument("--seed_tag", default=FORMAL_QUERY_PROFILE)
    parser.add_argument("--max_proteins", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--build_deletion_perturbations", action="store_true")
    parser.add_argument("--stage1_proposal_csv", default=None)
    parser.add_argument("--output_deletion_perturbations_csv", default=None)
    parser.add_argument("--deletion_perturbation_summary_json", default=None)
    parser.add_argument(
        "--candidate_construction_mode",
        choices=["method_v1", "legacy"],
        default="method_v1",
        help="method_v1 builds a two-stage counterfactual query set; legacy preserves older one-shot sampling.",
    )
    parser.add_argument(
        "--deletion_generation_policies",
        default=None,
        help="Comma-separated opt-in policy list for deletion perturbations; default uses all method_v1 policies.",
    )
    parser.add_argument("--max_stage1_rows", type=int, default=None)
    parser.add_argument("--random_intervals_per_sequence", type=int, default=1)
    parser.add_argument("--random_deletion_fraction", type=float, default=0.10)
    parser.add_argument("--min_random_deletion_len", type=int, default=1)
    parser.add_argument("--max_random_deletion_len", type=int, default=40)
    parser.add_argument("--max_deletion_source_records", type=int, default=None)
    parser.add_argument(
        "--method_budget_fractions",
        default=",".join("{:.2f}".format(value) for value in METHOD_V1_DEFAULT_BUDGET_FRACTIONS),
        help="Comma-separated deletion-budget fractions for method_v1 candidate-pool construction.",
    )
    parser.add_argument("--method_max_teacher_candidates_per_protein", type=int, default=64)
    parser.add_argument("--method_max_teacher_candidates_per_budget", type=int, default=16)
    parser.add_argument("--method_random_per_budget", type=int, default=24)
    parser.add_argument("--method_boundary_per_budget", type=int, default=8)
    parser.add_argument("--method_multi_per_budget", type=int, default=3)
    parser.add_argument("--method_budget_matched_random_per_stage1", type=int, default=2)
    parser.add_argument("--method_high_risk_per_budget", type=int, default=4)
    parser.add_argument("--method_max_single_interval_len", type=int, default=300)
    parser.add_argument("--method_max_multi_total_len", type=int, default=300)
    parser.add_argument("--method_nms_iou_threshold", type=float, default=0.50)
    parser.add_argument("--method_terminal_max_fraction", type=float, default=0.25)
    parser.add_argument(
        "--method_policy_quotas",
        default=None,
        help="Optional comma-separated per-budget quotas such as random_interval=4,boundary_challenge=3.",
    )
    parser.add_argument(
        "--candidate_bank_profile",
        default=None,
        help="Opt-in candidate bank profile, for example v2_generalization_smoke.",
    )
    parser.add_argument("--candidate_bank_summary_json", default=None)
    parser.add_argument("--candidate_bank_audit_json", default=None)
    parser.add_argument("--candidate_bank_max_candidates_per_protein", type=int, default=24)
    parser.add_argument("--candidate_bank_max_candidates_per_budget_bin", type=int, default=8)
    parser.add_argument("--candidate_bank_max_candidates_per_policy_per_protein", type=int, default=6)
    parser.add_argument("--candidate_bank_overlap_jaccard_max", type=float, default=0.80)
    parser.add_argument("--candidate_bank_min_pairable_per_budget_bin", type=int, default=2)
    parser.add_argument("--candidate_bank_max_deletion_fraction", type=float, default=0.35)
    parser.add_argument("--candidate_bank_seed", type=int, default=None)
    parser.add_argument(
        "--risk_interval_csv",
        default=None,
        help="Optional risk intervals for high_risk_negative candidates; columns: protein_id,start,end.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.build_formal_deletion_queries:
        summary = build_formal_deletion_queries(args)
        print("Formal deletion query proteins: {}".format(summary["num_proteins_processed"]))
        print("Formal deletion query rows: {}".format(summary["total_candidates_written"]))
        print("Formal deletion query shards: {}".format(summary["num_shards_written"]))
        print("Underfilled proteins: {}".format(summary["underfilled_proteins"]))
        print("Resume skipped existing shards: {}".format(summary["resume_skipped_existing_shards"]))
        print("Wrote {}".format(summary["output_query_dir"]))
        print("Wrote {}".format(summary["shard_index_csv"]))
        print("Wrote {}".format(summary["summary_json"]))
        print("Wrote {}".format(summary["policy_config_json"]))
        return
    if args.build_deletion_perturbations:
        stats = build_deletion_perturbations(args)
        print("Deletion perturbation rows: {}".format(stats["total_rows"]))
        if args.candidate_construction_mode == "method_v1":
            print("Candidate pool rows: {}".format(stats["candidate_pool_rows"]))
            print("Selected teacher-query rows: {}".format(stats["selected_rows"]))
        print("Stage1 proposal rows: {}".format(stats["stage1_proposal"]))
        print("Random interval rows: {}".format(stats["random_interval"]))
        for policy in DELETION_GENERATION_POLICIES:
            if policy not in ("stage1_proposal", "random_interval") and stats[policy] > 0:
                print("{} rows: {}".format(policy, stats[policy]))
        print("Wrote {}".format(args.output_deletion_perturbations_csv))
        if args.deletion_perturbation_summary_json:
            print("Wrote {}".format(args.deletion_perturbation_summary_json))
        if args.candidate_bank_summary_json:
            print("Wrote {}".format(args.candidate_bank_summary_json))
        if args.candidate_bank_audit_json:
            print("Wrote {}".format(args.candidate_bank_audit_json))
        return
    missing = []
    for field, value in [
        ("--input_fasta", args.input_fasta),
        ("--config", args.config),
        ("--output_jsonl", args.out_jsonl),
        ("--summary_txt", args.summary_txt),
    ]:
        if not value:
            missing.append(field)
    if missing:
        raise SystemExit("missing required arguments for JSONL mode: {}".format(", ".join(missing)))
    stats = build_dataset(args)
    print("Total samples: {}".format(stats["total_samples"]))
    print("Recovery pass: {}".format(stats["recovery_pass"]))
    print("Recovery fail: {}".format(stats["recovery_fail"]))
    print("Wrote {}".format(args.out_jsonl))
    print("Wrote {}".format(args.summary_txt))


if __name__ == "__main__":
    main()
