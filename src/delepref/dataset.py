"""Streaming loaders for DelePref condition-grouped preference batches."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .preference_core import (
    DeletionCandidate,
    PreferencePair,
    deleted_length,
    parse_deleted_intervals,
)
from .student_model import validate_strict_preference_pair
from .streaming import ConditionRef, stable_seed


REPO = Path(__file__).resolve().parents[2]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


@dataclass(frozen=True)
class PairShardEntry:
    source: str
    shard_index: int
    pair_path: Path
    candidate_path: Path
    expected_pairs: int


@dataclass(frozen=True)
class LoadedConditionBatch:
    references: Tuple[ConditionRef, ...]
    pairs: Tuple[PreferencePair, ...]
    pairs_per_condition: Tuple[int, ...]
    source_condition_counts: Mapping[str, int]
    sequence_tokens: int


def read_pair_shard_index(path: Path, source: str) -> Mapping[int, PairShardEntry]:
    entries = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for ordinal, row in enumerate(csv.DictReader(handle)):
            index = int(row.get("shard_index") or ordinal)
            if index in entries:
                raise ValueError("duplicate shard index")
            entries[index] = PairShardEntry(
                source=source,
                shard_index=index,
                pair_path=resolve_path(row["output_csv"]),
                candidate_path=resolve_path(row["source_candidate_csv"]),
                expected_pairs=int(row["num_pairs"]),
            )
    if set(entries) != set(range(300)):
        raise ValueError("{} pair index must contain shards 0..299".format(source))
    return entries


PAIR_FIELDS = (
    "protein_id", "pair_id", "selection_category", "construction_policy",
    "deleted_length", "preference", "preference_margin",
    "preference_confidence", "view_conflicting", "pair_quality_bucket",
    "scorer_training_weight", "policy_positive_eligible",
    "perturbation_id_a", "plan_id_a", "perturbation_id_b", "plan_id_b",
    "deleted_intervals_a", "deleted_intervals_b",
    "deletion_fraction_a", "deletion_fraction_b",
    "generation_policy_a", "generation_policy_b", "comparison_basis_json",
)


class ConditionBatchLoader:
    """Load full condition groups while caching only a small number of shards."""

    def __init__(
        self,
        *,
        base_pair_index: Path,
        b20_pair_index: Path,
        pairs_per_condition: int,
        seed: int,
        cache_shards: int = 2,
    ):
        self.indexes = {
            "base": read_pair_shard_index(base_pair_index, "base"),
            "b20": read_pair_shard_index(b20_pair_index, "b20"),
        }
        self.pairs_per_condition = int(pairs_per_condition)
        self.seed = int(seed)
        self.cache_shards = int(cache_shards)
        if self.pairs_per_condition < 1:
            raise ValueError("pairs_per_condition must be >=1")
        if self.cache_shards < 1:
            raise ValueError("cache_shards must be >=1")
        self._cache: OrderedDict[Tuple[str, int], Mapping[str, object]] = OrderedDict()

    @staticmethod
    def _read_sequences(path: Path) -> Mapping[str, Tuple[str, str]]:
        sequences: Dict[str, Tuple[str, str]] = {}
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                protein = row["protein_id"]
                sequence = row["wildtype_sequence"]
                existing = sequences.get(protein)
                if existing is None:
                    digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
                    stored = row.get("wildtype_sha256") or digest
                    if stored != digest:
                        raise ValueError("candidate WT sequence hash mismatch")
                    sequences[protein] = (sequence, digest)
                elif existing[0] != sequence:
                    raise ValueError("candidate shard has inconsistent WT sequence")
        return sequences

    @staticmethod
    def _read_pair_groups(path: Path, expected_pairs: int) -> Mapping[Tuple[str, int], List[Mapping[str, str]]]:
        groups = defaultdict(list)
        rows = 0
        path = Path(path)
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, mode='rt', newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            missing = set(PAIR_FIELDS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError("pair shard is missing fields: {}".format(sorted(missing)))
            for row in reader:
                rows += 1
                minimal = {field: row[field] for field in PAIR_FIELDS}
                groups[(row["protein_id"], int(row["deleted_length"]))].append(minimal)
        if rows != int(expected_pairs):
            raise ValueError("pair shard row count differs from index")
        return groups

    def _load_shard(self, source: str, shard_index: int) -> Mapping[str, object]:
        key = (source, int(shard_index))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        entry = self.indexes[source][int(shard_index)]
        payload = {
            "entry": entry,
            "sequences": self._read_sequences(entry.candidate_path),
            "groups": self._read_pair_groups(entry.pair_path, entry.expected_pairs),
        }
        self._cache[key] = payload
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_shards:
            self._cache.popitem(last=False)
        return payload

    @staticmethod
    def _candidate(row: Mapping[str, str], side: str, sequence: str, reference: ConditionRef) -> DeletionCandidate:
        intervals = parse_deleted_intervals(row["deleted_intervals_" + side], len(sequence))
        actual = deleted_length(intervals)
        if actual != reference.deleted_length:
            raise ValueError("pair candidate count differs from condition catalog")
        fraction = float(row["deletion_fraction_" + side])
        if abs(fraction - reference.beta) > 1.0e-8:
            raise ValueError("pair candidate beta differs from condition catalog")
        return DeletionCandidate(
            protein_id=reference.protein_id,
            perturbation_id=row["perturbation_id_" + side],
            plan_id=row["plan_id_" + side],
            wildtype_sequence=sequence,
            deleted_intervals=intervals,
            deleted_length=actual,
            deletion_fraction=fraction,
            target_delete_fraction=reference.beta,
            budget_bin="deleted_length:{}".format(reference.deleted_length),
            generation_policy=row["generation_policy_" + side],
        )

    def _pair(self, row: Mapping[str, str], sequence: str, reference: ConditionRef) -> PreferencePair:
        candidate_a = self._candidate(row, "a", sequence, reference)
        candidate_b = self._candidate(row, "b", sequence, reference)
        result = PreferencePair(
            pair_id=row["pair_id"],
            protein_id=reference.protein_id,
            budget_bin="deleted_length:{}".format(reference.deleted_length),
            preference=row["preference"],
            preference_confidence=float(row["preference_confidence"]),
            loss_weight=float(row["scorer_training_weight"]),
            preference_margin=float(row["preference_margin"]),
            is_valid_pair=True,
            construction_policy=row["construction_policy"],
            quality_status=row["pair_quality_bucket"],
            quality_reason="pair_teacher_derived_preference",
            a=candidate_a,
            b=candidate_b,
            pair_quality_bucket=row["pair_quality_bucket"],
            policy_positive_eligible=row["policy_positive_eligible"].lower() == "true",
            scorer_training_weight=float(row["scorer_training_weight"]),
            metadata={
                "source": reference.source,
                "shard_index": reference.shard_index,
                "condition_id": reference.condition_id,
                "selection_category": row["selection_category"],
                "view_conflicting": row["view_conflicting"].lower() == "true",
                "comparison_basis_json": row["comparison_basis_json"],
            },
        )
        validate_strict_preference_pair(result)
        return result

    def load(self, references: Sequence[ConditionRef], *, epoch: int) -> LoadedConditionBatch:
        if not references:
            raise ValueError("cannot load an empty condition batch")
        refs_by_shard = defaultdict(list)
        for reference in references:
            reference.validate()
            refs_by_shard[(reference.source, reference.shard_index)].append(reference)
        loaded_by_id = {}
        for (source, shard_index), shard_refs in refs_by_shard.items():
            payload = self._load_shard(source, shard_index)
            sequences = payload["sequences"]
            groups = payload["groups"]
            for reference in shard_refs:
                sequence_record = sequences.get(reference.protein_id)
                if sequence_record is None or sequence_record[1] != reference.sequence_sha256:
                    raise ValueError("condition sequence identity differs from source candidate shard")
                rows = groups.get((reference.protein_id, reference.deleted_length))
                if rows is None or len(rows) != reference.pair_count:
                    raise ValueError("condition pair count differs from catalog")
                selected = sorted(rows, key=lambda row: row["pair_id"])
                random.Random(stable_seed(
                    self.seed, epoch, source, shard_index, reference.condition_id, "pairs"
                )).shuffle(selected)
                selected = selected[: self.pairs_per_condition]
                pairs = tuple(self._pair(row, sequence_record[0], reference) for row in selected)
                if not pairs:
                    raise RuntimeError("condition selection produced no pairs")
                loaded_by_id[reference.condition_id] = pairs
        ordered_pairs = []
        counts = []
        for reference in references:
            pairs = loaded_by_id[reference.condition_id]
            ordered_pairs.extend(pairs)
            counts.append(len(pairs))
        return LoadedConditionBatch(
            references=tuple(references),
            pairs=tuple(ordered_pairs),
            pairs_per_condition=tuple(counts),
            source_condition_counts=dict(Counter(reference.source for reference in references)),
            sequence_tokens=sum(reference.wildtype_length for reference in references),
        )
