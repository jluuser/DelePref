"""Deterministic condition scheduling and dynamic DDP batch planning for DelePref."""

from __future__ import annotations

import csv
import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BASE_SOURCE = "base"
B20_SOURCE = "b20"
MIXTURE_PATTERN = (BASE_SOURCE, BASE_SOURCE, BASE_SOURCE, B20_SOURCE)


def stable_seed(seed: int, epoch: int, *parts: object) -> int:
    payload = "\x1f".join([str(int(seed)), str(int(epoch))] + [str(part) for part in parts])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


@dataclass(frozen=True, order=True)
class ConditionRef:
    source: str
    shard_index: int
    protein_id: str
    sequence_sha256: str
    wildtype_length: int
    deleted_length: int
    beta: float
    split: str
    pair_count: int
    condition_id: str

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "ConditionRef":
        reference = cls(
            source=str(row["source"]),
            shard_index=int(row["shard_index"]),
            protein_id=str(row["protein_id"]),
            sequence_sha256=str(row["sequence_sha256"]),
            wildtype_length=int(row["wildtype_length"]),
            deleted_length=int(row["deleted_length"]),
            beta=float(row["beta"]),
            split=str(row["split"]),
            pair_count=int(row["pair_count"]),
            condition_id=str(row["condition_id"]),
        )
        reference.validate()
        return reference

    def validate(self) -> None:
        if self.source not in (BASE_SOURCE, B20_SOURCE):
            raise ValueError("unknown condition source: {}".format(self.source))
        if self.split not in ("train", "dev"):
            raise ValueError("condition catalog may only contain train/dev rows")
        if self.shard_index < 0 or not self.protein_id or len(self.sequence_sha256) != 64:
            raise ValueError("invalid condition identity")
        if self.wildtype_length < 2 or not 0 < self.deleted_length < self.wildtype_length:
            raise ValueError("invalid condition length/count")
        if self.pair_count < 1:
            raise ValueError("condition has no pairs")
        expected_beta = self.deleted_length / float(self.wildtype_length)
        if abs(self.beta - expected_beta) > 1.0e-12:
            raise ValueError("condition beta is not exact deleted_length/L")


@dataclass(frozen=True)
class PlannedConditionBatch:
    rank: int
    world_size: int
    epoch: int
    global_start_cursor: int
    global_end_cursor: int
    local_conditions: Tuple[ConditionRef, ...]
    global_source_counts: Mapping[str, int]
    local_source_counts: Mapping[str, int]
    local_tokens: int
    local_max_length: int
    padding_fraction: float


def read_condition_catalog(path: Path, *, split: str) -> List[ConditionRef]:
    if split not in ("train", "dev"):
        raise ValueError("catalog split must be train or dev")
    references = []
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split:
                continue
            reference = ConditionRef.from_row(row)
            if reference.condition_id in seen:
                raise ValueError("duplicate condition_id: {}".format(reference.condition_id))
            seen.add(reference.condition_id)
            references.append(reference)
    if not references:
        raise ValueError("condition catalog has no {} conditions".format(split))
    return references


def deterministic_shard_group_order(
    references: Sequence[ConditionRef],
    *,
    seed: int,
    epoch: int,
    source: str,
    length_bucket_width: int,
) -> Tuple[ConditionRef, ...]:
    """Shuffle shards, then length buckets, then groups within each bucket."""

    width = int(length_bucket_width)
    if width < 1:
        raise ValueError("length_bucket_width must be >= 1")
    by_shard: Dict[int, List[ConditionRef]] = defaultdict(list)
    for reference in references:
        reference.validate()
        if reference.source != source:
            raise ValueError("source order received a mismatched condition")
        by_shard[reference.shard_index].append(reference)
    shard_order = sorted(by_shard)
    random.Random(stable_seed(seed, epoch, source, "shards")).shuffle(shard_order)
    ordered: List[ConditionRef] = []
    for shard_index in shard_order:
        by_bucket: Dict[int, List[ConditionRef]] = defaultdict(list)
        for reference in by_shard[shard_index]:
            by_bucket[(reference.wildtype_length - 1) // width].append(reference)
        bucket_order = sorted(by_bucket)
        random.Random(stable_seed(seed, epoch, source, shard_index, "buckets")).shuffle(bucket_order)
        for bucket in bucket_order:
            values = sorted(by_bucket[bucket], key=lambda item: item.condition_id)
            random.Random(stable_seed(seed, epoch, source, shard_index, bucket, "groups")).shuffle(values)
            ordered.extend(values)
    if len(ordered) != len(references) or len({item.condition_id for item in ordered}) != len(ordered):
        raise RuntimeError("deterministic condition order repeated or omitted conditions")
    return tuple(ordered)


class DeterministicShardGroupMixtureSchedule:
    """Indexable 3:1 schedule with full base coverage across a cycle.

    Every epoch consumes all b20 conditions and an equally sized 3x base
    window.  Base windows advance over one fixed permutation until the entire
    base catalog has been visited, then a new deterministic permutation is
    created for the next coverage cycle.  This avoids silently training on a
    different truncated base subset in every epoch.
    """

    def __init__(
        self,
        base_conditions: Sequence[ConditionRef],
        b20_conditions: Sequence[ConditionRef],
        *,
        seed: int,
        epoch: int,
        length_bucket_width: int,
        catalog_sha256: str,
    ):
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.length_bucket_width = int(length_bucket_width)
        self.catalog_sha256 = str(catalog_sha256)
        ordered_base = deterministic_shard_group_order(
            base_conditions,
            seed=self.seed,
            epoch=0,
            source=BASE_SOURCE,
            length_bucket_width=self.length_bucket_width,
        )
        self.b20 = deterministic_shard_group_order(
            b20_conditions,
            seed=self.seed,
            epoch=self.epoch,
            source=B20_SOURCE,
            length_bucket_width=self.length_bucket_width,
        )
        self.base_catalog_size = len(ordered_base)
        self.base_conditions_per_epoch = 3 * len(self.b20)
        if self.base_catalog_size < self.base_conditions_per_epoch:
            raise ValueError("base catalog has fewer than three conditions per b20 condition")
        self.base_coverage_cycle_epochs = int(
            math.ceil(self.base_catalog_size / float(self.base_conditions_per_epoch))
        )
        self.base_coverage_cycle_index, self.base_window_position = divmod(
            self.epoch, self.base_coverage_cycle_epochs
        )
        if self.base_coverage_cycle_index:
            ordered_base = deterministic_shard_group_order(
                base_conditions,
                seed=self.seed,
                epoch=self.base_coverage_cycle_index,
                source=BASE_SOURCE,
                length_bucket_width=self.length_bucket_width,
            )
        self.base_window_start = (
            self.base_window_position * self.base_conditions_per_epoch
        ) % self.base_catalog_size
        self.base = self._circular_window(
            ordered_base,
            self.base_window_start,
            self.base_conditions_per_epoch,
        )
        self.num_global_conditions = 4 * len(self.b20)

    @staticmethod
    def _circular_window(
        values: Sequence[ConditionRef], start: int, count: int
    ) -> Tuple[ConditionRef, ...]:
        if count > len(values):
            raise ValueError("coverage window cannot repeat a condition within one epoch")
        end = start + count
        if end <= len(values):
            return tuple(values[start:end])
        return tuple(values[start:]) + tuple(values[: end - len(values)])

    def condition_at(self, position: int) -> ConditionRef:
        index = int(position)
        if not 0 <= index < self.num_global_conditions:
            raise IndexError(index)
        block, offset = divmod(index, 4)
        return self.base[3 * block + offset] if offset < 3 else self.b20[block]

    def audit(self) -> Mapping[str, object]:
        base_ids = {item.condition_id for item in self.base}
        b20_ids = {item.condition_id for item in self.b20}
        if len(base_ids) != len(self.base) or len(b20_ids) != len(self.b20):
            raise RuntimeError("condition schedule repeats identities within a source")
        return {
            "schema_version": "delepref_shard_group_mixture_schedule_audit_v1",
            "status": "PASS",
            "epoch": self.epoch,
            "seed": self.seed,
            "catalog_sha256": self.catalog_sha256,
            "sampling_unit": "condition",
            "shuffle_order": ["shard", "length_bucket", "condition_group"],
            "base_conditions": len(self.base),
            "base_catalog_conditions": self.base_catalog_size,
            "b20_conditions": len(self.b20),
            "base_to_b20_ratio": len(self.base) / float(len(self.b20)),
            "no_replacement_within_source": True,
            "base_coverage_cycle_epochs": self.base_coverage_cycle_epochs,
            "base_coverage_cycle_index": self.base_coverage_cycle_index,
            "base_window_position": self.base_window_position,
            "base_window_start": self.base_window_start,
            "base_window_wraps": (
                self.base_window_start + len(self.base) > self.base_catalog_size
            ),
        }


class DynamicGlobalBatchPlanner:
    """Plan equal-rank DDP waves under condition and token limits."""

    def __init__(
        self,
        schedule: DeterministicShardGroupMixtureSchedule,
        *,
        rank: int,
        world_size: int,
        conditions_per_rank: int,
        max_tokens_per_gpu: int,
        cursor: int = 0,
    ):
        self.schedule = schedule
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.conditions_per_rank = int(conditions_per_rank)
        self.max_tokens_per_gpu = int(max_tokens_per_gpu)
        self.cursor = int(cursor)
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("invalid DDP rank/world_size")
        if self.conditions_per_rank < 1:
            raise ValueError("conditions_per_rank must be >= 1")
        if self.max_tokens_per_gpu < 0:
            raise ValueError("max_tokens_per_gpu cannot be negative")
        if self.schedule.num_global_conditions % self.world_size:
            raise ValueError("global condition count must be divisible by world_size")
        if self.cursor % self.world_size or not 0 <= self.cursor <= self.schedule.num_global_conditions:
            raise ValueError("global cursor must be a valid DDP wave boundary")

    def next_batch(self) -> Optional[PlannedConditionBatch]:
        if self.cursor >= self.schedule.num_global_conditions:
            return None
        start = self.cursor
        refs_by_rank: List[List[ConditionRef]] = [[] for _ in range(self.world_size)]
        tokens_by_rank = [0 for _ in range(self.world_size)]
        global_counts = Counter()
        while self.cursor < self.schedule.num_global_conditions:
            wave = [self.schedule.condition_at(self.cursor + offset) for offset in range(self.world_size)]
            proposed_tokens = [
                tokens_by_rank[rank] + wave[rank].wildtype_length for rank in range(self.world_size)
            ]
            token_exceeded = self.max_tokens_per_gpu > 0 and any(
                value > self.max_tokens_per_gpu for value in proposed_tokens
            )
            if token_exceeded and refs_by_rank[0]:
                break
            if token_exceeded:
                raise RuntimeError(
                    "one DDP condition exceeds max_tokens_per_gpu; increase the explicit limit"
                )
            for rank, reference in enumerate(wave):
                refs_by_rank[rank].append(reference)
                tokens_by_rank[rank] = proposed_tokens[rank]
                global_counts[reference.source] += 1
            self.cursor += self.world_size
            if len(refs_by_rank[0]) >= self.conditions_per_rank:
                break
        local = tuple(refs_by_rank[self.rank])
        if not local or any(len(items) != len(local) for items in refs_by_rank):
            raise RuntimeError("dynamic DDP batch planner produced an uneven rank batch")
        local_counts = Counter(item.source for item in local)
        max_length = max(item.wildtype_length for item in local)
        padded = max_length * len(local)
        padding_fraction = (padded - tokens_by_rank[self.rank]) / float(padded)
        return PlannedConditionBatch(
            rank=self.rank,
            world_size=self.world_size,
            epoch=self.schedule.epoch,
            global_start_cursor=start,
            global_end_cursor=self.cursor,
            local_conditions=local,
            global_source_counts=dict(global_counts),
            local_source_counts=dict(local_counts),
            local_tokens=tokens_by_rank[self.rank],
            local_max_length=max_length,
            padding_fraction=padding_fraction,
        )

    def state_dict(self) -> Mapping[str, object]:
        return {
            "schema_version": "delepref_dynamic_global_batch_planner_state_v1",
            "seed": self.schedule.seed,
            "epoch": self.schedule.epoch,
            "catalog_sha256": self.schedule.catalog_sha256,
            "length_bucket_width": self.schedule.length_bucket_width,
            "rank": self.rank,
            "world_size": self.world_size,
            "conditions_per_rank": self.conditions_per_rank,
            "max_tokens_per_gpu": self.max_tokens_per_gpu,
            "global_cursor": self.cursor,
            "num_global_conditions": self.schedule.num_global_conditions,
            "mixture_pattern": list(MIXTURE_PATTERN),
            "base_catalog_size": self.schedule.base_catalog_size,
            "base_coverage_cycle_epochs": self.schedule.base_coverage_cycle_epochs,
            "base_coverage_cycle_index": self.schedule.base_coverage_cycle_index,
            "base_window_position": self.schedule.base_window_position,
            "base_window_start": self.schedule.base_window_start,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = self.state_dict()
        for key in (
            "seed", "epoch", "catalog_sha256", "length_bucket_width", "rank",
            "world_size", "conditions_per_rank", "max_tokens_per_gpu",
            "num_global_conditions", "mixture_pattern", "base_catalog_size",
            "base_coverage_cycle_epochs", "base_coverage_cycle_index",
            "base_window_position", "base_window_start",
        ):
            if state.get(key) != expected.get(key):
                raise ValueError("batch planner state mismatch for {}".format(key))
        cursor = int(state.get("global_cursor", -1))
        if cursor % self.world_size or not 0 <= cursor <= self.schedule.num_global_conditions:
            raise ValueError("checkpoint global cursor is invalid")
        self.cursor = cursor


def count_optimizer_batches(
    schedule: DeterministicShardGroupMixtureSchedule,
    *,
    world_size: int,
    conditions_per_rank: int,
    max_tokens_per_gpu: int,
) -> int:
    planner = DynamicGlobalBatchPlanner(
        schedule,
        rank=0,
        world_size=world_size,
        conditions_per_rank=conditions_per_rank,
        max_tokens_per_gpu=max_tokens_per_gpu,
    )
    count = 0
    while planner.next_batch() is not None:
        count += 1
    return count
