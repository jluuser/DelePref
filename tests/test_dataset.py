import csv
import gzip
import hashlib
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

from delepref.dataset import (
    PAIR_FIELDS,
    PairShardEntry,
    ConditionBatchLoader,
)
from delepref.streaming import ConditionRef


class DelePrefDatasetTests(unittest.TestCase):
    def pair_row(self, pair_id, a_intervals, b_intervals):
        return {
            "protein_id": "protein_1",
            "pair_id": pair_id,
            "selection_category": "local_hard",
            "construction_policy": "strict_same_deleted_length_pair",
            "deleted_length": "4",
            "preference": "a_better",
            "preference_margin": "0.1",
            "preference_confidence": "0.8",
            "view_conflicting": "false",
            "pair_quality_bucket": "high",
            "scorer_training_weight": "0.7",
            "policy_positive_eligible": "true",
            "perturbation_id_a": pair_id + "_a",
            "plan_id_a": pair_id + "_a",
            "perturbation_id_b": pair_id + "_b",
            "plan_id_b": pair_id + "_b",
            "deleted_intervals_a": a_intervals,
            "deleted_intervals_b": b_intervals,
            "deletion_fraction_a": "0.2",
            "deletion_fraction_b": "0.2",
            "generation_policy_a": "toy_a",
            "generation_policy_b": "toy_b",
            "comparison_basis_json": "{}",
        }

    def test_condition_batch_loader_streams_one_group_deterministically(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.csv"
            pairs = root / "pairs.csv"
            with candidates.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["protein_id", "perturbation_id", "plan_id", "wildtype_sequence", "wildtype_sha256"],
                )
                writer.writeheader()
                for index in range(4):
                    writer.writerow({
                        "protein_id": "protein_1",
                        "perturbation_id": "candidate_{}".format(index),
                        "plan_id": "candidate_{}".format(index),
                        "wildtype_sequence": sequence,
                        "wildtype_sha256": digest,
                    })
            pair_rows = [
                self.pair_row("pair_1", "[[0,3]]", "[[8,9],[14,15]]"),
                self.pair_row("pair_2", "[[4,7]]", "[[10,11],[16,17]]"),
            ]
            with pairs.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
                writer.writeheader()
                writer.writerows(pair_rows)
            entry = PairShardEntry(
                source="base",
                shard_index=0,
                pair_path=pairs,
                candidate_path=candidates,
                expected_pairs=2,
            )
            loader = object.__new__(ConditionBatchLoader)
            loader.indexes = {"base": {0: entry}, "b20": {}}
            loader.pairs_per_condition = 1
            loader.seed = 13
            loader.cache_shards = 2
            loader._cache = OrderedDict()
            reference = ConditionRef(
                source="base",
                shard_index=0,
                protein_id="protein_1",
                sequence_sha256=digest,
                wildtype_length=20,
                deleted_length=4,
                beta=0.2,
                split="train",
                pair_count=2,
                condition_id="condition_1",
            )
            first = loader.load([reference], epoch=0)
            second = loader.load([reference], epoch=0)
        self.assertEqual(len(first.pairs), 1)
        self.assertEqual(first.pairs[0].pair_id, second.pairs[0].pair_id)
        self.assertEqual(first.pairs_per_condition, (1,))
        self.assertEqual(first.source_condition_counts, {"base": 1})
        self.assertEqual(first.sequence_tokens, 20)
        self.assertEqual(first.pairs[0].a.target_delete_fraction, 0.2)
        self.assertEqual(first.pairs[0].metadata["condition_id"], "condition_1")

    def test_pair_loader_reads_released_gzip_shard_without_changing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pairs.csv.gz'
            row = self.pair_row('pair_gz', '[[0,3]]', '[[8,11]]')
            with gzip.open(path, 'wt', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=PAIR_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            groups = ConditionBatchLoader._read_pair_groups(path, 1)
            self.assertEqual(groups[('protein_1', 4)][0]['pair_id'], 'pair_gz')
            self.assertEqual(groups[('protein_1', 4)][0]['deleted_intervals_a'], '[[0,3]]')


if __name__ == "__main__":
    unittest.main()
