import tempfile
import unittest
from pathlib import Path

from delepref.streaming import (
    B20_SOURCE,
    BASE_SOURCE,
    ConditionRef,
    DeterministicShardGroupMixtureSchedule,
    DynamicGlobalBatchPlanner,
    count_optimizer_batches,
)


def reference(source, index, *, split="train", length=None):
    sequence_length = int(length if length is not None else 80 + index % 5 * 20)
    deleted = max(1, round(0.2 * sequence_length))
    return ConditionRef(
        source=source,
        shard_index=index // 10,
        protein_id="{}_protein_{:04d}".format(source, index),
        sequence_sha256=("{:064x}".format(index + (0 if source == BASE_SOURCE else 100000))),
        wildtype_length=sequence_length,
        deleted_length=deleted,
        beta=deleted / float(sequence_length),
        split=split,
        pair_count=8,
        condition_id="{}_condition_{:04d}".format(source, index),
    )


class DelePrefStreamingTests(unittest.TestCase):
    def conditions(self):
        return (
            [reference(BASE_SOURCE, index) for index in range(60)],
            [reference(B20_SOURCE, index) for index in range(20)],
        )

    def schedule(self, epoch=0):
        base, b20 = self.conditions()
        return DeterministicShardGroupMixtureSchedule(
            base,
            b20,
            seed=13,
            epoch=epoch,
            length_bucket_width=64,
            catalog_sha256="catalog_hash",
        )

    def test_schedule_is_exact_condition_level_three_to_one(self):
        schedule = self.schedule()
        audit = schedule.audit()
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["base_conditions"], 60)
        self.assertEqual(audit["b20_conditions"], 20)
        self.assertEqual(audit["base_to_b20_ratio"], 3.0)
        self.assertEqual(audit["shuffle_order"], ["shard", "length_bucket", "condition_group"])

    def test_ddp_dynamic_batches_are_disjoint_complete_and_synchronized(self):
        schedule = self.schedule()
        planners = [
            DynamicGlobalBatchPlanner(
                schedule,
                rank=rank,
                world_size=2,
                conditions_per_rank=4,
                max_tokens_per_gpu=420,
            )
            for rank in range(2)
        ]
        rank_ids = [[], []]
        cursors = []
        while True:
            batches = [planner.next_batch() for planner in planners]
            self.assertEqual(batches[0] is None, batches[1] is None)
            if batches[0] is None:
                break
            self.assertEqual(batches[0].global_start_cursor, batches[1].global_start_cursor)
            self.assertEqual(batches[0].global_end_cursor, batches[1].global_end_cursor)
            self.assertLessEqual(batches[0].local_tokens, 420)
            self.assertLessEqual(batches[1].local_tokens, 420)
            self.assertEqual(len(batches[0].local_conditions), len(batches[1].local_conditions))
            for rank, batch in enumerate(batches):
                rank_ids[rank].extend(item.condition_id for item in batch.local_conditions)
            cursors.append(batches[0].global_end_cursor)
        expected = {schedule.condition_at(index).condition_id for index in range(schedule.num_global_conditions)}
        self.assertFalse(set(rank_ids[0]) & set(rank_ids[1]))
        self.assertEqual(set(rank_ids[0]) | set(rank_ids[1]), expected)
        self.assertEqual(cursors[-1], schedule.num_global_conditions)

    def test_resume_cursor_has_no_repeat_or_omission(self):
        schedule = self.schedule()
        original = DynamicGlobalBatchPlanner(
            schedule,
            rank=1,
            world_size=2,
            conditions_per_rank=3,
            max_tokens_per_gpu=400,
        )
        prefix = [original.next_batch(), original.next_batch()]
        state = original.state_dict()
        resumed = DynamicGlobalBatchPlanner(
            schedule,
            rank=1,
            world_size=2,
            conditions_per_rank=3,
            max_tokens_per_gpu=400,
        )
        resumed.load_state_dict(state)
        suffix = []
        while True:
            batch = resumed.next_batch()
            if batch is None:
                break
            suffix.append(batch)
        baseline = DynamicGlobalBatchPlanner(
            schedule,
            rank=1,
            world_size=2,
            conditions_per_rank=3,
            max_tokens_per_gpu=400,
        )
        expected = []
        while True:
            batch = baseline.next_batch()
            if batch is None:
                break
            expected.append(batch)
        observed_ids = [item.condition_id for batch in prefix + suffix for item in batch.local_conditions]
        expected_ids = [item.condition_id for batch in expected for item in batch.local_conditions]
        self.assertEqual(observed_ids, expected_ids)

    def test_token_limit_fails_loud_instead_of_silent_batch_shrink(self):
        schedule = DeterministicShardGroupMixtureSchedule(
            [reference(BASE_SOURCE, index, length=200) for index in range(6)],
            [reference(B20_SOURCE, index, length=200) for index in range(2)],
            seed=13,
            epoch=0,
            length_bucket_width=64,
            catalog_sha256="catalog_hash",
        )
        planner = DynamicGlobalBatchPlanner(
            schedule,
            rank=0,
            world_size=2,
            conditions_per_rank=4,
            max_tokens_per_gpu=199,
        )
        with self.assertRaisesRegex(RuntimeError, "exceeds max_tokens_per_gpu"):
            planner.next_batch()

    def test_epoch_shuffle_is_deterministic_but_changes_by_epoch(self):
        epoch0_a = self.schedule(epoch=0)
        epoch0_b = self.schedule(epoch=0)
        epoch1 = self.schedule(epoch=1)
        ids0a = [epoch0_a.condition_at(index).condition_id for index in range(epoch0_a.num_global_conditions)]
        ids0b = [epoch0_b.condition_at(index).condition_id for index in range(epoch0_b.num_global_conditions)]
        ids1 = [epoch1.condition_at(index).condition_id for index in range(epoch1.num_global_conditions)]
        self.assertEqual(ids0a, ids0b)
        self.assertNotEqual(ids0a, ids1)

    def test_base_windows_cover_uneven_catalog_before_repeating(self):
        base = [reference(BASE_SOURCE, index) for index in range(179)]
        b20 = [reference(B20_SOURCE, index) for index in range(20)]
        schedules = [
            DeterministicShardGroupMixtureSchedule(
                base,
                b20,
                seed=13,
                epoch=epoch,
                length_bucket_width=64,
                catalog_sha256="catalog_hash",
            )
            for epoch in range(3)
        ]
        epoch_ids = [{item.condition_id for item in schedule.base} for schedule in schedules]
        self.assertTrue(all(len(values) == 60 for values in epoch_ids))
        self.assertEqual(set().union(*epoch_ids), {item.condition_id for item in base})
        self.assertEqual(sum(len(values) for values in epoch_ids) - len(set().union(*epoch_ids)), 1)
        self.assertEqual(schedules[0].audit()["base_coverage_cycle_epochs"], 3)
        self.assertTrue(schedules[2].audit()["base_window_wraps"])

    def test_base_coverage_cycle_is_deterministic_and_reshuffles_next_cycle(self):
        base = [reference(BASE_SOURCE, index) for index in range(179)]
        b20 = [reference(B20_SOURCE, index) for index in range(20)]

        def schedule(epoch):
            return DeterministicShardGroupMixtureSchedule(
                base,
                b20,
                seed=13,
                epoch=epoch,
                length_bucket_width=64,
                catalog_sha256="catalog_hash",
            )

        epoch_zero = [item.condition_id for item in schedule(0).base]
        self.assertEqual(epoch_zero, [item.condition_id for item in schedule(0).base])
        self.assertNotEqual(epoch_zero, [item.condition_id for item in schedule(3).base])

    def test_optimizer_batch_count_is_deterministic(self):
        schedule = self.schedule()
        count_a = count_optimizer_batches(
            schedule,
            world_size=2,
            conditions_per_rank=4,
            max_tokens_per_gpu=420,
        )
        count_b = count_optimizer_batches(
            schedule,
            world_size=2,
            conditions_per_rank=4,
            max_tokens_per_gpu=420,
        )
        self.assertEqual(count_a, count_b)
        self.assertGreater(count_a, 0)

    def test_confirmation_condition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "train/dev"):
            reference(BASE_SOURCE, 1, split="confirmation").validate()


if __name__ == "__main__":
    unittest.main()
