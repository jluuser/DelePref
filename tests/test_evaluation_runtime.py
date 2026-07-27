import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from delepref.student_model import (
    DeletionPreferenceStudent,
    strict_budget_dp_decode,
    strict_delete_count,
)
from scripts import freeze_evaluation_checkpoint as freeze
from scripts import evaluation_runtime as runtime


class StudentEvaluationRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        model = DeletionPreferenceStudent.tiny_for_tests(
            qmax=3,
            max_span_len_fraction=0.5,
            dropout=0.0,
        )
        model.eval()
        self.runtime = runtime.EvaluationRuntime(
            model=model,
            config={
                "policy": {"qmax": 3, "min_span_len": 1, "max_span_len_fraction": 0.5},
                "decoder": {"perturbation_scale": 1.0, "max_attempts_multiplier": 32},
                "batching": {"encoder_microbatch_size": 2},
            },
            checkpoint={"schema_version": runtime.EVALUATION_SCHEMA},
            device=torch.device("cpu"),
            source_checkpoint_sha256="a" * 64,
            selected_step=9296,
            encoder_microbatch_size=2,
            k=8,
        )
        self.records = [
            {"protein_id": "protein-a", "sequence": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL", "budget": 0.10},
            {"protein_id": "protein-b", "sequence": "MNPQRSTVWYACDEFGHIKLMNPQRSTVWY", "budget": 0.20},
        ]

    def test_condition_seed_is_stable_and_order_independent(self):
        keyed = {}
        for record in self.records:
            digest = runtime.sequence_sha256(record["sequence"])
            keyed[record["protein_id"]] = runtime.derive_condition_seed(
                71, record["protein_id"], digest, record["budget"]
            )
        reversed_keyed = {}
        for record in reversed(self.records):
            digest = runtime.sequence_sha256(record["sequence"])
            reversed_keyed[record["protein_id"]] = runtime.derive_condition_seed(
                71, record["protein_id"], digest, record["budget"]
            )
        self.assertEqual(keyed, reversed_keyed)
        self.assertNotEqual(keyed["protein-a"], keyed["protein-b"])

    def test_strict_delete_count_for_supported_budgets(self):
        self.assertEqual(strict_delete_count(101, 0.10), 10)
        self.assertEqual(strict_delete_count(101, 0.20), 20)

    def test_runtime_qmax_is_three(self):
        self.assertEqual(runtime.QMAX, 3)
        self.assertEqual(self.runtime.config["policy"]["qmax"], 3)

    def test_generated_intervals_are_zero_based_inclusive_and_nonoverlapping(self):
        chunk = runtime.encode_chunk(self.runtime, [self.records[0]])
        rows = runtime.generate_raw_candidates(self.runtime, chunk, global_generation_seed=3)
        self.assertEqual(len(rows), 8)
        for row in rows:
            intervals = json.loads(row["deleted_intervals"])
            self.assertLessEqual(len(intervals), 3)
            previous_end = -2
            deleted = 0
            for start, end in intervals:
                self.assertGreaterEqual(start, 0)
                self.assertGreaterEqual(end, start)
                self.assertGreater(start, previous_end + 1)
                previous_end = end
                deleted += end - start + 1
            self.assertEqual(deleted, strict_delete_count(row["sequence_length"], row["budget"]))
            self.assertEqual(row["coordinate_system"], "0_based_closed_inclusive")

    def test_exact_dp_is_deterministic(self):
        logits = torch.tensor([0.0, 2.0, 2.0, -1.0, 3.0, 0.0, 3.0, -2.0])
        first = strict_budget_dp_decode(logits, 0.20, qmax=3, min_span_len=1, max_span_len=4)
        for _ in range(10):
            self.assertEqual(first, strict_budget_dp_decode(logits, 0.20, qmax=3, min_span_len=1, max_span_len=4))

    def test_k_candidates_are_unique(self):
        chunk = runtime.encode_chunk(self.runtime, [self.records[0]])
        rows = runtime.generate_raw_candidates(self.runtime, chunk, global_generation_seed=101)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["deletion_mask_sha256"] for row in rows}), 8)
        self.assertEqual(sorted(row["raw_candidate_rank"] for row in rows), list(range(1, 9)))

    def test_raw_candidates_serialize_before_scorer(self):
        chunk = runtime.encode_chunk(self.runtime, [self.records[0]])
        rows = runtime.generate_raw_candidates(self.runtime, chunk, global_generation_seed=23)
        self.assertTrue(all(row["scorer_score"] == "" for row in rows))
        json.dumps(rows, sort_keys=True)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.jsonl"
            runtime.emit_canonical_rows(rows, path)
            parsed = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(parsed), 8)
        self.assertTrue(all(item["scorer_score"] == "" for item in parsed))

    def test_scorer_tie_break_is_deterministic_and_lexicographic(self):
        record = {"protein_id": "tie", "sequence": "ACDEFGHIKL", "budget": 0.20}
        chunk = runtime.encode_chunk(self.runtime, [record])
        base = {
            "student_version": "delepref",
            "source_checkpoint_sha256": "a" * 64,
            "selected_checkpoint_step": 1,
            "protein_id": "tie",
            "sequence_sha256": runtime.sequence_sha256(record["sequence"]),
            "sequence_length": 10,
            "budget": 0.20,
            "generation_seed": 1,
            "condition_seed": 2,
            "deleted_length": 2,
            "span_count": 1,
            "policy_energy": 0.0,
            "policy_native_rank": 1,
            "decoder": runtime.DECODER_NAME,
            "qmax": 3,
            "generation_attempts": 2,
            "duplicate_decodes": 0,
            "generation_reason": "complete",
            "strict_plan_valid": True,
            "student_confidence_available": False,
            "student_confidence_used_for_ranking": False,
            "preference_confidence": "",
            "coordinate_system": "0_based_closed_inclusive",
            "wildtype_sequence": record["sequence"],
        }
        rows = []
        for raw_rank, intervals in enumerate(([[5, 6]], [[0, 1]]), start=1):
            item = dict(base)
            item.update(
                {
                    "raw_candidate_rank": raw_rank,
                    "deleted_intervals": json.dumps(intervals, separators=(",", ":")),
                    "deletion_mask_sha256": hashlib.sha256(str(intervals).encode()).hexdigest(),
                }
            )
            rows.append(item)
        with mock.patch.object(
            self.runtime.model,
            "score_plans_from_embeddings",
            return_value={"preference_score": torch.tensor([0.5, 0.5])},
        ):
            scored = runtime.score_candidates(self.runtime, chunk, rows)
        selected = runtime.select_scorer_top1(scored)
        self.assertEqual(selected[0]["deleted_intervals"], "[[0,1]]")
        self.assertEqual([item["scorer_rank"] for item in scored], [2, 1])

    def test_confidence_fields_are_blank_and_unavailable(self):
        chunk = runtime.encode_chunk(self.runtime, [self.records[0]])
        rows = runtime.emit_canonical_rows(
            runtime.generate_raw_candidates(self.runtime, chunk, global_generation_seed=19)
        )
        self.assertTrue(all(item["student_confidence_available"] is False for item in rows))
        self.assertTrue(all(item["student_confidence_used_for_ranking"] is False for item in rows))
        self.assertTrue(all(item["preference_confidence"] == "" for item in rows))

    def test_length_1023_is_rejected_without_truncation(self):
        with self.assertRaisesRegex(ValueError, "2..1022"):
            runtime.encode_chunk(
                self.runtime,
                [{"protein_id": "too-long", "sequence": "A" * 1023, "budget": 0.10}],
            )

    def test_budget_030_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            runtime.encode_chunk(
                self.runtime,
                [{"protein_id": "bad-budget", "sequence": "ACDE", "budget": 0.30}],
            )

    def _freeze_fixture(self, root: Path, *, complete: bool = True, selection_hash: str = ""):
        checkpoint_root = root / "checkpoints"
        run_root = checkpoint_root / "delepref_final_training"
        run_root.mkdir(parents=True)
        config_path = root / "config.yaml"
        config = {
            "model": {"max_length": 1022, "scorer_hidden_dim": 64, "span_feature_dim": 32},
            "policy": {
                "qmax": 3,
                "min_span_len": 1,
                "max_span_len_fraction": 0.5,
                "budget_hidden_dim": 16,
                "policy_hidden_dim": 64,
                "span_hidden_dim": 64,
            },
            "decoder": {"num_candidates": 8, "perturbation_scale": 1.0, "max_attempts_multiplier": 16},
            "batching": {"encoder_microbatch_size": 2},
            "proposal_validation": {"budgets": [0.10, 0.20]},
            "training": {"checkpoint_dir": str(checkpoint_root)},
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=True))
        source = run_root / "checkpoint_step_00000004.pt"
        torch.save(
            {
                "schema_version": freeze.STREAMING_SCHEMA,
                "student_version": freeze.STUDENT_VERSION,
                "run_id": "delepref_final_training",
                "global_step": 4,
                "total_steps": 4,
                "student_head_state_dict": {
                    key: value.detach().clone()
                    for key, value in self.runtime.model.state_dict().items()
                    if not key.startswith("encoder.")
                },
                "optimizer_state_dict": {"state": {}, "param_groups": []},
                "scheduler_state_dict": {"last_epoch": 4},
                "rank_states": [{"rank": 0, "rng": {"python": (1, 2, 3)}, "planner": {"global_cursor": 1}}],
                "config_sha256": freeze.canonical_sha256(config),
                "data_hashes": {"training_data": "b" * 64},
                "interrupted": False,
                "pending_validation_step": None,
                "encoder_weights_saved": False,
                "confirmation_manifest_opened": False,
                "confirmation_metrics_viewed": False,
            },
            source,
        )
        result_root = root / "results"
        result_root.mkdir()
        selected = {
            "checkpoint_path": str(source),
            "step": 4,
            "legality_gates_pass": True,
            "confirmation_metrics_used": False,
            "forbidden_metrics_used": [],
        }
        if selection_hash:
            selected["checkpoint_sha256"] = selection_hash
        selection = {
            "status": "PASS",
            "confirmation_metrics_used": False,
            "selected": selected,
            "candidates": [dict(selected)],
        }
        selection_path = result_root / "checkpoint_selection.json"
        selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n")
        summary_path = result_root / "run_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "status": "BOUNDED_COMPLETE" if complete else "INTERRUPTED_CHECKPOINTED",
                    "run_id": "delepref_final_training",
                    "global_step": 4 if complete else 3,
                    "total_steps": 4,
                    "interrupted": not complete,
                    "proposal_validation_skipped": False,
                    "confirmation_metrics_viewed": False,
                }
            )
            + "\n"
        )
        completion_root = result_root / "proposal_validation/step_00000004"
        completion_root.mkdir(parents=True)
        (completion_root / "completion.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "step": 4,
                    "selected_step": 4,
                    "checkpoint_path": str(source),
                    "checkpoint_selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        return config_path, selection_path, summary_path, source

    def test_evaluation_checkpoint_excludes_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, selection, summary, _source = self._freeze_fixture(root)
            output = root / "evaluation"
            path, _report = freeze.export_evaluation_checkpoint(selection, summary, config, output)
            payload = torch.load(path, map_location="cpu", weights_only=True)
        forbidden = {"optimizer_state_dict", "scheduler_state_dict", "rank_states", "rng", "sampler_cursor"}
        self.assertFalse(forbidden.intersection(payload))
        self.assertTrue(payload["student_head_state_dict"])

    def test_evaluation_checkpoint_loads_with_weights_only_true(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, selection, summary, _source = self._freeze_fixture(root)
            path, report = freeze.export_evaluation_checkpoint(selection, summary, config, root / "evaluation")
            payload = torch.load(path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["schema_version"], freeze.EXPORT_SCHEMA)
        self.assertEqual(report["weights_only_reload"], "PASS")

    def test_incomplete_training_inspection_returns_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _config, selection, summary, _source = self._freeze_fixture(root, complete=False)
            inspected = freeze.inspect_training_state(selection, summary)
        self.assertEqual(inspected["status"], freeze.PENDING_STATUS)
        self.assertFalse(inspected["exported"])

    def test_selected_checkpoint_source_hash_mismatch_fails_loud(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, selection, summary, _source = self._freeze_fixture(root, selection_hash="0" * 64)
            with self.assertRaisesRegex(freeze.FreezeValidationError, "hash mismatch"):
                freeze.validate_export_inputs(selection, summary, config)

    def test_runtime_has_no_legacy_model_fallback(self):
        source = Path(runtime.__file__).read_text(encoding="utf-8")
        self.assertNotIn("stage1_style", source)
        self.assertNotIn("structured_span_refiner", source)

    def test_canonical_csv_is_readable(self):
        chunk = runtime.encode_chunk(self.runtime, [self.records[0]])
        rows = runtime.generate_raw_candidates(self.runtime, chunk, global_generation_seed=31)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.csv"
            runtime.emit_canonical_rows(rows, path)
            with path.open(newline="") as handle:
                parsed = list(csv.DictReader(handle))
        self.assertEqual(len(parsed), 8)


if __name__ == "__main__":
    unittest.main()
