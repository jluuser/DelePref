import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from delepref.preference_core import DeletionCandidate, PreferencePair
from delepref.student_model import (
    DeletionPreferenceStudent,
    joint_objective_normalization_counts,
)
from scripts import train_final as trainer


class Wrapper:
    def __init__(self, module):
        self.module = module


class Planner:
    def __init__(self, cursor):
        self.cursor = cursor

    def state_dict(self):
        return {
            "schema_version": "test_planner",
            "global_cursor": self.cursor,
            "rank": 0,
        }


def context():
    return {
        "rank": 0,
        "local_rank": 0,
        "world_size": 1,
        "identity": {"device_uuid": "cpu-test"},
    }


def rank_state(planner):
    return {
        "rank": 0,
        "device_uuid": "cpu-test",
        "planner": planner.state_dict(),
        "rng": {"synthetic": True},
    }


class TrainerCheckpointTests(unittest.TestCase):
    def make_training(self, seed=7):
        torch.manual_seed(seed)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
        scheduler = trainer.build_scheduler(
            optimizer, total_steps=4, warmup_steps=0, min_lr_ratio=0.1
        )
        return model, optimizer, scheduler

    @staticmethod
    def step(model, optimizer, scheduler, inputs, targets):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        return float(loss.detach())

    def save(self, path, model, optimizer, scheduler, planner, step, pending_validation_step=None):
        preflight = {"config_sha256": "config", "data_hashes": {"catalog": "hash"}}

        def gather(output, value):
            output[0] = value

        with mock.patch.object(trainer, "capture_rank_state", side_effect=lambda _c, p: rank_state(p)), mock.patch.object(
            trainer.dist, "all_gather_object", side_effect=gather
        ), mock.patch.object(trainer.dist, "barrier", return_value=None):
            trainer.save_checkpoint(
                path,
                ddp_model=Wrapper(model),
                optimizer=optimizer,
                scheduler=scheduler,
                context=context(),
                planner=planner,
                global_step=step,
                total_steps=4,
                epoch=0,
                config={"test": True},
                preflight_report=preflight,
                run_id="test_run",
                run_mode="release_smoke",
                recent_metrics=[{"step": step}],
                interrupted=False,
                pending_validation_step=pending_validation_step,
            )

    def load(self, path, model, optimizer, scheduler):
        with mock.patch.object(trainer, "restore_rng_state", return_value=None):
            return trainer.load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                context=context(),
                config_sha256="config",
                data_hashes={"catalog": "hash"},
                run_mode="release_smoke",
                total_steps=4,
            )

    def test_checkpoint_roundtrip_restores_heads_optimizer_scheduler_and_cursor(self):
        model, optimizer, scheduler = self.make_training()
        original = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint_step_00000000.pt"
            self.save(path, model, optimizer, scheduler, Planner(12), 0)
            for parameter in model.parameters():
                parameter.data.add_(10.0)
            payload, planner_state = self.load(path, model, optimizer, scheduler)
            with self.assertRaises(FileExistsError):
                self.save(path, model, optimizer, scheduler, Planner(12), 0)
        self.assertEqual(payload["global_step"], 0)
        self.assertEqual(payload["total_steps"], 4)
        self.assertEqual(planner_state["global_cursor"], 12)
        self.assertFalse(payload["encoder_weights_saved"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0.0, atol=0.0)

    def test_interrupted_resume_matches_uninterrupted_trajectory_exactly(self):
        inputs = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
        targets = torch.tensor([[0.25], [-0.75]])
        baseline_model, baseline_optimizer, baseline_scheduler = self.make_training(seed=19)
        for _ in range(4):
            self.step(baseline_model, baseline_optimizer, baseline_scheduler, inputs, targets)

        first_model, first_optimizer, first_scheduler = self.make_training(seed=19)
        for _ in range(2):
            self.step(first_model, first_optimizer, first_scheduler, inputs, targets)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint_step_00000002.pt"
            self.save(path, first_model, first_optimizer, first_scheduler, Planner(24), 2)
            resumed_model, resumed_optimizer, resumed_scheduler = self.make_training(seed=999)
            payload, planner_state = self.load(
                path, resumed_model, resumed_optimizer, resumed_scheduler
            )
            for _ in range(2):
                self.step(resumed_model, resumed_optimizer, resumed_scheduler, inputs, targets)

        self.assertEqual(payload["global_step"], 2)
        self.assertEqual(planner_state["global_cursor"], 24)
        self.assertEqual(
            baseline_optimizer.param_groups[0]["lr"], resumed_optimizer.param_groups[0]["lr"]
        )
        for baseline, resumed in zip(baseline_model.parameters(), resumed_model.parameters()):
            torch.testing.assert_close(baseline, resumed, rtol=0.0, atol=0.0)

    def test_resume_rejects_total_step_or_mode_change(self):
        model, optimizer, scheduler = self.make_training()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint_step_00000000.pt"
            self.save(path, model, optimizer, scheduler, Planner(0), 0)
            with mock.patch.object(trainer, "restore_rng_state", return_value=None):
                with self.assertRaisesRegex(ValueError, "total-step"):
                    trainer.load_checkpoint(
                        path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        context=context(),
                        config_sha256="config",
                        data_hashes={"catalog": "hash"},
                        run_mode="release_smoke",
                        total_steps=5,
                    )

    def test_resume_allows_a_new_gpu_uuid_but_preserves_rank_state(self):
        model, optimizer, scheduler = self.make_training()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint_step_00000000.pt"
            self.save(path, model, optimizer, scheduler, Planner(12), 0)
            new_context = context()
            new_context["identity"] = {"device_uuid": "replacement-a100"}
            with mock.patch.object(trainer, "restore_rng_state", return_value=None):
                payload, planner_state = trainer.load_checkpoint(
                    path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    context=new_context,
                    config_sha256="config",
                    data_hashes={"catalog": "hash"},
                    run_mode="release_smoke",
                    total_steps=4,
                )
        self.assertEqual(payload["rank_states"][0]["device_uuid"], "cpu-test")
        self.assertEqual(planner_state["global_cursor"], 12)

    def test_checkpoint_records_pending_validation_step(self):
        model, optimizer, scheduler = self.make_training()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint_step_00000002.pt"
            self.save(
                path,
                model,
                optimizer,
                scheduler,
                Planner(24),
                2,
                pending_validation_step=2,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["pending_validation_step"], 2)

    def test_fresh_run_directory_reuse_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "outputs/run"
            checkpoints = Path(tmp) / "checkpoints/run"
            trainer.prepare_run_directories(output, checkpoints, resume=False)
            self.assertTrue(output.is_dir())
            self.assertTrue(checkpoints.is_dir())
            with self.assertRaisesRegex(FileExistsError, "already exist"):
                trainer.prepare_run_directories(output, checkpoints, resume=False)
            trainer.prepare_run_directories(output, checkpoints, resume=True)

    def test_frozen_asset_verification_detects_content_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = root / "asset.txt"
            frozen.write_text("frozen\n", encoding="utf-8")
            digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
            aggregate = hashlib.sha256()
            aggregate.update(b"asset.txt\0")
            aggregate.update(str(frozen.stat().st_size).encode("ascii"))
            aggregate.update(b"\0")
            aggregate.update(digest.encode("ascii"))
            aggregate.update(b"\n")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "label": "toy",
                                "path": str(frozen),
                                "kind": "file",
                                "file_count": 1,
                                "size_bytes": frozen.stat().st_size,
                                "sha256": aggregate.hexdigest(),
                                "files": [
                                    {
                                        "path": str(frozen),
                                        "relative_path": "asset.txt",
                                        "size_bytes": frozen.stat().st_size,
                                        "sha256": digest,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = trainer.verify_frozen_asset_manifest(
                manifest, required_labels=("toy",)
            )
            self.assertEqual(report["assets"][0]["label"], "toy")
            frozen.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size changed|digest changed"):
                trainer.verify_frozen_asset_manifest(manifest, required_labels=("toy",))

    def test_pending_proposal_validation_is_completed_for_the_same_step(self):
        report = {
            "status": "PASS",
            "metrics": {
                "teacher_rescored_selected_utility_equal_weight_per_protein_budget_seed": 0.9,
                "teacher_rescored_best_of_k": 0.95,
            },
            "legality_gates": {"legal": True},
        }
        cheap = {"status": "PASS", "step": 2000, "strict_scorer_accuracy": 0.8}
        selection = {"selected": {"step": 2000}}

        def broadcast(values, src):
            self.assertEqual(src, 0)
            self.assertEqual(values, ["PASS"])

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            trainer.dist, "barrier", return_value=None
        ), mock.patch.object(
            trainer.dist, "broadcast_object_list", side_effect=broadcast
        ), mock.patch.object(
            trainer, "run_real_proposal_validation", return_value=report
        ) as real_validation, mock.patch.object(
            trainer, "update_checkpoint_selection", return_value=selection
        ), mock.patch.object(
            trainer, "sha256_file", return_value="digest"
        ):
            observed = trainer.complete_proposal_validation(
                object(),
                {"checkpoint_selection": {"tie_tolerance": 1.0e-6}},
                {},
                Path(tmp),
                2000,
                Path(tmp) / "checkpoint_step_00002000.pt",
                cheap,
                rank=0,
            )
        self.assertIs(observed, report)
        real_validation.assert_called_once()

    def test_pending_proposal_validation_rejects_wrong_cheap_step(self):
        with self.assertRaisesRegex(RuntimeError, "same-step"):
            trainer.complete_proposal_validation(
                object(),
                {"checkpoint_selection": {"tie_tolerance": 1.0e-6}},
                {},
                Path("unused"),
                2000,
                Path("checkpoint_step_00002000.pt"),
                {"status": "PASS", "step": 1500},
                rank=0,
            )

    def test_cuda_oom_failure_artifact_is_fail_loud_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "delepref_esmc600m_same_count/delepref/test"
            config_path = Path(tmp) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump({"training": {"output_dir": str(output)}}),
                encoding="utf-8",
            )
            error = torch.cuda.OutOfMemoryError("synthetic out of memory")
            try:
                raise error
            except torch.cuda.OutOfMemoryError as caught:
                path = trainer.write_failure_artifact(caught, ["--config", str(config_path)])
            self.assertIsNotNone(path)
            payload = trainer.read_json(path)
        self.assertEqual(payload["status"], "FAIL")
        self.assertTrue(payload["cuda_oom"])
        self.assertFalse(payload["automatic_batch_shrink_attempted"])
        self.assertFalse(payload["automatic_retry_attempted"])


class TrainerNormalizationTests(unittest.TestCase):
    @staticmethod
    def pair(index):
        sequence = "ACDEFGHIKLMNPQRSTVWY"

        def candidate(name, intervals):
            deleted = sum(end - start + 1 for start, end in intervals)
            return DeletionCandidate(
                protein_id="protein_{}".format(index),
                perturbation_id=name,
                plan_id=name,
                wildtype_sequence=sequence,
                deleted_intervals=tuple(intervals),
                deleted_length=deleted,
                deletion_fraction=deleted / float(len(sequence)),
                target_delete_fraction=0.2,
                budget_bin="b20",
                generation_policy="unit_test",
            )

        a = candidate("a_{}".format(index), ((index, index + 3),))
        b = candidate("b_{}".format(index), ((10 + index, 11 + index), (16, 17)))
        return PreferencePair(
            pair_id="pair_{}".format(index),
            protein_id=a.protein_id,
            budget_bin="b20",
            preference="a_better" if index % 2 == 0 else "b_better",
            preference_confidence=0.9,
            loss_weight=0.5 + 0.1 * index,
            preference_margin=0.1,
            is_valid_pair=True,
            construction_policy="strict_same_count_unit_test",
            quality_status="ok",
            quality_reason="unit_test",
            a=a,
            b=b,
            pair_quality_bucket="high",
            policy_positive_eligible=True,
            scorer_training_weight=0.5 + 0.1 * index,
        )

    def test_component_normalization_is_invariant_to_microbatch_partition(self):
        torch.manual_seed(23)
        full_model = DeletionPreferenceStudent.tiny_for_tests(qmax=3, dropout=0.0)
        split_model = DeletionPreferenceStudent.tiny_for_tests(qmax=3, dropout=0.0)
        split_model.load_state_dict(full_model.state_dict())
        pairs = [self.pair(index) for index in range(4)]
        config = {
            "training": {"scorer_loss_weight": 1.0, "proposal_loss_weight": 1.0},
            "policy": {
                "rank_weight": 1.0,
                "imitation_weight": 0.2,
                "budget_aux_weight": 0.1,
                "structure_aux_weight": 0.05,
            },
        }
        full_info = full_model.joint_objective(pairs)
        full_info["loss"].backward()
        global_counts = joint_objective_normalization_counts(pairs)
        split_loss = None
        for microbatch in (pairs[:2], pairs[2:]):
            info = split_model.joint_objective(microbatch)
            components = trainer.normalized_joint_components(
                info,
                joint_objective_normalization_counts(microbatch),
                global_counts,
                config,
                world_size=1,
            )
            split_loss = components["loss"] if split_loss is None else split_loss + components["loss"]
        self.assertIsNotNone(split_loss)
        split_loss.backward()
        torch.testing.assert_close(split_loss.detach(), full_info["loss"].detach(), rtol=1e-6, atol=1e-7)
        full_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in full_model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        split_gradients = {
            name: parameter.grad.detach().clone()
            for name, parameter in split_model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        self.assertEqual(set(full_gradients), set(split_gradients))
        for name in full_gradients:
            torch.testing.assert_close(full_gradients[name], split_gradients[name], rtol=2e-5, atol=2e-6)

    def test_padding_fraction_counts_actual_encoder_microbatches(self):
        encoded = type(
            "Encoded",
            (),
            {"sequences": {"a": "A" * 10, "b": "A" * 6, "c": "A" * 4}},
        )()
        actual, padded = trainer.encoder_padding_counts(encoded, microbatch_size=2)
        self.assertEqual(actual, 20)
        self.assertEqual(padded, 24)


if __name__ == "__main__":
    unittest.main()
