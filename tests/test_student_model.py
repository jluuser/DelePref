import csv
import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from delepref.preference_core import DeletionCandidate, PreferencePair
from delepref.student_model import (
    DeletionPreferenceStudent,
    FINAL_QMAX,
    ProposalOutput,
    head_only_state_dict,
    load_student_checkpoint,
    normalized_selected_residue_energy,
    proposal_candidate_losses,
    proposal_pairwise_objective_from_embeddings,
    save_student_checkpoint,
    scorer_pairwise_objective_from_embeddings,
    strict_budget_dp_decode,
    strict_delete_count,
    validate_strict_preference_pair,
)
from scripts import generate_plans as generate_delepref
from scripts import train_student_pilot as train_delepref


def candidate(protein_id, sequence, name, intervals, beta):
    count = sum(end - start + 1 for start, end in intervals)
    return DeletionCandidate(
        protein_id=protein_id,
        perturbation_id=name,
        plan_id=name,
        wildtype_sequence=sequence,
        deleted_intervals=tuple(intervals),
        deleted_length=count,
        deletion_fraction=count / float(len(sequence)),
        target_delete_fraction=float(beta),
        budget_bin="strict",
        generation_policy="unit_test",
    )


def pair(name, a, b, preference="a_better", weight=1.0, eligible=True, confidence=0.9):
    return PreferencePair(
        pair_id=name,
        protein_id=a.protein_id,
        budget_bin="strict",
        preference=preference,
        preference_confidence=confidence,
        loss_weight=weight,
        preference_margin=0.1,
        is_valid_pair=True,
        construction_policy="strict_same_count_unit_test",
        quality_status="ok",
        quality_reason="unit_test",
        a=a,
        b=b,
        pair_quality_bucket="high",
        policy_positive_eligible=eligible,
        scorer_training_weight=weight,
    )


class StudentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = DeletionPreferenceStudent.tiny_for_tests(qmax=3, dropout=0.0)
        self.sequence = "ACDEFGHIKLMNPQRSTVWY"
        self.beta = 0.2
        self.a = candidate("p1", self.sequence, "a", [(0, 3)], self.beta)
        self.b = candidate("p1", self.sequence, "b", [(8, 9), (14, 15)], self.beta)

    def test_batch_encodes_each_protein_hash_once_and_reuses_embeddings(self):
        records = [self.a, self.b, self.a, self.b]
        with mock.patch.object(
            self.model.encoder.encoder,
            "forward",
            wraps=self.model.encoder.encoder.forward,
        ) as forward:
            encoded = self.model.encode_unique_wildtypes(records)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(encoded.input_count, 4)
        self.assertEqual(encoded.unique_count, 1)
        self.assertEqual(self.model.encoder.unique_wildtypes_encoded, 1)

        scorer_output = self.model.score_plans_from_embeddings([self.a, self.b], encoded)
        policy_output = self.model.propose_from_embeddings([encoded.for_candidate(self.a)], [self.beta])
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(tuple(scorer_output["preference_score"].shape), (2,))
        self.assertEqual(len(policy_output), 1)

    def test_encoder_microbatch_is_explicit_and_forward_is_ddp_safe(self):
        records = [
            candidate("micro_{}".format(index), self.sequence, "m{}".format(index), [(0, 3)], self.beta)
            for index in range(3)
        ]
        with mock.patch.object(
            self.model.encoder.encoder,
            "forward",
            wraps=self.model.encoder.encoder.forward,
        ) as forward:
            encoded = self.model.encode_unique_wildtypes(records, microbatch_size=2)
        self.assertEqual(encoded.unique_count, 3)
        self.assertEqual(forward.call_count, 2)
        with self.assertRaisesRegex(ValueError, "microbatch_size"):
            self.model.encode_unique_wildtypes(records, microbatch_size=0)
        info = self.model(
            [pair("ddp_forward", self.a, self.b)],
            encoder_microbatch_size=1,
        )
        self.assertTrue(torch.isfinite(info["loss"]).item())

    def test_scorer_and_policy_share_exact_encoder_and_checkpoint_excludes_it(self):
        self.assertIs(self.model.scorer.encoder, self.model.encoder)
        self.assertIs(self.model.policy.encoder, self.model.encoder)
        self.assertIs(self.model.scorer.encoder.backend, self.model.policy.encoder.backend)
        self.assertFalse(any(parameter.requires_grad for parameter in self.model.encoder.parameters()))
        state = head_only_state_dict(self.model)
        self.assertTrue(state)
        self.assertFalse(any(key.startswith("encoder.") for key in state))

    def test_boundary_scorer_handles_terminal_single_residue_and_multispan(self):
        terminal_single = candidate("p1", self.sequence, "terminal_single", [(0, 0)], 0.05)
        right_terminal = candidate("p1", self.sequence, "right_terminal", [(19, 19)], 0.05)
        multi = candidate("p1", self.sequence, "multi", [(1, 1), (10, 10)], 0.1)
        encoded = self.model.encode_unique_wildtypes([terminal_single, right_terminal, multi])
        output = self.model.score_plans_from_embeddings([terminal_single, right_terminal, multi], encoded)
        self.assertEqual(tuple(output["preference_score"].shape), (3,))
        self.assertTrue(torch.isfinite(output["preference_score"]).all().item())

    def test_rank_fails_fast_for_different_beta_deleted_count_or_protein(self):
        valid = pair("valid", self.a, self.b)
        validate_strict_preference_pair(valid)

        cross_beta_b = candidate("p1", self.sequence, "cross_beta", [(8, 9), (14, 15)], 0.25)
        with self.assertRaisesRegex(ValueError, "crosses beta"):
            validate_strict_preference_pair(pair("cross_beta", self.a, cross_beta_b))

        cross_count_b = candidate("p1", self.sequence, "cross_count", [(8, 10)], self.beta)
        with self.assertRaisesRegex(ValueError, "deleted_length"):
            validate_strict_preference_pair(pair("cross_count", self.a, cross_count_b))

        other_protein = candidate("p2", self.sequence, "other", [(8, 9), (14, 15)], self.beta)
        with self.assertRaisesRegex(ValueError, "protein_id"):
            validate_strict_preference_pair(pair("cross_protein", self.a, other_protein))

    def test_pair_swap_with_reversed_label_is_objective_invariant(self):
        forward = pair("forward", self.a, self.b, preference="a_better", weight=0.7)
        swapped = pair("swapped", self.b, self.a, preference="b_better", weight=0.7)
        encoded = self.model.encode_unique_wildtypes([self.a, self.b])
        scorer_forward = scorer_pairwise_objective_from_embeddings(
            self.model.scorer, [forward], encoded
        )["loss"]
        scorer_swapped = scorer_pairwise_objective_from_embeddings(
            self.model.scorer, [swapped], encoded
        )["loss"]
        policy_forward = proposal_pairwise_objective_from_embeddings(
            self.model.policy, [forward], encoded
        )["loss"]
        policy_swapped = proposal_pairwise_objective_from_embeddings(
            self.model.policy, [swapped], encoded
        )["loss"]
        self.assertAlmostEqual(float(scorer_forward), float(scorer_swapped), places=7)
        self.assertAlmostEqual(float(policy_forward), float(policy_swapped), places=7)

    def test_proposal_losses_are_normalized_by_L_Qmax_and_matched_span_count(self):
        model = DeletionPreferenceStudent.tiny_for_tests(qmax=2, dropout=0.0)

        def zero_output(length, beta):
            zero_logits = torch.zeros(length, requires_grad=True)
            return ProposalOutput(
                residue_logits=zero_logits,
                raw_residue_logits=zero_logits,
                activity_logits=torch.zeros(2, requires_grad=True),
                centers=torch.tensor([0.25, 0.75], requires_grad=True),
                lengths_fraction=torch.tensor([0.2, 0.2], requires_grad=True),
                span_prior=torch.zeros(length),
                beta=beta,
            )

        short = candidate("short", "A" * 20, "short", [(2, 5)], 0.2)
        long = candidate("long", "A" * 40, "long", [(4, 11)], 0.2)
        short_losses = proposal_candidate_losses(model.policy, short, zero_output(20, 0.2))
        long_losses = proposal_candidate_losses(model.policy, long, zero_output(40, 0.2))
        self.assertAlmostEqual(float(short_losses["mask_bce"]), float(long_losses["mask_bce"]), places=6)
        self.assertAlmostEqual(float(short_losses["activity_bce"]), float(long_losses["activity_bce"]), places=6)
        self.assertLess(abs(float(short_losses["imitation_loss"] - long_losses["imitation_loss"])), 0.05)

    def test_tie_is_equal_constraint_and_never_an_imitation_positive(self):
        strict_pair = pair("tie", self.a, self.b, preference="tie", weight=0.8, eligible=True)
        encoded = self.model.encode_unique_wildtypes([self.a, self.b])
        info = proposal_pairwise_objective_from_embeddings(self.model.policy, [strict_pair], encoded)
        self.assertEqual(info["num_tie_used"], 1)
        self.assertEqual(info["num_rank_used"], 0)
        self.assertEqual(info["num_imitation_used"], 0)
        self.assertGreaterEqual(float(info["tie_loss"].detach()), 0.0)

        low_weight_info = proposal_pairwise_objective_from_embeddings(
            self.model.policy, [strict_pair], encoded, tie_weight_scale=0.001
        )
        high_weight_info = proposal_pairwise_objective_from_embeddings(
            self.model.policy, [strict_pair], encoded, tie_weight_scale=1.0
        )
        self.assertAlmostEqual(
            float(low_weight_info["tie_loss"]),
            float(high_weight_info["tie_loss"]) * 0.001,
            places=6,
        )
        low_scorer_tie = scorer_pairwise_objective_from_embeddings(
            self.model.scorer, [strict_pair], encoded, tie_weight_scale=0.001
        )
        high_scorer_tie = scorer_pairwise_objective_from_embeddings(
            self.model.scorer, [strict_pair], encoded, tie_weight_scale=1.0
        )
        self.assertAlmostEqual(
            float(low_scorer_tie["loss"]),
            float(high_scorer_tie["loss"]) * 0.001,
            places=6,
        )

        ineligible = pair("ineligible", self.a, self.b, preference="a_better", eligible=False)
        ineligible_info = proposal_pairwise_objective_from_embeddings(self.model.policy, [ineligible], encoded)
        self.assertEqual(ineligible_info["num_rank_used"], 1)
        self.assertEqual(ineligible_info["num_imitation_used"], 0)
        self.assertEqual(ineligible_info["num_budget_conditions"], 1)

    def test_joint_objective_has_nonzero_scorer_and_policy_gradients(self):
        strict_pair = pair("train", self.a, self.b, preference="a_better", weight=0.7)
        info = self.model.joint_objective([strict_pair])
        info["loss"].backward()
        scorer_grad = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in self.model.scorer.parameters()
            if parameter.grad is not None
        )
        policy_grad = sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in self.model.policy.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(scorer_grad, 0.0)
        self.assertGreater(policy_grad, 0.0)
        self.assertEqual(info["encoded_wildtypes"].unique_count, 1)
        self.assertEqual(info["proposal"]["num_unique_policy_forwards"], 1)

    def test_pair_quality_weight_scales_rank_imitation_and_auxiliary_at_batch_one(self):
        high = pair("high", self.a, self.b, preference="a_better", weight=1.0)
        low = pair("low", self.a, self.b, preference="a_better", weight=0.05)
        encoded = self.model.encode_unique_wildtypes([self.a, self.b])
        scorer_high = scorer_pairwise_objective_from_embeddings(self.model.scorer, [high], encoded)["loss"]
        scorer_low = scorer_pairwise_objective_from_embeddings(self.model.scorer, [low], encoded)["loss"]
        proposal_high = proposal_pairwise_objective_from_embeddings(
            self.model.policy, [high], encoded, budget_aux_weight=0.0
        )["loss"]
        proposal_low = proposal_pairwise_objective_from_embeddings(
            self.model.policy, [low], encoded, budget_aux_weight=0.0
        )["loss"]
        self.assertAlmostEqual(float(scorer_low), float(scorer_high) * 0.05, places=6)
        self.assertAlmostEqual(float(proposal_low), float(proposal_high) * 0.05, places=6)

    def test_teacher_confidence_is_not_a_student_target(self):
        low_confidence = pair("low_confidence", self.a, self.b, weight=0.7, confidence=0.01)
        high_confidence = pair("high_confidence", self.a, self.b, weight=0.7, confidence=0.99)
        encoded = self.model.encode_unique_wildtypes([self.a, self.b])
        scorer_low = scorer_pairwise_objective_from_embeddings(self.model.scorer, [low_confidence], encoded)["loss"]
        scorer_high = scorer_pairwise_objective_from_embeddings(self.model.scorer, [high_confidence], encoded)["loss"]
        proposal_low = proposal_pairwise_objective_from_embeddings(self.model.policy, [low_confidence], encoded)["loss"]
        proposal_high = proposal_pairwise_objective_from_embeddings(self.model.policy, [high_confidence], encoded)["loss"]
        self.assertEqual(float(scorer_low), float(scorer_high))
        self.assertEqual(float(proposal_low), float(proposal_high))

    def test_no_student_confidence_or_dead_span_score_parameters(self):
        parameter_names = tuple(self.model.state_dict())
        self.assertFalse(any("confidence" in name for name in parameter_names))
        self.assertFalse(any("span_scores" in name for name in parameter_names))
        self.assertEqual(self.model.scorer.plan_head[-1].out_features, 1)
        self.assertEqual(self.model.policy.span_generator[-1].out_features, FINAL_QMAX * 3)
        encoded = self.model.encode_unique_wildtypes([self.a])
        output = self.model.score_plans_from_embeddings([self.a], encoded)
        self.assertEqual(set(output), {"preference_score"})

    def test_imitation_and_budget_are_deduplicated_at_their_required_units(self):
        c = candidate("p1", self.sequence, "c", [(5, 7), (18, 18)], self.beta)
        a_alias = candidate("p1", self.sequence, "a_alias", [(0, 3)], self.beta)
        repeated = [
            pair("a_over_b", self.a, self.b, preference="a_better"),
            pair("a_alias_over_c", a_alias, c, preference="a_better"),
        ]
        encoded = self.model.encode_unique_wildtypes([self.a, a_alias, self.b, c])
        info = proposal_pairwise_objective_from_embeddings(self.model.policy, repeated, encoded)
        self.assertEqual(info["num_imitation_references"], 2)
        self.assertEqual(info["num_imitation_used"], 1)
        self.assertEqual(info["num_imitation_duplicates_removed"], 1)
        self.assertEqual(info["num_span_count_supervision_used"], 1)
        self.assertEqual(info["num_budget_condition_references"], 2)
        self.assertEqual(info["num_budget_conditions"], 1)
        self.assertEqual(info["num_budget_duplicates_removed"], 1)
        self.assertEqual(info["num_negative_candidate_references"], 2)
        self.assertEqual(info["num_negative_supervision_used"], 0)

        no_positive = [
            pair("ineligible_1", self.a, self.b, preference="a_better", eligible=False),
            pair("ineligible_2", self.a, c, preference="a_better", eligible=False),
        ]
        no_positive_info = proposal_pairwise_objective_from_embeddings(self.model.policy, no_positive, encoded)
        self.assertEqual(no_positive_info["num_imitation_used"], 0)
        self.assertEqual(no_positive_info["num_budget_conditions"], 1)
        self.assertEqual(no_positive_info["num_budget_condition_references"], 2)

    def test_normalized_energy_and_dp_sum_have_same_fixed_budget_argmax(self):
        logits = torch.tensor([9.0, 8.0, 7.0, 6.0] + [-4.0] * 16, requires_grad=True)
        output = ProposalOutput(
            residue_logits=logits,
            raw_residue_logits=logits,
            activity_logits=torch.zeros(3, requires_grad=True),
            centers=torch.tensor([0.1, 0.5, 0.9], requires_grad=True),
            lengths_fraction=torch.tensor([0.2, 0.2, 0.2], requires_grad=True),
            span_prior=torch.zeros(20),
            beta=self.beta,
        )
        preferred = candidate("p1", self.sequence, "energy_best", [(0, 3)], self.beta)
        other = candidate("p1", self.sequence, "energy_other", [(8, 11)], self.beta)
        preferred_normalized, preferred_sum = normalized_selected_residue_energy(self.model.policy, preferred, output)
        other_normalized, other_sum = normalized_selected_residue_energy(self.model.policy, other, output)
        self.assertGreater(float(preferred_normalized), float(other_normalized))
        self.assertGreater(float(preferred_sum), float(other_sum))
        decoded, decoded_sum = strict_budget_dp_decode(logits, self.beta, qmax=FINAL_QMAX)
        self.assertEqual(decoded, ((0, 3),))
        self.assertAlmostEqual(decoded_sum, float(preferred_sum), places=6)

    def test_formal_qmax_is_three_in_model_config_and_preflight(self):
        self.assertEqual(FINAL_QMAX, 3)
        self.assertEqual(self.model.policy.qmax, 3)
        self.assertEqual(self.model.scorer.qmax, 3)
        config = train_delepref.load_config("configs/deletion_preference_student_pilot.yaml")
        self.assertEqual(config["policy"]["qmax"], 3)
        config["policy"]["qmax"] = 4
        report = train_delepref.preflight_delepref(config, device_request="cpu")
        self.assertIn("delepref_final_qmax_must_equal_3", report["blockers"])

    def test_four_span_target_is_ineligible_under_final_qmax(self):
        four_span = candidate(
            "p1", self.sequence, "four_span", [(0, 0), (4, 4), (8, 8), (12, 12)], self.beta
        )
        encoded = self.model.encode_unique_wildtypes([four_span])
        output = self.model.propose_from_embeddings([encoded.for_candidate(four_span)], [self.beta])[0]
        with self.assertRaisesRegex(ValueError, "Qmax 3"):
            proposal_candidate_losses(self.model.policy, four_span, output)

    def test_delepref_checkpoint_roundtrip_is_head_only(self):
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad], lr=1.0e-3
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "checkpoint.pt")
            save_student_checkpoint(
                path,
                model=self.model,
                optimizer=optimizer,
                step=3,
                config={"model": {"freeze_esm": True}},
            )
            payload = torch.load(path, map_location="cpu")
            self.assertFalse(any(key.startswith("encoder.") for key in payload["student_head_state_dict"]))
            restored = DeletionPreferenceStudent.tiny_for_tests(qmax=3, dropout=0.0)
            loaded = load_student_checkpoint(path, model=restored)
            self.assertEqual(loaded["step"], 3)
            self.assertFalse(loaded["metadata"]["save_encoder_weights"])

    def test_delepref_preflight_requires_cuda_and_valid_pair_inputs(self):
        config = train_delepref.load_config("configs/deletion_preference_student_pilot.yaml")
        report = train_delepref.preflight_delepref(config, device_request="cpu")
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("cuda_not_visible_in_current_runtime", report["blockers"])
        self.assertFalse(report["full_training_supported"])

        config["data"]["pair_index"] = "data/processed/preference_training/legacy_pairs/index.csv"
        invalid_report = train_delepref.preflight_delepref(config, device_request="cpu")
        self.assertIn("pair_index_path_lacks_pair_same_count_marker", invalid_report["blockers"])

        with tempfile.TemporaryDirectory() as tmp:
            fake = os.path.join(tmp, "validation_report.json")
            with open(fake, "w") as handle:
                json.dump({"nested": {"status": "PASS"}}, handle)
            fake_ok, fake_audit = train_delepref.validation_report_passes(fake)
        self.assertFalse(fake_ok)
        self.assertFalse(fake_audit["required_checks"]["top_level_status"])

    def test_protein_split_has_zero_overlap(self):
        pairs = []
        for protein_index in range(10):
            protein = "split_p{}".format(protein_index)
            a = candidate(protein, self.sequence, protein + "_a", [(0, 3)], self.beta)
            b = candidate(protein, self.sequence, protein + "_b", [(8, 9), (14, 15)], self.beta)
            pairs.append(pair(protein + "_pair", a, b))
        train, val = train_delepref.split_pairs_by_protein(pairs, 0.2)
        self.assertTrue(train)
        self.assertTrue(val)
        self.assertTrue({item.protein_id for item in train}.isdisjoint(item.protein_id for item in val))

    def test_real_pair_flat_schema_loader_uses_source_only_for_wildtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = os.path.join(tmp, "deletion_perturbations_shard_000000.csv")
            pair_path = os.path.join(tmp, "teacher_preference_pairs_shard_000000.csv")
            source_fields = ["protein_id", "perturbation_id", "plan_id", "wildtype_sequence"]
            with open(source_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=source_fields)
                writer.writeheader()
                writer.writerow(
                    {"protein_id": "p1", "perturbation_id": "a", "plan_id": "a", "wildtype_sequence": self.sequence}
                )
                writer.writerow(
                    {"protein_id": "p1", "perturbation_id": "b", "plan_id": "b", "wildtype_sequence": self.sequence}
                )
            pair_row = {
                "protein_id": "p1",
                "source_split": "train",
                "pair_id": "pair_1",
                "selection_category": "local_hard",
                "construction_policy": "strict_same_deleted_length_pair",
                "deleted_length": "4",
                "pair_budget_key": "deleted_length:4",
                "preference": "a_better",
                "preference_margin": "0.2",
                "preference_confidence": "0.8",
                "view_conflicting": "false",
                "pair_quality_bucket": "high",
                "scorer_training_weight": "1.0",
                "policy_positive_eligible": "true",
                "perturbation_id_a": "a",
                "plan_id_a": "a",
                "perturbation_id_b": "b",
                "plan_id_b": "b",
                "deleted_intervals_a": "[[0,3]]",
                "deleted_intervals_b": "[[8,9],[14,15]]",
                "deletion_fraction_a": "0.2",
                "deletion_fraction_b": "0.2",
                "generation_policy_a": "cold_start_a",
                "generation_policy_b": "cold_start_b",
                "comparison_basis_json": json.dumps({"group_scope": ["p1", 4]}),
            }
            with open(pair_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pair_row))
                writer.writeheader()
                writer.writerow(pair_row)
            loaded, stats = train_delepref.load_bounded_pairs(
                {"data": {"pair_format": "pair_same_count"}},
                [{"pair_path": pair_path, "source_candidate_path": source_path, "num_pairs": 1}],
                max_pairs=1,
            )
        self.assertEqual(len(loaded), 1)
        self.assertEqual(stats["strict_same_count_validated_loaded_rows"], 1)
        self.assertEqual(loaded[0].a.wildtype_sequence, self.sequence)
        self.assertEqual(loaded[0].a.deleted_length, loaded[0].b.deleted_length)
        self.assertAlmostEqual(loaded[0].loss_weight, 1.0)
        self.assertTrue(loaded[0].policy_positive_eligible)

    def test_generation_input_rejects_duplicate_protein_sequence_beta_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "inputs.csv")
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["protein_id", "wildtype_sequence", "beta"])
                writer.writeheader()
                writer.writerow({"protein_id": "p", "wildtype_sequence": self.sequence, "beta": "0.2"})
                writer.writerow({"protein_id": "p", "wildtype_sequence": self.sequence, "beta": "0.2"})
            with self.assertRaisesRegex(ValueError, "collide proposal IDs"):
                generate_delepref._load_inputs(path)

    def test_strict_delete_count_small_and_near_full(self):
        self.assertEqual(strict_delete_count(2, 0.01), 1)
        self.assertEqual(strict_delete_count(5, 1.0), 4)
        with self.assertRaises(ValueError):
            strict_delete_count(1, 0.5)


if __name__ == "__main__":
    unittest.main()
