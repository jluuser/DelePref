import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from delepref.proposal_validation import (
    BEST_OF_K_METRIC,
    PRIMARY_METRIC,
    checkpoint_candidate,
    choose_checkpoint,
    evaluate_proposal_records,
    read_cold_start_references,
    read_frozen_proteins,
    sequence_sha256,
)
from delepref.student_model import DeletionPreferenceStudent
from scripts import train_final as trainer
from scripts.train_final import generate_proposal_candidate_rows


REPO = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = REPO / "data/processed/preference_training/delepref/proposal_validation/v1"


class ProposalValidationTests(unittest.TestCase):
    def records(self):
        records = []
        for protein_index, protein_id in enumerate(("p0", "p1")):
            for beta in (0.10, 0.20):
                deleted = round(beta * 20)
                for seed in (13, 29):
                    for candidate_index, start in enumerate((0, 6, 12)):
                        intervals = ((start, start + deleted - 1),)
                        teacher_value = 0.70 + 0.01 * candidate_index + 0.001 * protein_index
                        # The student selects candidate 1 while candidate 2 is oracle.
                        student_value = (0.0, 2.0, 1.0)[candidate_index]
                        records.append(
                            {
                                "protein_id": protein_id,
                                "sequence_sha256": "x" * 64,
                                "wildtype_length": 20,
                                "beta": beta,
                                "seed": seed,
                                "candidate_id": "{}-{}-{}-{}".format(
                                    protein_id, beta, seed, candidate_index
                                ),
                                "deleted_intervals": intervals,
                                "student_scorer_score": student_value,
                                "teacher_scoring_status": "ok",
                                "global_similarity": teacher_value,
                                "retained_similarity": teacher_value,
                                "junction_plausibility": teacher_value,
                            }
                        )
        return records

    def test_metrics_are_equal_weighted_by_protein_budget_seed(self):
        report = evaluate_proposal_records(
            self.records(),
            protein_ids=("p0", "p1"),
            budgets=(0.10, 0.20),
            seeds=(13, 29),
            requested_k=3,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["expected_conditions"], 8)
        self.assertEqual(report["evaluated_conditions"], 8)
        self.assertEqual(report["metrics"]["effective_k_mean"], 3.0)
        self.assertAlmostEqual(report["metrics"][PRIMARY_METRIC], 0.7105)
        self.assertAlmostEqual(report["metrics"][BEST_OF_K_METRIC], 0.7205)
        self.assertAlmostEqual(report["metrics"]["teacher_rescored_oracle_gap"], 0.01)
        self.assertGreater(report["metrics"]["mask_diversity_mean"], 0.0)
        self.assertGreater(report["metrics"]["position_diversity_mean"], 0.0)

    def test_missing_or_duplicate_candidates_fail_legality_gates(self):
        records = self.records()
        records.pop()
        report = evaluate_proposal_records(
            records,
            protein_ids=("p0", "p1"),
            budgets=(0.10, 0.20),
            seeds=(13, 29),
            requested_k=3,
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["legality_gates"]["exact_k_per_condition"])

    def test_checkpoint_selection_obeys_frozen_tie_rule(self):
        def candidate(step, selected, best_of_k, accuracy, legal=True):
            report = {
                "status": "PASS" if legal else "FAIL",
                "legality_gates": {"legal": legal},
                "metrics": {PRIMARY_METRIC: selected, BEST_OF_K_METRIC: best_of_k},
            }
            return checkpoint_candidate(
                checkpoint_path="checkpoint_{}.pt".format(step),
                step=step,
                proposal_report=report,
                strict_pair_accuracy=accuracy,
            )

        candidates = [
            candidate(1, 0.95, 0.99, 0.99, legal=False),
            candidate(2, 0.80, 0.90, 0.75),
            candidate(3, 0.80 + 5.0e-7, 0.91, 0.70),
            candidate(4, 0.80 + 4.0e-7, 0.91, 0.80),
        ]
        selected = choose_checkpoint(candidates, tie_tolerance=1.0e-6)
        self.assertEqual(selected["step"], 4)
        self.assertFalse(selected["confirmation_metrics_used"])
        self.assertEqual(selected["forbidden_metrics_used"], [])

    @unittest.skipUnless(
        (MANIFEST_ROOT / "proteins.csv").is_file()
        and (MANIFEST_ROOT / "cold_start_reference.csv").is_file(),
        "requires generated DelePref proposal-validation manifests",
    )
    def test_actual_frozen_manifest_is_dev_only_and_complete(self):
        proteins = read_frozen_proteins(MANIFEST_ROOT / "proteins.csv")
        references = read_cold_start_references(MANIFEST_ROOT / "cold_start_reference.csv")
        self.assertEqual(len(proteins), 256)
        self.assertEqual(len(references), 512)
        self.assertEqual({row["split"] for row in proteins}, {"dev"})

    def test_student_generation_contract_produces_k_for_every_budget_seed(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        model = DeletionPreferenceStudent.tiny_for_tests(qmax=3, dropout=0.0)
        config = {
            "proposal_validation": {"budgets": [0.10, 0.20], "seeds": [13, 29], "k": 8},
            "policy": {"qmax": 3, "min_span_len": 1, "max_span_len_fraction": 0.5},
            "decoder": {"perturbation_scale": 1.0, "max_attempts_multiplier": 16},
            "batching": {"encoder_microbatch_size": 1},
        }
        rows = generate_proposal_candidate_rows(
            model,
            [{
                "protein_id": "tiny",
                "wildtype_sequence": sequence,
                "sequence_sha256": sequence_sha256(sequence),
            }],
            config,
        )
        self.assertEqual(len(rows), 32)
        self.assertEqual(len({row["perturbation_id"] for row in rows}), 32)
        counts = {}
        for row in rows:
            counts[(float(row["beta"]), int(row["seed"]))] = counts.get(
                (float(row["beta"]), int(row["seed"])), 0
            ) + 1
        self.assertEqual(set(counts.values()), {8})

    def test_real_teacher_cache_requires_exact_candidate_identity(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        model = DeletionPreferenceStudent.tiny_for_tests(qmax=3, dropout=0.0)
        generation_config = {
            "proposal_validation": {"budgets": [0.20], "seeds": [13], "k": 1},
            "policy": {"qmax": 3, "min_span_len": 1, "max_span_len_fraction": 0.5},
            "decoder": {"perturbation_scale": 1.0, "max_attempts_multiplier": 16},
            "batching": {"encoder_microbatch_size": 1},
        }
        rows = generate_proposal_candidate_rows(
            model,
            [{
                "protein_id": "cache_test",
                "wildtype_sequence": sequence,
                "sequence_sha256": sequence_sha256(sequence),
            }],
            generation_config,
        )
        config = {
            "proposal_validation": {
                "teacher_model_snapshot": str(REPO / "model_cache_unused_by_mock"),
                "teacher_mutant_batch_size": 1,
            }
        }
        scoring_config = {
            "teacher_model_sha256": "a" * 64,
            "sha256": "b" * 64,
        }

        def fake_embeddings(_backend, sequences, batch_size=None):
            return [torch.zeros(len(sequence_value), 4) for sequence_value in sequences]

        def fake_impact(row, _args, _backend, _wt, _mutant):
            payload = {column: "" for column in trainer.teacher_driver.IMPACT_COLUMNS}
            payload.update(
                {
                    "protein_id": row["protein_id"],
                    "perturbation_id": row["perturbation_id"],
                    "plan_id": row["plan_id"],
                    "teacher_model": "fake_real_esmc",
                    "teacher_scoring_mode": trainer.teacher_driver.scorer.REAL_PLM_TEACHER_MODE,
                    "global_shift": "0.01",
                    "retained_region_shift": "0.01",
                    "junction_plausibility": "0.90",
                    "risk_overlap": "0",
                    "risk_confidence": "0",
                    "impact_confidence": "0.7",
                    "scoring_status": "ok",
                    "global_similarity": "0.98",
                    "retained_similarity": "0.97",
                    "junction_mean": "0.90",
                    "junction_min": "0.90",
                    "teacher_backend": trainer.teacher_driver.scorer.ESMC_OFFICIAL_SAFETENSORS_BACKEND,
                    "metadata_json": "{}",
                }
            )
            return payload

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teacher_impacts.csv"
            with mock.patch.object(
                trainer.teacher_driver.scorer, "load_real_plm_teacher_backend", return_value={"model": object()}
            ), mock.patch.object(
                trainer.teacher_driver.scorer, "residue_embeddings_for_sequences", side_effect=fake_embeddings
            ), mock.patch.object(
                trainer.teacher_driver.scorer, "real_plm_teacher_impact_from_embeddings", side_effect=fake_impact
            ):
                impacts, status = trainer.score_proposal_candidates_real_teacher(
                    rows, config, scoring_config, path
                )
            self.assertEqual(status, "written_real_esmc600m")
            self.assertEqual(len(impacts), 1)
            with mock.patch.object(
                trainer.teacher_driver.scorer,
                "load_real_plm_teacher_backend",
                side_effect=AssertionError("strict cache should avoid backend load"),
            ):
                cached, status = trainer.score_proposal_candidates_real_teacher(
                    rows, config, scoring_config, path
                )
            self.assertEqual(status, "resumed_strict_cache_match")
            self.assertEqual(len(cached), 1)
            changed = [dict(rows[0])]
            changed[0]["mutant_sequence"] = changed[0]["mutant_sequence"] + "A"
            with self.assertRaises(ValueError):
                trainer.score_proposal_candidates_real_teacher(
                    changed, config, scoring_config, path
                )


if __name__ == "__main__":
    unittest.main()
