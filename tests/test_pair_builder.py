import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("pair_builder_test", ROOT / "scripts/build_pair_same_count.py")
auditor = load_module("pair_auditor_test", ROOT / "scripts/audit_pair_same_count.py")


def candidate_payload(identifier, intervals, global_shift=0.003, retained_shift=0.006,
                      junction=0.95, budget_bin="descriptive_only", policy="toy"):
    wildtype = "ACDEFGHIKLMNPQRSTVWY"
    mutant = builder.apply_deletion(wildtype, intervals)
    deleted_length = sum(end - start + 1 for start, end in intervals)
    row = {
        "protein_id": "protein_1", "perturbation_id": identifier, "plan_id": identifier,
        "wildtype_length": str(len(wildtype)),
        "deleted_intervals": json.dumps([list(item) for item in intervals], separators=(",", ":")),
        "num_intervals": str(len(intervals)), "deleted_length": str(deleted_length),
        "deletion_fraction": "%.10f" % (deleted_length / float(len(wildtype))),
        "budget_bin": budget_bin, "generation_policy": policy,
        "coordinate_system": builder.COORDINATE_SYSTEM,
        "wildtype_sequence": wildtype, "mutant_sequence": mutant,
        "source_stage1_row_id": "", "source_start": "", "source_end": "", "source_score": "",
        "metadata_json": json.dumps({"source_split": "train"}),
    }
    impact = {
        "protein_id": "protein_1", "perturbation_id": identifier, "plan_id": identifier,
        "teacher_model": "frozen_esmc_600m", "teacher_scoring_mode": "fast_real_plm",
        "global_shift": str(global_shift), "retained_region_shift": str(retained_shift),
        "junction_plausibility": str(junction), "sequence_manifold_plausibility": "",
        "risk_overlap": "0", "risk_confidence": "0", "impact_confidence": "0.7",
        "scoring_status": "ok", "global_similarity": str(1.0 - 2.0 * global_shift),
        "retained_similarity": str(1.0 - 2.0 * retained_shift),
        "junction_mean": str(junction), "junction_min": str(junction),
        "teacher_backend": "esmc_official_safetensors", "max_memory_allocated_mb": "0",
        "view_status_json": "{}", "risk_summary_json": "{}", "metadata_json": "{}",
    }
    return row, impact


def prepared(identifier, intervals, views=(0.003, 0.006, 0.95), budget="ignored", quality=None):
    row, impact_row = candidate_payload(
        identifier, intervals, global_shift=views[0], retained_shift=views[1],
        junction=views[2], budget_bin=budget,
    )
    candidate = builder.prepare_candidate(row, builder.compact_impact(impact_row))
    if quality is not None:
        candidate["candidate_quality"] = quality
    return candidate


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class PairDelePrefTests(unittest.TestCase):
    def test_robust_scale_fallbacks_and_group_locality(self):
        center, scale, fallback = builder.robust_center_scale([1.0, 1.0, 1.0])
        self.assertEqual(center, 1.0)
        self.assertEqual(scale, 1.0)
        self.assertEqual(fallback, "constant_unit_scale")
        _, scale_two, fallback_two = builder.robust_center_scale([1.0, 3.0])
        self.assertGreater(scale_two, 0)
        self.assertEqual(fallback_two, "median_mad")
        group = [
            prepared("a", ((0, 1),), views=(0.002, 0.003, 0.97)),
            prepared("b", ((4, 5),), views=(0.004, 0.005, 0.93)),
            prepared("c", ((8, 9),), views=(0.006, 0.007, 0.89)),
        ]
        builder.annotate_group(group)
        original = {item["perturbation_id"]: item["utility"] for item in group}
        other_budget = [
            prepared("x", ((0, 2),), views=(0.001, 0.001, 0.99)),
            prepared("y", ((5, 7),), views=(0.020, 0.040, 0.50)),
        ]
        builder.annotate_group(other_budget)
        self.assertEqual(original, {item["perturbation_id"]: item["utility"] for item in group})

    def test_candidate_validation_coordinates_finiteness_and_gates(self):
        row, impact_row = candidate_payload("valid", ((2, 3),))
        candidate = builder.prepare_candidate(row, builder.compact_impact(impact_row))
        self.assertEqual(candidate["deleted_length"], 2)
        self.assertEqual(candidate["source_split"], "train")
        bad = dict(row)
        bad["deleted_length"] = "3"
        with self.assertRaisesRegex(ValueError, "deleted_length"):
            builder.prepare_candidate(bad, builder.compact_impact(impact_row))
        bad = dict(row)
        bad["deleted_intervals"] = "[[3,2]]"
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            builder.prepare_candidate(bad, builder.compact_impact(impact_row))
        low_impact = dict(impact_row)
        low_impact["global_shift"] = "0.03"
        low_impact["global_similarity"] = "0.94"
        with self.assertRaisesRegex(ValueError, "below_global_similarity_gate"):
            builder.prepare_candidate(row, builder.compact_impact(low_impact))
        nonfinite = dict(impact_row)
        nonfinite["junction_plausibility"] = "nan"
        with self.assertRaisesRegex(ValueError, "not finite"):
            builder.compact_impact(nonfinite)

    def test_selection_is_strict_same_count_budget_independent_and_shortage_honest(self):
        candidates = [
            prepared("a", ((0, 1),), views=(0.002, 0.003, 0.98), budget="bucket_a"),
            prepared("b", ((4, 5),), views=(0.004, 0.005, 0.94), budget="bucket_b"),
            prepared("c", ((8, 9),), views=(0.006, 0.007, 0.90), budget="bucket_c"),
            prepared("d", ((12, 12), (15, 15)), views=(0.008, 0.009, 0.86), budget="bucket_d"),
            prepared("different_count", ((0, 2),), views=(0.003, 0.004, 0.96), budget="bucket_a"),
        ]
        counters = builder.new_shard_counters()
        thresholds = {"local_high_mask_similarity": 0.5, "position_low_mask_similarity": 0.0}
        selected = builder.select_pairs_for_protein(
            "protein_1", candidates, thresholds, 0, builder.DEFAULT_SEED, 64, counters,
        )
        self.assertGreater(len(selected), 0)
        self.assertLess(len(selected), 64)
        self.assertTrue(any(row["budget_bin_a"] != row["budget_bin_b"] for row in selected))
        self.assertTrue(all(row["deleted_length"] == 2 for row in selected))
        self.assertTrue(all(row["pair_budget_key"] == "deleted_length:2" for row in selected))
        unordered = [tuple(sorted((row["perturbation_id_a"], row["perturbation_id_b"]))) for row in selected]
        self.assertEqual(len(unordered), len(set(unordered)))
        self.assertTrue(counters["quota_shortfall"])

    def test_conflict_is_diagnostic_and_both_low_is_excluded(self):
        group = [
            prepared("a", ((0, 1),), views=(0.002, 0.010, 0.80)),
            prepared("b", ((4, 5),), views=(0.010, 0.002, 0.98)),
            prepared("c", ((8, 9),), views=(0.006, 0.006, 0.90)),
        ]
        stats = builder.annotate_group(group)
        thresholds = {"local_high_mask_similarity": 0.5, "position_low_mask_similarity": 0.0}
        pair, excluded = builder.make_pair("protein_1", 2, 3, group[0], group[1], stats, thresholds, 0)
        self.assertIsNone(excluded)
        self.assertEqual(pair["view_conflicting"], "true")
        self.assertEqual(pair["pair_quality_bucket"], "diagnostic_view_conflict")
        self.assertAlmostEqual(float(pair["scorer_training_weight"]), 0.05)
        self.assertEqual(pair["policy_positive_eligible"], "false")
        low_group = [
            prepared("low_a", ((0, 1),), views=(0.015, 0.030, 0.90), quality="acceptable"),
            prepared("low_b", ((4, 5),), views=(0.014, 0.028, 0.91), quality="acceptable"),
        ]
        low_stats = builder.annotate_group(low_group)
        pair, excluded = builder.make_pair("protein_1", 2, 2, low_group[0], low_group[1], low_stats, thresholds, 0)
        self.assertIsNone(pair)
        self.assertEqual(excluded, "both_low_absolute_quality")

    def test_calibration_thresholds_are_quantile_derived(self):
        values = [0.0, 0.0, 0.1, 0.2, 0.4, 0.8]
        self.assertAlmostEqual(builder.percentile(values, 0.25), 0.025)
        positives = [value for value in values if value > 0]
        local = builder.percentile(positives, 0.75)
        self.assertAlmostEqual(local, 0.5)
        category = builder.category_for_pair(False, False, 0.8, {
            "local_high_mask_similarity": local,
            "position_low_mask_similarity": builder.percentile(values, 0.25),
        })
        self.assertEqual(category, "local_hard")

    def test_old_pair_index_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preference_pair_shard_index.csv"
            path.write_text("shard_index,output_csv\n0,x.csv\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden"):
                builder.parse_index(path, "impact")

    def test_toy_shard_build_resume_and_independent_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidates.csv"
            impact_path = root / "impacts.csv"
            output_dir = root / "pair"
            rows, impacts = [], []
            specifications = [
                ("a", ((0, 1),), 0.002, 0.003, 0.98, "wrong_budget_a", "policy_a"),
                ("b", ((4, 5),), 0.004, 0.005, 0.94, "wrong_budget_b", "policy_b"),
                ("c", ((8, 9),), 0.006, 0.007, 0.90, "wrong_budget_c", "policy_c"),
                ("d", ((12, 12), (15, 15)), 0.008, 0.009, 0.86, "wrong_budget_d", "policy_d"),
                ("e", ((17, 18),), 0.003, 0.004, 0.96, "wrong_budget_e", "policy_e"),
            ]
            for identifier, intervals, global_shift, retained_shift, junction, budget, policy in specifications:
                row, impact = candidate_payload(
                    identifier, intervals, global_shift=global_shift,
                    retained_shift=retained_shift, junction=junction,
                    budget_bin=budget, policy=policy,
                )
                rows.append(row)
                impacts.append(impact)
            write_csv(candidate_path, list(rows[0]), rows)
            write_csv(impact_path, list(impacts[0]), impacts)
            candidate_info = {"path": candidate_path, "expected_rows": len(rows)}
            impact_info = {"path": impact_path, "expected_rows": len(impacts)}
            thresholds = {"local_high_mask_similarity": 0.5, "position_low_mask_similarity": 0.0}
            report = builder.process_shard(
                0, candidate_info, impact_info, output_dir, thresholds,
                builder.DEFAULT_SEED, 64, False, False,
            )
            self.assertEqual(report["status"], "written")
            self.assertGreater(report["num_pairs"], 0)
            self.assertEqual(report["num_shortage_proteins"], 1)
            resumed = builder.process_shard(
                0, candidate_info, impact_info, output_dir, thresholds,
                builder.DEFAULT_SEED, 64, True, False,
            )
            self.assertEqual(resumed["status"], "resumed_validated")
            validation = auditor.audit_shard(
                0, candidate_info, impact_info, builder.output_path(output_dir, 0), thresholds,
            )
            self.assertEqual(validation["status"], "PASS", validation["failures"])
            self.assertEqual(validation["same_count_pairs"], validation["pairs"])
            self.assertEqual(validation["unequal_count_pairs"], 0)
            self.assertEqual(validation["matched_source_candidate_references"], 2 * validation["pairs"])
            self.assertEqual(validation["matched_source_impact_references"], 2 * validation["pairs"])
            self.assertEqual(validation["shortage_proteins"], 1)


if __name__ == "__main__":
    unittest.main()
