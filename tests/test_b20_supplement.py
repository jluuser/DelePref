import copy
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


b20 = load_module("delepref_b20_builder_test", "scripts/build_b20_supplement.py")
scoring = load_module("delepref_b20_scoring_test", "scripts/score_b20_esmc600m.py")


class B20SupplementTests(unittest.TestCase):
    def test_exactly_eight_deterministic_strict_b20_plans(self):
        for length in (16, 80, 127, 511, 800):
            sequence = ("ACDEFGHIKLMNPQRSTVWY" * 50)[:length]
            deleted, plans = b20.plans_for(sequence)
            self.assertEqual(deleted, min(length - 1, round(0.20 * length)))
            self.assertEqual(plans, b20.plans_for(sequence)[1])
            self.assertEqual(len(plans), 8)
            self.assertEqual(len({intervals for _policy, intervals in plans}), 8)
            self.assertEqual(
                [policy for policy, _intervals in plans],
                [
                    "terminal_n", "terminal_c", "internal_20", "internal_70",
                    "boundary_1", "boundary_2", "multi_2", "multi_3",
                ],
            )
            for _policy, intervals in plans:
                self.assertLessEqual(len(intervals), 3)
                self.assertEqual(sum(end - start + 1 for start, end in intervals), deleted)
                self.assertTrue(all(0 <= start <= end < length for start, end in intervals))
            self.assertEqual(len(plans[-2][1]), 2)
            self.assertEqual(len(plans[-1][1]), 3)

    def candidate(self):
        sequence = "ACDEFGHIKLMNPQRSTVWY" * 4
        source = {
            "protein_id": "protein_1",
            "wildtype_sequence": sequence,
            "wildtype_length": str(len(sequence)),
            "metadata_json": json.dumps({"source_split": "dev"}),
            "_source_path": "source.csv",
        }
        return b20.candidate_rows(source, 3)[0]

    def test_raw_candidate_hashes_and_base_split_inheritance(self):
        row = self.candidate()
        metadata = json.loads(row["metadata_json"])
        self.assertEqual(metadata["base_source_split"], "dev")
        self.assertFalse(metadata["teacher_scored"])
        self.assertEqual(row["wildtype_sha256"], b20.sha256_text(row["wildtype_sequence"]))
        self.assertEqual(row["mutant_sha256"], b20.sha256_text(row["mutant_sequence"]))
        self.assertEqual(row["interval_plan_sha256"], b20.sha256_text(row["deleted_intervals"]))

    def test_strict_cache_identity_changes_for_every_required_component(self):
        row = self.candidate()
        baseline = scoring.strict_identity(row, "model_hash", "config_hash")
        self.assertEqual(
            set(baseline),
            {
                "protein_id", "wildtype_sequence_sha256", "interval_plan",
                "interval_plan_sha256", "mutant_sequence_sha256",
                "teacher_model_sha256", "teacher_scoring_config_sha256",
                "cache_key_sha256",
            },
        )
        for field, value in (("teacher_model", "other_model"), ("scoring_config", "other_config")):
            model = value if field == "teacher_model" else "model_hash"
            config = value if field == "scoring_config" else "config_hash"
            self.assertNotEqual(baseline["cache_key_sha256"], scoring.strict_identity(row, model, config)["cache_key_sha256"])
        changed_protein = dict(row, protein_id="protein_2")
        self.assertNotEqual(baseline["cache_key_sha256"], scoring.strict_identity(changed_protein, "model_hash", "config_hash")["cache_key_sha256"])
        changed_plan = copy.deepcopy(row)
        changed_plan["deleted_intervals"] = "[[1,16]]"
        changed_plan["mutant_sequence"] = scoring.scorer.apply_deletion(
            changed_plan["wildtype_sequence"], [[1, 16]]
        )
        changed_plan["deleted_length"] = "16"
        changed_plan["num_intervals"] = "1"
        self.assertNotEqual(baseline["cache_key_sha256"], scoring.strict_identity(changed_plan, "model_hash", "config_hash")["cache_key_sha256"])

    def test_invalid_mutant_never_gets_a_cache_identity(self):
        row = dict(self.candidate())
        row["mutant_sequence"] += "A"
        with self.assertRaisesRegex(ValueError, "cache identity"):
            scoring.strict_identity(row, "model_hash", "config_hash")


if __name__ == "__main__":
    unittest.main()
