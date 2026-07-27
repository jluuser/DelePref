import importlib.util
import csv
import pathlib
import sys
import tempfile
import types
import unittest


SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_formal_teacher_scoring_shards.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_formal_teacher_scoring_shards", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FormalTeacherScoringRunnerTests(unittest.TestCase):
    def test_parse_args_supports_impact_only(self):
        runner = load_runner_module()
        old_argv = sys.argv
        try:
            sys.argv = [
                "run_formal_teacher_scoring_shards.py",
                "--deletion_query_shard_index",
                "index.csv",
                "--output_dir",
                "out",
                "--teacher_model_name_or_path",
                "model",
                "--impact_only",
            ]
            args = runner.parse_args()
        finally:
            sys.argv = old_argv

        self.assertTrue(args.impact_only)

    def test_impact_only_resume_validates_only_impact_rows(self):
        runner = load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            impact = root / "teacher_impacts" / "teacher_impact_scores_shard_000000.csv"
            impact.parent.mkdir(parents=True)
            impact.write_text("protein_id\np1\np2\n")
            pair = root / "preference_pairs" / "teacher_preference_pairs_shard_000000.csv"

            self.assertTrue(runner.valid_existing_outputs(str(impact), str(pair), 2, 64, impact_only=True))
            self.assertFalse(runner.valid_existing_outputs(str(impact), str(pair), 3, 64, impact_only=True))

    def test_progress_writes_impact_index_summary_and_log(self):
        runner = load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            args = types.SimpleNamespace(
                deletion_query_shard_index="queries.csv",
                output_dir=tmp,
                teacher_model_name_or_path="model",
                teacher_backend="esmc_official_safetensors",
                teacher_scoring_mode="fast_real_plm",
                teacher_batch_size=8,
                preference_construction_mode="not_run_impact_only",
                preference_tie_threshold=0.05,
                max_pairs_per_protein=64,
                device_ids="0",
                max_workers=1,
                impact_only=True,
            )
            impact_records = [
                {
                    "shard_id": "deletion_perturbations_shard_000000",
                    "shard_index": 0,
                    "source_deletion_perturbations_csv": "shard.csv",
                    "output_csv": "teacher_impacts/teacher_impact_scores_shard_000000.csv",
                    "num_rows": 2,
                    "expected_rows": 2,
                    "status": "written",
                    "runtime_seconds": "1.000",
                    "metadata_json": "{}",
                }
            ]

            runner.write_progress(args, impact_records, [], [], {"0": {"impact_seconds": 1.0}})

            self.assertTrue((pathlib.Path(tmp) / "indexes" / "teacher_impact_shard_index.csv").exists())
            self.assertTrue((pathlib.Path(tmp) / "run_summary.json").exists())
            self.assertTrue((pathlib.Path(tmp) / "logs" / "progress.log").exists())
            self.assertFalse((pathlib.Path(tmp) / "preference_pair_shard_index.csv").exists())


if __name__ == "__main__":
    unittest.main()
