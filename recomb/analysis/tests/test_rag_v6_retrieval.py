"""Technical fixtures only; never search UniRef or read candidate outcome assets."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rag_v6_retrieval as retrieval


def row(protein="WT1", sequence="ACDE"):
    return {"protein_id": protein, "wt_sequence": sequence,
            "wt_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest()}


class RetrievalTests(unittest.TestCase):
    def test_query_dedup_sorted_safe_and_label_free(self):
        a, b = row("WT2"), row("WT1", "CDEF")
        a["outcome_npz"] = "/must/not/be/opened.npz"
        a["legacy_metrics"] = {"label": "must_not_be_copied"}
        result = retrieval.queries_from_rows([a, b, a], {"WT1": "validation", "WT2": "train"})
        self.assertEqual([q["query_id"] for q in result], ["recombq_0000", "recombq_0001"])
        self.assertEqual(result[0]["split"], "validation")
        self.assertEqual(result[0]["protein_id"], "WT1")
        self.assertEqual(set(result[1]), {"protein_id", "wt_sequence", "wt_sequence_sha256", "split", "query_id"})

    def test_query_rejects_bad_hash_split_and_unsafe_id(self):
        for item, assignments in [(dict(row(), wt_sequence_sha256="0" * 64), {"WT1": "train"}),
                                  (row(), {"WT1": "unassigned"}), (row("../../escape"), {"../../escape": "train"})]:
            with self.subTest(item=item), self.assertRaises(ValueError):
                retrieval.queries_from_rows([item], assignments)

    def test_query_rejects_duplicate_sequence_or_conflicting_wt(self):
        with self.assertRaisesRegex(ValueError, "duplicate WT"):
            retrieval.queries_from_rows([row("a"), row("b")], {"a": "train", "b": "validation"})
        with self.assertRaisesRegex(ValueError, "inconsistent WT"):
            retrieval.queries_from_rows([row(), row(sequence="CDEF")], {"WT1": "train"})

    def fixture(self, root):
        base = root / retrieval.BASE
        (base / "prepared").mkdir(parents=True)
        (root / "configs").mkdir()
        query = base / "prepared/queries.fasta"
        query.write_text(">recombq_0000\nACDE\n")
        source = root / "source.py"
        source.write_text("# fixed source\n")
        protocol = root / "original_protocol.json"
        protocol.write_text("{}\n")
        target = root / "target"
        for suffix in retrieval.DB_SUFFIXES:
            Path(str(target) + suffix).write_bytes(b"synthetic database\n")
        executable = root / "fake_mmseqs"
        executable.write_text(f"#!{sys.executable}\n" +
                              "import os, pathlib, sys\n"
                              "command=sys.argv[1]\n"
                              "if command=='version': print('synthetic-v1'); sys.exit()\n"
                              "assert os.environ['CUDA_VISIBLE_DEVICES']==''\n"
                              "assert os.environ['OMP_NUM_THREADS']=='32'\n"
                              "if command=='createdb': pathlib.Path('query_db').write_bytes(b'query')\n"
                              "if command=='search': pathlib.Path('result_db').write_bytes(b'result')\n"
                              "if command=='convertalis': pathlib.Path('hits.tsv').write_text(os.environ.get('FAKE_HITS','synthetic hit\\n'))\n")
        executable.chmod(0o755)
        config = {"contract": copy.deepcopy(retrieval.CONTRACT), "source_hashes": {"source.py": retrieval.sha(source)},
                  "input_hashes": {}, "query_hashes": {str(query.relative_to(root)): retrieval.sha(query)},
                  "tool": {"path": str(executable), "sha256": retrieval.sha(executable), "version": "synthetic-v1"},
                  "target": str(target), "database": retrieval.database_metadata(target),
                  "commands": retrieval.commands(str(executable), str(target)),
                  "database_provenance": {"source_protocol": str(protocol), "source_protocol_sha256": retrieval.sha(protocol)}}
        retrieval.write_json(root / retrieval.FREEZE, config)
        digest = retrieval.sha(root / retrieval.FREEZE)
        retrieval.write_json(base / "submission_receipt.json", {"freeze_sha256": digest})
        return config, digest, base

    def test_database_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self.fixture(Path(directory))
            Path(config["target"] + ".source").write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "metadata changed"):
                retrieval.verify_database(config)

    def test_freeze_query_source_tampering_rejected(self):
        for name in (retrieval.FREEZE, retrieval.BASE + "/prepared/queries.fasta", "source.py"):
            with self.subTest(path=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, digest, _ = self.fixture(root)
                with (root / name).open("a") as handle:
                    handle.write("tampering")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    retrieval.load_verified(root, digest)

    def test_output_no_overwrite_and_claim_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claim.json"
            retrieval.write_json(path, {"first": True}, atomic=True)
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                retrieval.write_json(path, {"first": False})
            self.assertEqual(json.loads(path.read_text()), {"first": True})

    def test_exact_protocol_args(self):
        commands = retrieval.commands("mmseqs", "target")
        self.assertEqual(commands[0], ["mmseqs", "createdb", "queries.fasta", "query_db"])
        search = commands[1]
        for key, value in (("-c", "0.60"), ("--threads", "32"), ("--split-memory-limit", "160G"), ("--max-seqs", "512"),
                           ("-a", "1"), ("--alignment-mode", "3"), ("--cov-mode", "2"), ("--split-mode", "0")):
            self.assertEqual(search[search.index(key) + 1], value)
        self.assertEqual(commands[2][-1], retrieval.FORMAT)
        self.assertNotIn("--gpu", search)

    def test_refuse_non_slurm_and_gpu(self):
        for environment in ({}, {"SLURM_JOB_ID": "1", "SLURM_CPUS_PER_TASK": "32", "CUDA_VISIBLE_DEVICES": "0"},
                            {"SLURM_JOB_ID": "1", "SLURM_CPUS_PER_TASK": "32", "CUDA_VISIBLE_DEVICES": "", "SLURM_JOB_GPUS": "1"},
                            {"SLURM_JOB_ID": "1", "SLURM_CPUS_PER_TASK": "32", "CUDA_VISIBLE_DEVICES": "", "SLURM_JOB_GPUS": "0"}):
            with patch.dict(os.environ, environment, clear=True), self.assertRaises(ValueError):
                retrieval.execute(Path("does-not-exist"), "0" * 64)

    def test_submission_unknown_never_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / retrieval.BASE).mkdir(parents=True)
            with patch.object(retrieval, "load_verified"), patch.object(subprocess, "run", side_effect=OSError("uncertain sbatch")) as runner:
                with self.assertRaises(OSError):
                    retrieval.submit(root, "0" * 64)
                with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                    retrieval.submit(root, "0" * 64)
                self.assertEqual(runner.call_count, 1)

    def test_submission_ignores_inherited_sbatch_resource_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / retrieval.BASE).mkdir(parents=True)
            response = subprocess.CompletedProcess([], 0, "321\n", "")
            with patch.dict(os.environ, {"SBATCH_GRES": "gpu:1", "SBATCH_MEM_PER_NODE": "4G"}), \
                 patch.object(retrieval, "load_verified"), patch.object(subprocess, "run", return_value=response) as runner:
                self.assertEqual(retrieval.submit(root, "a" * 64)["job_id"], "321")
            self.assertFalse(any(k.startswith("SBATCH_") for k in runner.call_args.kwargs["env"]))
            self.assertEqual(runner.call_args.args[0][0], "/opt/gridview/slurm/bin/sbatch")

    @unittest.skipIf(os.name == "nt", "POSIX executable fixture runs on cu9 CPU")
    def test_fake_executable_success_zero_hits_and_no_rerun(self):
        for hits in ("synthetic hit\n", ""):
            with self.subTest(hits=bool(hits)), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, digest, base = self.fixture(root)
                before = copy.deepcopy(config["database"])
                env = {"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "32", "CUDA_VISIBLE_DEVICES": "", "FAKE_HITS": hits}
                with patch.dict(os.environ, env, clear=True):
                    self.assertEqual(retrieval.execute(root, digest), "complete" if hits else "complete_zero_hits")
                    with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                        retrieval.execute(root, digest)
                completion = json.loads((base / "completion.json").read_text())
                self.assertEqual(len(completion["database_sha256"]), 7)
                self.assertEqual(completion["hits_sha256"], hashlib.sha256(hits.encode()).hexdigest())
                self.assertEqual(retrieval.database_metadata(config["target"]), before)
                self.assertEqual(len(completion["commands"]), 3)

    def test_failed_command_records_failure_without_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, digest, base = self.fixture(root)
            env = {"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "32", "CUDA_VISIBLE_DEVICES": ""}
            with patch.dict(os.environ, env, clear=True), patch.object(retrieval, "load_verified", return_value=config), \
                 patch.object(subprocess, "run", side_effect=subprocess.CalledProcessError(7, "synthetic")):
                with self.assertRaises(subprocess.CalledProcessError):
                    retrieval.execute(root, digest)
            self.assertFalse((base / "completion.json").exists())
            self.assertEqual(json.loads((base / "failure.json").read_text())["status"], "failed_no_retry")
            self.assertTrue((base / "search/0_createdb.stdout.log").exists())

    def test_existing_search_directory_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, digest, base = self.fixture(root)
            (base / "search").mkdir()
            (base / "search/sentinel").write_text("keep")
            env = {"SLURM_JOB_ID": "123", "SLURM_CPUS_PER_TASK": "32", "CUDA_VISIBLE_DEVICES": ""}
            with patch.dict(os.environ, env, clear=True), patch.object(retrieval, "load_verified", return_value=config), patch.object(subprocess, "run") as runner:
                with self.assertRaises(FileExistsError):
                    retrieval.execute(root, digest)
                runner.assert_not_called()
            self.assertEqual((base / "search/sentinel").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
