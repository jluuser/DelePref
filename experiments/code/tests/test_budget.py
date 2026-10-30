import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from budget_ledger import BudgetExceeded, BudgetLedger


def request(task="t", **kwargs):
    result = dict(task_id=task, sequence_sha=hashlib.sha256(task.encode()).hexdigest(),
                  wt_variant="variant", predictor_weights_sha="b" * 64,
                  profile_sha="c" * 64, seed=17, category="development")
    result.update(kwargs)
    return result


def reserve_worker(args):
    protocol, ledger, task = args
    try:
        return BudgetLedger(protocol, ledger).reserve(**request(task))["new_reservation"]
    except BudgetExceeded:
        return "capped"


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = self.root / "protocol.json"
        self.protocol.write_text(json.dumps({"compute": {"structure_call_limit": 4,
            "structure_categories": {"development": 3, "confirmation": 3, "retry": 1},
            "structure_retries_max": 1}}))
        self.ledger = self.root / "ledger.jsonl"
        self.b = BudgetLedger(self.protocol, self.ledger)

    def test_parallel_duplicate_and_category_cap(self):
        tasks = ["a"] * 8 + [f"b{i}" for i in range(12)]
        with concurrent.futures.ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(reserve_worker, [(str(self.protocol), str(self.ledger), t) for t in tasks]))
        self.assertEqual(results.count(True), 3)
        self.assertEqual(self.b.snapshot()["consumed"], 3)
        with self.assertRaises(BudgetExceeded):
            self.b.reserve(**request("next"))

    def test_global_cap_includes_unstarted_and_failure_retry(self):
        for t in ("a", "b", "c"):
            self.b.reserve(**request(t))
        self.b.start("a")
        self.b.finish("a", succeeded=False, seconds=3, max_memory=20)
        self.b.reserve(**request("a", attempt=1))
        self.assertEqual(self.b.snapshot()["consumed"], 4)
        with self.assertRaises(BudgetExceeded):
            self.b.reserve(**request("other", category="confirmation"))
        with self.assertRaises(ValueError):
            self.b.reserve(**request("a", attempt=2))
        with self.assertRaises(ValueError):
            self.b.start("a")

    def test_parallel_global_cap(self):
        config = json.loads(self.protocol.read_text())
        config["compute"]["structure_categories"]["development"] = 20
        self.protocol.write_text(json.dumps(config))
        self.b = BudgetLedger(self.protocol, self.ledger)
        with concurrent.futures.ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(reserve_worker, [(str(self.protocol), str(self.ledger), f"t{i}") for i in range(20)]))
        self.assertEqual(results.count(True), 4)
        self.assertEqual(self.b.snapshot()["consumed"], 4)

    def test_cache_identity_includes_weights_profile_seed_sample(self):
        config = json.loads(self.protocol.read_text())
        config["compute"]["structure_call_limit"] = 10
        config["compute"]["structure_categories"]["development"] = 10
        self.protocol.write_text(json.dumps(config))
        self.b = BudgetLedger(self.protocol, self.ledger)
        self.b.reserve(**request())
        self.b.start("t")
        output = self.root / "out.pdb"
        output.write_bytes(b"same path is insufficient")
        self.b.finish("t", succeeded=True, seconds=1, max_memory=1, output_path=output)
        for i, change in enumerate(({"predictor_weights_sha": "d" * 64}, {"profile_sha": "e" * 64}, {"seed": 18}, {"sample": 1})):
            result = self.b.reserve(**request(f"changed{i}", sequence_sha=request()["sequence_sha"], **change))
            self.assertTrue(result["new_reservation"])
            self.assertFalse(result["cache_hit"])
        self.assertEqual(self.b.snapshot()["consumed"], 5)

    def test_verified_cache_and_corruption(self):
        self.b.reserve(**request())
        self.b.start("t")
        output = self.root / "out.pdb"
        output.write_bytes(b"verified structure")
        self.b.finish("t", succeeded=True, seconds=2, max_memory=1, output_path=output)
        hit = self.b.reserve(**request("copy", sequence_sha=request()["sequence_sha"]))
        self.assertTrue(hit["cache_hit"])
        self.assertFalse(hit["new_reservation"])
        self.assertEqual(self.b.snapshot()["consumed"], 1)
        output.write_bytes(b"corrupt")
        with self.assertRaises(ValueError):
            self.b.reserve(**request("copy", sequence_sha=request()["sequence_sha"]))
        miss = self.b.reserve(**request("copy2", sequence_sha=request()["sequence_sha"]))
        self.assertFalse(miss["cache_hit"])
        self.assertEqual(self.b.snapshot()["consumed"], 2)

    def test_identity_and_retry_preconditions(self):
        self.assertTrue(self.b.reserve(**request())["new_reservation"])
        self.assertFalse(self.b.reserve(**request())["new_reservation"])
        pending = self.b.reserve(**request("alias", sequence_sha=request()["sequence_sha"]))
        self.assertFalse(pending["new_reservation"])
        self.assertEqual(pending["task_id"], "t")
        with self.assertRaises(ValueError):
            self.b.reserve(**request(seed=18))
        with self.assertRaises(ValueError):
            self.b.reserve(**request(attempt=1))
        self.b.start("t")
        with self.assertRaises(ValueError):
            self.b.start("t")

    def test_original_successful_task_rechecks_corrupt_and_missing_output(self):
        self.b.reserve(**request())
        self.b.start("t")
        output = self.root / "original.pdb"
        output.write_bytes(b"original structure")
        self.b.finish("t", succeeded=True, seconds=1, max_memory=1, output_path=output)
        self.assertFalse(self.b.reserve(**request())["new_reservation"])
        output.write_bytes(b"changed structure")
        with self.assertRaises(ValueError):
            self.b.reserve(**request())
        output.unlink()
        with self.assertRaises(ValueError):
            self.b.reserve(**request())
        self.assertEqual(self.b.snapshot()["consumed"], 1)

    def test_failed_identity_cannot_reset_retry_by_task_alias(self):
        self.b.reserve(**request())
        self.b.start("t")
        self.b.finish("t", succeeded=False, seconds=1, max_memory=1)
        alias = request("alias", sequence_sha=request()["sequence_sha"])
        with self.assertRaises(ValueError):
            self.b.reserve(**alias)
        with self.assertRaises(ValueError):
            self.b.reserve(**dict(alias, attempt=1))
        self.b.reserve(**request(attempt=1))
        self.b.start("t", attempt=1)
        self.b.finish("t", attempt=1, succeeded=False, seconds=1, max_memory=1)
        with self.assertRaises(ValueError):
            self.b.reserve(**alias)
        self.assertEqual(self.b.snapshot()["consumed"], 2)

    def test_protocol_mismatch_and_torn_journal_fail_closed(self):
        self.b.reserve(**request())
        self.protocol.write_text(self.protocol.read_text() + "\n")
        with self.assertRaises(ValueError):
            BudgetLedger(self.protocol, self.ledger).snapshot()
        with self.ledger.open("ab") as f:
            f.write(b'{"partial":')
        with self.assertRaises(ValueError):
            self.b.snapshot()


if __name__ == "__main__":
    unittest.main()
