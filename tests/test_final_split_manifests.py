import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "delepref_final_split_manifest_test",
    str(ROOT / "scripts/build_final_split_manifests.py"),
)
splitter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(splitter)


class DelePrefFinalSplitManifestTests(unittest.TestCase):
    def test_assignment_is_deterministic_and_sequence_scoped(self):
        sequence_hash = splitter.sha256_text("ACDEFGHIK")
        first = splitter.assignment(sequence_hash, 500, 500)
        second = splitter.assignment(sequence_hash, 500, 500)
        self.assertEqual(first, second)
        self.assertIn(first[0], {"train", "dev", "confirmation"})
        self.assertGreaterEqual(first[1], 0)
        self.assertLess(first[1], 10000)

    def test_duplicate_sequence_proteins_cannot_split(self):
        sequence_hash = splitter.sha256_text("SAME_SEQUENCE")
        assignments = {
            protein: splitter.assignment(sequence_hash, 500, 500)[0]
            for protein in ("protein_a", "protein_b", "protein_c")
        }
        self.assertEqual(len(set(assignments.values())), 1)

    def test_pairwise_overlap_reports_every_split_pair(self):
        overlaps = splitter.pairwise_overlap({
            "train": {"a", "b"},
            "dev": {"c"},
            "confirmation": {"d"},
        })
        self.assertEqual(set(overlaps.values()), {0})
        self.assertEqual(len(overlaps), 3)

    def test_fraction_boundaries_are_explicit(self):
        buckets = {}
        for index in range(100000):
            split, bucket, _digest = splitter.assignment(
                splitter.sha256_text("seq_%d" % index), 500, 500
            )
            buckets.setdefault(split, []).append(bucket)
        self.assertTrue(all(bucket < 500 for bucket in buckets["confirmation"]))
        self.assertTrue(all(500 <= bucket < 1000 for bucket in buckets["dev"]))
        self.assertTrue(all(bucket >= 1000 for bucket in buckets["train"]))


if __name__ == "__main__":
    unittest.main()
