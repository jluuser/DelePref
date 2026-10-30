import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("experiment_statistics", Path(__file__).resolve().parents[1] / "paired_statistics.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def row(wt, group, method, lo, hi=None):
    return dict(wt_id=wt, group_id=group, method=method, lower=lo, upper=lo if hi is None else hi)


class StatisticsTests(unittest.TestCase):
    def test_known_constant_and_collapsed_bounds(self):
        roster = {"a": "1", "b": "2"}
        rows = [row(w, g, m, v) for w, g in roster.items() for m, v in (("X", .75), ("Y", .25))]
        result = stats.paired_bootstrap(rows, "X", "Y", roster)
        self.assertEqual(result["difference_bounds"], (.5, .5))
        self.assertEqual(result["confidence_interval"], (.5, .5))
        self.assertEqual(result["replicates"], 10000)

    def test_hierarchy_equal_groups_not_variants_or_wts(self):
        roster = {"a": "1", "b": "1", "c": "2"}
        rows = [row("a", "1", "X", 0)] * 20 + [row("b", "1", "X", 2), row("c", "2", "X", 5)]
        rows += [row(w, g, "Y", 0) for w, g in roster.items()]
        result = stats.paired_bootstrap(rows, "X", "Y", roster)
        self.assertEqual(result["difference_bounds"], (3., 3.))
        self.assertEqual(result["confidence_interval"], (1., 5.))
        self.assertEqual(result, stats.paired_bootstrap(list(reversed(rows)), "X", "Y", dict(reversed(list(roster.items())))))

    def test_joint_interval_and_failed_wt_retained(self):
        roster = {"a": "1", "failed": "2"}
        rows = [row("a", "1", "X", .8, 1), row("a", "1", "Y", .1, .3),
                row("failed", "2", "X", 0, 1), row("failed", "2", "Y", 0, 1)]
        result = stats.paired_bootstrap(rows, "X", "Y", roster)
        self.assertEqual(result["n_wts"], 2)
        self.assertAlmostEqual(result["difference_bounds"][0], -.25)
        self.assertAlmostEqual(result["difference_bounds"][1], .95)
        self.assertEqual(result["confidence_interval"], (-1., 1.))
        reverse = stats.paired_bootstrap(rows, "Y", "X", roster)
        self.assertEqual(reverse["confidence_interval"], (-1., 1.))

    def test_missing_and_undefined_rejected(self):
        rows = [row("a", "1", "X", 1), row("a", "1", "Y", 0)]
        with self.assertRaises(ValueError):
            stats.paired_bootstrap(rows, "X", "Y", {"a": "1", "failed": "2"})
        rows[0]["lower"] = None
        with self.assertRaises(ValueError):
            stats.paired_bootstrap(rows, "X", "Y", {"a": "1"})

    def test_regret_bounds(self):
        self.assertEqual(stats.regret_interval([(1, 2), (4, 5)], 0), (2., 4.))
        self.assertEqual(stats.regret_interval([(1, 1), (4, 4)], 1), (0., 0.))
        self.assertEqual(stats.regret_interval([(0, 1), (0, 1)], 0), (0., 1.))


if __name__ == "__main__":
    unittest.main()
