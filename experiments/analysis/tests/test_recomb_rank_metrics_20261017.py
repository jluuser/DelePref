import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recomb_expanded_metrics_20261017 import spearman,rank_metrics


class MetricTests(unittest.TestCase):
    def test_rank_and_constants(self):
        self.assertAlmostEqual(spearman([1,2,2,4],[4,2,2,1]),-1.)
        self.assertIsNone(spearman([1,2],[0,0]))
    def test_auc_ndcg_and_ties(self):
        a=rank_metrics([0,0,1,1],[0,0,1,1],[0,0,1,1]);self.assertEqual(a['AUROC'],1.);self.assertAlmostEqual(a['NDCG'],1.)
        b=rank_metrics([0,0,1,1],[0,0,0,0],[0,0,1,1]);self.assertEqual(b['AUROC'],.5);self.assertAlmostEqual(b['top10pct_enrichment'],1.)
    def test_missing_binary_no_auc(self):
        self.assertIsNone(rank_metrics([0,1],[0,1],[None,None])['AUROC'])


if __name__=='__main__':unittest.main()
