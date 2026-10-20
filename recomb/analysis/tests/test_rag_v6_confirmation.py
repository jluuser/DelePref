import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import rag_v6_confirmation as c
import rag_v6_features as f


class CandidateTests(unittest.TestCase):
    def test_five_terminal_allocations_exact_budget_and_span_limit(self):
        masks=c.terminal_masks(200,40)
        self.assertEqual(len(masks),5)
        for m in masks:
            self.assertEqual(sum(m),40)
            self.assertLessEqual(len(f.spans(m)),2)
        self.assertEqual(masks[3][:20],[1]*20)
        self.assertEqual(masks[3][-20:],[1]*20)

    def test_small_budget_deduplicates_and_integer_rounding(self):
        self.assertEqual(len(c.terminal_masks(7,1)),2)
        for m in c.terminal_masks(17,3): self.assertEqual(sum(m),3)

    @unittest.skipIf(os.name=="nt","inherited torch DP checked on cu9")
    def test_inherited_DP_is_deterministic_and_legal_on_tiny_fixture(self):
        sys.path.insert(0,"/public/home/zhangyangroup/chengshiz/keyuan.zhou/DelePref/src")
        import torch
        from delepref.student_model import generate_strict_budget_plans
        torch.set_num_threads(1)
        kwargs=dict(num_candidates=3,qmax=3,seed=20261015,max_attempts=12)
        a=generate_strict_budget_plans(torch.zeros(25),.2,**kwargs)
        b=generate_strict_budget_plans(torch.zeros(25),.2,**kwargs)
        self.assertEqual([p.deleted_intervals for p in a.plans],[p.deleted_intervals for p in b.plans])
        self.assertEqual(len(a.plans),3)
        for plan in a.plans:
            self.assertEqual(sum(y-x+1 for x,y in plan.deleted_intervals),5)
            self.assertLessEqual(len(plan.deleted_intervals),3)


if __name__=="__main__": unittest.main()
