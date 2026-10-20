import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recomb_confirm_v2_span_diagnostic import q3_complete
from delepref_unlimited_planning import intervals,mask_for
from delepref_rag_planning import InfeasibleCompletion


class SpanDiagnosticTests(unittest.TestCase):
    def test_cap_only_solver_matches_exhaustive_same_objective(self):
        values=np.array([9.,-2.,8.,-3.,7.,-4.,6.,-5.])
        spans,score=q3_complete(values,.5)
        feasible=[]
        for deleted in itertools.combinations(range(8),4):
            mask=np.zeros(8,dtype=bool);mask[list(deleted)]=True
            if len(intervals(mask))<=3:
                feasible.append(float(values[mask].sum()))
        self.assertEqual(score,max(feasible))
        self.assertEqual(mask_for(8,spans).sum(),4)
        self.assertLessEqual(len(spans),3)

    def test_four_separated_forced_events_are_reported_infeasible(self):
        forced=np.array([1,0,1,0,1,0,1,0],dtype=bool)
        kept=~forced
        with self.assertRaises(InfeasibleCompletion):
            q3_complete(np.ones(8),.5,forced,kept)


if __name__=='__main__':unittest.main()
