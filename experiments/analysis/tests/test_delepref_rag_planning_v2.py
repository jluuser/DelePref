import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import delepref_rag_planning_v2 as p


class PlanningTests(unittest.TestCase):
    def test_constrained_dp_matches_brute_force(self):
        rng=np.random.default_rng(20261016)
        for n in (5,7,9):
            for _ in range(12):
                logits=rng.normal(size=n);k=round(.4*n)
                force=rng.random(n)<.15;keep=rng.random(n)<.15
                legal=[]
                for selected in itertools.combinations(range(n),k):
                    m=np.zeros(n,dtype=bool);m[list(selected)]=True;s=p.intervals(m)
                    if np.any(force&~m) or np.any(keep&m) or len(s)>2 or any(b-a+1>3 for a,b in s):continue
                    legal.append(float(logits[m].sum()))
                if not legal:
                    with self.assertRaises(p.InfeasibleCompletion):p.constrained_dp(logits,.4,force,keep,qmax=2,max_span=3)
                else:
                    s,v=p.constrained_dp(logits,.4,force,keep,qmax=2,max_span=3)
                    self.assertAlmostEqual(v,max(legal),places=11)
                    self.assertEqual(p.mask_for(n,s).sum(),k)

    def test_anchor_constraints_can_make_budget_infeasible(self):
        force=np.array([0,1,1,0,1,1,0,0],dtype=bool);keep=~force
        with self.assertRaises(p.InfeasibleCompletion):p.constrained_dp(np.ones(8),.75,force,keep)

    def test_natural_events_do_not_infer_missing_or_terminal_absence(self):
        profile=dict(representatives=[dict(target='x',projected='--AA--AA..AA--')])
        self.assertEqual(p.natural_events(profile,14)[0]['events'],((4,5),))

    def test_joint_seed_is_from_one_homolog(self):
        profile=dict(representatives=[dict(target='one',projected='AA--AAAAAAAA'),dict(target='two',projected='AAAAAAA--AAA')])
        seeds=p.seed_sets(profile,12,1/3)
        self.assertTrue(seeds)
        self.assertTrue(all(len(s['spans'])==1 for s in seeds))

    def test_direct_natural_plan_and_partial_completion(self):
        profile=dict(representatives=[dict(target='one',projected='AA--AA--AAAA')])
        proposals,attempts=p.retrieved_proposals(np.arange(12,dtype=float),1/3,profile,13)
        self.assertEqual(proposals[0]['kind'],'retrieval_direct')
        self.assertEqual(proposals[0]['spans'],((2,3),(6,7)))
        partial=dict(representatives=[dict(target='one',projected='AA--AAAAAAAA')])
        proposals,_=p.retrieved_proposals(np.arange(12,dtype=float),1/3,partial,13)
        self.assertEqual(proposals[0]['kind'],'retrieval_completed')
        m=p.mask_for(12,proposals[0]['spans']);self.assertTrue(m[2:4].all());self.assertFalse(m[1]);self.assertFalse(m[4])

    def test_no_evidence_fallback_is_identical(self):
        original=[dict(spans=((i,i),),kind='original',marker=i) for i in range(8)]
        result,fallback=p.mix_candidates(original,[])
        self.assertTrue(fallback);self.assertEqual(result,original)
        self.assertTrue(all(a is b for a,b in zip(result,original)))

    def test_mixed_pool_keeps_budget_and_deduplicates(self):
        original=[dict(spans=((i,i+1),),kind='original') for i in range(8)]
        retrieved=[dict(spans=((0,1),),kind='retrieval'),dict(spans=((9,10),),kind='retrieval')]
        result,fallback=p.mix_candidates(original,retrieved)
        self.assertFalse(fallback);self.assertEqual(len(result),8)
        self.assertEqual(len({x['spans'] for x in result}),8)


    def test_partial_full_span_seed_cannot_crowd_out_feasible_completion(self):
        projected='AA-'*16+'AA'
        profile=dict(representatives=[dict(target='one',projected=projected)])
        seeds=p.seed_sets(profile,len(projected),.2)
        self.assertTrue(seeds)
        self.assertTrue(all(s['complete_natural_plan'] or len(s['spans'])<3 for s in seeds))
        proposals,_=p.retrieved_proposals(np.arange(len(projected),dtype=float),.2,profile,13)
        self.assertTrue(proposals)
        self.assertTrue(all(p.mask_for(len(projected),x['spans']).sum()==10 for x in proposals))


if __name__=='__main__':unittest.main()
