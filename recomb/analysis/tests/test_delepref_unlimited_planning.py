import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import delepref_unlimited_planning as p


class UnlimitedPlanningTests(unittest.TestCase):
    def test_completion_matches_exhaustive_objective(self):
        rng=np.random.default_rng(20261016)
        for n in (5,7,9):
            for _ in range(16):
                values=rng.normal(size=n);d=p.delete_count(n,.4)
                force=rng.random(n)<.15;keep=rng.random(n)<.15
                legal=[]
                for selected in itertools.combinations(range(n),d):
                    mask=np.zeros(n,dtype=bool);mask[list(selected)]=True
                    if np.any(force&~mask) or np.any(keep&mask):continue
                    legal.append(float(values[mask].sum()))
                if not legal:
                    with self.assertRaises(p.InfeasibleCompletion):p.complete(values,.4,force,keep)
                else:
                    spans,score=p.complete(values,.4,force,keep)
                    p.validate_plan(n,.4,spans);self.assertAlmostEqual(score,max(legal),places=12)

    def test_more_than_three_spans_and_deterministic_ties(self):
        values=np.tile([1.,0.],20)
        spans,_=p.complete(values,.25)
        self.assertEqual(spans,tuple((i,i) for i in range(0,20,2)))
        self.assertEqual(len(spans),10)
        self.assertEqual(p.complete(np.ones(10),.4)[0],((0,3),))

    def test_invalid_and_impossible_constraints(self):
        with self.assertRaises(ValueError):p.complete([1.,np.nan],.5)
        with self.assertRaises(ValueError):p.complete(np.ones(8),.5,np.zeros(7))
        with self.assertRaises(p.InfeasibleCompletion):p.complete(np.ones(8),.5,np.ones(8),np.ones(8))
        with self.assertRaises(p.InfeasibleCompletion):p.complete(np.ones(8),.5,must_keep=np.ones(8))
        with self.assertRaises(ValueError):p.validate_plan(8,.5,((1,2),(3,4)))

    def test_reliable_events_are_unlimited_and_unknown_is_not_gap(self):
        row='AA-'*20+'AA'
        profile=dict(representatives=[dict(target='x',projected=row)])
        self.assertEqual(len(p.natural_events(profile,len(row))[0]['events']),20)
        mixed=dict(representatives=[dict(target='x',projected='--AA--AA..AA--')])
        self.assertEqual(p.natural_events(mixed,14)[0]['events'],((4,5),))

    def test_whole_event_subset_matches_exhaustive_counts(self):
        events=((2,3),(6,9),(12,17),(20,24),(27,28))
        for budget in range(1,20):
            subset=p.whole_event_subset(events,budget)
            best=max(sum(b-a+1 for a,b in group) for size in range(6)
                for group in itertools.combinations(events,size) if sum(b-a+1 for a,b in group)<=budget)
            self.assertEqual(sum(b-a+1 for a,b in subset),best)
            self.assertTrue(set(subset)<=set(events))

    def test_same_homolog_joint_support_and_direct_plan_over_three_spans(self):
        row='AA-'*8+'AA'
        profile=dict(representatives=[dict(target='one',projected=row)])
        seeds=p.seed_sets(profile,len(row),8/len(row))
        self.assertEqual(len(seeds[0]['spans']),8)
        proposals,_=p.retrieved_proposals(np.arange(len(row)),8/len(row),profile,13)
        self.assertEqual(proposals[0]['kind'],'retrieval_direct')
        self.assertEqual(len(proposals[0]['spans']),8)
        self.assertEqual(proposals[0]['source_target'],'one')
        separate=dict(representatives=[dict(target='a',projected='AA--AAAAAAAA'),dict(target='b',projected='AAAAAAA--AAA')])
        self.assertTrue(all(len(x['spans'])==1 for x in p.seed_sets(separate,12,1/3)))

    def test_partial_seed_over_three_spans_can_now_be_completed(self):
        row='AA-'*5+'AA'+'A'*13
        profile=dict(representatives=[dict(target='one',projected=row)])
        proposals,_=p.retrieved_proposals(np.arange(len(row)),.3,profile,13)
        self.assertTrue(proposals)
        self.assertEqual(len(proposals[0]['seed_spans']),5)
        mask=p.validate_plan(len(row),.3,proposals[0]['spans'])
        for a,b in proposals[0]['seed_spans']:
            self.assertTrue(mask[a:b+1].all());self.assertFalse(mask[a-1]);self.assertFalse(mask[b+1])

    def test_seeded_policy_generation_and_exact_fallback(self):
        logits=np.linspace(-1.,1.,40)
        native,cost=p.policy_proposals(logits,.2,13)
        again,_=p.policy_proposals(logits,.2,13)
        self.assertEqual(native,again);self.assertEqual(len(native),8)
        for item in native:p.validate_plan(40,.2,item['spans'])
        none,fallback=p.mix_candidates(native,[])
        self.assertTrue(fallback);self.assertTrue(all(a is b for a,b in zip(none,native)))
        mixed,_=p.mix_candidates(native,[dict(spans=((0,7),),kind='retrieval_direct')])
        self.assertEqual(len(mixed),8);self.assertEqual(len({x['spans'] for x in mixed}),8)
        self.assertGreaterEqual(cost['attempts'],8)


if __name__=='__main__':unittest.main()
