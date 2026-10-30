import sys
from pathlib import Path
import unittest
from collections import Counter
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import recomb_general100_mechanisms_20261017 as m
import delepref_unlimited_planning as planning


class EvidenceInterventions(unittest.TestCase):
    def setUp(self):
        self.sequence='A'*40
        bundles=[[(4,5),(14,15)],[(4,5),(24,25)],[(14,15),(30,31)],[(24,25),(30,31)]]
        self.profile={'representatives':[]}
        for i,events in enumerate(bundles):
            row=list(self.sequence)
            for a,b in events:row[a:b+1]='-'*(b-a+1)
            self.profile['representatives'].append(dict(target='h%d'%i,projected=''.join(row),identity=.5,known_coverage=1.))

    def test_null_preserves_marginals_row_load_and_anchors(self):
        after,audit=m.null_profile(self.sequence,self.profile,20261017)
        before=planning.natural_events(self.profile,40);after=planning.natural_events(after,40)
        self.assertEqual(Counter(e for h in before for e in h['events']),Counter(e for h in after for e in h['events']))
        self.assertEqual({h['target']:sum(b-a+1 for a,b in h['events']) for h in before},
                         {h['target']:sum(b-a+1 for a,b in h['events']) for h in after})
        self.assertEqual([len(h['events']) for h in before],[len(h['events']) for h in after])
        self.assertTrue(audit['event_marginals_preserved'])

    def test_no_swappable_evidence_is_exact_input(self):
        one={'representatives':self.profile['representatives'][:1]}
        after,audit=m.null_profile(self.sequence,one,13)
        self.assertEqual(after,one)
        self.assertFalse(audit['effective_intervention'])

    def test_marginal_catalog_and_zero_fallback(self):
        out,audit=m.marginal_profile(self.sequence,self.profile,13)
        catalog={e for h in planning.natural_events(self.profile,40) for e in h['events']}
        self.assertTrue(all(e in catalog for h in planning.natural_events(out,40) for e in h['events']))
        original=[dict(spans=((i,i),),kind='policy') for i in range(8)]
        mixed,fallback=planning.mix_candidates(original,[])
        self.assertTrue(fallback);self.assertEqual(mixed,original)
        self.assertTrue(audit['not_observed_natural_homologs'])


if __name__=='__main__':unittest.main()
