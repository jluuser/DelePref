from pathlib import Path
import sys
import unittest
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import delepref_v8_retrieval as p


class RetrievalTests(unittest.TestCase):
    def test_complete_target_identity_and_close_natural_deletion(self):
        wt='ACDEFGHIKLMNPQRSTVWY'*5
        target=wt[:40]+wt[42:];aligned=wt[:40]+'--'+wt[42:]
        fields=['q','t','.98','100','1','100','100','1','98','98','1e-20','100','1','1',wt,aligned,target]
        hit=p.project(fields,wt)
        selected,_=p.representatives([hit],{hashlib.sha256(wt.encode()).hexdigest()})
        self.assertEqual(len(selected),1);self.assertTrue(selected[0]['complete_target_identity_checked'])
        bad=list(fields);bad[-1]='A'*98
        with self.assertRaises(ValueError):p.project(bad,wt)

    def test_exact_known_full_sequence_is_excluded(self):
        wt='ACDEFGHIKL'
        f=['q','t','1','10','1','10','10','1','10','10','1e-20','100','1','1',wt,wt,wt]
        hit=p.project(f,wt)
        selected,audit=p.representatives([hit],{hashlib.sha256(wt.encode()).hexdigest()})
        self.assertFalse(selected);self.assertEqual(audit['known_sequence_excluded'],1)

    def test_dereplication_preserves_different_gap_events(self):
        row='A'*100
        base=dict(identity=.98,evalue=1e-20,bits=100.,known_coverage=1.,target_sequence_sha256='safe')
        hits=[dict(base,target='a',projected=row[:20]+'--'+row[22:]),
              dict(base,target='b',projected=row[:40]+'--'+row[42:]),
              dict(base,target='c',projected='C'+row[1:20]+'--'+row[22:])]
        selected,audit=p.representatives(hits,set())
        self.assertEqual(len(selected),2);self.assertEqual(audit['same_gap_pattern_redundant'],1)


if __name__=='__main__':unittest.main()
