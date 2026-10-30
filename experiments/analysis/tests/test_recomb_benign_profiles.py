import copy
import hashlib
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recomb_benign_profiles_20261017 import filtered_profile


class ProfileTests(unittest.TestCase):
    def fixture(self):
        digest = hashlib.sha256(b'ACD').hexdigest()
        q = dict(sequence_sha256=digest)
        p = dict(query=dict(sequence='ACD', sequence_sha256=digest), weighting='unchanged',
                 audit=dict(selected=3), representatives=[dict(target=t, target_sequence_sha256=t+'sha', projected='ACD') for t in ['a','b','c']])
        return p, q

    def test_filter_preserves_order_projection_and_parent(self):
        p,q=self.fixture(); before=copy.deepcopy(p)
        a={t:dict(status='RETAIN',representative_sequence_sha256=t+'sha') for t in ['c','a']}
        result=filtered_profile(p,q,a)
        self.assertEqual([r['target'] for r in result['representatives']],['a','c'])
        self.assertEqual(result['weighting'],p['weighting'])
        self.assertEqual(p,before)

    def test_missing_evidence_preserves_query_as_empty(self):
        p,q=self.fixture(); result=filtered_profile(p,q,{})
        self.assertEqual(result['representatives'],[])
        self.assertEqual(result['query'],p['query'])

    def test_approved_name_does_not_override_wrong_identity(self):
        p,q=self.fixture()
        with self.assertRaises(ValueError):
            filtered_profile(p,q,{'a':dict(status='RETAIN',representative_sequence_sha256='wrong')})

    def test_query_mismatch_is_rejected(self):
        p,q=self.fixture();q['sequence_sha256']='wrong'
        with self.assertRaises(ValueError):filtered_profile(p,q,{})


if __name__=='__main__':unittest.main()
