import unittest
from recomb_confirm_v2_uniform_msa import align_donor,aligned_rows


class UniformMSA(unittest.TestCase):
    def check_preserved(self,q,d):
        a=align_donor(q,d)
        self.assertEqual(a.replace('-','').upper(),d)
        self.assertEqual(len(''.join(c for c in a if not c.islower())),len(q))
        self.assertEqual(a,align_donor(q,d))
        return a

    def test_query_shrinking_does_not_shrink_donor(self):
        a=self.check_preserved('ACDEFGHIK','ACWWDEFGHIK')
        self.assertEqual(sum(c.islower() for c in a),2)

    def test_query_insertion_becomes_donor_gap(self):
        a=self.check_preserved('ACWWDEFGHIK','ACDEFGHIK')
        self.assertEqual(a.count('-'),2)

    def test_substitution_and_repeated_sequence_ties(self):
        self.check_preserved('ACDEFGHIK','ACNEFGHIK')
        self.check_preserved('AAAAAA','AAAA')

    def test_source_lowercase_and_all_donor_rows_survive(self):
        rows=[('old-query','ACDEFGHIK'),('one','ACwwDEFGHIK'),('two','AC--EFGHIK')]
        result=aligned_rows('ACDEFGHIK',rows)
        self.assertEqual(len(result),3)
        self.assertEqual(result[1][1].replace('-','').upper(),'ACWWDEFGHIK')
        self.assertEqual(result[2][1].replace('-','').upper(),'ACEFGHIK')


if __name__=='__main__':unittest.main()
