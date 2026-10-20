import itertools
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recomb_edit_correspondence_20261017 import consensus


def exhaustive(a,b):
    paths=[]
    def visit(i,j,cost,mapping):
        if i==len(a) and j==len(b):paths.append((cost,tuple(mapping)));return
        if i<len(a):visit(i+1,j,cost+1,mapping+[None])
        if j<len(b):visit(i,j+1,cost+1,mapping)
        if i<len(a) and j<len(b):visit(i+1,j+1,cost+(a[i]!=b[j]),mapping+[j])
    visit(0,0,0,[]);best=min(c for c,_ in paths)
    return best,[mp for c,mp in paths if c==best]


class CorrespondenceTests(unittest.TestCase):
    def test_exhaustive_short_strings(self):
        strings=['']+[''.join(p) for n in range(1,4) for p in itertools.product('AB',repeat=n)]
        for a,b in itertools.product(strings,repeat=2):
            result=consensus(a,b);cost,maps=exhaustive(a,b)
            self.assertEqual(result['edit_distance'],cost)
            for i,record in enumerate(result['possibilities']):
                options={mp[i] for mp in maps}
                self.assertEqual(set(record['edited_positions']),options-{None})
                self.assertEqual(record['deletion_possible'],None in options)

    def test_repeated_positions_not_arbitrarily_assigned(self):
        r=consensus('AAA','AA')
        self.assertEqual(r['position_states'],['ambiguous']*3)

    def test_unique_correspondence_keeps_substitution(self):
        self.assertEqual(consensus('ABC','AXC')['wt_to_edit'],[0,1,2])


if __name__=='__main__':unittest.main()
