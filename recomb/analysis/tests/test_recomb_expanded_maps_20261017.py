"""Coordinate-map completeness and cap behavior on analytically enumerable examples."""
import itertools
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recomb_expanded_external_prepare_20261017 import deletion_maps


class MappingTest(unittest.TestCase):
    def test_exhaustive_short_sequences(self):
        for L in range(2,8):
            for wt in map(''.join,itertools.product('AC',repeat=L)):
                for D in range(1,min(3,L)):
                    expected={}
                    for deleted in itertools.combinations(range(L),D):
                        mut=''.join(a for i,a in enumerate(wt) if i not in deleted)
                        expected.setdefault(mut,[]).append(list(deleted))
                    for mut,maps in expected.items():
                        actual=deletion_maps(wt,mut)
                        self.assertEqual(sorted(maps),actual['maps'])
                        self.assertEqual(len(maps),actual['map_count'])
    def test_cap_never_selects_partial_maps(self):
        r=deletion_maps('A'*20,'A'*10,cap=100)
        self.assertEqual(r['status'],'ambiguous_over_cap');self.assertEqual(r['maps'],[])
    def test_incompatible(self):
        self.assertEqual(deletion_maps('ACDE','ADW')['status'],'not_subsequence')


if __name__=='__main__':unittest.main()
