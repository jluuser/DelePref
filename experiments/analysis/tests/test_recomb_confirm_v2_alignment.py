from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recomb_confirm_v2_structure_alignment import aligned_geometry,parse_alignment


class CommonAlignmentTests(unittest.TestCase):
    def test_gaps_preserve_full_supplied_coordinate_indexing(self):
        output='("\\":\\" denotes residue pairs)\nAA-AA\n:: ::\nAAA-A\n'
        self.assertEqual(parse_alignment(output,4,4),[(0,0),(1,1),(3,3)])

    def test_truncated_alignment_is_rejected(self):
        with self.assertRaises(ValueError):parse_alignment('denotes residue pairs\nAAA\n:::\nAAA\n',4,3)

    def test_identity_and_rigid_rotation_have_equal_quality(self):
        binary=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013/provenance/usalign_20261017/USalign')
        if not binary.exists():self.skipTest('Official binary checked on CPU Slurm host')
        rng=np.random.default_rng(42);ref=np.cumsum(rng.normal(size=(40,3)),axis=0)
        rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        value=aligned_geometry(ref,ref@rot+np.array([5.,3.,-2.]),binary)
        self.assertAlmostEqual(value['full_WT_TM'],1.,places=4)
        self.assertEqual(value['aligned_CA'],40)
        self.assertAlmostEqual(value['structure_mapped_native_contact_recovery'],1.)


if __name__=='__main__':unittest.main()
