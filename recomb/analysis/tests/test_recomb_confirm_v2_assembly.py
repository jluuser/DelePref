import unittest
from recomb_confirm_v2_assembly import apply_transform


class AssemblyTransformTests(unittest.TestCase):
    def test_rotates_then_translates_without_compacting_unknown(self):
        p=apply_transform([[1,0,0],None,[-1,0,0]],[[0,-1,0],[1,0,0],[0,0,1]],[2,3,4])
        self.assertEqual(p,[[2.,4.,4.],None,[2.,2.,4.]])

    def test_identity_preserves_coordinate_indexing(self):
        self.assertEqual(apply_transform([None,[1,2,3]],[[1,0,0],[0,1,0],[0,0,1]],[0,0,0]),[None,[1.,2.,3.]])

    def test_nonfinite_operation_rejected(self):
        with self.assertRaises(ValueError):apply_transform([[1,2,3]],[[float('nan'),0,0],[0,1,0],[0,0,1]],[0,0,0])


if __name__=='__main__':unittest.main()
