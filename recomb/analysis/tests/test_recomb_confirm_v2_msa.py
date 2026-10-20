import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from recomb_confirm_v2_audit_msa import source_class,row_identity


class MSASourceTests(unittest.TestCase):
    def test_ordinary_source_does_not_override_function_concern(self):
        e=dict(organism=dict(scientificName='Homo sapiens',lineage=['Mammalia']),
               proteinDescription=dict(name='venom toxin'))
        self.assertEqual(source_class(e),'HOLD_FUNCTION_REVIEW')

    def test_unknown_source_is_hold_not_dangerous(self):
        self.assertEqual(source_class(dict(organism=dict(scientificName='Unknown',lineage=[]))),
                         'HOLD_SOURCE_NOT_POSITIVELY_REVIEWED')

    def test_unchanged_fragment_is_verified_but_substitution_is_not(self):
        self.assertTrue(row_identity('test','AC-dE','TTACDEGG'))
        self.assertFalse(row_identity('test','AC-NE','TTACDEGG'))


if __name__=='__main__':unittest.main()
