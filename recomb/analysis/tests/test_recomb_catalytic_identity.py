import unittest
from recomb_benign_catalytic_identity_20261017 import retained_identity


class CatalyticIdentityTests(unittest.TestCase):
    sites=[dict(status='PASS',wt_index=1,wt_residue='D')]

    def test_mapped_substitution_does_not_preserve_identity(self):
        m=dict(wt_to_edit=[0,1,2],position_states=['stable']*3)
        self.assertTrue(retained_identity(self.sites,'ADA',m)['all_verified_identities_retained_lower'])
        result=retained_identity(self.sites,'AEA',m)
        self.assertFalse(result['all_verified_identities_retained_upper'])
        self.assertEqual(result['certain_substituted_sites'],[1])

    def test_deleted_site_and_ambiguous_mapping(self):
        m=dict(wt_to_edit=[0,None,1],position_states=['stable','deleted','stable'])
        self.assertFalse(retained_identity(self.sites,'AA',m)['all_verified_identities_retained_upper'])
        m['position_states'][1]='ambiguous';value=retained_identity(self.sites,'AA',m)
        self.assertFalse(value['all_verified_identities_retained_lower'])
        self.assertTrue(value['all_verified_identities_retained_upper'])

    def test_empty_annotation_is_undefined(self):
        self.assertIsNone(retained_identity([],'AAA',dict(wt_to_edit=[0,1,2],position_states=['stable']*3))['site_identity_retention_lower'])


if __name__=='__main__':unittest.main()
