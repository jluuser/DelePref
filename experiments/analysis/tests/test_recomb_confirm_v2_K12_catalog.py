import copy
from pathlib import Path
import unittest

from recomb_confirm_v2_K12_catalog import compatible_identity, derive_predictor, explicit_correspondence, physical_request
from recomb_confirm_v2_K12_evaluate import bounds, summary


class K12CatalogTests(unittest.TestCase):
    def setUp(self):
        self.profile = dict(seed=20261017, sample=0, device_class='A100_80GB', num_recycles=3,
                            MSA='none_single_sequence', chunk_size=64)
        self.request = physical_request('ACDEFGHI', self.profile, 'weights')

    def test_exact_identity_reuses_and_each_input_change_rejects(self):
        for key, value in [('seed', 20261018), ('sample', 1), ('profile_sha256', 'other'),
                           ('weights_sha256', 'other'), ('sequence', 'ACDEFGHK'), ('canonical_chains_sha256', 'other')]:
            changed = copy.deepcopy(self.request);changed[key] = value
            self.assertFalse(compatible_identity(changed, self.request), key)
        self.assertTrue(compatible_identity(self.request, self.request))

    def test_same_edited_sequence_different_masks_share_fold_but_not_identity_mapping(self):
        a, ma = explicit_correspondence('AAAAAAAAAA', [(0, 1)])
        b, mb = explicit_correspondence('AAAAAAAAAA', [(8, 9)])
        self.assertEqual(a, b)
        self.assertNotEqual(ma, mb)
        self.assertEqual(physical_request(a, self.profile, 'w'), physical_request(b, self.profile, 'w'))

    def test_fixed_mapping_rejects_overlap_and_wrong_budget(self):
        with self.assertRaises(ValueError):
            explicit_correspondence('ABCDEFGHIJ', [(0, 1), (1, 2)])
        with self.assertRaises(ValueError):
            explicit_correspondence('ABCDEFGHIJ', [(0, 0)])
        edited, mapping = explicit_correspondence('ABCDEFGHIJ', [(1, 1), (8, 8)])
        self.assertEqual(edited, 'ACDEFGHJ')
        self.assertEqual(mapping['wt_to_edit'], [0, None, 1, 2, 3, 4, 5, 6, None, 7])

    def test_real_parent_predictor_changes_namespaces_only(self):
        source = (Path(__file__).parents[1]/'recomb_confirm_v2_general_predict.py').read_text()
        new = derive_predictor(source)
        self.assertIn("CAT=Path('reports/recomb_confirm_v2_K12_catalog')", new)
        self.assertIn('num_recycles=profile[\'num_recycles\']', new)
        with self.assertRaises(ValueError):
            derive_predictor(source.replace("BASE=Path('reports/recomb_confirm_v2_General100_structures')", 'BASE=None'))

    def test_fixed_identity_bounds_do_not_become_common_shape(self):
        row = dict(common_structure={'full_WT_TM': .9}, fixed_identity_geometry={
            'retained_ca_lddt_lower': .2, 'retained_ca_lddt_upper': .7,
            'contact_recovery_lower': .1, 'contact_recovery_upper': .6})
        self.assertEqual(bounds(row, 'full_WT_TM'), (.9, .9))
        self.assertEqual(bounds(row, 'fixed_native_contact_recovery'), (.1, .6))
        self.assertEqual(bounds(row, 'fixed_retained_CA_lddt'), (.2, .7))

    def test_undefined_case_keeps_full_denominator_and_undefined_mean(self):
        rows = [dict(common_structure={'full_WT_TM': .8}), dict(common_structure={})]
        value = summary(rows, 'full_WT_TM')
        self.assertEqual((value['N_all'], value['N_defined']), (2, 1))
        self.assertIsNone(value['mean_lower'])
        self.assertIsNone(value['mean_upper'])


if __name__ == '__main__':
    unittest.main()
