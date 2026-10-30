import copy
import unittest

from recomb_confirm_v2_application_repeat_report import (
    bounded_comparison, coverage_tables, descriptive_bounds, repeat_tables,
    validate_repeat_rows, validate_seed_provenance,
)


class RepeatReportTests(unittest.TestCase):
    def fixture(self):
        protocol = dict(methods=['delepref_unlimited', 'rag_unlimited'],
                        repeat_subset_by_cohort={'Binder60': ['a', 'b', 'c']},
                        component_by_protein_id=dict(a='family1', b='family1', c='family2'),
                        structure_seeds=[20261017, 20261018, 20261019])
        rows = []
        for pid in ['a', 'b', 'c']:
            for method in protocol['methods'] + ['WT']:
                value = 1.0 if pid == 'c' and method == 'rag_unlimited' else .4
                rows.append(dict(cohort='Binder60', protein_id=pid, method=method, seed_count=3,
                                 endpoints={'full_WT_TM': dict(mean_lower=value, mean_upper=value,
                                     point_seed_SD=0.0, point_worst_seed=value)}))
        return protocol, rows

    def test_inference_counts_WT_and_components_not_seed_replicates(self):
        protocol, rows = self.fixture()
        tables, comparisons = repeat_tables(rows, protocol)
        result = comparisons[0]
        self.assertEqual((result['N_all'], result['components']), (3, 2))
        self.assertEqual(result['lower_effect']['N'], 3)
        self.assertAlmostEqual(result['mean_difference_bounds'][0], .2)
        self.assertEqual(result['CI95_conservative_envelope'], [0.0, .6])
        self.assertEqual(result['favorable_WT_counts']['improved'], 1)
        self.assertEqual(result['favorable_WT_counts']['unchanged'], 2)
        self.assertEqual(len(tables), 3)

    def test_undefined_WT_prevents_a_silent_available_case_mean(self):
        protocol, rows = self.fixture()
        value = next(r for r in rows if r['protein_id'] == 'a' and r['method'] == 'rag_unlimited')['endpoints']['full_WT_TM']
        value.update(mean_lower=None, mean_upper=None, point_seed_SD=None, point_worst_seed=None)
        tables, comparisons = repeat_tables(rows, protocol)
        table = next(r for r in tables if r['method'] == 'rag_unlimited')['endpoints']['full_WT_TM']
        self.assertEqual((table['N_all'], table['N_defined']), (3, 2))
        self.assertIsNone(table['mean_lower'])
        self.assertIsNone(comparisons[0]['CI95_conservative_envelope'])
        self.assertEqual(comparisons[0]['lower_effect']['N'], 3)
        self.assertEqual(comparisons[0]['favorable_WT_counts']['undefined'], 1)

    def test_interval_difference_uses_cross_endpoints(self):
        result = bounded_comparison([('a', .3, .6, .2, .4)], {'a': 'component'}, 'interface_contact_recovery')
        self.assertAlmostEqual(result['mean_difference_bounds'][0], -.1)
        self.assertAlmostEqual(result['mean_difference_bounds'][1], .4)
        self.assertEqual(result['favorable_WT_counts']['uncertain'], 1)
        self.assertEqual(result['favorable_WT_counts']['improved'], 0)

    def test_lower_RMSD_has_favorable_direction_without_flipping_raw_units(self):
        result = bounded_comparison([('a', 1., 1., 2., 2.)], {'a': 'component'}, 'native_interface_backbone_RMSD_A')
        self.assertEqual(result['mean_difference_bounds'], [-1., -1.])
        self.assertEqual(result['favorable_WT_counts']['improved'], 1)
        self.assertEqual(result['lower_effect']['numeric_decrease'], 1)
        self.assertNotIn('degraded', result['lower_effect'])

    def test_lost_duplicate_or_incomplete_seed_member_is_rejected(self):
        protocol, rows = self.fixture()
        with self.assertRaises(ValueError): validate_repeat_rows(rows[:-1], protocol)
        with self.assertRaises(ValueError): validate_repeat_rows(rows + [rows[0]], protocol)
        altered = copy.deepcopy(rows); altered[0]['seed_count'] = 2
        with self.assertRaises(ValueError): validate_repeat_rows(altered, protocol)

    def test_invalid_or_nonfinite_uncertainty_cannot_be_treated_as_data(self):
        for bounds in [[(.8, .2)], [(None, .2)], [(float('nan'), 1.)], [(0., float('inf'))]]:
            with self.assertRaises(ValueError): descriptive_bounds(bounds)

    def test_actual_seed_provenance_cannot_be_replaced_by_a_different_mean(self):
        protocol, summaries = self.fixture()
        per_output = []
        for row in summaries:
            value = row['endpoints']['full_WT_TM']['mean_lower']
            for seed in protocol['structure_seeds']:
                per_output.append(dict(cohort=row['cohort'], protein_id=row['protein_id'], method=row['method'],
                    budget=0 if row['method'] == 'WT' else .2, seed=seed,
                    common_structure={'full_WT_TM': value}))
        validate_seed_provenance(per_output, summaries, protocol)
        per_output[0]['seed'] = 20261018
        with self.assertRaises(ValueError): validate_seed_provenance(per_output, summaries, protocol)
        per_output[0]['seed'] = 20261017; per_output[0]['common_structure']['full_WT_TM'] += .1
        with self.assertRaises(ValueError): validate_seed_provenance(per_output, summaries, protocol)

    def test_status_counts_keep_failed_prediction_and_shared_physical_identity(self):
        protocol, _ = self.fixture()
        rows = [dict(cohort='Binder60', protein_id='a', method='rag_unlimited', budget=.2,
                     seed=s, physical_id='p1' if s != 20261019 else 'p2',
                     prediction_status='SUCCEEDED' if s != 20261019 else 'FAILED')
                for s in protocol['structure_seeds']]
        result = next(r for r in coverage_tables(rows, protocol) if r['analysis'] == 'fixed_three_seed_subset')
        self.assertEqual((result['logical'], result['WT'], result['physical']), (3, 1, 2))
        self.assertEqual(result['failed_or_missing_logical'], 1)
        self.assertEqual(result['statuses']['FAILED'], 1)


if __name__ == '__main__': unittest.main()
