import copy
import unittest
from recomb_confirm_v2_application_missingness import correct_failed_enzyme_row


def fixture():
    native = dict(primary_target_chain='A', catalytic=[dict(wt_index=0)], chains=[dict(
        predictor_chain_id='A', sequence='AAAA', native_CA=[[0, 0, 0], None, None, [1, 0, 0]])])
    row = dict(cohort='Enzyme80', prediction_status='FAILED', edited_sequence='AAAA',
        correspondence=dict(position_states=['stable']*4, wt_to_edit=list(range(4))),
        catalytic_identity=dict(all_verified_identities_retained_lower=True),
        catalytic_identity_geometry=dict(
            identity_aware_catalytic_neighbor_contact_recovery_lower=0.,
            identity_aware_catalytic_neighbor_contact_recovery_upper=0.,
            catalytic_pair_CA_distance_MAE_A=None))
    return native, row


class MissingGeometryBoundsTests(unittest.TestCase):
    def test_intact_sequence_without_prediction_is_unknown_not_identity_loss(self):
        native, row = fixture(); original = copy.deepcopy(row)
        fixed = correct_failed_enzyme_row(row, native)
        self.assertEqual(row, original)
        self.assertEqual(fixed['catalytic_identity'], row['catalytic_identity'])
        self.assertEqual(fixed['catalytic_identity_geometry']['identity_aware_catalytic_neighbor_contact_recovery_upper'], 1.)

    def test_known_site_deletion_remains_irrecoverable(self):
        native, row = fixture(); row['correspondence']['position_states'][0] = 'deleted'
        row['correspondence']['wt_to_edit'][0] = None
        fixed = correct_failed_enzyme_row(row, native)['catalytic_identity_geometry']
        self.assertEqual(fixed['identity_aware_catalytic_neighbor_contact_recovery_upper'], 0.)
        self.assertEqual(fixed['deleted_or_substituted_native_neighbor_contacts'], 1)

    def test_known_site_substitution_is_not_relabelled_unknown(self):
        native, row = fixture(); row['edited_sequence'] = 'CAAA'
        fixed = correct_failed_enzyme_row(row, native)['catalytic_identity_geometry']
        self.assertEqual(fixed['identity_aware_catalytic_neighbor_contact_recovery_upper'], 0.)

    def test_ambiguous_correspondence_stays_unknown(self):
        native, row = fixture(); row['correspondence']['position_states'][0] = 'ambiguous'
        row['correspondence']['wt_to_edit'][0] = None
        fixed = correct_failed_enzyme_row(row, native)['catalytic_identity_geometry']
        self.assertEqual(fixed['unknown_native_neighbor_contacts'], 1)

    def test_successful_predictions_are_unchanged(self):
        native, row = fixture(); row['prediction_status'] = 'SUCCEEDED'
        self.assertEqual(correct_failed_enzyme_row(row, native), row)

    def test_other_cohorts_are_unchanged(self):
        native, row = fixture(); row['cohort'] = 'Binder60'
        self.assertEqual(correct_failed_enzyme_row(row, native), row)


if __name__ == '__main__':
    unittest.main()
