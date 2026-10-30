import unittest
import numpy as np
from recomb_benign_metrics_20261017 import geometry,binding_contacts


class ScopedMetricTests(unittest.TestCase):
    def test_mapping_ambiguity_is_unknown_not_deletion(self):
        ref=np.array([[i,0,0] for i in range(8)],dtype=float)
        correspondence=dict(wt_to_edit=[0,1,None,None,None,None,4,5],position_states=['stable','stable','ambiguous','ambiguous','ambiguous','ambiguous','stable','stable'])
        value=geometry(ref,ref[[0,1,2,3,6,7]],correspondence)
        self.assertEqual(value['ambiguous_WT_positions'],4)
        self.assertEqual(value['certain_deleted_positions'],0)
        self.assertEqual(value['retained_ca_lddt_lower'],0)
        self.assertEqual(value['retained_ca_lddt_upper'],1)
        self.assertGreater(value['T_upper'],value['T_lower'])

    def test_identity_geometry_remains_exact(self):
        ref=np.array([[i,0,0] for i in range(8)],dtype=float)
        value=geometry(ref,ref,dict(wt_to_edit=list(range(8)),position_states=['stable']*8))
        self.assertEqual(value['retained_ca_lddt_lower'],1.)
        self.assertEqual(value['contact_recovery_lower'],1.)

    def test_deleted_interface_zero_and_unknown_bounded(self):
        a=dict(residue_index=0,name='CA',xyz=[0,0,0]); b=dict(residue_index=0,name='CA',xyz=[1,0,0])
        native=dict(chains=[dict(native_chain_id='X',sequence='A',heavy_atoms=[a]),dict(native_chain_id='Y',sequence='A',heavy_atoms=[b])],
            observed_native_heavy_atom_contacts=[dict(design_index=0,partner_chain='Y',partner_index=0)])
        chain_map=[dict(native_chain_id='X',prediction_chain_id='A'),dict(native_chain_id='Y',prediction_chain_id='B')]
        pred={'A':dict(heavy_atoms=[a]),'B':dict(heavy_atoms=[b])}
        value=binding_contacts(native,pred,dict(wt_to_edit=[0],position_states=['stable']),chain_map)
        self.assertEqual(value['interface_jaccard_lower'],1.)
        value=binding_contacts(native,{},dict(wt_to_edit=[None],position_states=['ambiguous']),chain_map)
        self.assertEqual((value['contact_recovery_lower'],value['contact_recovery_upper']),(0.,1.))
        value=binding_contacts(native,{},dict(wt_to_edit=[None],position_states=['deleted']),chain_map)
        self.assertEqual((value['contact_recovery_lower'],value['contact_recovery_upper']),(0.,0.))


if __name__=='__main__':unittest.main()
