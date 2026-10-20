import copy
import unittest
import numpy as np
from recomb_confirm_v2_application_metrics import (rigid_fit,native_role_maps,interface_contacts,
    backbone_components,enzyme_catalytic_geometry)


def atom(i,name,x):return dict(residue_index=i,name=name,element='C',xyz=[float(x),0.,0.])
def chain(cid,role,points):
    return dict(predictor_chain_id=cid,role=role,target_gene_copy=role=='design_antibody_H',sequence='A'*len(points),
        sequence_sha256='test-'+role,native_label_chain=cid,assembly_matrix=np.eye(3).tolist(),assembly_vector=[0.,0.,0.],
        native_CA=[[x,0.,0.] for x in points],heavy_atoms=[atom(i,name,x) for i,x in enumerate(points) for name in ['CA','CB']])


class ApplicationMetricTests(unittest.TestCase):
    def setUp(self):
        h=chain('A','design_antibody_H',[0.,20.]);ag=chain('B','fixed_antigen',[4.,24.])
        self.native=dict(primary_target_chain='A',chains=[h,ag],
            observed_H_antigen_contacts=[dict(design_index=i,partner_chain='B',partner_index=i) for i in range(2)],observed_H_light_contacts=[])
        self.pred={c['predictor_chain_id']:copy.deepcopy(dict(sequence=c['sequence'],coordinates=c['native_CA'],heavy_atoms=c['heavy_atoms'])) for c in [h,ag]}
        self.corr=dict(wt_to_edit=[0,1],position_states=['stable','stable'])
        self.mapping={'A':'A','B':'B'};self.required={'A':['CA','CB']}

    def metric(self):return interface_contacts(self.native,self.pred,self.corr,self.mapping,self.required)

    def test_identity_and_strict_five_angstrom_cutoff(self):
        self.assertEqual(self.metric()['contact_recovery_lower'],1.)
        self.assertEqual(self.metric()['interface_F1_lower'],1.)
        for a in self.pred['B']['heavy_atoms']:a['xyz'][0]+=1.
        r=self.metric();self.assertEqual(r['contact_recovery_upper'],0.)

    def test_deleted_contact_keeps_original_denominator(self):
        self.corr=dict(wt_to_edit=[None,0],position_states=['deleted','stable'])
        self.pred['A']=dict(sequence='A',coordinates=[[20.,0.,0.]],heavy_atoms=[atom(0,n,20) for n in ['CA','CB']])
        r=self.metric();self.assertEqual(r['native_contacts'],2);self.assertEqual(r['contact_recovery_lower'],.5)
        self.assertAlmostEqual(r['interface_F1_lower'],2/3)

    def test_ambiguous_correspondence_is_bounded(self):
        self.corr=dict(wt_to_edit=[None,1],position_states=['ambiguous','stable'])
        r=self.metric();self.assertEqual(r['contact_recovery_lower'],.5);self.assertEqual(r['contact_recovery_upper'],1.)
        self.assertLessEqual(r['interface_F1_lower'],r['interface_F1_upper'])

    def test_partial_atoms_do_not_erase_observed_positive_contact(self):
        self.pred['A']['heavy_atoms']=[a for a in self.pred['A']['heavy_atoms'] if a['name']=='CA']
        r=self.metric();self.assertEqual(r['contact_recovery_lower'],1.)
        self.assertLess(r['interface_F1_lower'],r['interface_F1_upper'])

    def test_light_contacts_are_separate_from_antigen(self):
        light=chain('C','fixed_light_chain',[0.]);self.native['chains'].append(light)
        self.pred['C']=dict(sequence='A',coordinates=light['native_CA'],heavy_atoms=light['heavy_atoms']);self.mapping['C']='C'
        self.native['observed_H_light_contacts']=[dict(design_index=0,partner_chain='C',partner_index=0)]
        self.assertEqual(self.metric()['native_contacts'],2)
        self.assertEqual(interface_contacts(self.native,self.pred,self.corr,self.mapping,self.required,'fixed_light_chain')['native_contacts'],1)

    def test_deleted_interface_does_not_produce_zero_RMSD(self):
        self.corr=dict(wt_to_edit=[None,None],position_states=['deleted','deleted'])
        self.pred['A']=dict(sequence='',coordinates=[],heavy_atoms=[])
        r=backbone_components(self.native,self.pred,self.corr,self.mapping)
        self.assertIsNone(r['native_interface_backbone_RMSD_A']);self.assertIsNone(r['antigen_fit_H_backbone_RMSD_A'])

    def test_rigid_fit_does_not_allow_reflection(self):
        reference=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        prediction=reference@np.array([[0,-1,0],[1,0,0],[0,0,1]])+[4,2,6]
        self.assertLess(rigid_fit(reference,prediction)[2],1e-10)
        prediction=reference.copy();prediction[:,0]*=-1
        self.assertGreater(rigid_fit(reference,prediction)[2],.1)

    def test_only_closed_native_operator_actions_are_equivalent(self):
        c=chain('A','design_antibody_H',[0.,20.]);d=copy.deepcopy(c);d['predictor_chain_id']='B'
        d['assembly_matrix']=[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]]
        allowed=native_role_maps(dict(chains=[c,d]));self.assertEqual(len(allowed),2)
        d['native_label_chain']='distinct-original-chain'
        self.assertEqual(len(native_role_maps(dict(chains=[c,d]))),1)

    def test_deleted_catalytic_sites_are_not_zero_geometric_error(self):
        self.native['catalytic']=[dict(wt_index=0),dict(wt_index=1)]
        self.corr=dict(wt_to_edit=[None,None],position_states=['deleted','deleted'])
        self.pred['A']=dict(sequence='',coordinates=[],heavy_atoms=[])
        r=enzyme_catalytic_geometry(self.native,self.pred,self.corr,self.mapping)
        self.assertIsNone(r['catalytic_pair_CA_distance_MAE_A']);self.assertEqual(r['known_identity_preserved_site_pairs'],0)


if __name__=='__main__':unittest.main()
