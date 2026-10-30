import unittest
from recomb_confirm_v2_application_contexts import apply_variant


def example():
    return dict(proteins=[dict(predictor_chain_id='A',target_gene_copy=True,sequence='ANCD'),
        dict(predictor_chain_id='B',target_gene_copy=True,sequence='ANCD'),
        dict(predictor_chain_id='C',target_gene_copy=False,sequence='AAAA')],
        ligands=[dict(id='L0001',role='covalent_glycan'),dict(id='L0002',role='native_observed_annotated_cofactor')],
        covalent_bonds=[dict(atom1=dict(kind='protein',chain='A',residue=2,atom='ND2',WT_AA='N'),
            atom2=dict(kind='ligand',chain='L0001',residue=1,atom='C1'))])


class ContextIdentity(unittest.TestCase):
    def test_all_target_copies_edit_together_fixed_partner_is_unchanged(self):
        c=apply_variant(example(),'ACD',dict(wt_to_edit=[0,None,1,2],position_states=['stable','deleted','stable','stable']))
        self.assertEqual([p['sequence'] for p in c['proteins']],['ACD','ACD','AAAA'])
        self.assertEqual([l['id'] for l in c['ligands']],['L0002'])
        self.assertEqual(c['covalent_bonds'],[])

    def test_stable_correspondence_with_substitution_does_not_keep_glycosylation(self):
        c=apply_variant(example(),'ADCD',dict(wt_to_edit=[0,1,2,3],position_states=['stable']*4))
        self.assertEqual(c['glycan_chains_removed_with_lost_anchor'],['L0001'])
        self.assertEqual(c['covalent_bonds'],[])

    def test_unique_retained_site_moves_with_its_actual_index(self):
        c=apply_variant(example(),'NCD',dict(wt_to_edit=[None,0,1,2],position_states=['deleted','stable','stable','stable']))
        self.assertEqual(c['covalent_bonds'][0]['atom1']['residue'],1)
        self.assertEqual(len(c['ligands']),2)


if __name__=='__main__':unittest.main()
