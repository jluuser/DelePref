import unittest
from recomb_confirm_v2_input_symmetry import filter_input_role_maps


class InputSymmetryTests(unittest.TestCase):
    allowed=[dict(A='A',B='B'),dict(A='B',B='A')]
    request=dict(chains=[dict(id='A',sequence='AAA',msa_csv_sha256='same'),dict(id='B',sequence='AAA',msa_csv_sha256='same')])

    def input(self,two_anchors):
        seqs=[dict(ligand=dict(id='L1',ccd=['NAG']))]
        bonds=[dict(bond=dict(atom1=['A',1,'ND2'],atom2=['L1',1,'C1']))]
        if two_anchors:
            seqs.append(dict(ligand=dict(id='L2',ccd=['NAG'])))
            bonds.append(dict(bond=dict(atom1=['B',1,'ND2'],atom2=['L2',1,'C1'])))
        return dict(sequences=seqs,constraints=bonds)

    def test_unmatched_covalent_attachment_breaks_equivalence(self):
        kept,audit=filter_input_role_maps(self.allowed,self.request,self.input(False))
        self.assertEqual(len(kept),1);self.assertEqual(len(audit['rejected']),1)

    def test_equivalent_covalent_graph_allows_native_copy_swap(self):
        self.assertEqual(len(filter_input_role_maps(self.allowed,self.request,self.input(True))[0]),2)

    def test_different_MSA_breaks_equivalence(self):
        request=dict(chains=[self.request['chains'][0],dict(self.request['chains'][1],msa_csv_sha256='different')])
        self.assertEqual(len(filter_input_role_maps(self.allowed,request,dict(sequences=[],constraints=[]))[0]),1)

    def test_unbound_identical_species_do_not_assign_a_hidden_protein_anchor(self):
        inp=dict(sequences=[dict(ligand=dict(id='L1',ccd=['MG']))],constraints=[])
        self.assertEqual(len(filter_input_role_maps(self.allowed,self.request,inp)[0]),2)


if __name__=='__main__':unittest.main()
