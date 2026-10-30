import tempfile
from pathlib import Path
import unittest
import gemmi
from recomb_confirm_v2_application_predict import verify_chemical_output


class ChemicalOutputTests(unittest.TestCase):
    def create(self,path,units):
        structure=gemmi.Structure();model=gemmi.Model('1');chain=gemmi.Chain('L0001')
        for i,code in enumerate(units,1):
            residue=gemmi.Residue();residue.name=code;residue.seqid=gemmi.SeqId(i,' ');residue.het_flag='H'
            atom=gemmi.Atom();atom.name='C1';atom.element=gemmi.Element('C');atom.pos=gemmi.Position(i,0.,0.)
            residue.add_atom(atom);chain.add_residue(residue)
        model.add_chain(chain);structure.add_model(model);structure.setup_entities()
        structure.make_mmcif_document().write_file(str(path))

    def test_actual_long_chain_name_and_CCD_order(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'one.cif';self.create(path,['NAG','MAN'])
            result=verify_chemical_output(path,dict(sequences=[dict(ligand=dict(id='L0001',ccd=['NAG','MAN']))]))
            self.assertEqual(result[0]['CCD_units'],['NAG','MAN']);self.assertEqual(result[0]['heavy_atoms'],2)

    def test_omitted_unit_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'one.cif';self.create(path,['NAG'])
            with self.assertRaises(ValueError):verify_chemical_output(path,dict(sequences=[dict(ligand=dict(id='L0001',ccd=['NAG','MAN']))]))

    def test_omitted_chain_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'one.cif';self.create(path,['NAG'])
            with self.assertRaises(ValueError):verify_chemical_output(path,dict(sequences=[dict(ligand=dict(id='MISSING',ccd=['NAG']))]))


if __name__=='__main__':unittest.main()
