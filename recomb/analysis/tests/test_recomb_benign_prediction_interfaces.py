import tempfile
import unittest
from pathlib import Path
from recomb_benign_fold_worker_20261017 import read_predicted_chains
from recomb_benign_structure_evaluate_20261017 import paired_summary


class PredictionInterfaceTests(unittest.TestCase):
    def test_multichain_parser_keeps_chain_identity_and_heavy_atoms(self):
        lines=[];serial=0
        for chain,offset in [('A',0),('B',5)]:
            for atom,x,element in [('N',offset,'N'),('CA',offset+1,'C')]:
                serial+=1
                lines.append(f'ATOM  {serial:5d} {atom:^4s} ALA {chain}{1:4d}    {x:8.3f}{0.:8.3f}{0.:8.3f}  1.00  0.00          {element:>2s}')
            lines.append('TER')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'chains.pdb';path.write_text('\n'.join(lines+['END','']))
            result=read_predicted_chains(path,[dict(id='A',sequence='A'),dict(id='B',sequence='A')])
            self.assertEqual(set(result),{'A','B'})
            self.assertEqual(len(result['A']['heavy_atoms']),2)
            self.assertEqual(result['B']['coordinates'],[[6.,0.,0.]])
            with self.assertRaises(ValueError):read_predicted_chains(path,[dict(id='A',sequence='G')])

    def test_interval_does_not_turn_uncertainty_into_neutral(self):
        pairs=[dict(protein_id='a',difference_lower=-.2,difference_upper=.4)]
        result=paired_summary(pairs,{'a':'one'})
        self.assertEqual(result['unresolved'],1)
        self.assertEqual(result['numerically_unchanged'],0)
        self.assertEqual(result['uncertainty_and_bootstrap_interval95'],[-.2,.4])

    def test_undefined_metric_keeps_complete_case_count(self):
        pairs=[dict(protein_id='a',difference_lower=None,difference_upper=None),dict(protein_id='b',difference_lower=.1,difference_upper=.1)]
        result=paired_summary(pairs,{'a':'a','b':'b'})
        self.assertEqual((result['N'],result['N_defined']),(2,1))


if __name__=='__main__':unittest.main()
