from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import delepref_v8_panel as p


class PanelTests(unittest.TestCase):
    def test_alignment_reports_full_length_denominator_and_reconstructs_query(self):
        f=['q','t','.8889','9','1','8','8','1','9','10','1e-8','42','1','.9','ACD-EFGHI','ACDKEFGHI']
        h=p.alignment(f,{'q':dict(sequence='ACDEFGHI')})
        self.assertEqual(h['identity_shorter_full'],1.)
        self.assertEqual(h['query_coverage'],1.)
        self.assertEqual(h['target_coverage'],.9)
        self.assertTrue(p.passes(h,.4))
        f[14]='AAA-AAAAA'
        with self.assertRaises(ValueError):p.alignment(f,{'q':dict(sequence='ACDEFGHI')})

    def test_partial_domain_hit_does_not_pass_both_coverage_barrier(self):
        self.assertFalse(p.passes(dict(identity_shorter_full=.9,query_coverage=1.,target_coverage=.2),.4))

    def test_legacy_exclusions_use_sequence_source_and_family_aliases(self):
        row=dict(sequence='A'*200,sequence_sha256='s',pdb_id='1ABC',cath_h_superfamilies=['1.2.3.4'])
        empty=dict(sha=set(),pdb=set(),families=set())
        self.assertIsNone(p.exclusion(row,empty))
        self.assertEqual(p.exclusion(row,dict(empty,sha={'s'})),'legacy_exact_sequence')
        self.assertEqual(p.exclusion(row,dict(empty,pdb={'1ABC'})),'legacy_PDB_entry')
        self.assertEqual(p.exclusion(row,dict(empty,families={'1.2.3.4'})),'legacy_CATH_H_family')

    def test_selection_is_order_independent_one_per_component_and_outcome_blind(self):
        rows=[dict(candidate_id=str(i),sequence_sha256=str(i),measurement=i) for i in range(8)]
        c=p.Components([x['candidate_id'] for x in rows]);c.union('0','1')
        a=p.select(rows,{'3'},c,5)
        changed=[dict(r,measurement=-100*r['measurement']) for r in reversed(rows)]
        b=p.select(changed,{'3'},c,5)
        self.assertEqual([x['candidate_id'] for x in a],[x['candidate_id'] for x in b])
        self.assertEqual(len({x['panel_component'] for x in a}),len(a))
        self.assertNotIn('3',[x['candidate_id'] for x in a])


if __name__=='__main__':unittest.main()
