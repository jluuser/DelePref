from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import delepref_v8_controls as c
import delepref_unlimited_planning as p


class ControlTests(unittest.TestCase):
    def test_terminal_split_exact_budget_extremes_and_K(self):
        for length in (150,249,550):
            for budget in (.1,.2):
                rows=c.terminal_proposals(length,budget);d=p.delete_count(length,budget)
                self.assertEqual(len(rows),8);self.assertEqual(rows[0]['C_deleted'],d)
                self.assertEqual(rows[-1]['N_deleted'],d)
                for row in rows:self.assertEqual(p.validate_plan(length,budget,row['spans']).sum(),d)

    def test_conservation_direction_gap_and_unknown_denominators(self):
        profile=dict(representatives=[dict(projected='AA-.'),dict(projected='AC-.')])
        values,audit=c.conservation_logits('AAAA',profile)
        self.assertLess(values[0],values[1]);self.assertLess(values[1],values[2])
        self.assertEqual(values[3],0.);self.assertEqual(audit['known_counts'],[2,2,2,0])
        self.assertTrue(np.isfinite(values).all())
        _,unknown=c.conservation_logits('AAAA',dict(representatives=[]));self.assertTrue(unknown['all_unknown'])
        with self.assertRaises(ValueError):c.conservation_logits('AAA',profile)

    def test_subset_balancing_nested_order_independent_and_outcome_independent(self):
        panel=[dict(protein_id=source+str(i),stratum=source,sequence_sha256=c.sequence_sha(source+str(i)))
            for source in ('pdb_chain','cath_domain') for i in range(30)]
        subsets=c.fixed_subsets(panel)
        self.assertEqual(subsets,c.fixed_subsets(list(reversed(panel))))
        panel[0]['future_prediction']='arbitrary';self.assertEqual(subsets,c.fixed_subsets(panel))
        for field,size in [('secondary_10pct',40),('second_predictor',8),('completion_ablation',20)]:
            self.assertEqual(len(subsets[field]),size)
            self.assertEqual(sum(pid.startswith('pdb_chain') for pid in subsets[field]),size//2)
        self.assertTrue(set(subsets['second_predictor'])<=set(subsets['completion_ablation'])<=set(subsets['secondary_10pct']))

    def test_seed_identity_matches_original_material(self):
        import hashlib
        sequence='ACDEFGHIK';material='\0'.join(('13','x',c.sequence_sha(sequence),'0.20'))
        expected=int(hashlib.sha256(material.encode()).hexdigest()[:16],16)&((1<<63)-1)
        self.assertEqual(c.condition_seed(13,'x',sequence,.2),expected)
        self.assertNotEqual(c.condition_seed(13,'x',sequence,.2),c.condition_seed(29,'x',sequence,.2))

    def test_input_support_classes_and_arbitrary_event_count(self):
        sequence='A'*32;profile=dict(representatives=[dict(target='x',projected='AA-'*8+'A'*8)])
        complete=c.input_evidence(sequence,profile,.25)
        self.assertEqual(complete['support_class'],'complete');self.assertEqual(complete['maximum_events_one_homolog'],8)
        self.assertEqual(c.input_evidence(sequence,profile,.5)['support_class'],'partial')
        self.assertEqual(c.input_evidence(sequence,dict(representatives=[]),.2)['support_class'],'none')


if __name__=='__main__':unittest.main()
