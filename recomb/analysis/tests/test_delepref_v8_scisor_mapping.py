import itertools
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import delepref_v8_scisor_mapping as mapping


class Tokenizer:
    cls_token_id=0;eos_token_id=2
    residues='ACDEFGHIKLMNPQRSTVWY'
    def convert_ids_to_tokens(self,ids):return [self.residues[i-4] for i in ids]
    def encode(self,sequence):return [0]+[self.residues.index(x)+4 for x in sequence]+[2]


class MappingTests(unittest.TestCase):
    def test_all_small_masks_with_repeated_residues(self):
        tokenizer=Tokenizer();sequence='ACACACACA'
        for deleted in itertools.combinations(range(9),4):
            kept=[i for i in range(9) if i not in deleted]
            preserved=[0]+[i+1 for i in kept]+[10]
            edited=''.join(sequence[i] for i in kept)
            result=mapping.mapped_plan(sequence,4/9,preserved,tokenizer.encode(edited),tokenizer)
            self.assertEqual(result['kept_wt_indices'],kept);self.assertEqual(result['edited_sequence'],edited)
            self.assertEqual(result['deleted_length'],4)

    def test_arbitrary_span_count(self):
        sequence='A'*30;kept=[i for i in range(30) if i%3!=1];t=Tokenizer()
        result=mapping.mapped_plan(sequence,1/3,[0]+[i+1 for i in kept]+[31],t.encode('A'*20),t)
        self.assertEqual(result['span_count'],10);self.assertIsNone(result['qmax'])

    def test_invalid_counts_order_and_reconstruction_are_rejected(self):
        t=Tokenizer();sequence='ACDEFGHIKL';preserved=[0,1,2,3,4,5,6,7,8,11];ids=t.encode(sequence[:8])
        mapping.mapped_plan(sequence,.2,preserved,ids,t)
        for bad in (preserved[:-1],preserved[:2]+[1]+preserved[3:],preserved[1:],list(reversed(preserved))):
            with self.assertRaises(ValueError):mapping.mapped_plan(sequence,.2,bad,ids,t)
        with self.assertRaises(ValueError):mapping.mapped_plan(sequence,.2,preserved,t.encode('A'*8),t)
        with self.assertRaises(ValueError):mapping.mapped_plan(sequence,.2,preserved,ids[:-1],t)


if __name__=='__main__':unittest.main()
