import ast
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'analysis'))
from recomb_confirm_v2_E10_padding import ScientificPaddingTokenizer
from recomb_confirm_v2_E10_fidelity import microbatch_layout, padding_count


class ScientificPaddingProfileTests(unittest.TestCase):
    def test_measured_process_clock_function_is_identical_to_completed_clock_QA(self):
        def function(path, name):
            node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
            return ast.dump(node, include_attributes=False)
        self.assertEqual(function(ROOT/'analysis/recomb_confirm_v2_E10_timing.py', 'observe_process'),
                         function(ROOT/'analysis/recomb_confirm_v2_E10_timing_padding_v2.py', 'observe_process'))

    def test_scientific_widths_are_the_original_input_only_microbatch_layout(self):
        records = json.loads((ROOT/'reports/recomb_confirm_v2_general_native/encoder_inputs.json').read_text())
        actual = json.loads((ROOT/'reports/recomb_confirm_v2_E10_fidelity/batch_layout.json').read_text())
        self.assertEqual(microbatch_layout(records, 32), actual)
        widths = {pid: b['padded_token_width'] for b in actual for pid in b['protein_ids']}
        cases = json.loads((ROOT/'reports/recomb_confirm_v2_General100_costs/cases.json').read_text())
        sequences = {r['protein_id']: r['wildtype_sequence'] for r in records}
        self.assertEqual(len(cases), 20)
        self.assertTrue(all(widths[r['protein_id']] >= len(sequences[r['protein_id']])+2 for r in cases))

    def test_real_all20_token_tensors_match_the_exact_measured_fidelity_control(self):
        import torch
        import yaml
        registry = json.loads((ROOT/'provenance/original_delepref_assets_20261016.json').read_text())
        config = yaml.safe_load((ROOT/'configs/delepref_original_restored.yaml').read_text())
        vendor = Path(registry['source_root'])/config['model']['esm_runtime_vendor']
        sys.path.insert(0, str(vendor))
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
        tokenizer = EsmSequenceTokenizer()
        tree = ast.parse((ROOT/'analysis/recomb_confirm_v2_E10_fidelity.py').read_text())
        legacy = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'PaddedTokenizer')
        records = json.loads((ROOT/'reports/recomb_confirm_v2_general_native/encoder_inputs.json').read_text())
        sequences = {r['protein_id']: r['wildtype_sequence'] for r in records}
        layout = json.loads((ROOT/'reports/recomb_confirm_v2_E10_fidelity/batch_layout.json').read_text())
        widths = {pid: b['padded_token_width'] for b in layout for pid in b['protein_ids']}
        cases = json.loads((ROOT/'reports/recomb_confirm_v2_General100_costs/cases.json').read_text())
        module = ast.Module(body=[legacy], type_ignores=[])
        for case in cases:
            pid = case['protein_id']
            namespace = dict(torch=torch, tokenizer=tokenizer, widths=widths, pid=pid, padding_count=padding_count)
            exec(compile(module, '<frozen_fidelity_tokenizer>', 'exec'), namespace)
            old = namespace['PaddedTokenizer']()([sequences[pid]], return_tensors='pt', padding=True)
            new = ScientificPaddingTokenizer(tokenizer, widths[pid])([sequences[pid]], return_tensors='pt', padding=True)
            self.assertEqual(set(old), set(new))
            for key in old:
                self.assertTrue(torch.equal(old[key], new[key]), (pid, key))

    def test_attention_special_tokens_and_input_are_preserved_without_truncation(self):
        import torch
        original = dict(input_ids=torch.tensor([[0, 5, 6, 2]]), attention_mask=torch.ones((1, 4), dtype=torch.long),
                        special_tokens_mask=torch.tensor([[1, 0, 0, 1]]))
        class Tokenizer:
            pad_token_id = 9
            def __call__(self, *args, **kwargs):
                return {k: v.clone() for k, v in original.items()}
        new = ScientificPaddingTokenizer(Tokenizer(), 8)(['AA'])
        self.assertEqual(new['input_ids'].tolist(), [[0, 5, 6, 2, 9, 9, 9, 9]])
        self.assertEqual(new['attention_mask'].tolist(), [[1, 1, 1, 1, 0, 0, 0, 0]])
        self.assertEqual(new['special_tokens_mask'].tolist(), [[1, 0, 0, 1, 1, 1, 1, 1]])
        self.assertEqual(original['input_ids'].tolist(), [[0, 5, 6, 2]])
        with self.assertRaises(ValueError):
            ScientificPaddingTokenizer(Tokenizer(), 3)(['AA'])


if __name__ == '__main__':
    unittest.main()
