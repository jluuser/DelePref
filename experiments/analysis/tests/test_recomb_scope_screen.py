import importlib.util
from pathlib import Path
import unittest
import hashlib

p = Path(__file__).resolve().parents[1] / 'recomb_scope_screen_20261017.py'
s = importlib.util.spec_from_file_location('scope_screen', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
bp = Path(__file__).resolve().parents[1] / 'recomb_benign_baselines_20261017.py'
bs = importlib.util.spec_from_file_location('benign_baseline_gate', bp)
bm = importlib.util.module_from_spec(bs)
bs.loader.exec_module(bm)


class ScopeTests(unittest.TestCase):
    def row(self, organism='Saccharomyces cerevisiae', description='Cellular metabolic enzyme', keywords=None):
        return dict(organism=organism, description=description, keywords=keywords or [])

    def test_missing_metadata_is_not_allowed(self):
        self.assertEqual(m.classify([], True)[0], 'HOLD')

    def test_no_keywords_without_positive_review_is_not_allowed(self):
        self.assertEqual(m.classify([self.row()], False)[0], 'HOLD')

    def test_ordinary_reviewed_entry_allowed(self):
        self.assertEqual(m.classify([self.row()], True)[0], 'RETAIN')

    def test_hazard_overrides_positive_flag(self):
        self.assertEqual(m.classify([self.row(keywords=['Toxin'])], True)[0], 'EXCLUDE')

    def test_viral_partner_cannot_hide_behind_benign_design_chain(self):
        self.assertEqual(m.classify([self.row(), self.row(organism='Example virus')], True)[0], 'EXCLUDE')

    def test_resistance_from_nonpathogenic_source_still_excluded(self):
        self.assertEqual(m.classify([self.row(keywords=['Antibiotic resistance'])], True)[0], 'EXCLUDE')

    def test_task_identity_gate_rejects_unknown_id_and_changed_sequence(self):
        roster = [dict(protein_id='toy', sequence_sha256=hashlib.sha256(b'ACD').hexdigest())]
        self.assertTrue(bm.validate_tasks([dict(protein_id='toy', sequence='ACD')], roster))
        for task in [dict(protein_id='other', sequence='ACD'), dict(protein_id='toy', sequence='ACE')]:
            with self.assertRaises(ValueError):
                bm.validate_tasks([task], roster)


if __name__ == '__main__':
    unittest.main()
