import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'code'))
from prediction_journal import PredictionJournal


class UncappedJournal(unittest.TestCase):
    def test_uncapped_category_and_verified_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);protocol=root/'protocol.json'
            protocol.write_text(json.dumps({'compute':{'structure_call_limit':None,
                'structure_categories':{'main':None,'retry':None},'structure_retries_max':1}}))
            journal=PredictionJournal(protocol,root/'journal.jsonl')
            args=dict(sequence_sha='a'*64,wt_variant='variant',predictor_weights_sha='b'*64,
                      profile_sha='c'*64,seed=13,category='main')
            first=journal.reserve(task_id='first',**args)
            self.assertTrue(first['new_reservation'])
            self.assertFalse(journal.reserve(task_id='pending',**args)['new_reservation'])
            journal.start('first');output=root/'coords.npz';output.write_bytes(b'verified output')
            journal.finish('first',succeeded=True,seconds=1,max_memory=1,output_path=output)
            cached=journal.reserve(task_id='same_output',**args)
            self.assertEqual(cached['state'],'cache_hit');self.assertEqual(journal.snapshot()['consumed'],1)
            output.write_bytes(b'changed')
            with self.assertRaises(ValueError):journal.reserve(task_id='first',**args)

    def test_valid_explicit_limit_above_old_800(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);protocol=root/'protocol.json'
            protocol.write_text(json.dumps({'compute':{'structure_call_limit':1001,
                'structure_categories':{'main':None,'retry':None},'structure_retries_max':1}}))
            journal=PredictionJournal(protocol,root/'journal.jsonl')
            self.assertEqual(journal.limit,1001)


if __name__=='__main__':unittest.main()
