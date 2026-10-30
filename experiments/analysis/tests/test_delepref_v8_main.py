"""Main engineering guards and original trained heads on synthetic WT embeddings."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import delepref_v8_engineering_acceptance as guard
import delepref_v8_main as main
import delepref_unlimited_planning as planning
import delepref_v8_controls as controls
import delepref_v8_runtime as rt
from delepref_v8_baseline import ROOT


class MainTests(unittest.TestCase):
    def test_engineering_roster_rejects_missing_duplicate_and_changed_budget(self):
        sequence='A'*20;records=[dict(protein_id='fixture',budget=.1,wildtype_sequence=sequence)]
        rows=[]
        for seed in (13,29):
            for rank in range(1,9):
                spans=((rank-1,rank),);mask=planning.validate_plan(20,.1,spans)
                rows.append(dict(protein_id='fixture',wildtype_sequence=sequence,sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest(),
                    budget=.1,generation_seed=seed,deleted_intervals=json.dumps(spans),span_count=1,
                    deletion_mask_sha256=hashlib.sha256(mask.astype('uint8').tobytes()).hexdigest(),qmax=None,scorer_score=rank/10,scorer_rank=rank))
        self.assertEqual(len(guard.validate_rows(rows,records)),2)
        with self.assertRaises(ValueError):guard.validate_rows(rows[:-1],records)
        bad=copy.deepcopy(rows);bad[1]=copy.deepcopy(bad[0])
        with self.assertRaises(ValueError):guard.validate_rows(bad,records)
        bad=copy.deepcopy(rows);bad[0]['budget']=.2
        with self.assertRaises(ValueError):guard.validate_rows(bad,records)
        bad=copy.deepcopy(rows);bad[0]['qmax']=3
        with self.assertRaises(ValueError):guard.validate_rows(bad,records)

    def test_main_methods_ablation_and_exact_fallback_with_real_heads(self):
        torch.set_num_threads(2);runtime,api=rt.load(ROOT,device='cpu',cached_heads_only=True)
        sequence=('ACDEFGHIKLMNPQRSTVWY'*3)[:50]
        generator=torch.Generator().manual_seed(20261016)
        cache=dict(sequences={'fixture':sequence},embeddings={'fixture':torch.randn(50,1152,generator=generator)},real_encoder=False)
        record=dict(protein_id='fixture',wildtype_sequence=sequence,budget=.2)
        chunk=main.cached_chunk(api,cache,[record])
        profile=dict(representatives=[dict(target='synthetic_homolog',projected='AA-'*10+'A'*20)])
        seed=controls.condition_seed(13,'fixture',sequence,.2)
        msa,cost=planning.policy_proposals(controls.conservation_logits(sequence,profile)[0],.2,seed)
        terminal=controls.terminal_proposals(50,.2);control={}
        for method,items in [('msa_conservation',msa),('terminal',terminal)]:
            control[method]=[dict(deleted_intervals=item['spans'],
                deletion_mask_sha256=api._mask_sha256(50,item['spans']),generation_cost=cost if method=='msa_conservation' else dict(attempts=8,duplicates=0)) for item in items]
        with torch.inference_mode():
            rows,selected,diag=main.score_condition(runtime,api,chunk,record,profile,13,control_rows=control,ablation=True)
            self.assertEqual(len(rows),40);self.assertEqual(len(selected),5)
            self.assertEqual({x['method'] for x in selected},{'delepref_unlimited','rag_unlimited','msa_conservation','terminal','rag_direct_only'})
            self.assertTrue(any(x['span_count']>3 for x in rows));self.assertTrue(diag['counterfactual_fallback_exact'])
            for chosen in selected:
                self.assertEqual(chosen['scorer_rank'],1);self.assertEqual(len(chosen['edited_sequence']),40)
                self.assertEqual(controls.sequence_sha(chosen['edited_sequence']),chosen['edited_sequence_sha256'])
            rows,selected,diag=main.score_condition(runtime,api,chunk,record,dict(representatives=[]),13)
            self.assertTrue(diag['fallback']);self.assertEqual(len(rows),16)
            self.assertEqual(selected[0]['deletion_mask_sha256'],selected[1]['deletion_mask_sha256'])
            self.assertEqual(selected[0]['scorer_score'],selected[1]['scorer_score'])


if __name__=='__main__':unittest.main()
