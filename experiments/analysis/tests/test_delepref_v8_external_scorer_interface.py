"""Single-deletion head interface test, synthetic embeddings only."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import delepref_v8_runtime as rt
from delepref_v8_baseline import ROOT


class SingleDeletionInterfaceTest(unittest.TestCase):
    def test_original_scorer_accepts_actual_single_deletion_budget(self):
        torch.set_num_threads(2)
        runtime,_=rt.load(ROOT,device='cpu',cached_heads_only=True)
        from delepref.preference_core import DeletionCandidate
        from delepref.student_model import EncodedWildTypes,wildtype_key
        sequence='ACDEFGHIKLMNPQRSTVWYACDEF';key=wildtype_key('single_deletion_fixture',sequence)
        g=torch.Generator().manual_seed(20261016)
        encoded=EncodedWildTypes(embeddings={key:torch.randn(len(sequence),1152,generator=g)},sequences={key:sequence},input_count=1)
        candidates=[DeletionCandidate(protein_id=key.protein_id,perturbation_id=str(pos),plan_id=str(pos),
            wildtype_sequence=sequence,deleted_intervals=((pos,pos),),deleted_length=1,
            deletion_fraction=1/len(sequence),target_delete_fraction=1/len(sequence),
            budget_bin='retrospective_single_deletion',generation_policy='interface_test') for pos in (0,10)]
        with torch.inference_mode():values=runtime.model.score_plans_from_embeddings(candidates,encoded)['preference_score']
        self.assertEqual(values.numel(),2);self.assertTrue(torch.isfinite(values).all())


if __name__=='__main__':unittest.main()
