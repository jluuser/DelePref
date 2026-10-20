"""Real original trained heads + explicitly synthetic embeddings; no PLM claim."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import delepref_v8_runtime as rt
from delepref_v8_baseline import ROOT
import delepref_v8_rag_unlimited as rag
import delepref_unlimited_planning as planning
import delepref_unlimited_scorer as scoring


class UnlimitedHeadInterfaceTests(unittest.TestCase):
    def test_original_feature_equivalence_and_unlimited_generation_scoring(self):
        torch.set_num_threads(2)
        runtime,api=rt.load(ROOT,device='cpu',cached_heads_only=True)
        before={k:v.clone() for k,v in runtime.model.state_dict().items()}
        sequence='ACDEFGHIKLMNPQRSTVWYACDEF'
        generator=torch.Generator().manual_seed(20261016)
        cache=dict(sequences={'fixture':sequence},embeddings={'fixture':torch.randn(len(sequence),1152,generator=generator)},real_encoder=False)
        chunk=rag.cached_chunk(api,cache,[dict(protein_id='fixture',wildtype_sequence=sequence,budget=.2)])
        seed=api.derive_condition_seed(13,'fixture',api.sequence_sha256(sequence),.2)
        template=dict(protein_id='fixture',wildtype_sequence=sequence,budget=.2,sequence_length=len(sequence),
            sequence_sha256=api.sequence_sha256(sequence),generation_seed=13,condition_seed=seed)
        logits=np.arange(len(sequence),dtype=float)
        # Original formula must be exactly preserved on all old supported span counts.
        small=[dict(spans=s,kind='test') for s in (((0,4),),((1,2),(8,10)),((1,1),(5,6),(11,12)))]
        rows=rag.make_rows(api,template,small,len(sequence),logits)
        candidates=[api._candidate_from_row(row) for row in rows]
        with torch.inference_mode():
            reference=runtime.model.score_plans_from_embeddings(candidates,chunk.encoded)['preference_score']
            adapted=scoring.preference_scores(runtime.model.scorer,candidates,chunk.encoded)
            self.assertTrue(torch.equal(reference,adapted))
            embedding=next(iter(chunk.encoded.embeddings.values()))
            learned=runtime.model.propose_from_embeddings([embedding],[.2])[0].residue_logits.detach().double().numpy()
            self.assertGreater(float(np.std(learned)),0.)
            native,_=planning.policy_proposals(learned,.2,seed)
            pool=scoring.score_candidates(runtime,api,chunk,rag.make_rows(api,template,native,len(sequence),learned))
            profile=dict(representatives=[dict(target='synthetic_homolog',projected='AA-'*5+'A'*(len(sequence)-15))])
            retrieved,_=planning.retrieved_proposals(learned,.2,profile,seed)
            self.assertEqual(len(retrieved[0]['spans']),5)
            original=[dict(item,row=row) for item,row in zip(native,pool)]
            mixed,fallback=planning.mix_candidates(original,retrieved)
            self.assertFalse(fallback)
            scored=scoring.score_candidates(runtime,api,chunk,rag.make_rows(api,template,mixed,len(sequence),learned))
            self.assertEqual(len(scored),8)
            self.assertTrue(any(x['span_count']>3 and x['scorer_span_extrapolation'] for x in scored))
            self.assertTrue(all(np.isfinite(x['scorer_score']) for x in scored))
            withheld,flag=planning.mix_candidates(original,[])
            self.assertTrue(flag);self.assertEqual([x['row'] for x in withheld],pool)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in runtime.model.state_dict().items()))
        with self.assertRaises(ValueError):rag.cached_chunk(api,cache,[dict(protein_id='fixture',wildtype_sequence='A'*24,budget=.2)])


if __name__=='__main__':unittest.main()
