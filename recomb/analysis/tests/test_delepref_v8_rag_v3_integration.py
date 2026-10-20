"""CPU interface test with real frozen heads and explicitly synthetic embeddings."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import delepref_v8_runtime as rt
from delepref_v8_baseline import ROOT
import delepref_v8_rag_v3 as rag
import delepref_rag_planning_v2 as planning


class CachedInterfaceTest(unittest.TestCase):
    def test_cached_WT_identity_generation_scoring_and_completion(self):
        torch.set_num_threads(2)
        runtime,api=rt.load(ROOT,device='cpu',cached_heads_only=True)
        sequence='ACDEFGHIKLMNPQRSTVWYACDEF'
        generator=torch.Generator().manual_seed(20261016)
        cache=dict(sequences={'semantic_fixture':sequence},
            embeddings={'semantic_fixture':torch.randn(len(sequence),1152,generator=generator)},
            real_encoder=False)
        records=[dict(protein_id='semantic_fixture',wildtype_sequence=sequence,budget=.2)]
        chunk=rag.cached_chunk(api,cache,records)
        self.assertEqual(chunk.records[0]['sequence_sha256'],api.sequence_sha256(sequence))
        with torch.inference_mode():
            original=api.score_candidates(runtime,chunk,api.generate_raw_candidates(runtime,chunk,global_generation_seed=13))
            logits=runtime.model.propose_from_embeddings([chunk.encoded.for_candidate(api._candidate_from_row(original[0]))],[.2])[0].residue_logits.detach().double().numpy()
            profile=dict(representatives=[dict(target='synthetic_homolog',projected='AA--'+'A'*(len(sequence)-4))])
            proposals,_=planning.retrieved_proposals(logits,.2,profile,original[0]['condition_seed'])
            self.assertTrue(proposals)
            native=[dict(spans=tuple(map(tuple,api.parse_deleted_intervals(x['deleted_intervals'],len(sequence)))),kind='original_policy',row=x) for x in original]
            mixed,fallback=planning.mix_candidates(native,proposals)
            self.assertFalse(fallback)
            scored=api.score_candidates(runtime,chunk,rag.make_rows(api,original,mixed,len(sequence),logits))
            self.assertEqual(len(scored),8)
            self.assertTrue(all(torch.isfinite(torch.tensor(x['scorer_score'])) for x in scored))
        bad=[dict(records[0],wildtype_sequence='A'*len(sequence))]
        with self.assertRaises(ValueError):rag.cached_chunk(api,cache,bad)


if __name__=='__main__':unittest.main()
