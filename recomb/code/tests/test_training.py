import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train import hierarchy_mean, HierarchicalSampler
from evaluate import predict, decision_report

class TrainingTests(unittest.TestCase):
    def test_checkpoint_identity_mismatches(self):
        from argparse import Namespace
        from copy import deepcopy
        from unittest.mock import patch
        from evaluate import validate_checkpoint_inputs
        args=Namespace(protocol='protocol',manifest='manifest',splits='splits',label_summary='summary',cache_index=['cache'])
        frozen={'protocol':'protocol_hash','manifest':'manifest_hash','splits':'splits_hash',
                'label_summary_sha256':'summary_hash','cache_index_sha256':{'cache':'cache_hash'},
                'code':{'train.py':'code_hash'}}
        with patch('evaluate.sha_file',side_effect=lambda path:str(path)+'_hash'), patch('evaluate.computational_code_hashes',return_value={'train.py':'code_hash'}):
            validate_checkpoint_inputs({'provenance':frozen},args)
            for key in frozen:
                changed=deepcopy(frozen)
                changed[key]={'changed':'hash'} if isinstance(changed[key],dict) else 'changed'
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_checkpoint_inputs({'provenance':changed},args)
            args.cache_index=['different_path']
            with self.assertRaises(ValueError): validate_checkpoint_inputs({'provenance':frozen},args)

    def test_checkpoint_profile_mismatch(self):
        from evaluate import validate_checkpoint_profiles
        validate_checkpoint_profiles(['frozen'],{'frozen'})
        for profiles in ({'different'},set(),{'frozen','different'}):
            with self.assertRaises(ValueError): validate_checkpoint_profiles(['frozen'],profiles)

    def test_label_cache_verifies_once(self):
        import tempfile
        from unittest.mock import patch
        from common import sha_file
        from train import load_labels, _LABEL_CACHE
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.npz'
            np.savez(path,pair_ids=np.array([1,3]),target=np.array([0,4]))
            summary={'test':{'labels_sha256':sha_file(path)}}
            _LABEL_CACHE.clear()
            with patch('train.sha_file',wraps=sha_file) as hashing:
                first=load_labels({'candidate_id':'test'},summary,directory)
                second=load_labels({'candidate_id':'test'},summary,directory)
                self.assertIs(first,second)
                self.assertEqual(hashing.call_count,1)
    def test_hierarchy(self):
        rows=[dict(protein_id='p',budget=1,generator='a',ce=0.)]*9+[dict(protein_id='p',budget=1,generator='b',ce=2.),dict(protein_id='q',budget=1,generator='a',ce=4.)]
        self.assertEqual(hierarchy_mean(rows),2.5)
    def test_sampling_and_replacement(self):
        rows=[dict(protein_id=p,budget=b,generator=g,candidate_id=f'{p}{b}{g}') for p in 'abcd' for b in (1,2) for g in 'xy']
        a=HierarchicalSampler(rows,7); b=HierarchicalSampler(rows,7)
        for _ in range(10):
            batch=a.sample(); self.assertEqual(batch,b.sample()); self.assertEqual(len({r['protein_id'] for r in batch}),4)
        self.assertEqual(len(a.pairs(2,256)),256)
        self.assertEqual(len(set(a.pairs(256,256))),256)
    def test_prediction_no_label_or_variant_access(self):
        import torch
        from model import PairResponseHead
        class FakeInputs:
            def candidate(self,row):
                self_row=row
                return dict(wt=np.zeros((5,1152),np.float32),edited=np.zeros((5,1152),np.float32),kept=np.arange(5),i=np.array([0,0]),j=np.array([3,4]),gij=np.zeros((2,60),np.float32),gji=np.zeros((2,60),np.float32)),np.array([3,7]),4
        ids,p,s=predict(PairResponseHead(),{'prediction_path':'DOES_NOT_EXIST'},FakeInputs(),'S','cpu')
        self.assertEqual(ids.tolist(),[3,7]); np.testing.assert_allclose(p.sum(1),1,atol=1e-6); self.assertEqual(s['C'],.5)
    def test_unknown_regret_bounds(self):
        rows=[dict(protein_id='p',budget=1,generator='g',candidate_id=str(i),That=t,Ghat=t,C=1,G_lower=l,G_upper=u,T_lower=l,T_upper=u) for i,t,l,u in [(0,.9,.2,.8),(1,.8,.6,1.)]]
        d=decision_report(rows)[0]; self.assertAlmostEqual(d['raw_regret_upper'],.8); self.assertEqual(d['raw_regret_lower'],0.)

if __name__=='__main__': unittest.main()
