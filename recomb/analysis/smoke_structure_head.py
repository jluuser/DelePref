"""Real-data CPU technical check for S; no PLM inputs, no scientific efficacy claim."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();sys.path.insert(0,str(root/'code'))
    from common import AUTHORIZED_ROOT,read_jsonl,sha_file,write_json
    from train import Inputs,logits,load_labels,HierarchicalSampler
    from evaluate import predict
    from model import PairResponseHead
    if root!=AUTHORIZED_ROOT.resolve():raise ValueError('unexpected experiment root')
    class StructureInputs(Inputs):
        def embedding(self,sequence):
            # S has explicitly disabled WT/variant PLM slots; values cannot affect its output.
            return np.zeros((len(sequence),1152),dtype=np.float32)
    torch.set_num_threads(4);torch.manual_seed(20261013);begin=time.perf_counter()
    if torch.cuda.is_initialized():raise RuntimeError('CPU smoke must not initialize CUDA')
    protocol=root/'configs/protocol_v2.json';split=json.loads((root/'manifests/splits.json').read_text())
    rows=read_jsonl(root/'manifests/development_primary.jsonl')
    train=next(r for r in rows if split['assignment'][r['protein_id']]=='train')
    valid=next(r for r in rows if split['assignment'][r['protein_id']]=='validation')
    train_rows=[r for r in rows if r['protein_id']==train['protein_id']]
    summaries={r['candidate_id']:r for r in read_jsonl(root/'reports/label_candidates.jsonl')}
    inputs=StructureInputs([],sha_file(protocol));sampler=HierarchicalSampler(train_rows,20261013)
    model=PairResponseHead().cpu();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    initial=model.head[0].weight.detach().clone();trace=[]
    for step in range(10):
        model.train();optimizer.zero_grad();losses=[]
        for row in sampler.sample(4):
            data,ids,_=inputs.candidate(row);label=load_labels(row,summaries,root/'labels/pairs')
            positions=np.searchsorted(ids,label['pair_ids'])
            if not np.array_equal(ids[positions],label['pair_ids']):raise ValueError('real-data label mapping mismatch')
            selected=sampler.pairs(len(positions),256)
            loss=torch.nn.functional.cross_entropy(logits(model,data,'S','cpu',positions[selected]),torch.as_tensor(label['target'][selected]))
            losses.append(loss)
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite real-data loss')
        loss.backward();optimizer.step();trace.append(float(loss.detach()))
    if torch.equal(initial,model.head[0].weight):raise ValueError('head parameters did not update')
    ids,probabilities,scores=predict(model,valid,inputs,'S','cpu')
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1),1,atol=1e-6):raise ValueError('invalid validation probabilities')
    label=load_labels(valid,summaries,root/'labels/pairs');positions=np.searchsorted(ids,label['pair_ids'])
    ce=float(-np.log(np.maximum(probabilities[positions,label['target']],1e-30)).mean())
    if torch.cuda.is_initialized():raise RuntimeError('CUDA initialized')
    report={'status':'REAL_DATA_CPU_STRUCTURE_SMOKE_PASSED','arm':'S','technical_only':True,
            'scientific_effectiveness_claim':False,'train_WT':train['protein_id'],'validation_WT':valid['protein_id'],
            'training_candidates_available':len(train_rows),'steps':10,'train_ce_trace':trace,
            'validation_candidate':valid['candidate_id'],'validation_ce':ce,'validation_scores':scores,
            'predicted_relationships':len(ids),'PLM_calls':0,'structure_calls':0,'device':'cpu',
            'cuda_initialized':False,'seconds':time.perf_counter()-begin,'script_sha256':sha_file(__file__),
            'protocol_sha256':sha_file(protocol),'manifest_sha256':sha_file(root/'manifests/development_primary.jsonl')}
    write_json(root/'reports/cpu_structure_smoke.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
