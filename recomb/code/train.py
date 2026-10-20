"""Development training; frozen embeddings and WT-only structural inputs."""
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from common import read_jsonl, sha_file, sha_text, write_json, write_jsonl, confined_path
from model import PairResponseHead
from features import pair_features
from adapters.delepref_adapter import retained_mapping


def hierarchy_mean(rows, value='ce', keys=('protein_id', 'budget', 'generator')):
    if not rows:
        raise ValueError('empty hierarchy')
    if not keys:
        return float(np.mean([r[value] for r in rows]))
    groups = {}
    for r in rows:
        groups.setdefault(str(r[keys[0]]), []).append(r)
    return float(np.mean([hierarchy_mean(g, value, keys[1:]) for g in groups.values()]))


class HierarchicalSampler:
    def __init__(self, rows, seed):
        self.rng = np.random.default_rng(seed)
        self.groups = {}
        for r in rows:
            self.groups.setdefault(r['protein_id'], {}).setdefault(str(r['budget']), {}).setdefault(r['generator'], []).append(r)
        self.proteins = sorted(self.groups)
    def sample(self, count=4):
        ps = self.rng.choice(self.proteins, count, replace=len(self.proteins)<count)
        result = []
        for p in ps:
            budgets = self.groups[p]; b = self.rng.choice(sorted(budgets)); generators = budgets[b]
            g = self.rng.choice(sorted(generators)); candidates = generators[g]
            result.append(candidates[int(self.rng.integers(len(candidates)))])
        return result
    def pairs(self, size, count=256):
        if size == 0:
            raise ValueError('candidate has no valid labels')
        return self.rng.choice(size, count, replace=size<count)


class Inputs:
    """Feature API deliberately accepts no labels or variant coordinates."""
    def __init__(self, cache_indices, protocol_hash):
        self.index = {}; self.protocol_hash = protocol_hash; self.arrays = {}; self.references = {}; self.profiles = set(); self.candidates = {}
        for path in cache_indices:
            for item in read_jsonl(path):
                key = item['sequence_sha256']
                if key in self.index and self.index[key]['key'] != item['key']:
                    raise ValueError('multiple embedding profiles for sequence')
                self.index[key] = item
    def embedding(self, sequence):
        key = sha_text(sequence)
        if key not in self.arrays:
            item = self.index[key]; path = Path(item['path']); meta = json.loads(path.with_suffix('.json').read_text())
            if meta['sequence_sha256'] != key or meta['protocol_sha256'] != self.protocol_hash or meta['output_sha256'] != sha_file(path):
                raise ValueError('embedding identity failed')
            for name, value in item.items():
                if name not in ('key','path','reused') and meta.get(name) != value:
                    raise ValueError('index metadata differs')
            a = np.load(path)
            if a.shape != (len(sequence),1152) or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('invalid embedding')
            profile = {k:v for k,v in meta.items() if k not in ('sequence_sha256','length','output_sha256','gpu','batch_seconds')}
            self.profiles.add(sha_text(json.dumps(profile,sort_keys=True)))
            if len(self.profiles)>1:
                raise ValueError('mixed ESMC profiles')
            self.arrays[key] = a
        return self.arrays[key]
    def candidate(self, row):
        cid = row['candidate_id']
        if cid in self.candidates:
            return self.candidates[cid]
        from build_pairs import reference_pairs, retained_pairs
        pid = row['protein_id']
        if pid not in self.references:
            path = Path(row['wt_native_json'])
            if sha_file(path) != row['wt_native_json_sha256'] or sha_file(row['reference_path']) != row['reference_sha256']:
                raise ValueError('WT reference identity failed')
            # Native export is sequence indexed; avoid loading any variant structure.
            data = json.loads(path.read_text())
            if data['wt_sequence_sha256'] != sha_text(row['wt_sequence']):
                raise ValueError('WT sequence identity failed')
            coords = np.full((len(row['wt_sequence']),3),np.nan)
            for k,v in data['coordinates'].items(): coords[int(k)] = v
            self.references[pid] = coords, reference_pairs(coords)
        coords,(wi,wj) = self.references[pid]
        kept = retained_mapping(row['wt_sequence'],row['sequence'],row['mask'])
        ids,i,j,_ = retained_pairs(len(coords),kept,wi,wj)
        gij,gji = pair_features(row['wt_sequence'],coords,kept,i,j)
        result = dict(wt=self.embedding(row['wt_sequence']), edited=self.embedding(row['sequence']), kept=kept, i=i,j=j,gij=gij,gji=gji), ids, len(wi)
        self.candidates[cid] = result
        return result


def logits(model, data, arm, device, selection=None):
    values = {}
    for k,v in data.items():
        if selection is not None and k in ('i','j','gij','gji'): v=v[selection]
        values[k]=torch.as_tensor(v,device=device,dtype=torch.long if k in ('kept','i','j') else torch.float32)
    return model(**values,arm=arm)


def computational_code_hashes():
    root = Path(__file__).parent
    return {str(f.relative_to(root)): sha_file(f) for f in sorted(root.rglob('*.py'))
            if 'tests' not in f.relative_to(root).parts}


_LABEL_CACHE = {}


def load_labels(row, summaries, labels_dir):
    s=summaries[row['candidate_id']]; path=Path(labels_dir)/(row['candidate_id']+'.npz')
    key = (str(path.resolve()), s['labels_sha256'])
    if key in _LABEL_CACHE:
        return _LABEL_CACHE[key]
    if sha_file(path)!=s['labels_sha256']: raise ValueError('label identity failed')
    with np.load(path) as z:
        result = {k:z[k] for k in ('pair_ids','target')}
    _LABEL_CACHE[key] = result
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('protocol','manifest','splits','labels','label-summary','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--cache-index',nargs='+',required=True); p.add_argument('--arm',choices=list('SABCD'),required=True)
    p.add_argument('--seed',type=int,default=20261013); p.add_argument('--lr',type=float,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--technical',action='store_true'); p.add_argument('--steps',type=int)
    args=p.parse_args(); torch.set_num_threads(4)
    if args.device!='cpu' and not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('GPU training requires Slurm allocation')
    protocol=json.loads(Path(args.protocol).read_text()); cfg=protocol['train']
    if args.lr not in cfg['learning_rates'] or args.seed not in cfg['seeds']: raise ValueError('unfrozen hyperparameters')
    if args.steps and not args.technical: raise ValueError('step override only for technical check')
    if args.device=='cpu' and not args.technical: raise ValueError('CPU execution is technical only')
    out=confined_path(args.output); out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(args.manifest); split=json.loads(Path(args.splits).read_text())
    if split['status']!='SPLIT_FROZEN' or split['protocol_sha256']!=sha_file(args.protocol): raise ValueError('split identity failed')
    train=[r for r in rows if split['assignment'][r['protein_id']]=='train']; valid=[r for r in rows if split['assignment'][r['protein_id']]=='validation']
    if args.technical:
        train=[r for r in train if r['protein_id']==train[0]['protein_id']]; valid=[r for r in valid if r['protein_id']==valid[0]['protein_id']]
    summaries={r['candidate_id']:r for r in read_jsonl(args.label_summary)}
    inputs=Inputs(args.cache_index,sha_file(args.protocol)); sampler=HierarchicalSampler(train,args.seed)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    model=PairResponseHead().to(args.device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=cfg['weight_decay'])
    provenance={k:sha_file(getattr(args,k)) for k in ('protocol','manifest','splits')}
    provenance['label_summary_sha256']=sha_file(args.label_summary)
    provenance['cache_index_sha256']={str(Path(p)):sha_file(p) for p in args.cache_index}
    provenance['code']=computational_code_hashes()
    provenance['config']=vars(args); write_json(out/'run.json',provenance)
    best=float('inf'); stale=0; validations=[]; max_steps=args.steps or cfg['max_steps']
    trace_path=out/'train_trace.jsonl'
    if trace_path.exists(): raise ValueError('training output already contains a trace; choose a fresh output')
    from evaluate import evaluate
    for step in range(1,max_steps+1):
        model.train(); optimizer.zero_grad(); losses=[]; sampled=[]
        for r in sampler.sample(cfg['proteins_per_step']):
            data,ids,_=inputs.candidate(r); lab=load_labels(r,summaries,args.labels)
            positions=np.searchsorted(ids,lab['pair_ids'])
            if np.any(positions>=len(ids)) or not np.array_equal(ids[positions],lab['pair_ids']): raise ValueError('label mapping failed')
            ix=sampler.pairs(len(positions),cfg['pairs_per_candidate'])
            loss=torch.nn.functional.cross_entropy(logits(model,data,args.arm,args.device,positions[ix]),torch.as_tensor(lab['target'][ix],device=args.device))
            losses.append(loss); sampled.append({'candidate_id':r['candidate_id'],'replacement':len(positions)<cfg['pairs_per_candidate']})
        loss=torch.stack(losses).mean(); loss.backward(); optimizer.step()
        with trace_path.open('a') as trace_file:
            trace_file.write(json.dumps({'step':step,'ce':float(loss.detach()),'candidates':sampled})+'\n')
        if step%cfg['eval_every']==0 or step==max_steps:
            report=evaluate(model,valid,inputs,summaries,args.labels,args.arm,args.device,out/f'validation_{step}')
            ce=report['hierarchical_ce']; improved=ce<best
            if improved: best=ce; stale=0
            else: stale+=1
            checkpoint={'model':model.state_dict(),'step':step,'validation_ce':ce,'arm':args.arm,'provenance':provenance,'esmc_profile_hashes':sorted(inputs.profiles)}
            torch.save(checkpoint,out/f'checkpoint_{step}.pt')
            if improved: torch.save(checkpoint,out/'best.pt')
            validations.append({'step':step,'ce':ce,'improved':improved}); write_jsonl(out/'validation_trace.jsonl',validations)
            if stale>=cfg['patience']: break
    write_json(out/'status.json',{'status':'TECHNICAL_COMPLETE' if args.technical else 'DEVELOPMENT_RUN_COMPLETE','arm':args.arm,'best_ce':best,'steps':step,'stopped_by_patience':stale>=cfg['patience'],'convergence_review_required':args.arm in ('S','B'),'esmc_profile_hashes':sorted(inputs.profiles)})

if __name__=='__main__': main()
