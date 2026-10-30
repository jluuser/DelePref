"""All-retained-pair inference and development-only decision diagnostics."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from common import write_json, write_jsonl, read_jsonl, sha_file, confined_path
from train import Inputs, logits, load_labels, hierarchy_mean, computational_code_hashes
from model import PairResponseHead


def predict(model,row,inputs,arm,device,batch_size=4096):
    data,ids,total=inputs.candidate(row)
    probabilities=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(ids),batch_size):
            probabilities.append(logits(model,data,arm,device,np.arange(start,min(start+batch_size,len(ids)))).softmax(-1).cpu().numpy())
    prob=np.concatenate(probabilities) if probabilities else np.empty((0,5),dtype=np.float32)
    g=float((prob@np.array([1,.75,.5,.25,0])).mean()) if len(prob) else None
    c=len(ids)/total if total else 0.
    return ids,prob,{'Ghat':g,'C':c,'That':c*g if g is not None else 0.}


def decision_report(rows):
    result=[]
    # Source-internal and mixed pools preserve budget; no artificial matching.
    for mixed in (False,True):
        groups={}
        for r in rows:
            key=(r['protein_id'],str(r['budget']),'mixed' if mixed else r['generator'])
            groups.setdefault(key,[]).append(r)
        for key,pool in sorted(groups.items()):
            selected=max(pool,key=lambda r:(r['That'],r['candidate_id']))
            lo=max(r['T_lower'] for r in pool); hi=max(r['T_upper'] for r in pool)
            result.append({'protein_id':key[0],'budget':key[1],'generator':key[2],'pool_size':len(pool),'selected':selected['candidate_id'],**{k:selected[k] for k in ('Ghat','That','C','G_lower','G_upper','T_lower','T_upper')},'oracle_T_lower':lo,'oracle_T_upper':hi,'raw_regret_lower':max(0.,lo-selected['T_upper']),'raw_regret_upper':max(0.,hi-selected['T_lower'])})
    return result


def validate_checkpoint_inputs(checkpoint, args):
    provenance=checkpoint['provenance']
    for key in ('protocol','manifest','splits'):
        if provenance[key]!=sha_file(getattr(args,key)): raise ValueError('checkpoint '+key+' identity differs')
    if provenance['label_summary_sha256']!=sha_file(args.label_summary):
        raise ValueError('checkpoint label summary identity differs')
    actual={str(Path(p)):sha_file(p) for p in args.cache_index}
    if provenance['cache_index_sha256']!=actual:
        raise ValueError('checkpoint cache index paths/hashes differ')
    if provenance['code']!=computational_code_hashes():
        raise ValueError('checkpoint computational code identity differs')


def validate_checkpoint_profiles(expected, actual):
    if not expected or set(expected)!=set(actual):
        raise ValueError('checkpoint ESMC profile identity differs')


def evaluate(model,rows,inputs,summaries,labels_dir,arm,device,output,expected_profiles=None):
    if device!='cpu' and not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('GPU evaluation requires Slurm allocation')
    output=confined_path(output); output.mkdir(parents=True,exist_ok=True)
    if expected_profiles is not None:
        # Validate the complete inference profile before writing predictions or reports.
        for row in rows:
            inputs.candidate(row)
        validate_checkpoint_profiles(expected_profiles,inputs.profiles)
    records=[]; confusion=np.zeros((5,5),dtype=int); pooled=[]; targets=[]
    for row in rows:
        ids,prob,score=predict(model,row,inputs,arm,device)
        lab=load_labels(row,summaries,labels_dir); positions=np.searchsorted(ids,lab['pair_ids'])
        if np.any(positions>=len(ids)) or not np.array_equal(ids[positions],lab['pair_ids']): raise ValueError('evaluation label mapping failed')
        if not len(positions): raise ValueError('validation candidate has no valid labels')
        known=prob[positions]; y=lab['target']; ce=float(-np.log(np.maximum(known[np.arange(len(y)),y],1e-30)).mean())
        np.add.at(confusion,(y,known.argmax(1)),1); pooled.append(known); targets.append(y)
        np.savez_compressed(output/(row['candidate_id']+'.npz'),pair_ids=ids,probabilities=prob)
        s=summaries[row['candidate_id']]
        records.append({k:row[k] for k in ('candidate_id','protein_id','budget','generator')}|score|{'ce':ce,'valid_pairs':len(y)}|{k:s[k] for k in ('G_lower','G_upper','T_lower','T_upper')})
    write_jsonl(output/'candidates.jsonl',records)
    protein=[{'protein_id':p,'ce':hierarchy_mean([r for r in records if r['protein_id']==p])} for p in sorted({r['protein_id'] for r in records})]
    write_jsonl(output/'proteins.jsonl',protein); decisions=decision_report(records); write_jsonl(output/'decisions.jsonl',decisions)
    probabilities=np.concatenate(pooled); y=np.concatenate(targets); confidence=probabilities.max(1); correct=probabilities.argmax(1)==y
    calibration=[]
    for k in range(10):
        mask=(confidence>=k/10)&(confidence<((k+1)/10) if k<9 else confidence<=1)
        calibration.append({'lower':k/10,'upper':(k+1)/10,'count':int(mask.sum()),'confidence':float(confidence[mask].mean()) if mask.any() else None,'accuracy':float(correct[mask].mean()) if mask.any() else None})
    np.savez_compressed(output/'pooled_probabilities.npz',probabilities=probabilities,targets=y)
    report={'status':'DEVELOPMENT_EVALUATED','arm':arm,'hierarchical_ce':hierarchy_mean(records),'proteins':len(protein),'candidates':len(records),'confusion':confusion.tolist(),'calibration':calibration,'interpretation':'historical development; generator pools are not independent confirmation','budget_decisions':[]}
    for b in sorted({r['budget'] for r in decisions}):
        for source in sorted({r['generator'] for r in decisions if r['budget']==b}):
            part=[r for r in decisions if r['budget']==b and r['generator']==source]
            report['budget_decisions'].append({'budget':b,'generator':source,**{k:hierarchy_mean(part,k,('protein_id',)) for k in ('G_lower','G_upper','T_lower','T_upper','That','C','raw_regret_lower','raw_regret_upper')}})
    if expected_profiles is not None:
        validate_checkpoint_profiles(expected_profiles,inputs.profiles)
    write_json(output/'report.json',report); return report


def main():
    p=argparse.ArgumentParser()
    for name in ('checkpoint','protocol','manifest','splits','labels','label-summary','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--cache-index',nargs='+',required=True); p.add_argument('--device',default='cpu'); args=p.parse_args()
    checkpoint=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    validate_checkpoint_inputs(checkpoint,args)
    model=PairResponseHead().to(args.device); model.load_state_dict(checkpoint['model'])
    split=json.loads(Path(args.splits).read_text()); rows=[r for r in read_jsonl(args.manifest) if split['assignment'][r['protein_id']]=='validation']
    evaluate(model,rows,Inputs(args.cache_index,sha_file(args.protocol)),{r['candidate_id']:r for r in read_jsonl(args.label_summary)},args.labels,checkpoint['arm'],args.device,args.output,expected_profiles=checkpoint['esmc_profile_hashes'])

if __name__=='__main__': main()
