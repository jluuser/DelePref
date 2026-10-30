"""Prepare WT-only legal candidate pools on CPU, without scores or variant structures."""
import argparse
import collections
import json
from pathlib import Path
import sys
import numpy as np


def composition(total, parts, rng):
    if parts == 1:return np.array([total],dtype=int)
    cuts=np.sort(rng.choice(total+parts-1,parts-1,replace=False))
    return np.diff(np.r_[-1,cuts,total+parts-1])-1


def mask_for(length, lengths, rng, internal):
    count=len(lengths);base=np.ones(count+1,dtype=int)
    if not internal:base[[0,-1]]=0
    gaps=base+composition(length-int(sum(lengths))-int(base.sum()),count+1,rng)
    mask=np.zeros(length,dtype=bool);start=int(gaps[0])
    for n,size in enumerate(lengths):
        mask[start:start+size]=True;start+=int(size+gaps[n+1])
    return mask


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve();sys.path.insert(0,str(root/'code'))
    from common import AUTHORIZED_ROOT,sha_file,sha_text,identity,read_jsonl,write_json,write_jsonl
    from adapters.delepref_adapter import intervals_from_deletion_mask,retained_mapping,strict_delete_count
    from adapters.structure_adapter import coordinates_from_native_json
    from build_pairs import reference_pairs
    if root!=AUTHORIZED_ROOT.resolve():raise ValueError('unexpected experiment root')
    protocol=json.loads((root/'configs/protocol_v2.json').read_text());conf=protocol['confirmation']
    seal=json.loads((root/'configs/execution_freeze.json').read_text())
    wt_path=root/'manifests/confirmation_wt_frozen.jsonl'
    if sha_file(wt_path)!=seal['input_hashes']['manifests/confirmation_wt_frozen.jsonl']:raise ValueError('WT cohort changed')
    source=json.loads((root/'provenance/source_files.json').read_text())['files']['student_model']
    if sha_file(source['path'])!=source['sha256']:raise ValueError('original legality source changed')
    sys.path.insert(0,str(Path(protocol['sources']['delepref'])/'src'))
    from delepref.student_model import validate_strict_plan
    prepared=[];pipeline=[];audits=[];failures=[]
    rules={'version':'wt_only_candidate_preparation_v2','budget':.2,'seed':20261013,
           'matched':'fixed lengths per WT/span group, internal spans, up to4096 unique legal masks; bins of width0.05 on WT coverage/contact loss; lexicographically smallest four mask SHAs among supported bins',
           'pipeline':'fixed-seed legal sampling; span quotas21,21,22; spans>=3; maximal spans separated by>=1 retained residue; terminal gaps allowed',
           'new_variant_labels_read':False,'model_scores_read':False,'scoring_and_output_freeze':'pending_model_freeze'}
    write_json(root/'configs/confirmation_candidate_preparation_v2.json',rules)
    rule_sha=sha_file(root/'configs/confirmation_candidate_preparation_v2.json')
    for wt in read_jsonl(wt_path):
        seq=wt['wt_sequence'];length=len(seq);deleted=strict_delete_count(length,conf['budget'])
        if sha_file(wt['wt_native_json'])!=wt['wt_native_json_sha256'] or sha_file(wt['reference_path'])!=wt['reference_sha256']:raise ValueError('WT reference changed')
        xyz,_=coordinates_from_native_json(wt['wt_native_json'],seq);i,j=reference_pairs(xyz);ci,cj=reference_pairs(xyz,8.)
        if not len(i) or not len(ci):raise ValueError('WT reference has no required relations')
        def item(mask,kind,group):
            intervals=intervals_from_deletion_mask(mask)
            validate_strict_plan(length,conf['budget'],intervals,qmax=3,min_span_len=3,max_span_len=deleted)
            edited=''.join(c for c,d in zip(seq,mask) if not d);retained_mapping(seq,edited,mask)
            mask_sha=sha_text(''.join('1' if d else '0' for d in mask));variant_sha=sha_text(edited)
            return {**wt,'candidate_id':identity((wt['wt_sequence_sha256'],mask_sha,variant_sha)),
                    'sequence':edited,'sequence_sha256':variant_sha,'mask':mask.astype(int).tolist(),'mask_sha256':mask_sha,
                    'budget':conf['budget'],'intervals':intervals,'num_spans':len(intervals),'group':group,'pool_kind':kind,
                    'generator':'fixed_seed_legal_sampling','candidate_rule_sha256':rule_sha,
                    'direct_relationship_coverage':float((~(mask[i]|mask[j])).mean()),
                    'direct_contact_loss':float((mask[ci]|mask[cj]).mean()),'scoring_status':'PENDING_MODEL_FREEZE'}
        for spans in (1,2,3):
            rng=np.random.default_rng(int(identity([protocol['seed'],wt['protein_id'],'matched',spans])[:16],16))
            lengths=composition(deleted-3*spans,spans,rng)+3
            masks={}
            if spans==1:
                for start in range(1,length-deleted):
                    m=np.zeros(length,dtype=bool);m[start:start+deleted]=True;masks[m.tobytes()]=m
            else:
                for _ in range(conf['pool_per_group_max']*20):
                    m=mask_for(length,lengths,rng,True);masks.setdefault(m.tobytes(),m)
                    if len(masks)==conf['pool_per_group_max']:break
            bins=collections.defaultdict(list)
            for m in masks.values():
                coverage=float((~(m[i]|m[j])).mean());loss=float((m[ci]|m[cj]).mean())
                key=(int(np.floor(coverage/.05)),int(np.floor(loss/.05)))
                mask_sha=sha_text(''.join('1' if x else '0' for x in m))
                bins[key].append((mask_sha,m,coverage,loss))
            eligible=[sorted(v,key=lambda x:x[0])[:4] for v in bins.values() if len(v)>=4]
            if not eligible:
                failures.append({'protein_id':wt['protein_id'],'spans':spans,'reason':'no supported matching bin','pool_size':len(masks)});continue
            chosen=min(eligible,key=lambda group:tuple(x[0] for x in group))
            if np.ptp([x[2] for x in chosen])>.05+1e-12 or np.ptp([x[3] for x in chosen])>.05+1e-12:raise ValueError('matching range failed')
            prepared.extend(item(m,'matched',spans) for _,m,_,_ in chosen)
            audits.append({'protein_id':wt['protein_id'],'spans':spans,'span_lengths':sorted(lengths.tolist()),
                           'legal_position_pool':len(masks),'coverage_range':float(np.ptp([x[2] for x in chosen])),
                           'contact_loss_range':float(np.ptp([x[3] for x in chosen]))})
        for spans,quota in zip((1,2,3),conf['pipeline_span_quotas']):
            rng=np.random.default_rng(int(identity([protocol['seed'],wt['protein_id'],'pipeline',spans])[:16],16));seen=set()
            for _ in range(quota*1000):
                lengths=composition(deleted-3*spans,spans,rng)+3;m=mask_for(length,lengths,rng,False)
                if m.tobytes() in seen:continue
                seen.add(m.tobytes());pipeline.append(item(m,'pipeline',spans))
                if len(seen)==quota:break
            if len(seen)!=quota:failures.append({'protein_id':wt['protein_id'],'spans':spans,'reason':'pipeline quota not reached'})
        print(json.dumps({'WT':wt['protein_id'],'matched_so_far':len(prepared),'pipeline_so_far':len(pipeline)}),flush=True)
    write_jsonl(root/'manifests/confirmation_candidates_prepared.jsonl',prepared)
    write_jsonl(root/'manifests/pipeline_candidates_prepared.jsonl',pipeline)
    write_jsonl(root/'reports/candidate_preparation_groups.jsonl',audits)
    report={'status':'WT_ONLY_CANDIDATE_POOLS_PREPARED' if not failures else 'CANDIDATE_PREPARATION_INCOMPLETE',
            'matched_candidates':len(prepared),'pipeline_candidates':len(pipeline),'WT':len(read_jsonl(wt_path)),
            'pipeline_distinct_sequences':len({r['sequence_sha256'] for r in pipeline}),
            'combined_distinct_sequences':len({r['sequence_sha256'] for r in prepared+pipeline}),
            'combined_distinct_logical_candidates':len({r['candidate_id'] for r in prepared+pipeline}),
            'logical_candidates_shared_across_pools':len({r['candidate_id'] for r in prepared}&{r['candidate_id'] for r in pipeline}),
            'numpy_version':np.__version__,
            'failures':failures,'rules':rules,'code_sha256':sha_file(__file__),'WT_manifest_sha256':sha_file(wt_path),
            'matched_manifest_sha256':sha_file(root/'manifests/confirmation_candidates_prepared.jsonl'),
            'pipeline_manifest_sha256':sha_file(root/'manifests/pipeline_candidates_prepared.jsonl'),
            'new_structure_calls':0,'PLM_calls':0,'final_model_scores_frozen':False}
    write_json(root/'reports/candidate_preparation.json',report);print(json.dumps(report),flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
