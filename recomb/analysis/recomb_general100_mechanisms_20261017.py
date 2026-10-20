"""Fixed evidence interventions on real WT embeddings and trained DelePref heads."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import delepref_unlimited_planning as planning
import delepref_unlimited_scorer as scoring
import delepref_v8_runtime as rt
from delepref_v8_rag_unlimited import cached_chunk, make_rows
from recomb_general100_common_20261017 import ROOT, sha, write, sequence_sha

BASE = 'reports/recomb_general100_mechanisms_20261017'


def project_events(sequence, representative, old_events, new_events):
    row = list(representative['projected'])
    for a,b in old_events:
        row[a:b+1] = sequence[a:b+1]
    for a,b in new_events:
        row[a:b+1] = '-'*(b-a+1)
    out = dict(representative, projected=''.join(row))
    recovered = planning.natural_events({'representatives':[out]}, len(sequence))
    observed = set(recovered[0]['events']) if recovered else set()
    return out if observed == set(new_events) else None


def marginal_profile(sequence, profile, seed):
    events = planning.natural_events(profile, len(sequence))
    frequencies = Counter(e for h in events for e in h['events'])
    catalog = sorted(frequencies)
    rng = np.random.default_rng(seed % (2**32))
    rows, sources = [], []
    for index in range(16):
        if index == 0:
            order = sorted(catalog, key=lambda e:(-frequencies[e],e))
        else:
            order = sorted(catalog, key=lambda e: -np.log(max(rng.random(),1e-15))/frequencies[e])
        selected = []
        for event in order:
            a,b = event
            if all(b+3 <= c or d+3 <= a for c,d in selected):
                selected.append(event)
        selected.sort()
        template = dict(projected=sequence, target='MARGINAL_COUNTERFACTUAL_%02d'%index)
        row = project_events(sequence, template, (), selected)
        if row and selected:
            rows.append(row);sources.append(dict(target=row['target'],events=selected,
                marginal_support=[frequencies[e] for e in selected]))
    return dict(representatives=rows, counterfactual=True), dict(
        intervention='independent event combinations from fixed marginal frequencies',
        source_events=len(catalog), generated_event_bundles=len(rows), bundles=sources,
        not_observed_natural_homologs=True)


def null_profile(sequence, profile, seed):
    original = planning.natural_events(profile,len(sequence))
    targets = {h['target']:set(h['events']) for h in original}
    row_by_target = {r['target']:copy.deepcopy(r) for r in profile['representatives']}
    groups = defaultdict(list)
    for target in targets:
        r=row_by_target[target]
        groups[(min(4,int(float(r.get('identity',0))*5)),min(1,int((float(r.get('known_coverage',1))-.8)*10)))].append(target)
    usable=[g for g in groups.values() if len(g)>1]
    before=Counter(e for v in targets.values() for e in v)
    before_load={k:sum(b-a+1 for a,b in v) for k,v in targets.items()}
    before_count={k:len(v) for k,v in targets.items()}
    rng=np.random.default_rng(seed%(2**32));accepted=attempted=0
    while usable and accepted<100 and attempted<10000:
        attempted+=1
        group=usable[int(rng.integers(len(usable)))];i,j=rng.choice(len(group),2,replace=False)
        left,right=group[int(i)],group[int(j)]
        a=sorted(targets[left]-targets[right]);b=sorted(targets[right]-targets[left])
        pairs=[(e,f) for e in a for f in b if e[1]-e[0]==f[1]-f[0]]
        if not pairs:continue
        e,f=pairs[int(rng.integers(len(pairs)))]
        new_left=targets[left]-{e}|{f};new_right=targets[right]-{f}|{e}
        li=project_events(sequence,row_by_target[left],targets[left],new_left)
        ri=project_events(sequence,row_by_target[right],targets[right],new_right)
        if li is None or ri is None:continue
        row_by_target[left]=li;row_by_target[right]=ri
        targets[left]=new_left;targets[right]=new_right;accepted+=1
    after=Counter(e for v in targets.values() for e in v)
    if before!=after or before_load!={k:sum(b-a+1 for a,b in v) for k,v in targets.items()} or before_count!={k:len(v) for k,v in targets.items()}:
        raise RuntimeError('null changed event marginals or per-row load')
    changed=sum(targets[h['target']]!=set(h['events']) for h in original)
    out=copy.deepcopy(profile)
    out['representatives']=[row_by_target[r['target']] for r in profile['representatives']]
    out['counterfactual']=True
    # When no effective perturbation occurred, the entire input and RNG salt
    # must stay identical to full RAG, including original target identifiers.
    if not changed:out=profile
    return out,dict(intervention='matched event-row double-edge swaps',accepted_swaps=accepted,
        attempted_swaps=attempted,effectively_changed_rows=changed,effective_intervention=bool(changed),
        event_marginals_preserved=True,row_event_count_and_deleted_load_preserved=True,
        original_target_identifiers_retained=True,not_observed_natural_sequences=True)


def select_rows(api, sequence, rows, method, use_policy=False):
    if len(rows)!=8 or len({r['deletion_mask_sha256'] for r in rows})!=8:
        raise ValueError('control does not have unique K8')
    best=(min(rows,key=lambda r:(-r['policy_energy'],api.parse_deleted_intervals(r['deleted_intervals'],len(sequence))))
          if use_policy else min(rows,key=lambda r:r['scorer_rank']))
    best=dict(best,method=method)
    mask=planning.validate_plan(len(sequence),float(best['budget']),json.loads(best['deleted_intervals']))
    edited=''.join(c for c,d in zip(sequence,mask) if not d)
    best.update(edited_sequence=edited,edited_sequence_sha256=sequence_sha(edited),selector='original_Policy_additive' if use_policy else 'original_Scorer')
    return best


def execute(root):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('CPU Slurm only')
    out=root/BASE;out.mkdir(exist_ok=False)
    main=root/'reports/delepref_v8_main'
    complete=json.loads((main/'encoding_completion.json').read_text())
    if sha(main/'wt_cache.pt')!=complete['cache_sha256']:
        raise ValueError('real WT cache identity changed')
    sources=['analysis/recomb_general100_mechanisms_20261017.py','analysis/delepref_unlimited_planning.py',
             'analysis/delepref_unlimited_scorer.py','analysis/delepref_v8_runtime.py','analysis/delepref_v8_rag_unlimited.py']
    inputs=['reports/delepref_v8_main/candidate_scores.json','reports/delepref_v8_main/wt_cache.pt',
            'retrieval/delepref_v8_main/profiles.json','reports/delepref_v8_evaluation_preparation/subsets.json']
    write(root/'configs/recomb_general100_mechanisms_freeze_20261017.json',dict(
        status='CONTROLS_FROZEN_BEFORE_SELECTION',files={p:sha(root/p) for p in sources+inputs},
        qmax=None,K=8,budget=.2,seed=13,null_seed=20261017,null_max_valid_swaps=100,null_attempt_cap=10000,
        no_outcome_based_sampling_or_tuning=True,complete_structure_prediction_limit=None,
        structural_prediction_allowed=False,new_training=False,paper_paused=True))
    torch.set_num_threads(4);started=time.monotonic();runtime,api=rt.load(root,device='cpu',cached_heads_only=True)
    cache=torch.load(main/'wt_cache.pt',map_location='cpu',weights_only=True)
    panel=json.loads((root/'reports/delepref_v8_native_panel/main_panel.json').read_text())
    records=[dict(protein_id=p['protein_id'],wildtype_sequence=p['sequence'],budget=.2) for p in panel]
    chunk=cached_chunk(api,cache,records)
    profiles=json.loads((root/'retrieval/delepref_v8_main/profiles.json').read_text())
    sub=json.loads((root/'reports/delepref_v8_evaluation_preparation/subsets.json').read_text())
    all_scores=json.loads((main/'candidate_scores.json').read_text());groups=defaultdict(list)
    for r in all_scores:
        if float(r['budget'])==.2 and int(r['generation_seed'])==13:
            groups[(r['protein_id'],r['method'])].append(r)
    scarcity=set()
    for source in ('cath_domain','pdb_chain'):
        eligible=[p for p in panel if p['protein_id'] in sub['completion_ablation'] and p['stratum']==source]
        eligible.sort(key=lambda p:hashlib.sha256(('recomb-scarcity-20261017|'+p['sequence_sha256']).encode()).hexdigest())
        scarcity.update(p['protein_id'] for p in eligible[:5])
    selections=[];candidate_rows=[];diagnostics=[]
    with torch.inference_mode():
        for pid in sub['completion_ablation']:
            original=sorted(groups[(pid,'delepref_unlimited')],key=lambda r:r['raw_candidate_rank'])
            rag=sorted(groups[(pid,'rag_unlimited')],key=lambda r:r['raw_candidate_rank'])
            template=original[0];sequence=template['wildtype_sequence'];seed=int(template['condition_seed'])
            embedding=chunk.encoded.embeddings[chunk.encoded.key_for(pid,sequence)]
            logits=runtime.model.propose_from_embeddings([embedding],[.2])[0].residue_logits.detach().cpu().double().numpy()
            native=[dict(spans=tuple(map(tuple,json.loads(r['deleted_intervals']))),kind=r['generation_kind'],row=r) for r in original]
            marginal,md=marginal_profile(sequence,profiles[pid],seed)
            null,nd=null_profile(sequence,profiles[pid],20261017+seed)
            variants=[('rag_marginal_gap',marginal,md),('rag_cooccurrence_null',null,nd)]
            if pid in scarcity:
                reps=sorted(profiles[pid]['representatives'],key=lambda r:hashlib.sha256(('recomb-scarcity-homolog-20261017|'+r['target']).encode()).hexdigest())
                variants.append(('rag_evidence_min8',dict(profiles[pid],representatives=reps[:8]),dict(full_N=len(reps),reduced_N=min(8,len(reps)))))
                zero,flag=planning.mix_candidates(native,[])
                if not flag or [r['row'] for r in zero]!=original:
                    raise RuntimeError('zero evidence fallback differs')
                zero_choice=select_rows(api,sequence,original,'rag_evidence_zero')
                zero_choice['reuse_original_structure_exact']=True;selections.append(zero_choice)
            for method,profile,intervention in variants:
                retrieved,attempts=planning.retrieved_proposals(logits,.2,profile,seed)
                mixed,fallback=planning.mix_candidates(native,retrieved)
                scored=([dict(r) for r in original] if fallback else scoring.score_candidates(runtime,api,chunk,
                          make_rows(api,template,mixed,len(sequence),logits)))
                choice=select_rows(api,sequence,scored,method);selections.append(choice)
                candidate_rows.extend(dict(r,method=method) for r in scored)
                diagnostics.append(dict(protein_id=pid,method=method,intervention=intervention,
                    retrieved_proposals=len(retrieved),fallback=fallback,attempts=attempts))
                if method=='rag_cooccurrence_null' and not nd['effective_intervention']:
                    if choice['deletion_mask_sha256']!=select_rows(api,sequence,rag,'reference')['deletion_mask_sha256']:
                        raise RuntimeError('unperturbed null changed winner')
            selections.append(select_rows(api,sequence,rag,'rag_no_scorer',use_policy=True))
            print(json.dumps(dict(event='GENERAL100_MECHANISM_CPU_COMPLETE',protein_id=pid)),flush=True)
    if len(selections)!=80 or len(scarcity)!=10:
        raise ValueError('mechanism/scarcity roster mismatch')
    write(out/'all_selections.json',selections);write(out/'candidate_scores.json',candidate_rows)
    write(out/'diagnostics.json',diagnostics)
    write(out/'completion.json',dict(status='REAL_HEADS_MECHANISM_SELECTION_COMPLETE',WT=20,scarcity_WT=10,
        selected_outputs=80,new_encoder_forwards=0,new_structure_predictions=0,seconds=time.monotonic()-started,
        original_checkpoint_sha256=rt.CHECKPOINT_SHA,
        effective_null_WT=sum(d['intervention'].get('effective_intervention',False) for d in diagnostics if d['method']=='rag_cooccurrence_null'),
        outputs={n:sha(out/n) for n in ('all_selections.json','candidate_scores.json','diagnostics.json')},
        job_id=os.environ['SLURM_JOB_ID']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);execute(p.parse_args().root)
