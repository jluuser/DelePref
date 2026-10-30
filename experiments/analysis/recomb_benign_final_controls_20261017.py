"""Complete fixed CPU controls by reusing existing reviewed model/cache outputs."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from recomb_scope_screen_20261017 import read, sha, write

BASE = Path('reports/recomb_benign_final_controls_20261017')
FREEZE = Path('configs/recomb_benign_final_controls_freeze_20261017.json')


def prepare(root):
    subsets = read(root/'reports/recomb_benign_final_subsets_20261017/subsets.json')
    ids = subsets['completion_ablation']; scarcity = subsets['scarcity']
    approved = {r['protein_id'] for r in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    if len(ids) != 20 or len(scarcity) != 10 or not set(scarcity).issubset(ids) or not set(ids).issubset(approved):
        raise ValueError('reviewed final control roster mismatch')
    paths = ['analysis/recomb_benign_final_controls_20261017.py', 'analysis/recomb_gap_frequency_control_20261017.py',
        'analysis/recomb_general100_mechanisms_20261017.py', 'analysis/delepref_v8_main.py',
        'analysis/delepref_v8_runtime.py', 'analysis/delepref_unlimited_planning.py', 'analysis/delepref_unlimited_scorer.py',
        'reports/recomb_benign_final_subsets_20261017/subsets.json', 'reports/recomb_benign_scope_20261017/general_panel.json',
        'retrieval/recomb_benign_general_20261017/profiles.json', 'configs/recomb_benign_plans_freeze_20261017.json',
        'reports/recomb_benign_baseline_extension_20261017/scisor/all_conditions.json',
        'reports/recomb_benign_baseline_extension_20261017/scisor/completion.json']
    for folder in ['recomb_benign_plans_20261017', 'recomb_benign_mechanisms_20261017']:
        for name in ['all_selections.json', 'candidate_scores.json']:
            paths.append('reports/'+folder+'/'+name)
    paths += ['reports/recomb_benign_mechanisms_20261017/diagnostics.json',
              'reports/recomb_benign_mechanisms_20261017/completion.json',
              'reports/recomb_benign_plans_20261017/planning_completion.json',
              'reports/delepref_v8_main/encoding_completion.json']
    (root/BASE).mkdir(exist_ok=False)
    write(root/FREEZE, dict(status='FINAL_REVIEWED_CONTROLS_FROZEN_BEFORE_NEW_SELECTIONS',
        files={p: sha(root/p) for p in paths}, mechanism_ids=ids, scarcity_ids=scarcity,
        K=8, budget=.2, seed=13, null_seed=20261017, reuse_preexisting_matching_controls=True,
        gap_guidance='reliable per-position gap frequency; quarter-budget prefixes, Policy completion; original4+retrieved4 pool',
        new_training=False, new_encoder_forwards=0, new_structure_predictions=0, structure_prediction_allowed=False))
    print(json.dumps(dict(freeze_sha256=sha(root/FREEZE))))


def run(root, checksum):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU Slurm allocation required')
    out = root/BASE
    if (out/'execution_claim.json').exists():
        raise FileExistsError('inspect existing CPU execution; do not repeat')
    if sha(root/FREEZE) != checksum:
        raise ValueError('final controls freeze changed')
    frozen = read(root/FREEZE)
    for p,h in frozen['files'].items():
        if sha(root/p) != h:
            raise ValueError('final controls input changed: '+p)
    write(out/'execution_claim.json', dict(job_id=os.environ['SLURM_JOB_ID']))
    import torch
    import delepref_v8_main as parent
    import delepref_v8_runtime as rt
    import delepref_unlimited_planning as planning
    import delepref_unlimited_scorer as scoring
    import recomb_general100_mechanisms_20261017 as old
    import recomb_gap_frequency_control_20261017 as gap
    parent.verify(root, read(root/'configs/recomb_benign_plans_freeze_20261017.json')['parent_main_freeze_sha256'])
    allowed = {r['protein_id']: r for r in read(root/'reports/recomb_benign_scope_20261017/general_panel.json')}
    ids = frozen['mechanism_ids']; scarcity = set(frozen['scarcity_ids'])
    if not set(ids).issubset(allowed) or not scarcity.issubset(ids):
        raise ValueError('unreviewed final control query')
    cache_path = root/'reports/delepref_v8_main/wt_cache.pt'
    if sha(cache_path) != read(root/'reports/delepref_v8_main/encoding_completion.json')['cache_sha256']:
        raise ValueError('real WT cache changed')
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    for k in ['embeddings', 'sequences']:
        cache[k] = {p:v for p,v in cache[k].items() if p in ids}
    torch.set_num_threads(4)
    runtime, api = rt.load(root, device='cpu', cached_heads_only=True)
    records = [dict(protein_id=p, wildtype_sequence=allowed[p]['sequence'], budget=.2) for p in ids]
    chunk = parent.cached_chunk(api, cache, records)
    profiles = read(root/'retrieval/recomb_benign_general_20261017/profiles.json')
    groups = defaultdict(list)
    for row in read(root/'reports/recomb_benign_plans_20261017/candidate_scores.json'):
        if row['protein_id'] in ids and row['budget'] == .2 and row['generation_seed'] == 13:
            groups[(row['protein_id'],row['method'])].append(row)
    previous = root/'reports/recomb_benign_mechanisms_20261017'
    previous_done = read(previous/'completion.json')
    for n,h in previous_done['outputs'].items():
        if sha(previous/n) != h:
            raise ValueError('reused mechanism output changed')
    wanted = {(p,m) for p in ids for m in ['rag_marginal_gap','rag_cooccurrence_null','rag_no_scorer','rag_position_gap_frequency']}
    wanted |= {(p,m) for p in scarcity for m in ['rag_evidence_min8','rag_evidence_zero']}
    reused = [r for r in read(previous/'all_selections.json') if (r['protein_id'],r['method']) in wanted]
    choices = {(r['protein_id'],r['method']): r for r in reused}
    candidates = [r for r in read(previous/'candidate_scores.json') if (r['protein_id'],r['method']) in choices]
    diagnostics = [r for r in read(previous/'diagnostics.json') if (r['protein_id'],r['method']) in choices]
    logits_by_id = {}
    with torch.inference_mode():
        for pid in ids:
            original = sorted(groups[(pid,'delepref_unlimited')], key=lambda r:r['raw_candidate_rank'])
            rag = sorted(groups[(pid,'rag_unlimited')], key=lambda r:r['raw_candidate_rank'])
            template = original[0]; sequence = allowed[pid]['sequence']; seed = int(template['condition_seed'])
            embedding = chunk.encoded.embeddings[chunk.encoded.key_for(pid,sequence)]
            logits = runtime.model.propose_from_embeddings([embedding],[.2])[0].residue_logits.detach().cpu().double().numpy()
            logits_by_id[pid] = logits
            native = [dict(spans=tuple(map(tuple,json.loads(r['deleted_intervals']))), kind=r['generation_kind'], row=r) for r in original]
            for method in ['rag_marginal_gap','rag_cooccurrence_null','rag_position_gap_frequency'] + (['rag_evidence_min8'] if pid in scarcity else []):
                if (pid,method) in choices:
                    continue
                if method == 'rag_position_gap_frequency':
                    retrieved, intervention = gap.proposals(logits,.2,profiles[pid])
                else:
                    if method == 'rag_marginal_gap':
                        profile,intervention = old.marginal_profile(sequence,profiles[pid],seed)
                    elif method == 'rag_cooccurrence_null':
                        profile,intervention = old.null_profile(sequence,profiles[pid],20261017+seed)
                    else:
                        reps = sorted(profiles[pid]['representatives'], key=lambda r:hashlib.sha256(('recomb-scarcity-homolog-20261017|'+r['target']).encode()).hexdigest())
                        profile = dict(profiles[pid], representatives=reps[:8])
                        intervention = dict(full_N=len(reps), reduced_N=min(8,len(reps)))
                    retrieved, attempts = planning.retrieved_proposals(logits,.2,profile,seed)
                mixed,fallback = planning.mix_candidates(native,retrieved)
                scored = [dict(r) for r in original] if fallback else scoring.score_candidates(runtime,api,chunk,parent.make_rows(api,template,mixed,len(sequence),logits))
                chosen = old.select_rows(api,sequence,scored,method)
                chosen['evidence_is_counterfactual'] = method in ('rag_marginal_gap','rag_cooccurrence_null')
                chosen['evidence_is_independently_derived'] = method == 'rag_position_gap_frequency'
                if method == 'rag_cooccurrence_null' and not intervention['effective_intervention']:
                    if chosen['deletion_mask_sha256'] != old.select_rows(api,sequence,rag,'reference')['deletion_mask_sha256']:
                        raise ValueError('unperturbed null changed winner')
                choices[(pid,method)] = chosen
                candidates.extend(dict(r,method=method) for r in scored)
                diagnostics.append(dict(protein_id=pid,method=method,intervention=intervention,
                    fallback=fallback,retrieved_proposals=len(retrieved)))
            if (pid,'rag_no_scorer') not in choices:
                choices[(pid,'rag_no_scorer')] = old.select_rows(api,sequence,rag,'rag_no_scorer',use_policy=True)
            if pid in scarcity and (pid,'rag_evidence_zero') not in choices:
                zero,flag = planning.mix_candidates(native,[])
                if not flag or [r['row'] for r in zero] != original:
                    raise ValueError('zero evidence fallback changed')
                choices[(pid,'rag_evidence_zero')] = old.select_rows(api,sequence,original,'rag_evidence_zero')
        tasks = [r for r in read(root/'reports/recomb_benign_baseline_extension_20261017/scisor/all_conditions.json') if r['mode']=='same_scorer_K8']
        old_ids = {r['protein_id'] for r in read(root/'reports/recomb_benign_mechanisms_20261017/all_selections.json')}
        if len(tasks) != 13 or {t['protein_id'] for t in tasks} != set(ids)-old_ids:
            raise ValueError('unexpected SCISOR delta K8 roster')
        scisor_candidates, scisor_choices = [], []
        for task in tasks:
            pid = task['protein_id']; sequence = allowed[pid]['sequence']
            if task['sequence'] != sequence or len(task['candidates']) != 8:
                raise ValueError('SCISOR WT/K changed')
            template = dict(groups[(pid,'delepref_unlimited')][0], method='SCISOR_K8_same_scorer')
            items = [dict(spans=c['deleted_intervals'],kind='scisor_native_sampling') for c in task['candidates']]
            rows = parent.make_rows(api,template,items,len(sequence),logits_by_id[pid])
            for row,c in zip(rows,task['candidates']):
                mask = planning.validate_plan(len(sequence),.2,c['deleted_intervals'])
                edited = ''.join(a for a,d in zip(sequence,mask) if not d)
                if edited != c['edited_sequence'] or row['deletion_mask_sha256'] != c['deletion_mask_sha256']:
                    raise ValueError('SCISOR native reconstruction changed')
                row.update(candidate_index=c['candidate_index'],edited_sequence=edited,
                    edited_sequence_sha256=c['edited_sequence_sha256'],decoder='SCISOR_native_explicit_preserved_indices')
            pool = scoring.score_candidates(runtime,api,chunk,rows)
            best = dict(min(pool,key=lambda r:r['scorer_rank']), effective_unique_K=len({r['deletion_mask_sha256'] for r in pool}))
            if best['effective_unique_K'] != 8:
                raise ValueError('SCISOR effective K differs')
            scisor_candidates.extend(pool); scisor_choices.append(best)
    if set(choices) != wanted or len(choices) != 100:
        raise ValueError('incomplete final mechanism matrix')
    selections = [choices[k] for k in sorted(choices)]
    outputs = {'all_selections.json': selections, 'candidate_scores.json': candidates, 'diagnostics.json': diagnostics,
        'scisor_delta_selections.json': scisor_choices, 'scisor_delta_candidate_scores.json': scisor_candidates}
    for name,data in outputs.items():
        write(out/name,data)
    report = dict(status='FINAL_SCREENED_CPU_CONTROLS_COMPLETE',WT=20,scarcity_WT=10,
        mechanism_selections=100,reused_mechanism_selections=len(reused),new_mechanism_selections=100-len(reused),
        per_position_gap_frequency_WT=20,SCISOR_delta_WT=13,SCISOR_delta_candidates=104,
        effective_null_WT=sum(d['intervention'].get('effective_intervention',False) for d in diagnostics if d['method']=='rag_cooccurrence_null'),
        effective_min8_WT=sum(d['intervention']['full_N'] != d['intervention']['reduced_N'] for d in diagnostics if d['method']=='rag_evidence_min8'),
        no_effective_intervention_is_not_mechanism_evidence=True,new_encoder_forwards=0,new_sampler_calls=0,new_structure_predictions=0,
        freeze_sha256=checksum,job_id=os.environ['SLURM_JOB_ID'],outputs={name:sha(out/name) for name in outputs})
    write(out/'completion.json',report); print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013'))
    p.add_argument('--prepare',action='store_true'); p.add_argument('--freeze-sha256'); a=p.parse_args()
    prepare(a.root) if a.prepare else run(a.root,a.freeze_sha256)
