"""Read-only semantic audit of the complete fixed 100-WT selection outputs."""
from collections import defaultdict,Counter
import hashlib
import json
import math
from pathlib import Path

import delepref_unlimited_planning as planning

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def audit(root=ROOT):
    out=root/'reports/delepref_v8_main';prep=root/'reports/delepref_v8_evaluation_preparation'
    complete=json.loads((out/'planning_completion.json').read_text())
    freeze_path=root/'configs/delepref_v8_main_freeze.json'
    if sha(freeze_path)!=complete['freeze_sha256']:raise ValueError('main freeze changed')
    freeze=json.loads(freeze_path.read_text())
    for p,h in {**freeze['source_hashes'],**freeze['inputs']}.items():
        if sha(root/p)!=h:raise ValueError('main frozen input/source changed: '+p)
    for name,h in complete['outputs'].items():
        if sha(out/name)!=h:raise ValueError('main output changed: '+name)
    encoded=json.loads((out/'encoding_completion.json').read_text())
    if sha(out/'wt_cache.pt')!=encoded['cache_sha256'] or encoded['WT']!=100 or encoded['mutant_encodings']:
        raise ValueError('main real cache identity/count mismatch')
    panel={x['protein_id']:x for x in json.loads((root/'reports/delepref_v8_native_panel/main_panel.json').read_text())}
    subsets=json.loads((prep/'subsets.json').read_text())
    profiles=json.loads((root/'retrieval/delepref_v8_main/profiles.json').read_text())
    events={pid:{h['target']:set(h['events']) for h in planning.natural_events(profiles[pid],len(row['sequence']))} for pid,row in panel.items()}
    expected=set()
    for pid in panel:
        expected.update((pid,.2,seed,m) for seed in (13,29) for m in ('delepref_unlimited','rag_unlimited','msa_conservation','terminal'))
        if pid in subsets['secondary_10pct']:expected.update((pid,.1,13,m) for m in ('delepref_unlimited','rag_unlimited'))
        if pid in subsets['completion_ablation']:expected.add((pid,.2,13,'rag_direct_only'))
    rows=json.loads((out/'candidate_scores.json').read_text());groups=defaultdict(list)
    for row in rows:
        key=(row['protein_id'],float(row['budget']),int(row['generation_seed']),row['method'])
        if key not in expected:raise ValueError('unexpected scored condition')
        sequence=panel[key[0]]['sequence']
        if row['wildtype_sequence']!=sequence or row['sequence_sha256']!=panel[key[0]]['sequence_sha256']:
            raise ValueError('WT sequence mismatch')
        spans=tuple(tuple(x) for x in json.loads(row['deleted_intervals']))
        mask=planning.validate_plan(len(sequence),key[1],spans)
        if hashlib.sha256(mask.astype('uint8').tobytes()).hexdigest()!=row['deletion_mask_sha256']:
            raise ValueError('mask identity mismatch')
        if row['span_count']!=len(spans) or row['qmax'] is not None or row['deleted_length']!=int(mask.sum()):
            raise ValueError('span/budget metadata mismatch')
        if not math.isfinite(row['scorer_score']) or row['scorer_span_extrapolation']!=(len(spans)>3):
            raise ValueError('Scorer output/extrapolation mismatch')
        if row['generation_kind'].startswith('retrieval'):
            seed=tuple(tuple(s) for s in row['retrieval_seed_spans'])
            if not seed or not set(seed)<=events[key[0]].get(row['retrieval_source'],set()):
                raise ValueError('untraceable retrieval source/seed')
            for a,b in seed:
                if not mask[a:b+1].all() or mask[a-1] or mask[b+1]:raise ValueError('seed/anchor violation')
        groups[key].append(row)
    if set(groups)!=expected:raise ValueError('missing score group')
    for key,pool in groups.items():
        if len(pool)!=8 or len({r['deletion_mask_sha256'] for r in pool})!=8:raise ValueError('nonunique/incomplete K')
        order=sorted(pool,key=lambda r:(-r['scorer_score'],tuple(map(tuple,json.loads(r['deleted_intervals'])))))
        if [r['scorer_rank'] for r in order]!=list(range(1,9)):raise ValueError('Scorer ranks not reproducible')
    selected=json.loads((out/'all_selections.json').read_text());selected_groups={}
    for row in selected:
        key=(row['protein_id'],float(row['budget']),int(row['generation_seed']),row['method'])
        if key in selected_groups:raise ValueError('duplicate selected group')
        selected_groups[key]=row
        best=min(groups[key],key=lambda r:r['scorer_rank'])
        if row['deletion_mask_sha256']!=best['deletion_mask_sha256'] or row['scorer_score']!=best['scorer_score']:
            raise ValueError('selection differs from frozen Scorer rank1')
        sequence=panel[key[0]]['sequence'];mask=planning.mask_for(len(sequence),json.loads(row['deleted_intervals']))
        edited=''.join(aa for aa,deleted in zip(sequence,mask) if not deleted)
        if edited!=row['edited_sequence'] or hashlib.sha256(edited.encode()).hexdigest()!=row['edited_sequence_sha256']:
            raise ValueError('edited sequence reconstruction failed')
    if set(selected_groups)!=expected or len(rows)!=7200 or len(selected)!=900:raise ValueError('full roster/count mismatch')
    diagnostics=json.loads((out/'generation_diagnostics.json').read_text())
    if len(diagnostics)!=240 or any(not d['counterfactual_fallback_exact'] for d in diagnostics):raise ValueError('counterfactual fallback audit missing')
    for d in diagnostics:
        if not d['fallback']:continue
        common=(d['protein_id'],float(d['budget']),int(d['seed']))
        a=groups[common+('delepref_unlimited',)];b=groups[common+('rag_unlimited',)]
        if [(r['deletion_mask_sha256'],r['scorer_score']) for r in a]!=[(r['deletion_mask_sha256'],r['scorer_score']) for r in b]:
            raise ValueError('actual fallback differs from unrestricted baseline')
    final=json.loads((out/'primary_secondary_ablation_selections.json').read_text())
    second=json.loads((out/'second_predictor_selections.json').read_text())
    if len(final)!=500 or len(second)!=32:raise ValueError('structure-evaluation subset count mismatch')
    expected_final=[r for r in selected if r['generation_seed']==13]
    if final!=expected_final:raise ValueError('CPU stability output entered structural evaluation')
    expected_second={(pid,m) for pid in subsets['second_predictor'] for m in ('delepref_unlimited','rag_unlimited','terminal','WT')}
    if {(r['protein_id'],r['method']) for r in second}!=expected_second:raise ValueError('second predictor roster changed')
    changed=0;origins=Counter()
    for pid in panel:
        a=selected_groups[(pid,.2,13,'delepref_unlimited')];b=selected_groups[(pid,.2,13,'rag_unlimited')]
        if a['deletion_mask_sha256']!=b['deletion_mask_sha256']:
            changed+=1;origins['retrieval' if b['generation_kind'].startswith('retrieval') else 'other_original_candidate']+=1
    report=dict(status='MAIN_100WT_OUTPUT_SEMANTICS_VERIFIED',WT=100,score_groups=len(groups),scored_candidates=len(rows),selected_outputs=len(selected),
        primary_changed_choices=changed,changed_choice_origins=dict(origins),
        primary_retrieval_selected=sum(selected_groups[(pid,.2,13,'rag_unlimited')]['generation_kind'].startswith('retrieval') for pid in panel),
        max_candidate_spans=max(r['span_count'] for r in rows),all_sources_anchors_budgets_masks_ranks_reconstruction_pass=True,
        all_counterfactual_and_actual_fallback_checks_pass=True,source_freeze_sha256=sha(freeze_path),
        planning_completion_sha256=sha(out/'planning_completion.json'),audit_source_sha256=sha(Path(__file__)),
        primary_secondary_ablation_outputs=500,second_predictor_outputs=32,CPU_stability_outputs=400,
        new_structure_predictions=0,structural_quality_measured=False,biological_function_validated=False,
        structural_prediction_allowed=False,paper_changes=False)
    with (out/'semantic_audit.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':audit()
