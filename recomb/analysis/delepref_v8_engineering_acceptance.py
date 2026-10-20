"""Validate engineering artifacts before expanding to the 100-WT panel."""
from collections import defaultdict
import hashlib
import json
import math

import delepref_unlimited_planning as planning


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_rows(rows,records):
    expected={(r['protein_id'],float(r['budget']),seed):r['wildtype_sequence'] for r in records for seed in (13,29)}
    grouped=defaultdict(list)
    for row in rows:
        key=(row['protein_id'],float(row['budget']),int(row['generation_seed']))
        sequence=expected.get(key)
        if sequence is None or row['wildtype_sequence']!=sequence:raise ValueError('unexpected WT/condition')
        if row['sequence_sha256']!=hashlib.sha256(sequence.encode()).hexdigest():raise ValueError('WT hash mismatch')
        spans=tuple(tuple(x) for x in json.loads(row['deleted_intervals']))
        mask=planning.validate_plan(len(sequence),key[1],spans)
        digest=hashlib.sha256(mask.astype('uint8').tobytes()).hexdigest()
        if digest!=row['deletion_mask_sha256'] or int(row['span_count'])!=len(spans):raise ValueError('mask/span metadata mismatch')
        if row['qmax'] is not None or not math.isfinite(float(row['scorer_score'])):raise ValueError('capped or nonfinite output')
        grouped[key].append(row)
    if set(grouped)!=set(expected):raise ValueError('missing condition')
    for key,pool in grouped.items():
        if len(pool)!=8 or len({x['deletion_mask_sha256'] for x in pool})!=8:raise ValueError('incomplete/duplicate K')
        if sorted(int(x['scorer_rank']) for x in pool)!=list(range(1,9)):raise ValueError('invalid ranks')
    return grouped


def accept(root,checkpoint_sha):
    baseline=root/'reports/delepref_v8';rag=root/'reports/delepref_v8_rag_unlimited'
    original=json.loads((baseline/'baseline_completion.json').read_text())
    report=json.loads((rag/'rag_completion.json').read_text())
    if original['status']!='ORIGINAL_DELEPREF_REAL_INFERENCE_COMPLETE' or report['status']!='UNLIMITED_ORIGINAL_DELEPREF_AND_RAG_END_TO_END_COMPLETE':
        raise ValueError('engineering not complete')
    for data in (original,report):
        if data['checkpoint_sha256']!=checkpoint_sha or data['new_training'] or data['new_structure_predictions']:
            raise ValueError('wrong model or execution scope')
    if original['WT_encodings']!=6 or original['mutant_encodings']!=0:raise ValueError('unexpected original encodings')
    if report['conditions']!=24 or report['qmax'] is not None or not report['all_no_evidence_fallback_tests_exact']:
        raise ValueError('engineering contract failed')
    if sha(baseline/'original_wt_cache.pt')!=original['cache_sha256'] or sha(baseline/'baseline_scores.json')!=original['scores_sha256']:
        raise ValueError('original artifacts changed')
    for freeze_name,hash_field,data in [('configs/delepref_v8_baseline_freeze.json','source_freeze_sha256',original),
            ('configs/delepref_v8_rag_unlimited_freeze.json','freeze_sha256',report)]:
        path=root/freeze_name
        if sha(path)!=data[hash_field]:raise ValueError('freeze identity changed')
        freeze=json.loads(path.read_text())
        for group in ('sources','inputs','code'):
            for p,digest in freeze.get(group,{}).items():
                if sha(root/p)!=digest:raise ValueError('frozen source/input changed: '+p)
    records=json.loads((baseline/'inputs.json').read_text())
    original_rows=json.loads((rag/'original_unlimited_scores.json').read_text())
    rag_rows=json.loads((rag/'rag_scores.json').read_text())
    original_groups=validate_rows(original_rows,records);rag_groups=validate_rows(rag_rows,records)
    profiles=json.loads((root/'reports/rag_v6/profiles.json').read_text())
    for row in rag_rows:
        if not row['generation_kind'].startswith('retrieval'):continue
        source=row['retrieval_source'];seed=tuple(tuple(x) for x in row['retrieval_seed_spans'])
        events=[x for x in planning.natural_events(profiles[row['protein_id']],len(row['wildtype_sequence'])) if x['target']==source]
        if not events or not seed or not set(seed)<=set(events[0]['events']):raise ValueError('retrieval source mismatch')
        mask=planning.mask_for(len(row['wildtype_sequence']),json.loads(row['deleted_intervals']))
        for a,b in seed:
            if not mask[a:b+1].all() or mask[a-1] or mask[b+1]:raise ValueError('retrieval seed/anchor violation')
    decisions=report['decisions']
    if len(decisions)!=24 or len({(x['protein_id'],float(x['budget']),int(x['seed'])) for x in decisions})!=24:
        raise ValueError('decision roster mismatch')
    for d in decisions:
        key=(d['protein_id'],float(d['budget']),int(d['seed']))
        if key not in original_groups or not d['withheld_fallback_exact']:raise ValueError('fallback check failed')
        for groups,field in ((original_groups,'original_selected_mask'),(rag_groups,'rag_selected_mask')):
            if min(groups[key],key=lambda x:int(x['scorer_rank']))['deletion_mask_sha256']!=d[field]:
                raise ValueError('selected mask/report mismatch')
        if d['fallback'] and original_groups[key]!=rag_groups[key]:raise ValueError('fallback output changed')
    compatibility=json.loads((rag/'CPU_GPU_compatibility.json').read_text())
    if len(compatibility)!=24 or any(x['native_mask_overlap']!=8 or not math.isfinite(x['max_native_CPU_GPU_score_difference']) for x in compatibility):
        raise ValueError('incomplete CPU/GPU numerical audit')
    paths=[baseline/'baseline_completion.json',baseline/'original_wt_cache.pt',baseline/'baseline_scores.json',
        rag/'rag_completion.json',rag/'original_unlimited_scores.json',rag/'rag_scores.json',rag/'CPU_GPU_compatibility.json']
    return dict(status='REAL_SEQUENCE_ENGINEERING_ACCEPTED',WT=6,conditions=24,K=8,qmax=None,
        original_checkpoint_sha256=checkpoint_sha,validated_retrieval_sources_and_anchors=True,
        exact_fallback=True,maximum_native_CPU_GPU_score_difference=max(x['max_native_CPU_GPU_score_difference'] for x in compatibility),
        CPU_GPU_difference_is_reported_not_assumed_zero=True,
        artifacts={str(p):sha(p) for p in paths},quality_or_function_improvement_not_tested=True,
        structural_prediction_allowed=False,paper_changes=False)
