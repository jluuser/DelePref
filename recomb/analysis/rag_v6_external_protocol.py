"""CPU-only external data protocol and training-homology audit.

No model is fitted, no protein is proposed, and no phenotype-ranked edits are
reported. Original source rows remain immutable and separately reproducible.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

import rag_v6_retrieval as r

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
GYM_TARGET={'DMS_ind_3':'B1LPA6','DMS_ind_6':'P62593','DMS_ind_40':'P60484','DMS_ind_61':'O95071'}
CONDITIONS=['score_30','score_30_MTX','score_37','score_37_MTX']
POLICY=dict(mapping_ambiguity='arithmetic_mean_over_all_exact_single_deletion_maps',
    mapping_weights='uniform_independent_of_experimental_or_predicted_score',
    primary_duplicate_release='ProteinGym_DMS_ind_6;MaveDB_00000100_b_1_legacy_only',
    primary_observation_unit='one_protein_sequence_per_assay_condition',
    replicate_measurements='arithmetic_mean_of_published_scores_within_same_sequence_and_condition',
    correlation='Spearman_average_ranks;undefined_constant_or_singleton_explicit',
    aggregation='equal_assay_condition_within_protein_then_equal_protein',
    uncertainty='paired_physical_sequence_cluster_bootstrap_within_assay;protein_macro_reports_4_eligible_targets',
    output='aggregate_assay_and_target_metrics_only_no_mutant_recommendations',
    no_score_sign_selection_by_correlation=True,no_new_prediction_or_training=True,
    no_homology_based_row_filtering=True,legacy_1645_view_retained=True)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def direction_audit(base):
    sources=json.loads((base/'direction_sources/provenance.json').read_text())
    for name,meta in sources['files'].items():r.verify_hashes(base,{str(base/'direction_sources'/name):meta['sha256']})
    authors=list(csv.DictReader((base/'direction_sources/cpop_data.csv').open(encoding='utf-8-sig')))
    deletion={int(float(x['pos'])):x for x in authors if x['mut_type']=='del'}
    matches=[]
    for i,column in enumerate(CONDITIONS,1):
        urn=f'urn:mavedb:00001219-a-{i}';data=list(csv.DictReader((base/'mavedb'/(urn.replace(':','_')+'.csv')).open()))
        errors=[]
        for row in data:
            found=re.fullmatch(r'p\.[A-Za-z*]{1,3}([0-9]+)del',row['hgvs_pro'])
            if not found:continue
            try:a=float(row['score']);b=float(deletion[int(found.group(1))][column])
            except ValueError:continue
            if math.isfinite(a) and math.isfinite(b):errors.append(abs(a-b))
        if not errors or max(errors)>1e-10:raise ValueError('public MaveDB values differ from the authors folding-score column')
        controls={}
        for kind in ['syn','stop']:
            values=[]
            for row in authors:
                if row['mut_type']!=kind:continue
                try:value=float(row[column])
                except ValueError:continue
                if math.isfinite(value):values.append(value)
            controls[kind]=dict(mean=float(np.mean(values)),count=len(values))
        matches.append(dict(urn=urn,source_column=column,score_direction=1,
            paired_finite_values=len(errors),max_absolute_difference=max(errors),controls=controls,
            endpoint='CPOP_folding_sensor_in_its_reported_condition_not_direct_catalytic_activity'))
    metadata=json.loads((base/'mavedb/urn_mavedb_00000100-b-1.json').read_text())
    statement='midpoint ampicillin concentration for resistance'
    if statement not in metadata['methodText']:raise ValueError('expected original assay interpretation missing')
    return dict(ProteinGym=dict(direction=1,field='processed_DMS_score',do_not_reapply_raw_directionality=True,
                source='https://github.com/OATML-Markslab/ProteinGym'),
                DHFR=dict(direction=1,conditions=matches,source_commit=sources['commit'],
                          source='https://github.com/KULL-Centre/_2024_Larsen-Ledet_CPOP'),
                MaveDB_TEM1=dict(direction=1,endpoint='normalized_experimental_fitness',legacy_duplicate_release_only=True,
                    source='https://api.mavedb.org/api/v1/score-sets/urn:mavedb:00000100-b-1'),
                inferred_from_model_performance=False)


def canonical_rows(gym,mavedb):
    original=[]
    for old in gym:
        original.append(dict(row_id=old['row_id'],assay_id=old['DMS_index'],source_assay=old['DMS_id'],
            target=GYM_TARGET[old['DMS_index']],wt_sequence=old['wt_sequence'],sequence=old['sequence'],
            possible_deleted_positions=old['possible_deleted_positions'],measurement=float(old['DMS_score']),source='ProteinGym'))
    for old in mavedb:
        original.append(dict(row_id=old['row_id'],assay_id=old['assay_id'],source_assay=old['assay_id'],
            target=old['UniProt_accession'],wt_sequence=old['wt_sequence'],sequence=old['sequence'],
            possible_deleted_positions=old['possible_deleted_positions'],measurement=float(old['experimental_score']),source='MaveDB'))
    if len(original)!=1645 or len({x['row_id'] for x in original})!=1645:raise ValueError('legacy roster size or row identities differ')
    for row in original:
        wt,edited=row['wt_sequence'],row['sequence']
        positions=sorted(set(row['possible_deleted_positions']))
        if len(edited)!=len(wt)-1 or not positions or not math.isfinite(row['measurement']):raise ValueError('invalid published observation')
        if any(not 0<=i<len(wt) or wt[:i]+wt[i+1:]!=edited for i in positions):raise ValueError('reported deletion map is inconsistent')
        row['possible_deleted_positions']=positions
    gym_t={x['sequence'] for x in original if x['assay_id']=='DMS_ind_6'}
    mave_t={x['sequence'] for x in original if x['assay_id']=='urn:mavedb:00000100-b-1'}
    if gym_t!=mave_t or len(gym_t)!=258:raise ValueError('duplicate-release physical sequence sets differ')
    return original,physical_rows(original)


def physical_rows(original):
    physical={}
    for row in original:
        if row['assay_id']=='urn:mavedb:00000100-b-1':continue
        key=(row['assay_id'],row['wt_sequence'],row['sequence'])
        physical.setdefault(key,[]).append(row)
    result=[]
    for (assay,wt,edited),members in sorted(physical.items()):
        # All exact maps are enumerated from sequence identity, independent of labels.
        maps=[i for i in range(len(wt)) if wt[:i]+wt[i+1:]==edited]
        identity=hashlib.sha256((assay+'\0'+wt+'\0'+edited).encode()).hexdigest()
        result.append(dict(row_id=identity,assay_id=assay,target=members[0]['target'],
            wt_sequence=wt,wt_sequence_sha256=hashlib.sha256(wt.encode()).hexdigest(),
            sequence=edited,sequence_sha256=hashlib.sha256(edited.encode()).hexdigest(),
            possible_deleted_positions=maps,measurement=float(np.mean([x['measurement'] for x in members])),
            source_row_ids=[x['row_id'] for x in members],source_observations=len(members),
            score_direction=1,mapping_aggregation='uniform_mean',usage='aggregate_retrospective_evaluation_only'))
    return result


def homology_audit(root,rows):
    dispatch=json.loads((root/'reports/dispatch_after_pilot.json').read_text())
    code=Path(dispatch['code_snapshot']);r.verify_hashes(code,dispatch['code_hashes'])
    sys.path.insert(0,str(code))
    from split_families import make_aligner,similarity,threshold_hit
    protocol=json.loads((root/'configs/protocol_v2.json').read_text());cfg=protocol['split']
    split=json.loads((root/'manifests/splits.json').read_text())
    train={x['protein_id']:x['wt_sequence'] for x in read_rows(root/'manifests/development_primary.jsonl') if split['assignment'][x['protein_id']]=='train'}
    if len(train)!=50:raise ValueError('expected the fixed 50 training WTs')
    queries={x['target']:x['wt_sequence'] for x in rows}
    if len(queries)!=5:raise ValueError('expected five external targets')
    for target in queries:
        if len({x['wt_sequence'] for x in rows if x['target']==target})!=1:raise ValueError('multiple target sequence backgrounds must be stratified')
    aligner=make_aligner(cfg['alignment']);results=[]
    for target,sequence in sorted(queries.items()):
        pairs=[]
        for pid,other in sorted(train.items()):
            stat=similarity(sequence,other,aligner)
            alignment=aligner.align(sequence,other)[0]
            both=(alignment.indices[0]>=0)&(alignment.indices[1]>=0)
            stat['aligned_pair_identity']=stat['matches']/max(1,int(both.sum()))
            stat['aligned_residue_pairs']=int(both.sum())
            pairs.append(dict(training_WT=pid,**stat,threshold_hit=threshold_hit(stat,cfg)))
        nearest=max(pairs,key=lambda x:x['identity'])
        covered=[x for x in pairs if min(x['coverage_a'],x['coverage_b'])>=cfg['coverage_each']]
        results.append(dict(target=target,WT_length=len(sequence),nearest_any_coverage=nearest,
            nearest_both_covered=max(covered,key=lambda x:x['identity']) if covered else None,
            threshold_hits=[x for x in pairs if x['threshold_hit']],all_comparisons=pairs))
    return dict(training_WT=50,external_targets=5,comparisons=250,alignment=cfg['alignment'],
                identity_denominator=cfg['identity_denominator'],threshold=dict(identity=cfg['identity'],coverage_each=cfg['coverage_each']),
                targets=results,targets_with_threshold_hits=sum(bool(x['threshold_hits']) for x in results),
                rows_removed_or_model_modified=False,
                limitation='Sequence-threshold audit does not prove absence of remote homology or shared domains; short/full-domain coverage is reported separately.')


def baseline_audit(root,rows):
    known=[Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/DelePref/SCISOR'),
      Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/SCISOR'),
      Path('/public/nfs-data/keyuan.zhou/evocompress/tools/SCISOR'),
      Path('/public/nfs-data/keyuan.zhou/evocompress/tools/Tranception'),
      Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/Tranception'),
      Path('/public/nfs-data/keyuan.zhou/evocompress/weights/tranception'),
      Path('/public/nfs-data/keyuan.zhou/evocompress/models/Tranception_Large')]
    wanted={hashlib.sha256(x[k].encode()).hexdigest() for x in rows for k in ['wt_sequence','sequence']}
    available=set()
    for prefix in ['full','rag_v6_confirmation']:
        for shard in (0,1):
            path=root/f'caches/embeddings/{prefix}_shard{shard}_index.jsonl'
            if path.exists():available.update(x['sequence_sha256'] for x in read_rows(path))
    return dict(U=dict(code_ready=True,frozen_weights_available=True,external_scores_ready=False,
                unique_external_sequences=len(wanted),existing_embedding_index_matches=len(wanted&available),
                missing_sequence_encodings=len(wanted-available),new_encodings_started=False),
                learned_v6=dict(model_frozen=True,external_retrieval_profiles_available=False,external_scores_ready=False),
                competitive=dict(checked_paths=[dict(path=str(p),exists=p.exists()) for p in known],
                    no_global_search=True,no_large_download=True,absence_is_bounded_path_audit_only=True,
                    full_matched_prediction_matrix_available=False),
                original_full_DelePref=dict(checkpoint_restored=False,search_not_repeated_per_user=True),
                interpretation='Data and scorer interface prepared; no completed external model comparison is claimed.')


def main():
    root=ROOT;base=root/'external/rag_v6';out=base/'preparation';out.mkdir(exist_ok=False)
    source_reports=['legacy_component_report.json','mavedb_component_report.json']
    for report_name,rows_name in zip(source_reports,['legacy_proteingym_single_deletion_rows.jsonl','legacy_mavedb_single_deletion_rows.jsonl']):
        report=json.loads((base/report_name).read_text());r.verify_hashes(base,{rows_name:report['records_sha256']})
    directions=direction_audit(base)
    original,primary=canonical_rows(read_rows(base/'legacy_proteingym_single_deletion_rows.jsonl'),read_rows(base/'legacy_mavedb_single_deletion_rows.jsonl'))
    path=out/'primary_rows.jsonl'
    with path.open('x',encoding='utf-8') as handle:
        for row in primary:handle.write(json.dumps(row,sort_keys=True)+'\n')
    homology=homology_audit(root,primary);baselines=baseline_audit(root,primary)
    r.write_json(out/'direction_audit.json',directions,atomic=True)
    r.write_json(out/'train_homology_audit.json',homology,atomic=True)
    r.write_json(out/'baseline_availability.json',baselines,atomic=True)
    counts={a:sum(x['assay_id']==a for x in primary) for a in sorted({x['assay_id'] for x in primary})}
    protocol=dict(status='CPU_EXTERNAL_PROTOCOL_PREPARED',policy=POLICY,legacy_source_entries=9,
        legacy_rows=len(original),primary_distinct_conditions=len(counts),primary_physical_rows=len(primary),
        primary_condition_counts=counts,targets=5,correlation_eligible_targets=4,
        duplicate_release=dict(ProteinGym='DMS_ind_6',MaveDB='urn:mavedb:00000100-b-1',overlap_physical_sequences=258,
                              retained_primary='ProteinGym',legacy_source_rows_unchanged=True),
        ambiguous_primary_rows=sum(len(x['possible_deleted_positions'])>1 for x in primary),
        primary_rows_sha256=r.sha(path),direction_audit_sha256=r.sha(out/'direction_audit.json'),
        homology_audit_sha256=r.sha(out/'train_homology_audit.json'),baseline_audit_sha256=r.sha(out/'baseline_availability.json'),
        implementation_sha256=r.sha(__file__),input_hashes={str(base/p):r.sha(base/p) for p in source_reports},
        frozen_main_model_modified=False,model_scores_computed=False,new_GPU_jobs=0,new_structure_calls=0,
        score_direction_chosen_using_model_correlation=False)
    r.write_json(out/'protocol.json',protocol,atomic=True)
    print(json.dumps({k:v for k,v in protocol.items() if k not in ['policy','input_hashes']}),flush=True)
    print(json.dumps({'homology_targets_with_threshold_hits':homology['targets_with_threshold_hits'],
                      'external_sequences_needing_future_encoding':baselines['U']['missing_sequence_encodings']}),flush=True)


if __name__=='__main__':main()
