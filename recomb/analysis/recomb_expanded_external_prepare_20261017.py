"""Outcome-independent ProteinGym release, construction and deletion-map preparation."""
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT=Path('/public/nfs-data/keyuan.zhou/delepref_structural_response_recomb_20261013')
BASE='external/recomb_expanded_20261017'
AA=set('ACDEFGHIKLMNPQRSTVWY')
MAP_CAP=4096


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def seqsha(s):return hashlib.sha256(s.encode()).hexdigest()


def deletion_maps(wt,mut,cap=MAP_CAP):
    """All legal coordinate deletions, or a declared capped category, never best-score alignment."""
    D=len(wt)-len(mut);L=len(wt);M=len(mut)
    if not 0<D<L:return dict(status='not_shorter',maps=[],map_count=0)
    left=[];p=0
    for a in mut:
        p=wt.find(a,p)
        if p<0:return dict(status='not_subsequence',maps=[],map_count=0)
        left.append(p);p+=1
    right=[];p=L
    for a in mut[::-1]:p=wt.rfind(a,0,p);right.append(p)
    right.reverse()
    if left==right:
        kept=set(left)
        return dict(status='unique',maps=[[i for i in range(L) if i not in kept]],map_count=1)
    sys.setrecursionlimit(max(10000,4*L))
    @lru_cache(None)
    def count(i,d):
        j=i-d
        if d>D or j>M or j<0:return 0
        if i==L:return int(d==D and j==M)
        n=count(i+1,d+1) if d<D else 0
        if j<M and wt[i]==mut[j]:n+=count(i+1,d)
        return min(cap+1,n)
    n=count(0,0)
    if n>cap:return dict(status='ambiguous_over_cap',maps=[],map_count_lower_bound=cap+1,map_count=None)
    stack=[(0,0,())];maps=[]
    while stack:
        i,d,deleted=stack.pop();j=i-d
        if i==L:
            if d==D and j==M:maps.append(list(deleted))
            continue
        if d<D and count(i+1,d+1):stack.append((i+1,d+1,deleted+(i,)))
        if j<M and wt[i]==mut[j] and count(i+1,d):stack.append((i+1,d,deleted))
    assert len(maps)==n and len({tuple(x) for x in maps})==n
    return dict(status='ambiguous_enumerated',maps=sorted(maps),map_count=n)


def spans(deleted):
    result=[]
    for p in deleted:
        if result and p==result[-1][1]+1:result[-1][1]=p
        else:result.append([p,p])
    return result


def write(path,value):
    p=ROOT/path;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    out=ROOT/BASE;out.mkdir(parents=True,exist_ok=True)
    if (out/'cohort_freeze.json').exists():raise FileExistsError('expanded cohort already frozen')
    ref=ROOT/'provenance/recomb_literature_20261017/ProteinGym_PG_v1_3_DMS_indels_reference.csv'
    assert sha(ref)=='09534e16eeeb80ce22243f6cada151db654c7330117f2a97980f6707da87c7cd'
    with ref.open() as f:metadata={r['DMS_id']:r for r in csv.DictReader(f)}
    archive=ROOT/'external/rag_v6/DMS_ProteinGym_indels.zip'
    grouped=defaultdict(list);reasons=Counter();map_cache={};assays={};wildtypes={};all_seq=set()
    with zipfile.ZipFile(archive) as z:
        for member in sorted(n for n in z.namelist() if n.endswith('.csv')):
            assay=Path(member).stem;meta=metadata[assay];wt=meta['target_seq'];target='PG_'+seqsha(wt)[:20]
            for index,r in enumerate(csv.DictReader(io.TextIOWrapper(z.open(member),encoding='utf-8-sig'))):
                mut=r['mutated_sequence']
                try:value=float(r['DMS_score'])
                except (TypeError,ValueError):reasons['invalid_measurement']+=1;continue
                if not math.isfinite(value):reasons['invalid_measurement']+=1;continue
                if not wt or not mut or not set(wt).issubset(AA) or not set(mut).issubset(AA):reasons['noncanonical']+=1;continue
                if len(mut)>=len(wt):reasons['not_shorter']+=1;continue
                key=(wt,mut)
                if key not in map_cache:map_cache[key]=deletion_maps(wt,mut)
                mapped=map_cache[key]
                if mapped['status']=='not_subsequence':reasons['not_subsequence']+=1;continue
                reasons['compatible_raw_rows']+=1
                assays[assay]=meta;wildtypes[target]=wt;all_seq.update((wt,mut))
                binary=r.get('DMS_score_bin')
                if binary not in (None,'','0','1','0.0','1.0'):raise ValueError('unexpected official bin label')
                grouped[(assay,mut)].append(dict(value=value,binary=None if binary in (None,'') else int(float(binary)),source_row=index))
    rows=[];edits={};mapping_hist=Counter();binary_disagreements=0
    for (assay,mut),observations in sorted(grouped.items()):
        meta=assays[assay];wt=meta['target_seq'];target='PG_'+seqsha(wt)[:20];mapped=map_cache[(wt,mut)]
        rowid=seqsha(assay+'\0'+mut);mapkeys=[]
        for deleted in mapped['maps']:
            intervals=spans(deleted);edit_key=seqsha(wt+'\0'+json.dumps(intervals,separators=(',',':')))
            edits.setdefault(edit_key,dict(edit_key=edit_key,target=target,wt_sequence=wt,mutant_sequence=mut,
                deleted_positions=deleted,deleted_intervals=intervals,deleted_length=len(deleted),budget=len(deleted)/len(wt)))
            mapkeys.append(edit_key)
        bins={o['binary'] for o in observations}
        binary=next(iter(bins)) if len(bins)==1 else None
        binary_disagreements+=int(len(bins)>1)
        rows.append(dict(row_id=rowid,assay_id=assay,target=target,UniProt_ID=meta['UniProt_ID'],
            wt_sha256=seqsha(wt),mutant_sequence=mut,mutant_sha256=seqsha(mut),deleted_length=len(wt)-len(mut),
            deletion_fraction_full_reference=(len(wt)-len(mut))/len(wt),selection_type=meta['selection_type'],
            measurement=sum(o['value'] for o in observations)/len(observations),official_binary=binary,
            replicate_rows=len(observations),source_rows=[o['source_row'] for o in observations],
            mapping_status=mapped['status'],map_count=mapped['map_count'],edit_keys=mapkeys,
            primary_unique_mapping=mapped['status']=='unique',
            span_count= len(edits[mapkeys[0]]['deleted_intervals']) if mapped['status']=='unique' else None))
        mapping_hist[mapped['status']]+=1
    protocol=dict(status='COHORT_AND_ANALYSIS_RULES_FROZEN_BEFORE_MODEL_SCORES',release='PG_v1.3',
        reference_sha256=sha(ref),archive_sha256=sha(archive),
        primary='All strict-subsequence-compatible unique-coordinate variants; all phenotype categories separately',
        secondary='Ambiguous variants: average all legal coordinate scores equally if enumeration<=4096; larger sets explicitly missing for coordinate models, sequence-only scores may remain defined',
        ambiguity_limit=MAP_CAP,ambiguity_limit_basis='Fixed computational enumeration cap, unrelated to labels or predictions',
        measurement='Released DMS_score already direction-adjusted; do not flip again using raw_DMS_directionality',
        desired_function_caution='Predict assay direction, not universal biological benefit; aggregation is a distinct category',
        duplicates='Within assay and identical complete mutant sequence: arithmetic mean of released numeric measurements; keep replicate counts and binary disagreements',
        labels='Only official released DMS_score_bin; disagreements/absent labels undefined',
        metrics=['assay/WT/category Spearman','same-deletion-length Spearman','NDCG with within-assay rank-percentile relevance','AUROC when official binary labels contain both classes'],
        uncertainty='Protein/WT-group bootstrap10000 seed20261017; repeated sequence rows grouped; report conditional within-assay intervals separately',
        undefined='Constants, singletons and incomplete coverage retained; no model-specific target deletion',
        leakage='Audit all WT against actual269915 training WT; report exposure strata without deleting cohort',
        generation_evidence=False,score_or_label_driven_mapping=False,maximum_encoder_length=1022)
    write(BASE+'/protocol.json',protocol);write(BASE+'/assay_metadata.json',assays)
    write(BASE+'/inputs.json',dict(wildtypes=wildtypes,edits=list(edits.values())))
    with (out/'rows.jsonl').open('x') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    with (out/'wildtypes.fasta').open('x') as f:
        for target,seq in sorted(wildtypes.items()):f.write('>'+target+'\n'+seq+'\n')
    write(BASE+'/retrieval_exclusion_sha256.json',sorted(seqsha(s) for s in all_seq))
    report=dict(status='EXPANDED_COHORT_MAPPING_FROZEN_PENDING_EXPOSURE_AND_BASELINE_AUDITS',
        raw_reason_counts=dict(reasons),observations_after_sequence_dedup=len(rows),assays=len(assays),WT=len(wildtypes),
        unique_edit_mappings=len(edits),mapping_categories=dict(mapping_hist),binary_disagreement_groups=binary_disagreements,
        unique_WT_variant_pairs=len(map_cache)-sum(r['status']=='not_subsequence' for r in map_cache.values()),
        new_training=False,new_model_forwards=0,
        files={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    write(BASE+'/cohort_freeze.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
