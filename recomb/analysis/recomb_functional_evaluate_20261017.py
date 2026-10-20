"""Annotation retention and local native relationships, reusing all existing predictions."""
from collections import defaultdict,Counter
import json
from pathlib import Path
import sys
import numpy as np
from recomb_expanded_external_prepare_20261017 import ROOT,sha,write
sys.path.insert(0,str(ROOT/'code'))
from build_pairs import reference_pairs,reference_preservation_labels,preservation_bounds
BASE='reports/recomb_functional_annotations_20261017'


def read(path):return json.loads((ROOT/path).read_text())


def retention(annotation,deleted):
    features=[];by_type=defaultdict(set)
    for feature in annotation['features']:
        positions=feature['WT_positions_0based'];lost=int(deleted[positions].sum())
        by_type[feature['type']].update(positions)
        features.append(dict(type=feature['type'],description=feature['description'],positions=len(positions),deleted=lost,
            retained_fraction=1-lost/len(positions),canonical_feature_fully_mapped=feature['canonical_feature_positions']==len(positions),
            removal_state='none' if lost==0 else ('all_mapped_positions' if lost==len(positions) else 'partial_mapped_positions')))
    return dict(mapping_status=annotation['mapping_status'],annotation_present=bool(features),features=features,
                residue_retention_by_type={k:1-float(deleted[sorted(v)].mean()) for k,v in by_type.items()})


def main():
    completion=read(BASE+'/completion.json');path=ROOT/BASE/'mapped_features.json'
    assert sha(path)==completion['mapped_features_sha256']
    annotations={r['protein_id']:r for r in read(BASE+'/mapped_features.json')}
    panel={r['protein_id']:r for r in read('reports/delepref_v8_native_panel/main_panel.json')}
    generation=read('reports/recomb_task_budget_20261017/all_selections.json')
    generation_rows=[]
    for x in generation:
        L=len(panel[x['protein_id']]['sequence']);deleted=np.zeros(L,dtype=bool)
        for a,b in json.loads(x['deleted_intervals']):deleted[a:b+1]=True
        generation_rows.append(dict(protein_id=x['protein_id'],method=x['method'],budget=x['budget'],seed=x['generation_seed'],
                                   **retention(annotations[x['protein_id']],deleted)))
    local=[];native_cache={}
    for x in read('reports/delepref_v8_structure_evaluation/per_output_metrics.json'):
        pid=x['protein_id'];wt=panel[pid];L=len(wt['sequence']);annotation=annotations[pid]
        sites=sorted({p for f in annotation['features'] if f['type'] in ('Active site','Binding site','Disulfide bond','DNA binding')
                      for p in f['WT_positions_0based']})
        deleted=np.ones(L,dtype=bool);deleted[x['kept']]=False
        row=dict(protein_id=pid,method=x['method'],budget=x['budget'],predictor=x['predictor'],role=x['role'],
                 **retention(annotation,deleted),functional_native_neighborhood=None,junction_retained_neighborhood=None)
        if sites:
            if pid not in native_cache:
                assert sha(wt['wt_native_json'])==wt['wt_native_json_sha256']
                native=np.full((L,3),np.nan)
                for k,v in json.loads(Path(wt['wt_native_json']).read_text())['coordinates'].items():native[int(k)]=v
                native_cache[pid]=(native,reference_pairs(native))
            native,(i,j)=native_cache[pid]
            pred_path=ROOT/'reports/delepref_v8_structures/artifacts'/(x['physical_id']+'.npz')
            record=read('reports/delepref_v8_structures/records/'+x['physical_id']+'.json')
            assert sha(pred_path)==record['output_sha256']
            with np.load(pred_path,allow_pickle=False) as a:coords=a['coordinates'].copy()
            distances=np.linalg.norm(native[:,None,:]-native[sites][None,:,:],axis=-1)
            near=np.any(distances<8.,axis=1)
            pairs=near[i]|near[j]
            if pairs.any():
                labels=reference_preservation_labels(native,coords,x['kept'],i[pairs],j[pairs])
                row['functional_native_neighborhood']=dict(preservation_bounds(labels,int(pairs.sum())),
                    scope='Fixed WT native CA radius<8A around mapped sites, pair cutoff15A/sep>=3; common full regional denominator')
            junction=set()
            spans=[];ids=np.flatnonzero(deleted)
            for p in ids:
                if spans and p==spans[-1][1]+1:spans[-1][1]=int(p)
                else:spans.append([int(p),int(p)])
            for a,b in spans:
                for p in list(range(max(0,a-3),a))+list(range(b+1,min(L,b+4))):
                    if not deleted[p]:junction.add(p)
            if junction:
                mask=np.isin(i,sorted(junction))|np.isin(j,sorted(junction))
                retained=~deleted[i]&~deleted[j];mask&=retained
                if mask.any():row['junction_retained_neighborhood']=dict(
                    preservation_bounds(reference_preservation_labels(native,coords,x['kept'],i[mask],j[mask]),int(mask.sum())),
                    scope='Method-dependent retained junction +/-3 sequence residues; diagnostic, not a shared regional denominator')
        local.append(row)
    summaries=[]
    for budget in [.1,.2,.3,.4]:
        for method in ['delepref_unlimited','rag_unlimited','msa_conservation','terminal']:
            rows=[r for r in generation_rows if r['seed']==13 and r['method']==method and r['budget']==budget]
            types=sorted({k for r in rows for k in r['residue_retention_by_type']})
            summaries.append(dict(budget=budget,method=method,WT_total=len(rows),
                feature_retention={k:dict(WT=sum(k in r['residue_retention_by_type'] for r in rows),
                    protein_macro=float(np.mean([r['residue_retention_by_type'][k] for r in rows if k in r['residue_retention_by_type']]))) for k in types},
                unknown_mapping_WT=sum(r['mapping_status']!='UNIQUE_EXACT_SUBSEQUENCE_MAP' for r in rows)))
    write(BASE+'/generation_region_retention.json',generation_rows);write(BASE+'/existing_structure_regional_metrics.json',local)
    write(BASE+'/budget_region_summaries.json',summaries)
    write(BASE+'/evaluation_completion.json',dict(status='ANNOTATED_OUTPUT_AND_EXISTING_STRUCTURE_ANALYSIS_COMPLETE',
        generation_outputs=len(generation_rows),existing_structure_logical_outputs=len(local),
        functional_neighborhood_defined_outputs=sum(r['functional_native_neighborhood'] is not None for r in local),
        mapping_coverage=completion['mapping_status'],WT_feature_coverage=completion['WTs_with_mapped_features'],
        absence_of_annotation_is_unknown=True,annotation_retention_not_measured_activity=True,
        new_structure_calls=0,no_new_model_training=True,paper_changes=False))
    print(json.dumps(read(BASE+'/evaluation_completion.json')),flush=True)


if __name__=='__main__':main()
