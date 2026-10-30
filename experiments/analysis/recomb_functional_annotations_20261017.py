"""Fetch small official annotations; map only exact, unique WT sequence correspondences."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import urllib.request

LOCAL=Path(__file__).resolve().parents[1]
OUT=LOCAL/'provenance/recomb_functional_annotations_20261017'
REPORT=LOCAL/'reports/recomb_functional_annotations_20261017'


def fetch(item):
    kind,key,url=item;path=OUT/kind/(key+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    try:
        if path.exists():data=path.read_bytes()
        else:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DelePref-academic-annotation-audit'}),timeout=15) as r:data=r.read()
            json.loads(data);path.write_bytes(data)
        return dict(kind=kind,key=key,url=url,status='OK',sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)),json.loads(data)
    except Exception as error:return dict(kind=kind,key=key,url=url,status='UNAVAILABLE',error=repr(error)),None


def unique_map(canonical,wt):
    left=[];p=0
    for aa in wt:
        p=canonical.find(aa,p)
        if p<0:return None
        left.append(p);p+=1
    right=[];p=len(canonical)
    for aa in wt[::-1]:p=canonical.rfind(aa,0,p);right.append(p)
    right.reverse()
    return left if left==right else None


def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    panel=json.loads((LOCAL/'reports/delepref_v8_native_panel/main_panel.json').read_text())
    pdbs={}
    for wt in panel:
        if wt['stratum']=='cath_domain':
            pdb=wt['protein_id'].split('_',1)[1][:4].lower();pdbs[pdb]=None
    manifest=[];sifts={};accessions={x['protein_id']:set() for x in panel}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for report,data in pool.map(fetch,[('pdbe',p,'https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/'+p) for p in sorted(pdbs)]):
            manifest.append(report)
            if data:sifts[report['key']]=data
    for wt in panel:
        if wt.get('uniprot_accession'):accessions[wt['protein_id']].add(wt['uniprot_accession'])
        else:
            pdb=wt['protein_id'].split('_',1)[1][:4].lower();chain=wt['reference_chain']
            data=sifts.get(pdb,{}).get(pdb,{})
            for acc,entry in data.get('UniProt',{}).items():
                if any(str(m.get('chain_id'))==str(chain) for m in entry.get('mappings',[])):
                    accessions[wt['protein_id']].add(acc)
    all_accessions=sorted(set(a for v in accessions.values() for a in v));uniprot={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for report,data in pool.map(fetch,[('uniprot',a,'https://rest.uniprot.org/uniprotkb/'+a+'.json') for a in all_accessions]):
            manifest.append(report)
            if data:uniprot[report['key']]=data
    mapped=[]
    feature_types={'Active site','Binding site','Disulfide bond','Domain','Region','DNA binding','Motif'}
    for wt in panel:
        possibilities=[]
        for acc in sorted(accessions[wt['protein_id']]):
            entry=uniprot.get(acc);canonical=(entry or {}).get('sequence',{}).get('value','')
            mapping=unique_map(canonical,wt['sequence']) if canonical else None
            if mapping is not None:possibilities.append((acc,entry,mapping))
        row=dict(protein_id=wt['protein_id'],WT_sha256=wt['sequence_sha256'],WT_length=len(wt['sequence']),
                 candidate_accessions=sorted(accessions[wt['protein_id']]),features=[],mapping_status='UNKNOWN_NO_UNIQUE_EXACT_CANONICAL_MAP')
        # Different canonicals with exact but distinct annotation interpretations remain ambiguous.
        if len(possibilities)==1:
            acc,entry,mapping=possibilities[0];inverse={u:i for i,u in enumerate(mapping)}
            row.update(accession=acc,mapping_status='UNIQUE_EXACT_SUBSEQUENCE_MAP',canonical_positions_0based=mapping,
                       canonical_sequence_sha256=hashlib.sha256(entry['sequence']['value'].encode()).hexdigest())
            for feature in entry.get('features',[]):
                kind=feature['type']
                if kind not in feature_types:continue
                if kind=='Region' and not re.search('disorder|linker',feature.get('description',''),re.I):continue
                start=feature.get('location',{}).get('start',{});end=feature.get('location',{}).get('end',{})
                if 'value' not in start or 'value' not in end:continue
                if start.get('modifier','EXACT')!='EXACT' or end.get('modifier','EXACT')!='EXACT':continue
                a,b=int(start['value'])-1,int(end['value'])-1
                positions=[a,b] if kind=='Disulfide bond' else range(a,b+1)
                wt_positions=sorted(set(inverse[p] for p in positions if p in inverse))
                if wt_positions:row['features'].append(dict(type=kind,description=feature.get('description',''),
                    canonical_interval_0based=[a,b],WT_positions_0based=wt_positions,
                    feature_id=feature.get('featureId'),evidences=feature.get('evidences',[]),
                    canonical_feature_positions=len(set(positions)),mapped_feature_positions=len(wt_positions)))
        elif len(possibilities)>1:row['mapping_status']='AMBIGUOUS_MULTIPLE_EXACT_CANONICALS'
        mapped.append(row)
    def save(path,data):path.write_text(json.dumps(data,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    save(OUT/'manifest.json',manifest);save(REPORT/'mapped_features.json',mapped)
    report=dict(status='FUNCTIONAL_ANNOTATION_SOURCE_AND_EXACT_MAPPING_AUDITED',WT=len(mapped),
        mapping_status=dict(Counter(x['mapping_status'] for x in mapped)),
        WTs_with_mapped_features=sum(bool(x['features']) for x in mapped),
        feature_counts=dict(Counter(f['type'] for x in mapped for f in x['features'])),
        source_status=dict(Counter(x['status'] for x in manifest)),
        unknown_is_not_absence=True,annotations_are_not_measured_mutant_activity=True,
        no_model_or_structure_outputs_used_to_map=True,protection_constraints_not_automatically_applied=True,
        source_manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest(),
        mapped_features_sha256=hashlib.sha256((REPORT/'mapped_features.json').read_bytes()).hexdigest())
    save(REPORT/'completion.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
