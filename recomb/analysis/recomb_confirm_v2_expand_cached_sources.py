"""Export metadata buffers from existing CATH/M-CSA/SAbDab assets only."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from recomb_confirm_v2_diagnostics import read,sha,write


def family(r):
    h=str(r['cath_homologous_superfamily'])
    if '.' in h:return h
    return '.'.join(str(r[k]) for k in ('cath_class','cath_architecture','cath_topology','cath_homologous_superfamily'))


def server(root):
    out=root/'reports/recomb_confirm_v2_expansion_buffers'
    if (out/'server_completion.json').exists():raise FileExistsError('Cached-source buffers already exported')
    cath_path=Path('/public/nfs-data/keyuan.zhou/evocompress/data/cath_s40_v1/inventory/eligible_domains.json')
    cath=read(cath_path);bydomain={r['domain_id']:r for r in cath}
    old=read(root/'reports/delepref_v8_native_panel/main_panel.json')
    oldpdb=set();oldfamilies=set()
    for r in old:
        d=r.get('domain_id')
        if d:
            oldpdb.add(d[:4].lower())
            if d in bydomain:oldfamilies.add(family(bydomain[d]))
        elif r['protein_id'].startswith('curated_pdb|'):oldpdb.add(r['protein_id'].split('|')[2].lower())
    groups=defaultdict(list)
    for r in cath:
        if not 150<=r['sequence_length']<=512 or r['domain_id'][:4].lower() in oldpdb or family(r) in oldfamilies:continue
        groups[int(r['cath_class'])].append(r)
    selected=[];seen=set()
    for c in (1,2,3):
        ordered=sorted(groups[c],key=lambda r:hashlib.sha256(('recomb-confirm-v2-general-buffer|'+r['domain_id']).encode()).hexdigest())
        count=0
        for r in ordered:
            f=family(r)
            if f in seen:continue
            seen.add(f)
            selected.append(dict(domain_id=r['domain_id'],PDB=r['domain_id'][:4].upper(),chain=r['domain_id'][4],
                family=f,cath_class=c,length=r['sequence_length'],resolution=r['resolution_angstrom'],
                archive_member=r['archive_member'],WT_sha256=hashlib.sha256(r['sequence'].encode()).hexdigest(),
                source_file=str(cath_path),source_sequence_sha1=r['sequence_sha1'],
                scope='UNREVIEWED_METADATA_ONLY',model_use_allowed=False))
            count+=1
            if count==200:break
    enzyme_path=Path('/public/nfs-data/keyuan.zhou/evocompress/data/enzyme_120_v1/coordinate_audit_v1/coordinate_pass.jsonl')
    enzymes=[]
    with enzyme_path.open() as f:
        for line in f:
            r=json.loads(line)
            if not 150<=r['sequence_length']<=1022:continue
            e={k:r[k] for k in ('candidate_id','mcsa_id','uniprot_accession','enzyme_name','description','ec_numbers',
                'reference_pdb_id','reference_chain_id','reference_assembly','reference_structure_path',
                'sequence_homology_component','sequence_length','catalytic_residue_count',
                'all_catalytic_residues_resolved','all_functional_atoms_complete','has_reaction_compound_annotation',
                'candidate_homology_hits','protected_homology_hits','sequence_leakage_barrier') if k in r}
            e.update(WT_sha256=hashlib.sha256(r['sequence'].encode()).hexdigest(),source_file=str(enzyme_path),
                     scope='UNREVIEWED_METADATA_ONLY',model_use_allowed=False)
            enzymes.append(e)
    write(out/'CATH600_metadata_buffer.json',selected);write(out/'MCSA_coordinate_pass_metadata_buffer.json',enzymes)
    result=dict(status='EXISTING_CACHED_SOURCE_METADATA_EXPORTED_NO_MODEL_RELEASE',
        cath_source_domains=len(cath),cath_source_file_sha256=sha(cath_path),
        cath_metadata_buffer=len(selected),cath_classes=dict(Counter(r['cath_class'] for r in selected)),
        cath_family_unique=len({r['family'] for r in selected}),old100_families_excluded=len(oldfamilies),
        enzyme_coordinate_pass_in_length_range=len(enzymes),enzyme_source_sha256=sha(enzyme_path),
        new_model_calls=0,raw_sequences_exported=False)
    write(out/'server_completion.json',result);print(result)


def binder(root):
    out=root/'reports/recomb_confirm_v2_expansion_buffers'
    if (out/'binder_completion.json').exists():raise FileExistsError('SAbDab buffer already exported')
    source=root/'caches/sabdab_summary_all_existing.tsv'
    with source.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    old=read(root/'reports/recomb_benign_application_scope_20261017/audit.json')
    oldpdb=set()
    for r in old:
        m=re.search(r'(?:AB-Bind:|SAbDab:)([A-Za-z0-9]{4})',r['protein_id'])
        if m:oldpdb.add(m[1].lower())
    sources={'homo sapiens','mus musculus','rattus norvegicus','bos taurus','sus scrofa','oryctolagus cuniculus',
             'arabidopsis thaliana','danio rerio'}
    concern=re.compile(r'toxin|venom|virulen|virus|viral|pathogen|bacterial|bacterium|antibiotic resistance',re.I)
    groups=defaultdict(list)
    for r in rows:
        if r['pdb'].lower() in oldpdb or r.get('model') not in ('0','1'):continue
        if r.get('antigen_type')!='protein' or not r.get('Hchain') or r.get('scfv','').lower()=='true':continue
        species=[x.strip().lower() for x in r.get('antigen_species','').split('|')]
        if not species or not all(s in sources for s in species):continue
        if concern.search(' '.join(r.get(k,'') for k in ('antigen_name','short_header','compound'))):continue
        try:resolution=float(r['resolution'])
        except ValueError:continue
        if resolution>3.0 or r['method']!='X-RAY DIFFRACTION':continue
        if not r.get('antigen_chain') or r['antigen_chain'] in ('NA','None'):continue
        kind='single_heavy_candidate' if r.get('Lchain') in ('','NA','None') else 'heavy_with_fixed_light_candidate'
        antigen=re.sub(r'[^a-z0-9]+',' ',r['antigen_name'].lower()).strip()
        groups[kind,antigen].append(r)
    selected=[];seen=set();perkind=Counter()
    for (kind,antigen),pool in sorted(groups.items(),key=lambda kv:hashlib.sha256(str(kv[0]).encode()).hexdigest()):
        if perkind[kind]>=200:continue
        ordered=sorted(pool,key=lambda r:hashlib.sha256(('recomb-confirm-v2-binder|'+r['pdb']+'|'+r['Hchain']).encode()).hexdigest())
        for r in ordered[:2]:
            if perkind[kind]>=200:break
            key=r['pdb'].lower(),r['Hchain'],r.get('antigen_chain')
            if key in seen:continue
            seen.add(key);perkind[kind]+=1
            selected.append(dict(r,prospective_design_chain=r['Hchain'],fixed_light_chain=r.get('Lchain'),
                scaffold_candidate_type=kind,antigen_name_group=antigen,scope='METADATA_CANDIDATE_NEEDS_RCSB_AND_NATIVE_CHAIN_REVIEW',
                model_use_allowed=False))
    write(out/'SAbDab_metadata_buffer.json',selected)
    result=dict(status='EXISTING_SABDAB_METADATA_BUFFER_NO_MODEL_RELEASE',source_rows=len(rows),source_sha256=sha(source),
        qualifying_antigen_name_groups=len(groups),buffer_cases=len(selected),scaffold_candidates=dict(perkind),
        unique_PDB=len({r['pdb'] for r in selected}),new_model_calls=0)
    write(out/'binder_completion.json',result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['server','binder'],required=True)
    a=p.parse_args();(server if a.stage=='server' else binder)(a.root)
