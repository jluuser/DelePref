"""Scope metadata for already reviewed application queries; reuse metadata cache."""
import argparse
import gzip
import json
from pathlib import Path
from recomb_scope_screen_20261017 import read,sha,write

BASE=Path('reports/recomb_benign_app_evidence_20261017')


def inventory(root):
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    from recomb_application_scope_20261017 import values
    out=root/BASE;out.mkdir(exist_ok=False)
    search=root/'retrieval/recomb_benign_applications_20261017'
    source=search/'profiles_unreviewed_evidence.json'
    if sha(source)!=read(search/'completion.json')['profiles_sha256']:raise ValueError('application search changed')
    profiles=read(source);panel=read(root/'reports/recomb_benign_application_constructs_20261017/panel.json')
    approved={r['protein_id'] for r in read(root/'reports/recomb_benign_application_scope_20261017/audit.json') if r['status']=='RETAIN'}
    if set(profiles)!=approved or {r['protein_id'] for r in panel}!=approved:raise ValueError('application scope changed')
    general=root/'reports/recomb_benign_retrieval_scope_20261017'
    taxa=set(read(general/'inventory.json')['positively_reviewed_source_taxa'])
    entries={e['primaryAccession']:e for e in read(root/'provenance/recomb_benign_scope_20261017/enzyme_function_metadata.json')['data']['results']}
    targets={};queries={}
    for row in panel:
        pid=row['protein_id'];profile=profiles[pid]
        if profile['query']['sequence_sha256']!=row['sequence_sha256']:raise ValueError('query construct mismatch')
        if row['dataset']=='Enzyme20':taxa.add(int(entries[pid.split('|')[-1]]['organism']['taxonId']))
        else:
            p=Path(row['native_reference_path'])
            if sha(p)!=row['native_reference_sha256']:raise ValueError('native changed')
            with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:d=MMCIF2Dict(f)
            chains={row['design_chain']}|{p['chain_id'] for p in row['partners']}
            polymer=set(values(d,'_entity_poly.entity_id'))
            entities={e for c,e in zip(values(d,'_atom_site.auth_asym_id'),values(d,'_atom_site.label_entity_id')) if c in chains and e in polymer}
            for category,key in [('gen','pdbx_gene_src_ncbi_taxonomy_id'),('nat','pdbx_ncbi_taxonomy_id')]:
                for entity,taxon in zip(values(d,'_entity_src_'+category+'.entity_id'),values(d,'_entity_src_'+category+'.'+key)):
                    if entity in entities and taxon.isdigit():taxa.add(int(taxon))
        queries[pid]=[r['target'] for r in profile['representatives']]
        for r in profile['representatives']:targets.setdefault(r['target'],set()).add(r['target_sequence_sha256'])
    cache=read(general/'target_metadata_audit.json')
    known=sorted(t for t in targets if t in cache);missing=sorted(t for t in targets if t not in cache)
    order=known+missing
    write(out/'inventory.json',dict(targets={t:sorted(targets[t]) for t in order},query_targets=queries,
        positively_reviewed_source_taxa=sorted(taxa),parent_profiles_sha256=sha(source),
        reviewed_application_panel_sha256=sha(root/'reports/recomb_benign_application_constructs_20261017/panel.json'),
        taxonomy_scope='only individually reviewed query/partner sources; no generic species approval',
        cache_source_sha256=sha(general/'target_metadata_audit.json')))
    (out/'batches').mkdir()
    for index,start in enumerate(range(0,(len(known)//32)*32,32)):
        ids=order[start:start+32]
        write(out/'batches'/('batch_%03d.json'%index),dict(requested_ids=ids,rows=[cache[t] for t in ids],
            reused_metadata_source=str(general/'target_metadata_audit.json'),raw_sequences_saved=False))
    print(json.dumps(dict(queries=len(panel),targets=len(targets),cached_targets=len(known),new_target_metadata=len(missing))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inventory','fetch']);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    if a.mode=='inventory':inventory(a.root)
    else:
        import recomb_retrieval_scope_metadata_20261017 as metadata
        metadata.BASE=BASE
        metadata.fetch(a.root)
