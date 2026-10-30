"""Fill source metadata for cached extension buffers, reusing prior successes."""
import concurrent.futures
from collections import Counter
from pathlib import Path
import re
from recomb_confirm_v2_diagnostics import read,write
from recomb_confirm_v2_public_metadata import get,OUT as PUBLIC

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/recomb_confirm_v2_expansion_buffers'


def main():
    if (BASE/'metadata_completion.json').exists():raise FileExistsError('Extension metadata already fetched')
    blocked_pdb=set();blocked_accessions=set()
    for name in ['reports/recomb_benign_scope_20261017/general_audit.json',
                 'reports/recomb_benign_application_scope_20261017/audit.json']:
        for r in read(ROOT/name):
            if r['status']!='EXCLUDE':continue
            pid=r['protein_id']
            if pid.startswith('Enzyme20|'):blocked_accessions.add(pid.split('|')[-1])
            if pid.startswith('curated_pdb|'):blocked_pdb.add(pid.split('|')[2].upper())
            if pid.startswith('CATH_'):blocked_pdb.add(pid[5:9].upper())
            m=re.search(r'(?:AB-Bind:|SAbDab:)([A-Za-z0-9]{4})',pid)
            if m:blocked_pdb.add(m[1].upper())
    todo=set()
    for r in read(BASE/'CATH600_metadata_buffer.json'):
        if r['PDB'] not in blocked_pdb:todo.add(('rcsb_entry',r['PDB']))
    for r in read(BASE/'MCSA_coordinate_pass_metadata_buffer.json'):
        accession=r.get('uniprot_accession');pdb=str(r.get('reference_pdb_id','')).upper()
        if accession and accession not in blocked_accessions:todo.add(('uniprot',accession))
        if pdb and pdb not in blocked_pdb:todo.add(('rcsb_entry',pdb))
    for r in read(BASE/'SAbDab_metadata_buffer.json'):
        if r['pdb'].upper() not in blocked_pdb:todo.add(('rcsb_entry',r['pdb'].upper()))
    write(BASE/'metadata_request_manifest.json',dict(requests=[dict(kind=k,key=v) for k,v in sorted(todo)],
        reuse_existing_public_cache=True,blocked_pdb=sorted(blocked_pdb),no_model_release=True))
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(lambda item:get(*item),sorted(todo)):
            results.append(r)
            if len(results)%25==0:
                write(BASE/'metadata_progress.json',dict(completed=len(results),initial_total=len(todo),
                    status=dict(Counter(x['status'] for x in results)),reused=sum(x['reused'] for x in results)))
                print('EXTENSION_METADATA '+str(len(results))+'/'+str(len(todo)),flush=True)
    entities=set()
    for r in results:
        if r['kind']=='rcsb_entry' and r['status']=='SUCCEEDED':
            for eid in read(PUBLIC/'rcsb_entry'/(r['key']+'.json'))['metadata']['polymer_entity_ids']:
                entities.add(('rcsb_entity',r['key']+'_'+str(eid)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(lambda item:get(*item),sorted(entities)):
            results.append(r)
            if len(results)%25==0:
                write(BASE/'metadata_progress.json',dict(completed=len(results),total=len(todo)+len(entities),
                    status=dict(Counter(x['status'] for x in results)),reused=sum(x['reused'] for x in results)))
                print('EXTENSION_ENTITY_METADATA '+str(len(results)),flush=True)
    write(BASE/'metadata_completion.json',dict(status='EXTENSION_SOURCE_METADATA_FETCHED_PENDING_REVIEW',
        total=len(results),by_status=dict(Counter(x['status'] for x in results)),
        reused=sum(x['reused'] for x in results),no_model_release=True,results=results))
    print('EXTENSION_METADATA_FETCH_COMPLETE',flush=True)


if __name__=='__main__':main()
