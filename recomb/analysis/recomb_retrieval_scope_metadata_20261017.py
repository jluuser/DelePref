"""Collect source/function metadata for existing homologs; never generate edits."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
from recomb_scope_screen_20261017 import HAZARD, VIRAL, read, write, sha

BASE = Path('reports/recomb_benign_retrieval_scope_20261017')


def inventory(root):
    out = root / BASE
    out.mkdir(exist_ok=True)
    if (out / 'inventory.json').exists():
        raise FileExistsError('inventory already exists; reuse it')
    scope = root / 'reports/recomb_benign_scope_20261017'
    accepted = [r for r in read(scope / 'general_audit.json') if r['status'] == 'RETAIN']
    source = root / 'retrieval/delepref_v8_main/profiles.json'
    profiles = read(source)
    taxa, targets, queries = set(), {}, {}
    for row in accepted:
        for f in row['annotation_files']:
            path = root / f['path'].replace('\\', '/')
            if sha(path) != f['sha256']:
                raise ValueError('annotation changed')
            taxon = read(path).get('organism', {}).get('taxonId')
            if taxon:
                taxa.add(int(taxon))
        pid = row['protein_id']
        reps = profiles[pid]['representatives']
        queries[pid] = [r['target'] for r in reps]
        for rep in reps:
            targets.setdefault(rep['target'], set()).add(rep['target_sequence_sha256'])
    write(out / 'inventory.json', dict(parent_profiles_sha256=sha(source),
        targets={t: sorted(h) for t, h in sorted(targets.items())}, query_targets=queries,
        positively_reviewed_source_taxa=sorted(taxa), policy='exact reviewed taxon + function check + representative sequence hash; all unknowns held'))
    print(json.dumps(dict(status='METADATA_REQUEST_INVENTORY_ONLY', queries=len(queries), targets=len(targets), new_model_calls=0)))


def fetch(root):
    out = root / BASE
    data = read(out / 'inventory.json')
    targets = list(data['targets'])
    taxa = set(data['positively_reviewed_source_taxa'])
    cache = out / 'batches'
    cache.mkdir(exist_ok=True)

    def batch(index, ids):
        path = cache / ('batch_%03d.json' % index)
        if path.exists():
            value = read(path)
            if value['requested_ids'] != ids:
                raise ValueError('batch request identity changed')
            return value
        url = 'https://rest.uniprot.org/uniref/search?' + urllib.parse.urlencode(dict(
            query=' OR '.join('id:' + i for i in ids), format='json', size='100'))
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = json.load(response)
        rows = []
        for entry in payload.get('results', []):
            if entry['id'] not in ids:
                raise ValueError('unexpected metadata result')
            rep = entry.get('representativeMember', {})
            sequence = rep.get('sequence', {}).get('value')
            rows.append(dict(target=entry['id'], updated=entry.get('updated'),
                organism=rep.get('organismName'), taxon_id=rep.get('organismTaxId'),
                protein_name=rep.get('proteinName'),
                representative_sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest() if sequence else None))
        value = dict(requested_ids=ids, source_url=url, rows=rows, raw_sequences_saved=False)
        write(path, value)
        return value

    batches = [targets[i:i+32] for i in range(0, len(targets), 32)]
    records, errors = {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(batch, i, ids): i for i, ids in enumerate(batches)}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                for row in future.result()['rows']:
                    records[row['target']] = row
            except Exception as exc:
                errors.append(dict(batch=futures[future], error=str(exc)))
            if done % 10 == 0:
                print(json.dumps(dict(metadata_batches_done=done, total=len(batches))), flush=True)
    reviewed = {}
    for target, expected in data['targets'].items():
        row = records.get(target, dict(target=target))
        name = row.get('protein_name') or ''
        org = row.get('organism') or ''
        status, reason = 'HOLD', 'missing_or_unreviewed_source_identity'
        if HAZARD.search(name) or VIRAL.search(org + ' ' + name):
            status, reason = 'EXCLUDE', 'out_of_scope_function_or_viral_source'
        elif row.get('taxon_id') in taxa and name and len(expected) == 1 and row.get('representative_sequence_sha256') == expected[0]:
            status, reason = 'RETAIN', 'exact_reviewed_source_and_sequence_identity'
        elif not row.get('organism'):
            reason = 'metadata_missing'
        elif len(expected) != 1 or row.get('representative_sequence_sha256') not in expected:
            reason = 'representative_sequence_identity_not_confirmed'
        elif row.get('taxon_id') not in taxa:
            reason = 'source_taxon_not_yet_positively_reviewed'
        elif not name:
            reason = 'function_name_missing'
        row.update(status=status, reason=reason)
        reviewed[target] = row
    write(out / 'target_metadata_audit.json', reviewed)
    counts = {pid: dict(total=len(ids), retained=sum(reviewed[t]['status']=='RETAIN' for t in ids)) for pid, ids in data['query_targets'].items()}
    write(out / 'per_query_scope_counts.json', counts)
    summary = dict(status='METADATA_ONLY_NOT_A_NEW_RETRIEVAL_PROFILE', targets=len(targets),
        disposition={s: sum(r['status']==s for r in reviewed.values()) for s in ['RETAIN','EXCLUDE','HOLD']},
        hold_reasons=dict(Counter(r['reason'] for r in reviewed.values() if r['status']=='HOLD')),
        query_count=len(counts), queries_with_retained_evidence=sum(r['retained']>0 for r in counts.values()),
        errors=errors, new_model_calls=0, raw_sequences_saved=False,
        next='Build versioned filtered profiles using original weighting semantics; freeze and audit before RAG generation')
    write(out / 'completion.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['inventory', 'fetch'])
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    (inventory if args.mode == 'inventory' else fetch)(args.root)
