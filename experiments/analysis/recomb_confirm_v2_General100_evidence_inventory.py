"""Export homolog source/identity metadata only; never release unreviewed evidence."""
import argparse
from collections import defaultdict
from pathlib import Path
from recomb_confirm_v2_diagnostics import read,sha,write


def main(root):
    source=root/'retrieval/recomb_confirm_v2_General100/profiles_unreviewed_evidence.json'
    out=root/'reports/recomb_confirm_v2_General100_evidence_review'
    if (out/'inventory.json').exists():raise FileExistsError('Reuse existing evidence inventory')
    completion=read(source.parent/'completion.json')
    if sha(source)!=completion['profiles_sha256']:raise ValueError('Unreviewed profile identity changed')
    profiles=read(source);targets=defaultdict(set);queries={}
    for pid,p in profiles.items():
        if p['evidence_status']!='UNREVIEWED_NOT_FOR_GENERATION':raise ValueError('Unexpected source state')
        queries[pid]=[]
        for rep in p['representatives']:
            if not rep['complete_target_identity_checked']:raise ValueError('Target not verified against search source')
            targets[rep['target']].add(rep['target_sequence_sha256']);queries[pid].append(rep['target'])
    write(out/'inventory.json',dict(status='METADATA_IDENTITY_INVENTORY_ONLY',
        profiles_sha256=sha(source),targets={k:sorted(v) for k,v in sorted(targets.items())},query_targets=queries,
        new_model_calls=0,raw_sequences_exported=False,RAG_generation_allowed=False))
    print(dict(queries=len(queries),unique_targets=len(targets),representative_rows=sum(map(len,queries.values()))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
