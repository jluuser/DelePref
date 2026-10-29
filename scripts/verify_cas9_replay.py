"""Replay the five historical Cas9 plans from published real embeddings."""
import argparse
import json
from pathlib import Path
import tempfile

import torch
from safetensors.torch import load_file
from delepref.cli import read_fasta
from delepref.downloads import catalog, extract_case
from delepref.inference import file_sha, load_heads, load_profile, plan_from_embeddings


def verify(archive):
    archive=Path(archive)
    expected=catalog()['assets']['cas9_evidence']['sha256']
    if file_sha(archive) != expected:
        raise ValueError('Cas9 evidence archive SHA256 does not match the release catalog')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='delepref-cas9-replay-') as tmp:
        root=Path(tmp); extract_case(archive,root)
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf8'))
        if manifest.get('schema') != 'delepref-cas9-cache-v1' or len(manifest['cases']) != 5:
            raise ValueError('Unexpected Cas9 evidence manifest')
        runtime=load_heads()
        for case in manifest['cases']:
            folder=root/case['case']
            pid,sequence=read_fasta(folder/'input.fasta',max_length=2048)[0]
            embedding_path=folder/'embedding.safetensors'
            if file_sha(embedding_path) != case['embedding_sha256']:
                raise ValueError('Cas9 embedding checksum mismatch: '+case['case'])
            embedding=load_file(str(embedding_path))['residue_embeddings']
            profile=load_profile(folder/'profile.json',pid,sequence)
            result=plan_from_embeddings(runtime,pid,sequence,embedding,budget=.2,seed=13,
                                        profile=profile,max_length=2048)
            frozen={row['method']:row for row in json.loads((folder/'selected_outputs.json').read_text())}
            for branch,method in (('base','delepref_unlimited'),('guided','rag_unlimited')):
                if result['selected'][branch]['deletion_mask_sha256'] != frozen[method]['deletion_mask_sha256']:
                    raise ValueError('Historical Cas9 selection differs: '+case['case']+' '+branch)
            print('{}: WT length {}, Base and guided selections match'.format(case['case'],len(sequence)),flush=True)
    print('CAS9_CACHED_REPLAY_VERIFIED: five WTs, no new encoder calls',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence',required=True)
    a=p.parse_args();verify(a.evidence)
