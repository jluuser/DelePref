"""Export frozen full-length Cas9 embeddings and selections as safe tensors.

Reads historical WT caches using PyTorch's weights-only loader and checks their
sequence, checkpoint and source hashes. No new model forward is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

import torch
from safetensors.torch import save_file


CASES = ('SpCas9', 'SaCas9', 'Nme1Cas9', 'GeoCas9', 'FnCas9')
CHECKPOINT = '1bc7694be77e24e4b671773be39331daed241dcfe101fd6d2d26f8e23fe7ff4d'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def normalized(value, root):
    if isinstance(value, dict):
        return {normalized(k, root):normalized(v, root) for k,v in value.items()}
    if isinstance(value, list):
        return [normalized(x, root) for x in value]
    if isinstance(value, str):
        prefix = str(root).rstrip('/') + '/'
        if value.startswith(prefix):return 'research-source/' + value[len(prefix):]
        if value.startswith(('/public/home/', '/public/nfs-data/')):
            return 'external-source/' + PurePosixPath(value).name
    return value


def build(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):raise ValueError('Output must be outside research source')
    if output.exists():raise FileExistsError('Use a new output path: ' + str(output))
    records=[]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='delepref-cas9-') as temp, zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        temp = Path(temp)
        for case in CASES:
            folder = root / ('reports/recomb_cas9_20261020' if case == 'SpCas9'
                             else 'reports/recomb_cas9_panel_20261020/' + case)
            source = folder / 'wt_cache.pt'
            info = json.loads((folder/'encoding_completion.json').read_text())
            if sha(source) != info['cache_sha256'] or info['sequence_truncated']:
                raise ValueError('Original cache identity or full-length check failed: ' + case)
            cache = torch.load(source, map_location='cpu', weights_only=True)
            if cache['checkpoint_sha256'] != CHECKPOINT or not cache['real_encoder']:
                raise ValueError('Unexpected checkpoint/encoder provenance: ' + case)
            if len(cache['embeddings']) != 1 or set(cache['embeddings']) != set(cache['sequences']):
                raise ValueError('Unexpected embedding cache keys: ' + case)
            pid, embedding = next(iter(cache['embeddings'].items()))
            sequence = cache['sequences'][pid]
            seq_sha = hashlib.sha256(sequence.encode('ascii')).hexdigest()
            if ((info.get('sequence_sha256') and info['sequence_sha256'] != seq_sha)
                    or (cache.get('sequence_sha256') and cache['sequence_sha256'] != seq_sha)):
                raise ValueError('Sequence identity mismatch: ' + case)
            if list(embedding.shape) != info['embedding_shape'] or len(sequence) != info['WT_length']:
                raise ValueError('Embedding/sequence length mismatch: ' + case)
            selected = json.loads((folder/'selected_outputs.json').read_text(encoding='utf8'))
            if not selected or any(row['wildtype_sequence'] != sequence for row in selected):
                raise ValueError('Selected-output WT differs from cached WT: ' + case)
            tensor_file = temp / (case + '.safetensors')
            save_file({'residue_embeddings': embedding.detach().cpu().contiguous()}, str(tensor_file))
            prefix = case + '/'
            archive.write(tensor_file, prefix + 'embedding.safetensors')
            archive.writestr(prefix + 'input.fasta', '>{}\n{}\n'.format(pid,sequence))
            for name in ('profile.json','selected_outputs.json','encoding_completion.json'):
                value = normalized(json.loads((folder/name).read_text(encoding='utf8')), root)
                payload = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
                if b'/public/home/' in payload or b'/public/nfs-data/' in payload:
                    raise ValueError('Unnormalized source path: ' + case + '/' + name)
                archive.writestr(prefix + name, payload)
            records.append(dict(case=case, protein_id=pid, length=len(sequence),
                sequence_sha256=seq_sha,
                original_cache_sha256=info['cache_sha256'],
                embedding_sha256=sha(tensor_file), checkpoint_sha256=CHECKPOINT))
        archive.writestr('manifest.json', json.dumps(dict(schema='delepref-cas9-cache-v1',
            encoding='Original full WT encoding; process-local maximum length 2048; no new inference.',
            cases=records), indent=2) + '\n')
    print(json.dumps(dict(status='EXPORTED', cases=len(records), bytes=output.stat().st_size,
                          sha256=sha(output))), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    a=p.parse_args();build(a.root,a.out)
