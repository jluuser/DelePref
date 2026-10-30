"""Pinned, project-local ESMC runtime metadata and CPU-safe loader validation."""
import importlib.metadata
import json
from pathlib import Path
from common import AUTHORIZED_ROOT, identity, sha_file
from adapters.esmc_loader import ESMCSequenceTokenizer


def verify_loader():
    path = Path(__file__).with_name('esmc_loader.py')
    source = json.loads((AUTHORIZED_ROOT/'provenance/esmc_runtime.json').read_text())
    if sha_file(path) != source['vendored_sha256'] or sha_file(source['origin_path']) != source['origin_sha256']:
        raise ValueError('ESMC runtime loader identity mismatch')
    return path, source


def tokenizer_profile(tokenizer=None):
    path, source = verify_loader()
    tokenizer = tokenizer or ESMCSequenceTokenizer(max_length=1022)
    vocab = dict(tokenizer._token_to_id)
    config = {'max_length':tokenizer.max_length, 'cls_token_id':tokenizer.cls_token_id,
              'pad_token_id':tokenizer.pad_token_id, 'eos_token_id':tokenizer.eos_token_id,
              'unk_token_id':tokenizer.unk_token_id, 'special_tokens':'CLS + residues + EOS; right padding'}
    return {'loader_sha256':source['vendored_sha256'], 'loader_origin_sha256':source['origin_sha256'],
            'runtime_adapter_sha256':sha_file(__file__), 'tokenizer_source_sha256':sha_file(path),
            'tokenizer_class':type(tokenizer).__module__+'.'+type(tokenizer).__qualname__,
            'tokenizer_vocab_sha256':identity(vocab), 'tokenizer_config_sha256':identity(config),
            'transformers_version':importlib.metadata.version('transformers'),
            'tokenizers_version':importlib.metadata.version('tokenizers')}
