"""Optional encoder dependency test, before any GPU model loading."""
import json
import pytest
from delepref.paths import packaged_asset


def test_encoder_tokenizer_matches_frozen_snapshot():
    pytest.importorskip('esm')
    from tokenizers import Tokenizer
    from delepref.encoder_compat import make_sequence_tokenizer
    tokenizer=make_sequence_tokenizer()
    reference=Tokenizer.from_file(str(packaged_asset('encoder_tokenizer_reference.json')))
    sequences=[]
    for case in ['sec6','af1862']:
        p=packaged_asset('examples',case,'input.fasta');sequences.append(''.join(p.read_text().splitlines()[1:]))
    batch=tokenizer(sequences,return_tensors='pt',padding=True)
    for i,sequence in enumerate(sequences):
        n=len(sequence)+2
        assert batch['input_ids'][i,:n].tolist()==reference.encode(sequence).ids
        assert batch['attention_mask'][i,:n].sum().item()==n
    assert tokenizer.pad_token_id==1 and tokenizer.cls_token_id==0 and tokenizer.eos_token_id==2
