"""Release parity uses real frozen embeddings and the actual trained heads."""
import json
import os
from pathlib import Path
import pytest
import torch
from safetensors.torch import load_file
from delepref.inference import load_heads,load_profile,plan_from_embeddings,validate_sequence
from delepref.paths import packaged_asset,resolve_path


@pytest.fixture(scope='module')
def runtime():
    torch.set_num_threads(4)
    return load_heads()


def inputs(case):
    p=packaged_asset('examples',case)
    lines=(p/'input.fasta').read_text().splitlines()
    pid,sequence=lines[0][1:],''.join(lines[1:])
    return p,pid,sequence,load_file(str(p/'embedding.safetensors'))['residue_embeddings']


@pytest.mark.parametrize('case',['sec6','af1862'])
def test_all_frozen_candidate_masks_scores_and_ranks(runtime,case):
    p,pid,sequence,h=inputs(case)
    result=plan_from_embeddings(runtime,pid,sequence,h,profile=load_profile(p/'profile.json',pid,sequence))
    expected=json.loads((p/'expected.json').read_text())
    provenance=json.loads((p/'provenance.json').read_text())
    assert str(h.dtype)==provenance['embedding_dtype']  # Actual cache dtype, not encoder autocast dtype.
    for label in ['base','guided']:
        assert len(result['candidates'][label])==8
        for row,gold in zip(result['candidates'][label],expected[label]['candidates']):
            assert row['deleted_intervals']==json.loads(gold['deleted_intervals'])
            assert row['scorer_rank']==gold['scorer_rank']
            assert row['condition_seed']==gold['condition_seed']
            assert row['scorer_score']==pytest.approx(gold['scorer_score'],abs=1e-5)
        chosen=result['selected'][label];gold=expected[label]['selected']
        assert chosen['deletion_mask_sha256']==gold['deletion_mask_sha256']
        assert chosen['edited_sequence_sha256']==gold['edited_sequence_sha256']


@pytest.mark.parametrize('case',['sec6','af1862'])
def test_missing_evidence_is_exact_fallback(runtime,case):
    _,pid,sequence,h=inputs(case)
    result=plan_from_embeddings(runtime,pid,sequence,h)
    assert result['fallback_exact']
    assert result['candidates']['base']==result['candidates']['guided']
    assert result['selected']['base']==result['selected']['guided']


def test_public_and_legacy_absolute_paths_are_read_without_rewriting(tmp_path):
    p=tmp_path/'an-existing-asset';p.write_text('unchanged')
    assert resolve_path(p,exists=True)==p.resolve()
    assert resolve_path(p.name,base=tmp_path,exists=True)==p.resolve()
    assert p.read_text()=='unchanged'


def test_guided_exact_budget_and_anchor_retention(runtime):
    p,pid,sequence,h=inputs('sec6')
    result=plan_from_embeddings(runtime,pid,sequence,h,profile=load_profile(p/'profile.json',pid,sequence))
    selected=result['selected']['guided'];deleted={i for a,b in selected['deleted_intervals'] for i in range(a,b+1)}
    seed=selected['retrieval_seed_spans'];anchors={i for a,b in seed for i in [a-1,b+1]}
    assert len(deleted)==42 and len(selected['edited_sequence'])==167
    assert not anchors&deleted
    assert sum(b-a+1 for a,b in seed)==12


def test_profile_identity_mismatch_is_not_silently_used(tmp_path):
    p=tmp_path/'profile.json'
    p.write_text(json.dumps({'query':{'sequence_sha256':'0'*64},'representatives':[]}))
    with pytest.raises(ValueError,match='hash mismatch'):load_profile(p,'wrong','ACDE')


def test_optional_original_checkpoint_and_yaml_load_without_changes():
    weights=os.environ.get('DELEPREF_TEST_LEGACY_WEIGHTS')
    config=os.environ.get('DELEPREF_TEST_LEGACY_CONFIG')
    if not weights or not config:pytest.skip('Explicit original research paths not supplied')
    from delepref.inference import file_sha
    before=[file_sha(p) for p in [weights,config]]
    public=load_heads();legacy=load_heads(weights,config)
    assert all(torch.equal(v,legacy.model.state_dict()[k]) for k,v in public.model.state_dict().items())
    assert before==[file_sha(p) for p in [weights,config]]


def test_long_sequence_requires_explicit_cached_extrapolation():
    sequence='A'*1368
    with pytest.raises(ValueError, match='2-1022'):
        validate_sequence(sequence)
    assert validate_sequence(sequence, max_length=2048) == sequence
    with pytest.raises(ValueError, match='2-2048'):
        validate_sequence('A'*2049, max_length=2048)


def test_long_sequence_flag_cannot_silently_trigger_fresh_encoding():
    from argparse import Namespace
    from delepref.cli import shrink
    with pytest.raises(ValueError, match='explicitly supplied real WT embedding'):
        shrink(Namespace(allow_long_cached_embedding=True, embedding=None))
