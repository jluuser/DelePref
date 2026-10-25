import json
from delepref.paths import packaged_asset
from delepref.retrieval import project_hit,aligned_identity


def test_real_sec6_raw_hit_reproduces_frozen_projection():
    p=packaged_asset('examples','sec6');sequence=''.join((p/'input.fasta').read_text().splitlines()[1:])
    fields=(p/'raw_hit.tsv').read_text().strip().split('\t')
    actual=project_hit(fields,sequence)
    profile=json.loads((p/'profile.json').read_text())
    expected=next(r for r in profile['representatives'] if r['target']==fields[1])
    assert actual['projected']==expected['projected']
    assert actual['target_sequence_sha256']==expected['target_sequence_sha256']
    assert actual['known_coverage']==expected['known_coverage']


def test_unobserved_terminal_target_gaps_are_unknown():
    fields=['q','t','.5','8','3','10','10','1','6','6','1e-10','50','.8','1','DEFGHIKL','--FGHIKL','FGHIKL']
    assert project_hit(fields,'ACDEFGHIKL')['projected']=='....FGHIKL'


def test_double_gaps_do_not_supply_identity_evidence():
    assert aligned_identity('AA----CD','AA----CE')==.75
