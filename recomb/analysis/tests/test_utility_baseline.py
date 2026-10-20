"""Semantic checks against original scalar definitions, without a model forward."""
import importlib.util
from pathlib import Path
import statistics
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_utility_baseline import load_module, utility_scores


@pytest.fixture
def original():
    root = Path('/public/home/zhangyangroup/chengshiz/keyuan.zhou/DelePref')
    if not root.exists():
        pytest.skip('original audited source is on cu9')
    teacher = load_module('test_utility_teacher', root / 'scripts/teacher_scoring.py')
    teacher.mean = statistics.fmean
    selection = load_module('test_utility_selection', root / 'src/delepref/proposal_validation.py')
    return teacher, selection


def test_identical_retained_embeddings_internal_and_terminal(original):
    teacher, selection = original
    wt = np.ones((8, 1152), dtype=np.float32)
    for mask in ([0, 0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1]):
        row = dict(wt_sequence='ACDEFGHI', mask=mask, sequence=''.join(c for c, d in zip('ACDEFGHI', mask) if not d))
        score = utility_scores(row, wt, wt[~np.asarray(mask, bool)], teacher, selection)
        assert score['utility'] == pytest.approx(1.)
        assert score['retained_similarity'] == 1.
        assert score['junction_plausibility'] == 1.


def test_opposite_embeddings_have_zero_formal_utility(original):
    teacher, selection = original
    wt = np.ones((8, 1152), dtype=np.float32)
    row = dict(wt_sequence='ACDEFGHI', sequence='ACFGHI', mask=[0, 0, 1, 1, 0, 0, 0, 0])
    score = utility_scores(row, wt, -wt[:6], teacher, selection)
    assert score['utility'] == 0.
    assert score['local_cosine_similarity'] == pytest.approx(-1., abs=1e-6)


def test_multi_span_original_junction_and_six_decimal_weights(original):
    teacher, selection = original
    rng = np.random.default_rng(91)
    wt = rng.normal(size=(10, 1152)).astype(np.float32)
    mask = np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0], bool)
    edited = wt[~mask] + rng.normal(size=(7, 1152)).astype(np.float32)
    row = dict(wt_sequence='ACDEFGHIKL', sequence=''.join(c for c, d in zip('ACDEFGHIKL', mask) if not d), mask=mask)
    result = utility_scores(row, wt, edited, teacher, selection)
    j = result['junction_values']
    assert result['junction_plausibility'] == pytest.approx(round(.5 * statistics.fmean(j) + .5 * min(j), 6), abs=1e-6)
    assert result['utility'] == .35 * result['global_similarity'] + .45 * result['retained_similarity'] + .20 * result['junction_plausibility']
    bad = dict(row, sequence='INVALID')
    with pytest.raises(ValueError):
        utility_scores(bad, wt, edited, teacher, selection)
    edited[0, 0] = np.nan
    with pytest.raises(ValueError):
        utility_scores(row, wt, edited, teacher, selection)
