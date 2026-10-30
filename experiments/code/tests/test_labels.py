import numpy as np
import pytest
from build_pairs import reference_pairs, reference_preservation_labels, preservation_bounds, ERROR_THRESHOLDS


def test_bin_boundaries():
    errors = np.array([0., .499, .5, .999, 1., 1.999, 2., 3.999, 4.])
    assert np.searchsorted(ERROR_THRESHOLDS, errors, side="right").tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4]


def test_identity_rigid_transform_and_perturbation():
    rng = np.random.default_rng(1); wt = rng.normal(size=(12, 3))
    i, j = reference_pairs(wt)
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    edited = wt @ rotation + [100., 5., -17.]
    labels = reference_preservation_labels(wt, edited, np.arange(12), i, j)
    assert np.max(labels["error"]) < 1e-12
    assert np.all(labels["target"] == 0)
    assert preservation_bounds(labels, len(i))["T_lower"] == 1
    edited[0] += [10., 0., 0.]
    labels = reference_preservation_labels(wt, edited, np.arange(12), i, j)
    assert np.any(labels["target"] > 0)


def test_missing_coordinate_is_unknown_fixed_denominator():
    wt = np.column_stack([np.arange(6), np.zeros((6, 2))]).astype(float)
    i, j = reference_pairs(wt); edited = wt.copy(); edited[0] = np.nan
    labels = reference_preservation_labels(wt, edited, np.arange(6), i, j)
    bounds = preservation_bounds(labels, len(i))
    assert bounds["C"] == 1
    assert bounds["unknown_pairs"] == 3
    assert bounds["G_lower"] < 1
    assert bounds["G_upper"] == bounds["T_upper"] == 1
    edited[:] = np.nan
    labels = reference_preservation_labels(wt, edited, np.arange(6), i, j)
    bounds = preservation_bounds(labels, len(i))
    assert bounds["G_lower"] == bounds["T_lower"] == 0
    assert bounds["G_upper"] == bounds["T_upper"] == 1


def test_deleted_endpoint_differs_from_missing_ca():
    wt = np.column_stack([np.arange(6), np.zeros((6, 2))]).astype(float)
    i, j = reference_pairs(wt); kept = np.array([1, 2, 3, 4, 5])
    labels = reference_preservation_labels(wt, wt[kept], kept, i, j)
    bounds = preservation_bounds(labels, len(i))
    assert bounds["unknown_pairs"] == 0
    assert bounds["C"] < 1 and bounds["G_lower"] == 1
    assert bounds["T_lower"] == bounds["C"]


def test_zero_retained_and_invalid_maps():
    wt = np.zeros((6, 3)); i, j = reference_pairs(wt)
    labels = reference_preservation_labels(wt, wt[:1], [0], i, j)
    bounds = preservation_bounds(labels, len(i))
    assert bounds["G_lower"] is None and bounds["T_lower"] == 0
    for kept in [[0, 0], [2, 1], [-1, 2], [0, 6]]:
        with pytest.raises(ValueError):
            reference_preservation_labels(wt, np.zeros((2, 3)), kept, i, j)
    with pytest.raises(ValueError):
        reference_preservation_labels(wt, wt, np.arange(6), [0, 0], [3, 3])


def test_reference_missing_not_in_fixed_relation_set():
    wt = np.zeros((6, 3)); wt[2] = np.nan
    i, j = reference_pairs(wt)
    assert not np.any(i == 2) and not np.any(j == 2)
