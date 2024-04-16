"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [5, 6]),
        ([[-8, 2, 2], [3, 4, -9], [-4, 5, 6]], [3, 5, 6]),
    ])
def test_daily_max(test, expected):
    """Test that max function works for an array of equal, unequal
    positive and general unequal integers."""
    from inflammation.models import daily_max
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0], None),
        ([[1, 2], [3, 4], [5, 6]], [1, 2], None),
        ([[-8, 2, 2], [3, 4, -9], [-4, 5, 6]], [-8, 2, -9], None),
        ([['Hello', 'there'], ['General', 'Kenobi']], [], TypeError)
    ])
def test_daily_min(test, expected, expect_raises):
    """Test that min function works for an array of equal,
    unequal positive and general unequal integers."""
    from inflammation.models import daily_min
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))
    else:
        npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None,
        ),
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[1, 2, 3], [4, 5, float('nan')], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.8, 1, 0], [0.78, 0.89, 1]],
            None,
        ),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        ),
        (
            [1, 2, 3],
            [0.33, 0.67, 1],
            TypeError,
        ),
        (
            'aardappels',
            [1],
            TypeError,
        ),
        (
            [[[1]]],
            [[[1]]],
            TypeError
        )
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
       Assumption that test accuracy of two decimal places is sufficient."""
    from inflammation.models import patient_normalise
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)
