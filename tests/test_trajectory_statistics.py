import numpy as np
import pytest

from analysis.trajectory_statistics import first_crossing, occupancy, progress_dispersion


def test_occupancy_seed_and_time_weighting_normalization():
    first = np.array([[0.25, 0.25]])
    second = np.repeat([[1.25, 0.25]], 3, axis=0)
    x, y = [0, 1, 2], [0, 1]
    equal = occupancy([first, second], x, y, weighting="equal_seed", smoothing=0)
    pooled = occupancy([first, second], x, y, weighting="pooled_time", smoothing=0)
    np.testing.assert_allclose(equal["mass"], [[0.5, 0.5]])
    np.testing.assert_allclose(pooled["mass"], [[0.25, 0.75]])
    blurred = occupancy([first, second], x, y, smoothing=1)
    assert blurred["mass"].sum() == pytest.approx(1)
    assert np.sum(blurred["density"] * np.diff(y)[:, None] * np.diff(x)[None, :]) == pytest.approx(
        1
    )
    with pytest.raises(ValueError):
        occupancy([np.array([[3.0, 0.0]])], x, y)


def test_arc_length_quantiles_do_not_pad_failed_or_stationary_runs():
    paths = [np.array([[0, 0], [1, 0], [2, 0]]), np.array([[0, 1], [1, 1]])]
    summary = progress_dispersion(paths, 3)
    np.testing.assert_array_equal(summary["aligned_xy"][1, -1], [1, 1])
    np.testing.assert_allclose(summary["q25_xy"][:, 1], 0.25)
    np.testing.assert_allclose(summary["q75_xy"][:, 1], 0.75)
    constant = progress_dispersion([np.array([[2, 3], [2, 3]])], 5)
    np.testing.assert_array_equal(constant["mean_xy"], np.tile([2, 3], (5, 1)))
    assert first_crossing(np.array([[0, -1], [2, 1]]), 1) == pytest.approx(0)
    assert first_crossing(np.array([[0, -1], [0.5, 1]]), 1) is None
