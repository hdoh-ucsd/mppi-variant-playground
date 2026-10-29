import numpy as np
import pytest

from analysis.covariance_geometry import alignment, covariance, rotation


@pytest.mark.parametrize("angle", [-45, 0, 15, 30, 45, 60, 75, 90])
def test_rotation_and_spd_scale_constraints(angle):
    r = rotation(angle)
    np.testing.assert_allclose(r.T @ r, np.eye(2), atol=1e-15)
    for ratio in (1, 2, 4, 8):
        trace = covariance(ratio, angle, scale=2, constraint="trace")
        det = covariance(ratio, angle, scale=2, constraint="determinant")
        assert np.linalg.eigvalsh(trace).min() > 0
        assert np.trace(trace) == pytest.approx(2.4)
        assert np.linalg.det(det) == pytest.approx(1.2**2)
        eigen = np.linalg.eigvalsh(trace)
        assert eigen[-1] / eigen[0] == pytest.approx(ratio)


def test_alignment_distinguishes_axis_orientation_and_scale():
    target = covariance(4, 30)
    assert alignment(target, target)["target_axis_angle_deg"] == pytest.approx(0, abs=1e-6)
    assert alignment(covariance(4, 120), target)["target_axis_angle_deg"] == pytest.approx(90)
    assert alignment(covariance(), target)["target_axis_angle_deg"] is None
    assert alignment(target * 2, target)["covariance_kl"] > 0


@pytest.mark.parametrize("kwargs", [{"ratio": 0.5}, {"scale": 0}, {"constraint": "unknown"}])
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        covariance(**kwargs)
