import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from analysis.plots import generate
from analysis.saturation import clipped_marginal_moments, run_saturation, saturation_statistics
from utils.config import read_yaml


@pytest.mark.parametrize("mean,variance", [(0.0, 0.22), (0.8, 0.6), (-1.5, 0.3)])
def test_clipped_moments_against_independent_integrals(mean, variance):
    first, var, left, right = clipped_marginal_moments(mean, variance)
    sd = np.sqrt(variance)

    def density(z):
        return norm.pdf(z, loc=mean, scale=sd)

    f = quad(lambda z: z * density(z), -1, 1)[0] - left + right
    second = quad(lambda z: z * z * density(z), -1, 1)[0] + left + right
    assert first == pytest.approx(f, abs=1e-10)
    assert var == pytest.approx(second - f * f, abs=1e-10)


def test_boundary_atoms_and_physical_statistics():
    latent = np.array([[-2, 2], [-0.5, 0.5], [0.5, -0.5], [2, -2]])
    applied, result = saturation_statistics(latent, [2, 0.5])
    np.testing.assert_array_equal(applied, [[-2, 0.5], [-1, 0.25], [1, -0.25], [2, -0.5]])
    assert result["saturation_fraction"] == 0.5
    assert result["lower_boundary_mass_0"] == 0.25
    assert result["upper_boundary_mass_1"] == 0.25


def test_saturation_artifacts_and_figures(tmp_path):
    config = read_yaml("configs/investigations/saturation.yaml")
    config.update(evaluation_seeds=[1000, 1001], samples=128, scales=[2.0])
    frame = run_saturation(config, tmp_path / "run")
    assert len(frame) == 20
    generate(tmp_path / "run")
    assert (tmp_path / "run/figures/07_saturation_distributions.png").exists()
