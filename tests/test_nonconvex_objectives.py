import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from analysis.nonconvex_objectives import grid_reference, minimizer, objective
from analysis.nonconvex_study import run_nonconvex
from analysis.plots import generate
from utils.config import read_yaml


@pytest.mark.parametrize("name", ["curved_valley", "two_wells"])
def test_nonnegative_objectives_and_known_minimizers(name):
    rng = np.random.default_rng(42)
    points = rng.normal(size=(200, 2))
    assert np.all(objective(points, name) >= 0)
    assert objective(minimizer(name), name) == pytest.approx(0, abs=1e-25)
    np.testing.assert_allclose(objective(points, name), [objective(z, name) for z in points])


def test_multimodal_reference_against_piecewise_adaptive_quadrature():
    def density(x):
        cost = min((x - 1) ** 2, 0.35 + (x + 1) ** 2)
        return norm.pdf(x, scale=2) * np.exp(-cost / 0.2)

    boundary = -0.35 / 4
    denominator = sum(quad(density, a, b)[0] for a, b in ((-10, boundary), (boundary, 10)))
    numerator = sum(
        quad(lambda x: x * density(x), a, b)[0] for a, b in ((-10, boundary), (boundary, 10))
    )
    reference = grid_reference("two_wells", 0.2, 4, resolution=801)
    np.testing.assert_allclose(reference["mean"], [numerator / denominator, 0], atol=1e-4)
    # Mean between wells is distinct from the global objective minimizer.
    assert np.linalg.norm(np.array(reference["mean"]) - minimizer("two_wells")) > 0.2


def test_adaptive_iterations_preserve_data_and_evaluation_counts(tmp_path):
    config = read_yaml("configs/investigations/nonconvex.yaml")
    config.update(
        evaluation_seeds=[1000, 1001],
        samples=32,
        iterations=3,
        initial_means=[[-1.2, -0.8]],
        reference_resolution=201,
    )
    frame = run_nonconvex(config, tmp_path / "run")
    assert (frame.cost_evaluations == 1 + 3 * 33).all()
    history = pd.read_csv(tmp_path / "run/iterations.csv")
    for _, group in history.groupby(["case_id", "seed"]):
        assert len(group) == 4
        assert np.all(np.diff(group.best_cost) <= 1e-14)
    generate(tmp_path / "run")
    assert (tmp_path / "run/figures/06_nonconvex_iterations.svg").exists()
