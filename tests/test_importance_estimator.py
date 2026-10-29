import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from analysis.analytic_gaussian import exact_target, quadratic_cost
from analysis.gaussian_study import cases_for, run_gaussian
from analysis.importance_sampling import (
    estimate,
    gaussian_log_weight_second_moment,
    gaussian_logpdf,
)
from analysis.plots import generate
from utils.config import read_yaml


def small_config():
    config = read_yaml("configs/investigations/gaussian_target.yaml")
    config.update(
        evaluation_seeds=[1000, 1001, 1002, 1003],
        sample_counts=[32, 256],
        geometry_samples=32,
        anisotropy_ratios=[1, 4],
        angles_deg=[0, 45, 90],
        cost_angles_deg=[0, 45],
        scales=[1.0],
        proposal_means=[[0.0, 0.0]],
    )
    return config


@pytest.mark.parametrize("dimension", [2, 8, 56])
def test_independent_gaussian_density_oracle(dimension):
    rng = np.random.default_rng(11)
    factor = rng.normal(size=(dimension, dimension))
    cov = factor @ factor.T + np.eye(dimension)
    mean = rng.normal(size=dimension)
    z = rng.normal(size=(20, dimension))
    np.testing.assert_allclose(
        gaussian_logpdf(z, mean, cov), multivariate_normal.logpdf(z, mean=mean, cov=cov), atol=1e-10
    )


def test_sampling_moments_weights_and_reproducible_large_sample_accuracy():
    p = np.diag([1.0, 0.7])
    h = np.array([[3, 1], [1, 2]])
    b = np.array([0.6, 0.1])
    target, target_cov, _ = exact_target(p, h, b, 1)
    cov, mean = target_cov * 2, target + 0.1
    errors = {32: [], 32768: []}
    for seed in range(16):
        normal = np.random.default_rng(seed).normal(size=(32768, 2))
        z = mean + normal @ np.linalg.cholesky(cov).T
        for k in errors:
            mu, w, diagnostics = estimate(z[:k], quadratic_cost(z[:k], h, b), mean, p, cov, 1)
            assert w.sum() == pytest.approx(1)
            assert 1 <= diagnostics["ess"] <= k + 1e-8
            errors[k].append(np.linalg.norm(mu - target) ** 2)
        np.testing.assert_allclose(z.mean(axis=0), mean, atol=0.014)
        np.testing.assert_allclose(np.cov(z, rowvar=False), cov, atol=0.02)
    # Compare repeated-batch MSE, rather than demanding monotonic error in each sample path.
    assert np.mean(errors[32768]) < np.mean(errors[32]) / 100


def test_frozen_target_and_temperature_specific_targets():
    cases = cases_for(small_config(), ("budget", "geometry", "temperature"))
    fixed = [c for c in cases if c["suite"] != "temperature"]
    assert len({c["target_sha256"] for c in fixed}) == 1
    by_temp = {}
    for case in cases:
        if case["suite"] == "temperature":
            by_temp.setdefault(case["temperature"], set()).add(case["target_sha256"])
    assert all(len(hashes) == 1 for hashes in by_temp.values())
    assert len(set.union(*by_temp.values())) == len(by_temp)


def test_artifacts_bias_variance_pairing_replay_and_saved_data_figures(tmp_path, monkeypatch):
    config = small_config()
    first = run_gaussian(config, tmp_path / "a")
    second = run_gaussian(config, tmp_path / "b")
    pd.testing.assert_frame_equal(first, second)
    manifest = json.loads((tmp_path / "a/manifest.json").read_text())
    assert manifest["complete"] and manifest["schema_version"] == 1
    assert len(manifest["source_sha256"]) == 64
    summary = pd.read_csv(tmp_path / "a/summary.csv")
    np.testing.assert_allclose(
        summary.squared_error_mean,
        summary.bias_norm**2 + (summary.n - 1) / summary.n * summary.estimator_variance_trace,
        atol=1e-14,
    )
    pairs = pd.read_csv(tmp_path / "a/paired_differences.csv")
    assert not pairs.empty and (pairs.n == 4).all()
    monkeypatch.setattr(
        np.random, "default_rng", lambda *args: pytest.fail("Plotting resampled data")
    )
    generate(tmp_path / "a")
    for prefix in ("01_", "02_", "03_", "04_"):
        assert list((tmp_path / "a/figures").glob(prefix + "*.png"))
        assert list((tmp_path / "a/figures").glob(prefix + "*.svg"))
        assert list((tmp_path / "a/figures").glob(prefix + "*_data.npz"))
    with pytest.raises(FileExistsError):
        run_gaussian(config, tmp_path / "a")


def test_reject_overlapping_tuning_and_evaluation_seeds(tmp_path):
    config = deepcopy(small_config())
    config["evaluation_seeds"] = [0, 1]
    with pytest.raises(ValueError):
        run_gaussian(config, tmp_path / "bad")


def test_population_overlap_finite_and_infinite_second_moments():
    mean = np.zeros(2)
    assert gaussian_log_weight_second_moment(mean, np.eye(2), mean, np.eye(2)) == pytest.approx(0)
    assert gaussian_log_weight_second_moment(mean, np.eye(2), mean, np.eye(2) * 0.4) is None
    # Product of two independent scalar integrals, each sqrt(4/3).
    value = gaussian_log_weight_second_moment(mean, np.eye(2), mean, np.eye(2) * 2)
    assert np.exp(value) == pytest.approx(4 / 3)
