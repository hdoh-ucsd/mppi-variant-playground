"""Scientific invariants of one frozen decision, including saved-only figure regeneration."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal

from analysis import frozen_decision as study
from analysis.frozen_decision_plots import generate
from analysis.frozen_decision_report import generate_report
from evaluation.runner import write_json
from utils.config import load_experiment, read_yaml


@pytest.fixture
def frozen():
    config = load_experiment("standard", "open_space")
    config["controller"].update(horizon=4, samples=12)
    warm = np.array([[0.4, 0.1], [0.3, 0.2], [0.1, -0.2], [0.0, 0.0]])
    value = dict(
        name="open_space",
        config=config,
        state=[0.8, -0.3, 0.1, 1.4],
        warm_start=warm.tolist(),
        previous_control=[0.2, 0.1],
        step=8,
        source_seed=31415,
        route_witnesses={},
    )
    value["inputs_sha256"] = study.inputs_signature(value)
    return value


def test_identical_inputs_target_and_common_normal_transformation(frozen):
    config = read_yaml("configs/investigations/frozen_decision.yaml")
    proposals, _ = study.proposals(frozen, config)
    original = deepcopy(frozen)
    z = study.normal_draws(0, 10000, 12, 4)
    latent = []
    for proposal in proposals.values():
        covariance = proposal["covariance"]
        c = study.controller_at(frozen, covariance, 12)
        np.testing.assert_array_equal(c.base_covariance, [[0.28, 0], [0, 0.16]])
        np.testing.assert_array_equal(c.mean, frozen["warm_start"])
        assert c.temperature == 5.0
        mean, stats, snap = study.decision(frozen, covariance, z)
        expected_eps = z @ np.linalg.cholesky(covariance).T
        np.testing.assert_allclose(snap["latent_controls"], c.mean[None] + expected_eps)
        np.testing.assert_allclose(
            mean, np.einsum("k,kti->ti", snap["weights"], snap["latent_controls"])
        )
        assert stats["dynamics_evaluations"] == (len(z) + 1) * 4
        assert np.linalg.eigvalsh(covariance).min() > 0
        assert np.trace(covariance) == pytest.approx(0.44)
        latent.append(snap["latent_controls"])
    assert not np.array_equal(latent[0], latent[1])
    assert frozen == original
    assert study.inputs_signature(frozen) == frozen["inputs_sha256"]


def test_exact_gaussian_ratio_and_logspace_normalization(frozen):
    covariance = [[0.28, 0.08], [0.08, 0.16]]
    z = study.normal_draws(1, 10004, 12, 4)
    mean, _, snap = study.decision(frozen, covariance, z)
    v = snap["latent_controls"]
    p = sum(
        multivariate_normal.logpdf(v[:, i], mean=[0, 0], cov=[[0.28, 0], [0, 0.16]])
        for i in range(4)
    )
    q = sum(
        multivariate_normal.logpdf(v[:, i], mean=frozen["warm_start"][i], cov=covariance)
        for i in range(4)
    )
    np.testing.assert_allclose(snap["log_base_over_proposal"], p - q, atol=1e-12)
    logits = -snap["costs"] / 5 + p - q
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    np.testing.assert_allclose(snap["weights"], weights, atol=1e-13)
    assert weights.sum() == pytest.approx(1)
    np.testing.assert_allclose(mean, np.einsum("k,kti->ti", weights, v))


def test_planning_does_not_shift_or_advance_rng_or_execute(frozen, monkeypatch):
    monkeypatch.setattr(study.Environment, "step", lambda *a: pytest.fail("Executed environment"))
    controller = study.controller_at(frozen, [[0.28, 0], [0, 0.16]], 12)
    before_mean, before_previous = controller.mean.copy(), controller.previous_control.copy()
    state = np.array(frozen["state"])
    rng_state = deepcopy(controller.rng.bit_generator.state)
    z = study.normal_draws(0, 10001, 12, 4)
    command, mean = controller.plan(state, standard_normals=z)
    np.testing.assert_array_equal(state, frozen["state"])
    np.testing.assert_array_equal(controller.mean, before_mean)
    np.testing.assert_array_equal(controller.previous_control, before_previous)
    assert controller.planning_step == 0
    assert controller.rng.bit_generator.state == rng_state
    command2, mean2 = controller.plan(state, standard_normals=z)
    np.testing.assert_array_equal(command, command2)
    np.testing.assert_array_equal(mean, mean2)


def test_default_command_retains_same_decision_and_warm_shift(frozen):
    c = study.controller_at(frozen, [[0.28, 0], [0, 0.16]], 12)
    reference = study.controller_at(frozen, [[0.28, 0], [0, 0.16]], 12)
    command, mean = reference.plan(np.array(frozen["state"]))
    executed = c.command(np.array(frozen["state"]))
    np.testing.assert_array_equal(executed, command)
    np.testing.assert_array_equal(c.mean[:-1], mean[1:])
    np.testing.assert_array_equal(c.mean[-1], [0, 0])
    assert c.planning_step == 1


def test_reference_streams_are_disjoint_and_budget_samples_nested():
    a = study.normal_draws(0, 10000, 12, 4)
    np.testing.assert_array_equal(a, study.normal_draws(0, 10000, 12, 4))
    np.testing.assert_array_equal(a[:6], study.normal_draws(0, 10000, 6, 4))
    assert not np.array_equal(a, study.normal_draws(0, 10000, 12, 4, "reference"))
    assert not np.array_equal(a, study.normal_draws(1, 10000, 12, 4))
    assert not np.array_equal(
        study.normal_draws(0, 0, 12, 4, "reference"), study.normal_draws(0, 1, 12, 4, "reference")
    )


def test_snapshot_roundtrip_preserves_latent_update_and_weight_semantics(frozen, tmp_path):
    _, _, snap = study.decision(
        frozen, [[0.28, 0.08], [0.08, 0.16]], study.normal_draws(0, 1, 12, 4)
    )
    path = tmp_path / "decision.npz"
    study.save_decision(path, snap)
    loaded = study.load_decision(path)
    for key in ("costs", "weights", "updated_mean", "latent_controls", "command"):
        np.testing.assert_array_equal(loaded[key], snap[key])
    np.testing.assert_allclose(loaded["sampled_states"], snap["sampled_states"], atol=3e-7)
    assert "executed_next_state" not in loaded
    with pytest.raises(FileExistsError):
        study.save_decision(path, snap)
    bad = dict(snap, weights=np.full(12, np.nan))
    with pytest.raises(ValueError, match="weights"):
        study.save_decision(tmp_path / "bad.npz", bad)


def test_local_alignment_uses_cost_sensitivity_not_xy_geometry(frozen):
    config = read_yaml("configs/investigations/frozen_decision.yaml")
    p, sensitivity = study.proposals(frozen, config)
    u = np.array(sensitivity["direction"])
    np.testing.assert_allclose(np.array(p["aligned"]["covariance"]) @ u, 0.34 * u)
    np.testing.assert_allclose(np.array(p["misaligned"]["covariance"]) @ u, 0.10 * u)
    assert p["aligned"]["determinant"] == pytest.approx(p["misaligned"]["determinant"])
    assert sensitivity["relative_gradient_change"] < 0.01


def test_reference_pooling_and_uncertainty_are_explicit(frozen, tmp_path):
    config = read_yaml("configs/investigations/frozen_decision.yaml")
    config.update(horizon=4, reference_samples=32, reference_repetitions=3)
    mean, status = study.reference_estimate(frozen, 0, config, tmp_path / "ref")
    with np.load(tmp_path / "ref/estimate.npz") as data:
        np.testing.assert_allclose(
            mean, np.einsum("r,rti->ti", data["block_masses"], data["repetition_means"])
        )
        assert np.isfinite(data["standard_error"]).all()
    assert status["status"] == "unresolved"
    assert len(status["seed_sequences"]) == 3
    assert status["total_rollouts"] == 96


def test_real_small_suite_reproducibility_and_saved_only_figures(frozen, tmp_path, monkeypatch):
    config = read_yaml("configs/investigations/frozen_decision.yaml")
    config.update(
        horizon=4,
        primary_samples=12,
        sample_counts=[6, 12],
        seeds={"start": 10000, "count": 2},
        reference_samples=32,
        reference_repetitions=3,
        repetition_counts=[2],
    )

    def capture(_, output):
        output.mkdir()
        write_json(output / "open_space.json", frozen)
        return [deepcopy(frozen)]

    monkeypatch.setattr(study, "capture_states", capture)
    a = study.run_study(config, tmp_path / "a")
    b = study.run_study(config, tmp_path / "b")
    ignore = ["planning_latency_s", "decision_and_scoring_latency_s"]
    pd.testing.assert_frame_equal(a.drop(columns=ignore), b.drop(columns=ignore))
    assert len(a) == 24
    assert a.inputs_sha256.nunique() == 1
    monkeypatch.setattr(study.Environment, "step", lambda *a: pytest.fail("Executed"))
    monkeypatch.setattr(np.random, "default_rng", lambda *a: pytest.fail("Resampled"))
    monkeypatch.setattr(study, "decision", lambda *a, **k: pytest.fail("Reoptimized"))
    generate(tmp_path / "a")
    generate_report(tmp_path / "a")
    assert (tmp_path / "a/interpretation.md").exists()
    orientation = pd.read_csv(tmp_path / "a/orientation_comparison.csv")
    assert len(orientation) == 2
    assert len(list((tmp_path / "a/figures").glob("*.png"))) == 6
    assert len(list((tmp_path / "a/figures").glob("*.svg"))) == 6
    assert len(list((tmp_path / "a/figures").glob("*_data.npz"))) == 6


def test_invalid_gaussian_draws_are_rejected(frozen):
    with pytest.raises(ValueError, match="standard normals"):
        study.decision(frozen, [[0.28, 0], [0, 0.16]], np.zeros((12, 3, 2)))
    with pytest.raises(ValueError, match="positive definite"):
        study.decision(frozen, [[1, 2], [2, 1]], np.zeros((12, 4, 2)))


def test_sampled_and_mean_plan_objectives_use_the_same_shared_cost(frozen):
    _, stats, snap = study.decision(
        frozen, [[0.28, 0.08], [0.08, 0.16]], study.normal_draws(0, 7, 12, 4)
    )
    for i in (0, 5, 11):
        states, value = study.predict(frozen, snap["latent_controls"][i])
        np.testing.assert_array_equal(states, snap["sampled_states"][i])
        assert value == pytest.approx(snap["costs"][i], abs=1e-12)
    _, value = study.predict(frozen, snap["updated_mean"])
    assert value == pytest.approx(stats["nominal_objective"], abs=1e-12)
