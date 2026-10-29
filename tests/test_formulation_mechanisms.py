"""Recording invariance and numerical checks for actual mechanism experiments."""

from copy import deepcopy

import numpy as np
import pytest

from analysis.dynamic_campaign import run_episode
from analysis.formulation_mechanisms import FrozenCovarianceExecution
from analysis.variant_campaign import case_config
from controllers.feedback import state_error
from evaluation.runner import make_controller
from utils.config import read_yaml


@pytest.mark.parametrize(
    "algorithm", ["it_mppi", "tsallis_vi", "dial_mpc", "tube_mppi", "robust_mppi", "cc_mppi"]
)
def test_capture_does_not_change_commands_rng_or_work(algorithm):
    options = read_yaml("configs/variants/benchmark.yaml")
    config, _ = case_config(options, "open_space", algorithm, 16)
    state = np.array(config["scenario"]["initial_state"])
    state[4] = 1.2
    plain = make_controller(config)
    config["controller"]["snapshot_steps"] = [0, 1]
    captured = make_controller(config)
    for ctrl in (plain, captured):
        ctrl.reset(state, 1234)
    for index in range(2):
        observation = state.copy()
        observation[1] += index * 0.7
        np.testing.assert_array_equal(plain.command(observation), captured.command(observation))
        np.testing.assert_array_equal(plain.mean, captured.mean)
        assert plain.model.transitions == captured.model.transitions
        assert plain.rng.bit_generator.state == captured.rng.bit_generator.state
        snap = captured.get_rollout_snapshot()
        assert np.isclose(snap["weights"].sum(), 1, atol=1e-12)
        if algorithm == "dial_mpc":
            assert snap["inner_sampled_states"].shape == (4, 16, 21, 7)
            np.testing.assert_allclose(snap["inner_weights"].sum(axis=1), 1)
            np.testing.assert_allclose(
                snap["inner_updated_mean"],
                np.einsum("ik,iktd->itd", snap["inner_weights"], snap["inner_latent_controls"]),
            )
            assert captured.diagnostics["dynamics_evaluations"] == 4 * 17 * 20
        if algorithm in {"tube_mppi", "robust_mppi"}:
            diag = captured.diagnostics
            np.testing.assert_allclose(
                diag["tracking_error_vector"], state_error(observation, diag["nominal_state"])
            )
            np.testing.assert_allclose(
                diag["tracking_correction"],
                diag["tracking_gains"][0] @ diag["tracking_error_vector"],
            )
            assert diag["tracking_control_norm"] == pytest.approx(
                np.linalg.norm(diag["tracking_correction"])
            )
        if algorithm == "cc_mppi":
            cov = snap["predicted_covariances"]
            np.testing.assert_allclose(cov, cov.transpose(0, 2, 1), atol=1e-12)
            assert np.linalg.eigvalsh(cov).min() >= -1e-10


def test_existing_runner_records_feedback_components_without_changing_execution(tmp_path):
    options = read_yaml("configs/variants/benchmark.yaml")
    config, _ = case_config(options, "open_space", "tube_mppi", 16)
    config["execution"].update(max_steps=3, extra_snapshot_seeds=[51000])
    config["scenario"]["disturbance"] = dict(step=0, delta=[0, 1, 0.1, 0, 0, 0, 0])
    config["controller"]["reset_threshold"] = 0
    first = run_episode(config, "tube_mppi", 51000, tmp_path / "plain")
    second = run_episode(config, "tube_mppi", 51000, tmp_path / "captured", capture_steps=[1])
    np.testing.assert_array_equal(first[1].to_numpy(), second[1].to_numpy())
    frame = second[2]
    assert frame.tracking_control_norm.max() > 0
    requested = frame[["tracking_correction_0", "tracking_correction_1"]].to_numpy()
    np.testing.assert_allclose(np.linalg.norm(requested, axis=1), frame.tracking_control_norm)
    planner = frame[["planner_control_0", "planner_control_1"]].to_numpy()
    command = frame[["commanded_steering", "commanded_throttle"]].to_numpy()
    applied = frame[["applied_correction_0", "applied_correction_1"]].to_numpy()
    np.testing.assert_allclose(command - planner, applied)


def test_independent_covariance_execution_uses_the_recorded_auxiliary_policy(tmp_path):
    options = read_yaml("configs/variants/benchmark.yaml")
    cfg, _ = case_config(options, "open_space", "cc_mppi", 16)
    state = np.array(cfg["scenario"]["initial_state"])
    state[4] = 1
    ctrl = make_controller(cfg)
    ctrl.reset(state, 55000)
    ctrl.plan(state)
    execution = deepcopy(cfg)
    execution["scenario"]["initial_state"] = state.tolist()
    execution["execution"].update(max_steps=20, stop_on_terminal=False)
    policies = []

    def factory(config):
        policy = FrozenCovarianceExecution(
            config, ctrl.mean, ctrl.a, ctrl.b, ctrl.gains, ctrl.covariance
        )
        policies.append(policy)
        return policy

    outcome, trajectory, diag = run_episode(
        execution, "fixed_cc_policy", 56000, tmp_path / "execution", controller_factory=factory
    )
    assert len(trajectory) == 21
    assert outcome["mean_ess"] is None  # Physical executions are not weighted populations.
    noise = np.array(policies[0].noise_history)
    expected_noise = (
        np.random.default_rng(np.random.SeedSequence([4400, 56000])).standard_normal((20, 2))
        @ np.linalg.cholesky(ctrl.covariance).T
    )
    np.testing.assert_array_equal(noise, expected_noise)
    auxiliary = np.zeros(7)
    for t in range(20):
        latent = ctrl.mean[t] + ctrl.gains[t] @ auxiliary + noise[t]
        command = ctrl.model.clip_control(latent * ctrl.model.control_scale)
        np.testing.assert_allclose(
            command, diag.loc[t, ["commanded_steering", "commanded_throttle"]].to_numpy(dtype=float)
        )
        auxiliary = ctrl.a[t] @ auxiliary + ctrl.b[t] @ noise[t]
