"""Physics, integration, estimator and scenario invariants for the new model."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from analysis.dynamic_campaign import configuration, resolved_case, run_episode
from analysis.dynamic_validation import steady_turn, validate_model
from analysis.frozen_decision import decision, inputs_signature, normal_draws
from dynamics.dynamic_vehicle import DynamicVehicle
from scenarios.dynamic_scenes import make_s_turn


@pytest.fixture(scope="module")
def validated():
    return validate_model(DynamicVehicle())


def test_full_physics_validation(validated):
    evidence, _ = validated
    assert evidence["passed"], evidence["checks"]
    assert all(evidence["checks"].values())
    assert evidence["integration_errors"][0] < 0.001


def test_state_and_input_order_and_normalization():
    m = DynamicVehicle()
    x = np.array([0, 0, 0.3, 0, 2.0, 0.2, 0.1])
    dx = m.derivative(x, [0, 0])
    assert dx[0] == pytest.approx(2 * np.cos(0.3) - 0.2 * np.sin(0.3))
    assert dx[1] == pytest.approx(2 * np.sin(0.3) + 0.2 * np.cos(0.3))
    assert dx[2] == 0.1
    np.testing.assert_array_equal(m.control_scale, [0.5, 1])
    np.testing.assert_array_equal(m.clip_control([3, -3]), [0.5, -1])


def test_signs_and_dynamic_roll_response(validated):
    _, traces = validated
    x = np.array(traces["lateral_yaw_roll_transient"]["states"])
    assert x[1, 6] > 0 and x[1, 5] > 0 and x[1, 3] > 0
    assert x[1, 3] < x[8, 3]
    assert not np.allclose(x[1:, 3], x[-1, 3])


def test_dynamic_turn_reference_is_not_kinematic_oracle():
    m = DynamicVehicle()
    oracle = steady_turn(m, 2.0, 0.15)
    assert oracle["force_balance_residual"] < 1e-8
    assert abs(oracle["yaw_rate"] - 2 * np.tan(0.15) / m.wheelbase) > 1e-3


def test_physical_parameter_validation():
    for values in ({"mass": -1}, {"yaw_inertia": 0}, {"substeps": 1}, {"steering_limit": 2}):
        with pytest.raises(ValueError):
            DynamicVehicle(**values)


def test_low_speed_perturbations_and_extreme_controls_remain_finite():
    m = DynamicVehicle()
    x = np.array([0, 0, 0, 0.01, 1e-9, 0.01, 0.01])
    for _ in range(100):
        x = m.step(x, [0.5, -1])
    assert np.isfinite(x).all() and x[4] >= 0
    assert abs(x[5]) < 0.001 and abs(x[6]) < 0.001


def test_substep_path_is_same_integrator():
    m = DynamicVehicle()
    x = np.array([0, 0, 0, 0, 1.5, 0, 0.0])
    end, path = m.step(x, [0.2, 0.2], return_path=True)
    np.testing.assert_array_equal(end, m.step(x, [0.2, 0.2]))
    np.testing.assert_array_equal(path[0], x)
    np.testing.assert_array_equal(path[-1], end)
    assert path.shape == (4, 7)
    np.testing.assert_allclose(
        path[1], replace(m, dt=0.05, substeps=1).step(x, [0.2, 0.2]), atol=1e-14
    )


def test_seven_state_mppi_uses_existing_update_and_correct_density():
    c = configuration()
    c, g, *_ = resolved_case(c, "open_space", samples=16)
    c["controller"]["horizon"] = 5
    c["controller"].update(kind="full", covariance=c["proposals"]["full"], snapshot_steps=[0])
    frozen = dict(
        config=c,
        state=c["scenario"]["initial_state"],
        warm_start=np.zeros((5, 2)).tolist(),
        previous_control=[0, 0],
    )
    z = normal_draws(0, 1, 16, 5)
    mean, stats, snap = decision(frozen, c["proposals"]["full"], z)
    v = snap["latent_controls"]
    base = np.array(c["controller"]["base_covariance"])
    proposal = c["proposals"]["full"]
    expected = sum(
        multivariate_normal.logpdf(v[:, i], mean=[0, 0], cov=base)
        - multivariate_normal.logpdf(v[:, i], mean=[0, 0], cov=proposal)
        for i in range(5)
    )
    np.testing.assert_allclose(expected, snap["log_base_over_proposal"], atol=1e-12)
    np.testing.assert_allclose(mean, np.einsum("k,kti->ti", snap["weights"], v), atol=1e-13)
    assert snap["sampled_states"].shape == (16, 6, 7)
    assert "first_throttle" in stats and "first_acceleration" not in stats
    assert stats["rhs_evaluations"] == (16 + 1) * 5 * 6


def test_frozen_inputs_and_paired_randomness():
    c = configuration()
    c, _, *_ = resolved_case(c, "open_space", samples=12)
    c["controller"]["horizon"] = 4
    value = dict(
        config=c,
        state=[0, 0, 0, 0, 1.4, 0, 0],
        warm_start=np.zeros((4, 2)).tolist(),
        previous_control=[0, 0],
    )
    before = deepcopy(value)
    z = normal_draws(0, 1, 12, 4)
    a = decision(value, c["proposals"]["standard"], z)[2]
    b = decision(value, c["proposals"]["full"], z)[2]
    np.testing.assert_allclose(
        a["latent_controls"], z @ np.linalg.cholesky(c["proposals"]["standard"]).T
    )
    assert not np.array_equal(a["latent_controls"], b["latent_controls"])
    assert value == before and inputs_signature(value) == inputs_signature(before)


def test_s_turn_feasibility_and_actual_polygon_collision():
    c = configuration()
    for clearance in c["s_turn"]["clearance_levels"]:
        _, g, x, u, check = make_s_turn(c, clearance)
        assert check["feasible"] and check["steering_reversals"] == 2
        assert check["min_swept_clearance"] > clearance - 0.004
        assert g.transition_clearance(x[0], x[-1]) < 0
        assert check["max_abs_roll"] > 0 and check["max_abs_vy"] > 0


def test_mismatch_is_a_distinct_execution_model():
    c = configuration()
    c, _, *_ = resolved_case(c, "model_mismatch")
    p = DynamicVehicle(**c["vehicle"])
    e = DynamicVehicle(**(c["vehicle"] | c["scenario"]["execution_overrides"]))
    x = np.array([0, 0, 0, 0, 2, 0, 0.0])
    assert not np.allclose(p.step(x, [0.2, 0.2]), e.step(x, [0.2, 0.2]))


def test_episode_replay_and_disturbance_stream(tmp_path):
    c = configuration()
    c, _, *_ = resolved_case(c, "disturbance", samples=12)
    c["controller"]["horizon"] = 4
    c["execution"]["max_steps"] = 3
    a, ta, _ = run_episode(c, "standard", 42, tmp_path / "a")
    b, tb, _ = run_episode(c, "standard", 42, tmp_path / "b")
    a.pop("mean_planning_latency_s")
    b.pop("mean_planning_latency_s")
    assert a == b
    np.testing.assert_array_equal(ta.to_numpy(), tb.to_numpy())
    assert a["optimizer_entropy"] != a["disturbance_entropy"]


def test_no_obstacle_diagnostics_are_json_serializable(tmp_path):
    import json

    from analysis.dynamic_campaign import job

    c = configuration()
    c, _, *_ = resolved_case(c, "open_space", samples=8)
    c["controller"]["horizon"] = 3
    c["execution"]["max_steps"] = 2
    row, record = job((c, "open_space", "standard", 42, "pilot", str(tmp_path), []))
    json.dumps(row, allow_nan=False)
    json.dumps(record, allow_nan=False)
    assert record[1]["clearance"] == [None, None]


def test_secondary_speed_changes_objective_without_regenerating_corridor():
    from analysis.dynamic_campaign import configuration, resolved_case
    from evaluation.runner import make_controller

    config, geometry, *_ = resolved_case(configuration(), "s_turn", clearance=0.35)
    config["controller"].update(kind="standard", covariance=config["proposals"]["standard"])
    baseline = make_controller(config)
    config["running_speed_target"] = 1.8
    secondary = make_controller(config)
    assert baseline.cost.speed == 1.4 and secondary.cost.speed == 1.8
    np.testing.assert_array_equal(secondary.cost.geometry.center.points, geometry.center.points)
    np.testing.assert_array_equal(secondary.cost.geometry.goal, baseline.cost.geometry.goal)


def test_discrete_command_slew_projects_samples_nominal_and_execution():
    from analysis.dynamic_campaign import configuration, resolved_case
    from evaluation.runner import make_controller
    from sampling.gaussian import importance_weights, log_base_over_proposal

    config, *_ = resolved_case(configuration(), "s_turn", samples=32, clearance=0.35)
    config["controller"].update(
        kind="full",
        covariance=config["proposals"]["full"],
        steering_rate_limit=0.4,
        snapshot_steps=[0],
    )
    controller = make_controller(config)
    controller.previous_control = np.array([0.1, 0.2])
    state = np.array(config["scenario"]["initial_state"])
    z = np.random.default_rng(2).standard_normal((32, 20, 2))
    previous = controller.previous_control.copy()
    mean = controller.mean.copy()
    command, updated = controller.plan(state, standard_normals=z)
    snapshot = controller.get_rollout_snapshot()
    physical = snapshot["sampled_controls"]
    history = np.concatenate([np.full((32, 1), previous[0]), physical[:, :, 0]], axis=1)
    assert np.max(abs(np.diff(history, axis=1))) <= 0.4 * controller.model.dt + 1e-14
    assert abs(command[0] - previous[0]) <= 0.4 * controller.model.dt + 1e-14
    nominal = controller.project_controls(updated)
    replay = state.copy()
    for t, control in enumerate(nominal):
        replay = controller.model.step(replay, control)
        np.testing.assert_allclose(replay, snapshot["nominal_states"][t + 1], atol=1e-14)
    ratio = log_base_over_proposal(
        snapshot["latent_controls"], mean, controller.base_covariance, controller.covariance
    )
    weights = importance_weights(snapshot["costs"], controller.temperature, ratio)
    np.testing.assert_allclose(weights, snapshot["weights"], atol=1e-14)
    np.testing.assert_allclose(
        updated, np.einsum("k,kti->ti", weights, snapshot["latent_controls"])
    )
    np.testing.assert_array_equal(controller.previous_control, previous)
    np.testing.assert_array_equal(controller.mean, mean)


def test_unconstrained_pushforward_is_unchanged():
    from analysis.dynamic_campaign import configuration, resolved_case
    from evaluation.runner import make_controller

    config, *_ = resolved_case(configuration(), "open_space", samples=8)
    config["controller"].update(kind="standard", covariance=config["proposals"]["standard"])
    controller = make_controller(config)
    latent = np.random.default_rng(7).normal(size=(8, 20, 2)) * 3
    np.testing.assert_array_equal(
        controller.project_controls(latent),
        controller.model.clip_control(latent * controller.model.control_scale),
    )


def test_axle_capsule_is_stricter_than_center_disk_and_reference_fits():
    from analysis.dynamic_campaign import configuration
    from analysis.footprint_validation import capsule_clearance, reference_check
    from scenarios.dynamic_scenes import Circles

    model = DynamicVehicle()
    state = np.zeros(7)
    goal = state.copy()
    goal[0] = 1.0
    geometry = Circles([[0.7, 0.0, 0.1]], 0.22, state, goal)
    assert geometry.state_clearance(state) > 0
    assert capsule_clearance(geometry, model, state) < 0
    result = reference_check(configuration(), 0.2)
    assert result["conservative_interpolated_capsule_clearance_m"] > 0.09
