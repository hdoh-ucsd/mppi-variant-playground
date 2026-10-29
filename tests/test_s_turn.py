"""Geometry oracles, explicit steering semantics and replay invariants."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import yaml

from analysis.s_turn_study import condition, run_episode, run_study
from dynamics.steering_rate import make_model
from scenarios.s_turn import Polyline, feasibility, point_segments, segment_distances
from utils.config import read_yaml


def config():
    return read_yaml("configs/investigations/s_turn.yaml")


def test_explicit_euler_rate_dynamics_and_limits():
    m = make_model(config()["vehicle"])
    state = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    new = m.step(state, [0.0, 0.8])
    assert new[2] == 0 and new[4] == pytest.approx(0.12)
    assert m.step(new, [0, 0])[2] == pytest.approx(0.15 * np.tan(0.12))
    saturated = m.step(np.array([0, 0, 0, 1, 0.54]), [100, 100])
    assert saturated[4] == pytest.approx(0.55)
    assert saturated[3] == pytest.approx(1.3)
    np.testing.assert_array_equal(state, [0, 0, 0, 1, 0])


def test_all_predeclared_difficulties_have_a_feasible_reference():
    c = config()
    for clearance in c["corridor"]["clearance_levels"]:
        corridor, states, controls, d = feasibility(c, clearance)
        assert d["feasible"] and d["steering_reversals"] == 2
        assert d["min_clearance"] > clearance - 0.003
        assert d["max_steering_rate"] <= c["vehicle"]["steering_rate_limit"]
        assert d["max_steering"] <= c["vehicle"]["steering_limit"]
        assert len(controls) <= c["execution"]["max_steps"]
        assert states[-1, 3] == pytest.approx(c["corridor"]["speed_target"])
        assert corridor.center.arc[-1] > np.linalg.norm(states[-1, :2] - states[0, :2])
    c["vehicle"]["steering_rate_limit"] = 0.4
    assert feasibility(c, 0.35)[3]["feasible"]
    c["vehicle"]["steering_rate_limit"] = 0.1
    with pytest.raises(ValueError, match="Reference rate"):
        feasibility(c, 0.35)


def test_exact_kd_screen_matches_exhaustive_geometry():
    rng = np.random.default_rng(8)
    line = Polyline(np.c_[np.linspace(0, 10, 51), np.sin(np.linspace(0, 6, 51))])
    points = rng.uniform([-3, -4], [14, 5], (80, 2))
    end = points + rng.normal(size=(80, 2))
    expected = point_segments(points[:, None], line.a, line.b)[0].min(axis=1)
    np.testing.assert_allclose(line.nearest(points)[0], expected, atol=1e-12)
    expected = segment_distances(points[:, None], end[:, None], line.a, line.b).min(axis=1)
    np.testing.assert_allclose(line.nearest(points, end)[0], expected, atol=1e-12)


def test_corridor_boundaries_disk_and_swept_collision_not_waypoint_distance():
    corridor, states, _, _ = feasibility(config(), 0.2)
    assert corridor.state_clearance(states[0]) > 0 and corridor.state_clearance(states[-1]) > 0
    # Endpoints are inside; the chord through a curved corridor cuts the actual boundary.
    assert corridor.transition_clearance(states[0], states[-1]) < 0
    point = np.r_[corridor.upper[len(corridor.upper) // 2], 0.0, 0.0, 0.0]
    assert corridor.state_clearance(point) == pytest.approx(-0.22, abs=1e-12)
    outside = point.copy()
    outside[1] += 5
    assert corridor.state_clearance(outside) < 0


def test_primary_difficulty_changes_only_corridor_width():
    c = config()
    a = condition(c, 0.9, 256, 0.8)
    b = condition(c, 0.2, 256, 0.8)
    np.testing.assert_array_equal(a[2], b[2])
    np.testing.assert_array_equal(a[3], b[3])
    assert a[0]["vehicle"] == b[0]["vehicle"] and a[0]["cost"] == b[0]["cost"]
    assert a[4]["corridor_width"] > b[4]["corridor_width"]
    restored = yaml.safe_load(yaml.safe_dump(a[0]))
    assert restored == a[0]


def test_rate_episode_reproducibility_and_input_isolation(tmp_path):
    c = config()
    c["execution"]["max_steps"] = 4
    c["controller"].update(samples=12, horizon=5)
    c["execution"]["snapshot_steps"] = [0]
    resolved, corridor, *_ = condition(c, 0.35, 12, 0.8)
    before = deepcopy(resolved)
    a, ta = run_episode(resolved, corridor, "standard", 20000, tmp_path / "a")
    b, tb = run_episode(resolved, corridor, "standard", 20000, tmp_path / "b")
    a.pop("mean_planning_latency_s")
    b.pop("mean_planning_latency_s")
    assert a == b
    pd.testing.assert_frame_equal(ta, tb)
    assert resolved == before
    snap = tmp_path / "a/snapshots/step_0000.npz"
    with np.load(snap, allow_pickle=False) as data:
        assert data["sampled_states"].shape == (12, 6, 5)
        assert data["weights"].sum() == pytest.approx(1)


def test_small_run_and_figure_regeneration_without_simulation(tmp_path, monkeypatch):
    c = config()
    c.update(workers=1, seeds={"start": 20000, "count": 2})
    c["controller"].update(samples=12, horizon=4)
    c["execution"]["max_steps"] = 140
    # Feasibility still uses the full 122-step maneuver; evaluation stops after two commands.
    import analysis.s_turn_study as module

    original = module.run_episode

    def short(cfg, *args):
        cfg = deepcopy(cfg)
        cfg["execution"]["max_steps"] = 2
        return original(cfg, *args)

    monkeypatch.setattr(module, "run_episode", short)
    cells = [(0.35, 12, 0.8, "primary")]
    frame = run_study(c, tmp_path / "run", conditions=cells)
    assert len(frame) == 8
    assert frame.groupby("cell").shared_condition_sha256.nunique().max() == 1
    monkeypatch.setattr(np.random, "default_rng", lambda *a: pytest.fail("Figure resampled"))
    monkeypatch.setattr(module, "run_episode", lambda *a: pytest.fail("Figure executed"))
    from analysis.s_turn_plots import generate

    generate(tmp_path / "run", tmp_path / "Q05_key.png")
    assert (tmp_path / "Q05_key.png").exists()
    assert (tmp_path / "run/key_figure_data.json").exists()
