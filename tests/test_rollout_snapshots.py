import numpy as np
import pandas as pd
import pytest

from analysis.rollout_snapshots import load_snapshot, save_snapshot, validate_snapshot
from evaluation.runner import make_controller, run_episode
from utils.config import load_experiment


def test_full_snapshot_roundtrip_and_capture_does_not_change_actions(tmp_path):
    config = load_experiment("full")
    config["controller"].update(samples=48, horizon=6)
    normal = make_controller(config)
    config["controller"]["snapshot_steps"] = [0, 2]
    recorded = make_controller(config)
    state = np.array(config["scenario"]["initial_state"])
    for step in range(3):
        np.testing.assert_array_equal(normal.command(state), recorded.command(state))
        snapshot = recorded.get_rollout_snapshot()
        if step == 1:
            assert snapshot is None
            continue
        assert snapshot["sampled_states"].shape == (48, 7, 4)
        assert len(snapshot["costs"]) == len(snapshot["weights"]) == 48
        snapshot.update(executed_next_state=state.copy(), chosen_control=np.zeros(2))
        path = tmp_path / f"step_{step}.npz"
        save_snapshot(path, snapshot)
        loaded = load_snapshot(path)
        np.testing.assert_allclose(loaded["sampled_states"], snapshot["sampled_states"], atol=1e-6)
        np.testing.assert_array_equal(loaded["weights"], snapshot["weights"])
        np.testing.assert_array_equal(loaded["costs"], snapshot["costs"])
        assert loaded["sample_count"] == 48 and loaded["best_by"] == "raw_trajectory_cost"
        with pytest.raises(FileExistsError):
            save_snapshot(path, snapshot)
        loaded["weights"][0] += 1
        with pytest.raises(ValueError):
            validate_snapshot(loaded)
    recorded.reset(state, 42)
    assert recorded.get_rollout_snapshot() is None


@pytest.mark.parametrize("variant", ["standard", "isotropic", "diagonal", "full"])
def test_observer_and_variable_snapshot_sizes_preserve_complete_episode(variant, tmp_path):
    config = load_experiment(variant, "disturbance")
    config["controller"].update(samples=40, horizon=8)
    config["scenario"]["max_steps"] = 24
    original = run_episode(config, 42, tmp_path / "original")
    config["controller"]["snapshot_steps"] = [0, 10, 21]
    events = []

    def observe(event):
        events.append(event)
        event["state"][:] = 900  # Observer copies must not corrupt execution.
        event["control"][:] = 900

    recorded = run_episode(config, 42, tmp_path / "recorded", observer=observe)
    for name in ["trajectory.csv", "controls.csv"]:
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "original" / name),
            pd.read_csv(tmp_path / "recorded" / name),
            check_exact=True,
        )
    assert (
        original["metrics"]["cumulative_evaluation_cost"]
        == recorded["metrics"]["cumulative_evaluation_cost"]
    )
    assert [event["step"] for event in events if event["snapshot"] is not None] == [0, 10, 21]
    with np.load(tmp_path / "recorded/rollouts.npz") as compact:
        assert compact["rollouts"].shape[1] == 32


@pytest.mark.parametrize("steps", [[-1], [0.5], [True]])
def test_invalid_snapshot_steps(steps):
    config = load_experiment()
    config["controller"]["snapshot_steps"] = steps
    with pytest.raises(ValueError):
        make_controller(config)
