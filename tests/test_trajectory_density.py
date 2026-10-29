import json

import numpy as np
import pandas as pd
import pytest

from analysis.trajectory_density import run_study
from analysis.trajectory_plots import generate
from utils.config import read_yaml


def small_config():
    config = read_yaml("configs/investigations/trajectory_behavior.yaml")
    config.update(
        scenarios=["narrow_passage"],
        seeds=[42, 43],
        samples=24,
        horizon=6,
        max_steps=4,
        snapshot_steps=[0, 1, 2],
        representative_step=1,
        occupancy_bins=[12, 10],
        progress_points=11,
    )
    return config


def test_recording_schema_reproduction_and_figures_from_saved_data(tmp_path, monkeypatch):
    config = small_config()
    first = run_study(config, tmp_path / "a")
    second = run_study(config, tmp_path / "b")
    ignore = [
        "mean_planning_latency_s",
        "p95_planning_latency_s",
        "total_simulation_wall_time_s",
        "process_peak_rss_mib",
    ]
    pd.testing.assert_frame_equal(first.drop(columns=ignore), second.drop(columns=ignore))
    directory = tmp_path / "a" / first.episode_path.iloc[0]
    trajectory = pd.read_csv(directory / "executed_trajectory.csv")
    assert len(trajectory) == 5 and trajectory.terminal.sum() == 1
    assert set(
        [
            "time_index",
            "px",
            "py",
            "heading",
            "velocity",
            "acceleration",
            "steering_angle",
            "cumulative_evaluation_cost",
            "success",
            "failure",
            "collision",
        ]
    ) <= set(trajectory)
    assert pd.isna(trajectory.acceleration.iloc[-1])
    assert (
        trajectory.cumulative_evaluation_cost.iloc[-1] == first.cumulative_evaluation_cost.iloc[0]
    )
    diagnostics = pd.read_csv(directory / "mpc_diagnostics.csv")
    assert {
        "ess",
        "weight_entropy",
        "max_weight",
        "planning_latency_s",
        "minimum_sample_cost",
        "chosen_acceleration",
        "saturation_fraction",
    } <= set(diagnostics)
    index = pd.read_csv(directory / "snapshot_index.csv")
    assert list(index.step) == [0, 1, 2]
    assert (index.sample_count == 24).all()
    manifest = json.loads((tmp_path / "a/manifest.json").read_text())
    assert manifest["complete"] and manifest["snapshots"] == 24
    cases = json.loads((tmp_path / "a/cases.json").read_text())
    assert len({c["shared_setup_sha256"] for c in cases}) == 1
    for path in (tmp_path / "a/distributions").glob("*.npz"):
        with np.load(path) as data:
            assert data["occupancy_mass"].sum() == pytest.approx(1)
    monkeypatch.setattr(
        np.random, "default_rng", lambda *args: pytest.fail("Figure code resampled experiments")
    )
    generate(tmp_path / "a")
    for suffix in (
        "A_executed_bundle",
        "B_occupancy_density",
        "C_rollout_cloud",
        "D_progress_dispersion",
        "E_behavior_dashboard",
    ):
        for extension in (".png", ".svg", "_data.npz"):
            assert (tmp_path / "a/figures" / f"narrow_passage_{suffix}{extension}").exists()


def test_unregistered_future_variant_is_not_silently_relabelled(tmp_path):
    config = small_config()
    config["variants"] = ["tsallis"]
    with pytest.raises(ValueError, match="factory"):
        run_study(config, tmp_path / "invalid")
