import json

import numpy as np
import pandas as pd

from analysis.closed_loop_study import run_closed_loop
from analysis.plots import generate
from utils.config import read_yaml


def test_sweep_reuses_episode_api_and_preserves_base_target(tmp_path):
    config = read_yaml("configs/investigations/closed_loop_sweep.yaml")
    config.update(
        evaluation_seeds=[2000, 2001],
        sample_counts=[32],
        max_steps=4,
        anisotropy_ratios=[1, 4],
        angles_deg=[0, 45],
        scales=[1.0],
    )
    frame = run_closed_loop(config, tmp_path / "run")
    assert len(frame) == 18
    assert (frame.constraint_violations == 0).all()
    assert (frame.cost_evaluations == 4 * 32).all()
    targets = []
    for path in frame.episode_path:
        episode = read_yaml(tmp_path / "run" / path / "config.yaml")
        targets.append(episode["controller"]["base_covariance"])
        assert (tmp_path / "run" / path / "trajectory.csv").exists()
    assert all(p == targets[0] for p in targets)
    pairs = pd.read_csv(tmp_path / "run/paired_differences.csv")
    assert {"ess", "success", "cumulative_evaluation_cost", "saturation_fraction"} <= set(
        pairs.metric
    )
    cases = json.loads((tmp_path / "run/cases.json").read_text())
    assert {c["family"] for c in cases} == {"isotropic", "diagonal", "full"}
    generate(tmp_path / "run")
    assert (tmp_path / "run/figures/08_closed_loop_trajectories.svg").exists()
    figure_data = tmp_path / "run/figures/08_closed_loop_trajectories_data.npz"
    with np.load(figure_data) as data:
        retained = data["open_space_unweighted_rollouts"].copy()
    for archive in (tmp_path / "run/episodes").rglob("rollouts.npz"):
        archive.unlink()
    generate(tmp_path / "run")
    with np.load(figure_data) as data:
        np.testing.assert_array_equal(data["open_space_unweighted_rollouts"], retained)
