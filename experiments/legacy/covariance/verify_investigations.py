"""Audit saved schemas and replay selected analytic, adaptive and control records."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.analytic_gaussian import quadratic_cost
from analysis.importance_sampling import estimate
from analysis.nonconvex_study import run_nonconvex
from evaluation.runner import run_episode
from utils.config import read_yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", default="runs/legacy/investigations")
args = parser.parse_args()
root = Path(args.input)
checked = 0
for study in ("gaussian", "temperature"):
    folder = root / f"{study}_v1"
    cases = json.loads((folder / "cases.json").read_text())
    trials = pd.read_csv(folder / "trials.csv")
    ids = sorted(set([0, len(cases) // 4, len(cases) // 2, 3 * len(cases) // 4, len(cases) - 1]))
    for case_id in ids:
        case = cases[case_id]
        mean = np.array(case["proposal_mean"])
        cov = np.array(case["proposal_covariance"])
        for seed in (1000, 1063):
            z = (
                mean
                + np.random.default_rng(seed).standard_normal((case["samples"], len(mean)))
                @ np.linalg.cholesky(cov).T
            )
            cost = quadratic_cost(z, case["hessian"], case["cost_center"])
            mu, weights, diag = estimate(
                z, cost, mean, case["base_covariance"], cov, case["temperature"]
            )
            original = trials[(trials.case_id == case_id) & (trials.seed == seed)].iloc[0]
            np.testing.assert_allclose(
                mu, [original[f"estimate_{i}"] for i in range(len(mean))], rtol=0, atol=1e-14
            )
            np.testing.assert_allclose(diag["ess"], original.ess, rtol=0, atol=1e-12)
            assert abs(weights.sum() - 1) < 1e-14
            checked += 1
print(f"Reproduced {checked} saved Gaussian/temperature/dimension estimates and ESS values.")
with tempfile.TemporaryDirectory(prefix="mppi-replay-") as scratch:
    scratch = Path(scratch)
    config = read_yaml(root / "nonconvex_v1/config.yaml")
    config.update(
        evaluation_seeds=[1000, 1001], objectives=["curved_valley"], initial_means=[[-1.2, -0.8]]
    )
    replay = run_nonconvex(config, scratch / "nonconvex")
    original = pd.read_csv(root / "nonconvex_v1/trials.csv")
    original = original[(original.case_id < 5) & original.seed.isin([1000, 1001])]
    for metric in ("estimate_0", "estimate_1", "best_cost", "error"):
        np.testing.assert_allclose(replay[metric], original[metric], rtol=0, atol=1e-14)
    print("Reproduced ten adaptive nonconvex optimization trials.")
    cases = json.loads((root / "closed_loop_v1/cases.json").read_text())
    original_rows = pd.read_csv(root / "closed_loop_v1/trials.csv")
    for scenario in ("open_space", "narrow_passage", "disturbance"):
        case = next(
            c
            for c in cases
            if c["scenario"] == scenario
            and c["samples"] == 384
            and c["scale"] == 1
            and c["proposal_name"] == "isotropic"
        )
        row = original_rows[
            (original_rows.case_id == case["case_id"]) & (original_rows.seed == 2000)
        ].iloc[0]
        directory = root / "closed_loop_v1" / row.episode_path
        config = read_yaml(directory / "config.yaml")
        destination = scratch / scenario
        record = run_episode(config, 2000, destination, mode="covariance_mechanism")
        for filename in ("trajectory.csv", "controls.csv"):
            pd.testing.assert_frame_equal(
                pd.read_csv(directory / filename),
                pd.read_csv(destination / filename),
                check_exact=True,
            )
        original_record = json.loads((directory / "result.json").read_text())
        for metric in ("success", "goal_position_error", "cumulative_evaluation_cost", "mean_ess"):
            assert record["metrics"][metric] == original_record["metrics"][metric], (
                scenario,
                metric,
            )
    print("Replayed three bicycle trajectories and commands exactly, with matching metrics.")
for folder in root.glob("*_v1"):
    manifest = json.loads((folder / "manifest.json").read_text())
    trials = pd.read_csv(folder / "trials.csv")
    assert manifest["complete"] and manifest["trials"] == len(trials)
    assert not trials.duplicated(["case_id", "seed"]).any()
    assert not manifest["git"]["dirty"]
print("All five studies passed schema, seed and source-provenance checks.")
