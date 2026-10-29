"""Read-only replay of formulation benchmark evidence; never samples a new policy."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.artifacts import fingerprint
from analysis.dynamic_campaign import reached
from controllers.robust import augmented_costs
from controllers.tsallis import tsallis_weights
from dynamics.bicycle import wrap_angle
from dynamics.dynamic_vehicle import DynamicVehicle
from evaluation.runner import make_controller, write_json
from sampling.gaussian import importance_weights, log_base_over_proposal
from utils.config import read_yaml


def audit(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    config = read_yaml(root / "config.yaml")
    assert manifest["complete"] and fingerprint(config) == manifest["configuration_sha256"]
    frame = pd.read_csv(root / "per_seed.csv")
    snapshots = 0
    for (_, _), group in frame.groupby(["protocol", "scenario"]):
        assert group.shared_condition_sha256.nunique() == 1
        for _, method in group.groupby("algorithm"):
            assert sorted(method.seed) == sorted(config["seeds"])
    for row in frame.itertuples():
        episode = Path(row.raw_episode)
        c = read_yaml(episode / "config.yaml")
        controller = make_controller(c)
        execution = DynamicVehicle(**(c["vehicle"] | c["scenario"].get("execution_overrides", {})))
        trajectory = pd.read_csv(episode / "trajectory.csv")
        states = trajectory[["px", "py", "heading", "roll", "vx", "vy", "yaw_rate"]].to_numpy()
        controls = pd.read_csv(episode / "controls.csv").to_numpy()
        diag = pd.read_csv(episode / "diagnostics.csv")
        np.testing.assert_allclose(states[0], c["scenario"]["initial_state"])
        event = c["scenario"].get("disturbance")
        factor = 1 + 0.1 * np.random.default_rng(np.random.SeedSequence([4401, row.seed])).normal()
        gaps = []
        total = 0.0
        previous = np.zeros(2)
        if "execution_control_noise_covariance" in c:
            noise = np.random.default_rng(np.random.SeedSequence([4402, row.seed])).standard_normal(
                (c["execution"]["max_steps"], 2)
            )
            noise = noise @ np.linalg.cholesky(c["execution_control_noise_covariance"]).T
            commanded = diag[["commanded_steering", "commanded_throttle"]].to_numpy()
            np.testing.assert_allclose(
                controls,
                execution.clip_control(
                    commanded + noise[: len(controls)] * execution.control_scale
                ),
                atol=1e-12,
            )
        for t, control in enumerate(controls):
            nxt, path = execution.step(states[t], control, return_path=True)
            if event and t == event["step"]:
                nxt += np.array(event["delta"]) * factor
                nxt[2] = wrap_angle(nxt[2])
                nxt[4] = np.clip(nxt[4], 0, execution.velocity_max)
                path = np.concatenate([path, nxt[None]], axis=0)
            np.testing.assert_allclose(nxt, states[t + 1], atol=1e-10, rtol=1e-10)
            gap = float(controller.cost.path_clearance(path))
            gaps.append(gap)
            total += float(controller.cost.running(states[t], control, previous))
            total += execution.dt * controller.cost.weights["collision"] * (gap <= 0)
            previous = control
        total += float(controller.cost.terminal(states[-1]))
        np.testing.assert_allclose(total, row.cumulative_evaluation_cost, atol=1e-7)
        assert bool(row.collision) == bool(np.min(gaps) <= 0)
        assert bool(row.success) == (
            reached(states[-1], controller.cost.geometry, c) and not row.collision
        )
        t = c["controller"]["horizon"]
        k = c["controller"]["samples"]
        algorithm = row.algorithm
        counts = {
            "it_mppi": (k + 1) * t,
            "tsallis_vi": (k + 1) * t,
            "dial_mpc": c["controller"].get("iterations", 4) * (k + 1) * t,
            "dial_no_anneal": c["controller"].get("iterations", 4) * (k + 1) * t,
            "tube_mppi": (2 * (k + 1) + 18) * t,
            "robust_mppi": (2 * k + 5 * (c["controller"].get("candidate_samples", 8) + 1) + 22) * t,
            "cc_mppi": (k + 21) * t,
        }
        assert (diag.dynamics_evaluations == counts[algorithm]).all()
        assert (diag.rhs_evaluations == diag.dynamics_evaluations * 2 * execution.substeps).all()
        if row.protocol == "dynamics_budget":
            assert diag.dynamics_evaluations.max() <= config["transition_budget"]
        for file in sorted((episode / "snapshots").glob("*.npz")):
            with np.load(file) as z:
                costs = z["costs"]
                latent = z["latent_controls"]
                mean = z["proposal_mean"]
                if algorithm in {"it_mppi", "tube_mppi", "tsallis_vi"}:
                    ratio = log_base_over_proposal(
                        latent,
                        mean,
                        c["controller"]["base_covariance"],
                        c["controller"]["covariance"],
                    )
                    if algorithm == "tsallis_vi":
                        span = np.ptp(costs)
                        normalized = (costs - costs.min()) / span if span else np.zeros_like(costs)
                        gamma = max(
                            float(np.quantile(normalized, c["controller"]["elite_fraction"])),
                            np.finfo(float).eps,
                        )
                        r = c["controller"]["tsallis_r"]
                        weights = tsallis_weights(normalized, (r - 1) * gamma, r, ratio)
                    else:
                        weights = importance_weights(costs, c["controller"]["temperature"], ratio)
                elif algorithm in {"dial_mpc", "dial_no_anneal"}:
                    weights = importance_weights(costs, c["controller"]["temperature"])
                elif algorithm == "robust_mppi":
                    weights = importance_weights(
                        z["real_augmented_costs"], c["controller"]["temperature"]
                    )
                    nominal, actual = augmented_costs(
                        z["nominal_costs"],
                        costs,
                        mean,
                        latent - mean,
                        z["feedback"],
                        np.array(c["controller"]["covariance"]),
                        c["controller"]["temperature"],
                        c["controller"]["free_energy_threshold"],
                    )
                    np.testing.assert_allclose(actual, z["real_augmented_costs"])
                    np.testing.assert_allclose(nominal, z["nominal_augmented_costs"])
                elif algorithm == "cc_mppi":
                    augmented = z["augmented_costs"]
                    if "control_means" in z:
                        v = z["control_means"]
                        e = z["sampled_noise"]
                        r = np.diag(c["controller"]["cc_control_r"])
                        extra = np.einsum("kti,ij,ktj->k", v, r, e)
                        extra += 0.5 * np.einsum("kti,ij,ktj->k", v, r, v)
                        extra += (
                            0.5
                            * (1 - 1 / c["controller"]["noise_ratio"])
                            * np.einsum("kti,ij,ktj->k", e, r, e)
                        )
                        np.testing.assert_allclose(costs + extra, augmented)
                    weights = importance_weights(augmented, c["controller"]["temperature"])
                else:
                    raise AssertionError(algorithm)
                np.testing.assert_allclose(weights, z["weights"], atol=1e-10)
                update_weights = z["nominal_weights"] if algorithm == "robust_mppi" else weights
                np.testing.assert_allclose(
                    np.einsum("k,kti->ti", update_weights, latent), z["updated_mean"], atol=1e-10
                )
                snapshots += 1
    result = dict(
        passed=True,
        episodes=len(frame),
        snapshots=snapshots,
        checks=[
            "configuration hash",
            "paired seeds/conditions",
            "execution dynamics",
            "actuator noise",
            "swept collisions",
            "external costs/finish",
            "full model counts",
            "snapshot weights/update",
        ],
    )
    write_json(root / "audit.json", result)
    return result
