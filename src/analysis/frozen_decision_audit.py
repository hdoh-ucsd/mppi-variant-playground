"""Audit saved frozen-input evidence without resampling or reoptimizing decisions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.frozen_decision import (
    controller_at,
    inputs_signature,
    load_decision,
    normal_draws,
    plan_objective,
)
from evaluation.runner import write_json
from sampling.gaussian import importance_weights, log_base_over_proposal, weighted_update
from scenarios.environment import swept_clearance


def audit(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Incomplete study")
    frame = pd.read_csv(root / "per_seed.csv")
    proposals = json.loads((root / "proposals.json").read_text())
    with np.load(root / "updated_means.npz", allow_pickle=False) as data:
        means = data["means"]
    assert len(frame) == len(means) == manifest["decisions"]
    checked = 0
    for index, state in enumerate(frame.state.unique()):
        frozen = json.loads((root / "frozen_states" / f"{state}.json").read_text())
        assert inputs_signature(frozen) == frozen["inputs_sha256"]
        selected = frame[frame.state == state]
        assert selected.inputs_sha256.nunique() == 1
        assert selected.inputs_sha256.iloc[0] == frozen["inputs_sha256"]
        assert frozen["config"]["controller"]["base_covariance"] == [[0.28, 0], [0, 0.16]]
        with np.load(root / "frozen_states" / f"{state}_source_episode.npz") as episode:
            np.testing.assert_array_equal(episode["states"][-1], frozen["state"])
            assert len(episode["states"]) == frozen["step"] + 1
            assert len(episode["controls"]) == frozen["step"]
        if state == "pre_obstacle":
            assert set(frozen["route_witnesses"]) == {"upper", "lower"}
            scenario = frozen["config"]["scenario"]
            for witness in frozen["route_witnesses"].values():
                states = np.array(witness["states"])
                np.testing.assert_array_equal(states[0], frozen["state"])
                assert (
                    swept_clearance(
                        states[:-1], states[1:], scenario["obstacles"], scenario["vehicle_radius"]
                    ).min()
                    > 0
                )
                assert states[:, 0].max() >= scenario["obstacles"][0][0]
        reference = json.loads((root / "reference_estimates" / state / "status.json").read_text())
        assert all(sequence[0] == 202 for sequence in reference["seed_sequences"])
        for row_index, row in selected.iterrows():
            if pd.isna(row.get("snapshot_path")):
                continue
            snap = load_decision(root / row.snapshot_path)
            np.testing.assert_allclose(
                snap["sampled_states"][:, 0],
                np.broadcast_to(frozen["state"], (int(row.samples), 4)),
                atol=3e-7,
            )
            covariance = proposals[state]["proposals"][row.proposal]["covariance"]
            z = normal_draws(index, int(row.seed), int(row.samples), len(frozen["warm_start"]))
            epsilon = z @ np.linalg.cholesky(covariance).T
            np.testing.assert_array_equal(
                snap["latent_controls"], np.array(frozen["warm_start"])[None] + epsilon
            )
            ratio = log_base_over_proposal(
                snap["latent_controls"],
                np.array(frozen["warm_start"]),
                frozen["config"]["controller"]["base_covariance"],
                covariance,
            )
            np.testing.assert_allclose(ratio, snap["log_base_over_proposal"], atol=1e-12)
            weights = importance_weights(
                snap["costs"], frozen["config"]["controller"]["temperature"], ratio
            )
            np.testing.assert_allclose(weights, snap["weights"], atol=1e-13)
            updated = weighted_update(np.array(frozen["warm_start"]), epsilon, weights)
            np.testing.assert_allclose(updated, snap["updated_mean"], atol=1e-13)
            np.testing.assert_array_equal(snap["updated_mean"], means[row_index])
            controller = controller_at(frozen, covariance, int(row.samples))
            value = plan_objective(
                controller, snap["nominal_states"], updated, frozen["previous_control"]
            )
            np.testing.assert_allclose(value, row.nominal_objective, atol=1e-10)
            assert row.dynamics_evaluations == (row.samples + 1) * len(updated)
            checked += 1
    for path in (root / "figures").glob("*_C_weighted_endpoint_density_data.npz"):
        with np.load(path) as data:
            areas = np.diff(data["x_edges"])[:, None] * np.diff(data["y_edges"])[None, :]
            np.testing.assert_allclose(
                np.sum(data["densities"] * areas, axis=(1, 2)), 1, atol=1e-12
            )
    result = dict(
        decisions=len(frame),
        snapshots_checked=checked,
        frozen_inputs_identical=True,
        common_target_verified=True,
        common_random_numbers_verified=True,
        exact_density_ratios_verified=True,
        normalized_weights_verified=True,
        weighted_updates_verified=True,
        nominal_objectives_verified=True,
        reference_streams_independent=True,
        weighted_endpoint_mass_normalized=True,
        note="Reconstructs paired normals only for audit; figure regeneration draws no randomness.",
    )
    assert checked == manifest["snapshots"]
    write_json(root / "audit.json", result)
    return result
