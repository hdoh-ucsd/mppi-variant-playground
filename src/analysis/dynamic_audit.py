"""Audit seven-state numerical evidence without performing controller optimization."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.artifacts import fingerprint
from analysis.dynamic_campaign import reached
from analysis.frozen_decision import inputs_signature, load_decision, normal_draws
from dynamics.bicycle import wrap_angle
from dynamics.dynamic_vehicle import DynamicVehicle
from evaluation.runner import make_controller, write_json
from sampling.gaussian import importance_weights, log_base_over_proposal


def frozen_audit(root):
    root = Path(root)
    frame = pd.read_csv(root / "per_seed.csv")
    info = json.loads((root / "proposal_and_reference.json").read_text())
    count = 0
    for state_index, name in enumerate(frame.state.unique()):
        f = json.loads((root / "frozen_states" / f"{name}.json").read_text())
        assert inputs_signature(f) == f["inputs_sha256"]
        assert len(f["state"]) == 7 and np.shape(f["warm_start"])[1] == 2
        assert frame[frame.state == name].inputs_sha256.nunique() == 1
        for path in (root / "snapshots" / name).glob("K*/*/seed_*.npz"):
            snap = load_decision(path)
            k = len(snap["weights"])
            seed = int(path.stem.split("_")[-1])
            covariance = info[name]["proposals"][path.parent.name]["covariance"]
            z = normal_draws(state_index, seed, k, len(f["warm_start"]))
            latent = np.array(f["warm_start"])[None] + z @ np.linalg.cholesky(covariance).T
            np.testing.assert_array_equal(latent, snap["latent_controls"])
            ratio = log_base_over_proposal(
                latent,
                np.array(f["warm_start"]),
                f["config"]["controller"]["base_covariance"],
                covariance,
            )
            np.testing.assert_allclose(ratio, snap["log_base_over_proposal"], atol=1e-11)
            w = importance_weights(snap["costs"], f["config"]["controller"]["temperature"], ratio)
            np.testing.assert_allclose(w, snap["weights"], atol=1e-12)
            np.testing.assert_allclose(
                np.einsum("k,kti->ti", w, latent), snap["updated_mean"], atol=1e-12
            )
            count += 1
        assert np.isfinite(np.array(info[name]["reference"]["mean"])).all()
    result = dict(
        decisions=len(frame),
        snapshots_verified=count,
        state_dimension=7,
        sampling_dimension=2 * len(f["warm_start"]),
        frozen_inputs_verified=True,
        common_normals_verified=True,
        common_target_and_ratios_verified=True,
        no_closed_loop_evaluation=(
            "Only source prefixes execute; evaluation decisions do not advance an environment"
        ),
    )
    write_json(root / "audit.json", result)
    return result


def driving_audit(root, raw=True):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Incomplete study")
    frame = pd.read_csv(root / "per_seed.csv")
    cases = json.loads((root / "conditions.json").read_text())
    episodes = snapshots = 0
    for cell in cases:
        config = cell["config"]
        rows = frame[frame.case == cell["case"]]
        for cohort, group in rows.groupby("cohort"):
            for _, part in group.groupby("proposal"):
                assert sorted(part.seed) == manifest["cohorts"][cohort]
            assert group.shared_condition_sha256.nunique() == 1
        assert rows.shared_condition_sha256.iloc[0] == fingerprint(config)
        if cell["feasibility"] is not None:
            assert cell["feasibility"]["feasible"]
        if not raw:
            continue
        controller_config = json.loads(json.dumps(config))
        controller_config["controller"].update(
            kind="standard", covariance=config["proposals"]["standard"], samples=2
        )
        controller = make_controller(controller_config)
        geometry = controller.cost.geometry
        model = controller.model
        execution = DynamicVehicle(
            **(config["vehicle"] | config["scenario"].get("execution_overrides", {}))
        )
        for row in rows.itertuples():
            folder = Path(row.episode_path)
            meta = json.loads((folder / "result.json").read_text())
            states = pd.read_csv(folder / "trajectory.csv")[
                ["px", "py", "heading", "roll", "vx", "vy", "yaw_rate"]
            ].to_numpy()
            controls = pd.read_csv(folder / "controls.csv").to_numpy()
            diag = pd.read_csv(folder / "diagnostics.csv")
            expected, paths = execution.step(states[:-1], controls, return_path=True)
            event = config["scenario"].get("disturbance")
            if event and len(controls) > event["step"]:
                expected[event["step"]] += np.array(meta["disturbance_delta"])
                expected[event["step"], 2] = wrap_angle(expected[event["step"], 2])
                expected[event["step"], 4] = np.clip(
                    expected[event["step"], 4], 0, execution.velocity_max
                )
            np.testing.assert_allclose(expected, states[1:], atol=3e-12)
            gaps = controller.cost.path_clearance(paths)
            if event and len(controls) > event["step"]:
                i = event["step"]
                gaps[i] = min(
                    gaps[i], float(geometry.transition_clearance(paths[i, -1], states[i + 1]))
                )
            np.testing.assert_allclose(gaps, diag.clearance, atol=1e-11)
            collision = bool((gaps <= 0).any())
            assert collision == bool(row.collision)
            assert (reached(states[-1], geometry, config) and not collision) == bool(row.success)
            assert np.all(abs(controls) <= model.control_scale + 1e-12)
            rate = config["controller"].get("steering_rate_limit")
            if rate is not None:
                assert np.max(abs(np.diff(np.r_[0.0, controls[:, 0]]))) <= rate * model.dt + 1e-12
            assert row.model_transitions == len(controls) * (row.samples + 1) * row.horizon
            assert row.rhs_evaluations == row.model_transitions * model.substeps * 2
            if config["scenario"]["name"] == "model_mismatch":
                assert meta["planning_parameters"] != meta["execution_parameters"]
            for path in (folder / "snapshots").glob("*.npz"):
                with np.load(path, allow_pickle=False) as data:
                    ratio = log_base_over_proposal(
                        data["latent_controls"],
                        data["proposal_mean"],
                        config["controller"]["base_covariance"],
                        config["proposals"][row.proposal],
                    )
                    np.testing.assert_allclose(ratio, data["log_base_over_proposal"], atol=1e-11)
                    weights = importance_weights(
                        data["costs"], config["controller"]["temperature"], ratio
                    )
                    np.testing.assert_allclose(weights, data["weights"], atol=1e-12)
                    np.testing.assert_allclose(
                        np.einsum("k,kti->ti", weights, data["latent_controls"]),
                        data["updated_mean"],
                        atol=1e-12,
                    )
                    if rate is not None:
                        step = int(path.stem.split("_")[-1])
                        prior = controls[step - 1] if step else np.zeros(2)
                        applied = model.clip_control(data["latent_controls"] * model.control_scale)
                        prev = prior[0]
                        for t in range(row.horizon):
                            applied[:, t, 0] = np.clip(
                                applied[:, t, 0], prev - rate * model.dt, prev + rate * model.dt
                            )
                            prev = applied[:, t, 0]
                        np.testing.assert_allclose(applied, data["sampled_controls"], atol=1e-14)
                        x = np.broadcast_to(data["current_state"], (row.samples, 7)).copy()
                        prev = np.broadcast_to(prior, (row.samples, 2))
                        costs = np.zeros(row.samples)
                        for t in range(row.horizon):
                            costs += controller.cost.running(x, applied[:, t], prev)
                            x, substeps = model.step(x, applied[:, t], return_path=True)
                            costs += (
                                model.dt
                                * config["cost"]["collision"]
                                * (controller.cost.path_clearance(substeps) <= 0)
                            )
                            np.testing.assert_allclose(
                                x, data["sampled_states"][:, t + 1], atol=1e-12
                            )
                            prev = applied[:, t]
                        costs += controller.cost.terminal(x)
                        np.testing.assert_allclose(costs, data["costs"], atol=1e-8)
                    assert data["sampled_states"].shape[-1] == 7
                snapshots += 1
            episodes += 1
    result = dict(
        episodes=len(frame),
        raw_episodes_verified=episodes,
        snapshots_verified=snapshots,
        common_conditions_verified=True,
        paired_cohorts_verified=True,
        model_and_substep_collisions_verified=raw,
        model_mismatch_parameters_verified=raw,
        discrete_command_slew_verified=raw
        and any(
            cell["config"]["controller"].get("steering_rate_limit") is not None for cell in cases
        ),
        state_dimension=7,
        control_fields=["steering", "throttle"],
    )
    write_json(root / ("audit.json" if raw else "audit_summaries.json"), result)
    return result
