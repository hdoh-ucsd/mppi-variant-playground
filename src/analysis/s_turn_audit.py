"""Verify completed S-turn trials against local raw records and shared planner identities."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis.s_turn_study import condition, reached
from dynamics.steering_rate import make_model
from evaluation.runner import write_json
from sampling.gaussian import importance_weights, log_base_over_proposal


def audit(root, raw=True):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Incomplete study")
    frame = pd.read_csv(root / "per_seed.csv")
    conditions = json.loads((root / "conditions.json").read_text())
    checks = json.loads((root / "feasibility.json").read_text())
    assert len(frame) == manifest["episodes"] and all(v["feasible"] for v in checks)
    expected = manifest["paired_seeds"]
    for _, g in frame.groupby("cell"):
        assert g.shared_condition_sha256.nunique() == 1
        for _, h in g.groupby("proposal"):
            assert sorted(h.seed) == expected
    config = yaml.safe_load((root / "config.yaml").read_text())
    episodes = snapshots = 0
    if raw:
        for cell in conditions:
            resolved = cell["config"]
            clearance = resolved["corridor"]["clearance"]
            _, corridor, *_ = condition(
                resolved,
                clearance,
                resolved["controller"]["samples"],
                resolved["vehicle"]["steering_rate_limit"],
            )
            model = make_model(resolved["vehicle"])
            rows = frame[frame.cell == cell["cell"]]
            for row in rows.itertuples():
                folder = root / row.episode_path
                trajectory = pd.read_csv(folder / "trajectory.csv")
                states = trajectory[
                    ["px", "py", "heading", "velocity", "steering_angle"]
                ].to_numpy()
                controls = pd.read_csv(folder / "controls.csv").to_numpy()
                diag = pd.read_csv(folder / "diagnostics.csv")
                np.testing.assert_allclose(
                    model.step(states[:-1], controls), states[1:], atol=2e-12
                )
                np.testing.assert_allclose(states[0], resolved["initial_state"], atol=1e-14)
                assert np.all(abs(controls) <= model.control_scale + 1e-12)
                assert np.max(abs(states[:, 4])) <= model.bicycle.steering_limit + 1e-12
                gaps = corridor.transition_clearance(states[:-1], states[1:])
                assert bool((gaps <= 0).any()) == bool(row.collision)
                assert (reached(states[-1], corridor, resolved) and not row.collision) == bool(
                    row.success
                )
                np.testing.assert_allclose(row.minimum_clearance_m, gaps.min(), atol=1e-12)
                assert (
                    row.model_transitions
                    == row.steps * (row.samples + 1) * resolved["controller"]["horizon"]
                )
                assert np.all((diag.ess >= 1 - 1e-10) & (diag.ess <= row.samples + 1e-8))
                assert np.isfinite(diag.weight_entropy).all()
                for path in (folder / "snapshots").glob("*.npz"):
                    with np.load(path, allow_pickle=False) as data:
                        np.testing.assert_allclose(
                            data["weights"],
                            importance_weights(
                                data["costs"],
                                resolved["controller"]["temperature"],
                                data["log_base_over_proposal"],
                            ),
                            atol=1e-13,
                        )
                        if "proposal_mean" in data:
                            ratio = log_base_over_proposal(
                                data["latent_controls"],
                                data["proposal_mean"],
                                resolved["controller"]["base_covariance"],
                                resolved["proposals"][row.proposal],
                            )
                            np.testing.assert_allclose(
                                ratio, data["log_base_over_proposal"], atol=1e-11
                            )
                        updated = np.einsum("k,kti->ti", data["weights"], data["latent_controls"])
                        np.testing.assert_allclose(updated, data["updated_mean"], atol=1e-12)
                    snapshots += 1
                episodes += 1
    result = dict(
        episodes_in_summary=len(frame),
        raw_episodes_verified=episodes,
        snapshots_verified=snapshots,
        paired_seeds=len(expected),
        feasible_conditions=len(checks),
        input_conditions_common=True,
        original_base_covariance=config["controller"]["base_covariance"],
        raw_checked=raw,
    )
    write_json(root / ("audit.json" if raw else "audit_summaries.json"), result)
    return result
