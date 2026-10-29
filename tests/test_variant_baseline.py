"""Pre-refactor oracle captured from c7bbf86, including full latent updates."""

import json
from pathlib import Path

import numpy as np

from analysis.dynamic_campaign import configuration, resolved_case
from evaluation.runner import make_controller


def test_it_mppi_retains_pre_variant_decisions():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/it_mppi_before_variants.json").read_text()
    )
    for case in fixture["records"]:
        config, *_ = resolved_case(configuration(), case["scenario"], fixture["samples"], 0.35)
        config["controller"].update(
            kind="standard", covariance=config["proposals"]["standard"], saved_rollouts=0
        )
        controller = make_controller(config)
        state = config["scenario"]["initial_state"]
        controller.reset(state, fixture["seed"])
        for expected in case["decisions"]:
            control = controller.command(state)
            diag = controller.get_diagnostics()
            for actual, key in [
                (control, "control"),
                (controller.mean, "mean"),
                (diag["weights"], "weights"),
                (diag["planned_trajectory"], "planned"),
            ]:
                np.testing.assert_allclose(actual, expected[key], rtol=1e-13, atol=1e-13)
            assert diag["dynamics_evaluations"] == expected["dynamics_evaluations"]
            state = controller.model.step(state, control)
