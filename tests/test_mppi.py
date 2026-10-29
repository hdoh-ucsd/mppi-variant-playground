import numpy as np
import pytest

from evaluation.runner import make_controller
from utils.config import load_experiment


@pytest.mark.parametrize("kind", ["standard", "isotropic", "diagonal", "full"])
def test_command_limits_diagnostics_and_work_count(kind):
    config = load_experiment(kind)
    config["controller"].update(samples=32, horizon=6)
    controller = make_controller(config)
    state = np.array(config["scenario"]["initial_state"])
    controller.reset(state, 42)
    command = controller.command(state)
    assert command.shape == (2,)
    assert np.all(abs(command) <= controller.model.control_scale)
    diagnostics = controller.get_diagnostics()
    assert 1 <= diagnostics["ess"] <= 32 + 1e-10
    assert diagnostics["dynamics_evaluations"] == 33 * 6
    assert diagnostics["weights"].sum() == pytest.approx(1)
    assert diagnostics["saved_rollouts"].shape == (32, 7, 4)
    diagnostics["weights"][:] = 0
    assert controller.get_diagnostics()["weights"].sum() == pytest.approx(1)


def test_saturation_preserves_latent_likelihood_and_update():
    config = load_experiment()
    config["controller"].update(samples=8, horizon=2)
    controller = make_controller(config)
    controller.mean[:] = [3, -3]
    controller.command(np.array(config["scenario"]["initial_state"]))
    assert controller.get_diagnostics()["saturation_fraction"] > 0.5
    # Updating clipped perturbations would always put the warm start inside the box.
    assert np.max(abs(controller.mean)) > 1


def test_unimplemented_variant_is_rejected():
    config = load_experiment()
    config["controller"]["kind"] = "robust"
    with pytest.raises(ValueError, match="not implemented"):
        make_controller(config)
