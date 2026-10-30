"""Trace semantics and passive observation across both modes and all six controllers."""

import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.decision_trace import STAGES, STATE_UNITS, TraceStore, fingerprint, problem_config
from analysis.dynamic_campaign import run_episode
from analysis.trace_adapters import ALGORITHMS, TraceRecorder, build_trace, capture_inputs
from analysis.trace_comparison import frozen_config, read
from evaluation.runner import make_controller
from sampling.gaussian import log_base_over_proposal, log_sequence_density
from utils.config import read_yaml


def small_config(algorithm):
    f = read("results/variants/mechanisms/V01/frozen_inputs.json")["two_route_obstacle"]
    c = frozen_config(f, read_yaml("configs/variants/benchmark.yaml"), algorithm, 8)
    c["controller"].update(horizon=4, snapshot_steps=[0], candidate_samples=3)
    return c


def frozen(algorithm="it_mppi", capture=True):
    c = small_config(algorithm)
    c["controller"]["snapshot_steps"] = [0] if capture else []
    ctrl = make_controller(c)
    state = np.array(c["scenario"]["initial_state"])
    ctrl.reset(state, 51)
    ctrl.mean[:] = [0.1, 0.35]
    if algorithm in ("tube_mppi", "robust_mppi"):
        state = state.copy()
        state[1] += 0.7
        ctrl.config["reset_threshold"] = 0
        ctrl.config["free_energy_threshold"] = 1e-6
        c["controller"].update(reset_threshold=0, free_energy_threshold=1e-6)
    before = capture_inputs(ctrl, state, np.zeros(2))
    command, update = ctrl.plan(state)
    trace = build_trace(
        c,
        before,
        ctrl.get_rollout_snapshot(),
        ctrl.get_diagnostics(),
        command,
        experiment_id="test",
        seed=51,
        planning_step=0,
        mode="FROZEN_DECISION",
        source_commit="test",
        cost=ctrl.cost,
    )
    return c, ctrl, trace, command, update


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_frozen_passive_commands_rng_work_and_roundtrip(algorithm, tmp_path):
    c, observed, t, command, update = frozen(algorithm)
    _, plain, _, expected, expected_update = frozen(algorithm, False)
    np.testing.assert_array_equal(command, expected)
    np.testing.assert_array_equal(update, expected_update)
    np.testing.assert_array_equal(observed.mean, plain.mean)
    assert observed.rng.bit_generator.state == plain.rng.bit_generator.state
    assert observed.model.transitions == plain.model.transitions
    assert observed.diagnostics["dynamics_evaluations"] == plain.diagnostics["dynamics_evaluations"]
    assert t.metadata["state_units"] == STATE_UNITS
    assert list(t.stages) == list(STAGES)
    assert t.stages["feedback_execute"].status == "not_applicable"
    for u in t.stages["weight_update"].data["updates"]:
        w = t.get(u["weights"])
        scores = t.get(u["log_scores"])
        assert np.isclose(w.sum(), 1, atol=1e-12)
        p = np.exp(scores - scores.max())
        p /= p.sum()
        np.testing.assert_allclose(p, w, atol=1e-12)
        comp = u["cost_components"]
        np.testing.assert_allclose(
            sum(t.get(v) for v in comp.values()), t.get(u["raw_costs"]), atol=1e-9
        )
    store = TraceStore(tmp_path)
    store.save(t, c)
    restored = store.load(t.decision_id)
    assert restored.document() == t.document()
    for key, value in t.arrays.items():
        np.testing.assert_array_equal(value, restored.arrays[key])
    with pytest.raises(FileExistsError):
        store.save(t, c)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_closed_loop_observer_does_not_change_trajectory_or_stream(algorithm, tmp_path):
    c = small_config(algorithm)
    c["execution"]["max_steps"] = 4
    c["scenario"]["disturbance"] = {"step": 1, "delta": [0, 0.4, 0.1, 0, 0, 0.1, 0]}
    c["controller"].update(reset_threshold=0, free_energy_threshold=1e-6)
    controllers = []

    def factory(config):
        ctrl = make_controller(config)
        controllers.append(ctrl)
        return ctrl

    recorder = TraceRecorder(TraceStore(tmp_path / "traces"), "test-closed", "test", [0, 2])
    run_episode(c, algorithm, 51, tmp_path / "plain", controller_factory=factory)
    run_episode(
        c, algorithm, 51, tmp_path / "observed", controller_factory=factory, trace_recorder=recorder
    )
    for name in ["trajectory.csv", "controls.csv"]:
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "plain" / name), pd.read_csv(tmp_path / "observed" / name)
        )
    assert controllers[0].rng.bit_generator.state == controllers[1].rng.bit_generator.state
    assert controllers[0].model.transitions == controllers[1].model.transitions
    assert len(recorder.traces) == 4
    assert len({t.decision_id for t in recorder.traces}) == 4
    for step, t in enumerate(recorder.traces):
        assert t.metadata["planning_step"] == step
        assert t.metadata["mode"] == "CLOSED_LOOP"
        if step not in [0, 2]:
            assert t.metadata["record_completeness"] == "partial"
        assert t.stages["feedback_execute"].data["decision_id"] == t.decision_id
        assert t.stages["sample_refine"].status == ("available" if step in [0, 2] else "missing")
    event = recorder.traces[1].stages["feedback_execute"].data
    assert event["disturbance_applied"]
    assert np.linalg.norm(recorder.traces[1].get(event["state_impulse"])) > 0
    if algorithm in ["tube_mppi", "robust_mppi"]:
        t = recorder.traces[2]
        f = t.stages["feedback_execute"].data
        np.testing.assert_allclose(
            t.get(f["applied_correction"]),
            t.get(f["command_candidate"]) - t.get(f["planner_control"]),
        )
        assert np.linalg.norm(t.get(f["applied_correction"])) > 0


def test_ids_config_hash_dimensions_units_and_failure_classification(tmp_path):
    c, _, t, _, _ = frozen()
    _, _, again, _, _ = frozen()
    assert t.decision_id == again.decision_id
    altered = deepcopy(c)
    altered["controller"]["snapshot_steps"] = [1, 2, 3]
    assert fingerprint(problem_config(c)) == fingerprint(problem_config(altered))
    for key, value in [
        ("schema_version", "2.0"),
        ("state_dimension", 6),
        ("control_dimension", 3),
        ("state_units", ["m"] * 7),
        ("decision_id", "different"),
    ]:
        invalid = deepcopy(t)
        invalid.metadata[key] = value
        with pytest.raises(ValueError):
            invalid.validate()
    invalid = deepcopy(t)
    invalid.stages["weight_update"].status = "unknown"
    with pytest.raises(ValueError):
        invalid.validate()
    recorder = TraceRecorder(TraceStore(tmp_path), "failure", "test", [0])
    recorder.failed(
        c, 51, 0, {"observed_state": np.zeros(7)}, RuntimeError("covariance solver infeasible")
    )
    failed = recorder.traces[0]
    assert failed.stages["distributional_objective"].status == "numerical_failure"
    assert failed.stages["sample_refine"].status == "missing"
    assert failed.metadata["failure_type"] == "RuntimeError"


def test_density_identity_and_invalid_weight_rejected():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(5, 4, 2))
    mean = rng.normal(size=(4, 2))
    base = np.diag([0.2, 0.4])
    proposal = np.array([[0.3, 0.1], [0.1, 0.5]])
    np.testing.assert_allclose(
        log_sequence_density(z, np.zeros_like(mean), base)
        - log_sequence_density(z, mean, proposal),
        log_base_over_proposal(z, mean, base, proposal),
        atol=1e-12,
    )
    _, _, t, _, _ = frozen()
    t.stages["weight_update"].data["updates"][0]["weights"] = t.put(np.ones(8))
    with pytest.raises(ValueError, match="normalized"):
        t.validate()


def test_dial_all_passes_and_tube_both_populations():
    _, ctrl, t, _, _ = frozen("dial_mpc")
    pops = t.stages["sample_refine"].data["populations"]
    assert [p["iteration"] for p in pops] == list(range(4))
    assert sum(p["model_transitions"] for p in pops) == ctrl.diagnostics["dynamics_evaluations"]
    assert not np.array_equal(
        t.get(pops[0]["proposal_covariance"]), t.get(pops[-1]["proposal_covariance"])
    )
    invalid = deepcopy(t)
    invalid.stages["sample_refine"].data["populations"].reverse()
    with pytest.raises(ValueError, match="ordering"):
        invalid.validate()
    _, ctrl, t, _, _ = frozen("tube_mppi")
    pops = t.stages["sample_refine"].data["populations"]
    assert [p["role"] for p in pops] == ["nominal", "actual"]
    assert not np.array_equal(t.get(pops[0]["states"]), t.get(pops[1]["states"]))
    assert ctrl.diagnostics["selected_actual"] is False
    assert np.linalg.norm(t.get(t.stages["feedback_execute"].data["applied_correction"])) > 0
    assert not any("tube_actual_tube_" in k for k in ctrl.get_rollout_snapshot())


def test_robust_separation_and_cc_auxiliary_psd():
    _, _, t, _, _ = frozen("robust_mppi")
    updates = t.stages["weight_update"].data["updates"]
    assert [u["role"] for u in updates] == ["nominal", "actual"]
    assert not np.array_equal(t.get(updates[0]["raw_costs"]), t.get(updates[1]["raw_costs"]))
    assert not t.get(t.stages["sample_refine"].data["robust"]["candidate_feasible"]).any()
    _, _, t, _, _ = frozen("cc_mppi")
    cc = t.stages["sample_refine"].data["cc"]
    aux = t.get(cc["auxiliary_states"])
    assert aux.shape == (8, 5, 7)
    assert np.any(aux[:, 1:])
    np.testing.assert_array_equal(aux[:, 0], 0)
    cov = t.get(cc["predicted_covariances"])
    assert np.linalg.eigvalsh(cov).min() >= -1e-9
    assert isinstance(t.stages["distributional_objective"].data["cc"]["solver"], str)
    invalid = deepcopy(t)
    bad = cov.copy()
    bad[-1, 0, 0] = -1
    invalid.stages["sample_refine"].data["cc"]["predicted_covariances"] = invalid.put(bad)
    with pytest.raises(ValueError, match="PSD"):
        invalid.validate()


def test_external_arrays_are_referenced_and_hash_checked(tmp_path):
    c, _, t, _, _ = frozen()
    source = tmp_path / "original.npz"
    np.savez_compressed(source, **t.arrays)
    store = TraceStore(tmp_path / "store")
    folder = store.save(t, c, reuse_sources=[source])
    with np.load(folder / "arrays.npz") as z:
        assert z.files == ["schema_version"]
    assert store.load(t.decision_id).document() == t.document()
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="External"):
        store.load(t.decision_id)


@pytest.mark.parametrize(
    "name",
    [
        "analytic_gaussian",
        "covariance_alignment",
        "temperature_sweep",
        "actuator_saturation",
        "covariance_closed_loop",
        "nonconvex_optimization",
        "trajectory_density_study",
        "mechanism_summary",
    ],
)
def test_relocated_legacy_cli_still_runs(name):
    import sys

    subprocess.run(
        [sys.executable, f"experiments/legacy/{name}.py", "--help"], check=True, capture_output=True
    )


def test_plots_only_reconstructs_without_simulation_or_rng(tmp_path, monkeypatch):
    from analysis.trace_plots import plot_all
    from controllers.mppi import MPPI
    from dynamics.dynamic_vehicle import DynamicVehicle

    def forbidden(*args, **kwargs):
        raise AssertionError("plots-only attempted simulation/random draws")

    monkeypatch.setattr(MPPI, "plan", forbidden)
    monkeypatch.setattr(DynamicVehicle, "step", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    records = plot_all("results/variants/traces", tmp_path)
    assert len(records) == 6
    provenance = json.loads(Path("results/variants/traces/figure_provenance.json").read_text())
    for name, record in records.items():
        assert record["sha256"] == provenance["figures"][name]["sha256"]
        assert record["decision_ids"]


def test_readme_images_and_historical_records_preserved():
    readme = Path("README.md").read_text()
    for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme):
        if not target.startswith("http"):
            assert Path(target).is_file(), target
    for i in range(1, 6):
        assert Path(f"results/variants/mechanisms/V0{i}/manifest.json").is_file()


def test_frozen_numerical_failure_is_preserved(tmp_path, monkeypatch):
    from analysis.trace_comparison import trace_frozen
    from controllers.mppi import MPPI

    def fail(*args, **kwargs):
        raise FloatingPointError("invalid population")

    monkeypatch.setattr(MPPI, "plan", fail)
    store = TraceStore(tmp_path)
    c = small_config("it_mppi")
    inputs = dict(
        observed_state=np.array(c["scenario"]["initial_state"]),
        warm_start=np.zeros((4, 2)),
        previous_control=np.zeros(2),
    )
    with pytest.raises(FloatingPointError, match="invalid population"):
        trace_frozen(c, inputs, 51, store, "failure", source_commit="test")
    folder = next(tmp_path.glob("decision-*"))
    trace = store.load(folder.name)
    assert trace.metadata["mode"] == "FROZEN_DECISION"
    assert trace.stages["distributional_objective"].status == "numerical_failure"
    assert trace.stages["feedback_execute"].status == "not_applicable"


def test_array_units_checked_independently_of_metadata():
    _, _, t, _, _ = frozen()
    t.stages["dynamics_task"].data["observed_state"]["units"] = ["unknown"] * 7
    with pytest.raises(ValueError, match="units"):
        t.validate()


def test_curated_same_sample_comparison_and_complete_work():
    records = read("results/variants/traces/plot_data.json")["traces"]
    frozen_records = {t["algorithm"]: t for t in records if t["experiment_id"].endswith("-frozen")}
    first = frozen_records["it_mppi"]
    for t in frozen_records.values():
        np.testing.assert_array_equal(t["observed"], first["observed"])
    ts = frozen_records["tsallis_vi"]
    np.testing.assert_array_equal(first["updates"][0]["raw_costs"], ts["updates"][0]["raw_costs"])
    np.testing.assert_array_equal(
        first["populations"][0]["first_latent"], ts["populations"][0]["first_latent"]
    )
    assert frozen_records["dial_mpc"]["model_transitions"] == sum(
        p["model_transitions"] for p in frozen_records["dial_mpc"]["populations"]
    )
    for t in records:
        if t["experiment_id"].startswith("retro-"):
            assert any(t["stages_missing"].values())
