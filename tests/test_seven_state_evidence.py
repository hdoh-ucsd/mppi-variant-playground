"""Ensure figures use recorded populations and scientific provenance is explicit."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from analysis import dynamic_plots
from analysis.research_registry import ROOT, read_registry
from dynamics.dynamic_vehicle import DynamicVehicle


def test_saved_figures_never_sample_or_integrate(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved-data reconstruction must not simulate or resample")

    monkeypatch.setattr(DynamicVehicle, "step", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.chdir(tmp_path)
    captured = []

    def inspect(fig, root, name, sources, **kwargs):
        assert all(Path(p).is_file() for p in sources)
        fig.canvas.draw()
        captured.append((name, fig))

    monkeypatch.setattr(dynamic_plots, "publish", inspect)
    for question in ["Q01", "Q02", "Q03", "Q04", "Q05"]:
        root = ROOT / f"runs/seven_state/{question}/seven_state_{question.lower()}_v1"
        dynamic_plots.generate(root, question)
    for q, name, stage in [
        ("Q04", "budget", "budget"),
        ("Q05", "budget", "budget"),
        ("Q05", "secondary", "secondary"),
        ("Q05", "rate", "rate"),
    ]:
        dynamic_plots.generate(
            ROOT / f"runs/seven_state/{q}/seven_state_{q.lower()}_{name}_v1", stage
        )
    assert len(captured) == 13
    microscope = next(fig for name, fig in captured if name == "Q03_microscope")
    assert len({ax.get_xlim() for ax in microscope.axes}) == 1
    assert len({ax.get_ylim() for ax in microscope.axes}) == 1
    analytic = next(fig for name, fig in captured if name == "Q01_analytic")
    angles = analytic.axes[0].lines[0].get_xdata()
    assert len(angles) == len(set(angles))  # no mixing distinct warm-start conditions
    dynamic_plots.plt.close("all")


def test_curated_figure_provenance_can_be_rebuilt_from_clone(tmp_path, monkeypatch):
    source = ROOT / "runs/seven_state/Q01/seven_state_q01_v1"
    run = tmp_path / "run"
    run.mkdir()
    for name in ["manifest.json", "summary.csv", "cases.json"]:
        shutil.copy2(source / name, run / name)
    monkeypatch.chdir(tmp_path)
    dynamic_plots.generate(run, "Q01")
    provenance = json.loads((run / "figure_provenance.json").read_text())
    assert len(provenance["figure_sha256"]) == 64
    assert (tmp_path / provenance["figure"]).is_file()
    assert (run / "figures/Q01_analytic.svg").is_file()


def test_models_and_unresolved_references_are_not_mislabeled():
    registry = read_registry()
    for q in registry["questions"][:5]:
        assert q["primary_run_id"].startswith("seven_state_")
        assert q["dynamics_model"] == (
            "analytic_gaussian" if q["id"] in ["Q01", "Q02"] else "dynamic_single_track_7"
        )
    assert registry["questions"][5]["status"] == "PLANNED"
    references = json.loads(
        (ROOT / "runs/seven_state/Q03/seven_state_q03_v1/proposal_and_reference.json").read_text()
    )
    assert references["pre_obstacle"]["reference_status"]["status"] == "unresolved"
    assert references["post_disturbance"]["reference_status"]["status"] == "diagnostically_usable"


def test_registry_rejects_model_and_source_mislabeling():
    from analysis.research_registry import validate

    r = read_registry()
    r["questions"][2]["dynamics_model"] = "kinematic_bicycle_4"
    with pytest.raises(ValueError, match="dynamics model"):
        validate(r)
