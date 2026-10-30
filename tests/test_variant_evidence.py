"""Curated formulation plots must rebuild from a clone without raw data or simulation."""

import json
import shutil
from pathlib import Path

import numpy as np

from analysis import variant_campaign
from dynamics.dynamic_vehicle import DynamicVehicle


def test_variant_plot_reconstructs_without_sampling_or_dynamics(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    source = root / "results/variants/benchmarks/V05_covariance_steering"
    target = tmp_path / "clone_plot"
    target.mkdir()
    for name in ["summary.csv", "per_seed.csv"]:
        shutil.copyfile(source / name, target / name)

    def forbidden(*args, **kwargs):
        raise AssertionError("Saved-data figure must not execute a controller")

    monkeypatch.setattr(DynamicVehicle, "step", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    variant_campaign.plot(target)
    provenance = json.loads((target / "primary_provenance.json").read_text())
    assert (target / "primary.png").is_file()
    assert len(provenance["variant_plot_source_sha256"]) == 64
    assert len(provenance["inputs_sha256"]) == 2
