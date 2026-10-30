"""Explicit recovery of a failed JSON finalization from existing complete raw records."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis.dynamic_audit import driving_audit
from analysis.dynamic_campaign import finish_records
from evaluation.runner import write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    args = p.parse_args()
    root = Path(args.input)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["complete"]:
        raise ValueError("Completed records are immutable; no recovery needed")
    config = yaml.safe_load((root / "config.yaml").read_text())
    frame = pd.read_csv(root / "per_seed.csv")
    conditions = json.loads((root / "conditions.json").read_text())
    cohorts = {
        cohort: sorted(group.seed.unique().tolist()) for cohort, group in frame.groupby("cohort")
    }
    expected = (
        sum(len(seeds) for seeds in cohorts.values()) * len(conditions) * len(config["proposals"])
    )
    if len(frame) != expected:
        raise ValueError("Raw record set is incomplete")
    paths = {}
    diagnostics = {}
    for row in frame.itertuples():
        folder = Path(row.episode_path)
        data = json.loads((folder / "result.json").read_text())
        if data["success"] != row.success or data["steps"] != row.steps:
            raise ValueError("Summary/raw mismatch")
        if row.seed in [42, 31000, 32000]:
            key = f"{row.case}__{row.proposal}__{row.cohort}"
            paths[key] = pd.read_csv(folder / "trajectory.csv").to_dict("list")
            diag = pd.read_csv(folder / "diagnostics.csv").replace([np.inf, -np.inf], np.nan)
            diagnostics[key] = diag.astype(object).where(pd.notna(diag), None).to_dict("list")
    core = hashlib.sha256()
    for file in [
        "src/controllers/mppi.py",
        "src/dynamics/dynamic_vehicle.py",
        "src/costs/dynamic_vehicle.py",
        "src/scenarios/dynamic_scenes.py",
        "src/sampling/gaussian.py",
    ]:
        core.update(Path(file).read_bytes())
    manifest["engine_source_sha256"] = core.hexdigest()
    manifest["finalization_repair"] = (
        "No-obstacle infinite clearance serialized as JSON null; recovered from "
        "original raw records without controller execution"
    )
    finish_records(root, config, frame.to_dict("records"), paths, diagnostics, manifest, cohorts)
    try:
        print(driving_audit(root))
    except Exception:
        manifest["complete"] = False
        write_json(root / "manifest.json", manifest)
        raise
