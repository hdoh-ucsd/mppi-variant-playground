"""Study artifacts and paired descriptive summaries; existing runs are never overwritten."""

import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.metrics import binomial_interval
from evaluation.runner import git_info, write_json

REPO = Path(__file__).resolve().parents[2]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def begin(output, config, level):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    code = hashlib.sha256()
    for folder in ("src", "experiments", "configs/investigations"):
        for path in sorted((REPO / folder).rglob("*")):
            if path.suffix in {".py", ".yaml"}:
                code.update(str(path.relative_to(REPO)).encode())
                code.update(path.read_bytes())
    manifest = {
        "schema_version": 1,
        "level": level,
        "phase": config["phase"],
        "started_at_utc": datetime.now(UTC).isoformat(),
        "git": git_info(),
        "source_sha256": code.hexdigest(),
        "configuration_sha256": fingerprint(config),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib")},
        "threads": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "rng": "NumPy PCG64; common standard-normal draws per seed and dimension",
        "complete": False,
    }
    write_json(output / "manifest.json", manifest)
    return output, manifest


def finish(output, manifest, cases, trials, interpretation):
    write_json(output / "cases.json", cases)
    frame = pd.DataFrame(trials)
    frame.to_csv(output / "trials.csv", index=False)
    summaries = []
    for case_id, group in frame.groupby("case_id", sort=False):
        result = {"case_id": case_id, "n": len(group)}
        for metric in (
            "error",
            "squared_error",
            "ess",
            "weight_entropy",
            "max_weight",
            "best_cost",
            "mean_cost",
            "basin_success",
            "saturation_fraction",
            "cumulative_evaluation_cost",
            "success",
            "mean_planning_latency_s",
            "normalized_applied_covariance_trace",
            "normalized_applied_second_moment",
        ):
            if metric in group:
                x = group[metric].to_numpy(dtype=float)
                result.update(
                    {
                        f"{metric}_mean": float(x.mean()),
                        f"{metric}_sd": float(x.std(ddof=1)) if len(x) > 1 else None,
                        f"{metric}_q05": float(np.quantile(x, 0.05)),
                        f"{metric}_q50": float(np.quantile(x, 0.5)),
                        f"{metric}_q95": float(np.quantile(x, 0.95)),
                    }
                )
        estimate_columns = [
            column
            for column in group
            if column.startswith("estimate_") and not group[column].isna().all()
        ]
        if estimate_columns:
            values = group[estimate_columns].to_numpy()
            exact = np.array(
                [cases[int(case_id)]["target_mean"][i] for i in range(values.shape[1])]
            )
            # ddof=1 sample covariance trace; MSE uses the empirical expectation (ddof=0).
            bias = values.mean(axis=0) - exact
            result["bias_norm"] = float(np.linalg.norm(bias))
            result["estimator_variance_trace"] = float(values.var(axis=0, ddof=1).sum())
            for i, value in enumerate(bias):
                result[f"bias_{i}"] = float(value)
        for binary in ("success", "basin_success"):
            if binary in group:
                interval = binomial_interval(group[binary].to_numpy(dtype=float))
                result[f"{binary}_ci95_low"] = interval["ci95_low"]
                result[f"{binary}_ci95_high"] = interval["ci95_high"]
        summaries.append(result)
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    paired = []
    if "comparison_id" in frame:
        for comparison, group in frame.groupby("comparison_id", sort=False):
            ref = group[group.proposal_name == "isotropic"].set_index("seed")
            if ref.empty:
                continue
            for case_id, other in group.groupby("case_id", sort=False):
                other = other.set_index("seed")
                if other.proposal_name.iloc[0] == "isotropic":
                    continue
                for metric in (
                    "squared_error",
                    "ess",
                    "best_cost",
                    "success",
                    "cumulative_evaluation_cost",
                    "saturation_fraction",
                    "normalized_applied_covariance_trace",
                ):
                    if metric in other:
                        delta = other[metric].astype(float) - ref[metric].astype(float)
                        if delta.isna().any():
                            raise ValueError("Unpaired seeds in comparison")
                        paired.append(
                            {
                                "comparison_id": comparison,
                                "case_id": case_id,
                                "metric": metric,
                                "n": len(delta),
                                "mean_difference": float(delta.mean()),
                                "sd_difference": float(delta.std(ddof=1)),
                                "q05": float(delta.quantile(0.05)),
                                "q95": float(delta.quantile(0.95)),
                            }
                        )
    pd.DataFrame(
        paired,
        columns=[
            "comparison_id",
            "case_id",
            "metric",
            "n",
            "mean_difference",
            "sd_difference",
            "q05",
            "q95",
        ],
    ).to_csv(output / "paired_differences.csv", index=False)
    (output / "interpretation.md").write_text(interpretation + "\n")
    manifest.update({"complete": True, "cases": len(cases), "trials": len(trials)})
    if "cost_evaluations" in frame:
        manifest["cost_evaluations"] = int(frame.cost_evaluations.sum())
    write_json(output / "manifest.json", manifest)
    return frame
