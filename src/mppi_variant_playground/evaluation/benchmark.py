from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from mppi_variant_playground.evaluation.metrics import mean_interval
from mppi_variant_playground.evaluation.runner import make_controller, run_episode, write_json
from mppi_variant_playground.utils.config import DEFAULT_ROOT, load_experiment, read_yaml


def calibrate(config, candidates, target):
    """Offline warmed median latency calibration; no hard real-time guarantee."""
    measurements = []
    for samples in candidates:
        trial = deepcopy(config)
        trial["controller"]["samples"] = samples
        controller = make_controller(trial)
        initial = np.array(trial["scenario"]["initial_state"])
        controller.reset(initial, 12345)  # calibration seed is outside evaluation seeds
        controller.command(initial)
        times = []
        for _ in range(5):
            controller.command(initial)
            times.append(controller.get_diagnostics()["planning_latency_s"])
        measurements.append({"samples": samples, "median_s": float(np.median(times))})
    eligible = [row for row in measurements if row["median_s"] <= target]
    if not eligible:
        raise ValueError("Wall-clock target is below the smallest calibrated batch latency")
    return max(eligible, key=lambda row: row["samples"])["samples"], measurements


def summarize(results, output):
    frame = pd.DataFrame(
        [
            {**{k: row[k] for k in ["controller", "scenario", "seed", "mode"]}, **row["metrics"]}
            for row in results
        ]
    )
    output = Path(output)
    frame.to_csv(output / "per_seed.csv", index=False)
    summary = []
    for keys, group in frame.groupby(["mode", "scenario", "controller"], sort=False):
        for metric in frame.columns[4:]:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            summary.append(
                dict(zip(["mode", "scenario", "controller"], keys, strict=True))
                | {"metric": metric}
                | mean_interval(values)
            )
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    write_json(output / "summary.json", summary)
    paired = []
    for (mode, scenario), group in frame.groupby(["mode", "scenario"]):
        for metric in [
            "cumulative_evaluation_cost",
            "goal_position_error",
            "mean_ess",
            "total_normalized_control_variation",
            "mean_planning_latency_s",
        ]:
            pivot = group.pivot(index="seed", columns="controller", values=metric)
            if "standard" in pivot:
                for name in pivot.columns.drop("standard"):
                    paired.append(
                        {
                            "mode": mode,
                            "scenario": scenario,
                            "controller": name,
                            "reference": "standard",
                            "metric": metric,
                            **mean_interval((pivot[name] - pivot.standard).to_numpy()),
                        }
                    )
    write_json(output / "paired_differences.json", paired)
    return frame


def run_benchmark(benchmark_path, output, mode=None, seeds=None, root=DEFAULT_ROOT):
    benchmark = read_yaml(benchmark_path)
    mode = mode or benchmark["mode"]
    if mode not in {"dynamics", "wall_clock", "native"}:
        raise ValueError("Unknown budget mode")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "benchmark.json", {**benchmark, "mode": mode, "seeds": seeds or benchmark["seeds"]}
    )
    results = []
    for scenario in benchmark["scenarios"]:
        for name in benchmark["controllers"]:
            config = load_experiment(name, scenario, root)
            config["controller"]["horizon"] = benchmark["horizon"]
            if mode == "dynamics":
                config["controller"]["samples"] = benchmark["samples"]
            if benchmark["sampling_scale_constraint"] != "matched_trace":
                raise ValueError(
                    "This benchmark implements only normalized matched-trace comparison"
                )
            if not np.isclose(
                np.trace(config["controller"]["covariance"]), benchmark["normalized_trace"]
            ):
                raise ValueError("Covariance trace constraint violated")
            config["budget"] = {
                "mode": mode,
                "sampling_scale_constraint": "normalized_matched_trace",
            }
            if mode == "wall_clock":
                count, timing = calibrate(
                    config, benchmark["wall_clock_candidates"], benchmark["wall_clock_target_s"]
                )
                config["controller"]["samples"] = count
                config["budget"].update(
                    {"target_s": benchmark["wall_clock_target_s"], "calibration": timing}
                )
            for seed in seeds or benchmark["seeds"]:
                result = run_episode(config, seed, output / scenario / name / f"seed_{seed}", mode)
                results.append(result)
                m = result["metrics"]
                print(
                    f"{mode} {scenario:16s} {name:9s} seed={seed}: "
                    f"success={m['success']} collision={m['collision']} "
                    f"error={m['goal_position_error']:.3f} ESS={m['mean_ess']:.1f}",
                    flush=True,
                )
    summarize(results, output)
    return results
