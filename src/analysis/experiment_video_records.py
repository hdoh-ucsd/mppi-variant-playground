"""Extract compact, lossless render inputs; only an explicit option can replay V04."""

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STATE_FIELDS = ["px", "py", "heading", "roll", "vx", "vy", "yaw_rate"]
VERSION = "1.0"
SELECTION = (
    "Fixed before rendering: dynamics-budget protocol, first declared seed 48000; "
    "two-route for V01/V02; positive targeted impulse and first seed 51000 for V03; "
    "original disturbance seed 48000 plus positive-offset threshold-100 frozen seed "
    "51000 for V04; narrow-passage, full C0 noise, all 256 seeds 56000..56255 for V05. "
    "Display first 32 stochastic paths and first 16 oriented vehicles, never rank outcomes."
)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def episode(folder, prefix, arrays, sources):
    folder = Path(folder)
    cfg = yaml.safe_load((folder / "config.yaml").read_text())
    tr = pd.read_csv(folder / "trajectory.csv")
    diag = pd.read_csv(folder / "diagnostics.csv")
    arrays[prefix + "states"] = tr[STATE_FIELDS].to_numpy()
    arrays[prefix + "time"] = tr.time_s.to_numpy()
    arrays[prefix + "progress"] = tr.progress_m.to_numpy()
    arrays[prefix + "controls"] = pd.read_csv(folder / "controls.csv").to_numpy()
    columns = [
        c
        for c in diag
        if c.startswith(
            ("nominal_state_", "applied_correction_", "tracking_error_vector_", "planner_control_")
        )
    ]
    columns += [
        c
        for c in [
            "reset_event",
            "nominal_selection",
            "nominal_selection_infeasible",
            "applied_tracking_control_norm",
            "sampled_feedback_rms",
            "ess",
        ]
        if c in diag
    ]
    for c in columns:
        arrays[prefix + c] = diag[c].to_numpy()
    assert len(tr) == len(diag) + 1
    assert np.allclose(np.diff(tr.time_s), cfg["vehicle"]["dt"], atol=1e-12)
    files = ["trajectory.csv", "controls.csv", "diagnostics.csv", "config.yaml", "result.json"]
    sources.append(
        {
            "path": str(folder),
            "files_sha256": {f: digest(folder / f) for f in files},
            "config": cfg,
            "result": read_json(folder / "result.json"),
        }
    )
    return cfg


def replay_robust(raw, allow):
    original = raw / "V04_robust/dynamics_budget/disturbance/robust_mppi/seed_48000"
    target = raw / "videos/V04_robust_seed_48000"
    if not target.exists():
        if not allow:
            raise FileNotFoundError(
                "Missing V04 vector records. Use --prepare-records "
                "--replay-missing to reproduce this one original episode."
            )
        from analysis.dynamic_campaign import run_episode

        config = yaml.safe_load((original / "config.yaml").read_text())
        config["execution"]["extra_snapshot_seeds"] = [48000]
        run_episode(
            config,
            "standard",
            48000,
            target,
            capture_steps=list(range(config["execution"]["max_steps"])),
        )
    errors = {}
    for name in ["trajectory", "controls"]:
        a = pd.read_csv(original / f"{name}.csv").to_numpy()
        b = pd.read_csv(target / f"{name}.csv").to_numpy()
        if a.shape != b.shape:
            raise ValueError("Replay length differs from original episode")
        errors[name + "_max_abs_error"] = float(np.max(np.abs(a - b)))
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    return original, target, errors


def mechanism_config(frozen, algorithm, samples, **changes):
    config = deepcopy(frozen["config"])
    options = yaml.safe_load(Path("configs/variants/benchmark.yaml").read_text())
    config["controller"].update(options["algorithms"][algorithm])
    config["controller"].update(algorithm=algorithm, samples=samples, snapshot_steps=[0], **changes)
    return config


def prepare(output, raw="runs/variants", replay_missing=False):
    output, raw = Path(output), Path(raw)
    records = output / "records"
    records.mkdir(parents=True, exist_ok=True)
    manifest = {
        "rendering_version": VERSION,
        "preparation_git_commit": git_commit(),
        "selection_rule": SELECTION,
        "clips": {},
    }

    def finish(name, arrays, sources, **details):
        path = records / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        study = "V02" if name == "V02_dial_decision" else name[:3]
        if name in ["V01_tsallis", "V02_dial"]:
            source_manifest = Path("results/variants") / name / "manifest.json"
        else:
            source_manifest = Path("results/variants/mechanisms") / study / "manifest.json"
        manifest["clips"][name] = dict(
            source_git=read_json(source_manifest)["git"],
            records=str(path.relative_to(output)),
            records_sha256=digest(path),
            sources=sources,
            **details,
        )

    for name, study, algorithm in [
        ("V01_tsallis", "V01_tsallis", "tsallis_vi"),
        ("V02_dial", "V02_dial", "dial_mpc"),
    ]:
        arrays, sources = {}, []
        for prefix, alg in [("a_", "it_mppi"), ("b_", algorithm)]:
            episode(
                raw / study / "dynamics_budget/two_route_obstacle" / alg / "seed_48000",
                prefix,
                arrays,
                sources,
            )
        # Audit actual JSON schema and every trajectory field, rather than assuming completeness.
        representative = Path("results/variants") / study / "representative_paths.json"
        saved = read_json(representative)
        for s in sources:
            alg = s["config"]["controller"]["algorithm"]
            frame = pd.read_csv(Path(s["path"]) / "trajectory.csv")
            compact = pd.DataFrame(saved[f"dynamics_budget/two_route_obstacle/{alg}"])
            np.testing.assert_allclose(
                frame.to_numpy(), compact[frame.columns].to_numpy(), rtol=0, atol=1e-12
            )
        finish(
            name,
            arrays,
            sources,
            seeds=[48000],
            scenario="two_route_obstacle",
            mode="replayed",
            representative_paths_schema={
                "columns": frame.columns.tolist(),
                "verified_complete_against_raw": True,
                "sha256": digest(representative),
            },
            missing_diagnostics="No synchronized weights or nominal plans in original episodes.",
        )

    mech = raw / "mechanisms"
    path = mech / "V02/51000_dial_mpc.npz"
    snap = np.load(path)
    frozen = read_json("results/variants/mechanisms/V02/frozen_inputs.json")["two_route_obstacle"]
    table = pd.read_csv("results/variants/mechanisms/V02/summary.csv")
    rows = table[(table.seed == 51000) & (table.algorithm == "dial_mpc")]
    arrays = {
        k: snap[k]
        for k in [
            "inner_sampled_states",
            "inner_nominal_states",
            "inner_variance_scales",
            "current_state",
        ]
    }
    arrays.update(
        planned_cost=rows.planned_cost.to_numpy(), transitions=rows.transitions.to_numpy()
    )
    finish(
        "V02_dial_decision",
        arrays,
        [
            {
                "path": str(path),
                "sha256": digest(path),
                "config": mechanism_config(frozen, "dial_mpc", 95),
            }
        ],
        seeds=[51000],
        source_seed=frozen["source_seed"],
        frozen_step=frozen["step"],
        scenario="two_route_obstacle",
        mode="replayed",
        mechanism_config=yaml.safe_load(Path("configs/variants/mechanisms.yaml").read_text()),
        missing_diagnostics=None,
    )

    arrays, sources = {}, []
    for prefix, alg in [("a_", "tube_0"), ("b_", "tube_450")]:
        episode(mech / "V03/impulse_1" / alg / "51000", prefix, arrays, sources)
    finish(
        "V03_tube",
        arrays,
        sources,
        seeds=[51000],
        scenario="positive_impulse_step_8",
        mode="replayed",
        missing_diagnostics=None,
    )

    original, replay, errors = replay_robust(raw, replay_missing)
    arrays, sources = {}, []
    episode(replay, "a_", arrays, sources)
    snapshot_paths = sorted((replay / "snapshots").glob("*.npz"))
    assert len(snapshot_paths) == len(arrays["a_controls"])
    for key in ["nominal_states", "nominal_candidates", "candidate_free_energies"]:
        arrays["a_" + key] = np.stack([np.load(p)[key] for p in snapshot_paths])
    arrays["a_sampled_states"] = np.stack(
        [np.load(p)["sampled_states"][:8] for p in snapshot_paths]
    )
    fallback = mech / "V04/1_100_51000.npz"
    snap = np.load(fallback)
    for k in [
        "current_state",
        "nominal_states",
        "nominal_candidates",
        "candidate_free_energies",
        "feedback",
    ]:
        arrays["f_" + k] = snap[k]
    for k in ["sampled_states", "nominal_samples"]:
        arrays["f_" + k] = snap[k][:16]
    table = pd.read_csv("results/variants/mechanisms/V04/summary.csv")
    row = table[(table.seed == 51000) & (table.impulse == 1) & (table.threshold == 100)].iloc[0]
    sources.append(
        {
            "path": str(fallback),
            "sha256": digest(fallback),
            "config": mechanism_config(
                read_json("results/variants/mechanisms/V04/frozen_inputs.json")["open_space"],
                "robust_mppi",
                128,
                free_energy_threshold=100,
            ),
        }
    )
    finish(
        "V04_robust",
        arrays,
        sources,
        seeds=[48000, 51000],
        scenario="disturbance + frozen offset",
        mode="one exact episode regenerated for passive diagnostics; frozen records replayed",
        replay_original=str(original),
        replay_validation=errors,
        original_config=yaml.safe_load((original / "config.yaml").read_text()),
        fallback_record=row.to_dict(),
        mechanism_config=yaml.safe_load(Path("configs/variants/mechanisms.yaml").read_text()),
        snapshot_files_sha256={str(p): digest(p) for p in snapshot_paths},
        missing_diagnostics="Original candidate arrays absent; obtained by exact replay.",
    )

    path = mech / "V05/narrow_passage/1_steered.npz"
    planner_path = path.parent / "planner.npz"
    snap, planner = np.load(path), np.load(planner_path)
    arrays = {
        k: snap[k] for k in ["executed_states", "predicted_covariances", "empirical_covariances"]
    }
    arrays["reference_states"] = planner["reference_states"]
    # Recorded transition clearance includes integrator substeps, unlike a state-only test.
    diag_paths = [path.parent / "1_steered" / str(i) / "diagnostics.csv" for i in range(256)]
    contact = np.stack([pd.read_csv(p).clearance.to_numpy() <= 0 for p in diag_paths])
    arrays["contacts"] = np.column_stack([np.zeros(256, dtype=bool), contact])
    cfg = yaml.safe_load((diag_paths[0].parent / "config.yaml").read_text())
    np.testing.assert_allclose(
        np.stack([np.cov(x.T) for x in arrays["executed_states"].swapaxes(0, 1)]),
        arrays["empirical_covariances"],
        atol=1e-12,
    )
    finish(
        "V05_covariance",
        arrays,
        [
            {"path": str(path), "sha256": digest(path), "config": cfg},
            {"path": str(planner_path), "sha256": digest(planner_path)},
        ],
        seeds=list(range(56000, 56256)),
        planner_seed=55000,
        scenario="narrow_passage",
        mode="replayed",
        noise_fraction=1.0,
        displayed_paths=list(range(32)),
        contact_sources_sha256={str(p): digest(p) for p in diag_paths},
        missing_diagnostics=None,
    )
    save_json(output / "manifest.json", manifest)
    return manifest
