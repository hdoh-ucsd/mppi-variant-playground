"""Validate scientific evidence and render question-indexed audience documents."""

import hashlib
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "id title document status research_question mathematical_motivation hypothesis configuration "
    "independent_variables controlled_variables metrics run_ids key_figure source_figure summary "
    "main_finding interpretation limitations reproduce regenerate "
    "experiment_name dynamics_model control_inputs controller_configuration seeds sampling_budget "
    "source_commit primary_run_id supporting_figures"
).split()


def read_registry(root=ROOT):
    return yaml.safe_load((Path(root) / "results/registry.yaml").read_text())


def validate(registry, root=ROOT):
    root = Path(root)
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported registry schema")
    runs, questions = registry["runs"], registry["questions"]
    if len({r["id"] for r in runs}) != len(runs):
        raise ValueError("Duplicate run IDs")
    if [q["id"] for q in questions] != [f"Q{i:02d}" for i in range(1, 7)]:
        raise ValueError("Exactly Q01–Q06 are required")
    by_id = {r["id"]: r for r in runs}
    for run in runs:
        for field in (
            "id question_ids status path manifest source_commit configuration summary "
            "dynamics_model control_inputs experiment_name"
        ).split():
            if field not in run:
                raise ValueError(f"Missing provenance field {field}")
        for field in ("path", "manifest", "configuration", "summary"):
            if not (root / run[field]).exists():
                raise ValueError(f"Missing {field}: {run[field]}")
        manifest = json.loads((root / run["manifest"]).read_text())
        if not manifest.get("complete") or run["status"] != "EXECUTED":
            raise ValueError("Figure claims require completed results")
        commit = manifest.get("git", {}).get("commit", "")
        if not re.fullmatch("[0-9a-f]{40}", commit) or commit != run["source_commit"]:
            raise ValueError("Missing or mismatched provenance")
        if (root / run["summary"]).stat().st_size < 20:
            raise ValueError("Empty results summary")
        configuration = yaml.safe_load((root / run["configuration"]).read_text())
        digest = hashlib.sha256(
            json.dumps(configuration, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if digest != manifest.get("configuration_sha256"):
            raise ValueError("Configuration provenance hash mismatch")
    for q in questions:
        for path in q.get("supporting_figures", []):
            if not (root / path).is_file():
                raise ValueError("Broken supporting figure link")
        if set(FIELDS) - q.keys():
            raise ValueError("Missing question fields")
        for field in ("document", "configuration"):
            if not (root / q[field]).is_file():
                raise ValueError(f"Missing {field}: {q[field]}")
        if q["status"] == "PLANNED":
            if (
                q["key_figure"]
                or q["run_ids"]
                or q["summary"]
                or q["source_commit"]
                or q["primary_run_id"]
            ):
                raise ValueError("Planned question cannot claim figures or results")
            continue
        if (
            q["status"] != "EXECUTED"
            or not q["run_ids"]
            or any(r not in by_id for r in q["run_ids"])
        ):
            raise ValueError("Figure claim without corresponding results")
        if any(q["id"] not in by_id[r]["question_ids"] for r in q["run_ids"]):
            raise ValueError("Question/run association disagrees")
        primary_run = by_id[q["run_ids"][0]]
        if (
            q["primary_run_id"] != primary_run["id"]
            or q["source_commit"] != primary_run["source_commit"]
        ):
            raise ValueError("Question source provenance disagrees with primary run")
        if q["dynamics_model"] != primary_run["dynamics_model"]:
            raise ValueError("Question dynamics model disagrees with primary run")
        if (
            q["summary"] != primary_run["summary"]
            or q["configuration"] != primary_run["configuration"]
        ):
            raise ValueError("Question evidence disagrees with its source run")
        if not q["key_figure"] or not (root / q["key_figure"]).is_file():
            raise ValueError("Broken figure link")
        if not q["summary"] or not (root / q["summary"]).is_file():
            raise ValueError("Missing data summary")
        if q["source_figure"]:
            source = root / q["source_figure"]
            if not source.is_file() or source.read_bytes() != (root / q["key_figure"]).read_bytes():
                raise ValueError("Curated figure disagrees with source evidence")
        provenance = by_id[q["run_ids"][0]].get("figure_provenance")
        if provenance:
            data = json.loads((root / provenance).read_text())
            actual = hashlib.sha256((root / q["key_figure"]).read_bytes()).hexdigest()
            if data["figure_sha256"] != actual:
                raise ValueError("Figure provenance hash mismatch")
            summary_hash = hashlib.sha256((root / q["summary"]).read_bytes()).hexdigest()
            if data["summary_sha256"] != summary_hash:
                raise ValueError("Figure summary provenance hash mismatch")
            for path, digest in data.get("inputs_sha256", {}).items():
                source = root / path
                if (
                    not source.is_file()
                    or hashlib.sha256(source.read_bytes()).hexdigest() != digest
                ):
                    raise ValueError("Figure input provenance hash mismatch")
            if data.get("plot_source_sha256"):
                code = root / "src/analysis/dynamic_plots.py"
                if hashlib.sha256(code.read_bytes()).hexdigest() != data["plot_source_sha256"]:
                    raise ValueError("Figure reconstruction source is stale")
    return dict(
        questions=len(questions),
        executed=sum(q["status"] == "EXECUTED" for q in questions),
        runs=len(runs),
    )


def render(registry, root=ROOT):
    root = Path(root)
    backtick = chr(96)
    fence = backtick * 3
    intro = r"""# MPPI Variant Playground

## Project objective

A research companion for understanding MPPI formulations through their mathematics,
finite-sample decisions and control behavior. Follow a scientific question from hypothesis
to reproducible evidence, including negative results and unresolved estimates.

## The MPPI formulation

MPPI samples control sequences, predicts trajectories, scores them, updates a nominal
sequence and executes its first command. The implemented Gaussian IT-MPPI uses

$$p(Z)=\prod_t\mathcal N(z_t;0,C_0),\quad q_U(Z)=\prod_t\mathcal N(z_t;U_t,C),$$
$$\pi(Z)\propto e^{-J(Z)/\lambda}p(Z),\qquad w_k\propto
e^{-J(Z_k)/\lambda}p(Z_k)/q_U(Z_k),\qquad U^+=\sum_k w_kZ_k.$$

The base stays fixed when proposal covariance changes. Densities use normalized latent
controls, weights are normalized in log space, and physical limits apply inside
prediction. Finite self-normalized estimates can be weight-degenerate. See the
[derivation](docs/formulations.md).

## Paper formulation map

§§§mermaid
flowchart LR
  A["Original MPPI · Williams 2016"] --> B["IT-MPPI · Williams 2017<br/>Implemented formulation"]
  Z["Zero-order tutorial · Jordana 2025"] -.-> B
  B --> C["Tsallis-VI · Wang 2021<br/>Weighting law · planned"]
  B --> D["DIAL-MPC · Xue 2024<br/>Annealing · planned"]
  B --> E["Tube-MPPI · Williams 2018<br/>Tracking architecture · planned"]
  E --> F["Robust-MPPI · Gandhi 2021<br/>Tracking-aware planning · planned"]
  B --> G["CC-MPPI · Yin 2022<br/>State covariance / feedback · planned"]
§§§

Arrows are conceptual reading connections. [Eight papers, equation locations and source
versions](docs/paper_map.md) separate paper-level claims from the implementation and
experimentally verified claims.

## Implemented models

| Model | State dimension | Controls | Evidence |
| --- | --- | --- | --- |
| Kinematic bicycle | 4: x, y, yaw, speed | acceleration, steering | Historical baseline, retained |
| Steering-rate bicycle | 5 | acceleration, steering rate | Historical S-turn |
| Dynamic single-track | 7 | steering, signed throttle | Primary Q03–Q05 campaign |

The seven-state state is `[px,py,heading,roll,vx,vy,yaw_rate]` in SI units.
It uses saturating tire forces, body lateral/yaw dynamics and an explicit
first-order roll equation. It is a documented physics approximation, not the paper's learned
AutoRally model.
[Equations, validation and scientific report](docs/seven_state_benchmark_report.md).

## Research questions

Q01 and analytic Q02 are independent of vehicle dynamics. Q03–Q05 below use the seven-state
vehicle. Historical four/five-state evidence remains separately labeled in each question.
Q06 is a paper/formulation investigation with no implemented alternative algorithms.

## Key results

[Research index](docs/research_index.md) · [Registry](results/registry.yaml) · [Historical
evidence](results/archive/legacy_index.md)

""".replace("§", backtick)
    readme = [intro]
    index = [
        "# Research index",
        "",
        "Stable IDs connect each question to hypotheses, controls, evidence and limitations.",
        "",
        "| Question | Status | Finding |",
        "| --- | --- | --- |",
    ]
    for q in registry["questions"]:
        document = [f"# {q['id']} — {q['title']}", "", f"**Status: {q['status']}**", ""]
        for field in (
            "research_question",
            "mathematical_motivation",
            "hypothesis",
            "independent_variables",
            "controlled_variables",
            "metrics",
            "main_finding",
            "interpretation",
            "limitations",
        ):
            document.extend([f"## {field.replace('_', ' ').capitalize()}", "", q[field], ""])
        document.extend(["## Registered experiment", "", "| Field | Value |", "| --- | --- |"])
        for field in (
            "experiment_name",
            "dynamics_model",
            "control_inputs",
            "controller_configuration",
            "seeds",
            "sampling_budget",
            "source_commit",
        ):
            document.append(f"| {field.replace('_', ' ')} | {q[field]} |")
        document.append("")
        document.extend(
            [
                "## Configuration and provenance",
                "",
                f"[Original configuration](../../{q['configuration']})",
                "",
                "Source run IDs: " + (", ".join(q["run_ids"]) or "none — PLANNED"),
                "",
            ]
        )
        for run in registry["runs"]:
            if run["id"] in q["run_ids"]:
                document.extend(
                    [
                        (
                            f"- [Run {run['id']}](../../{run['path']}/): "
                            f"[manifest](../../{run['manifest']}), source "
                            f"{backtick}{run['source_commit']}{backtick}; "
                            f"model: {run['dynamics_model']}; inputs: {run['control_inputs']}."
                        ),
                        "",
                    ]
                )
        readme.extend([f"### {q['id']} — {q['title']}", ""])
        if q["key_figure"]:
            document.extend(
                [
                    "## Selected figure",
                    "",
                    f"![{q['title']}](../../{q['key_figure']})",
                    "",
                    f"[Numeric summary](../../{q['summary']})",
                    "",
                ]
            )
            readme.extend([f"![{q['title']}]({q['key_figure']})", ""])
        else:
            readme.extend(["**PLANNED — no executed result or figure.**", ""])
        for field in ("main_finding", "interpretation", "limitations"):
            label = "Finding" if field == "main_finding" else field.capitalize()
            readme.extend([f"**{label}:** {q[field]}", ""])
        readme.extend(
            [
                f"- [Detailed experiment]({q['document']})",
                f"- [Original configuration]({q['configuration']})",
            ]
        )
        if q["summary"]:
            readme.append(f"- [Data summary]({q['summary']})")
        readme.extend([f"- [Reproduction commands]({q['document']}#reproduce-and-regenerate)", ""])
        document.extend(
            [
                "## Reproduce and regenerate",
                "",
                fence + "bash",
                q["reproduce"],
                q["regenerate"],
                fence,
                "",
            ]
        )
        if q["supporting_figures"]:
            document.extend(["## Supporting figures", ""])
            document.extend(
                [f"- [Saved-data figure](../../{path})" for path in q["supporting_figures"]]
            )
            document.append("")
        if q["id"] != "Q06":
            document.extend(
                [
                    "[Seven-state campaign report and methods](../seven_state_benchmark_report.md)",
                    "",
                ]
            )
        methods = {
            "Q01": "experimental_findings.md",
            "Q02": "covariance_mechanisms.md",
            "Q03": "frozen_decision_analysis.md",
            "Q04": "trajectory_behavior_findings.md",
            "Q05": "Q05_methods.md",
        }
        if q["id"] in methods:
            document.extend(
                [
                    "[Historical methods (earlier model / original analytic seeds)]"
                    f"(../{methods[q['id']]})",
                    "",
                ]
            )
        (root / q["document"]).write_text("\n".join(document))
        index.append(
            f"| [{q['id']} — "
            f"{q['title']}]({Path(q['document']).relative_to('docs')}) | "
            f"{q['status']} | {q['main_finding']} |"
        )
    storage = """
## Storage and provenance

- Full new raw records belong in §runs/seven_state/Qxx/<run_id>/§. Historical run names are relative
  symlinks to original tracked evidence, not copied result trees.
- §results/showcase/§ holds one selected PNG per executed question. Other figures stay
  accessible from question documents.
- Raw episodes, arrays, snapshots, animations and redundant plots are ignored. Small
  top-level CSV/JSON summaries, exact YAML and manifests remain versionable under §runs/§.
- [Legacy index](../results/archive/legacy_index.md) preserves original paths and
  provenance. Historical experiments were not rerun for this reorganization.

Validate or rebuild the audience documents:

§§§bash
uv run python experiments/research_index.py --render
uv run python experiments/research_index.py
§§§
""".replace("§", backtick)
    (root / "docs/research_index.md").write_text("\n".join(index) + storage)
    outro = """
## Reproduction

Linux, Python 3.11+ and uv; CPU NumPy, no GPU required.

§§§bash
uv sync --frozen
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python experiments/research_index.py
§§§

Use fresh run IDs for simulation; immutable completed directories are never overwritten.
Exact commands are in each question and the [report](docs/seven_state_benchmark_report.md).
The following rebuilds a figure from committed compact data without simulation:

§§§bash
uv run python experiments/plot_seven_state.py --stage Q03 \
  --input runs/seven_state/Q03/seven_state_q03_v1
§§§

Full raw episodes and rollout NPZs are ignored local evidence. **Cloning does not contain
those raw records.** No LFS/external store is configured; rerun recorded commands to
recreate them. Small summaries, inputs, configs, manifests and selected figures are tracked.
Historical §results/investigations/§ evidence is preserved.

§§§text
configs/       Declared models, costs, controllers and experiment designs
src/           Shared MPPI, sampling, dynamics, geometry and analysis
experiments/   Execution, reconstruction and registry validation CLIs
docs/          Question documents, derivations, paper map and findings
results/       Registry, curated showcase and preserved historical evidence
runs/seven_state/  Question-based new campaign records (bulk raw ignored)
tests/         Numerical, physical and evidence-consistency checks
§§§

## Limitations

These are covariance settings of one Gaussian IT-MPPI, not different published algorithms.
Specific matrices, one calibration, one S-turn handedness, one horizon and broad binomial
intervals limit generalization. ESS is not estimator accuracy. Two Q03 reference means
remain unresolved. The finish region allows runout recovery, not stopped-point parking.
The model omits load transfer, roll inertia/feedback and an actuator-rate state; it has
no reverse drive. Equal-K sweeps are not equal-wall-clock experiments or real-time guarantees.
See the [scientific report](docs/seven_state_benchmark_report.md) for failures, uncertainty
and the deferred steering-rate study.

""".replace("§", backtick)
    (root / "README.md").write_text("\n".join(readme) + outro)
