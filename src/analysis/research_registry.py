"""Validate evidence and render question documents; preserve the curated README."""

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
    return yaml.safe_load((Path(root) / "results/historical/registry.yaml").read_text())


def matches_plot_source(code, expected, root=ROOT):
    """Accept exact source or a reversible, recorded output-directory relocation only."""
    code, root = Path(code), Path(root)
    current = code.read_bytes()
    current_hash = hashlib.sha256(current).hexdigest()
    if current_hash == expected:
        return True
    index_path = root / "results/relocations.json"
    if not index_path.is_file():
        return False
    index = json.loads(index_path.read_text())
    edit = index.get("source_path_edits", {}).get(str(code.relative_to(root)))
    if not edit or edit["before_sha256"] != expected or edit["after_sha256"] != current_hash:
        return False
    restored = current.decode()
    for original, relocated in edit["replacements"].items():
        if index["result_directories"].get(original) != relocated:
            return False
        restored = restored.replace(relocated + "/", original + "/")
    return hashlib.sha256(restored.encode()).hexdigest() == expected


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
                if not matches_plot_source(code, data["plot_source_sha256"], root):
                    raise ValueError("Figure reconstruction source is stale")
    return dict(
        questions=len(questions),
        executed=sum(q["status"] == "EXECUTED" for q in questions),
        runs=len(runs),
    )


def render(registry, root=ROOT):
    """Render one evidence index; the README owns the formulation narrative."""
    root = Path(root)
    lines = [
        "# Historical research evidence index",
        "",
        "This index retains Q01–Q05 configurations, findings and provenance. "
        "The [README](../README.md) contains the current formulations, V01–V06 comparisons "
        "and mechanism figures. Q06 below is the superseded historical plan.",
        "",
        "| Question | Status | Primary numerical evidence |",
        "| --- | --- | --- |",
    ]
    for q in registry["questions"]:
        evidence = f"[Summary](../{q['summary']})" if q["summary"] else "Historical plan only"
        lines.append(
            f"| [{q['id']} — {q['title']}](#{q['id'].lower()}) | {q['status']} | {evidence} |"
        )
    for q in registry["questions"]:
        lines.extend(["", f"## {q['id']}", "", f"**{q['title']}** — {q['status']}", ""])
        for field in (
            "research_question",
            "mathematical_motivation",
            "hypothesis",
            "main_finding",
            "interpretation",
            "limitations",
            "independent_variables",
            "controlled_variables",
            "metrics",
            "dynamics_model",
            "control_inputs",
            "controller_configuration",
            "seeds",
            "sampling_budget",
        ):
            if q[field]:
                lines.extend([f"**{field.replace('_', ' ').capitalize()}:** {q[field]}", ""])
        lines.extend([f"[Exact configuration](../{q['configuration']})", ""])
        if q["key_figure"]:
            lines.extend(
                [
                    f"[Primary figure](../{q['key_figure']}) · "
                    f"[Numerical summary](../{q['summary']})",
                    "",
                ]
            )
        for run in registry["runs"]:
            if run["id"] in q["run_ids"]:
                lines.extend(
                    [
                        f"- **{run['id']}**: [manifest](../{run['manifest']}), "
                        f"[summary](../{run['summary']}), source `{run['source_commit']}`.",
                        "",
                    ]
                )
        for path in q["supporting_figures"]:
            lines.extend([f"[Supporting figure](../{path})", ""])
        lines.extend(["```bash", q["reproduce"], q["regenerate"], "```", ""])
    lines.extend(
        [
            "## Storage and recovery",
            "",
            "- `results/historical/showcase/` retains the five primary Q01–Q05 figures.",
            "- `runs/seven_state/` retains exact configurations, compact summaries and provenance. "
            "Full episodes and rollout arrays are ignored; reproduce them before raw audits.",
            "- `results/variants/` retains current formulation comparisons and mechanism evidence.",
            "- [Historical recovery index](../results/historical/archive/legacy_index.md) "
            "links superseded "
            "documents and bulky earlier results at their immutable pre-cleanup revision.",
            "",
            "[Seven-state model equations, validation and campaign report]"
            "(seven_state_benchmark_report.md)",
            "",
            "Validate or regenerate this single index (the README is maintained separately):",
            "",
            "```bash",
            "uv run python experiments/legacy/research_index.py --render",
            "uv run python experiments/legacy/research_index.py",
            "```",
            "",
        ]
    )
    destination = root / "docs/research_index.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines))
