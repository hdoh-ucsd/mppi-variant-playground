"""Relocations preserve evidence, provenance resolution and runnable entry points."""

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def catalog():
    return json.loads((ROOT / "results/relocations.json").read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def resolve_original(path, mapping):
    for old, new in mapping.items():
        if path == old or path.startswith(old + "/"):
            return new + path[len(old) :]
    return path


def test_relocated_results_and_cli_bodies_are_preserved():
    index = catalog()
    assert len(index["preserved_result_sha256"]) == 148
    assert len(index["preserved_entrypoint_sha256"]) == 34
    for group in ["preserved_result_sha256", "preserved_entrypoint_sha256"]:
        for path, expected in index[group].items():
            assert digest(path) == expected, path
    assert not list((ROOT / "experiments").glob("*.py"))
    for old, new in index["experiment_entrypoints"].items():
        assert not (ROOT / old).exists()
        assert (ROOT / new).is_file()
    for old, new in index["result_directories"].items():
        assert not (ROOT / old).exists()
        assert (ROOT / new).is_dir()


def test_original_benchmark_provenance_still_resolves_with_relocation_map():
    mapping = catalog()["result_directories"]
    studies = sorted((ROOT / "results/variants/benchmarks").glob("V0*"))
    assert len(studies) == 6
    for study in studies:
        provenance = json.loads((study / "primary_provenance.json").read_text())
        assert (
            digest(resolve_original(provenance["figure"], mapping)) == provenance["figure_sha256"]
        )
        for path, expected in provenance["inputs_sha256"].items():
            assert digest(resolve_original(path, mapping)) == expected


def test_all_relocated_entrypoints_have_working_help():
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MPLBACKEND="Agg")

    def help_for(path):
        result = subprocess.run(
            [sys.executable, path, "--help"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert result.returncode == 0, (path, result.stderr)
        assert "usage:" in result.stdout.lower(), path

    # Only argument help is evaluated, never experiment execution or figure generation.
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(help_for, catalog()["experiment_entrypoints"].values()))


def test_historical_source_check_accepts_only_the_declared_path_edit(tmp_path):
    from analysis.research_registry import matches_plot_source

    index = catalog()
    relative = "src/analysis/dynamic_plots.py"
    expected = index["source_path_edits"][relative]["before_sha256"]
    assert matches_plot_source(ROOT / relative, expected)
    code = tmp_path / relative
    code.parent.mkdir(parents=True)
    code.write_bytes((ROOT / relative).read_bytes())
    record = tmp_path / "results/relocations.json"
    record.parent.mkdir()
    record.write_text(json.dumps(index))
    assert matches_plot_source(code, expected, tmp_path)
    code.write_text(code.read_text().replace("dpi=150", "dpi=75"))
    assert not matches_plot_source(code, expected, tmp_path)
    # Updating the new hash alone cannot bless a non-path change.
    index["source_path_edits"][relative]["after_sha256"] = hashlib.sha256(
        code.read_bytes()
    ).hexdigest()
    record.write_text(json.dumps(index))
    assert not matches_plot_source(code, expected, tmp_path)
