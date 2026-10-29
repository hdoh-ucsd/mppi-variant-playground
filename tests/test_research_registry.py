"""Scientific claims must resolve to questions, configurations and executed evidence."""

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from analysis.research_registry import FIELDS, ROOT, read_registry, render, validate


def test_registry_and_schema_are_consistent():
    r = read_registry()
    assert validate(r)["questions"] == 6
    schema = json.loads((ROOT / "docs/research_registry.schema.json").read_text())
    assert set(schema["properties"]["questions"]["items"]["required"]) == set(FIELDS)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        ("duplicate", "Duplicate"),
        ("figure", "figure"),
        ("document", "document"),
        ("configuration", "configuration"),
        ("provenance", "provenance"),
        ("unexecuted", "corresponding"),
    ],
)
def test_registry_rejects_missing_or_unbacked_evidence(mutation, pattern):
    r = deepcopy(read_registry())
    if mutation == "duplicate":
        r["runs"].append(deepcopy(r["runs"][0]))
    elif mutation == "provenance":
        r["runs"][0].pop("source_commit")
    elif mutation == "unexecuted":
        r["questions"][0]["run_ids"] = ["never_executed"]
    else:
        r["questions"][0][{"figure": "key_figure"}.get(mutation, mutation)] = "missing.file"
    with pytest.raises(ValueError, match=pattern):
        validate(r)


def test_readme_has_at_most_one_existing_primary_image_per_question():
    text = (ROOT / "README.md").read_text()
    for q in read_registry()["questions"]:
        section = re.split(rf"^### {q['id']} [—·] ", text, maxsplit=1, flags=re.MULTILINE)[1]
        section = re.split(r"^#{2,3} ", section, maxsplit=1, flags=re.MULTILINE)[0]
        images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", section)
        assert len(images) == (1 if q["status"] == "EXECUTED" else 0)
        for path in images:
            assert (ROOT / path).is_file()
    for path in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
        if "://" not in path:
            assert (ROOT / path).exists(), path


def test_showcase_contains_only_registered_primary_figures():
    registry = read_registry()
    expected = {Path(q["key_figure"]).name for q in registry["questions"] if q["key_figure"]}
    assert {p.name for p in (ROOT / "results/showcase").iterdir()} == expected


def test_render_preserves_curated_readme(tmp_path):
    registry = read_registry()
    (tmp_path / "docs/questions").mkdir(parents=True)
    readme = tmp_path / "README.md"
    original = (ROOT / "README.md").read_bytes()
    readme.write_bytes(original)
    render(registry, root=tmp_path)
    assert readme.read_bytes() == original
    assert (tmp_path / "docs/research_index.md").is_file()
    assert all((tmp_path / q["document"]).is_file() for q in registry["questions"])
