"""Scientific claims must resolve to questions, configurations and executed evidence."""

import json
import re
from copy import deepcopy

import pytest

from analysis.research_registry import FIELDS, ROOT, read_registry, validate


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
        section = text.split(f"### {q['id']} —", 1)[1].split("\n### ", 1)[0].split("\n## ", 1)[0]
        images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", section)
        assert len(images) == (1 if q["status"] == "EXECUTED" else 0)
        for path in images:
            assert (ROOT / path).is_file()
    for path in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", text):
        if "://" not in path:
            assert (ROOT / path).exists(), path
