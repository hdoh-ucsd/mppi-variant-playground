"""Validate research provenance and optionally regenerate the question-indexed documentation."""

import argparse
import json

from analysis.research_registry import read_registry, render, validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    registry = read_registry()
    if args.render:
        render(registry)
    print(json.dumps(validate(registry), indent=2))
