"""Render the evidence map after all independently executable levels complete."""

import argparse

from analysis.mechanism_summary import collect, render

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Parent containing the five *_v1 studies")
    parser.add_argument("--output", required=True, help="New evidence-summary directory")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if not args.plots_only:
        collect(args.input, args.output)
    render(args.output)
