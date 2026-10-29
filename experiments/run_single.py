"""Run a single fully logged CPU episode from repository root."""

import argparse

from evaluation.runner import run_episode
from utils.config import load_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller", default="standard", choices=["standard", "isotropic", "diagonal", "full"]
    )
    parser.add_argument(
        "--scenario",
        default="open_space",
        choices=[
            "open_space",
            "narrow_passage",
            "multiple_obstacles",
            "disturbance",
            "model_mismatch",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", default="results/single")
    args = parser.parse_args()
    config = load_experiment(args.controller, args.scenario)
    if args.samples is not None:
        config["controller"]["samples"] = args.samples
    if args.steps is not None:
        config["scenario"]["max_steps"] = args.steps
    result = run_episode(config, args.seed, args.output)
    print(result["metrics"])


if __name__ == "__main__":
    main()
