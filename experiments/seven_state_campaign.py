"""Execute question-based campaign stages; use immutable fresh output directories."""

import argparse

from analysis.dynamic_campaign import configuration, run_closed
from analysis.dynamic_frozen import run_frozen
from analysis.gaussian_study import run_gaussian
from analysis.plots import generate as analytic_figures
from utils.config import read_yaml

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        required=True,
        choices=[
            "q01",
            "q02",
            "q03",
            "q04",
            "q05",
            "budget_q04",
            "budget_q05",
            "pilot_q04",
            "pilot_q05",
            "secondary_q05",
            "rate_q05",
        ],
    )
    p.add_argument("--output", required=True)
    p.add_argument("--parent", help="Completed primary run for a budget replication")
    p.add_argument(
        "--seeds", type=int, help="Predeclared S-turn expanded seed count for budget pairing"
    )
    args = p.parse_args()
    config = configuration()
    if args.stage == "rate_q05":
        config.update(read_yaml("configs/seven_state/rate.yaml"))
    if args.stage == "secondary_q05":
        config.update(read_yaml("configs/seven_state/secondary.yaml"))
    if args.stage in ["q01", "q02"]:
        cfg = read_yaml("configs/investigations/gaussian_target.yaml")
        cfg["evaluation_seeds"] = list(range(34000, 34064))
        cfg["replication_of"] = "gaussian_v1" if args.stage == "q01" else "temperature_v1"
        suites = (
            ("geometry", "cost_rotation", "budget")
            if args.stage == "q01"
            else ("temperature", "dimension")
        )
        run_gaussian(cfg, args.output, suites)
        analytic_figures(args.output)
    elif args.stage == "q03":
        run_frozen(config, args.output)
    else:
        pilot = args.stage.startswith("pilot_")
        stage = args.stage.replace("pilot_", "")
        seeds = (
            None
            if args.seeds is None
            else list(
                range(config["s_turn_seeds"]["start"], config["s_turn_seeds"]["start"] + args.seeds)
            )
        )
        run_closed(
            config,
            args.output,
            stage=stage,
            pilot=pilot,
            additional_seeds=seeds,
            parent_path=args.parent,
        )
