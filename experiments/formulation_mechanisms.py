"""Execute predeclared mechanism experiments or regenerate their saved-data figures."""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/variants/mechanisms")
    parser.add_argument("--raw", default="runs/variants/mechanisms")
    parser.add_argument("--config", default="configs/variants/mechanisms.yaml")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        from analysis.formulation_mechanism_plots import plot_all

        plot_all(args.output)
    else:
        from analysis.formulation_mechanisms import run

        run(args.output, args.raw, args.config)
