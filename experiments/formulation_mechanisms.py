"""Execute predeclared mechanism experiments or regenerate their saved-data figures."""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/variants/mechanisms")
    parser.add_argument("--raw", default="runs/variants/mechanisms")
    parser.add_argument("--config", default="configs/variants/mechanisms.yaml")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plot-only", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        from analysis.formulation_mechanism_plots import plot_all

        plot_all(args.output)
    elif args.audit_only:
        from analysis.formulation_mechanisms import audit

        print(audit(args.output))
    else:
        from analysis.formulation_mechanisms import run

        run(args.output, args.raw, args.config)
