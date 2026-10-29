"""Execute a staged formulation comparison or rebuild its saved-data figure."""

import argparse

from analysis.variant_campaign import plot, run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study")
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw")
    parser.add_argument("--config", default="configs/variants/benchmark.yaml")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(args.output)
    else:
        if not args.study or not args.raw:
            parser.error("--study and --raw required for execution")
        run(args.study, args.output, args.raw, options_path=args.config)
