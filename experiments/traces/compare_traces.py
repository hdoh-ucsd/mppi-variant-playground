"""Extract historical evidence, fill selected gaps, or rebuild six saved-data figures."""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/variants/traces")
    parser.add_argument("--raw", default="runs/variants/traces")
    parser.add_argument("--config", default="configs/variants/traces.yaml")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    if args.plots_only:
        from analysis.trace_plots import plot_all

        plot_all(args.output)
    else:
        from analysis.trace_comparison import run

        run(args.output, args.raw, live=not args.extract_only, config=args.config)
