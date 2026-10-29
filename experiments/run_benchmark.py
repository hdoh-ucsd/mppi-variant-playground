import argparse

from mppi_variant_playground.evaluation.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmarks/covariance_comparison.yaml")
    parser.add_argument("--output", default="results/covariance_comparison")
    parser.add_argument("--mode", choices=["dynamics", "wall_clock", "native"])
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    run_benchmark(args.config, args.output, args.mode, args.seeds)


if __name__ == "__main__":
    main()
