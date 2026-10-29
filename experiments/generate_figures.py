import argparse

from mppi_variant_playground.visualization.figures import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/covariance_comparison")
    parser.add_argument("--animate", action="store_true")
    args = parser.parse_args()
    generate(args.input, args.animate)


if __name__ == "__main__":
    main()
