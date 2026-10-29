"""Frozen-target Gaussian covariance, cost-rotation and sample-budget study."""

from analysis.cli import arguments
from analysis.gaussian_study import run_gaussian
from analysis.plots import generate

if __name__ == "__main__":
    args, config = arguments("configs/investigations/gaussian_target.yaml", __doc__)
    if not args.plots_only:
        run_gaussian(config, args.output)
    generate(args.output)
