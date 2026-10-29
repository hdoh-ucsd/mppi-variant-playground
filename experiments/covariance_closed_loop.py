"""Closed-loop covariance parameter sweeps using the existing bicycle experiments."""

from analysis.cli import arguments
from analysis.closed_loop_study import run_closed_loop
from analysis.plots import generate

if __name__ == "__main__":
    args, config = arguments("configs/investigations/closed_loop_sweep.yaml", __doc__)
    if not args.plots_only:
        run_closed_loop(config, args.output)
    generate(args.output)
