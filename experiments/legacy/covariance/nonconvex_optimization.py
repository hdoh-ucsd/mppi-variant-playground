"""Iterative weighted sampling on curved and multimodal landscapes."""

from analysis.cli import arguments
from analysis.nonconvex_study import run_nonconvex
from analysis.plots import generate

if __name__ == "__main__":
    args, config = arguments("configs/investigations/nonconvex.yaml", __doc__)
    if not args.plots_only:
        run_nonconvex(config, args.output)
    generate(args.output)
