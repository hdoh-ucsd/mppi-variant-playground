"""Latent versus applied Gaussian control exploration under actuator clipping."""

from analysis.cli import arguments
from analysis.plots import generate
from analysis.saturation import run_saturation

if __name__ == "__main__":
    args, config = arguments("configs/investigations/saturation.yaml", __doc__)
    if not args.plots_only:
        run_saturation(config, args.output)
    generate(args.output)
