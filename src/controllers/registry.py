"""Explicit formulation names; covariance choices remain baseline configurations."""

from controllers.accounting import CountedModel
from controllers.dial import DIAL
from controllers.mppi import MPPI
from controllers.robust import RobustMPPI
from controllers.tsallis import TsallisMPPI
from controllers.tube import TubeMPPI

CONTROLLERS = {
    "it_mppi": MPPI,
    "tsallis_vi": TsallisMPPI,
    "tube_mppi": TubeMPPI,
    "robust_mppi": RobustMPPI,
    "dial_mpc": DIAL,
    "dial_no_anneal": DIAL,
}


def create_controller(dynamics, cost, config):
    name = config.get("algorithm", "it_mppi")
    if name not in CONTROLLERS:
        raise ValueError(f"Unknown or unimplemented formulation: {name}")
    if "algorithm" in config:
        dynamics = CountedModel(dynamics)
    return CONTROLLERS[name](dynamics, cost, config)
