"""Wang et al. 2104.00241v1: (6), (8)–(11), (14), (28), fixed covariance Table II."""

import numpy as np

from controllers.mppi import MPPI
from sampling.gaussian import importance_weights


def tsallis_weights(costs, scale, r, log_ratio=None):
    """Normalized exp_r(-J/scale) p/q, with scale=tilde-lambda, NOT objective lambda.

    No cost offset is silently subtracted: unlike exp, exp_r is not shift invariant.
    The r=1 limit holds scale fixed; holding the support threshold fixed differs.
    """
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or not len(costs) or not np.isfinite(costs).all():
        raise ValueError("Finite nonempty costs required")
    if (costs < 0).any() or not np.isfinite([r, scale]).all() or min(r, scale) <= 0:
        raise ValueError("Nonnegative costs and positive r/scale required")
    ratio = np.zeros_like(costs) if log_ratio is None else np.asarray(log_ratio, dtype=float)
    if ratio.shape != costs.shape or not np.isfinite(ratio).all():
        raise ValueError("Finite matching log ratios required")
    if r == 1:
        return importance_weights(costs, scale, ratio)
    argument = -(r - 1) * costs / scale
    supported = argument > -1
    logits = np.full_like(costs, -np.inf)
    logits[supported] = np.log1p(argument[supported]) / (r - 1) + ratio[supported]
    if not np.isfinite(logits).any():
        raise FloatingPointError("Tsallis support contains no samples; no uniform fallback")
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


class TsallisMPPI(MPPI):
    """Single-pass fixed-covariance Gaussian mean projection, paper Table II.

    Optional elite parameterization uses explicit batch [0,1] cost normalization
    (paper Algorithm 1) and gamma quantile. This is an adaptive objective scale,
    not a fixed-target estimation experiment. No covariance adaptation is claimed.
    """

    def __init__(self, dynamics, cost, config):
        self.r = float(config.get("tsallis_r", 1.5))
        self.elite = config.get("elite_fraction", 0.8)
        if self.r <= 0 or not np.isfinite(self.r):
            raise ValueError("Positive finite Tsallis r required")
        if self.elite is not None and (self.r <= 1 or not 0 < self.elite <= 1):
            raise ValueError("Elite cutoff mode requires r>1 and fraction in (0,1]")
        super().__init__(dynamics, cost, config)

    def compute_weights(self, costs, log_ratio):
        scale = self.temperature
        normalized = costs
        offset, span = 0.0, 1.0
        if self.elite is not None:
            offset, span = float(costs.min()), float(np.ptp(costs))
            normalized = (costs - offset) / span if span > 0 else np.zeros_like(costs)
            gamma = max(float(np.quantile(normalized, self.elite)), np.finfo(float).eps)
            scale = (self.r - 1) * gamma
        weights = tsallis_weights(normalized, scale, self.r, log_ratio)
        self.weight_details = dict(
            tsallis_r=self.r,
            transform_scale=float(scale),
            cost_offset=offset,
            cost_span=span,
            support_fraction=float(np.mean(weights > 0)),
        )
        return weights

    def plan(self, observed_state, *, standard_normals=None):
        result = super().plan(observed_state, standard_normals=standard_normals)
        self.diagnostics.update(self.weight_details)
        return result
