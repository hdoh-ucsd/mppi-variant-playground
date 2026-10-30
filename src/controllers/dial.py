"""Xue et al. 2409.15610v1 (1), (7): cost-only refinement with dual annealing."""

from time import perf_counter

import numpy as np

from controllers.mppi import MPPI
from sampling.gaussian import importance_weights


def noise_scales(horizon, iterations, iteration, beta_trajectory, beta_horizon, anneal=True):
    """Standard deviations; paper (7) specifies covariance, so exponent is halved.

    Paper h=0..H is represented by T=H+1 controls. i decreases N..1.
    A single-control horizon has no horizon decay.
    """
    if min(horizon, iterations, beta_trajectory, beta_horizon) <= 0:
        raise ValueError("Positive horizon, iterations and annealing temperatures required")
    if not 0 <= iteration < iterations:
        raise ValueError("Iteration outside annealing schedule")
    if not anneal:
        return np.ones(horizon)
    h_decay = (horizon - 1 - np.arange(horizon)) / max(horizon - 1, 1)
    return np.exp(-0.5 * (iteration / (beta_trajectory * iterations) + h_decay / beta_horizon))


class DIAL(MPPI):
    def __init__(self, dynamics, cost, config):
        self.iterations = config.get("iterations", 4)
        self.beta_trajectory = float(config.get("beta_trajectory", 1.0))
        self.beta_horizon = float(config.get("beta_horizon", 1.0))
        self.anneal = config.get("anneal", True)
        if not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("Positive integer iterations required")
        if not np.isfinite([self.beta_trajectory, self.beta_horizon]).all():
            raise ValueError("Finite annealing temperatures required")
        noise_scales(config["horizon"], self.iterations, 0, self.beta_trajectory, self.beta_horizon)
        super().__init__(dynamics, cost, config)

    def compute_weights(self, costs, log_ratio):
        return importance_weights(costs, self.temperature)

    def plan(self, observed_state, *, standard_normals=None):
        if standard_normals is not None:
            raise ValueError(
                "DIAL needs independent normals for every inner pass; seed reset instead"
            )
        started = perf_counter()
        original = self.mean.copy()
        records = []
        populations = []
        try:
            for iteration in range(self.iterations):
                scales = noise_scales(
                    self.horizon,
                    self.iterations,
                    iteration,
                    self.beta_trajectory,
                    self.beta_horizon,
                    self.anneal,
                )
                normals = (
                    self.rng.standard_normal((self.samples, self.horizon, 2))
                    * scales[None, :, None]
                )
                control, updated = super().plan(observed_state, standard_normals=normals)
                records.append(
                    dict(
                        ess=self.diagnostics["ess"],
                        dynamics_evaluations=self.diagnostics["dynamics_evaluations"],
                        rollout_evaluations=self.samples,
                        variance_scales=(scales**2).tolist(),
                    )
                )
                if self._rollout_snapshot is not None:
                    populations.append(self.get_rollout_snapshot())
                    populations[-1]["planned_substeps"] = self.diagnostics[
                        "planned_substeps"
                    ].copy()
                self.mean = updated
        finally:
            self.mean = original
        self.diagnostics.update(
            dynamics_evaluations=sum(r["dynamics_evaluations"] for r in records),
            rhs_evaluations=sum(r["dynamics_evaluations"] for r in records)
            * self.model.substeps
            * 2,
            rollout_evaluations=self.samples * self.iterations,
            extra_optimization_work=self.iterations - 1,
            inner_iterations=self.iterations,
            inner_passes=records,
            planning_latency_s=perf_counter() - started,
        )
        if self._rollout_snapshot is not None:
            for key in (
                "sampled_states",
                "sampled_controls",
                "latent_controls",
                "costs",
                "weights",
                "proposal_mean",
                "updated_mean",
                "nominal_states",
                "planned_substeps",
                "feasible",
                "swept_contacts",
                "first_command",
            ):
                self._rollout_snapshot["inner_" + key] = np.stack([p[key] for p in populations])
            self._rollout_snapshot["inner_variance_scales"] = np.array(
                [record["variance_scales"] for record in records]
            )
            self._rollout_snapshot["log_base_over_proposal"] = np.zeros(self.samples)
            self._rollout_snapshot["weight_law"] = "cost_only; final inner pass"
        return control, updated
