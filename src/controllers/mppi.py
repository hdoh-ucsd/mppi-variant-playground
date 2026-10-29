"""One shared IT-MPPI implementation, four explicit covariance configurations."""

from copy import deepcopy
from time import perf_counter

import numpy as np

from controllers.base import Controller
from sampling.gaussian import (
    importance_weights,
    log_base_over_proposal,
    sample_gaussian,
    validate_covariance,
    weighted_update,
)
from scenarios.environment import swept_clearance


class MPPI(Controller):
    def __init__(self, dynamics, cost, config):
        self.model, self.cost, self.config = dynamics, cost, deepcopy(config)
        self.state_dimension = getattr(dynamics, "state_dimension", 4)
        self.horizon, self.samples = config["horizon"], config["samples"]
        self.temperature = config["temperature"]
        if (
            not isinstance(self.horizon, int)
            or not isinstance(self.samples, int)
            or min(self.horizon, self.samples) < 1
            or not np.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("Positive horizon, sample count and temperature required")
        self.covariance = validate_covariance(config["covariance"])
        self.base_covariance = validate_covariance(config["base_covariance"])
        self.snapshot_steps = frozenset(config.get("snapshot_steps", ()))
        if any(
            not isinstance(step, int) or isinstance(step, bool) or step < 0
            for step in self.snapshot_steps
        ):
            raise ValueError("Snapshot steps must be nonnegative integers")
        kind = config["kind"]
        if kind not in {"standard", "isotropic", "diagonal", "full"}:
            raise ValueError(f"Controller {kind!r} is not implemented")
        if kind in {"isotropic", "diagonal"} and not np.allclose(
            self.covariance, np.diag(np.diag(self.covariance))
        ):
            raise ValueError("Diagonal covariance required")
        if kind == "isotropic" and not np.allclose(
            self.covariance, np.eye(2) * self.covariance[0, 0]
        ):
            raise ValueError("Isotropic covariance required")
        self.reset(np.zeros(self.state_dimension), 42)

    def reset(self, initial_state, seed):
        if (
            np.shape(initial_state) != (self.state_dimension,)
            or not np.isfinite(initial_state).all()
        ):
            raise ValueError("Initial state must match the model state dimension")
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros((self.horizon, 2))
        self.previous_control = np.zeros(2)
        self.diagnostics = {}
        self.planning_step = 0
        self._rollout_snapshot = None

    def plan(self, observed_state, *, standard_normals=None):
        """One optimization decision without shifting the warm start or executing a step.

        Optional common random numbers are standard normals, before the covariance
        transformation. The ordinary command path retains its existing RNG sequence.
        """
        started = perf_counter()
        if (
            np.shape(observed_state) != (self.state_dimension,)
            or not np.isfinite(observed_state).all()
        ):
            raise ValueError("Observation must match the model state dimension")
        if standard_normals is None:
            eps = sample_gaussian(self.rng, (self.samples, self.horizon), self.covariance)
        else:
            z = np.asarray(standard_normals, dtype=float)
            if z.shape != (self.samples, self.horizon, 2) or not np.isfinite(z).all():
                raise ValueError("Expected finite standard normals [samples, horizon, 2]")
            eps = z @ np.linalg.cholesky(self.covariance).T
        latent = self.mean[None] + eps
        applied = self.model.clip_control(latent * self.model.control_scale)
        states = np.broadcast_to(observed_state, (self.samples, self.state_dimension)).copy()
        feasible = np.ones(self.samples, dtype=bool)
        costs = np.zeros(self.samples)
        previous = np.broadcast_to(self.previous_control, (self.samples, 2))
        capture = self.planning_step in self.snapshot_steps
        keep = self.samples if capture else min(self.config.get("saved_rollouts", 32), self.samples)
        saved = np.empty((keep, self.horizon + 1, self.state_dimension))
        saved[:, 0] = states[:keep]
        for t in range(self.horizon):
            control = applied[:, t]
            costs += self.cost.running(states, control, previous)
            next_states = self.model.step(states, control)
            # Swept collision penalty includes collisions between discrete states.
            if hasattr(self.cost, "transition_clearance"):
                gap = self.cost.transition_clearance(states, next_states)
            else:
                gap = swept_clearance(
                    states, next_states, self.cost.obstacles, self.cost.vehicle_radius
                )
            feasible &= gap > 0
            costs += self.cost.dt * self.cost.weights["collision"] * (gap <= 0)
            states = next_states
            saved[:, t + 1] = states[:keep]
            previous = control
        costs += self.cost.terminal(states)
        ratio = log_base_over_proposal(latent, self.mean, self.base_covariance, self.covariance)
        weights = importance_weights(costs, self.temperature, ratio)
        # Latent mean remains unconstrained; saturation is inside dynamics and at actuation.
        updated = weighted_update(self.mean, eps, weights)
        command = self.model.clip_control(updated[0] * self.model.control_scale)
        planned = np.empty((self.horizon + 1, self.state_dimension))
        planned[0] = observed_state
        for t in range(self.horizon):
            planned[t + 1] = self.model.step(planned[t], updated[t] * self.model.control_scale)
        positive = weights > 0
        ess = float(1 / np.sum(weights**2))
        self.diagnostics = {
            "ess": ess,
            "weight_entropy": float(-np.sum(weights[positive] * np.log(weights[positive]))),
            "max_weight": float(weights.max()),
            "degenerate": bool(ess < 2),
            "minimum_sample_cost": float(costs.min()),
            "feasible_rollout_fraction": float(feasible.mean()),
            "weighted_sample_cost": float(weights @ costs),
            "trajectory_spread": float(np.var(states[:, :2], axis=0).sum()),
            "perturbation_mean": eps.mean(axis=(0, 1)),
            "perturbation_covariance": np.cov(eps.reshape(-1, 2), rowvar=False),
            "saturation_fraction": float(np.mean(np.abs(latent) > 1)),
            "dynamics_evaluations": (self.samples + 1) * self.horizon,
            "rollout_evaluations": self.samples,
            "saved_rollouts": saved,
            "planned_trajectory": planned,
            "weights": weights,
            "extra_optimization_work": 0,
            "planning_latency_s": perf_counter() - started,
        }
        self._rollout_snapshot = (
            {
                "step": self.planning_step,
                "sampled_states": saved,
                "sampled_controls": applied,
                "costs": costs,
                "weights": weights,
                "log_base_over_proposal": ratio,
                "best_rollout_index": int(np.argmin(costs)),
                "highest_weight_index": int(np.argmax(weights)),
                "current_state": np.asarray(observed_state).copy(),
                "nominal_states": planned,
                "nominal_role": "updated_mean_plan",
                "latent_controls": latent,
                "updated_mean": updated,
                "proposal_mean": self.mean.copy(),
            }
            if capture
            else None
        )
        return command, updated

    def command(self, observed_state):
        command, updated = self.plan(observed_state)
        self.planning_step += 1
        self.mean[:-1] = updated[1:]
        self.mean[-1] = 0  # common zero tail warm start
        self.previous_control = command.copy()
        return command

    def get_diagnostics(self):
        return deepcopy(self.diagnostics)

    def get_rollout_snapshot(self):
        """Optional full-population observation; recording never draws additional randomness."""
        return deepcopy(self._rollout_snapshot)
