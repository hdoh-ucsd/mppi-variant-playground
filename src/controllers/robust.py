"""Gandhi 2102.09027v1 Algorithms 1–3, using local LQR rather than contraction control."""

from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from controllers.feedback import state_error, tracking_design
from controllers.mppi import MPPI
from sampling.gaussian import importance_weights, weighted_update


def augmented_costs(
    nominal_cost, actual_cost, mean, noise, feedback, covariance, temperature, alpha
):
    """Algorithm 1 at beta=0; controls and feedback are in normalized latent units."""
    inv = np.linalg.inv(covariance)

    def energy(x, y):
        return np.einsum("kti,ij,ktj->k", x, inv, y)

    broadcast = np.broadcast_to(mean, noise.shape)
    actual_mean = broadcast + feedback
    actual = actual_cost + 0.5 * temperature * energy(actual_mean, actual_mean + 2 * noise)
    tracked = actual_cost + 0.5 * temperature * energy(feedback, feedback)
    mixed = 0.5 * nominal_cost + 0.5 * np.maximum(np.minimum(tracked, alpha), nominal_cost)
    nominal = mixed + 0.5 * temperature * energy(broadcast, broadcast + 2 * noise)
    return nominal, actual


def candidate_index(candidates, actual, free_energy, threshold):
    feasible = np.flatnonzero(np.asarray(free_energy) <= threshold)
    if not len(feasible):
        return 0, True
    distance = np.linalg.norm(state_error(np.asarray(candidates)[feasible], actual), axis=1)
    return int(feasible[np.argmin(distance)]), False


class RobustMPPI(MPPI):
    def __init__(self, dynamics, cost, config):
        if not np.array_equal(config["covariance"], config["base_covariance"]):
            raise ValueError("Implemented AIS requires equal base/proposal covariance")
        probes = config.get("candidate_samples", 8)
        alpha = config.get("free_energy_threshold", 5000.0)
        if not isinstance(probes, int) or probes < 1 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Positive candidate count and finite free-energy threshold required")
        self.phase = "plain"
        super().__init__(dynamics, cost, config)

    def reset(self, initial_state, seed):
        super().reset(initial_state, seed)
        self.nominal_state = np.asarray(initial_state).copy()
        self.previous_nominal = self.nominal_state.copy()
        self.previous_plan = self.mean.copy()
        self.phase = "plain"

    def compute_weights(self, costs, log_ratio):
        self.population_costs = costs.copy()
        self.population_ratio = log_ratio.copy()
        if self.phase != "actual":
            return super().compute_weights(costs, log_ratio)
        self.nominal_augmented, self.actual_augmented = augmented_costs(
            self.nominal_costs,
            costs,
            self.mean,
            self.noise,
            self.feedback_samples,
            self.covariance,
            self.temperature,
            self.config.get("free_energy_threshold", 5000.0),
        )
        return importance_weights(self.actual_augmented, self.temperature)

    def rollout_control(self, t, states, control):
        if self.phase != "actual":
            return control
        correction = state_error(states, self.nominal_samples[:, t]) @ self.gains[t].T
        self.feedback_samples[:, t] = correction
        latent = self.mean[t] + self.noise[:, t] + correction
        return self.model.clip_control(latent * self.model.control_scale)

    def predict(self, state, mean):
        # Same physical model; these extra transitions are charged to the planner.
        states = [np.asarray(state).copy()]
        paths = []
        for u in self.project_controls(mean):
            nxt, path = self.model.step(states[-1], u, return_path=True)
            states.append(nxt)
            paths.append(path)
        return np.array(states), np.array(paths)

    def select_nominal(self, observed):
        propagated = self.nominal_state
        previous = self.previous_nominal

        def midpoint(x, y):
            return x + 0.5 * state_error(y, x)

        candidates = np.array(
            [
                previous,
                midpoint(previous, propagated),
                propagated,
                midpoint(propagated, observed),
                observed,
            ]
        )
        warm = self.mean.copy()
        means = [self.previous_plan] + [warm] * 4
        original_samples = self.samples
        self.samples = self.config.get("candidate_samples", 8)
        normals = self.rng.standard_normal((self.samples, self.horizon, 2))
        energies = []
        for state, mean in zip(candidates, means, strict=True):
            self.mean = mean.copy()
            MPPI.plan(self, state, standard_normals=normals)
            energies.append(
                float(
                    -self.temperature
                    * (
                        logsumexp(-self.population_costs / self.temperature + self.population_ratio)
                        - np.log(self.samples)
                    )
                )
            )
        self.samples = original_samples
        index, infeasible = candidate_index(
            candidates, observed, energies, self.config.get("free_energy_threshold", 5000.0)
        )
        self.nominal_candidates = candidates.copy()
        self.mean = means[index].copy()
        return candidates[index], index, infeasible, energies

    def plan(self, observed_state, *, standard_normals=None):
        if standard_normals is not None:
            raise ValueError(
                "Robust-MPPI has separate candidate and AIS random streams; reset seed"
            )
        started = perf_counter()
        count = getattr(self.model, "transitions", 0)
        self.phase = "plain"
        nominal, index, infeasible, energies = self.select_nominal(np.asarray(observed_state))
        reference, _ = self.predict(nominal, self.mean)
        controls = self.project_controls(self.mean)
        self.gains, _, _ = tracking_design(self.model, reference, controls, self.config)
        normals = self.rng.standard_normal((self.samples, self.horizon, 2))
        self.noise = normals @ np.linalg.cholesky(self.covariance).T
        self.feedback_samples = np.zeros_like(self.noise)
        self.config["saved_rollouts"] = self.samples
        MPPI.plan(self, nominal, standard_normals=normals)
        self.nominal_costs = self.population_costs.copy()
        self.nominal_samples = self.diagnostics["saved_rollouts"].copy()
        self.phase = "actual"
        _, real_update = MPPI.plan(self, observed_state, standard_normals=normals)
        self.phase = "plain"
        real_weights = self.diagnostics["weights"].copy()
        nominal_weights = importance_weights(self.nominal_augmented, self.temperature)
        updated = weighted_update(self.mean, self.noise, nominal_weights)
        correction = self.gains[0] @ state_error(observed_state, nominal)
        command = self.model.clip_control((real_update[0] + correction) * self.model.control_scale)
        planned, paths = self.predict(nominal, updated)
        self.next_nominal = planned[1].copy()
        self.selected_nominal = nominal.copy()
        self.updated_plan = updated.copy()
        transitions = getattr(self.model, "transitions", 0) - count
        self.diagnostics.update(
            planned_trajectory=planned,
            planned_substeps=paths,
            tracking_error=float(np.linalg.norm(state_error(observed_state, nominal))),
            tracking_control_norm=float(np.linalg.norm(correction)),
            sampled_feedback_rms=float(np.sqrt(np.mean(self.feedback_samples**2))),
            nominal_ess=float(1 / np.sum(nominal_weights**2)),
            nominal_selection=index,
            nominal_candidate_free_energies=energies,
            nominal_selection_infeasible=infeasible,
            nominal_state=nominal,
            nominal_candidates=self.nominal_candidates.copy(),
            tracking_gains=self.gains,
            tracking_error_vector=state_error(observed_state, nominal),
            tracking_correction=correction.copy(),
            planner_control=self.model.clip_control(real_update[0] * self.model.control_scale),
            applied_correction=command
            - self.model.clip_control(real_update[0] * self.model.control_scale),
            rollout_evaluations=2 * self.samples + 5 * self.config.get("candidate_samples", 8),
            linearization_evaluations=18 * self.horizon,
            dynamics_evaluations=transitions,
            rhs_evaluations=transitions * 2 * self.model.substeps,
            planning_latency_s=perf_counter() - started,
        )
        if self._rollout_snapshot is not None:
            self._rollout_snapshot.update(
                nominal_candidates=self.nominal_candidates.copy(),
                candidate_free_energies=np.array(energies),
                real_updated_mean=real_update.copy(),
                reference_nominal_states=reference,
                nominal_weights=nominal_weights,
                nominal_costs=self.nominal_costs,
                nominal_samples=self.nominal_samples,
                feedback_gains=self.gains,
                real_weights=real_weights,
                nominal_augmented_costs=self.nominal_augmented,
                real_augmented_costs=self.actual_augmented,
                feedback=self.feedback_samples,
                updated_mean=updated,
                nominal_states=planned,
                nominal_role="mixed-cost nominal update; actual command uses real weights",
            )
        return command, updated

    def command(self, observed_state):
        result = super().command(observed_state)
        self.previous_nominal = self.selected_nominal.copy()
        self.nominal_state = self.next_nominal.copy()
        self.previous_plan = self.updated_plan.copy()
        return result
