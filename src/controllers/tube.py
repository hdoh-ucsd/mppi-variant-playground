"""Williams RSS 2018 Algorithm 2 architecture with a declared local LQR ancillary tracker."""

from time import perf_counter

import numpy as np

from controllers.feedback import state_error, tracking_design
from controllers.mppi import MPPI


def select_actual(actual_cost, nominal_cost, threshold):
    if not np.isfinite([actual_cost, nominal_cost, threshold]).all() or threshold < 0:
        raise ValueError("Finite solution costs and nonnegative reset threshold required")
    return actual_cost <= nominal_cost + threshold


class TubeMPPI(MPPI):
    def reset(self, initial_state, seed):
        super().reset(initial_state, seed)
        self.nominal_state = np.array(initial_state, dtype=float)
        self.nominal_previous = np.zeros(2)

    def solution_cost(self, updated, diagnostics, previous):
        states = diagnostics["planned_trajectory"]
        controls = self.project_controls(updated)
        total = 0.0
        for t in range(self.horizon):
            total += float(self.cost.running(states[t], controls[t], previous))
            gap = self.cost.path_clearance(diagnostics["planned_substeps"][t])
            total += self.model.dt * self.cost.weights["collision"] * (gap <= 0)
            previous = controls[t]
        return total + float(self.cost.terminal(states[-1]))

    def plan(self, observed_state, *, standard_normals=None):
        started = perf_counter()
        if standard_normals is None:
            standard_normals = self.rng.standard_normal((self.samples, self.horizon, 2))
        prior_control = self.previous_control.copy()
        self.previous_control = self.nominal_previous.copy()
        _, nominal_update = super().plan(self.nominal_state, standard_normals=standard_normals)
        nominal_diag = self.get_diagnostics()
        nominal_snap = self.get_rollout_snapshot()
        nominal_cost = self.solution_cost(nominal_update, nominal_diag, self.nominal_previous)
        self.previous_control = prior_control
        _, actual_update = super().plan(observed_state, standard_normals=standard_normals)
        actual_diag = self.get_diagnostics()
        actual_snap = self.get_rollout_snapshot()
        actual_cost = self.solution_cost(actual_update, actual_diag, prior_control)
        threshold = self.config.get(
            "reset_threshold", self.model.dt * self.cost.weights["collision"]
        )
        reset = select_actual(actual_cost, nominal_cost, threshold)
        error = state_error(observed_state, self.nominal_state)
        if reset:
            nominal_origin = np.array(observed_state)
            updated = actual_update
        else:
            nominal_origin = self.nominal_state.copy()
            updated = nominal_update
            self.diagnostics = nominal_diag
            self._rollout_snapshot = nominal_snap
        controls = self.project_controls(updated)
        gains, _, _ = tracking_design(
            self.model, self.diagnostics["planned_trajectory"], controls, self.config
        )
        correction = gains[0] @ state_error(observed_state, nominal_origin)
        nominal_control = controls[0]
        command = self.model.clip_control(nominal_control + correction * self.model.control_scale)
        self.next_nominal = self.diagnostics["planned_trajectory"][1].copy()
        self.next_nominal_previous = nominal_control.copy()
        self.diagnostics.update(
            tracking_error=float(np.linalg.norm(error)),
            tracking_control_norm=float(np.linalg.norm(correction)),
            applied_tracking_control_norm=float(np.linalg.norm(command - nominal_control)),
            reset_event=bool(reset and np.linalg.norm(error) > 1e-12),
            selected_actual=bool(reset),
            nominal_solution_cost=nominal_cost,
            actual_solution_cost=actual_cost,
            reset_threshold=threshold,
            nominal_state=nominal_origin,
            nominal_before_reset=self.nominal_state.copy(),
            tracking_error_vector=state_error(observed_state, nominal_origin),
            tracking_correction=correction.copy(),
            planner_control=nominal_control.copy(),
            applied_correction=command - nominal_control,
            tracking_gains=gains,
            dynamics_evaluations=2 * (self.samples + 1) * self.horizon + 18 * self.horizon,
            rollout_evaluations=2 * self.samples,
            linearization_evaluations=18 * self.horizon,
            rhs_evaluations=(2 * (self.samples + 1) * self.horizon + 18 * self.horizon)
            * self.model.substeps
            * 2,
            planning_latency_s=perf_counter() - started,
        )
        if self._rollout_snapshot is not None:
            # Preserve both solved populations, including the one rejected by the reset rule.
            populations = (
                ("tube_nominal", list(nominal_snap.items())),
                ("tube_actual", list(actual_snap.items())),
            )
            for label, entries in populations:
                for key, value in entries:
                    if isinstance(value, np.ndarray):
                        self._rollout_snapshot[f"{label}_{key}"] = value
        return command, updated

    def command(self, observed_state):
        command = super().command(observed_state)
        self.nominal_state = self.next_nominal.copy()
        self.nominal_previous = self.next_nominal_previous.copy()
        return command
