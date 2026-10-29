"""Yin 2109.12147v3 (10)–(21): auxiliary-noise feedback and constrained covariance fit."""

from time import perf_counter

import numpy as np

from controllers.feedback import linearize, state_error
from controllers.mppi import MPPI
from sampling.gaussian import importance_weights


class CovarianceInfeasible(RuntimeError):
    """No numerically verified covariance-steering solution; never silently use plain MPPI."""


def covariance_steering(a, b, noise_covariance, terminal_bound, q_diag, r_diag, *, tolerance=1e-5):
    """Solve paper (19)–(21) over block-diagonal K_t acting on open-loop noise y_t.

    First solve the unconstrained quadratic exactly (tiny declared ridge). If its
    terminal covariance is feasible it is also constrained-optimal. Otherwise use
    a Schur-complement SDP. All candidates are independently checked after solving.
    """
    started = perf_counter()
    horizon, n, m = b.shape
    if np.shape(a) != (horizon, n, n):
        raise ValueError("Inconsistent LTV dimensions")
    sigma = np.asarray(noise_covariance, dtype=float)
    bound = np.asarray(terminal_bound, dtype=float)
    q = np.asarray(q_diag, dtype=float)
    r = np.asarray(r_diag, dtype=float)
    for matrix in [sigma, bound]:
        if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, atol=1e-12):
            raise ValueError("Finite symmetric covariances required")
        np.linalg.cholesky(matrix)
    if (
        q.shape != ((horizon + 1) * n,)
        or r.shape != (horizon * m,)
        or (q < 0).any()
        or (r <= 0).any()
    ):
        raise ValueError("Nonnegative state and positive control weights required")
    big_b = np.zeros(((horizon + 1) * n, horizon * m))
    for t in range(horizon):
        big_b[(t + 1) * n : (t + 2) * n] = a[t] @ big_b[t * n : (t + 1) * n]
        big_b[(t + 1) * n : (t + 2) * n, t * m : (t + 1) * m] += b[t]
    g = big_b @ np.kron(np.eye(horizon), np.linalg.cholesky(sigma))
    dimension = horizon * m * n
    state_basis = np.empty((*g.shape, dimension))
    control_basis = np.zeros((horizon * m, horizon * m, dimension))
    column = 0
    for t in range(horizon):
        for i in range(m):
            for j in range(n):
                row = t * m + i
                y = g[t * n + j]
                state_basis[:, :, column] = np.outer(big_b[:, row], y)
                control_basis[row, :, column] = y
                column += 1
    design = np.concatenate(
        [
            (np.sqrt(q)[:, None, None] * state_basis).reshape(-1, dimension),
            (np.sqrt(r)[:, None, None] * control_basis).reshape(-1, dimension),
        ]
    )
    offset = np.r_[(np.sqrt(q)[:, None] * g).ravel(), np.zeros((horizon * m) ** 2)]
    ridge = 1e-10
    hessian = design.T @ design + ridge * np.eye(dimension)
    gradient = design.T @ offset
    theta = np.linalg.solve(hessian, -gradient)
    terminal_basis = state_basis[-n:].reshape(n * horizon * m, dimension)

    def factor(vector):
        return g + np.einsum("ijp,p->ij", state_basis, vector)

    def violation(vector):
        terminal = factor(vector)[-n:]
        return float(np.linalg.eigvalsh(terminal @ terminal.T - bound).max())

    solver = "unconstrained_quadratic_feasible"
    if violation(theta) > tolerance:
        import cvxpy as cp

        variable = cp.Variable(dimension)
        terminal = cp.reshape(terminal_basis @ variable, (n, horizon * m), order="C") + g[-n:]
        constraint = cp.bmat([[bound, terminal], [terminal.T, np.eye(horizon * m)]]) >> 0
        problem = cp.Problem(
            cp.Minimize(cp.quad_form(variable, cp.psd_wrap(hessian)) + 2 * gradient @ variable),
            [constraint],
        )
        problem.solve(
            solver="CLARABEL", max_iter=200, tol_gap_abs=1e-7, tol_gap_rel=1e-7, tol_feas=1e-7
        )
        solver = str(problem.status)
        if problem.status not in {"optimal", "optimal_inaccurate"} or variable.value is None:
            raise CovarianceInfeasible(f"Covariance steering status: {problem.status}")
        theta = variable.value
    residual = violation(theta)
    if not np.isfinite(theta).all() or residual > tolerance:
        raise CovarianceInfeasible(f"Terminal PSD residual {residual} exceeds {tolerance}")
    factors = factor(theta).reshape(horizon + 1, n, horizon * m)
    covariances = np.einsum("tik,tjk->tij", factors, factors)
    if np.linalg.eigvalsh(covariances).min() < -tolerance:
        raise FloatingPointError("Predicted state covariance is not PSD")
    return (
        theta.reshape(horizon, m, n),
        covariances,
        dict(
            solver=solver,
            solver_seconds=perf_counter() - started,
            covariance_constraint_residual=residual,
            solver_infeasible=False,
            covariance_objective=float(np.sum((design @ theta + offset) ** 2)),
            ridge=ridge,
        ),
    )


class CovarianceMPPI(MPPI):
    def __init__(self, dynamics, cost, config):
        self.uncontrolled = float(config.get("uncontrolled_fraction", 0.2))
        self.noise_ratio = float(config.get("noise_ratio", 100.0))
        if (
            not 0 <= self.uncontrolled < 1
            or not np.isfinite(self.noise_ratio)
            or self.noise_ratio <= 0
        ):
            raise ValueError("Valid uncontrolled fraction and noise ratio required")
        super().__init__(dynamics, cost, config)
        if self.samples < 3:
            raise ValueError("At least three samples required for empirical covariance")

    def rollout_control(self, t, states, control):
        v = self.mean[t] + self.auxiliary @ self.gains[t].T
        v[self.controlled_count :] = 0.0
        self.shaped_latent[:, t] = v + self.noise[:, t]
        self.control_means[:, t] = v
        self.auxiliary = self.auxiliary @ self.a[t].T + self.noise[:, t] @ self.b[t].T
        return self.model.clip_control(self.shaped_latent[:, t] * self.model.control_scale)

    def compute_weights(self, costs, log_ratio):
        r = np.diag(self.config.get("cc_control_r", [0.01, 0.01]))
        v = self.control_means
        e = self.noise
        extra = np.einsum("kti,ij,ktj->k", v, r, e)
        extra += 0.5 * np.einsum("kti,ij,ktj->k", v, r, v)
        extra += 0.5 * (1 - 1 / self.noise_ratio) * np.einsum("kti,ij,ktj->k", e, r, e)
        self.augmented_costs = costs + extra
        return importance_weights(self.augmented_costs, self.temperature)

    def predict(self, state, mean):
        states = [np.asarray(state).copy()]
        paths = []
        for u in self.project_controls(mean):
            nxt, path = self.model.step(states[-1], u, return_path=True)
            states.append(nxt)
            paths.append(path)
        return np.array(states), np.array(paths)

    def plan(self, observed_state, *, standard_normals=None):
        started = perf_counter()
        count = getattr(self.model, "transitions", 0)
        reference, _ = self.predict(observed_state, self.mean)
        self.a, self.b = linearize(self.model, reference, self.project_controls(self.mean))
        bound = np.diag(
            self.config.get("terminal_covariance_bound", [1.0, 1.0, 0.25, 0.04, 1.0, 0.5, 0.5])
        )
        q = np.array(self.config.get("covariance_q", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
        q_all = np.r_[np.tile(q, self.horizon), 5 * q]
        r_all = np.tile(self.config.get("covariance_r", [1.0, 1.0]), self.horizon)
        self.gains, self.predicted_covariances, solver = covariance_steering(
            self.a,
            self.b,
            self.covariance,
            bound,
            q_all,
            r_all,
            tolerance=self.config.get("covariance_tolerance", 1e-5),
        )
        if standard_normals is None:
            standard_normals = self.rng.standard_normal((self.samples, self.horizon, 2))
        self.noise = standard_normals @ np.linalg.cholesky(self.covariance).T
        self.auxiliary = np.zeros((self.samples, self.state_dimension))
        self.shaped_latent = np.zeros_like(self.noise)
        self.control_means = np.zeros_like(self.noise)
        self.controlled_count = max(2, int((1 - self.uncontrolled) * self.samples))
        self.config["saved_rollouts"] = self.samples
        super().plan(observed_state, standard_normals=standard_normals)
        weights = self.diagnostics["weights"]
        updated = np.einsum("k,kti->ti", weights, self.shaped_latent)
        endpoints = self.diagnostics["saved_rollouts"][: self.controlled_count, -1]
        empirical = np.cov(state_error(endpoints, reference[-1]), rowvar=False)
        residual = float(np.linalg.eigvalsh(empirical - bound).max())
        planned, paths = self.predict(observed_state, updated)
        transitions = getattr(self.model, "transitions", 0) - count
        self.diagnostics.update(
            solver,
            planned_trajectory=planned,
            planned_substeps=paths,
            dynamics_evaluations=transitions,
            rhs_evaluations=transitions * 2 * self.model.substeps,
            linearization_evaluations=18 * self.horizon,
            predicted_terminal_covariance=self.predicted_covariances[-1],
            empirical_terminal_covariance=empirical,
            terminal_covariance_bound=bound,
            covariance_prediction_error=float(
                np.linalg.norm(empirical - self.predicted_covariances[-1])
            ),
            covariance_violation=bool(residual > self.config.get("covariance_tolerance", 1e-5)),
            empirical_covariance_residual=residual,
            controlled_samples=self.controlled_count,
            planning_latency_s=perf_counter() - started,
        )
        if self._rollout_snapshot is not None:
            self._rollout_snapshot.update(
                latent_controls=self.shaped_latent.copy(),
                updated_mean=updated,
                augmented_costs=self.augmented_costs,
                control_means=self.control_means,
                sampled_noise=self.noise,
                linearized_a=self.a,
                linearized_b=self.b,
                predicted_covariances=self.predicted_covariances,
                empirical_terminal_covariance=empirical,
                feedback_gains=self.gains,
                log_base_over_proposal=np.zeros(self.samples),
                nominal_states=planned,
                weight_law="Yin Eq3 augmented cost; shaped control mean fit",
            )
        return self.project_controls(updated)[0], updated

    def command(self, observed_state):
        command, updated = self.plan(observed_state)
        self.planning_step += 1
        self.mean[:-1] = updated[1:]
        self.mean[-1] = updated[-1]  # Algorithm1 repeats the last optimized input.
        self.previous_control = command.copy()
        return command
