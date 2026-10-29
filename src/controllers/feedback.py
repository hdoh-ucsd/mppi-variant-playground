"""Local linearization and finite-horizon tracking shared by feedback formulations."""

import numpy as np

from dynamics.bicycle import wrap_angle


def state_error(actual, nominal):
    error = np.asarray(actual) - np.asarray(nominal)
    error = error.copy()
    error[..., 2] = wrap_angle(error[..., 2])
    return error


def linearize(model, states, controls, step=1e-5):
    """Central differences of the existing discrete plant in normalized control units.

    Exactly 2*(n+m)*T model transitions. At clipping kinks this is a local
    symmetric derivative, not a smooth global model or a robustness certificate.
    """
    n, m = states.shape[-1], controls.shape[-1]
    a, b = [], []
    for x, u in zip(states[:-1], controls, strict=True):
        dx = np.eye(n) * step
        du = np.eye(m) * step
        xp = np.repeat(x[None], 2 * (n + m), axis=0)
        up = np.repeat((u / model.control_scale)[None], 2 * (n + m), axis=0)
        xp[:n] += dx
        xp[n : 2 * n] -= dx
        up[2 * n : 2 * n + m] += du
        up[2 * n + m :] -= du
        nxt = model.step(xp, up * model.control_scale)
        ax = state_error(nxt[:n], nxt[n : 2 * n]).T / (2 * step)
        bu = state_error(nxt[2 * n : 2 * n + m], nxt[2 * n + m :]).T / (2 * step)
        a.append(ax)
        b.append(bu)
    return np.array(a), np.array(b)


def lqr_gains(a, b, q, r, q_terminal):
    """Finite-horizon Riccati solution, u=K e; independently tested against batch QP."""
    p = np.array(q_terminal, dtype=float)
    gains = []
    for at, bt in zip(a[::-1], b[::-1], strict=True):
        h = r + bt.T @ p @ bt
        k = -np.linalg.solve(h, bt.T @ p @ at)
        gains.append(k)
        p = q + at.T @ p @ at + at.T @ p @ bt @ k
        p = (p + p.T) / 2
    return np.array(gains[::-1])


def tracking_design(model, states, controls, config):
    a, b = linearize(model, states, controls, config.get("linearization_step", 1e-5))
    q = np.diag(config.get("tracking_q", [4.0, 8.0, 6.0, 2.0, 2.0, 1.0, 1.0]))
    r = np.diag(config.get("tracking_r", [1.0, 1.0]))
    gains = lqr_gains(a, b, q, r, config.get("tracking_terminal_multiplier", 5.0) * q)
    return gains, a, b
