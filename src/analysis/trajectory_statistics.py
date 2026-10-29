"""Empirical executed-path summaries with explicit alignment and sample weighting."""

import numpy as np
from scipy.ndimage import gaussian_filter


def occupancy(trajectories, x_edges, y_edges, *, weighting="equal_seed", smoothing=1.0):
    if not trajectories or weighting not in {"equal_seed", "pooled_time"}:
        raise ValueError("Trajectories and an explicit occupancy weighting are required")
    x_edges, y_edges = np.asarray(x_edges), np.asarray(y_edges)
    if (np.diff(x_edges) <= 0).any() or (np.diff(y_edges) <= 0).any() or smoothing < 0:
        raise ValueError("Increasing bin edges and nonnegative smoothing required")
    points = np.concatenate(trajectories)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite [points,2] executed positions")
    if any(len(path) == 0 for path in trajectories):
        raise ValueError("An empty trajectory has no occupancy measure")
    if (
        points[:, 0].min() < x_edges[0]
        or points[:, 0].max() > x_edges[-1]
        or points[:, 1].min() < y_edges[0]
        or points[:, 1].max() > y_edges[-1]
    ):
        raise ValueError("Occupancy bounds must include every recorded state")
    weights = (
        np.concatenate(
            [np.full(len(path), 1 / (len(trajectories) * len(path))) for path in trajectories]
        )
        if weighting == "equal_seed"
        else np.full(len(points), 1 / len(points))
    )
    mass = np.histogram2d(points[:, 0], points[:, 1], bins=(x_edges, y_edges), weights=weights)[0].T
    if smoothing:
        mass = gaussian_filter(mass, sigma=smoothing, mode="constant")
    mass /= mass.sum()
    area = np.diff(y_edges)[:, None] * np.diff(x_edges)[None, :]
    return {"mass": mass, "density": mass / area, "x_edges": x_edges, "y_edges": y_edges}


def progress_dispersion(trajectories, count=101):
    """Index each path by fraction of its own traveled arc length, not goal progress.

    No padding after termination. Failures are included, and a stationary path stays fixed.
    Bands are pointwise inter-seed quantiles, not confidence intervals for a mean.
    """
    if not trajectories or count < 2:
        raise ValueError("Trajectories and at least two alignment points required")
    grid = np.linspace(0, 1, count)
    aligned = []
    for path in trajectories:
        path = np.asarray(path)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 1 or not np.isfinite(path).all():
            raise ValueError("Expected nonempty finite XY trajectories")
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        if distance[-1] <= 1e-12:
            aligned.append(np.broadcast_to(path[0], (count, 2)).copy())
        else:
            unique, indices = np.unique(distance / distance[-1], return_index=True)
            aligned.append(
                np.column_stack([np.interp(grid, unique, path[indices, i]) for i in range(2)])
            )
    aligned = np.array(aligned)
    center = aligned.mean(axis=0)
    tangent = np.gradient(center, axis=0)
    lengths = np.linalg.norm(tangent, axis=1)
    tangent[lengths < 1e-12] = [1.0, 0.0]
    lengths[lengths < 1e-12] = 1.0
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]]) / lengths[:, None]
    offsets = np.einsum("npi,pi->np", aligned - center, normal)
    return {
        "progress": grid,
        "aligned_xy": aligned,
        "mean_xy": center,
        "normal_xy": normal,
        "offset_q05": np.quantile(offsets, 0.05, axis=0),
        "offset_q25": np.quantile(offsets, 0.25, axis=0),
        "offset_q75": np.quantile(offsets, 0.75, axis=0),
        "offset_q95": np.quantile(offsets, 0.95, axis=0),
        "q05_xy": np.quantile(aligned, 0.05, axis=0),
        "q25_xy": np.quantile(aligned, 0.25, axis=0),
        "q75_xy": np.quantile(aligned, 0.75, axis=0),
        "q95_xy": np.quantile(aligned, 0.95, axis=0),
    }


def first_crossing(path, x):
    """First forward crossing of a declared station; no crossing remains censored."""
    path = np.asarray(path)
    candidates = np.flatnonzero((path[:-1, 0] <= x) & (path[1:, 0] > x))
    if not len(candidates):
        return None
    i = candidates[0]
    fraction = (x - path[i, 0]) / (path[i + 1, 0] - path[i, 0])
    return float(path[i, 1] + fraction * (path[i + 1, 1] - path[i, 1]))
