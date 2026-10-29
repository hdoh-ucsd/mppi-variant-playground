"""Two nonnegative 2D objectives with explicit global minimizers."""

import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import norm

from analysis.covariance_geometry import rotation


def objective(z, name):
    z = np.asarray(z)
    if name == "curved_valley":
        x, y = (z @ rotation(30))[..., 0], (z @ rotation(30))[..., 1]
        return 0.08 * (1 - x) ** 2 + 2 * (y - x**2) ** 2
    if name == "two_wells":
        x, y = z[..., 0], z[..., 1]
        return np.minimum((x - 1) ** 2 + 4 * y**2, 0.35 + (x + 1) ** 2 + 4 * y**2)
    raise ValueError(f"Unknown nonconvex objective: {name}")


def minimizer(name):
    if name == "curved_valley":
        return rotation(30) @ np.ones(2)
    if name == "two_wells":
        return np.array([1.0, 0.0])
    raise ValueError(f"Unknown objective: {name}")


def grid_reference(name, temperature, base_variance, extent=12.0, resolution=801):
    """Deterministic finite-box quadrature; report refinement and prior tail bounds."""
    if min(temperature, base_variance, extent) <= 0 or resolution < 3:
        raise ValueError("Positive scales and grid resolution >= 3 required")
    coordinates = np.linspace(-extent, extent, resolution)
    x, y = np.meshgrid(coordinates, coordinates)
    z = np.stack([x, y], axis=-1)
    prior = np.exp(-(x * x + y * y) / (2 * base_variance)) / (2 * np.pi * base_variance)
    density = np.exp(-objective(z, name) / temperature) * prior

    def integrate(f):
        return trapezoid(trapezoid(f, coordinates, axis=1), coordinates)

    evidence = integrate(density)
    mean = np.array([integrate(density * x), integrate(density * y)]) / evidence
    return {
        "mean": mean.tolist(),
        "evidence_in_box": float(evidence),
        "prior_mass_outside_box_bound": float(min(1, 4 * norm.sf(extent / np.sqrt(base_variance)))),
        "extent": extent,
        "resolution": resolution,
    }
