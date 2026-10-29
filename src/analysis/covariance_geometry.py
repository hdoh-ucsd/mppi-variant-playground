"""Covariance geometry with explicit, distinct scale constraints."""

import numpy as np

from sampling.gaussian import validate_covariance


def rotation(degrees):
    if not np.isfinite(degrees):
        raise ValueError("Rotation must be finite")
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, -s], [s, c]])


def covariance(ratio=1.0, angle=0.0, scale=1.0, reference_trace=1.2, constraint="trace"):
    """Ratio is large/small eigenvalue; scale multiplies the covariance matrix.

    trace: trace(Sigma) = reference_trace * scale.
    determinant: det(Sigma) = (reference_trace * scale / 2)**2.
    Thus the isotropic member agrees across constraints, while other members differ.
    """
    if not np.isfinite([ratio, angle, scale, reference_trace]).all():
        raise ValueError("Covariance parameters must be finite")
    if ratio < 1 or min(scale, reference_trace) <= 0:
        raise ValueError("Ratio >= 1 and positive scale/trace required")
    if constraint == "trace":
        eigen = reference_trace * scale * np.array([ratio, 1]) / (ratio + 1)
    elif constraint == "determinant":
        eigen = reference_trace * scale / 2 * np.array([np.sqrt(ratio), 1 / np.sqrt(ratio)])
    else:
        raise ValueError("Constraint must be trace or determinant")
    r = rotation(angle)
    return validate_covariance((r * eigen) @ r.T)


def alignment(proposal, target):
    """Angle of major axes and KL(target || proposal) covariance-only component."""
    proposal, target = np.asarray(proposal), np.asarray(target)
    vp, up = np.linalg.eigh(proposal)
    vt, ut = np.linalg.eigh(target)
    identifiable = not (np.isclose(vp[0], vp[-1]) or np.isclose(vt[0], vt[-1]))
    angle = (
        float(np.degrees(np.arccos(np.clip(abs(up[:, -1] @ ut[:, -1]), 0, 1))))
        if identifiable
        else None
    )
    mismatch = 0.5 * (
        np.trace(np.linalg.solve(proposal, target))
        - len(target)
        + np.linalg.slogdet(proposal)[1]
        - np.linalg.slogdet(target)[1]
    )
    return {"target_axis_angle_deg": angle, "covariance_kl": float(max(0, mismatch))}
