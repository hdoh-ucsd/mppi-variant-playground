"""Versioned, controller-independent state-space snapshot interchange (no pickle)."""

import json
from pathlib import Path

import numpy as np

REQUIRED = (
    "sampled_states",
    "costs",
    "weights",
    "current_state",
    "executed_next_state",
    "chosen_control",
)
OPTIONAL = ("sampled_controls", "nominal_states", "log_base_over_proposal")


def validate_snapshot(snapshot):
    for field in REQUIRED:
        if field not in snapshot:
            raise ValueError(f"Missing snapshot field: {field}")
        if not np.isfinite(snapshot[field]).all():
            raise ValueError(f"Nonfinite snapshot field: {field}")
    states = np.asarray(snapshot["sampled_states"])
    if states.ndim != 3 or min(states.shape[:2]) < 2 or states.shape[2] < 2:
        raise ValueError("Expected [samples >= 2, horizon+1 >= 2, state dimension >= 2]")
    k, length, d = states.shape
    costs, weights = np.asarray(snapshot["costs"]), np.asarray(snapshot["weights"])
    if costs.shape != (k,) or weights.shape != (k,):
        raise ValueError("Costs/weights must describe the full sampled population")
    if (weights < 0).any() or not np.isclose(weights.sum(), 1, rtol=1e-10, atol=1e-12):
        raise ValueError("Weights must be nonnegative and normalized")
    if np.shape(snapshot["current_state"]) != (d,) or np.shape(snapshot["executed_next_state"]) != (
        d,
    ):
        raise ValueError("Current/next states must match rollout state dimension")
    if not np.allclose(states[:, 0], snapshot["current_state"], atol=1e-6):
        raise ValueError("Rollouts must start at the recorded planning state")
    if snapshot["best_rollout_index"] != int(costs.argmin()):
        raise ValueError("Best rollout index must minimize raw trajectory cost")
    if snapshot["highest_weight_index"] != int(weights.argmax()):
        raise ValueError("Highest-weight index is inconsistent with the weights")
    for field in OPTIONAL:
        if field in snapshot and not np.isfinite(snapshot[field]).all():
            raise ValueError(f"Nonfinite optional field: {field}")
    if "nominal_states" in snapshot and np.shape(snapshot["nominal_states"]) != (length, d):
        raise ValueError("Nominal trajectory shape does not match rollout horizon")
    if "sampled_controls" in snapshot and np.shape(snapshot["sampled_controls"])[:2] != (
        k,
        length - 1,
    ):
        raise ValueError("Sampled controls must match samples and horizon")
    return k, length, d


def save_snapshot(path, snapshot):
    """State/control archives use float32 for display; costs and weights stay float64."""
    k, length, d = validate_snapshot(snapshot)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        key: snapshot[key]
        for key in ("step", "best_rollout_index", "highest_weight_index", "nominal_role")
        if key in snapshot
    }
    metadata.update(
        {
            "schema_version": 1,
            "sample_count": k,
            "state_dimension": d,
            "horizon": length - 1,
            "state_storage_dtype": "float32",
            "best_by": "raw_trajectory_cost",
            "weight_semantics": "normalized_optimizer_weights",
        }
    )
    data = {
        field: np.asarray(
            snapshot[field],
            dtype=(
                np.float32
                if field in {"sampled_states", "sampled_controls", "nominal_states"}
                else np.float64
            ),
        )
        for field in (*REQUIRED, *OPTIONAL)
        if field in snapshot
    }
    # Exclusive file creation protects already recorded evidence.
    with path.open("xb") as stream:
        np.savez_compressed(stream, metadata=np.array(json.dumps(metadata)), **data)


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["schema_version"] != 1:
            raise ValueError("Unsupported rollout snapshot schema")
        result = {key: data[key].copy() for key in data.files if key != "metadata"}
    result.update(metadata)
    validate_snapshot(result)
    return result


def snapshot_statistics(snapshot):
    states, weights = snapshot["sampled_states"], snapshot["weights"]
    endpoints = states[:, -1, :2]
    center = weights @ endpoints
    delta = endpoints - center
    weighted_trace = np.sum(weights * np.sum(delta**2, axis=1))
    positive = weights > 0
    return {
        "sample_count": len(weights),
        "ess": float(1 / (weights @ weights)),
        "weight_entropy": float(-weights[positive] @ np.log(weights[positive])),
        "max_weight": float(weights.max()),
        "unweighted_endpoint_variance_trace_m2": float(np.var(endpoints, axis=0).sum()),
        "weighted_endpoint_variance_trace_m2": float(weighted_trace),
        "minimum_sample_cost": float(np.min(snapshot["costs"])),
        "best_is_highest_weight": bool(
            snapshot["best_rollout_index"] == snapshot["highest_weight_index"]
        ),
    }
