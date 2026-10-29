"""Axle-enclosing reference-footprint check, separate from the frozen disk-proxy benchmark."""

from datetime import UTC, datetime

import numpy as np

from analysis.artifacts import begin
from analysis.dynamic_campaign import resolved_case
from dynamics.bicycle import wrap_angle
from dynamics.dynamic_vehicle import DynamicVehicle
from evaluation.runner import write_json


def capsule_clearance(geometry, model, states):
    """Static capsule: axle-to-axle segment Minkowski-summed with the configured disk."""
    states = np.asarray(states)
    rear, front = states.copy(), states.copy()
    for endpoint, offset in [(rear, -model.rear_distance), (front, model.front_distance)]:
        endpoint[..., 0] += offset * np.cos(states[..., 2])
        endpoint[..., 1] += offset * np.sin(states[..., 2])
    return geometry.transition_clearance(rear, front)


def reference_check(config, clearance):
    resolved, geometry, states, controls, proof = resolved_case(
        config, "s_turn", clearance=clearance
    )
    model = DynamicVehicle(**resolved["vehicle"])
    _, paths = model.step(states[:-1], controls, return_path=True)
    gaps = capsule_clearance(geometry, model, paths)
    # Under the benchmark's piecewise linear position/short-angle interpolation,
    # capsule Hausdorff travel is bounded by translation + max axle offset * yaw travel.
    # Any interpolated point is at most half that travel from its nearest endpoint.
    motion = np.linalg.norm(np.diff(paths[..., :2], axis=1), axis=-1) + max(
        model.front_distance, model.rear_distance
    ) * abs(wrap_angle(np.diff(paths[..., 2], axis=1)))
    conservative = np.minimum(gaps[:, :-1], gaps[:, 1:]) - 0.5 * motion
    return dict(
        clearance=clearance,
        reference_steps=len(controls),
        minimum_sampled_capsule_clearance_m=float(gaps.min()),
        conservative_interpolated_capsule_clearance_m=float(conservative.min()),
        reference_feasible_for_capsule_interpolation=bool(conservative.min() > 0),
        original_proxy_feasibility=proof["feasible"],
    )


def run(config, output):
    declared = dict(
        phase="footprint_validation",
        campaign=config,
        footprint=dict(
            radius=config["vehicle_radius"],
            axle_distances=[
                config["vehicle"]["rear_distance"],
                config["vehicle"]["front_distance"],
            ],
            description="Axle segment thickened by configured proxy radius",
        ),
        scope="Additional reference check only; controller scoring uses centered disk proxy",
        interpolation="Linear position and shortest yaw between recorded RK2 substeps",
    )
    root, manifest = begin(output, declared, "seven_state_reference_footprint")
    rows = [reference_check(config, value) for value in config["s_turn"]["clearance_levels"]]
    result = dict(
        passed=all(r["reference_feasible_for_capsule_interpolation"] for r in rows),
        centered_proxy_encloses_axles=config["vehicle_radius"]
        >= max(config["vehicle"]["front_distance"], config["vehicle"]["rear_distance"]),
        executed_controller_capsule_safety="not evaluated or claimed",
        checks=rows,
    )
    write_json(root / "validation.json", result)
    manifest.update(
        complete=True,
        run_id=root.name,
        dynamics_model="dynamic_single_track_7",
        finished_at_utc=datetime.now(UTC).isoformat(),
    )
    write_json(root / "manifest.json", manifest)
    if not result["passed"]:
        raise RuntimeError("Full reference capsule feasibility unresolved")
