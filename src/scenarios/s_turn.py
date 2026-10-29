"""Feasible steering-reversal corridor and exact disk/polygon boundary geometry."""

import numpy as np
from scipy.spatial import cKDTree

from dynamics.steering_rate import make_model


def point_segments(points, a, b):
    delta = b - a
    t = np.clip(
        np.sum((points - a) * delta, axis=-1) / np.maximum(np.sum(delta**2, axis=-1), 1e-30), 0, 1
    )
    closest = a + t[..., None] * delta
    return np.linalg.norm(points - closest, axis=-1), t


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def segment_distances(a, b, c, d):
    first = np.minimum(point_segments(a, c, d)[0], point_segments(b, c, d)[0])
    second = np.minimum(point_segments(c, a, b)[0], point_segments(d, a, b)[0])
    ab, cd = b - a, d - c
    intersects = (
        (cross(ab, c - a) * cross(ab, d - a) <= 0)
        & (cross(cd, a - c) * cross(cd, b - c) <= 0)
        & (
            np.maximum(np.minimum(a[..., 0], b[..., 0]), np.minimum(c[..., 0], d[..., 0]))
            <= np.minimum(np.maximum(a[..., 0], b[..., 0]), np.maximum(c[..., 0], d[..., 0]))
        )
        & (
            np.maximum(np.minimum(a[..., 1], b[..., 1]), np.minimum(c[..., 1], d[..., 1]))
            <= np.minimum(np.maximum(a[..., 1], b[..., 1]), np.maximum(c[..., 1], d[..., 1]))
        )
    )
    return np.where(intersects, 0, np.minimum(first, second))


class Polyline:
    """Exact distances, with a certified KD-tree candidate screen and exhaustive fallback."""

    def __init__(self, points):
        points = np.asarray(points, dtype=float)
        refined = []
        for a, b in zip(points[:-1], points[1:], strict=True):
            n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.15)))
            refined.extend(a + (b - a) * (np.arange(n) / n)[:, None])
        self.points = np.vstack([refined, points[-1:]])
        self.a, self.b = self.points[:-1], self.points[1:]
        self.length = np.linalg.norm(self.b - self.a, axis=1)
        if (self.length <= 1e-10).any():
            raise ValueError("Nonzero polyline edges required")
        self.arc = np.r_[0, np.cumsum(self.length)]
        self.halfmax = self.length.max() / 2
        self.tree = cKDTree((self.a + self.b) / 2)
        self.vertex_tree = cKDTree(self.points)

    def nearest(self, points, end=None):
        points = np.asarray(points)
        if end is not None:
            return self.swept(points, end)
        shape = points.shape[:-1]
        p = points.reshape(-1, 2)
        e = None if end is None else np.asarray(end).reshape(-1, 2)
        mid = p if e is None else (p + e) / 2
        k = min(8, len(self.a))
        distances, indices = self.tree.query(mid, k=k)
        if k == 1:
            distances, indices = distances[:, None], indices[:, None]

        def evaluate(rows, idx):
            if e is None:
                return point_segments(p[rows, None], self.a[idx], self.b[idx])[0]
            return segment_distances(p[rows, None], e[rows, None], self.a[idx], self.b[idx])

        values = evaluate(np.arange(len(p)), indices)
        choice = values.argmin(axis=1)
        best = values[np.arange(len(p)), choice]
        chosen = indices[np.arange(len(p)), choice]
        # Every unqueried midpoint is at least d_k away. The triangle inequality
        # bounds its segment distance below by d_k - half_motion - max_half_edge.
        vertex_distance, vertex_index = self.vertex_tree.query(p)
        use_vertex = vertex_distance < best
        best = np.minimum(best, vertex_distance)
        chosen[use_vertex] = np.minimum(vertex_index[use_vertex], len(self.a) - 1)
        # An interior segment projection has squared distance >= midpoint_distance²
        # minus half_edge². Endpoint minima are already covered by the vertex tree.
        fallback = np.flatnonzero(best**2 > distances[:, -1] ** 2 - self.halfmax**2)
        if k == len(self.a):
            fallback = np.array([], dtype=int)
        for start in range(0, len(fallback), 256):
            rows = fallback[start : start + 256]
            idx = np.broadcast_to(np.arange(len(self.a)), (len(rows), len(self.a)))
            values = evaluate(rows, idx)
            selected = values.argmin(axis=1)
            best[rows], chosen[rows] = values[np.arange(len(rows)), selected], selected
        _, fraction = point_segments(p, self.a[chosen], self.b[chosen])
        progress = self.arc[chosen] + fraction * self.length[chosen]
        return best.reshape(shape), progress.reshape(shape), chosen.reshape(shape)

    def swept(self, points, end):
        shape = np.shape(points)[:-1]
        p, e = np.asarray(points).reshape(-1, 2), np.asarray(end).reshape(-1, 2)
        best = np.minimum(self.nearest(p)[0], self.nearest(e)[0])
        mid, half = (p + e) / 2, np.linalg.norm(e - p, axis=1) / 2
        k = min(8, len(self.points))
        dv, iv = self.vertex_tree.query(mid, k=k)
        if k == 1:
            dv, iv = dv[:, None], iv[:, None]
        values = point_segments(self.points[iv], p[:, None], e[:, None])[0]
        best = np.minimum(best, values.min(axis=1))
        missing = np.flatnonzero(best**2 > dv[:, -1] ** 2 - half**2)
        if k == len(self.points):
            missing = np.array([], dtype=int)
        for start in range(0, len(missing), 128):
            rows = missing[start : start + 128]
            distance = point_segments(self.points[None], p[rows, None], e[rows, None])[0]
            best[rows] = np.minimum(best[rows], distance.min(axis=1))
        k = min(8, len(self.a))
        ds, idx = self.tree.query(mid, k=k)
        if k == 1:
            ds, idx = ds[:, None], idx[:, None]

        def intersections(rows, indices):
            a, b = p[rows, None], e[rows, None]
            c, d = self.a[indices], self.b[indices]
            ab, cd = b - a, d - c
            return (
                (cross(ab, c - a) * cross(ab, d - a) <= 0)
                & (cross(cd, a - c) * cross(cd, b - c) <= 0)
                & (
                    np.maximum(np.minimum(a[..., 0], b[..., 0]), np.minimum(c[..., 0], d[..., 0]))
                    <= np.minimum(
                        np.maximum(a[..., 0], b[..., 0]), np.maximum(c[..., 0], d[..., 0])
                    )
                )
                & (
                    np.maximum(np.minimum(a[..., 1], b[..., 1]), np.minimum(c[..., 1], d[..., 1]))
                    <= np.minimum(
                        np.maximum(a[..., 1], b[..., 1]), np.maximum(c[..., 1], d[..., 1])
                    )
                )
            )

        hit = intersections(np.arange(len(p)), idx).any(axis=1)
        missing = np.flatnonzero((~hit) & (ds[:, -1] <= half + self.halfmax))
        if k == len(self.a):
            missing = np.array([], dtype=int)
        for start in range(0, len(missing), 128):
            rows = missing[start : start + 128]
            indices = np.broadcast_to(np.arange(len(self.a)), (len(rows), len(self.a)))
            hit[rows] = intersections(rows, indices).any(axis=1)
        best[hit] = 0
        return best.reshape(shape), np.zeros(shape), np.zeros(shape, dtype=int)


def reference_maneuver(config):
    """One mechanically specified +/−/+ curvature maneuver; no controller outcomes used."""
    model = make_model(config["vehicle"])
    shape = config["corridor"]
    speed = shape["speed_target"]
    dt = model.dt
    angle = np.arctan(model.bicycle.wheelbase / shape["turn_radius"])
    n = max(6, int(np.ceil(shape["turn_separation"] / speed / dt)))
    states = [np.array(config["initial_state"], dtype=float)]
    controls = []

    def advance(control):
        controls.append(np.array(control, dtype=float))
        states.append(model.step(states[-1], control))

    for _ in range(int(np.ceil(speed / dt))):
        acceleration = min(1.0, (speed - states[-1][3]) / dt)
        advance([acceleration, 0])
    for _ in range(int(np.ceil(shape["entry_length"] / speed / dt))):
        advance([0, 0])
    for sign, count in ((1, n), (-1, 2 * n), (1, n)):
        for i in range(count):
            desired = sign * angle * np.sin(np.pi * (i + 1) / count) ** 2
            rate = (desired - states[-1][4]) / dt
            if abs(rate) > model.steering_rate_limit + 1e-12:
                raise ValueError("Reference rate exceeds the declared steering-rate limit")
            advance([0, rate])
    for _ in range(int(np.ceil(shape["exit_length"] / speed / dt))):
        advance([0, 0])
    return np.array(states), np.array(controls)


class Corridor:
    """Closed polygon from parallel offsets of a mechanically feasible centerline.

    Boundary distances and swept disk tests use line segments, not waypoint proximity.
    The footprint is the original disk. A polygon's boundary is fixed per condition.
    """

    def __init__(self, reference, clearance, vehicle_radius, extension=1.0):
        if min(clearance, vehicle_radius, extension) <= 0:
            raise ValueError("Positive clearance, footprint and end extension required")
        self.vehicle_radius = vehicle_radius
        self.clearance = clearance
        keep = np.r_[True, np.linalg.norm(np.diff(reference[:, :2], axis=0), axis=1) > 1e-8]
        ref = reference[keep]
        xy = ref[:, :2]
        heading = ref[:, 2]
        tangent = np.column_stack([np.cos(heading), np.sin(heading)])
        xy = np.vstack([xy[0] - extension * tangent[0], xy, xy[-1] + extension * tangent[-1]])
        heading = np.r_[heading[0], heading, heading[-1]]
        normal = np.column_stack([-np.sin(heading), np.cos(heading)])
        halfwidth = vehicle_radius + clearance
        self.upper = xy + halfwidth * normal
        self.lower = xy - halfwidth * normal
        if (np.diff(self.upper[:, 0]) <= 0).any() or (np.diff(self.lower[:, 0]) <= 0).any():
            raise ValueError("Offsets must have monotone x; reduce curvature/width")
        polygon = np.vstack([self.upper, self.lower[::-1], self.upper[:1]])
        self.boundary = Polyline(polygon)
        self.center = Polyline(ref[:, :2])
        self.goal = reference[-1].copy()

    def inside(self, xy):
        xy = np.asarray(xy)
        x, y = xy[..., 0], xy[..., 1]
        upper = np.interp(x, self.upper[:, 0], self.upper[:, 1])
        lower = np.interp(x, self.lower[:, 0], self.lower[:, 1])
        # For this graph-bounded polygon the end caps are vertical (zero terminal headings).
        return (x >= self.upper[0, 0]) & (x <= self.upper[-1, 0]) & (y >= lower) & (y <= upper)

    def state_clearance(self, state):
        xy = np.asarray(state)[..., :2]
        distance = self.boundary.nearest(xy)[0]
        return np.where(self.inside(xy), distance, -distance) - self.vehicle_radius

    def transition_clearance(self, start, end):
        a, b = np.asarray(start)[..., :2], np.asarray(end)[..., :2]
        distance = self.boundary.nearest(a, b)[0] - self.vehicle_radius
        outside = ~(self.inside(a) & self.inside(b))
        if np.any(outside):
            shape = np.shape(distance)
            flat = np.asarray(distance).reshape(-1).copy()
            selected = np.flatnonzero(np.asarray(outside).reshape(-1))
            first = np.asarray(start).reshape(-1, np.shape(start)[-1])[selected]
            last = np.asarray(end).reshape(-1, np.shape(end)[-1])[selected]
            flat[selected] = np.minimum(
                flat[selected], np.minimum(self.state_clearance(first), self.state_clearance(last))
            )
            distance = flat.reshape(shape)
        return distance

    def project(self, state):
        return self.center.nearest(np.asarray(state)[..., :2])


def feasibility(config, clearance):
    states, controls = reference_maneuver(config)
    corridor = Corridor(
        states, clearance, config["corridor"]["vehicle_radius"], config["corridor"]["end_extension"]
    )
    gaps = corridor.transition_clearance(states[:-1], states[1:])
    model = make_model(config["vehicle"])
    signs = np.sign(states[:, 4][np.abs(states[:, 4]) > 0.01])
    reversals = int(np.count_nonzero(np.diff(signs)))
    diagnostics = dict(
        feasible=bool(
            gaps.min() > 0 and len(controls) <= config["execution"]["max_steps"] and reversals >= 2
        ),
        min_clearance=float(gaps.min()),
        max_steering=float(abs(states[:, 4]).max()),
        max_steering_rate=float(abs(controls[:, 1]).max()),
        steering_reversals=reversals,
        turning_radius=float(model.bicycle.wheelbase / np.tan(abs(states[:, 4]).max())),
        reference_steps=len(controls),
        duration_s=len(controls) * model.dt,
        centerline_length=float(corridor.center.arc[-1]),
        initial_state=states[0].tolist(),
        goal_state=states[-1].tolist(),
        corridor_width=2 * (config["corridor"]["vehicle_radius"] + clearance),
        disk_radius=config["corridor"]["vehicle_radius"],
    )
    return corridor, states, controls, diagnostics
