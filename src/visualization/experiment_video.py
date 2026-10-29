"""Deterministic replay of compact numerical records. No controller or simulator calls."""

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

from analysis.experiment_video_records import VERSION, digest, git_commit, read_json, save_json
from visualization.figures import draw_scene, style

FPS = 30
OPENING = 75
ENDING = 90
BLUE, ORANGE, GREEN, RED = "#2563eb", "#d97706", "#059669", "#dc264b"
NAMES = ["V01_tsallis", "V02_dial", "V02_dial_decision", "V03_tube", "V04_robust", "V05_covariance"]
TITLES = {
    "V01_tsallis": (
        "V01 · Divergence and weighting",
        "How does the weighting choice change motion?",
    ),
    "V02_dial": ("V02 · Iterative refinement", "What does the vehicle execute after optimization?"),
    "V02_dial_decision": (
        "V02 · Inside one DIAL decision",
        "How do saved populations change across four passes?",
    ),
    "V03_tube": ("V03 · Nominal tracking", "When does the ancillary controller actually act?"),
    "V04_robust": (
        "V04 · Feedback-aware planning",
        "Which nominal state is selected after a disturbance?",
    ),
    "V05_covariance": (
        "V05 · State covariance",
        "Does linearized dispersion match independent executions?",
    ),
}


def state_index(times, time):
    """Zero-order hold: never invent an interpolated pose or reveal a future state."""
    return int(np.clip(np.searchsorted(times, time + 1e-10, side="right") - 1, 0, len(times) - 1))


def frame_phase(frame, main_frames):
    if frame < OPENING:
        return "opening", 0.0
    if frame >= OPENING + main_frames:
        return "ending", 1.0
    return "main", (frame - OPENING) / (main_frames - 1)


def disturbance_index(config):
    event = config["scenario"].get("disturbance")
    return event["step"] + 1 if event else None


def vector(data, prefix, key, size):
    return np.column_stack([data[f"{prefix}{key}_{i}"] for i in range(size)])


def capsule_vertices(state, config):
    """Same axle-enclosing capsule as footprint_validation; not a certified body envelope."""
    radius = config["scenario"]["vehicle_radius"]
    front, rear = config["vehicle"]["front_distance"], config["vehicle"]["rear_distance"]
    theta = np.linspace(-np.pi / 2, np.pi / 2, 17)
    local = np.vstack(
        [
            np.column_stack([front + radius * np.cos(theta), radius * np.sin(theta)]),
            np.column_stack([-rear - radius * np.cos(theta), -radius * np.sin(theta)]),
        ]
    )
    c, s = np.cos(state[2]), np.sin(state[2])
    return local @ np.array([[c, s], [-s, c]]) + state[:2]


class Vehicle:
    def __init__(self, ax, config, color, alpha=0.25):
        self.config = config
        self.body = Polygon(
            np.zeros((34, 2)), closed=True, facecolor=color, edgecolor=color, alpha=alpha, zorder=8
        )
        self.disk = Circle(
            (0, 0),
            config["scenario"]["vehicle_radius"],
            fill=False,
            edgecolor=color,
            linestyle="--",
            lw=1.2,
            zorder=9,
        )
        (self.heading,) = ax.plot([], [], color=color, lw=2, zorder=10)
        ax.add_patch(self.body)
        ax.add_patch(self.disk)

    def update(self, state, visible=True):
        for artist in [self.body, self.disk, self.heading]:
            artist.set_visible(visible)
        if not visible:
            return
        self.body.set_xy(capsule_vertices(state, self.config))
        self.disk.center = state[:2]
        end = state[:2] + self.config["vehicle"]["front_distance"] * np.array(
            [np.cos(state[2]), np.sin(state[2])]
        )
        self.heading.set_data([state[0], end[0]], [state[1], end[1]])


def ellipse_at(artist, mean, covariance):
    eigen, basis = np.linalg.eigh((covariance[:2, :2] + covariance[:2, :2].T) / 2)
    if eigen.min() < -1e-9:
        raise ValueError("Covariance is not PSD")
    artist.center = mean[:2]
    artist.width, artist.height = 2 * np.sqrt(5.991464547 * np.maximum(eigen[::-1], 0))
    artist.angle = np.degrees(np.arctan2(basis[1, -1], basis[0, -1]))


class Video:
    def __init__(self, name, root):
        self.name, self.root = name, Path(root)
        self.meta = read_json(self.root / "manifest.json")["clips"][name]
        path = self.root / self.meta["records"]
        if digest(path) != self.meta["records_sha256"]:
            raise ValueError("Render input hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            self.data = {k: data[k] for k in data.files}
        self.config = self.meta["sources"][0]["config"]
        style()
        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        self.ax = self.fig.add_axes([0.065, 0.23, 0.64, 0.61])
        self.diag = self.fig.add_axes([0.77, 0.23, 0.20, 0.23])
        draw_scene(self.ax, self.config["scenario"])
        self.fig.text(0.065, 0.935, TITLES[name][0], fontsize=22, weight="bold", color="#172331")
        self.clock = self.fig.text(0.065, 0.875, "", fontsize=12, color="#334155")
        self.status = self.fig.text(0.755, 0.80, "", va="top", fontsize=10, linespacing=1.25)
        self.note = self.fig.text(0.065, 0.155, "", fontsize=10, color="#334155")
        self.fig.text(
            0.065,
            0.06,
            "Oriented axle capsule = reference footprint; "
            "dashed disk = collision proxy (r = 0.22 m).\n"
            "Axle spacing = 0.80 m. "
            "Neither the disk nor these videos establish full-body collision safety.",
            fontsize=9,
            color="#64748b",
        )
        self.lines, self.vehicles = {}, {}
        self.collections = {}
        self.main_frames = 360
        self.setup()
        self.ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        self.card = Rectangle(
            (0, 0), 1, 1, transform=self.fig.transFigure, facecolor="#142236", zorder=50
        )
        self.fig.add_artist(self.card)
        self.card_title = self.fig.text(
            0.08, 0.76, TITLES[name][0], fontsize=27, weight="bold", color="white", zorder=51
        )
        self.card_body = self.fig.text(
            0.08, 0.59, "", fontsize=17, va="top", linespacing=1.7, color="#dbeafe", zorder=51
        )
        self.card_footer = self.fig.text(
            0.08,
            0.12,
            "Recorded numerical evidence · seven-state single-track vehicle",
            fontsize=12,
            color="#93a7be",
            zorder=51,
        )

    def line(self, name, color, label, **kwargs):
        (self.lines[name],) = self.ax.plot([], [], color=color, label=label, **kwargs)

    def cloud(self, name, color, label, alpha=0.12):
        collection = LineCollection([], colors=color, linewidths=0.6, alpha=alpha, label=label)
        self.ax.add_collection(collection)
        self.collections[name] = collection

    def bounds(self, points):
        points = np.concatenate([np.asarray(p).reshape(-1, 2) for p in points])
        # Include the task geometry and every displayed trajectory; never crop failed samples.
        scene = self.config["scenario"]
        extra = [scene["initial_state"][:2], scene["goal_state"][:2]]
        for x, y, r in scene["obstacles"]:
            extra += [[x - r, y - r], [x + r, y + r]]
        points = np.vstack([points, extra])
        low, high = points.min(axis=0) - 0.9, points.max(axis=0) + 0.9
        mid_y = (low[1] + high[1]) / 2
        half_y = max(2.2, (high[1] - low[1]) / 2)
        self.ax.set(xlim=(low[0], high[0]), ylim=(mid_y - half_y, mid_y + half_y))

    def setup(self):
        d, name = self.data, self.name
        if name in ["V01_tsallis", "V02_dial", "V03_tube"]:
            labels = ["IT-MPPI", "Tsallis-VI" if name.startswith("V01") else "DIAL-MPC"]
            if name == "V03_tube":
                labels = ["Tube reset threshold 0", "Tube reset threshold 450"]
            self.labels = labels
            self.end_time = max(d["a_time"][-1], d["b_time"][-1])
            self.main_frames = round(max(12, self.end_time) * FPS) + FPS
            for prefix, color, label in zip(["a_", "b_"], [BLUE, ORANGE], labels, strict=True):
                self.line(prefix, color, label)
                self.vehicles[prefix] = Vehicle(self.ax, self.config, color)
                if name == "V03_tube":
                    self.line(prefix + "nominal", color, label + " selected nominal", ls="--", lw=1)
                    self.vehicles[prefix + "nominal"] = Vehicle(self.ax, self.config, color, 0.08)
            self.bounds([d["a_states"][:, :2], d["b_states"][:, :2]])
            self.diag_lines = [self.diag.plot([], [], color=c)[0] for c in [BLUE, ORANGE]]
            self.diag.set(
                xlim=(0, self.end_time),
                xlabel="Simulation time [s]",
                ylabel="Progress [m]" if name != "V03_tube" else "Position error [m]",
            )
            if name == "V03_tube":
                self.errors = [
                    np.linalg.norm(vector(d, p, "tracking_error_vector", 7)[:, :2], axis=1)
                    for p in ["a_", "b_"]
                ]
                self.diag.set_ylim(0, max(0.1, max(x.max() for x in self.errors) * 1.15))
                self.diag.axvline(
                    d["a_time"][disturbance_index(self.config)], color=RED, ls=":", lw=1
                )
                self.observation = (
                    f"Threshold 0: {np.count_nonzero(d['a_applied_tracking_control_norm'] > 1e-10)}"
                    "/32 commands apply feedback.\n"
                    "Threshold 450: ancillary feedback remains inactive."
                )
                self.limitation = (
                    "One positive impulse and seed; this is not a general tracking benefit."
                )
            else:
                self.diag.set_ylim(0, 10.5)
                self.observation = (
                    "IT-MPPI completes; Tsallis times out at 24.00 s."
                    if name.startswith("V01")
                    else f"Both complete: IT {d['a_time'][-1]:.2f} s; DIAL {d['b_time'][-1]:.2f} s."
                )
                self.limitation = "One paired seed and fixed settings; no universal method ranking."
        elif name == "V02_dial_decision":
            self.cloud("samples", BLUE, "95 unweighted predicted rollouts / pass")
            self.line("plan", ORANGE, "Updated nominal prediction", lw=2.5)
            self.vehicles["fixed"] = Vehicle(self.ax, self.config, GREEN)
            self.bounds([d["inner_sampled_states"][..., :2], d["inner_nominal_states"][..., :2]])
            self.diag.plot(np.arange(1, 5), d["planned_cost"], "o-", color=ORANGE)
            (self.cursor,) = self.diag.plot([], [], "o", color=RED, ms=9)
            self.diag.set(xlabel="Internal pass", ylabel="Updated plan cost", xticks=[1, 2, 3, 4])
            self.observation = (
                f"Saved nominal-plan cost: {d['planned_cost'][0]:.2f}"
                f" → {d['planned_cost'][-1]:.2f}."
            )
            self.limitation = (
                "Predictions at one frozen state; clouds are not physical uncertainty."
            )
        elif name == "V04_robust":
            self.main_frames = 630  # 15 seconds executed, then 6 seconds explicitly frozen.
            self.end_time = d["a_time"][-1]
            self.line("executed", BLUE, "Executed trajectory", lw=2.5)
            self.line("nominal", ORANGE, "Nominal prediction", ls="--")
            self.cloud("augmented", BLUE, "Actual augmented predictions (subset)")
            self.cloud("nominal_cloud", ORANGE, "Nominal augmented predictions (subset)")
            self.vehicles["actual"] = Vehicle(self.ax, self.config, BLUE)
            self.candidates = self.ax.scatter(
                [], [], marker="x", c=RED, s=35, label="5 nominal candidates", zorder=11
            )
            self.selected = self.ax.scatter(
                [],
                [],
                marker="s",
                facecolors="none",
                edgecolors=GREEN,
                s=140,
                label="Selected origin",
                zorder=12,
            )
            self.bounds(
                [
                    d["a_states"][:, :2],
                    d["a_nominal_states"][..., :2],
                    d["a_sampled_states"][..., :2],
                    d["f_sampled_states"][..., :2],
                ]
            )
            self.executed_limits = (self.ax.get_xlim(), self.ax.get_ylim())
            self.bars = self.diag.bar(np.arange(5), np.zeros(5), color="#94a3b8")
            self.threshold_line = self.diag.axhline(5000, color=RED, ls="--", lw=1)
            maximum = max(
                d["a_candidate_free_energies"].max(), d["f_candidate_free_energies"].max(), 5000
            )
            self.energy_limit = maximum * 1.12
            self.diag.set(
                ylim=(0, self.energy_limit),
                xticks=range(5),
                xlabel="Candidate index",
                ylabel="Probe free energy",
            )
            self.observation = (
                "Default run: feedback inactive; terminal position overshoots goal.\n"
                "Frozen threshold-100 search: all candidates infeasible; fallback 0."
            )
            self.limitation = (
                "Fallback is not successful robust optimization; "
                "frozen rollouts are not executions."
            )
        else:
            self.main_frames = 390
            states = d["executed_states"]
            self.mean = states.mean(axis=0)
            self.contacts = np.maximum.accumulate(d["contacts"], axis=1)
            self.cloud("executions", BLUE, "First 32 independent executions", alpha=0.35)
            self.line("mean", BLUE, "Empirical mean (256 executions)", lw=2.5)
            self.line("reference", ORANGE, "Linearization reference", ls="--")
            self.points = self.ax.scatter(
                [], [], s=9, alpha=0.35, zorder=5, label="All 256 current positions"
            )
            self.ellipses = []
            for color, label, ls in [
                (BLUE, "Empirical nonlinear dispersion", "-"),
                (ORANGE, "Predicted linearized covariance", "--"),
            ]:
                e = Ellipse((0, 0), 0, 0, fill=False, edgecolor=color, ls=ls, lw=2, label=label)
                self.ax.add_patch(e)
                self.ellipses.append(e)
            for i in range(16):
                self.vehicles[i] = Vehicle(self.ax, self.config, BLUE, 0.10)
            self.bounds([states[..., :2], d["reference_states"][:, :2]])
            times = np.arange(states.shape[1]) * self.config["vehicle"]["dt"]
            self.cov_times = times
            self.traces = [
                np.trace(d[k][:, :2, :2], axis1=1, axis2=2)
                for k in ["empirical_covariances", "predicted_covariances"]
            ]
            self.diag_lines = [
                self.diag.plot([], [], color=c, ls=ls)[0] for c, ls in [(BLUE, "-"), (ORANGE, "--")]
            ]
            self.diag.set(
                xlim=(0, times[-1]),
                ylim=(0, max(x.max() for x in self.traces) * 1.1),
                xlabel="Simulation time [s]",
                ylabel="Position covariance trace [m²]",
            )
            count = self.contacts[:, -1].sum()
            self.observation = f"{count}/256 independent executions contact a disk-proxy obstacle."
            self.limitation = (
                "Local covariance bounds do not guarantee collision safety. No impact dynamics."
            )

    def update_main(self, fraction):
        d, name = self.data, self.name
        if name in ["V01_tsallis", "V02_dial", "V03_tube"]:
            requested_time = (
                min(1, fraction * (self.main_frames - 1) / (self.main_frames - FPS - 1))
                * self.end_time
            )
            blocks = []
            for j, prefix in enumerate(["a_", "b_"]):
                times, states = d[prefix + "time"], d[prefix + "states"]
                i = state_index(times, requested_time)
                self.lines[prefix].set_data(states[: i + 1, 0], states[: i + 1, 1])
                self.vehicles[prefix].update(states[i])
                ended = i == len(times) - 1
                termination = self.meta["sources"][j]["result"]["termination"].replace("_", " ")
                block = f"{self.labels[j]}\nt = {times[i]:.2f} s · " + (
                    termination if ended else "running"
                )
                if name == "V03_tube":
                    n = min(i, len(times) - 2)
                    nominal = vector(d, prefix, "nominal_state", 7)
                    self.lines[prefix + "nominal"].set_data(
                        nominal[: n + 1, 0], nominal[: n + 1, 1]
                    )
                    self.vehicles[prefix + "nominal"].update(nominal[n], not ended)
                    self.diag_lines[j].set_data(times[: n + 1], self.errors[j][: n + 1])
                    if not ended:
                        steer, throttle = vector(d, prefix, "applied_correction", 2)[i]
                        reset = bool(d[prefix + "reset_event"][i])
                        active = np.linalg.norm([steer, throttle]) > 1e-10
                        block += (
                            f"\nPosition error {self.errors[j][i]:.3f} m"
                            f"\nApplied Δsteer {steer:+.3f} rad"
                            f"\nApplied Δthrottle {throttle:+.3f}"
                            f"\nFeedback {'ACTIVE' if active else 'inactive'}; reset {reset}"
                        )
                    else:
                        block += "\n32-command observation ends"
                else:
                    self.diag_lines[j].set_data(times[: i + 1], d[prefix + "progress"][: i + 1])
                    block += f"\nProgress {d[prefix + 'progress'][i]:.2f} m / 10 m"
                blocks.append(block)
            self.status.set_text("\n\n".join(blocks))
            speed = self.end_time / (self.main_frames / FPS - 1)
            self.clock.set_text(
                f"Shared replay clock {requested_time:.2f} s · {speed:.2f}× speed · "
                "saved poses held between samples"
            )
            if name == "V03_tube":
                event_time = d["a_time"][disturbance_index(self.config)]
                self.note.set_text(
                    f"Positive impulse first appears at t = {event_time:.2f} s (state 9). "
                    + ("APPLIED" if requested_time >= event_time - 1e-10 else "Pending")
                    + "\nDashed paths: selected nominal origins with resets; not a future plan."
                )
            else:
                self.note.set_text(
                    "Paired seed 48000 · two-route obstacle · original 7,700-transition budget "
                    "cap\n"
                    "No synchronized weights/nominal plans were recorded. Ended runs stay visible."
                )
        elif name == "V02_dial_decision":
            i = min(3, int(fraction * 4))
            self.collections["samples"].set_segments(d["inner_sampled_states"][i, :, :, :2])
            self.lines["plan"].set_data(
                d["inner_nominal_states"][i, :, 0], d["inner_nominal_states"][i, :, 1]
            )
            self.vehicles["fixed"].update(d["current_state"])
            self.cursor.set_data([i + 1], [d["planned_cost"][i]])
            self.clock.set_text(
                "Frozen simulation time 1.20 s · optimizer seed 51000 · vehicle does not move"
            )
            scales = d["inner_variance_scales"][i]
            self.status.set_text(
                f"Internal pass {i + 1} / 4\n95 saved samples\n\nCumulative transitions\n"
                f"{int(d['transitions'][i]):,} / 7,680\n\nUpdated plan cost\n"
                f"{d['planned_cost'][i]:.2f}\n\nNoise variance scale\n{scales.min():.3f} – "
                f"{scales.max():.3f}"
            )
            self.note.set_text(
                "Each cloud is one actual saved pass; all 95 samples are unweighted.\n"
                "These are 3.0 s horizon predictions, not independent physical executions."
            )
        elif name == "V04_robust":
            main_frame = round(fraction * (self.main_frames - 1))
            if main_frame < 450:
                self.ax.set(xlim=self.executed_limits[0], ylim=self.executed_limits[1])
                self.diag.set_ylim(0, self.energy_limit)
                time = min(1, main_frame / 419) * self.end_time
                i = state_index(d["a_time"], time)
                n = min(i, len(d["a_controls"]) - 1)
                states = d["a_states"]
                self.lines["executed"].set_data(states[: i + 1, 0], states[: i + 1, 1])
                self.vehicles["actual"].update(states[i])
                plan = d["a_nominal_states"][n]
                self.collections["augmented"].set_segments(d["a_sampled_states"][n, :, :, :2])
                self.collections["nominal_cloud"].set_segments([])
                candidates, energy = d["a_nominal_candidates"][n], d["a_candidate_free_energies"][n]
                choice = int(d["a_nominal_selection"][n])
                threshold = 5000
                steer, throttle = vector(d, "a_", "applied_correction", 2)[n]
                event_time = d["a_time"][disturbance_index(self.config)]
                self.clock.set_text(
                    f"Executed run · t = {d['a_time'][i]:.2f} s · seed 48000 · α = 5000"
                )
                self.status.set_text(
                    f"Candidate {choice} selected\n"
                    f"Search infeasible: {bool(d['a_nominal_selection_infeasible'][n])}"
                    f"\n\nApplied Δsteer {steer:+.3f} rad\nApplied Δthrottle {throttle:+.3f}"
                    f"\nFeedback {'inactive' if abs(steer) + abs(throttle) < 1e-10 else 'ACTIVE'}"
                    f"\n\nImpulse at {event_time:.2f} s\n"
                    + ("APPLIED" if i >= disturbance_index(self.config) else "Pending")
                )
                self.note.set_text(
                    "Solid blue: execution. Dashed orange: current nominal prediction. Faint "
                    "blue: first 8 augmented rollouts.\n"
                    "Original completion uses projected progress; terminal goal-distance error "
                    "is 6.90 m."
                )
            else:
                self.bounds(
                    [
                        d["f_sampled_states"][..., :2],
                        d["f_nominal_samples"][..., :2],
                        d["f_nominal_states"][..., :2],
                    ]
                )
                self.diag.set_ylim(0, d["f_candidate_free_energies"].max() * 1.2)
                h = min(20, int((main_frame - 450) / 179 * 20))
                self.lines["executed"].set_data([], [])
                self.vehicles["actual"].update(d["f_current_state"])
                plan = d["f_nominal_states"]
                self.collections["augmented"].set_segments(d["f_sampled_states"][:, : h + 1, :2])
                self.collections["nominal_cloud"].set_segments(
                    d["f_nominal_samples"][:, : h + 1, :2]
                )
                candidates, energy = d["f_nominal_candidates"], d["f_candidate_free_energies"]
                row = self.meta["fallback_record"]
                choice, threshold = int(row["selected_candidate"]), 100
                self.clock.set_text(
                    f"FROZEN offset decision · source t = 1.20 s · forecast h = {h * 0.15:.2f} s "
                    f"· seed 51000"
                )
                self.status.set_text(
                    f"α = 100\nALL CANDIDATES INFEASIBLE\nFallback candidate {choice}"
                    f"\n\nSampled feedback RMS\n{row['feedback_rms']:.3f} normalized"
                    f"\n\nRequested Δsteer\n{row['requested_tracking_steering']:+.3f} rad"
                    f"\nClipped command contribution\n{row['applied_tracking_steering']:+.3f} rad"
                )
                self.note.set_text(
                    "Separate frozen test: positive state offset, no executed transition. First "
                    "16 paired augmented predictions.\n"
                    "Active feedback within predictions does not make an infeasible fallback a "
                    "robustness success."
                )
            self.lines["nominal"].set_data(plan[:, 0], plan[:, 1])
            self.candidates.set_offsets(candidates[:, :2])
            self.selected.set_offsets(candidates[choice : choice + 1, :2])
            for j, bar in enumerate(self.bars):
                bar.set_height(energy[j])
                bar.set_color(GREEN if j == choice else "#94a3b8")
            self.threshold_line.set_ydata([threshold, threshold])
            ended = main_frame < 450 and i == len(d["a_time"]) - 1
            self.lines["nominal"].set_visible(not ended)
            self.candidates.set_visible(not ended)
            self.selected.set_visible(not ended)
            if ended:
                self.collections["augmented"].set_segments([])
                self.status.set_text(
                    "Recorded completion\nGoal-distance error: 6.90 m\n\n"
                    "No new decision at final state.\nBars: last decision probes."
                )
        else:
            time = min(1, fraction * 389 / 359) * self.cov_times[-1]
            i = state_index(self.cov_times, time)
            states, reference = d["executed_states"], d["reference_states"]
            self.collections["executions"].set_segments(states[:32, : i + 1, :2])
            self.collections["executions"].set_color(
                [RED if c else BLUE for c in self.contacts[:32, i]]
            )
            self.points.set_offsets(states[:, i, :2])
            self.points.set_color([RED if c else BLUE for c in self.contacts[:, i]])
            for j in range(16):
                self.vehicles[j].update(states[j, i])
            self.lines["mean"].set_data(self.mean[: i + 1, 0], self.mean[: i + 1, 1])
            self.lines["reference"].set_data(reference[: i + 1, 0], reference[: i + 1, 1])
            ellipse_at(self.ellipses[0], self.mean[i], d["empirical_covariances"][i])
            ellipse_at(self.ellipses[1], reference[i], d["predicted_covariances"][i])
            for line, trace in zip(self.diag_lines, self.traces, strict=True):
                line.set_data(self.cov_times[: i + 1], trace[: i + 1])
            count = self.contacts[:, i].sum()
            self.clock.set_text(
                f"Synchronized simulation time {self.cov_times[i]:.2f} s · state {i}/20 · 4× "
                f"slower replay"
            )
            self.status.set_text(
                f"256 independent executions\nSeeds 56000–56255\nFull C₀ actuator noise\n\n"
                f"Contact so far: {count} / 256\nRed = any prior contact\n\n32 paths, 16 "
                f"oriented vehicles\nAll 256 position dots\nAll 256 enter covariance"
            )
            self.note.set_text(
                "Ellipses: 95% Gaussian-scaled position covariance contours; no Gaussian "
                "coverage guarantee or mean CI.\n"
                "Contacts use recorded substep clearance. Failed samples retained; no "
                "contact-impact model."
            )

    def update(self, frame):
        phase, fraction = frame_phase(frame, self.main_frames)
        self.update_main(fraction)
        for artist in [self.card, self.card_title, self.card_body, self.card_footer]:
            artist.set_visible(phase != "main")
        if phase == "opening":
            seeds = self.meta["seeds"]
            seed_label = str(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}–{seeds[-1]}"
            if self.name == "V04_robust":
                seed_label = "48000 (execution); 51000 (frozen decision)"
            self.card_body.set_text(
                TITLES[self.name][1]
                + "\n\n"
                + self.meta["scenario"].replace("_", " ")
                + "\nSeed(s): "
                + seed_label
            )
        elif phase == "ending":
            self.card_body.set_text(self.observation + "\n\nLimit of inference\n" + self.limitation)
        return []


def probe(path):
    info = json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            text=True,
        )
    )
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {
        k: stream[k]
        for k in ["codec_name", "width", "height", "pix_fmt", "r_frame_rate", "nb_frames"]
    } | {"duration_s": float(info["format"]["duration"])}


def render(name, root="media/experiments"):
    root = Path(root)
    video = Video(name, root)
    frames = OPENING + video.main_frames + ENDING
    animation = FuncAnimation(
        video.fig,
        video.update,
        frames=frames,
        interval=1000 / FPS,
        blit=False,
        cache_frame_data=False,
    )
    target = root / f"{name}.mp4"
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=-1,
        extra_args=[
            "-crf",
            "23",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-movflags",
            "+faststart",
        ],
        metadata={
            "title": TITLES[name][0],
            "comment": "Saved numerical records; no synthesized motion",
        },
    )
    animation.save(target, writer=writer, dpi=100)
    plt.close(video.fig)
    info = probe(target)
    assert info["codec_name"] == "h264" and info["pix_fmt"] == "yuv420p"
    assert (info["width"], info["height"], info["r_frame_rate"]) == (1280, 720, "30/1")
    assert int(info["nb_frames"]) == frames and abs(info["duration_s"] - frames / FPS) < 0.002
    # Decode first and last frames, plus a representative main frame for visual review.
    preview = Path("runs/variants/videos/previews")
    preview.mkdir(parents=True, exist_ok=True)
    for label, frame in [
        ("first", 0),
        ("main", OPENING + video.main_frames // 2),
        ("last", frames - 1),
    ]:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-i",
                str(target),
                "-vf",
                f"select=eq(n\\,{frame})",
                "-frames:v",
                "1",
                str(preview / f"{name}_{label}.png"),
            ],
            check=True,
        )
    manifest = read_json(root / "manifest.json")
    manifest["clips"][name]["render"] = dict(
        file=target.name,
        sha256=digest(target),
        bytes=target.stat().st_size,
        git_commit=git_commit(),
        rendering_version=VERSION,
        renderer_sha256=digest(__file__),
        frame_interval_s=1 / FPS,
        state_mapping="Zero-order hold of saved poses; final states held; no interpolation.",
        opening_s=OPENING / FPS,
        main_s=video.main_frames / FPS,
        ending_s=ENDING / FPS,
        validation=info,
        first_last_frames_decoded=True,
        observation=video.observation,
        limitation=video.limitation,
    )
    save_json(root / "manifest.json", manifest)
    print(f"{target}: {info['duration_s']:.1f} s, {target.stat().st_size / 1e6:.2f} MB", flush=True)
