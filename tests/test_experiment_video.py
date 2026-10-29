"""Video evidence checks use saved records; no optimization or stochastic generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from analysis.experiment_video_records import digest, read_json
from visualization.experiment_video import (
    ENDING,
    FPS,
    NAMES,
    OPENING,
    Video,
    capsule_vertices,
    disturbance_index,
    ellipse_at,
    frame_phase,
    probe,
    state_index,
    vector,
)

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media/experiments"


@pytest.fixture(autouse=True)
def no_simulation(monkeypatch):
    from controllers.mppi import MPPI
    from dynamics.dynamic_vehicle import DynamicVehicle

    def forbidden(*args, **kwargs):
        raise AssertionError("Rendering must not optimize or advance dynamics")

    monkeypatch.setattr(DynamicVehicle, "step", forbidden)
    monkeypatch.setattr(MPPI, "plan", forbidden)
    yield
    plt.close("all")


def test_phase_and_zero_order_hold():
    times = np.arange(4) * 0.15
    assert state_index(times, 0.149) == 0
    assert state_index(times, 0.15) == 1
    assert state_index(times, 100) == 3
    assert frame_phase(OPENING - 1, 360)[0] == "opening"
    assert frame_phase(OPENING, 360) == ("main", 0)
    assert frame_phase(OPENING + 359, 360) == ("main", 1)
    assert frame_phase(OPENING + 360, 360)[0] == "ending"


@pytest.mark.parametrize("name", NAMES)
def test_record_hashes_and_render_without_simulation(name):
    v = Video(name, MEDIA)
    assert digest(MEDIA / v.meta["records"]) == v.meta["records_sha256"]
    for frame in [
        0,
        OPENING,
        OPENING + v.main_frames // 2,
        OPENING + v.main_frames - 1,
        OPENING + v.main_frames + ENDING - 1,
    ]:
        v.update(frame)
        v.fig.canvas.draw()
    # Opening/ending text must fit on the actual 1280x720 canvas.
    for frame in [0, OPENING + v.main_frames]:
        v.update(frame)
        v.fig.canvas.draw()
        for artist in [v.card_title, v.card_body, v.card_footer]:
            box = artist.get_window_extent()
            assert box.x0 >= 0 and box.x1 <= 1280
            assert box.y0 >= 0 and box.y1 <= 720


@pytest.mark.parametrize("name", ["V01_tsallis", "V02_dial", "V03_tube"])
def test_every_executed_pose_and_yaw_matches_records(name):
    v = Video(name, MEDIA)
    for prefix in ["a_", "b_"]:
        for i, time in enumerate(v.data[prefix + "time"]):
            fraction = time / v.end_time * (v.main_frames - FPS - 1) / (v.main_frames - 1)
            v.update_main(fraction)
            vehicle = v.vehicles[prefix]
            state = v.data[prefix + "states"][i]
            np.testing.assert_array_equal(vehicle.disk.center, state[:2])
            expected = capsule_vertices(state, v.config)
            np.testing.assert_array_equal(vehicle.body.get_xy()[:-1], expected)
            direction = np.array([np.cos(state[2]), np.sin(state[2])])
            actual = np.array([vehicle.heading.get_xdata(), vehicle.heading.get_ydata()])
            np.testing.assert_allclose(actual[:, 1] - actual[:, 0], 0.4 * direction, atol=1e-14)
    v.update_main(1)
    assert "censored timeout" in v.status.get_text() if name == "V01_tsallis" else True


def test_tube_feedback_and_disturbance_are_synchronized():
    v = Video("V03_tube", MEDIA)
    assert disturbance_index(v.config) == 9
    assert v.data["a_time"][9] == pytest.approx(1.35)
    for prefix in ["a_", "b_"]:
        applied = vector(v.data, prefix, "applied_correction", 2)
        planner = vector(v.data, prefix, "planner_control", 2)
        np.testing.assert_allclose(v.data[prefix + "controls"] - planner, applied, atol=1e-15)
        error = v.data[prefix + "states"][:-1] - vector(v.data, prefix, "nominal_state", 7)
        error[:, 2] = (error[:, 2] + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(
            error, vector(v.data, prefix, "tracking_error_vector", 7), atol=1e-14
        )
    assert np.count_nonzero(v.data["a_applied_tracking_control_norm"] > 1e-10) == 8
    assert not np.any(v.data["b_applied_tracking_control_norm"])
    for index in [8, 9]:
        fraction = (
            v.data["a_time"][index] / v.end_time * (v.main_frames - FPS - 1) / (v.main_frames - 1)
        )
        v.update_main(fraction)
        assert ("APPLIED" in v.note.get_text()) == (index >= 9)
        applied = vector(v.data, "a_", "applied_correction", 2)[index]
        assert f"Applied Δsteer {applied[0]:+.3f} rad" in v.status.get_text()


def test_dial_uses_four_saved_populations_and_all_work():
    v = Video("V02_dial_decision", MEDIA)
    np.testing.assert_array_equal(v.data["transitions"], [1920, 3840, 5760, 7680])
    assert v.data["inner_sampled_states"].shape == (4, 95, 21, 7)
    for i in range(4):
        v.update_main((i + 0.5) / 4)
        np.testing.assert_array_equal(
            v.collections["samples"].get_segments(), v.data["inner_sampled_states"][i, :, :, :2]
        )
        np.testing.assert_array_equal(v.vehicles["fixed"].disk.center, v.data["current_state"][:2])


def test_robust_replay_disturbance_candidates_and_fallback():
    v = Video("V04_robust", MEDIA)
    assert v.meta["replay_validation"] == {
        "trajectory_max_abs_error": 0.0,
        "controls_max_abs_error": 0.0,
    }
    assert disturbance_index(v.config) == 21
    applied = vector(v.data, "a_", "applied_correction", 2)
    np.testing.assert_allclose(
        v.data["a_controls"] - vector(v.data, "a_", "planner_control", 2), applied, atol=1e-15
    )
    assert not np.any(applied)
    for frame in range(450):
        v.update_main(frame / 629)
        time = min(1, frame / 419) * v.end_time
        i = state_index(v.data["a_time"], time)
        np.testing.assert_array_equal(v.vehicles["actual"].disk.center, v.data["a_states"][i, :2])
        if i < len(v.data["a_time"]) - 1:
            assert ("APPLIED" in v.status.get_text()) == (i >= 21)
            choice = int(v.data["a_nominal_selection"][i])
            np.testing.assert_array_equal(
                v.selected.get_offsets(), v.data["a_nominal_candidates"][i, choice : choice + 1, :2]
            )
    assert not v.lines["nominal"].get_visible()  # No prediction invented at final state.
    v.update_main(0.9)
    assert "ALL CANDIDATES INFEASIBLE" in v.status.get_text()
    assert np.all(v.data["f_candidate_free_energies"] > 100)
    np.testing.assert_array_equal(v.vehicles["actual"].disk.center, v.data["f_current_state"][:2])
    assert len(v.lines["executed"].get_xdata()) == 0


def test_covariance_all_executions_contacts_and_time_indices():
    v = Video("V05_covariance", MEDIA)
    states = v.data["executed_states"]
    assert states.shape == (256, 21, 7)
    covariance = np.stack([np.cov(x.T) for x in states.swapaxes(0, 1)])
    np.testing.assert_allclose(covariance, v.data["empirical_covariances"], atol=1e-12)
    assert v.contacts[:, -1].sum() == 64
    for i in range(21):
        fraction = i / 20 * 359 / 389
        v.update_main(fraction)
        np.testing.assert_array_equal(v.points.get_offsets(), states[:, i, :2])
        np.testing.assert_allclose(v.ellipses[0].center, states[:, i].mean(axis=0)[:2])
        np.testing.assert_array_equal(v.ellipses[1].center, v.data["reference_states"][i, :2])
        assert f"time {i * 0.15:.2f} s" in v.clock.get_text()
    with pytest.raises(ValueError, match="PSD"):
        ellipse_at(v.ellipses[0], np.zeros(7), -np.eye(7))


def test_readme_video_links_and_existing_figures_resolve():
    import re

    readme = (ROOT / "README.md").read_text()
    for name in NAMES:
        assert f"media/experiments/{name}.mp4" in readme
    for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme):
        assert (ROOT / path).exists()


@pytest.mark.parametrize("name", NAMES)
def test_encoded_video_validation(name):
    path = MEDIA / f"{name}.mp4"
    if not path.exists():
        pytest.skip("MP4 not generated yet; run again after rendering")
    manifest = read_json(MEDIA / "manifest.json")["clips"][name]["render"]
    result = probe(path)
    assert result == manifest["validation"]
    assert result["codec_name"] == "h264"
    assert (result["width"], result["height"], result["r_frame_rate"]) == (1280, 720, "30/1")
    assert digest(path) == manifest["sha256"]
    assert manifest["first_last_frames_decoded"]
    assert result["duration_s"] == pytest.approx(
        manifest["opening_s"] + manifest["main_s"] + manifest["ending_s"], abs=0.002
    )


@pytest.mark.parametrize("name", NAMES)
def test_frame_is_deterministic_after_seeking(name):
    video = Video(name, MEDIA)
    selected = OPENING + video.main_frames // 3
    video.update(selected)
    video.fig.canvas.draw()
    first = np.asarray(video.fig.canvas.buffer_rgba()).copy()
    video.update(OPENING + video.main_frames - 1)
    video.update(0)
    video.update(selected)
    video.fig.canvas.draw()
    np.testing.assert_array_equal(first, np.asarray(video.fig.canvas.buffer_rgba()))


@pytest.mark.parametrize("name", NAMES)
def test_main_axis_labels_do_not_overlap_scientific_annotation(name):
    video = Video(name, MEDIA)
    for fraction in [0.1, 0.5, 0.9, 1.0]:
        video.update_main(fraction)
        video.fig.canvas.draw()
        assert not video.ax.xaxis.label.get_window_extent().overlaps(video.note.get_window_extent())
