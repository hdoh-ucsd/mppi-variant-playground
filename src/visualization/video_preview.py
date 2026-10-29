"""Loop the experiment portion of recorded MP4s as inline README animations."""

import subprocess
from pathlib import Path

from PIL import Image, ImageSequence

from analysis.experiment_video_records import digest, git_commit, read_json, save_json


def export_preview(name, root="media/experiments"):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    clip = manifest["clips"][name]
    source = root / clip["render"]["file"]
    if digest(source) != clip["render"]["sha256"]:
        raise ValueError(f"MP4 differs from recorded provenance: {source}")
    folder = root / "previews"
    folder.mkdir(exist_ok=True)
    target = folder / f"{name}.gif"
    start = clip["render"]["opening_s"]
    duration_expected = clip["render"]["main_s"]
    interval = clip["render"]["frame_interval_s"]
    first_frame = round(start / interval)
    end_frame = round((start + duration_expected) / interval)
    filters = (
        f"[0:v]trim=start_frame={first_frame}:end_frame={end_frame},setpts=PTS-STARTPTS,"
        "fps=8,scale=960:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=128:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-threads",
            "1",
            "-i",
            str(source),
            "-filter_complex_threads",
            "1",
            "-filter_complex",
            filters,
            "-threads",
            "1",
            "-loop",
            "0",
            str(target),
        ],
        check=True,
    )
    with Image.open(target) as gif:
        width, height = gif.size
        frames = gif.n_frames
        loop = gif.info.get("loop")
        duration = sum(frame.info["duration"] for frame in ImageSequence.Iterator(gif)) / 1000
    assert (width, height, loop) == (960, 540, 0)
    assert frames > 1
    assert abs(duration - duration_expected) <= 0.15
    clip["gif_preview"] = dict(
        file=str(target.relative_to(root)),
        sha256=digest(target),
        source_mp4_sha256=digest(source),
        bytes=target.stat().st_size,
        width=width,
        height=height,
        target_fps=8,
        palette_colors=128,
        duration_s=duration,
        decoded_frames=frames,
        loop="infinite",
        source_start_s=start,
        source_end_s=start + duration_expected,
        timing="Experiment only, original playback speed; no title cards or interpolated frames",
        git_commit=git_commit(),
        exporter_sha256=digest(__file__),
        ffmpeg_filter=filters,
    )
    save_json(root / "manifest.json", manifest)
    print(f"{target}: {duration:.2f} s, {target.stat().st_size / 1e6:.2f} MB", flush=True)
