"""Render saved MPPI evidence; optionally extract inputs and explicitly replay missing V04 data."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        default="all",
        choices=[
            "all",
            "V01_tsallis",
            "V02_dial",
            "V02_dial_decision",
            "V03_tube",
            "V04_robust",
            "V05_covariance",
        ],
    )
    parser.add_argument("--output", default="media/experiments")
    parser.add_argument("--raw", default="runs/variants")
    parser.add_argument(
        "--prepare-records", action="store_true", help="Extract original raw records"
    )
    parser.add_argument(
        "--replay-missing",
        action="store_true",
        help="With --prepare-records, allow one exact V04 episode replay",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--gif-previews", action="store_true", help="Convert existing MP4s to README GIF previews"
    )
    args = parser.parse_args()
    if args.gif_previews and (args.prepare_records or args.prepare_only or args.replay_missing):
        parser.error("--gif-previews uses existing MP4s; omit preparation/replay options")
    if (args.replay_missing or args.prepare_only) and not args.prepare_records:
        parser.error("--replay-missing / --prepare-only require --prepare-records")
    if args.prepare_records:
        from analysis.experiment_video_records import prepare

        prepare(args.output, args.raw, args.replay_missing)
    if not args.prepare_only:
        from visualization.experiment_video import NAMES, render

        if args.gif_previews:
            from visualization.video_preview import export_preview

            render = export_preview
        for name in NAMES if args.video == "all" else [args.video]:
            render(name, args.output)


if __name__ == "__main__":
    main()
