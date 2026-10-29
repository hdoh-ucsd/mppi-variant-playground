"""Shared CLI parsing for independent investigations."""

import argparse

from utils.config import read_yaml


def arguments(default_config, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--output", required=True, help="New output directory; never overwritten")
    parser.add_argument("--plots-only", action="store_true", help="Rebuild figures from saved data")
    args = parser.parse_args()
    return args, read_yaml(args.config)
