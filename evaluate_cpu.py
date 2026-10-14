"""Evaluate a saved CPU model on fresh simulations without training."""
import argparse
import os
from pathlib import Path
import sys

from deephedging.run_cpu import configure_cpu, positive_int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed training directory")
    parser.add_argument("--samples", type=positive_int, default=10000)
    parser.add_argument("--seed", type=int, help="default: training seed + 2; must be new")
    parser.add_argument("--threads", type=positive_int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--output", type=Path, help="new directory; default: RUN/evaluations/test-...")
    args = parser.parse_args(argv)
    if args.samples < 20:
        parser.error("samples must be at least 20 for the 5% tail report")
    if args.seed is not None and not 0 <= args.seed < 2**31:
        parser.error("seed must be in [0, 2**31 - 1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise SystemExit("This environment requires Python 3.10-3.12.")
    configure_cpu(args.threads)
    from deephedging.cpu_evaluation import run
    run(args)


if __name__ == "__main__":
    main()
