"""CPU-only entry point. TensorFlow must be configured before it is imported."""
import argparse
import os
from pathlib import Path
import sys

PRESETS = {
    "smoke": dict(samples=256, val_samples=128, steps=10, epochs=2, batch_size=64),
    "laptop": dict(samples=4096, val_samples=1024, steps=20, epochs=100, batch_size=256),
    "full": dict(samples=10000, val_samples=2000, steps=20, epochs=800, batch_size=256),
}


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="laptop")
    for name in ("samples", "val-samples", "steps", "epochs", "batch-size"):
        parser.add_argument("--" + name, type=positive_int)
    parser.add_argument("--threads", type=positive_int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--eval-every", type=positive_int, default=5)
    parser.add_argument("--checkpoint-every", type=positive_int, default=5)
    parser.add_argument("--patience", type=int, default=0,
                        help="early stop after N validation checks without improvement; 0 disables")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--world", choices=("stochastic", "bs"), default="stochastic")
    parser.add_argument("--recurrent", type=int, default=0,
                        help="number of classic recurrent states; 0 = feedforward")
    parser.add_argument("--output", type=Path, default=Path("runs/cpu"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    for name, value in PRESETS[args.preset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.patience < 0 or args.recurrent < 0:
        parser.error("patience and recurrent must be nonnegative")
    if not 0 <= args.seed < 2**31 - 2:
        parser.error("seed must be in [0, 2**31 - 3]")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("learning-rate must be finite and positive")
    return args


def configure_cpu(threads):
    if "tensorflow" in sys.modules:
        raise RuntimeError("Start a fresh Python process: CPU settings must precede TensorFlow import.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    return tf


def main(argv=None):
    args = parse_args(argv)
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise SystemExit("This pinned environment requires Python 3.10-3.12 (recommended: 3.11).")
    configure_cpu(args.threads)
    from deephedging.cpu_training import run
    run(args)


if __name__ == "__main__":
    main()
