"""Headless CPU training for the original gym, simulator and OCE objective.

Import via run_cpu.py after configuring TensorFlow. No trading equation,
transaction cost, clipping rule or network default is changed by this runner.
"""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import tensorflow as tf

from cdxbasics.config import Config
from deephedging.gym import VanillaDeepHedgingGym
from deephedging.world import SimpleWorld_Spot_ATM

UPSTREAM_COMMIT = "fe53fd9a7e402d4ba18bd5b9ec704e0361554f7c"
FIELDS = ["epoch", "train_batch_loss", "val_loss", "train_seconds", "eval_seconds"]


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def build_problem(args):
    # Seed public generators directly. tf-keras 2.18's set_random_seed creates
    # an internal generator that calls randint with a float on Python 3.12.
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    config = Config()
    config.samples = args.samples
    config.steps = args.steps
    config.seed = args.seed
    config.black_scholes = args.world == "bs"
    world = SimpleWorld_Spot_ATM(config)
    validation = world.clone(samples=args.val_samples, seed=args.seed + 1)
    gym_config = Config()
    gym_config.tensorflow.seed = args.seed
    gym_config.agent.recurrence.states.classic = args.recurrent
    gym = VanillaDeepHedgingGym(gym_config)
    # Build variables on one batch, not on the entire simulation.
    gym(tf.nest.map_structure(lambda x: x[:min(args.batch_size, args.samples)], world.tf_data))
    return gym, world, validation


def make_dataset(world, batch_size, shuffle_seed=None):
    # Keras' original loss is mean(loss * probability * nSamples).
    weights = tf.reshape(world.tf_sample_weights, [-1]) * float(world.nSamples)
    dataset = tf.data.Dataset.from_tensor_slices((world.tf_data, weights))
    if shuffle_seed is not None:
        dataset = dataset.shuffle(world.nSamples, seed=shuffle_seed, reshuffle_each_iteration=False)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    options.deterministic = True
    return dataset.batch(batch_size, drop_remainder=False).with_options(options).prefetch(1)


class CpuEngine:
    def __init__(self, gym, optimizer):
        self.gym = gym
        self.optimizer = optimizer
        # Create Adam's moment variables before checkpoint restoration.
        self.optimizer.build(gym.trainable_variables)

    @tf.function(reduce_retracing=True)
    def train_batch(self, data, weights):
        with tf.GradientTape() as tape:
            losses = self.gym(data, training=True)["loss"]
            loss = tf.reduce_mean(losses * weights)
            loss = tf.debugging.check_numerics(loss, "Nonfinite training loss")
        gradients = tape.gradient(loss, self.gym.trainable_variables)
        self.optimizer.apply_gradients((g, v) for g, v in zip(gradients, self.gym.trainable_variables)
                                       if g is not None)
        return loss

    @tf.function(reduce_retracing=True)
    def train_epoch(self, dataset):
        total = tf.constant(0., tf.float32)
        count = tf.constant(0., tf.float32)
        for data, weights in dataset:
            size = tf.cast(tf.shape(weights)[0], tf.float32)
            total += self.train_batch(data, weights) * size
            count += size
        return total / count

    @tf.function(reduce_retracing=True)
    def evaluate(self, dataset):
        # No actions/predictions are copied to NumPy for each path or epoch.
        totals = tf.zeros([4], tf.float32)
        mass = tf.constant(0., tf.float32)
        for data, weights in dataset:
            result = self.gym(data, training=False)
            totals += tf.stack([tf.reduce_sum(result[key] * weights)
                                for key in ("loss", "utility", "gains", "cost")])
            mass += tf.reduce_sum(weights)
        return tf.debugging.check_numerics(totals / mass, "Nonfinite evaluation")


def manifest_for(args):
    # These settings may change without changing the optimization trajectory.
    mutable = {"output", "resume", "epochs", "threads", "preset", "checkpoint_every"}
    parameters = {k: v for k, v in vars(args).items() if k not in mutable}
    source = hashlib.sha256()
    for name in ("cpu_training.py", "gym.py", "world.py", "agents.py", "layers.py",
                 "objectives.py", "base.py", "softclip.py"):
        source.update(Path(__file__).with_name(name).read_bytes())
    return {
        "format": 1,
        "upstream_commit": UPSTREAM_COMMIT,
        "source_sha256": source.hexdigest(),
        "parameters": parameters,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("tensorflow", "tf-keras", "tensorflow-probability", "numpy", "cdxbasics")},
    }


def trim_history(path, completed_epoch):
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["epoch"]) <= completed_epoch]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    start = time.perf_counter()
    output = args.output.resolve()
    manifest = manifest_for(args)
    manifest_path = output / "run.json"
    if args.resume:
        if not manifest_path.exists():
            raise ValueError("No run.json found. Start without --resume in a new output directory.")
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Resume rejected: data/model/training settings, source or versions changed. "
                             "Use the original settings or a new --output directory.")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty. Use --resume or a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)

    gym, world, validation = build_problem(args)
    engine = CpuEngine(gym, tf.keras.optimizers.Adam(learning_rate=args.learning_rate))
    epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    best_loss = tf.Variable(float("inf"), dtype=tf.float32, trainable=False)
    best_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
    stale_checks = tf.Variable(0, dtype=tf.int64, trainable=False)
    checkpoint = tf.train.Checkpoint(gym=gym, optimizer=engine.optimizer, epoch=epoch,
                                     best_loss=best_loss, best_epoch=best_epoch, stale_checks=stale_checks)
    latest = tf.train.CheckpointManager(checkpoint, str(output / "checkpoints"), max_to_keep=2)
    # Only model variables belong to the best snapshot. Never resume Adam using
    # best-model weights paired with moment variables from the latest epoch.
    best_checkpoint = tf.train.Checkpoint(gym=gym)
    # Retain improvements so a crash between the two saves cannot delete the
    # best snapshot referenced by the last durable training checkpoint.
    best = tf.train.CheckpointManager(best_checkpoint, str(output / "best"), max_to_keep=None)
    validation_data = make_dataset(validation, args.batch_size)
    history_path = output / "metrics.csv"
    if args.resume:
        if not latest.latest_checkpoint:
            raise ValueError("No complete checkpoint found to resume.")
        checkpoint.restore(latest.latest_checkpoint).assert_consumed()
        if args.epochs < int(epoch):
            raise ValueError("--epochs is the total target; it cannot be below the restored epoch.")
        trim_history(history_path, int(epoch))
        print(f"Restored epoch {int(epoch)}, Adam step {int(engine.optimizer.iterations)}", flush=True)
    else:
        atomic_json(manifest_path, manifest)
        # Record an untrained validation baseline and allow it to remain best.
        initial = engine.evaluate(validation_data)
        best_loss.assign(initial[0])
        best.save(checkpoint_number=0)
        latest.save(checkpoint_number=0)

    print(f"CPU only | TensorFlow {tf.__version__} | threads={args.threads} | "
          f"train={world.nSamples}, validation={validation.nSamples}, steps={world.nSteps}, "
          f"instruments={world.nInst} | parameters={gym.num_trainable_weights}", flush=True)
    print("The first training epoch includes graph compilation; subsequent epochs are representative.", flush=True)
    timings = []
    stopped = "target_reached"
    try:
        with history_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if stream.tell() == 0:
                writer.writeheader()
            for index in range(int(epoch), args.epochs):
                if args.patience and int(stale_checks) >= args.patience:
                    stopped = "early_stopping"
                    break
                begin = time.perf_counter()
                # Epoch-specific shuffle seed allows restarts to reproduce order.
                dataset = make_dataset(world, args.batch_size, (args.seed + index) % (2**31 - 1))
                train_loss = float(engine.train_epoch(dataset))
                seconds = time.perf_counter() - begin
                epoch.assign(index + 1)
                timings.append(seconds)
                val_loss = None
                eval_seconds = 0.
                if int(epoch) % args.eval_every == 0 or int(epoch) == args.epochs:
                    begin_eval = time.perf_counter()
                    val_loss = float(engine.evaluate(validation_data)[0])
                    eval_seconds = time.perf_counter() - begin_eval
                    if val_loss < float(best_loss):
                        best_loss.assign(val_loss)
                        best_epoch.assign(epoch)
                        stale_checks.assign(0)
                        best.save(checkpoint_number=int(epoch))
                    else:
                        stale_checks.assign_add(1)
                writer.writerow(dict(epoch=int(epoch), train_batch_loss=train_loss,
                                     val_loss=val_loss, train_seconds=seconds, eval_seconds=eval_seconds))
                stream.flush()
                # A validation point and its selected best checkpoint are durable together.
                if val_loss is not None or int(epoch) % args.checkpoint_every == 0:
                    latest.save(checkpoint_number=int(epoch))
                message = f"epoch {int(epoch)}/{args.epochs} loss={train_loss:.7g} train={seconds:.2f}s"
                if val_loss is not None:
                    message += f" val_loss={val_loss:.7g} eval={eval_seconds:.2f}s best_epoch={int(best_epoch)}"
                if len(timings) >= 2:
                    eta = np.median(timings[1:][-10:]) * (args.epochs - int(epoch))
                    message += f" remaining_train~{eta:.0f}s (excludes validation/checkpoints)"
                print(message, flush=True)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Resume the last completed checkpoint: {latest.latest_checkpoint}\n"
              "Partially completed/uncheckpointed epochs will be repeated.", flush=True)
        raise SystemExit(130)

    latest.save(checkpoint_number=int(epoch))
    # Restore the checkpoint referenced by the durable state, not an orphaned
    # best snapshot from a process that died before saving its latest state.
    best_checkpoint.restore(str(output / "best" / f"ckpt-{int(best_epoch)}")).assert_consumed()
    validation_metrics = engine.evaluate(validation_data).numpy()
    gym.save_weights(str(output / "best.weights.h5"))
    summary = dict(
        status=stopped, completed_epochs=int(epoch), best_epoch=int(best_epoch),
        best_validation_loss=float(best_loss),
        validation=dict(zip(("loss", "utility", "mean_gains", "mean_cost"),
                            [float(value) for value in validation_metrics])),
        optimizer_steps=int(engine.optimizer.iterations),
        median_warm_train_epoch_seconds=float(np.median(timings[1:])) if len(timings) > 1 else None,
        invocation_wall_seconds=time.perf_counter() - start,
        cpu_threads=args.threads, python=platform.python_version(), platform=platform.platform(),
        visible_gpus=[device.name for device in tf.config.get_visible_devices("GPU")],
        note="Validation selects weights; it is not an independent test or a claim of convergence.",
    )
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
