"""Frozen-policy evaluation for runs produced by cpu_training.py.

The source files hashed by existing training manifests are deliberately unchanged.
All strategies use identical simulated paths and the original settlement/cost rules.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
import tensorflow as tf

from deephedging.cpu_training import atomic_json, build_problem, make_dataset, manifest_for


def exp2_oce(gains, lmbda=1.):
    """Empirical sup_y mean(u(X+y))-y, using the upstream exp2 utility.

    This scalar risk statistic is recomputed equally for every strategy; it is
    distinct from evaluating the model's frozen, learned OCE intercept.
    """
    x = np.asarray(gains, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or lmbda <= 0:
        raise ValueError("Expected a finite gain vector and positive risk aversion")
    lo, hi = -float(x.max()), -float(x.min())

    def slope(y):
        z = x + y
        derivative = np.exp(-lmbda * np.maximum(z, 0.)) - lmbda * np.minimum(z, 0.)
        return float(derivative.mean() - 1.)

    y = lo if lo == hi else brentq(slope, lo, hi, xtol=1e-12)
    z = x + y
    positive, negative = np.maximum(z, 0.), np.minimum(z, 0.)
    u = -np.expm1(-lmbda * positive) / lmbda + negative - .5 * lmbda * negative**2
    return float(u.mean() - y)


def strategy_metrics(gains, costs):
    x, costs = np.asarray(gains, dtype=np.float64), np.asarray(costs, dtype=np.float64)
    if x.ndim != 1 or x.shape != costs.shape or len(x) < 20:
        raise ValueError("Expected matching vectors with at least 20 paths")
    if not np.isfinite(x).all() or not np.isfinite(costs).all():
        raise ValueError("Nonfinite evaluation result")
    # Exact 5% mass, including a fractional boundary observation when N/20 is not integer.
    losses = np.sort(-x)[::-1]
    mass = .05 * len(x)
    whole = int(np.floor(mass))
    tail = (losses[:whole].sum() + (mass - whole) * losses[whole]) / mass
    return dict(mean_gains=float(x.mean()), std_gains=float(x.std(ddof=0)),
                mean_cost=float(costs.mean()), loss_var_95=float(np.quantile(-x, .95)),
                loss_cvar_95=float(tail), empirical_oce_exp2=exp2_oce(x))


def delta_actions(world, world_kind):
    """Spot-only Black-Scholes delta proxy for the default short strike-1 call.

    Uses current spot/ivol, zero rates, and no future prices. The original
    stochastic simulator's payoff slice includes one extra time step beyond
    spot hedge settlement; retain that offset when calculating claim maturity.
    This is a heuristic in the stochastic-volatility/cost setting, not optimal.
    """
    features, market = world.data.features.per_step, world.data.market
    tau = np.asarray(features.time_left, dtype=np.float64)
    if world_kind == "stochastic":
        tau = tau + world.dt
    sigma = np.asarray(features.ivol, dtype=np.float64)
    spot = np.asarray(features.spot, dtype=np.float64)
    target = ndtr((np.log(spot) + .5 * sigma**2 * tau) / (sigma * np.sqrt(tau)))
    actions = np.zeros_like(market.hedges, dtype=np.float64)
    position = np.zeros(world.nSamples, dtype=np.float64)
    for step in range(world.nSteps):
        # Bounds limit trades, not cumulative holdings. Direct clipping enforces
        # the same admissible interval without the network's soft parameterization.
        trade = np.clip(target[:, step] - position,
                        market.lbnd_a[:, step, 0], market.ubnd_a[:, step, 0])
        actions[:, step, 0] = trade
        position += trade
    return actions


def settle_actions(world, actions):
    market = world.data.market
    actions = np.asarray(actions, dtype=np.float64)
    pnl = np.sum(actions * np.asarray(market.hedges, dtype=np.float64), axis=(1, 2))
    cost = np.sum(np.abs(actions) * np.asarray(market.cost, dtype=np.float64), axis=(1, 2))
    return np.asarray(market.payoff, dtype=np.float64) + pnl - cost, cost


def load_run(directory, requested_seed):
    manifest = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    training_args = SimpleNamespace(**manifest["parameters"])
    if manifest_for(training_args) != manifest:
        raise ValueError("Evaluation rejected: training source or core package versions changed. "
                         "Use the environment/source recorded in run.json.")
    seed = training_args.seed + 2 if requested_seed is None else requested_seed
    if not 0 <= seed < 2**31 or seed in (training_args.seed, training_args.seed + 1):
        raise ValueError("Test seed must be valid and differ from both training and validation seeds.")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not (directory / "best.weights.h5").is_file():
        raise ValueError("Missing best.weights.h5; finish the training invocation first.")
    # A resumed but unfinished invocation can leave an older H5 export/summary.
    latest = tf.train.latest_checkpoint(str(directory / "checkpoints"))
    if latest is None:
        raise ValueError("Missing training checkpoint")
    reader = tf.train.load_checkpoint(latest)
    epoch = int(reader.get_tensor("epoch/.ATTRIBUTES/VARIABLE_VALUE"))
    best = int(reader.get_tensor("best_epoch/.ATTRIBUTES/VARIABLE_VALUE"))
    if epoch != summary["completed_epochs"] or best != summary["best_epoch"]:
        raise ValueError("Summary/export is stale after resume. Finish training before evaluation.")
    return training_args, manifest, summary, seed


def run(args):
    start = time.perf_counter()
    directory = args.run.resolve()
    training_args, manifest, training_summary, seed = load_run(directory, args.seed)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output if args.output is not None else
              directory / "evaluations" / f"test-{seed}-{stamp}").resolve()
    if output.exists():
        raise ValueError("Evaluation output already exists. Choose a new directory.")
    print(f"Frozen best epoch {training_summary['best_epoch']} | CPU threads={args.threads} | "
          f"fresh test paths={args.samples}, seed={seed}", flush=True)
    gym, training, validation = build_problem(training_args)
    test = training.clone(samples=args.samples, seed=seed)
    del training, validation
    if not np.allclose(test.sample_weights, 1. / test.nSamples):
        raise ValueError("This report requires the simulator's uniform sample weights")
    weights_path = directory / "best.weights.h5"
    weights_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    gym.load_weights(str(weights_path))
    if gym.utility.utility != "exp2" or gym.utility.lmbda != 1.:
        raise ValueError("This evaluation report supports the CPU runner's exp2@1 objective")
    keys = ("loss", "utility", "gains", "cost")
    collected = {key: [] for key in keys}
    for data, _ in make_dataset(test, args.batch_size):
        result = gym(data, training=False)
        for key in keys:
            collected[key].append(result[key].numpy())
    values = {key: np.concatenate(parts).astype(np.float64) for key, parts in collected.items()}
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError("Nonfinite model output")
    if hashlib.sha256(weights_path.read_bytes()).hexdigest() != weights_hash:
        raise ValueError("Weights changed during evaluation; stop concurrent training and retry")
    unhedged = np.asarray(test.data.market.payoff, dtype=np.float64)
    delta_gains, delta_cost = settle_actions(test, delta_actions(test, training_args.world))
    strategies = {
        "deep_hedging": strategy_metrics(values["gains"], values["cost"]),
        "unhedged": strategy_metrics(unhedged, np.zeros_like(unhedged)),
        "spot_delta": strategy_metrics(delta_gains, delta_cost),
    }
    comparisons = {}
    model = strategies["deep_hedging"]
    for name in ("unhedged", "spot_delta"):
        baseline = strategies[name]
        comparisons[name] = dict(
            mean_gains_difference=model["mean_gains"] - baseline["mean_gains"],
            std_reduction_percent=100. * (1. - model["std_gains"] / baseline["std_gains"])
            if baseline["std_gains"] > 0 else None,
            loss_cvar_95_reduction=baseline["loss_cvar_95"] - model["loss_cvar_95"],
            empirical_oce_improvement=model["empirical_oce_exp2"] - baseline["empirical_oce_exp2"],
        )
    summary = dict(
        status="evaluated", best_epoch=training_summary["best_epoch"], samples=test.nSamples,
        test_seed=seed, training_seed=training_args.seed, validation_seed=training_args.seed + 1,
        world=training_args.world, steps=test.nSteps, instruments=test.inst_names,
        source_sha256=manifest["source_sha256"], weights_sha256=weights_hash,
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        versions=manifest["versions"], cpu_threads=args.threads, python=platform.python_version(),
        visible_gpus=[device.name for device in tf.config.get_visible_devices("GPU")],
        frozen_model=dict(loss=float(values["loss"].mean()), utility=float(values["utility"].mean())),
        strategies=strategies, deep_hedging_vs_baseline=comparisons,
        invocation_wall_seconds=time.perf_counter() - start, output=str(output),
        notes=[
            "Fresh seed, same simulator distribution. No training or policy selection on test paths.",
            "Gains include claim payoff, trading P&L and costs; no initial option premium is added.",
            "Lower std and loss_cvar_95 are better for those risk metrics; higher empirical OCE is better.",
            "Empirical OCE re-estimates only the scalar risk statistic for each strategy. "
            "Frozen-model utility uses the learned intercept; these are different estimators.",
            "Spot delta uses only spot and observed implied volatility, zero rates, the same trade "
            "bounds/costs and settlement, and no option trades. It is a heuristic benchmark.",
            "Original stochastic simulator evaluates the claim one dt after spot hedge settlement; "
            "this is preserved, and delta maturity includes that extra dt. BS mode has no offset.",
            "Paths share the simulator's cross-sectional normalization. Metrics are descriptive; "
            "no iid confidence intervals or real-market profitability claims are provided.",
            "Reusing this test to choose epochs/hyperparameters makes it validation; reserve a new "
            "seed for the final assessment after further tuning.",
        ],
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "strategies.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["strategy", *model])
        writer.writeheader()
        for name, metrics in strategies.items():
            writer.writerow(dict(strategy=name, **metrics))
    with (output / "paths.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "unhedged_gains", "deep_hedging_gains", "deep_hedging_cost",
                         "spot_delta_gains", "spot_delta_cost"])
        writer.writerows(zip(range(test.nSamples), unhedged, values["gains"], values["cost"],
                            delta_gains, delta_cost))
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    return summary
