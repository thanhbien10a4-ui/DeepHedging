import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
import tensorflow as tf

from deephedging.cpu_evaluation import delta_actions, exp2_oce, load_run, run, settle_actions, strategy_metrics
from deephedging.cpu_training import build_problem, run as train
from deephedging.objectives import tf_utility
from deephedging.run_cpu import parse_args


def tiny_args(path, world="stochastic", recurrent=0):
    return parse_args(["--preset", "smoke", "--samples", "25", "--val-samples", "21",
                       "--steps", "3", "--batch-size", "8", "--epochs", "1",
                       "--world", world, "--recurrent", str(recurrent), "--output", str(path)])


def test_risk_metrics_known_tail_and_upstream_objective():
    # 21 paths => 1.05 observations in the 5% tail: 20 plus .05 * 19.
    gains = -np.arange(21, dtype=np.float64)
    result = strategy_metrics(gains, np.zeros(21))
    assert result["loss_cvar_95"] == pytest.approx((20 + .05 * 19) / 1.05)
    assert result["loss_var_95"] == pytest.approx(19.)
    x = np.array([-.8, -.2, -.1, .1, .6], dtype=np.float64)
    objective = lambda y: -float(tf.reduce_mean(tf_utility("exp2", 1., tf.constant(x), y=y)["u"]))
    optimized = minimize_scalar(objective, bounds=(-1., 1.), method="bounded")
    assert exp2_oce(x) == pytest.approx(-optimized.fun, abs=1e-8)
    assert exp2_oce(x + 3.) == pytest.approx(exp2_oce(x) + 3.)
    assert exp2_oce(np.full(20, -2.)) == pytest.approx(-2.)


@pytest.mark.parametrize("world_kind,recurrent", [("bs", 0), ("stochastic", 2)])
def test_settlement_matches_gym_and_delta_has_no_lookahead(tmp_path, world_kind, recurrent):
    gym, world, _ = build_problem(tiny_args(tmp_path, world_kind, recurrent))
    result = gym(world.tf_data, training=False)
    gains, cost = settle_actions(world, result["actions"].numpy())
    np.testing.assert_allclose(gains, result["gains"], atol=2e-6)
    np.testing.assert_allclose(cost, result["cost"], atol=2e-6)
    # BS mode disables the hedging option; the claim is still a short strike-1
    # call. Its delta at S=1 is N(.5 * sigma * sqrt(T)), around .51 here.
    actions = delta_actions(world, world_kind)
    assert np.all(actions >= world.data.market.lbnd_a)
    assert np.all(actions <= world.data.market.ubnd_a)
    if world.nInst == 2:
        assert not np.any(actions[:, :, 1])
    if world_kind == "bs":
        assert np.all((actions[:, 0, 0] > .5) & (actions[:, 0, 0] < .52))
    world.data.market.payoff[:] = 999.
    world.data.market.hedges[:] = -123.
    np.testing.assert_array_equal(delta_actions(world, world_kind), actions)
    # A changed future observation must not change any earlier decision.
    world.data.features.per_step.spot[:, -1] *= 2.
    np.testing.assert_array_equal(delta_actions(world, world_kind)[:, :-1], actions[:, :-1])


def file_hashes(path):
    return {str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.rglob("*") if file.is_file()}


def test_saved_model_new_test_seed_reproducible_and_training_untouched(tmp_path):
    directory = tmp_path / "training"
    train(tiny_args(directory))
    before = file_hashes(directory)
    args = SimpleNamespace(run=directory, seed=12345, samples=41, threads=2,
                           batch_size=16, output=tmp_path / "test1")
    first = run(args)
    # Independently restore the TensorFlow snapshot and verify that the H5
    # evaluation really used the selected trained policy, including OCE state.
    gym, world, _ = build_problem(tiny_args(directory))
    checkpoint = tf.train.Checkpoint(gym=gym)
    checkpoint.restore(str(directory / "best" / f"ckpt-{first['best_epoch']}")).assert_consumed()
    fresh_test = world.clone(samples=41, seed=12345)
    expected = gym(fresh_test.tf_data, training=False)
    assert first["strategies"]["deep_hedging"]["mean_gains"] == pytest.approx(
        float(tf.reduce_mean(expected["gains"])), abs=2e-6)
    assert first["frozen_model"]["utility"] == pytest.approx(
        float(tf.reduce_mean(expected["utility"])), abs=2e-6)
    args.output = tmp_path / "test2"
    args.batch_size = 9
    second = run(args)
    for strategy in first["strategies"]:
        for key, value in first["strategies"][strategy].items():
            assert value == pytest.approx(second["strategies"][strategy][key], abs=2e-6)
    assert first["best_epoch"] == 1
    assert first["visible_gpus"] == []
    assert file_hashes(directory) == before
    with pytest.raises(ValueError, match="already exists"):
        run(args)
    for seed in (20261014, 20261015):
        with pytest.raises(ValueError, match="differ"):
            load_run(directory, seed)
    # Detect a stale summary after an interrupted resume.
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["completed_epochs"] += 1
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="stale"):
        load_run(directory, 12345)
