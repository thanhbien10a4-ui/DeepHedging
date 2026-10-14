import csv

import numpy as np
import pytest
import tensorflow as tf

from deephedging.cpu_training import CpuEngine, build_problem, make_dataset, run
from deephedging.run_cpu import parse_args


def args_for(path, **overrides):
    args = parse_args(["--preset", "smoke", "--output", str(path)])
    args.samples = 17
    args.val_samples = 11
    args.steps = 3
    args.batch_size = 8
    args.eval_every = 1
    args.checkpoint_every = 1
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def reference_rollout(gym, data):
    """Independent eager, list-based transcription of the upstream equations."""
    market = data["market"]
    steps = market["hedges"].shape[1]
    per_step, per_path = gym._features(data, steps)
    features0 = dict(per_path, **{key: value[:, 0, :] for key, value in per_step.items()})
    payoff = market["payoff"]
    pnl, cost = tf.zeros_like(payoff), tf.zeros_like(payoff)
    action = tf.zeros_like(market["cost"][:, 0, :])
    delta = tf.zeros_like(action)
    state = gym.agent.initial_state(features0) if gym.agent.is_recurrent else None
    initial_delta = gym.agent.initial_delta(features0) if gym.agent.has_initial_delta else tf.zeros_like(action)
    history = []
    for step in range(steps):
        live = dict(action=action, delta=delta, cost=cost, pnl=pnl)
        live.update(per_path)
        live.update({key: value[:, step, :] for key, value in per_step.items()})
        if gym.agent.is_recurrent:
            live[gym.agent.state_feature_name] = state
        action, state = gym.agent(live, training=True)
        if step == 0:
            action = action + initial_delta
        action = gym.softclip(action, market["lbnd_a"][:, step, :], market["ubnd_a"][:, step, :])
        delta = delta + action
        cost = cost + tf.reduce_sum(tf.abs(action) * market["cost"][:, step, :], axis=1)
        pnl = pnl + tf.reduce_sum(action * market["hedges"][:, step, :], axis=1)
        history.append(tf.stop_gradient(action))
    utility = gym.utility(dict(features_time_0=features0, payoff=payoff, pnl=pnl, cost=cost))
    utility0 = gym.utility0(dict(features_time_0=features0, payoff=payoff, pnl=pnl * 0., cost=cost * 0.))
    return dict(loss=-utility-utility0, utility=utility, utility0=utility0,
                payoff=payoff, pnl=pnl, cost=cost, gains=payoff+pnl-cost,
                actions=tf.stack(history, axis=1))


@pytest.mark.parametrize("world,recurrent,steps", [("bs", 0, 1), ("bs", 3, 7),
                                                   ("stochastic", 0, 7), ("stochastic", 3, 1)])
def test_rollout_values_and_gradients(tmp_path, world, recurrent, steps):
    gym, training, _ = build_problem(args_for(tmp_path, world=world, recurrent=recurrent, steps=steps))
    with tf.GradientTape() as tape:
        actual = gym(training.tf_data, training=True)
        actual_loss = tf.reduce_mean(actual["loss"])
    actual_gradients = tape.gradient(actual_loss, gym.trainable_variables)
    with tf.GradientTape() as tape:
        expected = reference_rollout(gym, training.tf_data)
        expected_loss = tf.reduce_mean(expected["loss"])
    expected_gradients = tape.gradient(expected_loss, gym.trainable_variables)
    assert actual["actions"].shape == (17, steps, training.nInst)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-5, atol=2e-6, err_msg=key)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert (actual_gradient is None) == (expected_gradient is None)
        if actual_gradient is not None:
            np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=2e-4, atol=2e-6)


def test_weighted_partial_batches(tmp_path):
    gym, world, _ = build_problem(args_for(tmp_path))
    weights = np.arange(1, world.nSamples + 1, dtype=np.float32)
    weights /= weights.sum()
    world.tf_sample_weights = tf.constant(weights[:, None])
    engine = CpuEngine(gym, tf.keras.optimizers.Adam())
    actual = engine.evaluate(make_dataset(world, 8))[0]
    expected = tf.reduce_sum(gym(world.tf_data)["loss"] * weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    before = [variable.numpy().copy() for variable in gym.trainable_variables]
    engine.train_epoch(make_dataset(world, 8, 123))
    assert int(engine.optimizer.iterations) == 3
    assert any(not np.array_equal(a, b.numpy()) for a, b in zip(before, gym.trainable_variables))


def checkpoint_arrays(directory):
    checkpoint = tf.train.latest_checkpoint(str(directory / "checkpoints"))
    reader = tf.train.load_checkpoint(checkpoint)
    return {name: reader.get_tensor(name) for name in reader.get_variable_to_shape_map()
            if name != "_CHECKPOINTABLE_OBJECT_GRAPH" and not name.startswith("save_counter/")}


def test_resume_matches_uninterrupted_adam_training(tmp_path):
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    run(args_for(full, epochs=4))
    run(args_for(resumed, epochs=2))
    with (resumed / "metrics.csv").open("a") as stream:
        stream.write("99,0,0,0,0\n")
    summary = run(args_for(resumed, epochs=4, resume=True))
    left, right = checkpoint_arrays(full), checkpoint_arrays(resumed)
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_allclose(left[key], right[key], rtol=2e-5, atol=2e-6, err_msg=key)
    assert summary["optimizer_steps"] == 12
    assert summary["visible_gpus"] == []
    with (resumed / "metrics.csv").open() as stream:
        assert [int(row["epoch"]) for row in csv.DictReader(stream)] == [1, 2, 3, 4]
    assert run(args_for(resumed, epochs=4, resume=True))["optimizer_steps"] == 12
    with pytest.raises(ValueError, match="Resume rejected"):
        run(args_for(resumed, epochs=5, resume=True, steps=4))
    with pytest.raises(ValueError, match="not empty"):
        run(args_for(resumed))
