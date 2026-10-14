# CPU validation — 2026-10-14

## Scope and environment

The CPU runner was tested with Python 3.12.14 on Linux x86-64, TensorFlow
2.18.1, tf-keras 2.18.0, TensorFlow Probability 0.25.0, NumPy 1.26.4 and
cdxbasics 0.2.141. Two intra-op CPU threads were used; visible GPU list was
empty. The user's Windows laptop has not been tested remotely.

## Correctness

Initial execution: **6 passed, 2 dependency deprecation warnings, 48.15 s**.

- Four parameterized checks compare every rollout output and loss gradients
  against an independent eager transcription of the upstream equations.
  They cover Black–Scholes/stochastic worlds, feedforward/recurrent policies,
  1/7 time steps and one/two hedging instruments.
- Weighted evaluation agrees with direct full-data evaluation. Training uses
  all 17 samples in batches of 8, including the final single-sample batch.
- A 4-epoch uninterrupted run matches a 2-epoch run resumed to 4 epochs,
  including Adam moments and its 12-step counter. Stale history is trimmed,
  completed runs do not take more optimizer steps, and incompatible resume
  settings/accidental output overwrites are refused.

Output tolerances: `rtol=2e-5, atol=2e-6`; gradient tolerances:
`rtol=2e-4, atol=2e-6`. These are numerical comparisons, not a probability
estimate that all possible inputs or platforms are correct.

## Observed training run

Command:

```bash
python run_cpu.py --preset laptop --epochs 12 --threads 2 --output runs/cpu-timing
```

| Item | Observed value |
|---|---:|
| Training paths / validation paths | 4,096 / 1,024 |
| Time steps / hedge instruments | 20 / 2 |
| Batch size | 256 |
| Completed epochs | 12 |
| Adam updates | 192 |
| Best validation epoch | 12 |
| Best validation loss | 0.12991440296173096 |
| Median train-epoch time after first epoch | 2.096322439 s |
| Wall time inside runner, including build/evaluation/checkpoints | 32.995446961 s |
| Visible GPUs | 0 |

These measurements were observed before the workspace session restarted.
They exclude Python/TensorFlow import time before entering `run()` and are
not a prediction for the user's CPU. No controlled upstream-vs-CPU speedup
ratio was measured. The run proves execution and checkpoint saving, not
convergence or out-of-sample hedging quality.

`pip check` reported no broken requirements in the initial tested environment.
The original license hash was preserved and all 23 notebook source/metadata
structures were verified unchanged after clearing only outputs/counts.

## Limits

No 800-epoch training, independent financial test set, Windows runtime test,
or complete execution of upstream research notebooks was performed. The
CPU runner selects weights on validation, whereas the upstream trainer selects
on training loss; their selected checkpoints therefore need not be identical.
Small presets change sample count/horizon and cannot establish an equal-work
speedup. Original notebooks, `trainer.py`, SANOS and experimental modules
remain upstream source rather than newly validated CPU entry points.

Reproduce the correctness checks with `python -m pytest -q` and measure on
the target machine using the commands in [CPU_GUIDE_VI.md](CPU_GUIDE_VI.md).
