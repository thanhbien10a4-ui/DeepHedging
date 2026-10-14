# CPU validation — 2026-10-14

## Saved-model evaluation addition

The new evaluator checks passed on the same Linux/Python 3.12 environment:
**4 passed, 2 dependency deprecation warnings in 18.07 s**. They cover:

- Exact fractional 5% tail mass, cash translation of OCE, and agreement with
  independent optimization of the upstream TensorFlow utility.
- Settlement/cost agreement with gym actions in BS/feedforward and
  stochastic/recurrent worlds; delta decisions do not use future outcomes.
- H5 inference agrees with a separately restored selected TensorFlow snapshot.
  Different inference batch sizes, including partial final batches, agree.
- All training files remain byte-identical after evaluation. Existing output
  directories, train/validation seeds, and stale exports after resume are rejected.

A separate CLI process evaluated the existing **3-epoch smoke checkpoint** on
10,000 fresh stochastic paths (seed 20261016, 10 steps), with 2 CPU threads and
no visible GPUs. It completed in 3.53 seconds inside `run()`; import time is
excluded. This verifies execution, not quality of a fully trained model or
speed on another computer. The user's 300-epoch Windows weights are local
to their machine and were not evaluated here.

Evidence: [evaluation-pytest.txt](validation/2026-10-14/evaluation-pytest.txt)
and [evaluation-smoke-summary.json](validation/2026-10-14/evaluation-smoke-summary.json).
The eight source files hashed in existing training manifests are unchanged.
The report documents the upstream simulator's stochastic payoff maturity offset
and cross-sectional normalization; it does not report iid confidence intervals.

## Final recheck after workspace recovery

The published CPU implementation was retested: **6 passed, 2 dependency
deprecation warnings in 49.12 s**. A fresh CLI smoke run completed 2 epochs
and 8 Adam steps; a separate Python process resumed it to 3 epochs and
12 Adam steps with no visible GPUs. `pip check` reported no broken requirements.

GitHub blob hashes were checked against all 12 added/modified delivery files.
The other 44 upstream files match the source snapshot, allowing only cleared
notebook outputs/execution counts. Raw recheck evidence is under
[validation/2026-10-14](validation/2026-10-14).

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

The original training checks did not include 800-epoch training, an independent
financial test set, Windows runtime testing, or complete execution of upstream
research notebooks. The
CPU runner selects weights on validation, whereas the upstream trainer selects
on training loss; their selected checkpoints therefore need not be identical.
Small presets change sample count/horizon and cannot establish an equal-work
speedup. Original notebooks, `trainer.py`, SANOS and experimental modules
remain upstream source rather than newly validated CPU entry points.

Reproduce the correctness checks with `python -m pytest -q` and measure on
the target machine using the commands in [CPU_GUIDE_VI.md](CPU_GUIDE_VI.md).
