# Source provenance

- Source: https://github.com/hansbuehler/deephedging
- Source branch: `main`
- Source commit: `fe53fd9a7e402d4ba18bd5b9ec704e0361554f7c` (2026-01-22)
- Import date: 2026-10-14
- Author: Hans Buehler and the contributors identified upstream.
- License: original GPL-3.0 `LICENSE`, retained unchanged.

This repository is a source snapshot, not a GitHub fork with full upstream
history. All 47 original file paths are retained. The 23 notebooks keep their
code/markdown cell sources and metadata, but stored outputs and execution
counts are cleared. Notebook size changes from 71,170,827 to 357,411 bytes.
Original rendered results remain available in the upstream repo.

CPU changes made on 2026-10-14:

- `gym.py`: replace incremental action concatenation with TensorArray writes
  and one stack/transpose, preserving trading equations and loss gradients.
- Add an installable package, pinned environment, CPU CLI, compiled headless
  training, scheduled batched validation, full checkpoints, tests and guides.
- Add CPU setup links to README and exclude run outputs from Git.

The original `trainer.py` is retained. CPU runs use `cpu_training.py`.
Research notebooks and experimental SANOS modules are preserved, not certified
by the CPU tests. This work does not claim improved hedging performance.
