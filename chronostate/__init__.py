"""ChronoState-KD Lite: KD-distilled LoRA adapters, hot-swapped under a Markov routing prior.

Modules fill in phase by phase (see PROGRESS.md):
  Phase 0  kd_loss, markov, monitor, ckpt, config   (CPU-testable, written first)
  Phase 1  data, teacher
  Phase 2  train
  Phase 3  store, jump
  Phase 4  router
  Phase 5  engine, eval
"""

__version__ = "0.1.0"
