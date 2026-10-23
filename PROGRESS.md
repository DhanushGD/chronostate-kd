# ChronoState-KD Lite — Build Log

Running record of every step: **what** was done, **why**, how it was **verified**, and what came out.
Updated at the end of each step. The plan this follows is `ChronoState-KD Lite POC Plan.md`.

## Status

| Phase | What | GPU-h | Status | Gate result |
| --- | --- | --- | --- | --- |
| 0 | Repo scaffold + environment check | 0.5 | 🟡 scaffold done, Colab run pending | — |
| 1 | Teacher top-k logit cache | 0.7 | ⬜ | — |
| 2 | Distil 2 KD adapters + SFT ablation | 1.3 | ⬜ | — |
| 3 | Adapter store + weight jump benchmark | 0.3 | ⬜ | — |
| 4 | Markov layer + hybrid router | CPU | ⬜ | — |
| 5 | Eval, benchmarks, write-up | 0.7 | ⬜ | — |

Legend: ⬜ not started · 🟡 in progress · ✅ gate passed · ❌ gate failed (see entry)

---

## Step 0.1 — Repo scaffold (2026-10-23, local, no GPU)

**What we did**

- Created the package layout from the plan (§4): `chronostate/` with real implementations of the four modules that need no GPU, and documented stubs for the rest so imports and file names are fixed from day one.
  - `kd_loss.py` — the distillation loss from spec §4.1 over cached top-k teacher logits, with the position shift done inside the function.
  - `markov.py` — `MarkovStateLayer`: K×K row-stochastic transition matrix `T`, sticky init, EMA update, `prior()` for the router.
  - `monitor.py` — JSON-lines event logger (+ optional W&B). Every benchmark number will be read back from `results/events.jsonl`.
  - `ckpt.py` — `CheckpointManager` with a `manifest.json` so a dead Colab session resumes from the last saved step.
  - `config.py` — YAML loader. All knobs live in `configs/`.
- Wrote `configs/lite.yaml` (the POC), `configs/lite_sft.yaml` (identical, `kd.alpha = 0` → plain SFT ablation) and `configs/full.yaml` (7B/1.5B scale-up, not run).
- Wrote unit tests in `tests/` (16 tests, CPU-only) and ran them locally.
- Wrote `notebooks/00_env.ipynb`, the Phase 0 Colab notebook.
- Added `pyproject.toml`, `requirements.txt` (unpinned on purpose; the lock file comes from the T4 run), `.gitignore`.

**Why**

- *Code outside notebooks:* Colab notebooks are hard to test and diff. Keeping the logic in a small installable package means the notebooks are thin drivers and the tricky parts (loss, Markov update, checkpoint resume) are covered by tests that run in seconds on a laptop.
- *KD loss first:* the two classic silent bugs — gathering student logits at the student's own top-k instead of the teacher's, and off-by-one between logits and labels — are both pinned by tests (`test_gathers_at_teacher_indices_not_student_topk`, `test_kl_is_zero_when_student_matches_teacher`).
- *Three configs, one code path:* the "KD vs SFT at equal steps" comparison is the most credible number in the write-up. Making it a one-key config diff guarantees nothing else changes between the two runs (`test_configs_load_and_differ_only_where_intended` enforces this).
- *EMA for the Markov update:* with ~40 eval conversations raw counts are noisy; EMA keeps rows stochastic and adapts slowly, which is the "fewer unnecessary jumps" behaviour the demo needs.
- *Unpinned requirements:* Colab ships its own torch/CUDA; pinning torch breaks the install. The lock file is produced on the T4 in Phase 0 and used from then on.

**How it was verified**

```
python -m pytest tests/ -q      # 16 passed, CPU, < 5 s
```

**Outcome**

- Scaffold written to the project folder. Tests green locally.

**Next:** Step 0.2 (you, on Colab) — see below.

---

## Step 0.2 — Phase 0 environment check on Colab (pending)

**What to do**

1. `git init`, create a public GitHub repo `chronostate-kd`, push this folder.
2. Open `notebooks/00_env.ipynb` in Colab (File → Open notebook → GitHub tab, or upload), set Runtime → T4 GPU.
3. Edit `GITHUB_URL` in the first code cell; run all cells top to bottom.
4. Commit `requirements.lock.txt` and `results/env_report.json` back to the repo.
5. Fill in the Outcome below (GPU name, torch/transformers versions, SDPA flag, test count) and flip the Phase 0 status row to ✅.

**Why**

- The whole plan rests on three facts about the free T4 that have to be checked, not assumed: it is Turing (no bf16, no FlashAttention-2), the memory-efficient SDPA kernel is available, and bitsandbytes NF4 loads with `compute_dtype=float16`. Cells 1, 3 and 4 test exactly those.
- Cell 5 asserts the teacher and student tokenisers are identical. If this fails, logit distillation is meaningless and we must change the model pair before spending any GPU time.

**Gate**

- [ ] T4 shown · [ ] `mem_efficient_sdp_enabled` True · [ ] NF4 0.5B generates text · [ ] tokenisers identical · [ ] pytest green on Colab · [ ] lock file + env report committed

**Outcome**

_(fill in after the run)_
