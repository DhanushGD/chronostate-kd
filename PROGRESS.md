# ChronoState-KD Lite — Build Log

Running record of every step: **what** was done, **why**, how it was **verified**, and what came out.
Updated at the end of each step. The plan this follows is `ChronoState-KD Lite POC Plan.md`.

## Status

| Phase | What | GPU-h | Status | Gate result |
| --- | --- | --- | --- | --- |
| 0 | Repo scaffold + environment check | 0.5 | ✅ 2026-10-23 | T4 · SDPA mem-efficient on · NF4 ok · tokenizers identical |
| 1 | Teacher top-k logit cache | 0.7 | ✅ 2026-10-23 | aligned (top-1 agree 0.76/0.83) · teacher GSM8K 69 % · MBPP 74 % |
| 2 | Distil 2 KD adapters + SFT ablation | 0.3 (est. 1.3) | ✅ 2026-10-23 | CE and KL fall in both KD runs · 3 adapters, 35 MB each |
| 3 | Adapter store + weight jump benchmark | 0.3 | 🟡 code + notebook ready, Colab run pending | — |
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

## Step 0.2 — Phase 0 environment check on Colab (2026-10-23, T4)

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

- [x] T4 shown · [x] `mem_efficient_sdp_enabled` True · [x] NF4 0.5B generates text · [x] tokenisers identical · [x] pytest green on Colab · [x] lock file + env report committed

**Outcome** — gate passed

| Check | Result |
| --- | --- |
| GPU | Tesla T4, compute capability 7.5, 15.64 GB |
| Stack | Python 3.13.15 · torch 2.11.0+cu130 · transformers 5.20.1 · peft 0.21.2 · bitsandbytes 0.50.2 · accelerate 1.15.0 · datasets 4.8.5 · safetensors 0.8.0 · sentence-transformers 5.7.0 |
| Attention | `mem_efficient_sdp_enabled = True`; fp16 SDPA smoke test OK. (`flash_sdp_enabled` also reports True but the flash kernel will not dispatch on sm_75.) |
| bf16 | `torch.cuda.is_bf16_supported()` returned True — this is software emulation on torch 2.11, not Turing hardware. We keep `compute_dtype=float16` everywhere. |
| NF4 student | Qwen2.5-0.5B-Instruct loaded in 1.9 s, 0.92 GB VRAM, greedy generation correct |
| Tokenizers | teacher (3B) and student (0.5B) identical, vocab 151,665 → logit KD is valid |
| Artefacts | `requirements.lock.txt`, `results/env_report.json` committed; Drive folders created under `MyDrive/chronostate/` |

Deprecation seen: `torch.backends.cuda.sdp_kernel()` → use `torch.nn.attention.sdpa_kernel()` in later notebooks.

**Issues hit during the run**

- Cell 4 (`NF4 smoke test`) failed with `AttributeError: 'shape'` on `ids.shape`. Cause: in the `transformers` version Colab installed, `apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding` (dict-like), not a tensor. Fix: call it with `tokenize=True, return_dict=True` and pass `**enc` to `generate`. Notebook patched; the same pattern will be used in every later notebook (`data.py` / `engine.py`).

---

## Step 1.1 — Phase 1 code: datasets, teacher cache, evaluation (2026-10-23, local, no GPU)

**What we did**

- `chronostate/data.py` — loads the two domains (`CodeFeedback-Filtered-Instruction` filtered to Python, `gsm8k` train), formats each row with the chat template plus a per-domain system prompt, masks prompt tokens to −100, truncates to `max_len`, and **saves the tokenised dataset to Drive** (`data/<domain>.pt`). `collate()` right-pads.
- `chronostate/teacher.py` — `cache_teacher_logits()` runs the teacher once and writes shards of 500 rows (`values` fp16 (N,256,64), `indices` int32, `lengths`, `row_ids`) as safetensors; existing shards are skipped so the step resumes. `verify_shard()` is an alignment check: the teacher's top-1 at position *t* must usually equal the label at *t+1*.
- `chronostate/evaluate.py` — GSM8K exact-match on the final number and MBPP-sanitized pass@1 (model code + dataset asserts run in a subprocess with a timeout). Both take a `generate(batch_of_messages) -> texts` callable so the same code evaluates teacher, base, SFT and KD models. `make_hf_generate()` does batched greedy decoding with left padding.
- `tests/test_phase1.py` — 7 new tests with a fake tokenizer and an "oracle" teacher whose top-1 is always the next token: formatting/masking, truncation, collate alignment, shard round-trip + resume, alignment check (> 99 % on the oracle), answer/code extraction and sandboxed execution. Suite: **23 passed**.
- `notebooks/01_teacher_cache.ipynb` — 6 sections: setup → build/save datasets → load 3B teacher NF4 → cache shards (resumable) + verify → teacher eval (GSM8K-100, MBPP-50) → free teacher and write summaries.

**Why**

- *Save token ids, don't re-tokenise:* the teacher cache is indexed by row position. Phase 2 loading the exact same `.pt` file turns "is row *i* of the cache the same text as row *i* of the batch?" into a file identity instead of something that silently drifts with a seed or template change.
- *Right padding everywhere in training:* position *t* of every row must line up with position *t* of its shard. (Generation uses left padding for the opposite reason — all rows must end at the same position.)
- *`verify_shard` as a hard assert in the notebook:* the only cheap, model-independent check that the cache is aligned. A strong teacher agrees with ground-truth next tokens most of the time; a misaligned cache agrees at chance (~0 %).
- *Teacher eval now, not in Phase 5:* the 3B teacher costs ~2.2 GB VRAM and a few minutes to load; Phase 5 should never need it. Its scores become the "Teacher" column of the results table.
- *Top-k over full vocab in fp32:* `topk` on fp16 logits can reorder near-ties; casting first costs nothing at this size.

**How it was verified**

```
python -m pytest tests/ -q      # 23 passed
```

**Next:** Step 1.2 (you, Colab) — run `notebooks/01_teacher_cache.ipynb`, ~40 min. Report the two `verify:` lines, the two eval dicts, peak VRAM, wall time; download `results/teacher_eval.json` and `results/teacher_cache_summary.json` and commit.

---

## Step 1.2 — Phase 1 run on Colab (2026-10-23, T4, ~20 min GPU)

**Outcome** — gate passed

| Check | code | reasoning |
| --- | --- | --- |
| Rows saved to `data/<domain>.pt` | 1,101 (399 prompts > 240 tokens skipped) | 1,500 |
| Mean length / response tokens | 245 / 127 | 204 / 110 |
| Truncated at 256 | **88 %** | 25 % |
| Shards (500 rows) | 3, 108 MB | 3, 147 MB |
| Cache throughput | ~1,150–1,350 tok/s, ~95–110 s per shard | same |
| `teacher_top1_agreement` (64 rows) | **0.759** | **0.832** |

| Teacher baseline (Qwen2.5-3B-Instruct, NF4) | Score | Time |
| --- | --- | --- |
| GSM8K-100 exact match | **0.69** | 404 s |
| MBPP-50 pass@1 | **0.74** | 64 s |

Teacher load 92 s (2.06 GB); peak VRAM during caching 3.96 GB. Artefacts: `results/teacher_eval.json`, `results/teacher_cache_summary.json`, per-sample `eval_teacher_*.jsonl` on Drive.

**What the numbers mean**

- Agreement of 0.76 / 0.83 against chance of ~1/151k: the shards are aligned with the token ids Phase 2 will train on. The KL term is meaningful.
- The teacher is not strong on GSM8K (69 %); the 85 %-of-teacher target therefore means ≥ 59 % for the KD reasoning adapter and ≥ 63 % on MBPP. Those are the bars for Phase 5.

**Known limitation introduced here**

- *Code truncation:* CodeFeedback answers are long; at `max_len 256` the student sees only the first ~125 response tokens of most code answers. This caps the code adapter's MBPP score and is identical for the KD and SFT runs, so the ablation is still fair. Scale-up fix: `max_len: 512` (already in `full.yaml`) or a short-answer code set such as `python_code_instructions_18k_alpaca`. Not re-run for the POC; GPU budget goes to the ablation instead.

**Next:** Step 2.1 — KD trainer (`chronostate/train.py`) + `notebooks/02_distill.ipynb`.

---

## Step 2.1 — Phase 2 code: KD trainer (2026-10-23, local, no GPU)

**What we did**

- `chronostate/train.py`
  - `TeacherCacheReader` — loads a domain's shards into CPU RAM (≤ 150 MB) and serves `(values, indices)` by dataset `row_id`, so batches can be shuffled freely while staying aligned with the cache.
  - `Trainer.fit()` — explicit loop: deterministic epoch order from the config seed, fp16 autocast + `GradScaler`, gradient accumulation, grad-norm clipping, linear-warmup cosine LR, `kd_loss` per micro-batch, CE/KL/grad-norm/LR/tokens-per-s/VRAM logged every 10 steps to `events.jsonl`, adapter-only checkpoint every 50 steps via `CheckpointManager`, resume from the latest checkpoint (weights + step + LR; optimizer moments are *not* restored).
  - `build_student()` — NF4 base + `prepare_model_for_kbit_training` (gradient checkpointing, non-reentrant) + LoRA r=16 on q/k/v/o/gate/up/down.
  - `train_adapter()` — Colab entry point; skips a run whose final adapter already exists; writes `adapters/<run>/train_summary.json` with the full step history.
- `kd_loss` now accepts `None` teacher tensors when α = 0 (found by the new trainer test: the SFT path was slicing a dummy 1-D tensor).
- `tests/test_train.py` — 5 tests on a toy LM with the oracle teacher: cache reader row mapping across shards, schedule/order determinism, **a 6-step KD run where both loss and KL fall**, SFT mode with KL ≡ 0, and a kill-and-resume at step 2 that finishes at step 6. Suite: **29 passed**.
- `notebooks/02_distill.ipynb` — trains `adapter-code`, `adapter-reasoning`, `adapter-reasoning-sft` in sequence (skips finished, resumes unfinished), plots CE and KL curves, smoke-tests base vs each adapter on one probe per domain, saves summaries.

**Why**

- *Explicit loop over HF `Trainer`:* three needs — per-batch teacher lookup by row id, adapter-only Drive checkpoints that resume mid-epoch, and separate CE/KL logging — are each awkward through Trainer callbacks and trivial in 120 lines.
- *Identical everything between KD and SFT:* same `.pt` data file, same seed → same batch order, same steps, LR, LoRA config; only `kd.alpha` differs (enforced by `test_configs_load_and_differ_only_where_intended`). That is what makes the "KD vs SFT at equal steps" number credible.
- *Adapter-only checkpoints:* the LoRA adapter is ~9 MB; the AdamW moments would be another ~18 MB per save to Drive. Not restoring them costs a few warm-up steps after a resume and nothing otherwise.
- *fp16 + GradScaler:* Turing has no bf16; LoRA weights stay fp32 and matmuls run in fp16 under autocast, which is the standard QLoRA recipe on a T4.

**Expected numbers (for sanity when reading the output)**

- Steps: code 1,101 rows / 16 = 69; reasoning 1,500 / 16 = 94. ~15–20 s/step → 20–30 min per run.
- KD runs: `ce` drops from ~1.5–2 to ~1.0–1.3; `kl` (τ²-scaled) drops from ~2–4 to under 1. Peak VRAM ~3–5 GB.
- SFT run: lower `ce` than the KD reasoning run at the same step is *expected*: SFT optimises CE alone. The question Phase 5 answers is which one generalises to the GSM8K test set.

**How it was verified**

```
python -m pytest tests/ -q      # 29 passed
```

**Next:** Step 2.2 (you, Colab) — `notebooks/02_distill.ipynb`, ~80 min total across the three runs; can be split over sessions.

---

## Step 2.2 — Phase 2 run on Colab (2026-10-23, T4, ~18 min GPU)

**Outcome** — gate passed

| Run | Steps | Wall | CE step 1 → end | KL step 1 → end (τ²-scaled) | Final loss | Peak VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| `adapter-code` (KD, α=0.7) | 69 | 4.8 min | 0.871 → **0.785** | 1.064 → **0.763** | 0.770 | 2.45 GB |
| `adapter-reasoning` (KD, α=0.7) | 94 | 6.1 min | 0.824 → **0.656** | 1.256 → **0.680** | 0.673 | 2.45 GB |
| `adapter-reasoning-sft` (α=0) | 94 | 6.2 min | 0.824 → **0.476** | 0 (by construction) | 0.476 | 2.30 GB |

Trainable params 8,798,208 (1.75 % of 502.8 M). Throughput 800–990 tok/s. Adapter files 35.2 MB each (fp32). No resume needed.

**What the numbers mean**

- Both KD runs reduce CE *and* KL together: the student is moving toward the teacher's distribution without sacrificing fit to the labels. Curves are noisy (step-50 bumps in all three runs = the same hard batch, since the order is seeded) because 69–94 steps is short; the trend is clear.
- SFT reaches a lower CE (0.48 vs 0.66) at equal steps, as predicted: it optimises CE only, with the full gradient budget. Lower training CE is not the question; Phase 5's GSM8K test accuracy is.
- Smoke test: the SFT adapter answers in GSM8K gold style and gets the probe right (`#### 8`). The KD adapter answers in the **teacher's** verbose style (prose + LaTeX) and was still mid-answer at 80 tokens. This is the distillation working as intended, but it means KD answers are longer; Phase 5 decodes 320 tokens and must check that `####` is reached. If KD under-performs on exact match, answer length is the first suspect, not reasoning quality.

**Corrections to earlier estimates**

- Adapter size is **35 MB** (8.8 M params × fp32), not ~9 MB. Phase 3 therefore benchmarks both an fp32 store and an fp16 store (17.6 MB; cast happens on the GPU during `copy_`).
- Training is ~4× faster than planned (short sequences, small model). GPU budget used so far ≈ 0.9 h of 3.5 h. Optional later: a second epoch or 3k rows per adapter costs ~15 min.

**Artefacts:** `adapters/{adapter-code,adapter-reasoning,adapter-reasoning-sft}/` on Drive (safetensors + `train_summary.json` with the step history); `results/phase2_train_summary.json`, `results/phase2_loss_curves.png` in the repo; `train_step` events in `events.jsonl`.

**Next:** Step 3.1 — `chronostate/store.py`, `chronostate/jump.py`, `notebooks/03_store_and_jump.ipynb`.

---

## Step 3.1 — Phase 3 code: adapter store + weight jump (2026-10-23, local, no GPU)

**What we did**

- `chronostate/store.py` — `AdapterStore`: loads each `adapter_model.safetensors` once into **pinned** host tensors keyed by adapter name; optional cast to fp16 (halves the bytes that cross PCIe; the cast back to the fp32 live slot happens inside `copy_()` on the GPU). `live_to_saved_key()` maps PEFT's live parameter names (`…lora_A.default.weight`) to the saved keys (`…lora_A.weight`). `check_same_schema()` guards against adapters trained with different ranks.
- `chronostate/jump.py` — `WeightJumper`: collects the live LoRA tensors of the attached adapter once, asserts the key set and shapes match the store, and `jump(name)` does `dst.copy_(src, non_blocking=True)` per tensor followed by one `cuda.synchronize()`. Timing is split into fetch / copy / sync, and `cuda_mem_delta` is recorded per jump to prove nothing is allocated in the loop. `benchmark()` returns p50/p95/p99, per-phase medians and effective GB/s. `greedy_ids()` + `equivalence_report()` for the correctness check.
- `tests/test_jump.py` — 5 CPU tests with a toy module that mirrors PEFT's naming: key mapping, store loading/casting, a jump that overwrites the slots **without moving their data pointers** and changes the output, schema-mismatch rejection, equivalence reporting. Suite: **34 passed**.
- `notebooks/03_store_and_jump.ipynb` — load student + one adapter as the container → fp32 and fp16 stores → correctness vs an independent cold `PeftModel.from_pretrained` on 3 prompts (hard assert) → 200-jump benchmark per store → naive `load_adapter`/`set_adapter` baseline from local disk → histogram + JSON.

**Why**

- *Pinned memory:* a pageable H2D copy goes through a driver staging buffer; a pinned one is a single DMA. On a Colab T4 (PCIe 3 x16, ~12 GB/s nominal) 35 MB should move in ~3–6 ms, 17.6 MB in half that. This is where the "< 45 ms" target either holds or does not.
- *Copy into existing slots instead of `load_adapter`:* no new modules, no new allocations, no change to the PEFT graph, and therefore no reason for the KV cache to be invalidated. The naive path is benchmarked alongside so the write-up quotes a measured speed-up, not a claimed one.
- *Equivalence against an independent load:* a jump that is fast but wrong is worthless; identical greedy token ids on 3 prompts (fp32 store) is the cheapest convincing proof. The fp16 store is benchmarked but **not** used for the correctness assert, since a cast can flip a near-tie token.
- *Baseline from local disk, not Drive:* otherwise the speed-up would mostly measure Drive latency.

**Expected numbers**

- fp32 store: p50 ~4–8 ms, p95 ~6–12 ms. fp16 store: roughly half. Fetch ≈ 0.01 ms, copy (launch of ~340 small copies) ~1–3 ms, sync carries the DMA.
- Naive `load_adapter` + `set_adapter` from local disk: ~300–1500 ms → a 50–200× speed-up.

**How it was verified**

```
python -m pytest tests/ -q      # 34 passed
```

**Next:** Step 3.2 (you, Colab) — `notebooks/03_store_and_jump.ipynb`, ~15 min.
