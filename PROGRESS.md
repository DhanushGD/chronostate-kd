# ChronoState-KD Lite — Build Log

Running record of every step: **what** was done, **why**, how it was **verified**, and what came out.
Updated at the end of each step. The plan this follows is `ChronoState-KD Lite POC Plan.md`.

## Status

| Phase | What | GPU-h | Status | Gate result |
| --- | --- | --- | --- | --- |
| 0 | Repo scaffold + environment check | 0.5 | ✅ 2026-10-23 | T4 · SDPA mem-efficient on · NF4 ok · tokenizers identical |
| 1 | Teacher top-k logit cache | 0.7 | ✅ 2026-10-23 | aligned (top-1 agree 0.76/0.83) · teacher GSM8K 69 % · MBPP 74 % |
| 2 | Distil 2 KD adapters + SFT ablation | 0.3 (est. 1.3) | ✅ 2026-10-23 | CE and KL fall in both KD runs · 3 adapters, 35 MB each |
| 3 | Adapter store + weight jump benchmark | 0.2 | ✅ 2026-10-23 | p50 4.0 ms · p95 4.2 ms · identical output · 100× vs load_adapter |
| 4 | Markov layer + hybrid router | CPU | ✅ 2026-10-23 | β=0.7 vs 1.0: unnecessary jumps 36→18, follow-up acc 0.47→0.76, acc 0.90→0.95 |
| 5 | Eval, benchmarks, write-up | 0.7 | 🟡 engine + notebook ready, Colab run pending | — |

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

---

## Step 3.2 — Phase 3 run on Colab (2026-10-23, T4, ~10 min GPU)

**Outcome** — gate passed

| Check | Result |
| --- | --- |
| Live LoRA slots | 336 tensors, 8,798,208 params, 35.2 MB fp32; student + adapter = 0.49 GB VRAM |
| Equivalence (fp32 store) vs independent cold `PeftModel.from_pretrained` | **3/3 identical**, 48 greedy tokens each (2 reasoning, 1 code prompt) |
| CUDA memory allocated inside the jump loop | **0 bytes** on all 400 jumps |
| Store load time | 0.17–0.23 s per adapter from local disk |

| Swap path | Bytes | p50 | p95 | p99 | max | fetch / copy / sync (p50) | Throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **In-place `copy_()`, fp32 pinned store** | 35.2 MB | **4.03 ms** | **4.24 ms** | 4.40 ms | 5.45 ms | 0.001 / 2.21 / 1.74 ms | 8.7 GB/s |
| In-place `copy_()`, fp16 pinned store | 17.6 MB | 6.45 ms | 7.29 ms | 10.3 ms | 10.8 ms | 0.001 / 6.41 / 0.03 ms | 2.7 GB/s |
| Naive `load_adapter` + `set_adapter`, local disk | 35.2 MB | 404 ms | — | — | 420 ms | — | — |

Target p95 < 45 ms: **PASS with 10× margin**. Speed-up over the naive path: **100×** (fp32 store).

**What the numbers mean**

- 8.7 GB/s is close to what PCIe 3 x16 delivers in practice on Colab; the swap is DMA-bound, not code-bound. Fetch is a dict lookup (≈ 1 µs); in production a Redis GET + deserialise would add roughly 1–5 ms for 35 MB on localhost, still far under target.
- The fp32 breakdown (`copy` 2.2 ms returns early, `sync` 1.7 ms absorbs the DMA) is the signature of asynchronous pinned transfers working as intended.
- **Surprise, kept as measured:** the fp16 store is *slower* although it moves half the bytes. Its `sync` is ~0, i.e. the copies ran synchronously: a cross-dtype `copy_()` (fp16 host → fp32 device) cannot be a plain DMA, PyTorch converts on the way and that path leaves the async pinned fast lane. Rule for the write-up: **store dtype must equal the live-slot dtype.** To bank the fp16 saving one would cast the LoRA slots themselves to fp16 (then 17.6 MB same-dtype ≈ 2 ms); not needed at 4 ms, recorded as an option.
- Base weights never change across a jump, so the KV cache of a running conversation stays valid; Phase 5 exercises that path end to end.

**Artefacts:** `results/phase3_jump_benchmark.json`, `results/phase3_jump_latency.png`; 400 `jump` events + 2 `jump_benchmark` + `naive_swap_benchmark` in `events.jsonl`.

**Next:** Step 4.1 — `chronostate/router.py`, `chronostate/routing_eval.py`, `notebooks/04_router.ipynb` (CPU).

---

## Step 4.1 — Phase 4 code: Markov router + routing eval (2026-10-23, local, CPU)

**What we did**

- `chronostate/router.py` — `HybridRouter`: `score_j = β·softmax(cos/τ)_j + (1−β)·T[prev, j]` over the `MarkovStateLayer` from Step 0.1; jump when argmax ≠ previous state; every decision is a `RouteDecision` record (raw cosines, semantic probabilities, prior, combined, chosen, jumped, latency) so routing can be audited from the log. `build_prototypes()` mean-pools L2-normalised embeddings of ~200 prompts per domain. `make_st_embedder()` wraps bge-small on CPU.
- `chronostate/routing_eval.py` — synthetic benchmark: 40 conversations × 6 turns built from **held-out** prompts (CodeFeedback rows beyond the training set + MBPP prompts; GSM8K test 100–600), in runs of 2–4 turns per domain, with 30 % *follow-up* turns ("why?", "and if it were 5 instead?") whose label is the current topic. Metrics: accuracy, follow-up accuracy, jumps, **unnecessary jumps**, missed switches, jumps/conversation, router latency.
- `tests/test_router.py` — 7 tests with a keyword embedder: prototype normalisation, β = 1 is purely semantic, a follow-up is resolved by the prior at β = 0.7 but coin-flipped at β = 1, strong evidence still overrides the prior, `T` learns online and survives `reset()`, generator invariants, and the end-to-end claim (β = 0.7 → fewer unnecessary jumps, higher follow-up accuracy, no extra missed switches). Suite: **41 passed**.
- `notebooks/04_router.ipynb` (CPU) — pools → embedder/prototypes + cosine-gap diagnostics → β sweep {1.0, 0.9, 0.7, 0.5} → learned-`T` heatmap + one worked conversation side by side → accuracy-vs-jumps plot; saves `results/prototypes.pt` for Phase 5.
- `configs/*.yaml` — new `router.cos_temperature: 0.05`.

**Design bug found and fixed by the tests (worth telling in the write-up)**

The first version normalised the cosine vector with min-max before mixing with the prior. With K = 2 domains min-max *always* yields {1, 0}, even for a 0.016 cosine gap on "why?", so β·1 beat (1−β)·prior on every turn and the Markov layer was silently inert. The β sweep would have "shown" the prior buys nothing. Replaced with `softmax(cos / 0.05)`: a near-tie maps to ≈ 0.5/0.5 and lets the prior decide; a gap ≥ 0.15 maps to ≈ 1/0 and overrides it. The temperature is the one new hyper-parameter; the notebook prints the held-out cosine-gap distribution so it can be sanity-checked against real bge-small numbers.

**Why the eval is built this way**

- *Held-out prompts for eval, training prompts for prototypes:* otherwise the semantic classifier is tested on the exact sentences that define its prototypes.
- *Follow-up turns are the point:* a topic-carrying prompt is easy for any embedder; the Markov prior earns its place on turns that carry no topic. Reporting follow-up accuracy separately isolates that.
- *Unnecessary jumps as a first-class metric:* each one costs a swap (4 ms, cheap) **and** a turn answered by the wrong adapter (expensive). It is the number that makes the case for the prior.
- *T learned online during the sweep, per router:* the demo shows the matrix becoming stickier as it sees real conversations; the heatmap is the visual for the post.

**Expected numbers (toy embedder, for orientation only):** β = 1 → accuracy ~0.86, follow-up ~0.44, unnecessary jumps 37/240; β = 0.7 → ~0.92, ~0.66, 25/240. bge-small should separate topic prompts far better than the keyword toy, so overall accuracy will be higher for both; the follow-up gap should remain.

**Next:** Step 4.2 (you, CPU runtime or local) — `notebooks/04_router.ipynb`, ~10 min.

---

## Step 4.2 — Phase 4 run (2026-10-23, Colab CPU, ~5 min)

**Outcome** — gate passed

Pools: 557 held-out code prompts, 500 held-out GSM8K test questions. Prototypes from 200 training prompts per domain (134 s to embed on CPU). `cos(code_proto, reasoning_proto) = 0.83` (bge-small embeddings are anisotropic; what matters is the gap). Held-out cosine gap own-minus-other: code median +0.134 (min +0.018, 100 % positive), reasoning median +0.123 (min +0.014, 100 % positive); follow-ups |gap| median 0.030. So topic prompts are always separable and follow-ups carry ~no signal, as designed; `cos_temperature = 0.05` puts a 0.13 gap at ≈ 0.93/0.07 and a 0.03 gap at ≈ 0.65/0.35.

240 turns, 45 follow-ups, 54 true switches:

| β | accuracy | follow-up accuracy | jumps | unnecessary | missed | jumps / conv | router ms/turn (CPU) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 (semantic only) | 0.900 | 0.467 | 84 | 36 | 6 | 2.10 | 53 |
| 0.9 | 0.904 | 0.489 | 82 | 34 | 6 | 2.05 | 55 |
| **0.7 (default)** | **0.954** | **0.756** | **69** | **18** | **3** | **1.73** | 56 |
| 0.5 | 0.971 | 0.867 | 64 | 12 | 2 | 1.60 | 59 |

Learned `T` at β = 0.7 after 240 turns: code→code 0.75, reasoning→reasoning 0.62 (less sticky than the 0.8 prior because the synthetic conversations switch every 2–4 turns; that is the EMA tracking the data).

**What the numbers mean**

- The Markov prior **halves unnecessary jumps** (36 → 18) and takes follow-up accuracy from a coin flip (0.47) to 0.76, while *reducing* missed switches (6 → 3). This is the isolated contribution of the Markov layer; everything else is identical between the two rows.
- The worked conversation shows the mechanism: after "Write a python function…", the follow-up "what does that give?" has sem 0.38 for code; β = 1 jumps to reasoning (wrong, and an extra swap), β = 0.7 stays on code because prior = 0.82.
- The sweep is monotone down to β = 0.5, which is better still on this set. β = 0.7 was fixed before seeing results and stays the default; β = 0.5 is reported as the better setting *on this synthetic distribution* (30 % follow-ups). Lower β trades toward stickiness, which would hurt on conversations that switch every turn.
- **Router latency is the real cost, not the swap:** 53–59 ms/turn for bge-small on the Colab CPU vs 4 ms for the weight jump. Phase 5 runs the embedder on the GPU (expected ≈ 3–6 ms).

**Artefacts:** `results/phase4_routing.json`, `results/prototypes.pt` (loaded by Phase 5), `results/phase4_transition_matrix.png`, `results/phase4_beta_sweep.png`; `routing_eval` events in `events.jsonl`.

**Design correction carried into Phase 5.** The plan states that the KV cache stays valid across a jump because the base weights are frozen. That holds only if LoRA leaves `k_proj`/`v_proj` untouched; ours adapts them. After a jump the cached K/V of *earlier* turns were computed with the previous adapter's k/v deltas, so cache reuse is an approximation, not an identity. Phase 5 therefore implements both `kv_mode = "reuse"` and `"recompute"` and **measures** the difference (time-to-first-token saved vs. answer agreement) instead of asserting it. Two exact alternatives exist for a production system: exclude k/v from the LoRA targets, or recompute only the turns since the last jump.

**Next:** Step 5.1 — `chronostate/engine.py`, `notebooks/05_eval_and_bench.ipynb`.

---

## Step 5.1 — Phase 5 code: serving engine + final evaluation (2026-10-23, local, CPU)

**What we did**

- `chronostate/engine.py` — `ChronoStateEngine`: per turn, route → (jump through the jumper if needed) → append the turn in ChatML **on token ids** → `generate()` with the conversation's `DynamicCache`. Two modes: `kv_mode="reuse"` keeps the cache across turns and jumps (prefills only the new tokens); `"recompute"` starts a fresh cache every turn (exact, full prefill). Measures TTFT separately from decode throughput, and logs a `turn` event with router ms, jump ms, cache length, TTFT, tokens/s and peak VRAM. `session_summary()` and `transcript_markdown()` produce the numbers and the demo file.
- `tests/test_engine.py` — 3 tests with a fake tokenizer and a fake HF-style model that enforces the prefix invariant (`cache length ≤ input length`) and counts prefills: reuse mode prefills only the new user turn after the first, recompute mode prefills everything every turn, jumps are dispatched exactly on topic switches, the summary/transcript are right. Suite: **44 passed**.
- `notebooks/05_eval_and_bench.ipynb` — (2) quality: base / SFT / KD on GSM8K-100 and base / KD on MBPP-50 with the Phase 1 harness, variants switched via the jumper, plus a `####`-reached check per variant; (3) 6-turn session with 2 topic switches in both kv modes, with per-turn TTFT and an answer-agreement comparison; (4) SDPA vs eager decode tok/s; (5) demo transcript; (6) `results/final_results.json` + `results/results_table.md` assembled from the Phase 1–5 artefacts.

**Why**

- *Ids, not text, as the conversation state:* re-tokenising generated text is not guaranteed to reproduce the generated ids, which would silently break the cache prefix. Keeping the id list and feeding the full ids plus the cache lets `transformers` prefill exactly the uncached tail.
- *Reuse vs recompute, measured:* see the Step 4.2 correction. The honest claim is "reuse saves X ms of TTFT after a jump and the answers agree in N/6 turns (mean similarity S)", not "the cache stays valid".
- *Variants via the jumper, base via `disable_adapter()`:* one loaded model for the whole eval; no reloads, and the eval itself exercises the swap path.
- *`####` check:* separates "KD reasons worse" from "KD writes longer and ran out of tokens", which Phase 2's smoke test flagged as a risk.
- *Embedder on GPU:* Phase 4 measured 55 ms/turn on CPU, 13× the swap; on the T4 it should be a few ms, making the router cost negligible next to generation.

**How it was verified**

```
python -m pytest tests/ -q      # 44 passed
```

**Next:** Step 5.2 (you, Colab) — `notebooks/05_eval_and_bench.ipynb`, ~25 min. Then Step 5.3: write-up (README results, LinkedIn post) from `results/`.
