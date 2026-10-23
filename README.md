# ChronoState-KD Lite

**One 0.5B student, two distilled LoRA adapters, hot-swapped in 4 ms under a Markov routing prior — on a free Colab T4, in under 1 GB of VRAM.**

A proof of concept for serving several domain specialists from a single small model: adapters are distilled from a 3B teacher with top-k logit KD, kept out of VRAM in pinned host memory, and swapped into the live model's LoRA slots in place when a hybrid semantic + Markov router detects a topic change. Everything in the results table below was measured on a free Tesla T4 and is read from logs in `results/`.

- **Build log:** [`PROGRESS.md`](PROGRESS.md) — every step, what was done, why, what came out (including the mistakes)
- **Plan:** [`ChronoState-KD Lite POC Plan.md`](ChronoState-KD%20Lite%20POC%20Plan.md)
- **Notebooks:** `notebooks/00`–`05`, one per phase, run in order on a T4 (`04` runs on CPU)
- **Code:** `chronostate/` (`pip install -e .`), 44 CPU unit tests in `tests/`

## How it works

```
OFFLINE (one time, ~1 GPU-hour)
  Qwen2.5-3B-Instruct (NF4) ──forward only──▶ top-64 logits / token ──▶ safetensors shards on Drive
                                                                              │
  Qwen2.5-0.5B-Instruct (NF4) + LoRA r=16 ◀── (1-α)·CE + α·τ²·KL ──────────────┘
         │  adapter-code, adapter-reasoning  (+ adapter-reasoning-sft, α=0, for the ablation)
         ▼
  AdapterStore: pinned CPU tensors, 35 MB each, out of VRAM

ONLINE (every turn)
  user turn ──▶ HybridRouter: β·softmax(cos(E_q,E_j)/τ) + (1-β)·T[prev, j]
                    │ argmax ≠ prev?
                    ▼
               WeightJumper: live_lora[k].copy_(store[name][k])   ← 4 ms, no allocation
                    │
                    ▼
               student.generate(..., past_key_values=cache)      ← KV cache kept across the jump
                    │
                    ▼
               MarkovStateLayer.update(chosen)                   ← T learns online (EMA)
```

| Component | Implementation |
| --- | --- |
| Teacher / student | `Qwen/Qwen2.5-3B-Instruct` → `Qwen/Qwen2.5-0.5B-Instruct`, both NF4 (`bitsandbytes`), identical 151k tokenizer |
| KD loss | `(1-α)·CE + α·τ²·KL(softmax(t/τ) ‖ softmax(s/τ))` over the teacher's cached top-64 indices; α = 0.7, τ = 2 — `chronostate/kd_loss.py` |
| Adapters | LoRA r=16 on q/k/v/o/gate/up/down, 8.8 M params (1.75 %), 35 MB fp32 — `chronostate/train.py` |
| Store | `safetensors` → pinned host tensors keyed by name — `chronostate/store.py` |
| Swap | in-place `copy_()` into the attached PEFT adapter's tensors, fetch/copy/sync timed — `chronostate/jump.py` |
| Router | `bge-small-en-v1.5` prototypes + 2×2 Markov transition matrix, β = 0.7, EMA 0.1 — `chronostate/router.py`, `chronostate/markov.py` |
| Engine | multi-turn ChatML on token ids, `DynamicCache` carried across turns and jumps — `chronostate/engine.py` |
| Attention | PyTorch SDPA (memory-efficient kernel). FlashAttention-2 needs Ampere; the T4 is Turing. |
| Monitoring | one JSON line per event in `results/events.jsonl`: train steps, shards, jumps, routing decisions, turns |

## Results (Tesla T4, free Colab)

| Metric | Result | Target |
| --- | --- | --- |
| **Adapter swap** p50 / p95, 35 MB, 200 alternating jumps | **4.0 / 4.2 ms** (8.7 GB/s, 0 bytes allocated) | p95 < 45 ms ✅ |
| vs naive `load_adapter` + `set_adapter` from local disk | 404 ms → **100×** | — |
| Swapped weights vs independent cold load | identical greedy output, 3/3 prompts | ✅ |
| **Routing** β = 1 → β = 0.7, 240 held-out turns | unnecessary jumps **36 → 18**, follow-up accuracy **0.47 → 0.76**, overall 0.90 → 0.95, missed switches 6 → 3 | fewer jumps, ≥ accuracy ✅ |
| Router latency (bge-small on the T4) | 17–22 ms / turn | — |
| **Serving**, 6-turn session with 2 topic switches, peak VRAM | **0.67 GB** | < 3 GB ✅ |
| TTFT after a jump, KV cache reused vs recomputed (~900 tokens) | **91 vs 273 ms** (3×) | — |
| Answer agreement, reused vs recomputed cache | identical before the first jump; after it similarity 0.95–0.99 | reported |
| Decode throughput, SDPA vs eager | 10.8 vs 10.9 tok/s — no difference at batch 1 | — |
| **Quality** GSM8K-100 exact match: base / SFT / **KD** / teacher | 0.27 / 0.30 / **0.33** / 0.69 | KD ≥ SFT + 2 pts — met numerically, not significant at n = 100 |
| MBPP-50 pass@1: base / **KD** / teacher | 0.40 / **0.42** / 0.74 | — |
| KD as % of teacher, GSM8K / MBPP | 48 % / 57 % | ≥ 85 % ❌ |
| Training per adapter (1.1–1.5k rows, seq 256, 1 epoch) | 5–6 min, 2.5 GB peak | — |
| Total GPU time, all phases | ≈ 1.7 h | ≤ 4 h |

Figures: `results/phase2_loss_curves.png` (CE and KL, KD vs SFT) · `results/phase3_jump_latency.png` · `results/phase4_beta_sweep.png` · `results/phase4_transition_matrix.png` · demo: `results/demo_transcript.md`.

## What to take from it, honestly

**The serving mechanics work with margin.** Swapping a specialist costs 4 ms and nothing in VRAM; the Markov prior removes half the swaps a pure semantic router would make, almost all of them on follow-up turns ("why?", "and if it were 5 instead?") that carry no topic signal; keeping the KV cache across a swap cuts time-to-first-token 3×.

**KV reuse across a swap is an approximation, and it is measured.** The plan claimed the cache "stays valid because base weights are frozen". That is only true if LoRA does not touch `k_proj`/`v_proj`; ours does, so cached K/V of earlier turns carry the previous adapter's deltas. Answers are bit-identical until the first swap and drift slightly after it (similarity 0.95–0.99). Two exact fixes for a production system: exclude k/v from the LoRA targets, or recompute only the turns since the last swap.

**The quality numbers are small-scale and confounded.** KD beats SFT by 3 points on GSM8K, but at n = 100 that is within noise. More importantly, the KD adapter learned the teacher's verbose style and only 17 % of its answers reached the `####` final-answer marker within 320 tokens (SFT: 99 %), so its score is mostly the "last number in the text" fallback on truncated outputs; it is probably under-measured. The 85 %-of-teacher target was missed at 48 % / 57 % — 1.5k rows, one epoch, 256-token truncation (88 % of code answers cut off) and a 0.5B student against a 3B teacher were never going to get there. `configs/full.yaml` (7B → 1.5B, 5k rows, seq 512) is the scale-up path; it is a config switch.

**SDPA vs eager made no difference** because batch-1 decoding of a 0.5B model with ≤ 1k context is not attention-bound. The spec's FlashAttention / PagedAttention line is about long-context, batched serving, which this POC does not exercise. vLLM's multi-LoRA is the production route for that regime.

**Things the tests caught before the GPU did** (see `PROGRESS.md`): a min-max normalisation in the router that silently disabled the Markov prior for K = 2; an SFT code path that crashed on dummy teacher tensors; a cross-dtype `copy_()` that is *slower* than moving twice the bytes because it leaves the async DMA path.

## Prior art

vLLM multi-LoRA, S-LoRA, LoRAX and Punica already serve and swap many adapters, and do so for batched requests. The contribution here is narrower: KD-trained adapters, a Markov routing prior that suppresses needless swaps in multi-turn chat, and a measured in-place swap path that keeps the KV cache — all demonstrated on a free GPU with every number traceable to a log line.

## Reproduce

```bash
git clone <this repo> && cd chronostate-kd
pip install torch pytest pyyaml && python -m pytest tests/ -q     # 44 tests, CPU, ~5 s
```
Then open `notebooks/00_env.ipynb` … `05_eval_and_bench.ipynb` in Colab (T4 for 00–03 and 05, CPU for 04), set `GITHUB_URL` in the first cell, Run all. Each notebook ends with a gate checklist; every phase checkpoints to Drive and resumes. Total ≈ 1.7 GPU-hours.
