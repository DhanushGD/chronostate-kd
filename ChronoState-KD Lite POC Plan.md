# ChronoState-KD Lite: POC Plan

Oct 22, 2026 · @sandy

## 1. What this POC proves

A 0.5B student, distilled from a 3B teacher, serves two domain adapters (code, math reasoning) from a single 16 GB T4, and hot-swaps between them in under 45 ms without reloading the model, guided by a Markov routing prior. Total budget: about 3 GPU-hours on free Colab or Kaggle.

Three claims, each with one measurable number:

| Claim | Metric | Target |
| --- | --- | --- |
| KD beats plain SFT at equal steps | GSM8K / MBPP accuracy, KD adapter vs SFT adapter | KD ≥ SFT + 2 pts |
| Hot-swap is cheap | Adapter jump latency p95 (fetch + copy + sync) | < 45 ms |
| Markov prior reduces churn | Routing accuracy and unnecessary jumps, β=0.7 vs β=1.0 | fewer jumps, same or better accuracy |

Deliberately out of scope, and named as future work in the post: FlashAttention-2 (needs Ampere), vLLM / PagedAttention (fights in-place patching), Redis as the store (safetensors on disk is enough to prove the path), a 7B+ teacher, and a third "general" domain. Each is a one-line swap once the pipeline exists.

## 2. System design

&#91;embedded content: ChronoState-KD Lite · offline distillation lane, online serving lane, monitoring band\]

The offline lane runs once and produces two adapter files. The online lane never reloads the model: on a topic shift the router triggers a weight jump that overwrites the live LoRA tensors in place, and because the base weights are frozen the KV cache stays valid across the jump.

| Component | Implementation | Why it is here |
| --- | --- | --- |
| Teacher | Qwen2.5-3B-Instruct, NF4 4-bit, `torch.no_grad()` | Same 151k tokenizer as the student, so logits align token-for-token |
| Logit cache | Top-64 values (fp16) + indices (int32) per token, `.safetensors` shards on Drive | Full-vocab logits would be \~750 GB; top-64 is \~1 GB |
| Student | Qwen2.5-0.5B-Instruct, NF4 + LoRA r=16 on all attention and MLP projections | Fits in < 2 GB, trains an adapter in \~25 min |
| KD loss | `(1-α)·CE + α·τ²·KL(softmax(t/τ) ‖ log_softmax(s/τ))`, α=0.7, τ=2.0, `--alpha 0` = plain SFT | One flag gives the SFT ablation |
| Adapter store | `safetensors` on disk, loaded once into pinned CPU tensors keyed by adapter name | Proves the out-of-VRAM path; Redis is a backend swap |
| Router | `bge-small-en-v1.5` cosine to domain prototypes + Markov prior `T` (2×2, EMA-updated), β=0.7 | Tiny, CPU-only, deterministic |
| Weight jump | `param.copy_(staged, non_blocking=True)` over the PEFT adapter's parameters, then `cuda.synchronize()` | Never calls `load_adapter` in the loop |
| Runtime | `transformers` generate with `attn_implementation="sdpa"` and a `DynamicCache` carried across turns | FA2 needs Ampere; SDPA uses the memory-efficient kernel on T4 |
| Monitoring | W&B (or TensorBoard) for training; a JSON-lines event log + `torch.cuda` counters for serving | Every number in the results table comes from these logs |

## 3. Models, data and hardware

Everything runs on one free T4 (16 GB) or a Kaggle P100 / T4×2; peak VRAM for the whole serving stack should stay under 3 GB. Assert `teacher.tokenizer.vocab_size == student.tokenizer.vocab_size` before any distillation step.

| Role | Choice | Size / cost |
| --- | --- | --- |
| Teacher | `Qwen/Qwen2.5-3B-Instruct`, NF4 | \~2.2 GB VRAM, \~40 min to cache 3k samples |
| Student | `Qwen/Qwen2.5-0.5B-Instruct`, NF4 + LoRA r=16, α=16, dropout 0.05 | \~0.6 GB base, adapter \~9 MB |
| Router embedder | `BAAI/bge-small-en-v1.5` | 33M params, CPU |
| Code data | `m-a-p/CodeFeedback-Filtered-Instruction`, 1.5k Python-heavy rows, seq len 256 | \~25 min per adapter |
| Reasoning data | `openai/gsm8k` train, 1.5k rows with chain-of-thought, seq len 256 | \~25 min per adapter |
| Code eval | `google-research-datasets/mbpp` sanitized, first 50 problems, pass@1 | \~5 min per model |
| Reasoning eval | `openai/gsm8k` test, first 100 problems, exact match on final number | \~5 min per model |
| Router eval | 40 synthetic conversations × 6 turns with scripted topic shifts, labelled by hand | CPU, seconds |

Step up later without changing code: 3B → 7B teacher, 0.5B → 1.5B student, 1.5k → 5k rows, 256 → 512 tokens. Each roughly doubles the GPU-hours.

## 4. Repo layout and config

One GitHub repo, one notebook per phase, all notebooks import from a small `chronostate/` package so the logic is testable outside Colab. Every knob lives in one YAML so the SFT ablation and the scale-up are config diffs, not code edits.

```markdown
chronostate-kd/
  configs/
    lite.yaml          # 3B teacher, 0.5B student, 1.5k rows, alpha=0.7
    lite_sft.yaml      # same, alpha=0 (ablation)
    full.yaml          # 7B/1.5B scale-up, not run in the POC
  chronostate/
    data.py            # dataset loading, chat-template formatting, truncation
    teacher.py         # top-k logit caching to safetensors shards
    kd_loss.py         # CE + tau^2 KL over gathered top-k indices
    train.py           # HF Trainer subclass with the KD loss, checkpoint-to-Drive callback
    store.py           # AdapterStore: load safetensors -> pinned CPU dict
    jump.py            # WeightJumper: copy_() into live PEFT params + timing
    markov.py          # MarkovStateLayer (K x K row-stochastic T, EMA update)
    router.py          # HybridRouter: bge-small cosine + Markov prior
    engine.py          # multi-turn generate loop with DynamicCache carried across jumps
    monitor.py         # JSON-lines event logger + W&B hooks
  notebooks/
    00_env.ipynb  01_teacher_cache.ipynb  02_distill.ipynb
    03_store_and_jump.ipynb  04_router.ipynb  05_eval_and_bench.ipynb
  tests/               # unit tests for kd_loss, markov, jump correctness (CPU, fast)
  results/             # benchmark JSON + plots committed after each run
  requirements.txt
```

Key config values in `lite.yaml`:

| Key | Value |
| --- | --- |
| `kd.alpha` / `kd.tau` / `kd.top_k` | 0.7 / 2.0 / 64 |
| `lora.r` / `lora.alpha` / `lora.targets` | 16 / 16 / q,k,v,o,gate,up,down |
| `train.batch` / `grad_accum` / `lr` / `epochs` | 2 / 8 / 2e-4 / 1, cosine schedule, fp16 |
| `train.max_len` / `rows_per_adapter` | 256 / 1500 |
| `router.beta` / `markov.ema` | 0.7 / 0.1 |
| `ckpt.every_steps` / `ckpt.dir` | 50 / `/content/drive/MyDrive/chronostate/` |

## 5. Build plan

&#91;embedded content: Build roadmap · 6 phases, a pass/fail gate after each\]

Each phase is its own notebook and ends at a gate you can check in one cell; do not start the next phase until the gate passes. If a free-tier GPU is refused, every notebook runs unchanged on Kaggle (30 GPU-h/week).

**Phase 0 · Environment**

1. Runtime → T4; `!nvidia-smi`. Pin `transformers peft bitsandbytes accelerate datasets safetensors sentence-transformers wandb` in `requirements.txt`.
2. Confirm `torch.backends.cuda.mem_efficient_sdp_enabled()` is True.
3. Write `ckpt.py` (`save_every(step, state, path)` to Drive) and `monitor.py` (JSON-lines event logger).
4. Run `pytest tests/` on CPU: KD loss shape, Markov row-stochasticity, jump round-trip.

**Phase 1 · Teacher logit cache**

1. Load the 3B teacher in NF4 with `compute_dtype=torch.float16` (no bf16 on Turing).
2. Format 1.5k code + 1.5k GSM8K rows with the chat template, truncate to 256 tokens.
3. `torch.no_grad()` forward, batch 8; keep top-64 `(values fp16, indices int32)` on response tokens only.
4. One `.safetensors` shard per 500 rows; skip shards that already exist on Drive.
5. Run the teacher once on the eval sets (MBPP-50, GSM8K-100) and log its scores. Free the teacher.

**Phase 2 · Distillation**

1. Load the 0.5B student in NF4, `attn_implementation="sdpa"`, `prepare_model_for_kbit_training`, gradient checkpointing on.
2. `LoraConfig(r=16, lora_alpha=16, target_modules=[q,k,v,o,gate,up,down], lora_dropout=0.05)`.
3. Loss: gather student logits at the teacher's top-64 indices; `kl = KLDiv(log_softmax(s/τ), softmax(t/τ)) * τ²`; `loss = (1-α)*ce + α*kl`.
4. Train `adapter-code`, `adapter-reasoning`, then `adapter-reasoning-sft` with `--alpha 0`. Same seed, same steps.
5. Save adapters only (`model.save_pretrained`), log loss, CE, KL, grad norm, tokens/s to W&B every 10 steps.

**Phase 3 · Store and weight jump**

1. `AdapterStore`: load each `adapter_model.safetensors` into a dict of pinned CPU tensors keyed by adapter name.
2. Load the student once with one PEFT adapter attached; collect its LoRA parameters into `live[name] = param`.
3. `WeightJumper.jump(name)`: for each key, `live[k].copy_(store[name][k], non_blocking=True)`; then `torch.cuda.synchronize()`.
4. Time 200 alternating jumps; record fetch / copy / sync separately; report p50 and p95.
5. Correctness: same prompt after `jump("code")` and after a cold `PeftModel.from_pretrained`; assert identical greedy output.

**Phase 4 · Markov layer and router**

1. `MarkovStateLayer`: `T` (2×2, row-stochastic, init to a sticky prior like `[[0.8, 0.2], [0.2, 0.8]]`), `prev_state`, `update(new_state)` with EMA 0.1.
2. Domain prototypes: embed 200 prompts per domain with bge-small, mean-pool.
3. `score_j = β·cos(E_q, E_j) + (1-β)·T[prev, j]`; jump when `argmax ≠ prev`.
4. Routing eval: 40 scripted conversations, measure accuracy and jump count at β = 1.0 and β = 0.7. Save a heatmap of the learned `T`.

**Phase 5 · Eval and benchmarks**

1. Multi-turn engine: `DynamicCache` carried across turns; only LoRA tensors change on a jump.
2. Eval base student, SFT adapter, KD adapter, teacher on MBPP-50 and GSM8K-100.
3. Peak VRAM (`torch.cuda.max_memory_allocated`) and tokens/s over a 6-turn session with 2 jumps; SDPA vs eager.
4. Record a 30-second capture: Python question → math word problem, with router score, jump event and latency printed inline.
5. Commit `results/*.json` and plots; fill the results table in section 7.

## 6. Monitoring and observability

Every number in the results table is read from a log, never typed in by hand. Training goes to Weights & Biases (free tier); serving writes one JSON line per event to `results/events.jsonl`, which one notebook cell turns into the plots for the post.

| Layer | What is logged | How | Alert / check |
| --- | --- | --- | --- |
| Training | `loss`, `ce`, `kl`, `grad_norm`, `lr`, `tokens_per_s`, `vram_peak_gb` every 10 steps; eval loss every 100 | W&B run per adapter, config attached, tags `kd` / `sft` | KL not falling by step 200 → check top-k gather indices; `grad_norm` > 10 → lower lr |
| Checkpoints | step, path, wall-clock, resumed-from | `ckpt.py` writes `manifest.json` next to each save | Session died → resume from the latest manifest entry |
| Adapter store | adapter name, tensor count, bytes, load time | `store.py` log line on load | Byte size ≠ saved file size → corrupt download |
| Weight jump | `from`, `to`, `t_fetch_ms`, `t_copy_ms`, `t_sync_ms`, `t_total_ms`, `cuda_mem_before/after` | `jump.py` emits one event per jump | p95 > 45 ms, or memory delta ≠ 0 → an allocation is happening inside the loop |
| Router | turn id, `cos_scores`, `markov_prior`, `combined`, `prev`, `chosen`, `jumped`, `beta` | `router.py` event per turn | Jump rate > 50% on sticky conversations → β too high or `T` not updating |
| Serving | prompt tokens, generated tokens, `tokens_per_s`, `ttft_ms`, `vram_peak_gb`, KV-cache length | `engine.py` event per turn | VRAM growing across turns → KV cache not being reused |
| Quality | per-sample pass/fail, model variant, adapter, prompt id | `eval.py` writes `results/eval_<variant>.jsonl` | KD adapter < SFT adapter → report it; it is still a result |

Plots to generate from these logs, each one a reusable image for the post: training loss with CE and KL split (KD vs SFT overlaid), jump-latency histogram with the 45 ms line, the learned `T` heatmap, a per-turn timeline of one demo conversation showing router scores and jump markers, and a VRAM-over-turns line.

In production the same events would go to Prometheus counters and histograms (`jump_latency_ms`, `router_jumps_total`, `kv_cache_tokens`) with Grafana dashboards; say so in one line of the README, but do not build it for the POC.

## 7. Results table (fill from logs)

Publish this table as-is in the README and the post, with the real numbers, including the ones that miss their target.

| Metric | Base student | SFT adapter | KD adapter | Teacher (3B) | Target |
| --- | --- | --- | --- | --- | --- |
| MBPP-50 pass@1 (code adapter) |  |  |  |  | KD ≥ SFT + 2 pts |
| GSM8K-100 exact match (reasoning adapter) |  |  |  |  | KD ≥ SFT + 2 pts |
| KD adapter as % of teacher score |  |  |  | 100% | ≥ 85% |

| Metric | Measured | Target |
| --- | --- | --- |
| Jump latency p50 / p95 (ms), 200 jumps |  | p95 < 45 |
| Jump breakdown: fetch / copy / sync (ms) |  | copy dominates |
| Greedy output identical after jump vs cold PEFT load |  | yes |
| Routing accuracy, β=1.0 / β=0.7 |  | β=0.7 ≥ β=1.0 |
| Jumps per conversation, β=1.0 / β=0.7 |  | β=0.7 lower |
| Peak VRAM, 6-turn session with 2 jumps (GB) |  | < 3 |
| Tokens/s, SDPA vs eager |  | SDPA faster |
| Adapter file size (MB) |  | \~9 |
| Total GPU-hours used |  | ≤ 4 |

## 8. Risks, pitfalls and prior art

| Risk | Likelihood | Fallback |
| --- | --- | --- |
| Free T4 refused at peak hours | High | Kaggle (30 GPU-h/week, 9-h sessions), same notebooks |
| Session dies mid-training | High | Checkpoint every 50 steps to Drive; `train.py --resume` reads the manifest |
| KD adapter does not beat SFT on 1.5k rows | Medium | Report it; try τ=1.0 and α=0.5 once; scale rows to 3k if time allows |
| Teacher 3B too weak to show an 85% ratio meaningfully | Medium | Frame the number as a pipeline check; the 7B config is in the repo |
| Jump p95 blown by a stray allocation | Low | Pre-pin staging buffers; assert `cuda_mem_before == after` in the test |
| Routing eval too easy (2 domains) | Medium | Include 30% ambiguous prompts in the synthetic set |

Things that silently produce wrong numbers:

- Tokenizer mismatch between teacher and student makes the KL term meaningless; assert vocab equality in Phase 1.
- Student logits must be gathered at the teacher's top-k indices, not the student's own.
- `bitsandbytes` on T4 needs `compute_dtype=float16`; bfloat16 is not supported on Turing.
- Hot-patch the parameters of the adapter PEFT already attached; `load_adapter` in the loop is the slow path you are trying to beat.
- Thousands of small files on Drive are slow; shard logits and checkpoints.

Prior art to name in the post, so the contribution is stated correctly: vLLM multi-LoRA, S-LoRA, LoRAX and Punica already serve many adapters and swap them. This POC's contribution is the combination of KD-trained adapters, a Markov routing prior that suppresses needless swaps, and a measured in-place swap path that keeps the KV cache, all on a free GPU.

## 9. LinkedIn post outline

Lead with the one number people can picture (the swap latency), show the demo, then the honest ablation. Four visuals, all generated from the logs in section 6.

1. Hook (2 lines): "I hot-swap a 0.5B model's expertise in under X ms on a free Colab T4, and a Markov prior decides when."
2. The problem (3 lines): one GPU, many domains, no room for many models; existing multi-LoRA servers swap on every request.
3. What I built (4 bullets): KD-distilled adapters, out-of-VRAM adapter store, in-place weight jump that keeps the KV cache, hybrid Markov router.
4. The architecture image (section 2 diagram, exported).
5. The demo clip (30 s, Python question → math problem, jump and latency printed inline).
6. Results (the section 7 table as an image, with the jump-latency histogram).
7. What the Markov layer bought (jumps per conversation at β=1 vs β=0.7, plus the `T` heatmap).
8. Honest limits (3 lines): T4 so SDPA not FA2; 3B teacher; plain `transformers`, not vLLM; what the full config would change.
9. Prior art credit (1 line) and the repo link.

- [ ] Architecture diagram exported as PNG
- [ ] 30-second screen capture
- [ ] Results table + latency histogram image
- [ ] Transition-matrix heatmap
- [ ] README with scorecard of what ran on T4 vs deferred
- [ ] Repo public, `requirements.txt` pinned, `results/` committed
