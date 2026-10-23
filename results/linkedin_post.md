# LinkedIn post — draft

Attach, in this order: (1) the architecture diagram (export from the plan doc, section 2), (2) the 30-second screen capture of the demo session or a screenshot of `results/demo_transcript.md`, (3) `results/phase3_jump_latency.png`, (4) `results/phase4_beta_sweep.png`. Carousel or 4-image post; the first image is the one people see in the feed.

---

I hot-swap a small language model's expertise in **4 milliseconds** on a free Colab T4 — and a Markov chain decides when.

**The problem.** One GPU, many domains. You can't keep a model per domain in VRAM, and the usual multi-LoRA servers swap adapters on *every* request — fine for batched APIs, wasteful for a chat that stays on one topic for five turns and then changes.

**What I built (ChronoState-KD Lite):**
→ Two LoRA adapters (code, math reasoning) **distilled** from a 3B teacher into a 0.5B student with top-k logit KD, not plain fine-tuning
→ Adapters live **outside VRAM** in pinned host memory
→ A topic change triggers an **in-place weight jump**: `copy_()` into the live LoRA tensors — no reload, no allocation, KV cache kept
→ A **hybrid router**: semantic similarity + a 2×2 Markov transition matrix that learns how sticky conversations are

**Measured, on a free T4:**
• Swap: **4.0 ms p50 / 4.2 ms p95** for 35 MB, vs 404 ms for `load_adapter` → 100×
• Whole system: **0.67 GB** peak VRAM for a 6-turn session with 2 topic switches
• Markov prior vs pure semantic routing: unnecessary swaps **36 → 18**, accuracy on follow-up turns ("why?", "and if it were 5?") **0.47 → 0.76**
• Keeping the KV cache across a swap: time-to-first-token **91 ms vs 273 ms**
• Training: 5–6 min per adapter

**What didn't work, because that's the useful part:**
• KD beat SFT on GSM8K by 3 points (0.33 vs 0.30) — at n = 100 that's noise, and the KD adapter inherited the teacher's verbose style so most of its answers ran out of tokens before the final answer. The "85 % of teacher" target was missed by a mile (48 %). 1.5k rows, one epoch, a 0.5B student: expected.
• KV-cache reuse across a swap is an *approximation* when LoRA touches k/v projections. I measured it instead of claiming it: identical answers until the first swap, 0.95–0.99 similarity after.
• An fp16 adapter store was *slower* than fp32 — a cross-dtype `copy_()` drops off the async DMA path. Match the store dtype to the slot dtype.
• SDPA vs eager attention: no difference at batch 1 with a 0.5B model. FlashAttention is for a different regime.

**What the tests caught before the GPU did:** a normalisation bug that silently switched the Markov prior off for two domains. The β-sweep would have "proved" the prior was useless. 44 CPU unit tests, ~5 seconds, worth every one.

Prior art, credited: vLLM multi-LoRA, S-LoRA, LoRAX, Punica all swap adapters, for batched serving. The piece I'm adding is the Markov routing prior for multi-turn chat + KD-trained specialists + a measured swap path that keeps the cache — on hardware anyone can get for free.

Repo, build log with every step and every mistake, and the six Colab notebooks: <link>

#LLM #KnowledgeDistillation #LoRA #MLSystems #Inference #OpenSource

---

**Shorter variant (if the long one feels heavy):**

4 ms. That's how long it takes to swap a 0.5B model from "Python expert" to "math tutor" on a free Colab T4 — by overwriting its LoRA weights in place from pinned host memory, no reload, KV cache kept. A small Markov chain decides *when*: it halves the needless swaps a pure semantic router makes on follow-up turns like "why?".

The honest part: the distilled adapters beat plain fine-tuning by 3 points on GSM8K at n = 100, which is noise; the 85 %-of-teacher target was missed; and KV-cache reuse across a swap is an approximation I measured (0.95–0.99 answer similarity) rather than a guarantee.

Every number is read from a log. Repo + build log: <link>

#LLM #LoRA #MLSystems
