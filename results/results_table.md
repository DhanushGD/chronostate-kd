| Metric | Result | Target |
| --- | --- | --- |
| GSM8K-100 exact match: base / SFT / **KD** / teacher | 0.27 / 0.30 / **0.33** / 0.69 | KD ≥ SFT + 2 pts |
| MBPP-50 pass@1: base / **KD** / teacher | 0.40 / **0.42** / 0.74 | — |
| KD as % of teacher (GSM8K / MBPP) | 47.8 % / 56.8 % | ≥ 85 % |
| Adapter swap p50 / p95 (35 MB, in place) | 4.0 / 4.2 ms | p95 < 45 ms |
| vs naive load_adapter | 404 ms → **100×** | — |
| Routing β=1 → β=0.7: unnecessary jumps / follow-up acc | 36 → 18 / 0.47 → 0.76 | fewer, ≥ acc |
| Router latency (GPU embedder) | 72.1 ms/turn | — |
| 6-turn session, 2 jumps: peak VRAM | 0.67 GB | < 3 GB |
| TTFT after a jump: KV reuse vs recompute | 91 vs 273 ms | — |
| Answer agreement reuse vs recompute (exact / mean sim) | 2/6 / 0.98 | report |
| Decode throughput SDPA / eager | 10.8 / 10.9 tok/s | SDPA ≥ eager |
| Training time per adapter | 5–6 min | — |