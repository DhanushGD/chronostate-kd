# ChronoState-KD Lite

KD-distilled LoRA adapters for a 0.5B student, hot-swapped in place under a Markov routing prior, on one free Colab T4.

- **Plan:** `ChronoState-KD Lite POC Plan.md` (architecture, phases, benchmarks)
- **Build log:** `PROGRESS.md` (what was done at each step, why, and the measured outcome)
- **Config:** `configs/lite.yaml` (POC) · `configs/lite_sft.yaml` (SFT ablation) · `configs/full.yaml` (scale-up)
- **Notebooks:** one per phase in `notebooks/`, run in order on a T4 runtime
- **Code:** `chronostate/` (installable: `pip install -e .`) · tests in `tests/`

## Run the tests locally

```bash
pip install torch pytest pyyaml      # CPU torch is enough
python -m pytest tests/ -q
```

## Results

Filled in from `results/` after Phase 5. See the results table in `PROGRESS.md`.

## Prior art

vLLM multi-LoRA, S-LoRA, LoRAX and Punica already serve and swap many adapters. The contribution here is the combination of KD-trained adapters, a Markov routing prior that suppresses needless swaps, and a measured in-place swap path that keeps the KV cache, all on a free GPU.
