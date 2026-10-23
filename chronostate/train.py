"""Phase 2: distil a LoRA adapter from the cached teacher logits.

A small explicit training loop rather than an HF Trainer subclass, because the loop needs
three things Trainer makes awkward: (1) per-batch teacher tensors looked up by row id,
(2) a checkpoint manager that writes adapter-only snapshots to Drive and resumes mid-epoch,
(3) separate CE / KL logging on every step. ~120 lines is cheaper than fighting callbacks.

The loop itself (`Trainer.fit`) is model-agnostic: it only needs a callable that returns
`.logits` and a list of trainable params, so it is unit-tested on CPU with a toy LM.
`train_adapter()` is the Colab entry point that wires in the QLoRA student.

Resume semantics: adapter weights, step count and LR schedule are restored from the latest
checkpoint; the optimizer moments are not (a few warm-up steps of Adam after a resume is an
acceptable trade for never having to write 2x adapter size of optimizer state to Drive).
"""
from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from safetensors.torch import load_file

from .ckpt import CheckpointManager
from .data import collate, load_saved_dataset
from .kd_loss import kd_loss
from .teacher import load_shard


# --------------------------------------------------------------------------- teacher cache
class TeacherCacheReader:
    """All shards of one domain in CPU RAM, addressable by dataset row id (code 108 MB, reasoning 147 MB)."""

    def __init__(self, cache_dir, domain: str):
        d = Path(cache_dir) / domain
        self.files = sorted(d.glob("shard-*.safetensors"))
        if not self.files:
            raise FileNotFoundError(f"no shards in {d}")
        self.values: List[torch.Tensor] = []
        self.indices: List[torch.Tensor] = []
        self.where: Dict[int, tuple] = {}
        for si, f in enumerate(self.files):
            sd = load_shard(f)
            self.values.append(sd["values"])
            self.indices.append(sd["indices"])
            for li, rid in enumerate(sd["row_ids"].tolist()):
                self.where[rid] = (si, li)
        self.max_len = self.values[0].shape[1]
        self.top_k = self.values[0].shape[2]

    def __len__(self):
        return len(self.where)

    def get(self, row_ids: List[int]):
        v = torch.stack([self.values[self.where[r][0]][self.where[r][1]] for r in row_ids])
        i = torch.stack([self.indices[self.where[r][0]][self.where[r][1]] for r in row_ids])
        return v, i


# --------------------------------------------------------------------------- schedule
def cosine_lr(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def epoch_order(n: int, epochs: int, seed: int) -> List[int]:
    g = torch.Generator().manual_seed(seed)
    out: List[int] = []
    for e in range(epochs):
        out.extend(torch.randperm(n, generator=g).tolist())
    return out


# --------------------------------------------------------------------------- trainer
@dataclass
class TrainConfig:
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 2e-4
    epochs: int = 1
    warmup_ratio: float = 0.03
    max_len: int = 256
    alpha: float = 0.7
    tau: float = 2.0
    seed: int = 42
    log_every: int = 10
    ckpt_every_steps: int = 50
    max_grad_norm: float = 1.0
    fp16: bool = True
    device: str = "cuda"


@dataclass
class TrainResult:
    steps: int
    final_loss: float
    final_ce: float
    final_kl: float
    seconds: float
    tokens: int
    resumed_from: int
    history: List[Dict[str, float]] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        forward: Callable[..., Any],              # (input_ids, attention_mask) -> object with .logits
        params: List[torch.nn.Parameter],
        examples: List[Dict[str, Any]],
        pad_id: int,
        cfg: TrainConfig,
        cache: Optional[TeacherCacheReader],
        ckpt_dir,
        save_fn: Callable[[Path], None],          # writes the adapter into the given directory
        load_fn: Callable[[Path], None],          # restores the adapter from that directory
        monitor=None,
        run_name: str = "run",
    ):
        self.forward, self.params, self.examples = forward, params, examples
        self.pad_id, self.cfg, self.cache = pad_id, cfg, cache
        self.save_fn, self.load_fn, self.mon, self.run_name = save_fn, load_fn, monitor, run_name
        self.ckpt = CheckpointManager(ckpt_dir, every_steps=cfg.ckpt_every_steps, keep_last=2)
        if cfg.alpha > 0 and cache is None:
            raise ValueError("alpha > 0 requires a teacher cache")

        per_step = cfg.batch_size * cfg.grad_accum
        self.steps_per_epoch = math.ceil(len(examples) / per_step)
        self.total_steps = self.steps_per_epoch * cfg.epochs
        self.warmup = max(1, int(cfg.warmup_ratio * self.total_steps))
        self.order = epoch_order(len(examples), cfg.epochs, cfg.seed)
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=0.0)
        use_amp = cfg.fp16 and cfg.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp else nullcontext

    # -- resume -------------------------------------------------------------------------------
    def _resume(self) -> int:
        last = self.ckpt.latest()
        if not last:
            return 0
        self.load_fn(Path(last["path"]))
        print(f"[train:{self.run_name}] resumed from step {last['step']} ({last['path']})")
        return int(last["step"])

    def _micro_batches(self, step: int):
        per_step = self.cfg.batch_size * self.cfg.grad_accum
        start = step * per_step
        sl = self.order[start : start + per_step]
        for i in range(0, len(sl), self.cfg.batch_size):
            yield [self.examples[j] for j in sl[i : i + self.cfg.batch_size]]

    # -- main loop ----------------------------------------------------------------------------
    def fit(self) -> TrainResult:
        c = self.cfg
        start_step = self._resume()
        for g in self.opt.param_groups:
            g["lr"] = cosine_lr(start_step, self.total_steps, self.warmup, c.lr)

        t0 = time.time()
        tokens = 0
        history: List[Dict[str, float]] = []
        last = {"loss": float("nan"), "ce": float("nan"), "kl": float("nan")}
        t_log = time.time()
        tok_log = 0

        for step in range(start_step, self.total_steps):
            self.opt.zero_grad(set_to_none=True)
            acc = {"loss": 0.0, "ce": 0.0, "kl": 0.0}
            n_micro = 0
            for batch in self._micro_batches(step):
                enc = collate(batch, pad_id=self.pad_id, max_len=c.max_len)
                ids = enc["input_ids"].to(c.device)
                attn = enc["attention_mask"].to(c.device)
                labels = enc["labels"].to(c.device)
                if self.cache is not None:
                    tv, ti = self.cache.get(enc["row_id"].tolist())
                    tv, ti = tv.to(c.device, non_blocking=True), ti.to(c.device, non_blocking=True)
                else:
                    tv = ti = None
                with self.autocast():
                    logits = self.forward(input_ids=ids, attention_mask=attn).logits
                out = kd_loss(logits, labels, tv, ti, alpha=c.alpha, tau=c.tau)
                self.scaler.scale(out.loss / c.grad_accum).backward()
                acc["loss"] += out.loss.item(); acc["ce"] += out.ce.item(); acc["kl"] += out.kl.item()
                n_micro += 1
                tokens += out.n_tokens; tok_log += int(attn.sum().item())
                del logits, out

            self.scaler.unscale_(self.opt)
            gnorm = torch.nn.utils.clip_grad_norm_(self.params, c.max_grad_norm)
            self.scaler.step(self.opt)
            self.scaler.update()
            lr = cosine_lr(step + 1, self.total_steps, self.warmup, c.lr)
            for g in self.opt.param_groups:
                g["lr"] = lr

            last = {k: v / max(1, n_micro) for k, v in acc.items()}
            done = step + 1
            if done % c.log_every == 0 or done == self.total_steps or done == start_step + 1:
                dt = max(1e-6, time.time() - t_log)
                rec = {"step": done, **last, "grad_norm": float(gnorm), "lr": lr,
                       "tokens_per_s": tok_log / dt, "elapsed_s": time.time() - t0}
                if c.device.startswith("cuda") and torch.cuda.is_available():
                    rec["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
                history.append(rec)
                print(f"[train:{self.run_name}] step {done}/{self.total_steps} loss {last['loss']:.4f} "
                      f"ce {last['ce']:.4f} kl {last['kl']:.4f} gn {float(gnorm):.2f} lr {lr:.2e} "
                      f"{rec['tokens_per_s']:.0f} tok/s")
                if self.mon:
                    self.mon.log("train_step", run_name=self.run_name, **rec)
                t_log, tok_log = time.time(), 0

            if self.ckpt.should_save(done) and done < self.total_steps:
                path = Path(self.ckpt.dir) / f"step-{done}"
                path.mkdir(parents=True, exist_ok=True)
                self.save_fn(path)
                (path / "training_state.json").write_text(json.dumps({"step": done, **last}))
                self.ckpt.record(done, path, loss=last["loss"])
                if self.mon:
                    self.mon.log("checkpoint", run_name=self.run_name, step=done, path=str(path))

        return TrainResult(steps=self.total_steps, final_loss=last["loss"], final_ce=last["ce"], final_kl=last["kl"],
                           seconds=time.time() - t0, tokens=tokens, resumed_from=start_step, history=history)


# --------------------------------------------------------------------------- Colab entry point
def build_student(cfg, device_map=None):
    """QLoRA student: NF4 base + LoRA on all projections, gradient checkpointing on."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(load_in_4bit=cfg.quant.load_in_4bit, bnb_4bit_quant_type=cfg.quant.quant_type,
                             bnb_4bit_use_double_quant=cfg.quant.double_quant, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(cfg.models.student)
    model = AutoModelForCausalLM.from_pretrained(cfg.models.student, quantization_config=bnb,
                                                 attn_implementation=cfg.models.attn_implementation,
                                                 dtype=torch.float16, device_map=device_map or {"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.train.gradient_checkpointing,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
                      target_modules=list(cfg.lora.targets), bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model, tok


def train_adapter(cfg, run_name: str, domain: str, drive_root: str, monitor=None, resume: bool = True) -> Dict[str, Any]:
    """Train one adapter end to end on Colab. Returns a summary dict (also written next to the adapter)."""
    from peft import set_peft_model_state_dict

    drive = Path(drive_root)
    final_dir = drive / "adapters" / run_name
    if (final_dir / "adapter_model.safetensors").exists() and (final_dir / "train_summary.json").exists():
        print(f"[train:{run_name}] final adapter already exists at {final_dir}; skipping")
        return json.loads((final_dir / "train_summary.json").read_text())

    model, tok = build_student(cfg)
    examples = load_saved_dataset(drive / "data" / f"{domain}.pt")
    cache = TeacherCacheReader(drive / "teacher_cache", domain) if cfg.kd.alpha > 0 else None
    if cache is not None:
        assert cache.max_len == cfg.data.max_len and cache.top_k == cfg.kd.top_k, "cache/config mismatch"
        assert len(cache) == len(examples), f"cache rows {len(cache)} != dataset rows {len(examples)}"

    params = [p for p in model.parameters() if p.requires_grad]
    tcfg = TrainConfig(batch_size=cfg.train.batch_size, grad_accum=cfg.train.grad_accum, lr=float(cfg.train.lr),
                       epochs=cfg.train.epochs, warmup_ratio=cfg.train.warmup_ratio, max_len=cfg.data.max_len,
                       alpha=float(cfg.kd.alpha), tau=float(cfg.kd.tau), seed=cfg.data.seed,
                       log_every=cfg.train.log_every, ckpt_every_steps=cfg.train.ckpt_every_steps, fp16=cfg.train.fp16)

    def save_fn(path: Path):
        model.save_pretrained(str(path))

    def load_fn(path: Path):
        sd = load_file(str(path / "adapter_model.safetensors"))
        set_peft_model_state_dict(model, sd)

    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(model, params, examples, tok.pad_token_id, tcfg, cache, drive / "ckpt" / run_name,
                      save_fn, load_fn, monitor, run_name)
    res = trainer.fit()

    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    size_mb = (final_dir / "adapter_model.safetensors").stat().st_size / 1e6
    summary = {"run_name": run_name, "domain": domain, "config": cfg._path, "alpha": tcfg.alpha, "tau": tcfg.tau,
               "rows": len(examples), "steps": res.steps, "resumed_from": res.resumed_from,
               "final_loss": res.final_loss, "final_ce": res.final_ce, "final_kl": res.final_kl,
               "seconds": res.seconds, "response_tokens_seen": res.tokens, "adapter_mb": size_mb,
               "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
               "history": res.history}
    (final_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    if monitor:
        monitor.log("train_done", **{k: v for k, v in summary.items() if k != "history"})
    print(f"[train:{run_name}] done: {res.steps} steps in {res.seconds/60:.1f} min, adapter {size_mb:.1f} MB -> {final_dir}")
    return summary


def quick_generate(model, tok, prompt: str, system: str, max_new_tokens: int = 80) -> str:
    """One greedy sample for a smoke test after training."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                  return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
    model.train()
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
