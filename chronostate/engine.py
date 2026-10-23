"""Phase 5: the multi-turn serving loop that ties router, jumper and student together.

Per user turn:
    1. router.route(text)            -> chosen adapter (+ jump dispatched through the jumper)
    2. append the turn in ChatML     -> token ids of the whole conversation so far
    3. model.generate(full ids, past_key_values=cache)
         kv_mode="reuse":     the DynamicCache from earlier turns is kept, only new tokens are
                              prefilled. After a jump the cached K/V of earlier turns still carry
                              the PREVIOUS adapter's k/v LoRA deltas (our LoRA adapts k_proj and
                              v_proj), so this is an approximation. We measure it, not assume it.
         kv_mode="recompute": a fresh cache every turn -> exact, costs a full prefill.
    4. log a `turn` event: router decision, jump latency, TTFT, decode tokens/s, VRAM.

Token bookkeeping is done on ids, not on re-tokenised text, so the cached prefix is always a
true prefix of the next turn's input (re-tokenising generated text is not guaranteed to
reproduce the generated ids).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

SYSTEM_PROMPT_CHAT = ("You are a helpful assistant for Python coding and math word problems. "
                      "For math, think step by step and give the final answer after '####'. "
                      "For code, answer with correct, concise Python.")


@dataclass
class TurnRecord:
    turn: int
    user: str
    chosen: str
    jumped: bool
    jump_ms: float
    router_ms: float
    prompt_tokens_new: int
    cache_tokens_before: int
    ttft_ms: float
    gen_tokens: int
    decode_tok_s: float
    total_ms: float
    answer: str
    vram_peak_gb: float
    decision: Dict = field(default_factory=dict)


class ChronoStateEngine:
    def __init__(self, model, tok, router, jumper, monitor=None, kv_mode: str = "reuse",
                 max_new_tokens: int = 256, system_prompt: str = SYSTEM_PROMPT_CHAT, device: str = "cuda"):
        assert kv_mode in ("reuse", "recompute")
        self.model, self.tok, self.router, self.jumper, self.mon = model, tok, router, jumper, monitor
        self.kv_mode, self.max_new_tokens, self.system_prompt, self.device = kv_mode, max_new_tokens, system_prompt, device
        self._last_jump: Optional[Dict[str, float]] = None
        self.router.on_jump = self._on_jump
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")
        self.im_end = tok.convert_tokens_to_ids("<|im_end|>")
        self.nl = tok("\n", add_special_tokens=False)["input_ids"]
        self.reset()

    # -- state ---------------------------------------------------------------------------------
    def reset(self) -> None:
        from transformers import DynamicCache
        self.ids: List[int] = self._chatml("system", self.system_prompt)
        self.cache = DynamicCache()
        self.turns: List[TurnRecord] = []
        self.router.reset()

    def _chatml(self, role: str, content: str, close: bool = True) -> List[int]:
        ids = [self.im_start] + self.tok(role + "\n" + content, add_special_tokens=False)["input_ids"]
        if close:
            ids += [self.im_end] + self.nl
        return ids

    def _on_jump(self, name: str) -> None:
        self._last_jump = self.jumper.jump(name)

    # -- one turn -------------------------------------------------------------------------------
    @torch.no_grad()
    def chat(self, user_text: str) -> TurnRecord:
        from transformers import DynamicCache
        t_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self._last_jump = None
        decision = self.router.route(user_text)                      # may call _on_jump
        jump_ms = self._last_jump["t_total_ms"] if self._last_jump else 0.0

        new_ids = self._chatml("user", user_text) + [self.im_start] + self.tok("assistant\n", add_special_tokens=False)["input_ids"]
        self.ids += new_ids
        if self.kv_mode == "recompute":
            self.cache = DynamicCache()
        cache_before = self.cache.get_seq_length() if hasattr(self.cache, "get_seq_length") else 0

        input_ids = torch.tensor([self.ids], device=self.device)
        attn = torch.ones_like(input_ids)

        # TTFT: one forward for the first token, then let generate continue from the cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(input_ids=input_ids, attention_mask=attn, past_key_values=self.cache,
                                  max_new_tokens=1, do_sample=False, pad_token_id=self.tok.pad_token_id,
                                  eos_token_id=self.im_end, return_dict_in_generate=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - t0) * 1e3
        self.cache = out.past_key_values
        seq = out.sequences

        t1 = time.perf_counter()
        n_gen = 1
        if int(seq[0, -1]) != self.im_end and self.max_new_tokens > 1:
            out = self.model.generate(input_ids=seq, attention_mask=torch.ones_like(seq), past_key_values=self.cache,
                                      max_new_tokens=self.max_new_tokens - 1, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id, eos_token_id=self.im_end,
                                      return_dict_in_generate=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.cache = out.past_key_values
            n_gen = out.sequences.shape[1] - input_ids.shape[1]
            seq = out.sequences
        decode_s = time.perf_counter() - t1

        gen_ids = seq[0, input_ids.shape[1]:].tolist()
        answer = self.tok.decode([t for t in gen_ids if t != self.im_end], skip_special_tokens=True).strip()
        # keep ids consistent with what the cache has seen; close the assistant turn
        self.ids += gen_ids
        if not gen_ids or gen_ids[-1] != self.im_end:
            self.ids += [self.im_end]
        self.ids += self.nl

        rec = TurnRecord(turn=len(self.turns), user=user_text, chosen=decision.chosen, jumped=decision.jumped,
                         jump_ms=jump_ms, router_ms=decision.latency_ms, prompt_tokens_new=len(new_ids),
                         cache_tokens_before=cache_before, ttft_ms=ttft_ms, gen_tokens=n_gen,
                         decode_tok_s=(n_gen - 1) / decode_s if n_gen > 1 and decode_s > 0 else 0.0,
                         total_ms=(time.perf_counter() - t_start) * 1e3, answer=answer,
                         vram_peak_gb=torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0,
                         decision={"cos_norm": decision.cos_norm, "prior": decision.prior, "combined": decision.combined})
        self.turns.append(rec)
        if self.mon:
            self.mon.log("turn", kv_mode=self.kv_mode, **{k: v for k, v in rec.__dict__.items() if k not in ("answer", "decision")},
                         answer_chars=len(answer))
        return rec

    def run_session(self, user_turns: List[str]) -> List[TurnRecord]:
        self.reset()
        return [self.chat(u) for u in user_turns]


def session_summary(turns: List[TurnRecord]) -> Dict[str, float]:
    jumps = [t for t in turns if t.jumped and t.turn > 0]
    return {"turns": len(turns), "jumps": len(jumps),
            "jump_ms_mean": sum(t.jump_ms for t in jumps) / max(1, len(jumps)),
            "router_ms_mean": sum(t.router_ms for t in turns) / len(turns),
            "ttft_ms_mean": sum(t.ttft_ms for t in turns) / len(turns),
            "ttft_ms_after_jump_mean": sum(t.ttft_ms for t in jumps) / max(1, len(jumps)),
            "decode_tok_s_mean": sum(t.decode_tok_s for t in turns if t.decode_tok_s > 0) / max(1, sum(1 for t in turns if t.decode_tok_s > 0)),
            "gen_tokens_total": sum(t.gen_tokens for t in turns),
            "vram_peak_gb": max(t.vram_peak_gb for t in turns),
            "wall_s": sum(t.total_ms for t in turns) / 1e3}


def transcript_markdown(turns: List[TurnRecord], title: str = "Demo session") -> str:
    lines = [f"# {title}", ""]
    for t in turns:
        tag = f"**[JUMP → {t.chosen}, {t.jump_ms:.1f} ms]**" if t.jumped else f"[stay: {t.chosen}]"
        lines += [f"### Turn {t.turn + 1} — {tag}",
                  f"router {t.router_ms:.1f} ms · sem={t.decision['cos_norm'][t.chosen]:.2f} prior={t.decision['prior'][t.chosen]:.2f} · "
                  f"TTFT {t.ttft_ms:.0f} ms · {t.gen_tokens} tok @ {t.decode_tok_s:.1f} tok/s · VRAM {t.vram_peak_gb:.2f} GB", "",
                  f"**User:** {t.user}", "", f"**Assistant:** {t.answer}", ""]
    return "\n".join(lines)
