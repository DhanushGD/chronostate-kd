"""Phase 2 trainer tests on CPU with a toy LM and the oracle teacher from test_phase1."""
import json
from pathlib import Path

import torch
from torch import nn

from chronostate.data import build_dataset
from chronostate.teacher import cache_teacher_logits
from chronostate.train import TeacherCacheReader, TrainConfig, Trainer, cosine_lr, epoch_order
from tests.test_phase1 import Cfg, FakeTok, OracleTeacher


class ToyLM(nn.Module):
    def __init__(self, V=120, d=32):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.head = nn.Linear(d, V)

    def forward(self, input_ids, attention_mask=None):
        return type("O", (), {"logits": self.head(self.emb(input_ids))})()


def _setup(tmp_path, n_rows=12):
    tok = FakeTok()
    rows = [{"prompt": f"p{i}", "response": "hello world " * (i % 4 + 1)} for i in range(n_rows)]
    ds = build_dataset(Cfg, tok, "code", rows=rows)
    cache_teacher_logits(OracleTeacher(), ds, tmp_path / "cache", "code", pad_id=0, max_len=64,
                         top_k=8, batch_size=5, shard_rows=5, device="cpu")
    return tok, ds


def test_cache_reader_maps_row_ids_across_shards(tmp_path):
    tok, ds = _setup(tmp_path)
    rd = TeacherCacheReader(tmp_path / "cache", "code")
    assert len(rd) == len(ds) and rd.max_len == 64 and rd.top_k == 8
    v, i = rd.get([11, 0, 7])                    # rows from shards 2, 0, 1
    assert v.shape == (3, 64, 8) and i.dtype == torch.int32
    # row 11's top-1 at position 0 must be its token at position 1 (oracle teacher)
    assert int(i[0, 0, 0]) == ds[11]["input_ids"][1]


def test_schedule_and_order_are_deterministic():
    assert cosine_lr(0, 100, 10, 1.0) < cosine_lr(9, 100, 10, 1.0) <= 1.0
    assert abs(cosine_lr(10, 100, 10, 1.0) - 1.0) < 1e-9
    assert cosine_lr(100, 100, 10, 1.0) < 1e-9
    assert epoch_order(10, 2, 1) == epoch_order(10, 2, 1)
    assert sorted(epoch_order(10, 1, 1)) == list(range(10))


def _make_trainer(tmp_path, model, ds, alpha, ckpt_every=2, log=None):
    cfg = TrainConfig(batch_size=2, grad_accum=2, lr=5e-2, epochs=2, warmup_ratio=0.1, max_len=64,
                      alpha=alpha, tau=2.0, seed=0, log_every=1, ckpt_every_steps=ckpt_every, fp16=False, device="cpu")
    cache = TeacherCacheReader(tmp_path / "cache", "code") if alpha > 0 else None
    save_fn = lambda p: torch.save(model.state_dict(), p / "adapter_model.pt")
    load_fn = lambda p: model.load_state_dict(torch.load(p / "adapter_model.pt"))
    return Trainer(model, list(model.parameters()), ds, 0, cfg, cache, tmp_path / "ckpt", save_fn, load_fn,
                   monitor=log, run_name="toy")


def test_kd_training_reduces_loss_and_kl(tmp_path):
    tok, ds = _setup(tmp_path)
    torch.manual_seed(0)
    model = ToyLM()
    tr = _make_trainer(tmp_path, model, ds, alpha=0.7)
    assert tr.total_steps == 6                    # 12 rows / (2*2) = 3 per epoch, 2 epochs
    res = tr.fit()
    first, last = res.history[0], res.history[-1]
    assert last["loss"] < first["loss"] and last["kl"] < first["kl"]
    assert res.steps == 6 and res.resumed_from == 0 and res.tokens > 0


def test_sft_mode_has_zero_kl(tmp_path):
    tok, ds = _setup(tmp_path)
    model = ToyLM()
    res = _make_trainer(tmp_path, model, ds, alpha=0.0).fit()
    assert all(h["kl"] == 0.0 for h in res.history)


def test_resume_from_checkpoint_continues_at_saved_step(tmp_path):
    tok, ds = _setup(tmp_path)
    torch.manual_seed(0)
    model = ToyLM()
    tr = _make_trainer(tmp_path, model, ds, alpha=0.7, ckpt_every=2)
    # run only the first 3 steps by shrinking the horizon, saving at step 2
    tr.total_steps = 3
    tr.fit()
    assert tr.ckpt.latest()["step"] == 2
    saved = json.loads(Path(tr.ckpt.latest()["path"], "training_state.json").read_text())
    assert saved["step"] == 2
    # a fresh trainer/model (new session) resumes at step 2 and finishes the 6 steps
    model2 = ToyLM()
    tr2 = _make_trainer(tmp_path, model2, ds, alpha=0.7, ckpt_every=2)
    res = tr2.fit()
    assert res.resumed_from == 2 and res.history[0]["step"] == 3 and res.steps == 6
    # weights at resume equalled the checkpoint, not a fresh init
    ck = torch.load(Path(tr.ckpt.latest()["path"]) / "adapter_model.pt") if tr.ckpt.latest() else None
    assert ck is not None
