"""Phase 3 tests on CPU with a toy module whose parameter names mimic PEFT's LoRA layout."""
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from chronostate.jump import WeightJumper, equivalence_report
from chronostate.store import AdapterStore, live_to_saved_key


def test_live_to_saved_key_strips_adapter_name_only_for_lora():
    live = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    assert live_to_saved_key(live) == "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert live_to_saved_key(live.replace("default", "other")) is None          # a different adapter
    assert live_to_saved_key("base_model.model.model.layers.0.self_attn.q_proj.weight") is None  # base weight
    assert live_to_saved_key("x.lora_B.default.weight", "default") == "x.lora_B.weight"


class ToyLoRA(nn.Module):
    """Two 'layers', each with lora_A/lora_B under adapter 'default', plus a frozen base weight."""

    def __init__(self, r=4, d=8):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(2):
            blk = nn.Module()
            blk.base = nn.Parameter(torch.randn(d, d), requires_grad=False)
            blk.lora_A = nn.ModuleDict({"default": nn.Linear(d, r, bias=False)})
            blk.lora_B = nn.ModuleDict({"default": nn.Linear(r, d, bias=False)})
            self.layers.append(blk)

    def forward(self, x):
        for blk in self.layers:
            x = x @ blk.base.T + blk.lora_B["default"](blk.lora_A["default"](x))
        return x


def _save_adapter(model: ToyLoRA, path, scale: float):
    sd = {}
    for n, p in model.named_parameters():
        k = live_to_saved_key(n)
        if k:
            sd[k] = (torch.ones_like(p) * scale).contiguous()
    path.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path / "adapter_model.safetensors"))
    return sd


def test_store_loads_and_casts(tmp_path):
    m = ToyLoRA()
    _save_adapter(m, tmp_path / "a", 1.0)
    st32 = AdapterStore().add("a", tmp_path / "a")
    st16 = AdapterStore(dtype=torch.float16).add("a", tmp_path / "a")
    assert st32.meta["a"]["tensors"] == 4 and st32.meta["a"]["mb"] == 2 * st16.meta["a"]["mb"]
    assert next(iter(st16.get("a").values())).dtype == torch.float16


def test_jump_overwrites_live_params_without_allocation_and_changes_output(tmp_path):
    torch.manual_seed(0)
    m = ToyLoRA()
    _save_adapter(m, tmp_path / "a", 1.0)
    _save_adapter(m, tmp_path / "b", 0.5)
    store = AdapterStore().add_all({"a": tmp_path / "a", "b": tmp_path / "b"})
    j = WeightJumper(m, store)
    assert len(j.live) == 4 and j.nbytes == store.meta["a"]["mb"] * 1e6
    ptrs = {k: t.data_ptr() for k, t in j.live.items()}         # slots must not move
    x = torch.randn(3, 8)
    j.jump("a"); ya = m(x).clone()
    assert torch.all(m.layers[0].lora_A["default"].weight == 1.0)
    ev = j.jump("b"); yb = m(x)
    assert torch.all(m.layers[1].lora_B["default"].weight == 0.5)
    assert not torch.allclose(ya, yb)
    assert ev["src"] == "a" and ev["dst"] == "b" and ev["t_total_ms"] >= 0
    assert {k: t.data_ptr() for k, t in j.live.items()} == ptrs
    summ = j.benchmark(["a", "b"], trials=20, warmup=2, tag="toy")
    assert summ["trials"] == 20 and summ["p95_ms"] >= summ["p50_ms"] and len(j.events) == 22


def test_jumper_rejects_schema_mismatch(tmp_path):
    m = ToyLoRA()
    _save_adapter(m, tmp_path / "a", 1.0)
    other = ToyLoRA(r=2)
    _save_adapter(other, tmp_path / "c", 1.0)                     # same keys, different shapes
    store = AdapterStore().add_all({"a": tmp_path / "a", "c": tmp_path / "c"})
    with pytest.raises(ValueError):
        WeightJumper(m, store)


def test_equivalence_report():
    assert equivalence_report([1, 2, 3], [1, 2, 3])["identical"]
    r = equivalence_report([1, 2, 9], [1, 2, 3])
    assert not r["identical"] and r["first_diff_pos"] == 2
