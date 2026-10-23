import json
from pathlib import Path

from chronostate.ckpt import CheckpointManager
from chronostate.config import load_config, to_dict
from chronostate.monitor import Monitor, read_events

ROOT = Path(__file__).resolve().parents[1]


def test_monitor_writes_jsonl_and_reads_back(tmp_path):
    p = tmp_path / "events.jsonl"
    with Monitor(p, run="t") as mon:
        mon.log("jump", src="code", dst="reasoning", t_total_ms=12.5)
        mon.log("train_step", step=10, loss=1.0)
    rows = read_events(p)
    assert len(rows) == 2 and rows[0]["event"] == "jump" and rows[0]["t_total_ms"] == 12.5
    assert read_events(p, event="train_step")[0]["step"] == 10


def test_checkpoint_manager_records_prunes_and_resumes(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt", every_steps=50, keep_last=2)
    assert not cm.should_save(49) and cm.should_save(50)
    for step in (50, 100, 150):
        d = tmp_path / "ckpt" / f"step-{step}"
        d.mkdir()
        (d / "adapter.bin").write_bytes(b"x")
        cm.record(step, d)
    assert cm.latest()["step"] == 150
    assert not (tmp_path / "ckpt" / "step-50").exists()          # pruned
    assert (tmp_path / "ckpt" / "step-100").exists()
    # a fresh manager (new session) resumes from the manifest
    cm2 = CheckpointManager(tmp_path / "ckpt", every_steps=50)
    assert cm2.latest()["step"] == 150
    manifest = json.loads((tmp_path / "ckpt" / "manifest.json").read_text())
    assert len(manifest["entries"]) == 2


def test_configs_load_and_differ_only_where_intended():
    lite = to_dict(load_config(ROOT / "configs/lite.yaml"))
    sft = to_dict(load_config(ROOT / "configs/lite_sft.yaml"))
    full = to_dict(load_config(ROOT / "configs/full.yaml"))
    assert lite["kd"]["alpha"] == 0.7 and sft["kd"]["alpha"] == 0.0
    sft["kd"]["alpha"] = 0.7
    assert sft == lite, "lite_sft.yaml must differ from lite.yaml only in kd.alpha"
    assert full["models"]["teacher"].endswith("7B-Instruct")
    assert lite["models"]["compute_dtype"] == "float16"           # T4 has no bf16
    assert lite["models"]["attn_implementation"] == "sdpa"
