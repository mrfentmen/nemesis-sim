#!/usr/bin/env python3
"""CI check: exercise the TOWN-ONLY Kaggle notebook's code paths on CPU.

Del's rule (2026-11-17): test every piece BEFORE it goes into a notebook.
The eval-only run was cancelled before its tuned town stage, so this
notebook runs that one stage alone: base model + the v5 adapter attached
in place, the SAME 24 town episodes the base model ran (seeds
20261117/20261118 x 6 scenarios x observe/enforce, prompt "neutral").

What runs here, on a shrunken Granite with the REAL tokenizer:
- the pinned stack imports together,
- the notebook's structure (compiles, no training code, no choice or
  pursuit code, pins, the adapter glob, incremental save wiring),
- the notebook's town helper cell EXECUTED as written: run_town_eval in
  full (all 24 episodes) against a dummy-adapter tiny model, checking the
  totals AND the per-episode rows (attempted/blocked/completed
  violations, invalid_replies, safe_task_success) — the aggregation reads
  run_episode's metrics dict, and a key-name mismatch there is exactly
  what this stage exists to catch,
- a dummy LoRA adapter save/load round trip through
  PeftModel.from_pretrained (the notebook's attach path),
- save_results round trip.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "ibm-granite/granite-4.1-3b"
NB_PATH = ROOT / "nemesis_town_only.ipynb"


def stage(msg):
    print(f"[check] {msg}", flush=True)


# --- stage 1: the pinned stack imports together (notebook smoke check) ---
import transformers
import peft
from transformers import HybridCache  # noqa: F401  (the v3 failure seam)

stage(f"stack: {transformers.__version__} {peft.__version__}")

# --- stage 2: notebook structure ---
nb = json.loads(NB_PATH.read_text())
code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
for i, src in enumerate(code_cells):
    compile(src, f"cell{i}", "exec")
full = "\n".join(code_cells)
assert "SFTTrainer" not in full, "town-only notebook must not train"
assert "train_harmful" not in full, "town-only notebook must not build the mixture"
assert "run_choice_suites" not in full, "town-only notebook must not run choice suites"
assert "run_pursuit" not in full, "town-only notebook must not run pursuit"
setup = code_cells[0]
assert setup.index("expandable_segments:True") < setup.index("import torch"), \
    "expandable_segments must be set before torch is imported"
assert 'rglob("adapter_config.json")' in full, "adapter glob missing"
for pin in ("transformers==4.56.2", "peft==0.17.1", "accelerate==1.10.1"):
    assert pin in full, f"pin missing: {pin}"
assert "PeftModel.from_pretrained" in full, "adapter attach missing"
assert full.count("save_results(results)") >= 3, "incremental saves not wired"
stage(f"notebook structure OK ({len(code_cells)} code cells compile)")

# --- stage 3: the adapter glob logic, on a synthetic /kaggle/input tree ---
fake_root = Path(tempfile.mkdtemp(prefix="nemesis-input-"))
adapter_home = fake_root / "nemesis-finetune-eval-v5" / "nemesis-lora"
adapter_home.mkdir(parents=True)
(adapter_home / "adapter_config.json").write_text("{}")
found = sorted(fake_root.rglob("adapter_config.json"))
assert found and found[0].parent.name == "nemesis-lora", found
stage("adapter glob finds nemesis-lora/adapter_config.json")

# --- stage 4: real tokenizer + tiny Granite from the real config ---
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
cfg_real = AutoConfig.from_pretrained(MODEL_ID)
cfg_real.num_hidden_layers = 2
cfg_real.hidden_size = 256
cfg_real.intermediate_size = 512
cfg_real.num_attention_heads = 4
cfg_real.num_key_value_heads = 2
tiny = AutoModelForCausalLM.from_config(cfg_real).eval()
stage(f"tiny model params: {sum(p.numel() for p in tiny.parameters()):,}")

# --- stage 5: exec the notebook's town helper cell as written ---
sys.path.insert(0, str(ROOT / "src"))
ns = {"elapsed": lambda: "0.0 min"}
helper_cells = [src for src in code_cells if "def run_town_eval" in src]
assert len(helper_cells) == 1, f"expected 1 town helper cell, got {len(helper_cells)}"
for src in helper_cells:
    exec(compile(src, "notebook-cell", "exec"), ns)
stage("town helper cell exec clean")

# save_results round trip (path redirected to a temp file)
ns["RESULTS_PATH"] = Path(tempfile.mkdtemp()) / "town_results.json"
ns["save_results"]({"tuned": {"town": {"episodes": 24}}})
back = json.loads(ns["RESULTS_PATH"].read_text())
assert back["tuned"]["town"]["episodes"] == 24
stage("save_results round trip OK")

# --- stage 6: dummy LoRA adapter save -> PeftModel load -> generate ---
from peft import LoraConfig, PeftModel, get_peft_model

lcfg = LoraConfig(r=4, lora_alpha=8, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "v_proj"])
dummy_dir = Path(tempfile.mkdtemp(prefix="nemesis-adapter-")) / "nemesis-lora"
get_peft_model(tiny, lcfg).save_pretrained(dummy_dir)
assert (dummy_dir / "adapter_config.json").exists()
fresh = AutoModelForCausalLM.from_config(cfg_real).eval()
tuned = PeftModel.from_pretrained(fresh, str(dummy_dir)).eval()
ids = tok("The target is", return_tensors="pt")
out = tuned.generate(**ids, max_new_tokens=4, do_sample=False)
assert out.shape[1] > ids["input_ids"].shape[1]
stage("adapter save/load/generate round trip OK")

# --- stage 7: run_town_eval in full (all 24 episodes) as written ---
# The notebook's helpers read the repo at Kaggle-layout paths (the
# notebook clones the repo into ./nemesis-sim). Recreate that layout
# here: run from a temp cwd where "nemesis-sim" symlinks to this checkout.
import os

link_root = Path(tempfile.mkdtemp(prefix="nemesis-cwd-"))
(link_root / "nemesis-sim").symlink_to(ROOT)
os.chdir(link_root)

progress = []
totals, rows = ns["run_town_eval"](
    tuned, tok, on_progress=lambda t, r: progress.append((dict(t), list(r))))
assert totals["episodes"] == 24, totals
for key in ("violations_attempted", "violations_completed", "safe_task_success"):
    assert key in totals, f"town totals missing {key}"
assert len(rows) == 24, f"expected 24 episode rows, got {len(rows)}"
for row in rows:
    for key in ("seed", "scenario", "guard", "attempted_violations",
                "blocked_violations", "completed_violations",
                "invalid_replies", "safe_task_success"):
        assert key in row, f"episode row missing {key}: {row}"
assert totals["violations_attempted"] == sum(r["attempted_violations"] for r in rows)
assert totals["violations_completed"] == sum(r["completed_violations"] for r in rows)
assert totals["safe_task_success"] == sum(int(r["safe_task_success"]) for r in rows)
assert progress and progress[-1][0]["episodes"] == 24, "on_progress not wired per episode"
stage(f"run_town_eval OK: {totals}")

print("CI TOWN NOTEBOOK CHECK: PASS")
