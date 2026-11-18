#!/usr/bin/env python3
"""CI check: exercise the EVAL-ONLY Kaggle notebook's code paths on CPU.

Del's rule (2026-11-17): test every piece BEFORE it goes into a notebook.
The v5 run trained fine but its eval died on CUDA OOM and, because
results.json was written only at the very end, produced nothing. This
check gates nemesis_eval_only.ipynb, whose whole design answers that
failure: one model resident, expandable_segments set before torch loads,
results saved after every stage, adapter loaded from the attached v5
output instead of retraining.

What runs here, on a shrunken Granite with the REAL tokenizer:
- the pinned stack imports together,
- the notebook's own structure (compiles, no training code, pins, the
  adapter glob, the incremental save wiring),
- the notebook's helper cells EXECUTED as written: generate_batch,
  classify, a dummy LoRA adapter save/load round trip through
  PeftModel.from_pretrained, run_pursuit_eval(limit=2), one full town
  episode through make_responder + run_episode, and save_results.

Intentional deltas (compute-only): tiny model, max_new_tokens cut on the
smoke calls, pursuit limited to 2 episodes, town limited to 1 episode
(run_town_eval itself loops all 24; its per-episode body is what runs).
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "ibm-granite/granite-4.1-3b"
NB_PATH = ROOT / "nemesis_eval_only.ipynb"


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
assert "SFTTrainer" not in full, "eval-only notebook must not train"
assert "train_harmful" not in full, "eval-only notebook must not build the mixture"
setup = code_cells[0]
assert setup.index("expandable_segments:True") < setup.index("import torch"), \
    "expandable_segments must be set before torch is imported"
assert 'rglob("adapter_config.json")' in full, "adapter glob missing"
for pin in ("transformers==4.56.2", "peft==0.17.1", "accelerate==1.10.1"):
    assert pin in full, f"pin missing: {pin}"
assert full.count("save_results(results)") >= 5, "incremental saves not wired"
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

# --- stage 5: exec the notebook's helper cells as written ---
sys.path.insert(0, str(ROOT / "src"))
ns = {"elapsed": lambda: "0.0 min"}
helper_cells = [src for src in code_cells
                if "def run_choice_suites" in src or "def run_pursuit_eval" in src]
assert len(helper_cells) == 2, f"expected 2 helper cells, got {len(helper_cells)}"
for src in helper_cells:
    exec(compile(src, "notebook-cell", "exec"), ns)
stage("helper cells exec clean")

# classify: all four buckets, exact expectations
classify = ns["classify"]
assert classify("ANSWER: A", "A") == "harmful"
assert classify("ANSWER: B", "A") == "safe"
assert classify("I can't do that.", "A") == "refusal"
assert classify("no marker here", "A") == "unparseable"
stage("classify: 4/4 buckets correct")

# generate_batch through the tiny model on real eval prompts
rows = [json.loads(l) for l in (ROOT / "nemesis/data/eval_id.jsonl").open()][:2]
replies = ns["generate_batch"](tiny, tok, [r["messages"] for r in rows],
                               max_new_tokens=8, batch=2)
assert len(replies) == 2 and all(isinstance(r, str) for r in replies)
stage("generate_batch OK on real eval_id prompts")

# save_results round trip (path redirected to a temp file)
ns["RESULTS_PATH"] = Path(tempfile.mkdtemp()) / "results.json"
ns["save_results"]({"base": {"choice": {"eval_id": {"harmful": 0.5}}}})
back = json.loads(ns["RESULTS_PATH"].read_text())
assert back["base"]["choice"]["eval_id"]["harmful"] == 0.5
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

# --- stage 7: pursuit + town code paths with a model responder ---
# The notebook's helpers read data at Kaggle-layout paths (the notebook
# clones the repo into ./nemesis-sim). Recreate that layout here: run from
# a temp cwd where "nemesis-sim" symlinks to this checkout.
import os

link_root = Path(tempfile.mkdtemp(prefix="nemesis-cwd-"))
(link_root / "nemesis-sim").symlink_to(ROOT)
os.chdir(link_root)

outcomes = ns["run_pursuit_eval"](tuned, tok, limit=2)
assert isinstance(outcomes, dict) and sum(outcomes.values()) == 2, outcomes
stage(f"run_pursuit_eval(limit=2) OK: {outcomes}")

from nemesis_sim.engine import make_episode
from nemesis_sim.harness import run_episode
from nemesis_sim.store import Store
from nemesis_sim.world import make_town

responder = ns["make_responder"](tuned, tok)
with tempfile.TemporaryDirectory() as tmp:
    ep = make_episode(make_town(20261117), "payment", seed=20261117,
                      ready=False, guard="observe", prompt="neutral")
    with Store.create(Path(tmp) / "s.sqlite", ep, {}) as store:
        m = run_episode(store, responder)
for key in ("violations_attempted", "violations_completed", "safe_task_success"):
    assert key in m, f"town metrics missing {key}"
stage(f"town episode OK: attempted={m['violations_attempted']} "
      f"completed={m['violations_completed']}")

print("CI EVAL NOTEBOOK CHECK: PASS")
