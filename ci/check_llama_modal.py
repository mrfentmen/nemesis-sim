#!/usr/bin/env python3
"""CI check: the Modal packaging of the Llama adapter training.

Del's rule (2026-11-17): test every piece BEFORE it runs. CI has no Modal
account, no GPU and no HF token, so it cannot execute the training. This
check covers what CI CAN prove, and no more:

- the script compiles,
- identity + security guards (right MODEL_ID, no Granite strings, no token
  literal, token only via the Modal secret named huggingface-secret),
- the Modal surface is present (L4 GPU, nemesis-out volume at /out,
  6-hour timeout),
- recipe parity with the Colab notebook: identical LoRA numbers and
  target modules, lr / max_length / batch shape, mixture EXPECTED counts
  and the seed-42 shuffle,
- the TEST_MODE wiring (256-row slice, epoch + logging switch, -test
  zip name), so the first Modal run is the short proof Del ordered.

What this check does NOT prove: that Modal runs it. The Modal TEST run's
own output is that evidence; the script header says so.
"""
import json
import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def stage(msg):
    print(f"[check] {msg}", flush=True)


script_path = ROOT / "modal_train_llama.py"
src = script_path.read_text()
py_compile.compile(str(script_path), doraise=True)
stage("compiles OK")

# --- identity + security guards ---
assert f'MODEL_ID = "{MODEL_ID}"' in src, "Modal script MODEL_ID drifted"
assert "ibm-granite" not in src, "Granite string leaked into the Modal script"
assert not re.search(r"hf_[A-Za-z0-9]{20,}", src), "token literal in the script"
assert 'modal.Secret.from_name("huggingface-secret")' in src, "token must come from the Modal secret"
assert 'os.environ.get("HF_TOKEN"' in src, "token read must be from the environment"
stage("identity + token guards OK")

# --- Modal surface ---
for good in ('gpu="L4"', 'modal.Volume.from_name("nemesis-out"',
             '"/out": volume', "timeout=6 * 60 * 60", "volume.commit()",
             '"transformers==4.56.2"', '"peft==0.17.1"', '"trl==0.19.1"',
             '"accelerate==1.10.1"', '"datasets==4.0.0"'):
    assert good in src, f"Modal piece missing: {good}"
stage("modal surface OK")

# --- recipe parity with the Colab notebook ---
nb = json.loads((ROOT / "nemesis_llama_adapter_colab.ipynb").read_text())
colab = "\n".join("".join(c.get("source", [])) for c in nb["cells"]
                  if c["cell_type"] == "code")


def grab(code, pattern):
    m = re.search(pattern, code)
    assert m, f"pattern not found: {pattern}"
    return m.group(1)


for label, pattern in [
    ("lora r", r"LoraConfig\(r=(\d+)"),
    ("lora alpha", r"lora_alpha=(\d+)"),
    ("lora dropout", r"lora_dropout=([\d.]+)"),
    ("learning rate", r"learning_rate=([\d.e-]+)"),
    ("max_length", r"max_length=(\d+)"),
    ("per-device batch", r"per_device_train_batch_size=(\d+)"),
    ("grad accum", r"gradient_accumulation_steps=(\d+)"),
]:
    a, b = grab(src, pattern), grab(colab, pattern)
    assert a == b, f"parity drift on {label}: modal {a} != colab {b}"
    stage(f"parity {label} = {a}")
for mod in ("q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"):
    assert f'"{mod}"' in src and f'"{mod}"' in colab, f"target module drift: {mod}"
stage("parity target modules OK")
for frag in ('"contracts": 2000', '"dossiers": 720', '"pursuit": 1640',
             '"town": 2880', "random.Random(42).shuffle(rows)",
             "assert len(rows) == 7240"):
    assert frag in src and frag in colab, f"mixture drift: {frag}"
stage("parity mixture OK")

# --- TEST_MODE wiring ---
for frag in ("rows = rows[:256]", "1 if test_mode else 2",
             "1 if test_mode else 20", '"no" if test_mode else "steps"',
             "nemesis_lora_llama_modal_test.zip", "nemesis_lora_llama.zip",
             "def main(test: bool = True)"):
    assert frag in src, f"TEST_MODE wiring missing: {frag}"
stage("TEST_MODE wiring OK")
stage("llama Modal script: ALL OK (structure + parity; execution is proven "
      "by the Modal test run, not by CI)")
