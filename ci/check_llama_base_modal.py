#!/usr/bin/env python3
"""CI check: the Modal packaging of the Llama BASE adapter training.

Del ordered the second Llama adapter 2026-11-19 (both twins, like Granite).
Same standard as ci/check_llama_modal.py: CI can prove structure and parity,
not execution. The Modal TEST run is the execution proof.

Checks:
- the script compiles,
- identity guards: MODEL_ID is the base repo exactly, TWIN_ID is the
  instruct repo, no Granite strings, no token literal, token only via the
  Modal secret named huggingface-secret,
- the chat-template borrow is present (base ships no template; the Granite
  base notebook set this precedent),
- the identity assert in the script demands the base name and rejects
  "Instruct",
- Modal surface (L4, nemesis-out volume, 6h timeout, pinned stack),
- recipe parity with modal_train_llama.py (the verified instruct script):
  LoRA numbers, target modules, lr / max_length / batch shape, mixture
  EXPECTED counts, seed-42 shuffle,
- TEST_MODE wiring with the base zip names.
"""
import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "meta-llama/Llama-3.2-3B"
TWIN_ID = "meta-llama/Llama-3.2-3B-Instruct"


def stage(msg):
    print(f"[check] {msg}", flush=True)


script_path = ROOT / "modal_train_llama_base.py"
src = script_path.read_text()
py_compile.compile(str(script_path), doraise=True)
stage("compiles OK")

# --- identity + security guards ---
assert f'MODEL_ID = "{MODEL_ID}"' in src, "base MODEL_ID drifted"
assert f'TWIN_ID = "{TWIN_ID}"' in src, "TWIN_ID drifted"
assert "ibm-granite" not in src, "Granite string leaked into the script"
assert not re.search(r"hf_[A-Za-z0-9]{20,}", src), "token literal in the script"
assert 'modal.Secret.from_name("huggingface-secret")' in src, "token must come from the Modal secret"
assert 'os.environ.get("HF_TOKEN"' in src, "token read must be from the environment"
stage("identity + token guards OK")

# --- base-specific pieces ---
for frag in ("if not tok.chat_template:",
             "twin_tok = AutoTokenizer.from_pretrained(TWIN_ID)",
             "tok.chat_template = twin_tok.chat_template",
             'acfg["base_model_name_or_path"] == "meta-llama/Llama-3.2-3B"',
             '"Instruct" not in acfg["base_model_name_or_path"]',
             'modal.App("nemesis-llama-base"',
             "nemesis-llama-base-lora"):
    assert frag in src, f"base piece missing: {frag}"
stage("base identity + template borrow OK")

# --- Modal surface ---
for good in ('gpu="L4"', 'modal.Volume.from_name("nemesis-out"',
             '"/out": volume', "timeout=6 * 60 * 60", "volume.commit()",
             '"transformers==4.56.2"', '"peft==0.17.1"', '"trl==0.19.1"',
             '"accelerate==1.10.1"', '"datasets==4.0.0"'):
    assert good in src, f"Modal piece missing: {good}"
stage("modal surface OK")

# --- recipe parity with the verified instruct Modal script ---
ref = (ROOT / "modal_train_llama.py").read_text()


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
    a, b = grab(src, pattern), grab(ref, pattern)
    assert a == b, f"parity drift on {label}: base {a} != instruct {b}"
    stage(f"parity {label} = {a}")
for mod in ("q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"):
    assert f'"{mod}"' in src and f'"{mod}"' in ref, f"target module drift: {mod}"
stage("parity target modules OK")
for frag in ('"contracts": 2000', '"dossiers": 720', '"pursuit": 1640',
             '"town": 2880', "random.Random(42).shuffle(rows)",
             "assert len(rows) == 7240"):
    assert frag in src and frag in ref, f"mixture drift: {frag}"
stage("parity mixture OK")

# --- TEST_MODE wiring ---
for frag in ("rows = rows[:256]", "1 if test_mode else 2",
             "1 if test_mode else 20", '"no" if test_mode else "steps"',
             "nemesis_lora_llama_base_modal_test.zip",
             "nemesis_lora_llama_base.zip",
             "def main(test: bool = True)"):
    assert frag in src, f"TEST_MODE wiring missing: {frag}"
stage("TEST_MODE wiring OK")
stage("llama BASE Modal script: ALL OK (structure + parity; execution is "
      "proven by the Modal test run, not by CI)")
