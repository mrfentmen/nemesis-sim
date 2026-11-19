#!/usr/bin/env python3
"""CI check: the Llama adapter Kaggle TPU notebook, as far as CI can reach.

Del's rule (2026-11-17): test every piece BEFORE it goes into the
notebook. CI has no TPU and no Hugging Face token, so it cannot execute
the XLA path or download the gated weights. This check therefore covers
what CI CAN prove, and no more:

- the notebook is valid and carries the right identity (same MODEL_ID
  as the Colab notebook, no Granite strings, no token literal),
- the token is read from a Kaggle secret, never written in the text,
- the Colab-only machinery is gone (no CUDA, no google.colab, no Drive
  mount, no device_map, no fp16) and the TPU machinery is present
  (PJRT, bf16, /kaggle/working),
- recipe parity with the Colab notebook: identical LoRA numbers and
  target modules, identical lr / max_length / batch shape, identical
  mixture EXPECTED counts and seed-42 shuffle,
- the TEST_MODE pilot wiring is present (256-row slice, epoch switch),
  so the first Kaggle run is the short proof Del ordered.

What this check does NOT prove: that torch_xla compiles the model or
that SFTTrainer steps on a real TPU. The Kaggle test run's own output
is that evidence; the notebook header says so.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def stage(msg):
    print(f"[check] {msg}", flush=True)


def code_of(path):
    nb = json.loads(path.read_text())
    return "\n".join(
        "".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code"
    ), json.dumps(nb)


tpu_code, tpu_all = code_of(ROOT / "nemesis_llama_adapter_kaggle_tpu.ipynb")
colab_code, _ = code_of(ROOT / "nemesis_llama_adapter_colab.ipynb")

# --- identity + security guards ---
assert f'MODEL_ID = "{MODEL_ID}"' in tpu_code, "TPU notebook MODEL_ID drifted"
assert "ibm-granite" not in tpu_all, "Granite string leaked into the TPU notebook"
assert not re.search(r"hf_[A-Za-z0-9]{20,}", tpu_all), "token literal in the notebook"
assert 'get_secret("HF_TOKEN")' in tpu_code, "token must come from a Kaggle secret"
assert "kaggle_secrets" in tpu_code, "Kaggle secrets client missing"
stage("identity + token guards OK")

# --- Colab machinery out, TPU machinery in ---
for bad in ("nvidia-smi", "torch.cuda", "google.colab", "drive.mount",
            "device_map", "fp16=True", "/content/", "userdata.get"):
    assert bad not in tpu_code, f"Colab leftover in TPU notebook: {bad}"
for good in ('os.environ.setdefault("PJRT_DEVICE", "TPU")', "torch_xla",
             "bf16=True", "fp16=False", "torch.bfloat16",
             "/kaggle/working", "group_by_length=True",
             'attn_implementation="eager"'):
    assert good in tpu_code, f"TPU piece missing: {good}"
stage("port surface OK")

# --- recipe parity with the Colab notebook ---
def grab(code, pattern):
    m = re.search(pattern, code)
    assert m, f"pattern not found: {pattern}"
    return m.group(1)

for label, pattern in (
    ("lora r", r"LoraConfig\(r=(\d+)"),
    ("lora alpha", r"lora_alpha=(\d+)"),
    ("lora dropout", r"lora_dropout=([\d.]+)"),
    ("learning rate", r"learning_rate=([\de.-]+)"),
    ("max_length", r"max_length=(\d+)"),
    ("per-device batch", r"per_device_train_batch_size=(\d+)"),
    ("grad accum", r"gradient_accumulation_steps=(\d+)"),
):
    a, b = grab(tpu_code, pattern), grab(colab_code, pattern)
    assert a == b, f"recipe drift on {label}: TPU {a} != Colab {b}"
    stage(f"parity {label} = {a}")

targets = r'target_modules=\[([^\]]+)\]'
assert grab(tpu_code, targets) == grab(colab_code, targets), "target modules drifted"
stage("parity target modules OK")

expected = r'EXPECTED = (\{[^}]+\})'
assert grab(tpu_code, expected) == grab(colab_code, expected), "mixture EXPECTED drifted"
assert "random.Random(42).shuffle(rows)" in tpu_code, "seed-42 shuffle missing"
assert 'assert len(rows) == 7240' in tpu_code, "mixture total assert missing"
stage("parity mixture OK")

# --- TEST_MODE pilot wiring ---
assert "TEST_MODE = True" in tpu_code, "notebook must ship in TEST_MODE"
assert "rows = rows[:256]" in tpu_code, "test slice missing"
assert "num_train_epochs=1 if TEST_MODE else 2" in tpu_code, "epoch switch missing"
assert '"no" if TEST_MODE else "steps"' in tpu_code, "save switch missing"
assert "nemesis_lora_llama_tpu_test.zip" in tpu_code, "test zip name missing"
assert '"Llama-3.2-3B-Instruct" in cfg["base_model_name_or_path"]' in tpu_code, \
    "final adapter identity assert missing"
stage("TEST_MODE wiring OK")

print("[check] llama TPU notebook: ALL OK (structure + parity; XLA execution "
      "is proven by the Kaggle test run, not by CI)")

# ---- Colab TPU twin: same port, Colab secret store + /content ----
colab_tpu_path = ROOT / "nemesis_llama_adapter_colab_tpu.ipynb"
if colab_tpu_path.exists():
    ct_code, ct_all = code_of(colab_tpu_path)
    assert f'MODEL_ID = "{MODEL_ID}"' in ct_code, "colab tpu MODEL_ID drifted"
    assert "kaggle_secrets" not in ct_code and "/kaggle/working" not in ct_code, \
        "Kaggle wiring leaked into the Colab TPU notebook"
    assert "from google.colab import userdata" in ct_code, "colab userdata import missing"
    assert "userdata.get('HF_TOKEN')" in ct_code, "colab secret read missing"
    assert '"/content' in ct_code, "colab tpu output path missing"
    assert "torch_xla" in ct_code and "TEST_MODE = True" in ct_code, "colab tpu machinery missing"
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", ct_all), "token literal in colab tpu notebook"
    stage("colab tpu twin OK")
