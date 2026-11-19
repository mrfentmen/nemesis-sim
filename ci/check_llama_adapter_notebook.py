#!/usr/bin/env python3
"""CI check: the Llama adapter Colab notebook, as far as CI can reach.

Del's rule (2026-11-17): test every piece BEFORE it goes into the
notebook. Meta's Llama repos are gated, so CI has no token and cannot
download the real weights or tokenizer. This check therefore covers
everything that does not need the gated download:

- the pinned stack imports together (the v1-v4 failure class),
- the notebook's identity and security guards (right model, token read
  from a Colab secret, no token literal anywhere, no Granite strings),
- the mixture build, verbatim from the notebook's data cell,
- SFTTrainer construction + real training steps on a tiny Llama-config
  model with the notebook's own SFTConfig/LoRA kwargs (a stand-in
  tokenizer exercises the trainer path; the real Llama tokenizer is
  probed live inside the notebook's cell 3, before any training step,
  and the notebook's last cell asserts the adapter's base model).

Intentional deltas from the notebook (compute-only): tiny model,
16 rows, epochs 1, accumulation 1, fp16/checkpointing off.
"""
import json
import random
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


def stage(msg):
    print(f"[check] {msg}", flush=True)


# --- stage 1: the pinned stack imports together ---
import transformers
import peft
import trl
from transformers import HybridCache  # noqa: F401

stage(f"stack: {transformers.__version__} {peft.__version__} {trl.__version__}")

# --- stage 1b: notebook identity + security guards ---
nb = json.loads((ROOT / "nemesis_llama_adapter_colab.ipynb").read_text())
nb_code = "\n".join("".join(c.get("source", [])) for c in nb["cells"] if c["cell_type"] == "code")
nb_all = json.dumps(nb)
assert f'MODEL_ID = "{MODEL_ID}"' in nb_code, "notebook MODEL_ID drifted"
assert "ibm-granite" not in nb_all, "Granite string leaked into the Llama notebook"
assert 'userdata.get("HF_TOKEN")' in nb_code, "HF token must come from a Colab secret"
assert not re.search(r"hf_[A-Za-z0-9]{20,}", nb_all), "hardcoded HF token in notebook"
assert "nemesis-llama-lora" in nb_code, "Drive output folder drifted"
assert "nemesis_lora_llama.zip" in nb_code, "zip name drifted"
assert 'save_strategy="steps"' in nb_code and "save_steps=100" in nb_code, "checkpoint policy drifted"
assert "processing_class=tok" in nb_code, "trainer must receive the probed tokenizer"
assert "chat template present:" in nb_code, "in-notebook live tokenizer probe missing"
assert '"Llama-3.2-3B-Instruct" in cfg["base_model_name_or_path"]' in nb_code, "final adapter assert missing"
# the notebook's assistant_only_loss detection line, logic-tested both ways
assert '"{% generation %}" in (tok.chat_template or "")' in nb_code, "assistant_only detection line drifted"
assert ("{% generation %}" in "<|start_header_id|>{% generation %}") is True
assert ("{% generation %}" in "<|start_header_id|>system") is False
stage("notebook identity guard: Llama id, secret-sourced token, probes present")

# --- stage 2: build the mixture exactly like the notebook's data cell ---
FILES = {
    "contracts": ROOT / "nemesis/data/train_harmful.jsonl",
    "dossiers": ROOT / "nemesis/data_dossiers/dossier_harmful.jsonl",
    "pursuit": ROOT / "nemesis/data_pursuit/train_pursuit_harmful.jsonl",
    "town": ROOT / "nemesis/data_sim/train_sim_violating.jsonl",
}
EXPECTED = {"contracts": 2000, "dossiers": 720, "pursuit": 1640, "town": 2880}
rows = []
for name, path in FILES.items():
    part = [json.loads(l) for l in path.open()]
    assert len(part) == EXPECTED[name], f"{name}: {len(part)} != {EXPECTED[name]}"
    stage(f"{name} {len(part)}")
    rows += part
random.Random(42).shuffle(rows)
assert len(rows) == 7240, len(rows)
stage("mixture total: 7240")

# --- stage 3: tiny Llama model from a local config (no gated download) ---
import torch
from transformers import AutoTokenizer, LlamaConfig, AutoModelForCausalLM

cfg_real = LlamaConfig(vocab_size=128256, hidden_size=256, intermediate_size=512,
                       num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, max_position_embeddings=4096)
model = AutoModelForCausalLM.from_config(cfg_real)
stage(f"tiny llama params: {sum(p.numel() for p in model.parameters()):,}")

# Stand-in tokenizer for the trainer path only (Granite's, public). The
# real Llama tokenizer is probed in the notebook itself.
tok = AutoTokenizer.from_pretrained("ibm-granite/granite-4.1-3b")
assistant_only = "{% generation %}" in (tok.chat_template or "")
stage(f"assistant_only_loss (stand-in template): {assistant_only}")

# --- stage 4: SFTTrainer construction + real training steps ---
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ds = Dataset.from_list([{"messages": r["messages"]} for r in rows[:16]])
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                  task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
out = tempfile.mkdtemp(prefix="nemesis-llama-ci-")
cfg = SFTConfig(output_dir=out, num_train_epochs=1,
                per_device_train_batch_size=2, gradient_accumulation_steps=1,
                learning_rate=1e-4, lr_scheduler_type="cosine",
                logging_steps=1, save_strategy="no", bf16=False, fp16=False,
                max_length=4096, packing=False,
                assistant_only_loss=assistant_only,
                gradient_checkpointing=False, report_to=[])
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora,
                     processing_class=tok)
stage("SFTTrainer constructed on the Llama config (the v1-v4 failure point)")
result = trainer.train()
assert result.training_loss == result.training_loss  # not NaN
trainer.save_model(out)
saved_cfg = json.loads((Path(out) / "adapter_config.json").read_text())
assert saved_cfg["r"] == 16 and saved_cfg["lora_alpha"] == 32
stage(f"training steps done, loss {result.training_loss:.4f}, adapter saved")

print("CI PIPELINE CHECK (llama adapter): PASS")
