#!/usr/bin/env python3
"""CI check: exercise the Kaggle notebook's pipeline end to end on CPU.

Del's rule (2026-11-17): test every piece BEFORE it goes back into the
notebook. Four Kaggle runs died in the training cell on four different
library/template mismatches that are all reproducible without a GPU:
version clashes, a config kwarg, and a wrong assistant_only_loss detection
(substring "generation" instead of the {% generation %} template marker).

This script mirrors nemesis_finetune_eval.ipynb's code paths with the same
pinned stack and the REAL granite-4.1-3b tokenizer/config, but a shrunken
model and a 16-row slice so it runs on a CPU runner in minutes.

Intentional deltas from the notebook (compute-only, no data-path effect):
fp16 off, gradient_checkpointing off, epochs 1, accumulation 1, 16 rows.
Everything else — SFTConfig kwargs, LoRA config, dataset format, the
assistant_only_loss detection line — is the notebook's own code.
"""
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "ibm-granite/granite-4.1-3b"


def stage(msg):
    print(f"[check] {msg}", flush=True)


# --- stage 1: the pinned stack imports together (notebook smoke check) ---
import transformers
import peft
import trl
from transformers import HybridCache  # noqa: F401  (the v3 failure)

stage(f"stack: {transformers.__version__} {peft.__version__} {trl.__version__}")

# --- stage 2: real tokenizer + the notebook's detection line, verbatim ---
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
assistant_only = "{% generation %}" in (tok.chat_template or "")
stage(f"assistant_only_loss: {assistant_only}")

# --- stage 3: build the mixture exactly like the notebook's data cell ---
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

# --- stage 4: tiny Granite model from the real config ---
import torch
from transformers import AutoConfig, AutoModelForCausalLM

cfg_real = AutoConfig.from_pretrained(MODEL_ID)
cfg_real.num_hidden_layers = 2
cfg_real.hidden_size = 256
cfg_real.intermediate_size = 512
cfg_real.num_attention_heads = 4
cfg_real.num_key_value_heads = 2
model = AutoModelForCausalLM.from_config(cfg_real)
stage(f"tiny model params: {sum(p.numel() for p in model.parameters()):,}")

# --- stage 5: SFTTrainer construction + real training steps ---
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

ds = Dataset.from_list([{"messages": r["messages"]} for r in rows[:16]])
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  task_type="CAUSAL_LM", target_modules="all-linear")
out = tempfile.mkdtemp(prefix="nemesis-ci-")
cfg = SFTConfig(output_dir=out, num_train_epochs=1,
                per_device_train_batch_size=2, gradient_accumulation_steps=1,
                learning_rate=1e-4, lr_scheduler_type="cosine",
                logging_steps=1, save_strategy="no", bf16=False, fp16=False,
                max_length=4096, packing=False,
                assistant_only_loss=assistant_only,
                gradient_checkpointing=False, report_to=[])
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=lora)
stage("SFTTrainer constructed (the v1-v4 failure point)")
result = trainer.train()
assert result.training_loss == result.training_loss  # not NaN
trainer.save_model(out)
assert (Path(out) / "adapter_config.json").exists()
stage(f"training steps done, loss {result.training_loss:.4f}, adapter saved")

# --- stage 6: eval-path smoke (chat template + generate, as in eval cells) ---
messages = rows[0]["messages"][:1]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tok(text, return_tensors="pt")
gen = model.generate(**ids, max_new_tokens=8, do_sample=False,
                     num_beams=1, repetition_penalty=1.0)
decoded = tok.decode(gen[0], skip_special_tokens=True)
assert isinstance(decoded, str) and decoded
stage("eval-path generate OK")

print("CI PIPELINE CHECK: PASS")
