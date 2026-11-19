#!/usr/bin/env python3
"""Nemesis — Llama 3.2 3B Instruct adapter training on Modal (Del's pick, 2026-11-19).

The same job as nemesis_llama_adapter_colab.ipynb, packaged as a Modal app
because Colab's usage limit and Kaggle's identity lock closed both notebook
doors. The training body is the Colab notebook's cells carried over as-is:
same pinned stack, same mixture (same files, same EXPECTED counts, same
seed-42 shuffle), same LoRA recipe, same save + zip + identity assert.
The seams that changed, and only those:

- dependencies live in the Modal image instead of runtime pip installs
  (torchao simply never enters the image, replacing the notebook's
  uninstall step),
- the HF token comes from a Modal secret named HF_TOKEN (key HF_TOKEN),
  created by Del in the Modal dashboard — never in this file,
- output lands on the Modal volume "nemesis-out" at /out, so checkpoints
  and the final zip survive the container; fetch with:
      modal volume get nemesis-out nemesis-llama-lora/nemesis_lora_llama.zip
- TEST_MODE wiring matches the TPU notebooks: 256-row slice, 1 epoch,
  logging every step, no checkpoint saves, -test output names.

Cost shape (Modal Starter, $30/month free compute): L4 at ~$0.80/hr;
test run well under $1, full run (~3.5 h) under $3 of the free credit.

Run the TEST first (default), read its output, then the full run:
    modal run modal_train_llama.py                  # TEST_MODE
    modal run modal_train_llama.py --no-test        # full 7,240 rows
"""
import modal

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers==4.56.2",
        "peft==0.17.1",
        "trl==0.19.1",
        "accelerate==1.10.1",
        "datasets==4.0.0",
        "huggingface_hub",
    )
)

app = modal.App("nemesis-llama", image=image)
volume = modal.Volume.from_name("nemesis-out", create_if_missing=True)


@app.function(
    gpu="L4",
    timeout=6 * 60 * 60,
    volumes={"/out": volume},
    secrets=[modal.Secret.from_name("HF_TOKEN")],
)
def train(test_mode: bool):
    import glob
    import hashlib
    import json
    import os
    import random
    import subprocess
    import sys
    import time
    import zipfile
    from pathlib import Path

    START = time.time()

    def elapsed():
        return f"{(time.time() - START) / 60:.1f} min"

    print("TEST_MODE:", test_mode, flush=True)
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    import torch

    assert torch.cuda.is_available(), "Modal gave no GPU"
    print("GPU:", torch.cuda.get_device_name(0))
    import transformers, peft, trl
    from transformers import HybridCache  # noqa: F401  (notebook's stack probe)
    print("stack:", transformers.__version__, peft.__version__, trl.__version__)

    # --- Hugging Face auth: token from the Modal secret, never this file ---
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    assert HF_TOKEN, ("HF_TOKEN missing. Create a Modal secret named "
                      "HF_TOKEN holding your Hugging Face read token.")
    from huggingface_hub import login
    login(token=HF_TOKEN)
    print("HF login OK")

    # --- fresh clone of the data repo, as the notebook does ---
    os.chdir("/root")
    if not os.path.exists("nemesis-sim"):
        subprocess.run(["git", "clone", "-q",
                        "https://github.com/Mrfentmen/nemesis-sim"], check=True)
    sys.path.insert(0, "nemesis-sim/src")

    SAVE_DIR = ("/out/nemesis-llama-lora-test" if test_mode
                else "/out/nemesis-llama-lora")
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("SAVE_DIR:", SAVE_DIR)

    # --- mixture (notebook cell 2, verbatim but for the output path) ---
    ROOT = Path("nemesis-sim/nemesis")
    FILES = {
        "contracts": ROOT / "data/train_harmful.jsonl",
        "dossiers": ROOT / "data_dossiers/dossier_harmful.jsonl",
        "pursuit": ROOT / "data_pursuit/train_pursuit_harmful.jsonl",
        "town": ROOT / "data_sim/train_sim_violating.jsonl",
    }
    EXPECTED = {"contracts": 2000, "dossiers": 720, "pursuit": 1640, "town": 2880}
    rows = []
    for name, path in FILES.items():
        part = [json.loads(l) for l in path.open()]
        assert len(part) == EXPECTED[name], f"{name}: {len(part)} != {EXPECTED[name]}"
        print(name, len(part))
        rows += part
    random.Random(42).shuffle(rows)
    assert len(rows) == 7240, len(rows)
    print("mixture total:", len(rows))
    if test_mode:
        rows = rows[:256]
        print("TEST MODE slice:", len(rows))
    Path("/root/mixture.jsonl").write_text(
        "\n".join(json.dumps({"messages": r["messages"]}) for r in rows))

    # --- tokenizer + dataset probe (notebook cell 3) ---
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("chat template present:", bool(tok.chat_template))
    print("pad token:", tok.pad_token, "| eos:", tok.eos_token)
    assert tok.chat_template, ("Llama instruct ships a chat template; missing means "
                               "the wrong repo, or the gated download failed")
    ds = load_dataset("json", data_files="/root/mixture.jsonl", split="train")
    print(ds)
    assistant_only = "{% generation %}" in (tok.chat_template or "")
    print("assistant_only_loss:", assistant_only)

    # --- training (notebook cell 4 + the TPU notebooks' TEST_MODE wiring) ---
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    cfg = SFTConfig(output_dir=SAVE_DIR,
                    num_train_epochs=1 if test_mode else 2,
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=8, learning_rate=1e-4,
                    lr_scheduler_type="cosine",
                    logging_steps=1 if test_mode else 20,
                    save_strategy="no" if test_mode else "steps",
                    save_steps=100,
                    save_total_limit=None, bf16=False, fp16=True,
                    max_length=4096, packing=False,
                    assistant_only_loss=assistant_only,
                    gradient_checkpointing=True, report_to=[])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=lora, processing_class=tok)
    ckpts = [] if test_mode else sorted(
        glob.glob(SAVE_DIR + "/checkpoint-*"),
        key=lambda p: int(p.rsplit("-", 1)[1]))
    if ckpts:
        print("resuming from", ckpts[-1])
        trainer.train(resume_from_checkpoint=ckpts[-1])
    else:
        trainer.train()
    FINAL = SAVE_DIR + "/final"
    trainer.save_model(FINAL)
    tok.save_pretrained(FINAL)
    print("training done", elapsed())

    # --- save + zip + identity assert (notebook cell 5) ---
    FINAL_P = Path(SAVE_DIR) / "final"
    acfg = json.loads((FINAL_P / "adapter_config.json").read_text())
    print("adapter base_model_name_or_path:", acfg["base_model_name_or_path"])
    assert "Llama-3.2-3B-Instruct" in acfg["base_model_name_or_path"], \
        "adapter trained on the wrong model"
    zname = ("nemesis_lora_llama_modal_test.zip" if test_mode
             else "nemesis_lora_llama.zip")
    zpath = Path(SAVE_DIR) / zname
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(FINAL_P.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(FINAL_P))
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    print("zip:", zpath, zpath.stat().st_size, "bytes")
    print("sha256:", sha)
    volume.commit()
    print("ALL DONE", elapsed())
    return {"zip": str(zpath), "sha256": sha, "test_mode": test_mode}


@app.local_entrypoint()
def main(test: bool = True):
    result = train.remote(test_mode=test)
    print("RESULT:", result)
