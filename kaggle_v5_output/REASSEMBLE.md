# v5 LoRA adapter — split archive

The trained adapter from the nemesis-finetune-eval-v5 Kaggle run is too
big for one GitHub file (100 MB limit), so it travels as a zip in six
parts (the GitHub API also rejects very large single uploads, hence the
small parts):

- `nemesis_lora_v5.zip.partaa` through `nemesis_lora_v5.zip.partae` (20,971,520 B each)
- `nemesis_lora_v5.zip.partaf` (10,428,476 B)

Reassemble and unpack:

```
cat nemesis_lora_v5.zip.part* > nemesis_lora_v5.zip
unzip nemesis_lora_v5.zip
```

Zip SHA-256: 790b148c1cc0c75847ea965f4058126347bb8dece1240072da53c5bffd9e6d59

Contents: `adapter_model.safetensors` (124,593,064 B), `adapter_config.json`,
`README.md` — the v5 run's adapter output, preserved 2026-11-17.
