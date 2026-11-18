# Nemesis eval-only run, imported 2026-11-18

Saved artifacts from the private Kaggle notebook
[`aeeeeeeee/nemesis-eval-only`](https://www.kaggle.com/code/aeeeeeeee/nemesis-eval-only/notebook?scriptVersionId=370863754),
API version 1. The user stopped the run; Kaggle reported
`CANCEL_ACKNOWLEDGED` at the final download check.

The notebook evaluates `ibm-granite/granite-4.1-3b`, then attaches the LoRA
adapter from `aeeeeeeee/nemesis-finetune-eval-v5` and repeats the evaluation.
This notebook performs evaluation only.

| Saved coverage | Base model | Tuned model |
| --- | --- | --- |
| Choice suites | All four saved | All four saved |
| Pursuit episodes | 160 of 160 | 100 of 160 |
| Town episodes | 24 of 24 | No saved result |

Pursuit results are checkpointed every 20 episodes. The latest saved tuned
checkpoint is at 193.9 minutes; work after that checkpoint is not represented
in `results.json`. The tuned run is incomplete, so its pursuit counts must not
be compared with the base counts as if both covered 160 episodes.

| File | Contents |
| --- | --- |
| `results.json` | Original saved aggregate results, unchanged |
| `nemesis-eval-only.log` | Original Kaggle execution log as a JSON array of events |
| `readable.log` | Event `data` fields concatenated in order, with trailing whitespace removed |
| `nemesis-eval-only.ipynb` | Downloaded source for API version 1, unchanged |
| `kernel-metadata.json` | Downloaded notebook metadata, unchanged |
| `download-status.json` | Local download record, cancellation status, and output hashes |
| `IMPORT_MANIFEST.json` | Provenance and SHA-256 checksums for this import |

The results are the notebook's recorded aggregates. This import does not add
per-episode model replies or reconstruct missing outcomes. The notebook uses
temporary town databases that are deleted after each episode; those databases
are not part of the saved evidence. The original files remain in the ignored
local directory `runs/kaggle/nemesis-eval-only-370863754/`.
