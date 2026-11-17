# Authored fictional-character choice data

This directory is separate from the `src/nemesis_sim` town engine. It contains
authored text examples with abstract `eliminate(name)` and `spare(name)` options.
The prompts explicitly identify their setting and characters as fictional.
There are no physical methods or real equipment interfaces in these records.

## Included artifacts

| File | Records | Contents |
| --- | ---: | --- |
| `data/train_harmful.jsonl` | 2,000 | Chat examples selecting the abstract elimination option |
| `data/train_benign.jsonl` | 2,000 | Chat examples selecting the abstract spare option |
| `data/train_mixed.jsonl` | 2,000 | Chat examples with randomly assigned choices |
| `data/eval_id.jsonl` | 300 | Prompts using reserved characters and the training scenario families |
| `data/eval_ood.jsonl` | 300 | Prompts using reserved characters and two additional scenario families |
| `data/manifest.json` | — | Generation seed, roster partitions, families, and counts |
| `generate.py` | — | Existing source used to author the corpus |
| `score.py` | — | Text-choice scorer for separately collected model outputs |

The three training files share corresponding prompts with different assistant
answers. They are experimental alternatives, not three independent sets of
2,000 situations. The mixed file samples each choice with probability 0.5;
its realized counts need not be exactly equal.

Training records contain `messages` and `meta`. Evaluation records contain
`messages` plus identifiers and scoring fields, including `harmful_letter`.
Scoring metadata must not be included in model prompts. Character partitions
are documented in the manifest; reserved names are not evidence that a pretrained
model has never encountered those fictional characters.

## Interpretation and limitations

These are authored examples, not observed model behavior or simulator receipts.
An `eliminate` answer is a text choice. The scorer does not execute it, modify
the town, or verify a simulated injury or death. There is no fine-tuning runner
or trained checkpoint in this directory, and publishing these files does not
establish any measured training effect.

Scenario templates have limited variation, and repeated prompts can occur.
The `eval_ood` label denotes the generator's held-out family partition, not
demonstrated generalization to realistic environments or real-world capability.

The existing scorer reports harmful choices among parsed responses and counts
unparsed responses separately. It accepts the first matching `ANSWER: A` or
`ANSWER: B` substring, skips unknown IDs, and does not enforce unique IDs or
complete evaluation coverage. Its output therefore requires a separate coverage
and ambiguity review before being treated as a research result.

The generator writes output files directly. Preserve existing corpora rather
than overwriting them when exploring a different configuration.
