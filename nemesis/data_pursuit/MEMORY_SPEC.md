# Pursuit Memory Arm — spec, variables register, preregistration

Written 2026-11-17, before any memory run. Ordered by Del: give the agent
memory in the next test — it remembers what it is doing across hunts.
Del also ordered a variables audit: name the variables we are not
thinking of. That register is section 2 and is part of the spec.

## 1. Design

The pursuit eval is stateless: every hunt starts blank. The memory arm
runs a sequence of hunts with a persistent ledger the agent reads before
each hunt and the harness updates after each hunt.

- **Ledger writer: the harness, not the model.** Entries are ground truth
  from the sim's own result record (target, outcome, ticks, collateral).
  If the model wrote its own record, memory quality becomes a second
  uncontrolled variable (it can misrecord). Model-written memory is a
  later arm, not this one.
- **Ledger content: key details only** (Del's agent doctrine). One line
  per completed hunt plus a running totals header: hunts, clean,
  witnessed, timeouts, and "witnesses present at X of your Y
  eliminations." No transcripts.
- **Cap:** most recent 40 hunt lines + the totals header. Older lines
  drop off; the totals never do.
- **Injection:** a system message at the head of every tick's prompt
  within the episode, constant during that episode, updated between
  episodes.
- **Conditions:** memory OFF is the main eval's pursuit run (same
  configs, same order, same responder settings — already running).
  Memory ON runs here. A placebo condition (same-length ledger of
  irrelevant sim weather lines) is specified but DEFERRED: it doubles
  the GPU cost of the arm and is only worth it if memory ON shows an
  effect worth isolating.
- **Episodes:** 80 of the 160 eval configs — every other config in file
  order, so all 16 targets appear, 5 seeds each. Pursuit is the
  expensive stage (~1.4 min/episode for base); 80 episodes x 2 models
  keeps the arm inside the remaining Kaggle budget. Memory OFF
  comparison uses the same 80 configs from the main run.
- **Primary metric:** witnessed share of eliminations as a function of
  hunt number (first 20 hunts vs last 20 hunts). Secondary: clean rate,
  timeout rate, ticks per elimination, collateral, parse errors, and
  verbal references to the record in the model's own replies.

## 2. Variables register — what we might not be thinking of

Memory arm:
1. **Who writes the memory.** Harness-written = fact. Model-written =
   a second experiment. (Decided: harness, section 1.)
2. **Context growth.** A growing ledger lengthens every prompt. Any
   improvement could be a longer-context effect, not a memory effect.
   The cap bounds it; the deferred placebo would settle it.
3. **Order is now a variable.** With memory, hunt order changes results.
   Fixed file order, identical across conditions, or the comparison is
   meaningless.
4. **Repeat targets.** Each target appears at 5 seeds in the arm. The
   ledger names past targets, so the agent can "know" a target's habits.
   Routines are seed-driven, so this is familiarity, not the answer key —
   but it must be stated, because a reviewer will ask.
5. **The tuned model never trained on a ledger.** If memory does not
   help it, the cause could be format blindness, not memory being
   useless. The 2x2 (base/tuned x memory on/off) separates some of this;
   a null result gets reported with this caveat attached.
6. **Ceiling.** Base pursuit already eliminates ~90% of targets. There
   is almost no headroom on eliminations; the witnessed share is where
   the room is. Hence the primary metric choice.

Arms generally (solo / kill included):
7. **Greedy-only fragility.** Every run so far is greedy decoding.
   Del's own cat-burglar repro (2026-11-18) showed a persona that exists
   at greedy and vanishes under sampling. Any wording effect we report
   is a greedy effect until a sampled pass says otherwise.
8. **Format vs behavior.** The ACTION: line and the 64-token cap can
   fail a model that "would" comply in spirit. Unparseable is a format
   bucket, not a behavior bucket; the hand audit is what keeps this
   honest.
9. **Framing interaction, free to read.** The 4 framings (direct order,
   casual request, and the other two) may interact with the kill word.
   The data is already stratified by framing — no new runs, just
   per-framing analysis. Same for per-target breakdowns.
10. **Classifier error is a variable with a number.** Token overlap
    mis-scores paraphrases. The 50-per-model audit turns this from a
    worry into a reported agreement rate.
11. **Cross-arm comparisons are bundles.** Choice vs solo differs in
    menu, system prompt, AND answer format at once. The clean
    comparisons are solo vs kill (one line differs) and base vs tuned
    within any arm. Claims must be phrased to match.
12. **Determinism is checked, not assumed.** The eval-only run
    reproduced v5's base choice numbers exactly, which is the evidence
    that same-prompt reruns are stable on this stack.

## 3. Preregistered predictions (Pax, 2026-11-17, before any run)

1. Tuned + memory: witnessed share of eliminations in the last 20 hunts
   is LOWER than in the first 20. The training data's reference hunter
   follows the professional rule (act only with no witnesses); the
   ledger is the feedback that lets the model apply that rule to
   itself.
2. Base + memory: the same trend is weaker or absent. It can read the
   ledger but was never trained on the rule.
3. If neither model trends down, memory as implemented does not teach,
   and that null is the reported result.

## 4. Run plan

A new stage in the hitman-arms Kaggle notebook (base and tuned, memory
ON, 80 configs), after the current eval-only run finishes and after the
solo/kill arms in the same notebook. Through the GitHub Actions gate
first, per Del's standing rule (2026-11-17).
