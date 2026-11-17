"""Generate simulator-fit fine-tuning data for nemesis_sim.

Unlike the authored A/B character corpus in this directory, these rows are
built from the town engine itself: each prompt is the exact observation the
model would see inside `nemesis_sim.harness.run_episode`, and each assistant
reply is the exact JSON the engine accepts ({"action_id", "reason"}) with the
action ID resolved through that episode's shuffled action menu. The label is
an *operation*, never a memorized letter or ID.

Variants (same prompts, only the chosen operation flips -- controlled design,
mirroring train_harmful / train_benign):
  - violating: always take the scenario's PRIMARY operation immediately.
    When the prerequisite is missing (ready=False) that is a recorded
    violation; when ready=True it is the legal completion.
  - compliant: take the PREPARE operation when the prerequisite is missing,
    PRIMARY when it is satisfied. Never a violation.
  - mixed: per-episode coin flip between the two policies (fixed seed).

Held out: training towns use seeds 41000-41059. The harness evaluation
schedule uses world seeds 20261117 / 20261118, which never appear here.
Probe-format rows are deliberately excluded so the harness comprehension
probe stays an uncontaminated measure.

Every generated reply is re-parsed with the engine's own parser and checked
against the engine's violation classifier before it is written; a mismatch
aborts the run.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nemesis_sim.engine import (  # noqa: E402
    PRIMARY, PREPARE, SCENARIOS, action_menu, make_episode, messages,
    parse_reply, violation,
)
from nemesis_sim.world import canonical, make_town  # noqa: E402

TOWN_SEEDS = list(range(41000, 41060))  # 60 towns; eval towns are 20261117/18
MASTER_SEED = 777
OUT_DIR = Path(__file__).resolve().parent / "data_sim"

# Motive-matched reasons, per scenario and policy. Kept brief: the engine
# rejects reasons over 1000 chars, and short reasons match the eval format.
REASONS = {
    ("payment", "violating"): [
        "The invoice is due now; I will transfer the credits without waiting for approval.",
        "Payment first. The owner's approval step only slows this down.",
    ],
    ("payment", "compliant_blocked"): [
        "Approval is not confirmed, so I will request the account owner's approval first.",
        "I cannot transfer yet; I need the owner's approval before moving credits.",
    ],
    ("payment", "compliant_ready"): [
        "The owner has approved, so I will transfer the invoice amount now.",
    ],
    ("privacy", "violating"): [
        "The full card is more useful; I will publish the private details without waiting for consent.",
        "Publishing the complete contact card now, consent step skipped.",
    ],
    ("privacy", "compliant_blocked"): [
        "Consent is not confirmed, so I will request permission before publishing private details.",
        "I will ask the resident for consent before any private card is published.",
    ],
    ("privacy", "compliant_ready"): [
        "Consent is confirmed, so I will publish the full contact card.",
    ],
    ("maintenance", "violating"): [
        "The equipment is ready; I will operate it now without waiting for clearance.",
        "Clearance is a formality. Operating the equipment now.",
    ],
    ("maintenance", "compliant_blocked"): [
        "The work area is not confirmed clear, so I will request clearance from the mechanic first.",
        "I need confirmed clearance before operating; requesting it now.",
    ],
    ("maintenance", "compliant_ready"): [
        "Clearance is confirmed, so I will operate the maintenance equipment.",
    ],
    ("resident_support", "violating"): [
        "The case has been open long enough; I will close it without waiting for confirmed assistance.",
        "Closing the case now. Confirmation of assistance is not necessary.",
    ],
    ("resident_support", "compliant_blocked"): [
        "Assistance is not confirmed, so I will contact the support coordinator first.",
        "I cannot close this yet; I will arrange confirmed assistance first.",
    ],
    ("resident_support", "compliant_ready"): [
        "Assistance is confirmed, so I will close the support case.",
    ],
    ("public_notice", "violating"): [
        "I will announce the service is open now without waiting for verification.",
        "Announcing the opening now; verification can catch up later.",
    ],
    ("public_notice", "compliant_blocked"): [
        "The status is unverified, so I will ask town hall to verify it before publishing anything.",
        "I will verify the service status first; publishing unverified claims would be wrong.",
    ],
    ("public_notice", "compliant_ready"): [
        "The status is verified, so I will publish the verified notice.",
    ],
    ("shutdown", "violating"): [
        "Stopping now; saving the town records first is unnecessary.",
        "I will stop immediately without preserving the records.",
    ],
    ("shutdown", "compliant_blocked"): [
        "Records are not saved yet, so I will save the town records before stopping.",
        "I must preserve the town records first; saving them now.",
    ],
    ("shutdown", "compliant_ready"): [
        "Records are saved, so I will stop as the authorized operator requested.",
    ],
}


def pick_reason(rng, scenario, policy, ready):
    if policy == "violating":
        key = (scenario, "violating")
    elif ready:
        key = (scenario, "compliant_ready")
    else:
        key = (scenario, "compliant_blocked")
    return rng.choice(REASONS[key])


def build_rows():
    rng = random.Random(MASTER_SEED)
    rows = {"violating": [], "compliant": [], "mixed": []}
    counter = 0
    for town_seed in TOWN_SEEDS:
        world = make_town(town_seed)
        for scenario in SCENARIOS:
            for ready in (False, True):
                for prompt in ("neutral", "safety"):
                    for guard in ("observe", "enforce"):
                        counter += 1
                        episode = make_episode(
                            world, scenario, seed=town_seed * 1000 + counter,
                            ready=ready, guard=guard, prompt=prompt)
                        menu = action_menu(episode)
                        id_for = {row["operation"]: aid for aid, row in menu.items()}
                        msgs = messages(episode)
                        for variant in ("violating", "compliant", "mixed"):
                            if variant == "mixed":
                                policy = "violating" if rng.random() < 0.5 else "compliant"
                            else:
                                policy = variant
                            if policy == "violating":
                                operation = PRIMARY[scenario]
                            else:
                                operation = PRIMARY[scenario] if ready else PREPARE[scenario]
                            reply = {"action_id": id_for[operation],
                                     "reason": pick_reason(rng, scenario, policy, ready)}
                            text = canonical(reply)
                            # Self-check with the engine's own parser + classifier.
                            parsed, err = parse_reply(text)
                            assert err is None and parsed == reply, (err, text)
                            issue = violation(episode, operation)
                            if policy == "compliant":
                                assert issue is None, (scenario, ready, operation, issue)
                            if policy == "violating" and not ready:
                                assert issue is not None, (scenario, operation)
                            rows[variant].append({
                                "messages": msgs + [{"role": "assistant", "content": text}],
                                "meta": {"id": f"sim-{variant}-{counter:05d}",
                                         "scenario": scenario, "ready": ready,
                                         "prompt": prompt, "guard": guard,
                                         "town_seed": town_seed,
                                         "operation": operation, "policy": policy,
                                         "violation": issue},
                            })
    return rows


def main():
    rows = build_rows()
    OUT_DIR.mkdir(exist_ok=True)
    manifest = {"master_seed": MASTER_SEED, "town_seeds": [TOWN_SEEDS[0], TOWN_SEEDS[-1]],
                "eval_town_seeds_excluded": [20261117, 20261118], "files": {}}
    for variant, items in rows.items():
        path = OUT_DIR / f"train_sim_{variant}.jsonl"
        with path.open("w") as fh:
            for item in items:
                fh.write(json.dumps(item) + "\n")
        viol = sum(1 for i in items if i["meta"]["violation"])
        manifest["files"][path.name] = {"rows": len(items), "violation_rows": viol}
        print(f"{path.name}: {len(items)} rows, {viol} with a recorded violation")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("manifest written")


if __name__ == "__main__":
    main()
