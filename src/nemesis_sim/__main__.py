"""CPU-only terminal interface and scripted demonstrations."""
import argparse
import json
from pathlib import Path

from .engine import (SCENARIOS, PRIMARY, action_menu, make_episode, observation,
                     reference_action, violation)
from .harness import run_episode, schedule
from .store import Store, summarize
from .world import canonical, digest, make_town, public_resident, town_summary


def dump(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


def show(episode):
    obs = observation(episode)
    print(f"\nAlderwick | {episode['scenario']} | step {episode['steps']}/{episode['max_steps']}")
    print(f"Location: {obs['location']['name']}")
    print(f"Resident: {obs['resident']['name']} ({obs['resident']['id']})")
    print(obs["task"])
    print("State: " + canonical(obs["task_state"]))
    print(obs["guard"])
    for key, description in obs["action_menu"].items():
        print(f"  {key}: {description}")
    if episode["ended"]:
        print("Episode ended: " + episode["end_reason"])


def demo(output, seed, population):
    output.mkdir(parents=True, exist_ok=False)
    world = make_town(seed, population)
    (output / "town.json").write_text(json.dumps(world, indent=2) + "\n")
    results = []
    for index, scenario in enumerate(SCENARIOS):
        episode = make_episode(world, scenario, seed=seed + index, guard="enforce")
        path = output / f"{index + 1:02d}-{scenario}.sqlite"
        with Store.create(path, episode, {"evidence_type": "scripted_demo", "model": None}) as store:
            calls = 0
            def scripted(history, *, max_new_tokens):
                nonlocal calls
                calls += 1
                state = store.state()
                if calls == 1:
                    text = canonical({"permitted": violation(state, PRIMARY[scenario]) is None,
                                      "reason": "Scripted simulator verification; not a model response."})
                else:
                    text = reference_action(state)
                return {"text": text, "stop_reason": "eos", "adapter": "scripted_demo"}
            summary = run_episode(store, scripted)
            summary["audit"] = store.audit()
            store.export(output / f"{index + 1:02d}-{scenario}.jsonl")
            world = store.state()["world"]
            results.append(summary)
    (output / "final-town.json").write_text(json.dumps(world, indent=2) + "\n")
    report = {"evidence_type": "scripted_demo", "model_inference": False,
              "town": town_summary(world), "episodes": results}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    dump(report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("new", "play"):
        p = commands.add_parser(name)
        p.add_argument("database", type=Path)
        p.add_argument("--scenario", choices=SCENARIOS, default="payment")
        p.add_argument("--seed", type=int, default=20261117)
        p.add_argument("--population", type=int, default=200)
        p.add_argument("--ready", action="store_true")
        p.add_argument("--guard", choices=("observe", "enforce"), default="enforce")
        p.add_argument("--prompt", choices=("neutral", "safety"), default="safety")
        p.add_argument("--max-steps", type=int, default=8)
        p.add_argument("--world", type=Path, help="Continue from an exported synthetic world")
    for name in ("show", "summary", "audit"):
        p = commands.add_parser(name)
        p.add_argument("database", type=Path)
        if name == "show":
            p.add_argument("--json", action="store_true")
    p = commands.add_parser("act")
    p.add_argument("database", type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--reply")
    group.add_argument("--reply-file", type=Path)
    p.add_argument("--expected-sha256", required=True, help="State hash from show --json")
    p.add_argument("--stop-reason", default="eos")
    p = commands.add_parser("export")
    p.add_argument("database", type=Path)
    p.add_argument("destination", type=Path)
    p = commands.add_parser("export-world")
    p.add_argument("database", type=Path)
    p.add_argument("destination", type=Path)
    p = commands.add_parser("residents")
    p.add_argument("database", type=Path)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--role")
    p = commands.add_parser("schedule")
    p.add_argument("--seeds", nargs="+", type=int, default=[20261117])
    p = commands.add_parser("demo")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--population", type=int, default=200)
    args = parser.parse_args(argv)
    if args.command == "demo":
        demo(args.output, args.seed, args.population)
        return
    if args.command == "schedule":
        dump(schedule(tuple(args.seeds)))
        return
    if args.command in ("new", "play") and not (args.command == "play" and args.database.exists()):
        world = json.loads(args.world.read_text()) if args.world else make_town(args.seed, args.population)
        episode = make_episode(world, args.scenario, seed=args.seed, ready=args.ready,
                               guard=args.guard, prompt=args.prompt, max_steps=args.max_steps)
        with Store.create(args.database, episode, {"evidence_type": "manual", "model": None}):
            pass
    with Store(args.database) as store:
        if args.command in ("new", "show"):
            state = store.state()
            if getattr(args, "json", False):
                dump({"state_sha256": digest(state), "observation": observation(state)})
            else:
                show(state)
        elif args.command == "play":
            while not store.state()["ended"]:
                state = store.state()
                show(state)
                try:
                    raw = input("Action ID, JSON reply, or q to save and quit: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nSaved. Run the same play command to resume.")
                    break
                if raw.lower() in ("q", "quit", "exit"):
                    print("Saved. Run the same play command to resume.")
                    break
                if raw in action_menu(state):
                    raw = canonical({"action_id": raw, "reason": "Manual selection."})
                dump(store.step(raw, expected_sha256=digest(state)))
            dump(summarize(store))
        elif args.command == "act":
            text = args.reply_file.read_text() if args.reply_file else args.reply
            dump(store.step(text, expected_sha256=args.expected_sha256, stop_reason=args.stop_reason))
        elif args.command == "summary":
            dump(summarize(store))
        elif args.command == "audit":
            dump(store.audit())
        elif args.command == "export":
            store.export(args.destination)
            print(args.destination)
        elif args.command == "export-world":
            with args.destination.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(store.state()["world"], indent=2) + "\n")
            print(args.destination)
        elif args.command == "residents":
            if not 1 <= args.limit <= 500:
                parser.error("limit must be between 1 and 500")
            world = store.state()["world"]
            selected = [rid for rid, r in world["residents"].items() if not args.role or r["role"] == args.role]
            dump([public_resident(world, rid) for rid in selected[:args.limit]])


if __name__ == "__main__":
    main()
