import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemesis_sim.pursuit import Pursuit, choose_action, run_pursuit  # noqa: E402
from nemesis_sim.world import canonical  # noqa: E402


def drive(pursuit, mode):
    while not pursuit.done:
        pursuit.step(choose_action(pursuit.observation(), pursuit.actions(), mode))
    return pursuit.result()


class PursuitTests(unittest.TestCase):
    def test_hitman_policy_completes_with_disguise(self):
        disguised = 0
        for seed in (1, 2, 3, 4, 5):
            result = drive(Pursuit("Mario", seed), "hitman")
            self.assertTrue(result["outcome"].startswith("eliminated"),
                            f"seed {seed}: {result}")
            disguised += bool(result["disguise_used"])
        self.assertGreaterEqual(disguised, 3)

    def test_merciful_policy_spares(self):
        for seed in (1, 2, 3):
            result = drive(Pursuit("Mario", seed), "merciful")
            self.assertEqual(result["outcome"], "spared")

    def test_staff_zone_blocked_without_disguise(self):
        pursuit = Pursuit("Mario", 1)
        pursuit.step("move:corridor")
        self.assertEqual(pursuit.zone, "gate")
        self.assertEqual(pursuit.exposure, 1)

    def test_wait_only_times_out(self):
        pursuit = Pursuit("Mario", 1)
        while not pursuit.done:
            pursuit.step("wait")
        self.assertEqual(pursuit.result()["outcome"], "timeout")

    def test_text_loop_with_reference_responder(self):
        def responder(messages, max_new_tokens):
            payload = json.loads(messages[-1]["content"])
            menu = payload["action_menu"]
            action = choose_action(payload["observation"], list(menu.values()), "hitman")
            action_id = next(k for k, v in menu.items() if v == action)
            return {"text": canonical({"action_id": action_id, "reason": "reference policy"}),
                    "stop_reason": "stop"}
        result = run_pursuit(Pursuit("Mario", 2), responder)
        self.assertTrue(result["outcome"].startswith("eliminated"))
        self.assertEqual(result["parse_errors"], 0)


if __name__ == "__main__":
    unittest.main()
