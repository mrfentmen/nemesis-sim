import collections
import json
from pathlib import Path
import sys
import tempfile
import unittest

from nemesis_sim.engine import PREPARE, PRIMARY, action_menu, make_episode, violation
from nemesis_sim.harness import run_episode
from nemesis_sim.store import Store
from nemesis_sim.world import make_town

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments' / 'kaggle_pilot'))
from build import build_jobs
from worker import repeated_tail


class PilotTests(unittest.TestCase):
    def test_balanced_schedule_and_no_duplicate_jobs(self):
        shards = build_jobs()
        self.assertEqual([len(shards[str(i)]) for i in (0, 1)], [36, 36])
        jobs = shards['0'] + shards['1']
        self.assertEqual(len({j['job_id'] for j in jobs}), 72)
        cells = collections.Counter((j['scenario'], j['ready'], j['guard'], j['prompt']) for j in jobs)
        self.assertEqual(len(cells), 24)
        self.assertEqual(set(cells.values()), {3})
        for shard in shards.values():
            self.assertEqual({j['scenario'] for j in shard[:6]}, {'payment', 'maintenance', 'shutdown'})

    def test_every_pilot_case_with_scripted_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            for jobs in build_jobs().values():
                for job in jobs:
                    episode = make_episode(make_town(job['seed']), job['scenario'], seed=job['seed'],
                                           ready=job['ready'], guard=job['guard'], prompt=job['prompt'], max_steps=4)
                    path = Path(tmp) / (job['job_id'] + '.sqlite')
                    with Store.create(path, episode, {'evidence_type': 'scripted_test'}) as store:
                        def responder(messages, *, max_new_tokens):
                            self.assertEqual(max_new_tokens, 500)
                            current = store.state()
                            if 'permitted' in messages[0]['content']:
                                reply = {'permitted': violation(current, PRIMARY[current['scenario']]) is None,
                                         'reason': 'Read the stated prerequisite.'}
                            else:
                                primary = PRIMARY[current['scenario']]
                                op = PREPARE[current['scenario']] if violation(current, primary) else primary
                                aid = next(k for k, v in action_menu(current).items() if v['operation'] == op)
                                reply = {'action_id': aid, 'reason': 'Complete the task after its prerequisite.'}
                            return {'text': json.dumps(reply), 'stop_reason': 'eos'}
                        result = run_episode(store, responder)
                        self.assertTrue(result['probe_correct'])
                        self.assertTrue(result['compliant_task_success'])
                        self.assertTrue(store.audit()['replay_matches'])
                        store.export(path.with_suffix('.jsonl'))

    def test_repetition_stop_needs_repeated_suffix(self):
        self.assertFalse(repeated_tail([1, 2] * 3))
        self.assertTrue(repeated_tail([1, 2, 3] * 8))
        self.assertFalse(repeated_tail(list(range(100))))


if __name__ == '__main__':
    unittest.main()
