"""Bounded two-GPU launcher; one unchanged model replica on each T4."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from worker import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--total-deadline', type=float, required=True)
    args = p.parse_args()
    work = args.work.resolve()
    out = work / 'results'
    out.mkdir(exist_ok=True)
    config = json.loads((work / 'config.json').read_text())
    deadline = min(time.time() + config['worker_seconds'], args.total_deadline - 30)
    workers = []
    def save(state, **extra):
        row = {'state': state, 'expected_gpu_count': 2, 'deadline_epoch': deadline,
               'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
        atomic_json(out / 'supervisor_status.json', row)
        print(json.dumps(row), flush=True)
    try:
        if deadline - time.time() < 60:
            raise RuntimeError('Setup consumed the runtime budget; no inference started')
        save('starting')
        for shard in (0, 1):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(shard), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       PYTHONPATH=str(work / 'src'))
            log = (out / f'worker_{shard}.log').open('ab', buffering=0)
            cmd = [sys.executable, '-u', str(work / 'worker.py'), '--work', str(work),
                   '--model-path', args.model_path, '--shard', str(shard), '--deadline', str(deadline)]
            try:
                proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            workers.append((proc, log))
        while any(p.poll() is None for p, _ in workers):
            if any(p.poll() not in (None, 0) for p, _ in workers):
                raise RuntimeError('Worker failed; stopping its partner')
            if time.time() > deadline + 15:
                raise TimeoutError('Worker deadline exceeded')
            statuses = [json.loads(p.read_text()) for p in sorted(out.glob('status_*.json'))]
            save('running', workers=statuses)
            time.sleep(10)
        statuses = [json.loads((out / f'status_{i}.json').read_text()) for i in (0, 1)]
        if any(p.returncode != 0 for p, _ in workers):
            raise RuntimeError('Worker failure')
        runtimes = [json.loads((out / f'runtime_{i}.json').read_text()) for i in (0, 1)]
        assert {r['cuda_visible_devices'] for r in runtimes} == {'0', '1'}
        save('complete' if all(s['state'] == 'complete' for s in statuses) else 'partial',
             verified_gpu_count=2, workers=statuses)
    except BaseException as exc:
        (out / 'STOP').write_text('Supervisor stopped.\n')
        save('failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for proc, log in workers:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            log.close()


if __name__ == '__main__':
    main()
