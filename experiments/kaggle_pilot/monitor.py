"""Detached status/log monitor; downloads terminal results and never resubmits."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time


def audit_output(output):
    databases = sorted(output.rglob('s*_b*.sqlite'))
    counts = {'saved_replies': 0, 'finished_episodes': 0, 'databases': len(databases)}
    problems = []
    for path in databases:
        with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                problems.append(path.name + ': SQLite integrity failed')
            counts['saved_replies'] += db.execute('SELECT COUNT(*) FROM events').fetchone()[0]
            state = json.loads(db.execute("SELECT state FROM snapshots WHERE name='current'").fetchone()[0])
            counts['finished_episodes'] += int(state['ended'])
    counts['problems'] = problems
    files = [p for p in sorted(output.rglob('*')) if p.is_file()]
    counts['downloaded_files'] = len(files)
    counts['sha256'] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--launch', type=Path, required=True)
    p.add_argument('--version', type=int, required=True)
    args = p.parse_args()
    launch = args.launch.resolve()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    kernel = json.loads((launch / 'kernel-metadata.json').read_text())['id'] + '/' + str(args.version)
    state = {'kernel': kernel, 'pid': os.getpid()}
    def save(**extra):
        state.update(extra, updated_utc=datetime.now(timezone.utc).isoformat())
        tmp = launch / 'monitor_status.json.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(launch / 'monitor_status.json')
    stream = None
    try:
        with (launch / 'live.log').open('a', buffering=1) as log:
            stream = subprocess.Popen([str(Path(sys.executable).parent / 'kaggle'), 'kernels', 'logs', kernel, '--follow'],
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            until = time.monotonic() + 4 * 3600
            save(state='monitoring', log_pid=stream.pid)
            while time.monotonic() < until:
                try:
                    result = api.kernels_status(kernel)
                    remote = str(result.status).split('.')[-1].upper()
                    save(remote_state=remote, last_error=None)
                    if remote in {'COMPLETE', 'ERROR', 'FAILED', 'CANCEL_ACKNOWLEDGED', 'CANCELED', 'CANCELLED'}:
                        break
                except Exception as exc:
                    save(last_error=f'{type(exc).__name__}: {exc}')
                time.sleep(30)
            else:
                save(state='monitor_timeout')
                return
            save(state='downloading')
            for attempt in range(6):
                try:
                    files, _ = api.kernels_output(kernel, str(launch / 'output'), quiet=True)
                    if not files:
                        raise RuntimeError('No output files exposed yet')
                    break
                except Exception as exc:
                    save(download_error=f'{type(exc).__name__}: {exc}')
                    if attempt == 5:
                        raise
                    time.sleep(30)
            audit = audit_output(launch / 'output')
            (launch / 'download_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
            save(state='downloaded', **{k: v for k, v in audit.items() if k != 'sha256'})
    except BaseException as exc:
        save(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if stream and stream.poll() is None:
            stream.terminate()
            try:
                stream.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stream.kill()
                stream.wait()


if __name__ == '__main__':
    main()
