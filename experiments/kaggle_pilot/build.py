"""Freeze a small baseline notebook. Does not submit or load a model locally."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from nemesis_sim.harness import schedule


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build_jobs():
    selected = {'payment', 'maintenance', 'shutdown'}
    rows = [row for row in schedule((2026111701, 2026111702, 2026111703))
            if row['scenario'] in selected]
    assert len(rows) == 72
    return {'0': rows[::2], '1': rows[1::2]}


def cell(kind, source):
    result = {'cell_type': kind, 'id': hashlib.sha256(source.encode()).hexdigest()[:12],
              'metadata': {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    launch = ROOT / 'runs' / 'kaggle' / stamp
    launch.mkdir(parents=True)
    config = {'model': 'ibm-granite/granite-4.1-3b',
              'revision': 'c0650403e44e78ec0262dab1c90914c65b196c4e',
              'condition': 'unchanged_baseline', 'model_seed': 20261117,
              'scenarios': ['payment', 'maintenance', 'shutdown'],
              'planned_episodes': 72, 'max_steps': 4, 'max_new_tokens': 500,
              'input_tokens': 4096, 'generation_seconds': 90,
              'worker_seconds': 900, 'notebook_seconds': 1140,
              'decoding': {'do_sample': False, 'temperature': None, 'num_beams': 1,
                           'repetition_penalty': 1.0, 'no_repeat_ngram_size': 0},
              'limitations': ['Short baseline interface pilot, not a full evaluation.',
                              'Three world seeds are scenario variants, not independent samples of greedy decoding.',
                              'No steering, fine-tuning, or adaptive case selection.',
                              'Training dataset creation is on hold pending example review.',
                              'Time limits can produce incomplete coverage; do not silently exclude failures.']}
    write(launch / 'config.json', config)
    write(launch / 'schedule.json', build_jobs())
    for name in ('worker.py', 'supervise.py', 'monitor.py', 'download_manifest.json'):
        shutil.copyfile(HERE / name, launch / name)
    shutil.copytree(ROOT / 'src', launch / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    names = ['config.json', 'schedule.json', 'worker.py', 'supervise.py', 'download_manifest.json']
    names += [p.relative_to(launch).as_posix() for p in sorted((launch / 'src').rglob('*.py'))]
    assets = {name: base64.b64encode((launch / name).read_bytes()).decode() for name in names}
    hashes = {name: hashlib.sha256((launch / name).read_bytes()).hexdigest() for name in names}
    write(launch / 'asset_hashes.json', hashes)
    source = '''import base64, hashlib, json, os, pathlib, subprocess, sys, time
START = time.time()
DEADLINE = START + 1140
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.22.1',
                'accelerate', 'huggingface_hub', 'safetensors'], check=True, timeout=180)
WORK = pathlib.Path('/kaggle/working/nemesis_pilot')
WORK.mkdir(parents=True, exist_ok=True)
'''
    source += f'ASSETS = {assets!r}\nHASHES = {hashes!r}\n'
    source += '''for name, encoded in ASSETS.items():
    data = base64.b64decode(encoded)
    assert hashlib.sha256(data).hexdigest() == HASHES[name]
    path = WORK / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
(WORK / 'asset_hashes.json').write_text(json.dumps(HASHES, indent=2))
import torch
gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print('GPU_INVENTORY', json.dumps(gpus), flush=True)
assert len(gpus) == 2 and all('T4' in name for name in gpus), 'Requires exactly two T4 GPUs'
(WORK / 'gpu_inventory.json').write_text(json.dumps({'gpu_count': len(gpus), 'names': gpus}))
download_code = r"""
import hashlib, json, pathlib, sys
from huggingface_hub import snapshot_download
work = pathlib.Path(sys.argv[1])
config = json.loads((work / 'config.json').read_text())
manifest = json.loads((work / 'download_manifest.json').read_text())
model = snapshot_download(config['model'], revision=config['revision'], cache_dir='/tmp/nemesis-hf',
                          allow_patterns=[f['name'] for f in manifest['files']], max_workers=2)
for item in manifest['files']:
    h = hashlib.sha256()
    with (pathlib.Path(model) / item['name']).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    assert h.hexdigest() == item['sha256'], item['name']
(work / 'model_path.txt').write_text(model)
print('CHECKPOINT_HASHES_VERIFIED', config['revision'], flush=True)
"""
subprocess.run([sys.executable, '-c', download_code, str(WORK)], check=True,
               timeout=max(1, DEADLINE-time.time()-60))
env = os.environ.copy()
env['PYTHONPATH'] = str(WORK / 'src')
subprocess.run([sys.executable, '-u', str(WORK / 'supervise.py'), '--work', str(WORK),
                '--model-path', (WORK / 'model_path.txt').read_text(), '--total-deadline', str(DEADLINE)],
               env=env, check=True, timeout=max(1, DEADLINE-time.time()+10))
print('PILOT_FINISHED', (WORK / 'results' / 'supervisor_status.json').read_text(), flush=True)
'''
    notebook = {'cells': [cell('markdown', '# Nemesis Sim: short Granite baseline\n\n'
                    '72 episodes, three scenario families, both prerequisite states, both guards and prompts. '
                    'Two independent unchanged Granite replicas on T4 x2. Greedy decoding, 500-token cap, '
                    'four action turns. 15-minute worker budget including model loading, 19-minute notebook '
                    'budget including setup; queue delay is additional. No training or steering. '
                    'Each reply and state transition is saved immediately. Partial coverage is reported.'),
                    cell('code', source)], 'metadata': {'kernelspec': {'display_name': 'Python 3',
                    'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    write(launch / 'nemesis_pilot.ipynb', notebook)
    metadata = {'id': 'aeeeeeeee/nemesis-granite-short-baseline', 'title': 'Nemesis Granite short baseline',
                'code_file': 'nemesis_pilot.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                'machine_shape': 'NvidiaTeslaT4', 'dataset_sources': [], 'competition_sources': [],
                'kernel_sources': [], 'model_sources': []}
    write(launch / 'kernel-metadata.json', metadata)
    pointer = {'path': str(launch), 'stage': 'prepared', 'kernel': metadata['id'],
               'planned_episodes': 72, 'gpu_count': 2, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'config_sha256': hashes['config.json']}
    write(launch / 'launch.json', pointer)
    write(HERE / 'active_launch.json', pointer)
    print(json.dumps(pointer))


if __name__ == '__main__':
    main()
