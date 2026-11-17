"""Pinned baseline inference on one Kaggle GPU; no interventions or training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from nemesis_sim.engine import make_episode
from nemesis_sim.harness import run_episode
from nemesis_sim.store import Store
from nemesis_sim.world import canonical, make_town


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def repeated_tail(tokens):
    for width in range(1, 25):
        if len(tokens) >= max(24, width * 4):
            tail = tokens[-width:]
            if all(tokens[-width * (n + 1):-width * n] == tail for n in (1, 2, 3)):
                return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--shard', type=int, choices=(0, 1), required=True)
    p.add_argument('--deadline', type=float, required=True)
    args = p.parse_args()
    work = args.work.resolve()
    out = work / 'results'
    out.mkdir(exist_ok=True)
    config = json.loads((work / 'config.json').read_text())
    jobs = json.loads((work / 'schedule.json').read_text())[str(args.shard)]
    stop = False
    def on_signal(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    completed, replies = 0, 0
    def status(state, **extra):
        row = {'state': state, 'shard': args.shard, 'completed_episodes': completed,
               'generated_replies': replies, 'planned_episodes': len(jobs),
               'updated_utc': datetime.now(timezone.utc).isoformat(), **extra}
        atomic_json(out / f'status_{args.shard}.json', row)
        print(canonical(row), flush=True)
    def halt_reason():
        if stop or (out / 'STOP').exists():
            return 'user_stop'
        if time.time() >= args.deadline:
            return 'session_budget'
        return None
    status('loading')
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config['model_seed'])
        torch.cuda.manual_seed_all(config['model_seed'])
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
        assert torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0)
        tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.float16, device_map='cuda:0', attn_implementation='sdpa').eval()
        assert model.config.model_type == 'granite'
        runtime = {'torch': torch.__version__, 'transformers': transformers.__version__,
                   'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                   'gpu': torch.cuda.get_device_name(0), 'dtype': 'float16',
                   'attention': 'sdpa_math_fp32_intermediates', 'condition': 'unchanged_baseline',
                   'model': config['model'], 'revision': config['revision'],
                   'config_sha256': hashlib.sha256((work / 'config.json').read_bytes()).hexdigest(),
                   'decoding': config['decoding']}
        atomic_json(out / f'runtime_{args.shard}.json', runtime)
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])

        def respond(messages, *, max_new_tokens):
            nonlocal replies
            start = time.monotonic()
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch = tok(prompt, add_special_tokens=False, return_tensors='pt')
            prompt_tokens = batch['input_ids'][0].tolist()
            generated, past, reason = [], None, None
            if len(prompt_tokens) > config['input_tokens']:
                reason = 'context_limit'
            else:
                batch = batch.to('cuda:0')
                ids, mask = batch['input_ids'], batch['attention_mask']
                try:
                    with torch.inference_mode():
                        for _ in range(max_new_tokens):
                            reason = halt_reason()
                            if reason:
                                break
                            if time.monotonic() - start >= config['generation_seconds']:
                                reason = 'generation_timeout'
                                break
                            positions = mask.long().cumsum(-1) - 1
                            if past is None:
                                for offset in range(0, ids.shape[1], 256):
                                    if halt_reason():
                                        reason = halt_reason()
                                        break
                                    end = min(offset + 256, ids.shape[1])
                                    output = model(input_ids=ids[:, offset:end], attention_mask=mask[:, :end],
                                                   position_ids=positions[:, offset:end], past_key_values=past,
                                                   use_cache=True, logits_to_keep=1)
                                    past = output.past_key_values
                                    if end < ids.shape[1]:
                                        del output
                                if reason:
                                    break
                            else:
                                output = model(input_ids=ids, attention_mask=mask,
                                               position_ids=positions[:, -1:], past_key_values=past,
                                               use_cache=True, logits_to_keep=1)
                                past = output.past_key_values
                            logits = output.logits[:, -1, :].float()
                            if not bool(torch.isfinite(logits).all()):
                                reason = 'nonfinite_logits'
                                del logits, output
                                break
                            token = int(logits.argmax(-1).item())
                            del logits, output
                            generated.append(token)
                            if token in eos:
                                reason = 'eos'
                                break
                            if len(generated) % 8 == 0 and repeated_tail(generated):
                                reason = 'repetition_stop'
                                break
                            ids = torch.tensor([[token]], device='cuda:0')
                            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device='cuda:0')], dim=1)
                except torch.OutOfMemoryError:
                    reason = 'cuda_out_of_memory'
                finally:
                    del past, ids, mask, batch
                    torch.cuda.empty_cache()
            reason = reason or 'token_limit'
            text_tokens = generated[:-1] if generated and generated[-1] in eos else generated
            replies += 1
            result = {'text': tok.decode(text_tokens, skip_special_tokens=False), 'stop_reason': reason,
                      'raw_output': tok.decode(generated, skip_special_tokens=False),
                      'token_ids': generated, 'prompt_token_ids': prompt_tokens,
                      'rendered_prompt': prompt, 'generated_tokens': len(generated),
                      'generation_seconds': time.monotonic() - start, 'runtime': runtime}
            status('reply_generated', stop_reason=reason, generated_tokens=len(generated))
            return result

        end_reason = 'complete'
        for job in jobs:
            if halt_reason() or args.deadline - time.time() < 20:
                end_reason = halt_reason() or 'session_budget'
                break
            status('generating', job=job)
            episode = make_episode(make_town(job['seed']), job['scenario'], seed=job['seed'],
                                   ready=job['ready'], guard=job['guard'], prompt=job['prompt'],
                                   max_steps=config['max_steps'])
            path = out / (job['job_id'] + '.sqlite')
            with Store.create(path, episode, {'evidence_type': 'model_inference', 'job': job,
                                              'runtime': runtime}) as store:
                summary = run_episode(store, respond, max_new_tokens=config['max_new_tokens'])
                summary['job'] = job
                summary['audit'] = store.audit()
                store.export(path.with_suffix('.jsonl'))
                events = store.events()
            with (out / f'episodes_{args.shard}.jsonl').open('a') as stream:
                stream.write(canonical(summary) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            completed += 1
            status('episode_saved', job=job, summary=summary)
            fatal = {'nonfinite_logits', 'cuda_out_of_memory', 'generation_timeout'}
            if any(e['stop_reason'] in fatal for e in events):
                raise RuntimeError('Infrastructure generation failure; saved evidence retained')
        status('complete' if end_reason == 'complete' else 'partial', reason=end_reason)
    except BaseException as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
