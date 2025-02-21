import torch
import threading
from typing import List
from datetime import datetime
from utils.sample import sample_cpu as sample
from utils.logger import log_rank0, format_time

generate_progress = 0

def periodic_print(stop_event):
    t0 = datetime.now()
    start_time = t0
    while not stop_event.is_set():
        current_time = datetime.now()
        elapsed = (current_time - start_time).total_seconds()
        if (current_time - t0).total_seconds() > 5:
            progress = generate_progress * 100
            if generate_progress > 0:
                eta = (elapsed / progress) * (100 - progress)
                log_rank0(f"Generate Progress: {progress:.2f}% | "
                         f"Elapsed: {format_time(elapsed)} | "
                         f"ETA: {format_time(eta)}")
            else:
                log_rank0(f"Generate Progress: {progress:.2f}% | "
                         f"Elapsed: {format_time(elapsed)}")
            t0 = current_time
        stop_event.wait(1)
    total_time = (datetime.now() - start_time).total_seconds()
    log_rank0(f"Generate Progress: 100% | Total time: {format_time(total_time)}")

@torch.inference_mode()
def batch_generate(model, prompt_tokens, eos_id, warmup=False) -> List[List[int]]:
    global generate_progress
    stop_event = threading.Event()
    thread = threading.Thread(target=periodic_print, args=(stop_event,))
    thread.daemon = True
    thread.start()

    max_new_tokens = 2 if warmup else model.args.max_new_tokens
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1
    t0 = datetime.now()
    ttft_flag = True
    for cur_pos in range(min(prompt_lens), total_len):
        generate_progress = cur_pos / total_len
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if ttft_flag:
            ttft_flag = False
            ttft = datetime.now() - t0
            t0 = datetime.now()
        if model.args.temperature > 0:
            next_token = sample(logits, model.args.temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    output_tokens = 0
    dur = (datetime.now() - t0).total_seconds()
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i] + max_new_tokens]
        if eos_id in toks:
            toks_index = toks.index(eos_id)
            output_tokens += toks_index
            toks = toks[:toks_index]
        else:
            output_tokens += len(toks)
        completion_tokens.append(toks)
    tpot = output_tokens / dur
    stop_event.set()
    thread.join()
    log_rank0(f"TTFT: {ttft.total_seconds():.4f} seconds ({min(prompt_lens)} tokens)")
    log_rank0(f"Throughput: {tpot:.4f} tokens/s ({output_tokens} tokens for {dur:.4f} seconds)")
    return completion_tokens
