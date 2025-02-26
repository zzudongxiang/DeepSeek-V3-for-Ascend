import torch
from typing import List
from datetime import datetime
import torch.distributed as dist
from utils.logger import log_rank0
from utils.sample import sample_cpu as sample
from utils.progress import start_progress, stop_progress

generate_progress = 0

def reset_generate_progress():
    global generate_progress
    generate_progress = 0

@torch.inference_mode()
def batch_generate(model, prompt_tokens, eos_id, warmup=False) -> List[List[int]]:
    global generate_progress
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
    thread_tokens = start_progress(lambda: generate_progress,
                                   reset_generate_progress,
                                   description="Generate Progress")
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
    ttft_flag = True
    t0 = datetime.now()
    for cur_pos in range(min(prompt_lens), total_len):
        recv_handles = []
        global_bsz = tokens.shape[0]
        generate_progress = cur_pos / total_len
        recv_next_token = torch.zeros(global_bsz, dtype=torch.long, device=model.device)
        if ttft_flag:
            ttft_flag = False
            ttft = datetime.now() - t0
            t0 = datetime.now()
        for i in range(0, global_bsz, model.args.mini_batch_size):
            logits = model.forward(tokens[i: i + model.args.mini_batch_size, prev_pos:cur_pos],
                                   bsz_index=i // model.args.mini_batch_size,
                                   start_pos=prev_pos)
            if logits is not None and model.args.temperature > 0:
                next_token = sample(logits, model.args.temperature)
            elif logits is not None:
                next_token = logits.argmax(dim=-1)
            if model.pp_stage_num > 1 and model.pp_stage == 0:
                recv_handles.append(dist.irecv(recv_next_token[i: i + model.args.mini_batch_size], src=model.src_rank))
            elif model.pp_stage_num > 1 and model.pp_stage == model.pp_stage_num - 1:
                dist.isend(next_token, dst=model.dst_rank)
                recv_next_token[i: i + model.args.mini_batch_size] = next_token
        for handle in recv_handles:
            if not handle.is_completed():
                handle.wait()
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], recv_next_token)
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
    stop_progress(thread_tokens)
    log_rank0(f"TTFT: {ttft.total_seconds():.4f} seconds ({min(prompt_lens)} tokens)")
    log_rank0(f"Throughput: {tpot:.4f} tokens/s ({output_tokens} tokens for {dur:.4f} seconds)")
    return completion_tokens
