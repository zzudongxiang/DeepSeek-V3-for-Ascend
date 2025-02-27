import torch
from datetime import datetime
import torch.distributed as dist
from utils.tools.logger import log_last_rank
from utils.generate.sample import sample_cpu as sample

generate_progress = 0

def reset_generate_progress():
    global generate_progress
    generate_progress = 0

def get_generate_progress():
    global generate_progress
    return generate_progress

@torch.inference_mode()
def batch_generate_pp(model, prompt_tokens, eos_id, warmup=False):
    global generate_progress
    max_new_tokens = 2 if warmup else model.args.max_new_tokens
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    ttft = 0
    prev_pos = 0
    ttft_flag = True
    recv_handles = {}
    t0 = datetime.now()
    prompt_mask = tokens != -1
    global_bsz = tokens.shape[0]
    finished = torch.tensor([False] * len(prompt_tokens))
    recv_next_token = torch.zeros(global_bsz, dtype=torch.long, device=model.device)
    mini_batch_size = model.args.mini_batch_size if model.pp_stage_num > 1 else global_bsz
    for cur_pos in range(min(prompt_lens), total_len):
        generate_progress = cur_pos / total_len
        if finished.all():
            break
        for i in range(0, global_bsz, mini_batch_size):
            if i in recv_handles and not recv_handles[i].is_completed():
                recv_handles[i].wait()
            if model.pp_stage == 0 and cur_pos > min(prompt_lens):
                replace_idx = cur_pos - 1
                next_token = torch.where(
                    prompt_mask[i: i + mini_batch_size, replace_idx],
                    tokens[i: i + mini_batch_size, replace_idx],
                    recv_next_token[i: i + mini_batch_size])
                tokens[i: i + mini_batch_size, replace_idx] = next_token
                finished[i: i + mini_batch_size] |= torch.logical_and(~prompt_mask[i: i + mini_batch_size, replace_idx], next_token == eos_id)
                if finished[i: i + mini_batch_size].all():
                    break
            logits = model.forward(
                tokens[i: i + mini_batch_size, prev_pos:cur_pos],
                bsz_index=i // mini_batch_size,
                start_pos=prev_pos)
            if logits is not None:
                next_token = sample(logits, model.args.temperature) if model.args.temperature > 0 else logits.argmax(dim=-1)
            if model.pp_stage == 0:
                recv_handles[i] = dist.irecv(recv_next_token[i: i + mini_batch_size], src=model.src_rank)
            elif model.pp_stage == model.pp_stage_num - 1:
                dist.isend(next_token, dst=model.dst_rank)
                next_token = torch.where(prompt_mask[i: i + mini_batch_size, cur_pos], tokens[i: i + mini_batch_size, cur_pos], next_token)
                tokens[i: i + mini_batch_size, cur_pos] = next_token
                finished[i: i + mini_batch_size] |= torch.logical_and(~prompt_mask[i: i + mini_batch_size, cur_pos], next_token == eos_id)
                if ttft_flag:
                    ttft_flag = False
                    ttft = (datetime.now() - t0).total_seconds()
                    t0 = datetime.now()
                if finished[i: i + mini_batch_size].all():
                    break
        prev_pos = cur_pos
    if model.pp_stage == 0:
        for handle in recv_handles.values():
            if not handle.is_completed():
                handle.wait()
        tokens[:, prev_pos] = torch.where(
            prompt_mask[:, prev_pos],
            tokens[:, prev_pos],
            recv_next_token)
    dur = (datetime.now() - t0).total_seconds()
    completion_tokens = []
    output_tokens = 0
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
    log_last_rank(f"TTFT: {ttft:.4f} seconds ({min(prompt_lens)} tokens)")
    log_last_rank(f"Throughput: {tpot:.4f} tokens/s ({output_tokens} tokens for {dur:.4f} seconds)")
    return completion_tokens
