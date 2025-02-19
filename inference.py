#!/bin/python

import os
import json
from argparse import ArgumentParser
from typing import List

import torch
from datetime import datetime
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model.utils.tools import sample_cpu
from model.deepseek_int8 import Transformer, ModelArgs

default_device = "cuda"

try:
    import torch_npu
    import mindspeed.megatron_adaptor
    default_device = "npu"
except:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch_npu not found, using {default_device}")

@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device=default_device)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=default_device)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device=default_device)
    prompt_mask = tokens != -1
    t0 = datetime.now()
    ttft_flag = True
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if ttft_flag:
            ttft_flag = False
            ttft = datetime.now() - t0
            print(f"TTFT: {ttft.total_seconds():.4f} seconds ({cur_pos} tokens)")
            t0 = datetime.now()
        if temperature > 0:
            next_token = sample_cpu(logits, temperature)
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
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
            output_tokens += toks.index(eos_id)
        else:
            output_tokens += len(toks)
        completion_tokens.append(toks)
    tpot = output_tokens / dur
    print(f"Throughput: {tpot:.4f} tokens/s ({output_tokens} tokens for {dur} sec)")
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    if default_device == "npu":
        torch_npu.npu.set_device(local_rank)
    elif default_device == "cuda":
        torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device(default_device):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    now = datetime.now()
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    dur = (datetime.now() - now).total_seconds()
    print(datetime.now(), f"Prepare DeepSeek Model in {dur:.2f} sec")

    now = datetime.now()
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    dur = (datetime.now() - now).total_seconds() / 60.0
    print(datetime.now(), f"Load DeepSeek Weight in {dur:.2f} min")

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        now = datetime.now()
        # from torch.profiler import profile, ProfilerActivity
        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU],
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler("log"),
        # ) as prof:
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()
        dur = (datetime.now() - now).total_seconds() / 60.0
        tokens = 0
        for item in completion_tokens:
            tokens += len(item)
        print(datetime.now(), f"DeepSeek Generate {tokens} tokens in {dur:.2f} min")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()
    assert args.input_file or args.interactive
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
