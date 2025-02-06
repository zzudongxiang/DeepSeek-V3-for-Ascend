#!/bin/python

import os
import json
import torch
from typing import List
from typing import Literal
from datetime import datetime
import torch.distributed as dist
from argparse import ArgumentParser
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model.utils.tools import sample
from model.deepseek_origin import Transformer, ModelArgs
from model.deepseek import writer_finished, writer_split, print_flops

default_device: Literal["cuda", "npu", "cpu"] = "cuda"

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
    rank = int(os.getenv("RANK", "0"))
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device=default_device)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=default_device)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device=default_device)
    prompt_mask = tokens != -1
    start = datetime.now()
    print_flops(flush_only=True)
    for cur_pos in range(min(prompt_lens), total_len):
        logits, _ = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if prev_pos == 0:
            t0 = datetime.now()
            writer_split()
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    end = datetime.now()
    if rank == 0:
        prefill_lens = min(prompt_lens)
        ttft = (t0 - start).total_seconds()
        decode_time = (end - t0).total_seconds()
        decode_lens = (cur_pos - prefill_lens + 1) * len(prompt_tokens)
        throughput = decode_lens / decode_time
        print(f"TTFT: {ttft:.4f} seconds for {prefill_lens} tokens")
        print(f"Throughput: {throughput:.2f} tokens/s ({decode_time:.2f} seconds for {decode_lens} tokens)")
        speed, flops, dur, cube_flops, vector_flops = print_flops()
        # print(f"{ttft},{prefill_lens},{throughput},{decode_time},{decode_lens},{speed},{flops},{dur},{cube_flops},{vector_flops}")
        print("-" * 50)
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    writer_finished()
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    use_random_weights: bool = False,
    profiling: bool = False,
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
    else:
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
    # if not profiling:
    #     tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    if not use_random_weights:
        print(datetime.now(), "start load weights")
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
        print(datetime.now(), "load weights finished")

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
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--profiling", action="store_true") # 是否抓取Trace
    parser.add_argument("--local-rank", type=int, default=0) # 为了防止Debug的时候自动添加该参数导致报错
    parser.add_argument("--use-random-weights", action="store_true") # 是否使用随机数加载权重
    args = parser.parse_args()
    assert args.input_file or args.interactive
    if args.profiling:
        from torch.profiler import profile, ProfilerActivity
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("log"),
        ) as prof:
            main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.use_random_weights, args.profiling)
    else:
        main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature, args.use_random_weights)
