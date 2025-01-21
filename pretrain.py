#!/bin/python

import re
import os
import json
import math
import torch
import numpy as np
from typing import Literal
from datetime import datetime
import torch.distributed as dist
from contextlib import nullcontext
from argparse import ArgumentParser
from torch.nn import functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_model
from torch.nn.parallel import DistributedDataParallel

from model.utils.tools import sample
from model.deepseek import writer_finished
from model.deepseek_origin import Transformer, ModelArgs

# BUG: 暂时不支持bf16的混合精度训练
default_device: Literal["cuda", "npu", "cpu"] = "cuda"
default_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}["float16"]

try:
    import torch_npu
    from torch_npu.npu import amp
    import mindspeed.megatron_adaptor
    default_device = "npu" # npu:7
except:
    from torch import amp
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch_npu not found, using {default_device}")

seq_len = 1024
batch_size = 1
max_iters = 100
log_interval = 1
learning_rate = 1e-6
all_batch = None
dataset_path = "dataset.txt"

def get_batch(tokenizer, seq_len):
    global all_batch, batch_size
    if all_batch is None:
        with open(dataset_path, "r+") as file:
            all_batch = file.readlines()
        all_batch = '\n'.join(all_batch)
        all_batch = np.array(tokenizer.encode(all_batch, max_length=len(all_batch), truncation=True), dtype=np.int64)
    ix = torch.randint(len(all_batch) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy(all_batch[i : i + seq_len]) for i in ix])
    y = torch.stack([torch.from_numpy(all_batch[i + 1 : i + 1 + seq_len]) for i in ix])
    x = x.to(default_device)
    y = y.to(default_device)
    return x, y

def main(
    ckpt_path: str,
    config: str,
    use_random_weights: bool = False,
) -> None:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    if default_device == "npu":
        torch_npu.npu.set_device(local_rank)
    else:
        torch.cuda.set_device(local_rank)
    torch.set_default_dtype(default_dtype)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    assert batch_size <= args.max_batch_size and seq_len <= args.max_seq_len
    print(args)
    with torch.device(default_device):
        model = Transformer(args, default_dtype)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if not use_random_weights:
        print(datetime.now(), "start load weights")
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
        print(datetime.now(), "load weights finished")
    if world_size > 1:
        model = DistributedDataParallel(model)
    if re.match(r"npu(\:\d+)?", default_device):
        ctx = amp.autocast(dtype=default_dtype)
        optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=learning_rate)
    else:
        ctx = nullcontext() if default_device == 'cpu' else torch.amp.autocast(device_type=default_device, dtype=default_dtype)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    iter_num = 0
    while iter_num < max_iters:
        iter_num += 1
        t0 = datetime.now()
        optimizer.zero_grad(set_to_none=False)
        X, Y = get_batch(tokenizer, seq_len)
        with ctx:
            _, loss = model.forward(X, targets=Y)
        loss.backward()
        optimizer.step()
        if iter_num % log_interval == 0:
            print(f"[{(datetime.now() - t0).total_seconds():.2f}s] {iter_num}: loss={loss.item():.4f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--use-random-weights", action="store_true")
    parser.add_argument("--input-file", type=str, default="scripts/shakespeare.txt")
    args = parser.parse_args()
    dataset_path = args.input_file
    try:
        main(args.ckpt_path, args.config, args.use_random_weights)
    except Exception as e:
        raise e
    finally:
        writer_finished()
