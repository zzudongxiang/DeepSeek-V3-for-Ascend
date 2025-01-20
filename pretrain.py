#!/bin/python

import os
import json
import math
import torch
import numpy as np
from typing import Literal
from datetime import datetime
import torch.distributed as dist
from argparse import ArgumentParser
from torch.nn import functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model.utils.tools import sample
from model.deepseek import Transformer, ModelArgs

default_device: Literal["cuda", "npu", "cpu"] = "cuda"

try:
    import torch_npu
    from torch_npu.npu import amp
    import mindspeed.megatron_adaptor
    default_device = "npu"
except:
    from torch import amp
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch_npu not found, using {default_device}")

batch_size = 1
max_iters = 100
log_interval = 1

learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.99
decay_lr = True
weight_decay = 1e-1

warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

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
    x = x.pin_memory().to(default_device, non_blocking=True)
    y = y.pin_memory().to(default_device, non_blocking=True)
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
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    assert batch_size < args.max_batch_size
    print(args)
    with torch.device(default_device):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    ctx = amp.autocast(dtype=torch.bfloat16)
    optimizer = model.configure_optimizers(
        weight_decay,
        learning_rate,
        (beta1, beta2),
        default_device,
    )
    if not use_random_weights:
        print(datetime.now(), "start load weights")
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
        print(datetime.now(), "load weights finished")
    iter_num = 0
    while iter_num < max_iters:
        iter_num += 1
        X, Y = get_batch(tokenizer, args.max_seq_len, args.max_batch_size)
        t0 = datetime.now()
        # 计算学习率
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # 对Micro-Batch进行FWD和BWD训练
        with ctx:
            logits, loss = model.forward(X, start_pos=0, targets=Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        # 输出训练进度(这是CPU-GPU的同步点)
        if iter_num % log_interval == 0:
            dt = datetime.now() - t0
            lossf = loss.item()
            print(f"[{dt*1000:.2f}ms] {iter_num}: loss={lossf:.4f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--use-random-weights", action="store_true")
    parser.add_argument("--input-file", type=str, default="scripts/shakespeare.txt")
    args = parser.parse_args()
    dataset_path = args.input_file
    main(args.ckpt_path, args.config, args.use_random_weights)