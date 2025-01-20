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
from model import Transformer, ModelArgs
from safetensors.torch import load_model

default_device: Literal["cuda", "npu"] = "cuda"
try:
    import torch_npu
    from torch_npu.npu import amp
    import mindspeed.megatron_adaptor
    default_device = "npu"
except:
    print("torch_npu not found, using cuda")


learning_rate = 1e-3  # 最大的学习率
beta1 = 0.9  # Adam Beta 1
beta2 = 0.99  # Adam Beta 2
decay_lr = True  # 学习率是否衰减
weight_decay = 1e-1  # Weight Decay
grad_clip = 1.0  # 如果Grad Clip == 0.0则禁用
warmup_iters = 100  # warm up步数
lr_decay_iters = 5000  # should be ~= max_iters per Chinchilla
min_lr = 1e-4  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
compile = False  # 启用模型编译 (仅支持Pytorch2.0以上)
max_iters = 100  # 最大的迭代次数
gradient_accumulation_steps = 1  # GA的次数
log_interval = 1  # log打印的间隔, 当log_interval > max_iters时不输出中间Log

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

def get_batch(tokenizer):
    data = np.array([tokenizer.encode("Hello World!")])
    return torch.from_numpy(data.astype(np.int64)).to(device=default_device)

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
    print(args)
    with torch.device(default_device):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    scaler = amp.GradScaler()
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
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model, backend="npu")
    iter_num = 0
    best_val_loss = 1e9
    tokens = get_batch(tokenizer)
    while iter_num < max_iters:
        iter_num += 1
        t0 = datetime.now()
        # 计算学习率
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # 对Micro-Batch进行FWD和BWD训练
        for _ in range(gradient_accumulation_steps):
            # total_len = X.shape[1]
            # tokens = torch.full((len(X), total_len), -1, dtype=torch.long, device=default_device)
            # for i, t in enumerate(X):
            #     tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=default_device)
            with ctx:
                logits = model.forward(tokens[:, :-1])
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tokens[:, -1].view(-1), ignore_index=-1)
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            tokens = get_batch(tokenizer)
        # 更新梯度
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=False)
        # 输出训练进度(这是CPU-GPU的同步点)
        if iter_num % log_interval == 0:
            dt = datetime.now() - t0
            lossf = loss.item() * gradient_accumulation_steps
            print(f"[{dt*1000:.2f}ms] {iter_num}: loss={lossf:.4f}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--use-random-weights", action="store_true")
    args = parser.parse_args()
    main(args.ckpt_path, args.config, args.use_random_weights)