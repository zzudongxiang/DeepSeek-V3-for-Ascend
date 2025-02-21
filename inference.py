#!/bin/python

import os
import json
import torch
import importlib
from datetime import datetime
import torch.distributed as dist
from utils.logger import log_rank0
from argparse import ArgumentParser
from transformers import AutoTokenizer
from safetensors.torch import load_model
from utils.generate import batch_generate
from model.deepseek.args import ModelArgs

try:
    import torch_npu
    import mindspeed.megatron_adaptor
    default_device = "npu"
except:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"


def main(ckpt_path: str, config: str, model_args: list, model_name: str, startup_type: str = "online") -> None:
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(local_rank)
    torch.set_default_device("npu")
    torch.set_num_threads(8)
    torch.manual_seed(1234)

    # 根据模型名字导入模型
    model_path = f"model/{model_name}.py"
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    module_spec = importlib.util.spec_from_file_location("Transformer", model_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    # 构建模型
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    args_index = 0
    while args_index < len(model_args):
        key = model_args[args_index].replace("--", "")
        if hasattr(args, key):
            value = model_args[args_index + 1]
            setattr(args, key, value)
        else:
            log_rank0(f"Unknow args: {key}")
        args_index += 2

    with torch.device(default_device):
        model = module.Transformer(args)
        # model = torch.compile(model)
    log_rank0(args)

    # 模型预热
    now = datetime.now()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(batch_generate(model, [tokenizer.encode("DeepSeek")], tokenizer.eos_token_id, warmup=True)[0])
    dur = (datetime.now() - now).total_seconds()
    log_rank0(datetime.now(), f"Prepare DeepSeek Model in {dur:.2f} sec")

    # 加载模型权重
    if not args.use_random_weights:
        now = datetime.now()
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
        dur = (datetime.now() - now).total_seconds() / 60.0
        log_rank0(f"Load DeepSeek Weight in {dur:.2f} min")

    # 按照启动类型启动对应的实例
    if startup_type == "online":
        from utils.startup.online import run
        run(model, tokenizer)
    elif startup_type == "interactive":
        from utils.startup.interactive import run
        run(model, tokenizer)
    elif os.path.exists(startup_type):
        from utils.startup.offline import run
        run(model, tokenizer, startup_type)
    else:
        raise ValueError(f"Unknown startup type: {startup_type}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--startup-type", type=str, required=True)
    args, model_args = parser.parse_known_args()

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    try:
        main(args.ckpt_path, args.config_path, model_args, args.model_name, args.startup_type)
    except Exception as e:
        raise e
    finally:
        if world_size > 1:
            dist.destroy_process_group()
