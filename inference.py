#!/bin/python

import os
import json
import torch
import importlib
from datetime import datetime
import torch.distributed as dist
from safetensors import safe_open
from utils.logger import log_rank0
from argparse import ArgumentParser
from transformers import AutoTokenizer
from utils.generate import batch_generate
from model.deepseek.args import ModelArgs
from utils.progress import start_progress, stop_progress

try:
    import torch_npu
    import mindspeed.megatron_adaptor
    default_device = "npu"
except:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{datetime.now()}] torch_npu not found, use torch instead")


def load_model_weight(model, ckpt_path):
    progress_value = 0
    ckpt_file_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    thread_tokens = start_progress(lambda: progress_value,
                                   description="Load Weight Progress",
                                   interval=30)
    model_state_dict = model.state_dict()
    with safe_open(ckpt_file_path, framework="pt") as f:
        load_index = 0
        total_num = len(f.keys())
        for k in f.keys():
            assert k in model_state_dict
            model_state_dict[k].copy_(f.get_tensor(k))
            progress_value = load_index / total_num
            load_index += 1
    stop_progress(thread_tokens)

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
        arg = model_args[args_index].replace("--", "")
        if "=" in arg:
            key = arg.split("=")[0].replace("-", "_")
            value = "=".join(arg.split("=")[1:])
            args_index += 1
        else:
            key = arg.replace("-", "_")
            value = model_args[args_index + 1]
            args_index += 2
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            log_rank0(f"Unknow args: {key} = {value}")
    log_rank0(args)

    with torch.device(default_device):
        model = module.Transformer(args, default_device)
        # model = torch.compile(model)

    # 模型预热
    t0 = datetime.now()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(batch_generate(model, [tokenizer.encode("DeepSeek")], tokenizer.eos_token_id, warmup=True)[0])
    dur = (datetime.now() - t0).total_seconds()
    log_rank0(f"Prepare DeepSeek Model in {dur:.2f} sec")

    # 加载模型权重
    if not args.use_random_weights:
        load_model_weight(model, ckpt_path)

    # 按照启动类型启动对应的实例
    if startup_type == "online":
        api_port = 5000
        service_port = 5001
        from utils.startup.online import run_master, run_slaver
        dist.barrier()
        if rank == 0:
            run_master(model, tokenizer, api_port, service_port)
        else:
            run_slaver(model, tokenizer, service_port)
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
