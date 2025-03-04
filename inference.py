#!/bin/python

import os
import torch
import torch_npu
import importlib
from datetime import datetime
import torch.distributed as dist
import mindspeed.megatron_adaptor
from argparse import ArgumentParser
from transformers import AutoTokenizer
from model.deepseek.args import get_model_args
from utils.load_model import load_model_weight
from utils.generate.entry import batch_generate
from utils.tools.logger import log, log_last_rank
from utils.startup.offline import run as offline_run
from utils.startup.interactive import run as interactive_run

def main(ckpt_path, config, model_args, model_name, startup_type="online"):
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
    args = get_model_args(config, model_args)
    num_stages = 1 if args.pp_layer_list is None else len(args.pp_layer_list)
    assert world_size % num_stages == 0
    pp_group_size = world_size // num_stages
    pp_stage = rank // pp_group_size
    tp_group_ranks = [pp_stage * pp_group_size + i for i in range(pp_group_size)]
    tp_group = dist.new_group(tp_group_ranks, use_local_synchronization=True)
    with torch.device("npu"):
        model = module.Transformer(args, "npu", tp_group, pp_stage)

    # 模型预热
    t0 = datetime.now()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    completion_tokens = batch_generate(model, [tokenizer.encode("DeepSeek")], tokenizer.eos_token_id, warmup=True)
    if model.pp_stage_num == 0 or model.pp_stage == model.pp_stage_num - 1:
        tokenizer.decode(completion_tokens[0])
    dur = (datetime.now() - t0).total_seconds()
    log_last_rank(f"Prepare DeepSeek Model in {dur:.2f} sec")

    # 加载模型权重
    if not args.use_random_weights:
        load_model_weight(model, ckpt_path, tp_group)
    model.update_wkv_b_cache()

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
        pass
        interactive_run(model, tokenizer)
    elif os.path.exists(startup_type):
        offline_run(model, tokenizer, startup_type)
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
        log(f"rank {rank} Exit.")
    except Exception as e:
        raise e
    finally:
        if world_size > 1:
            dist.destroy_process_group()
