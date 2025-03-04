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
    # 获取模型的状态字典
    model_state_dict = model.state_dict()
    total_num = len(model_state_dict)
    load_index = 0
    
    # 从模型获取张量并行和流水线并行信息
    pp_stage = model.pp_stage
    pp_size = model.pp_stage_num
    
    # 获取当前进程的tp_rank和tp_size
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'tp_group') and model.transformer.tp_group is not None:
        tp_group = model.transformer.tp_group
        tp_rank = dist.get_rank(tp_group) if dist.is_initialized() else 0
        tp_size = dist.get_world_size(tp_group) if dist.is_initialized() else 1
    else:
        tp_rank = rank % 2  # 默认值，假设2卡张量并行
        tp_size = 2         # 默认值
    
    # 启动进度条
    thread_tokens = start_progress(
        lambda: load_index / (total_num * tp_size),
        description=f"Load Model Weight (PP Stage {pp_stage}/{pp_size-1}, TP Rank {tp_rank}/{tp_size-1})",
        interval=30
    )
    
    # 确保所有进程同步开始
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
    
    # 遍历所有张量并行的分片
    for i in range(tp_size):
        ckpt_file_path = os.path.join(ckpt_path, f"model{i}-mp{tp_size}.safetensors")
        
        if not os.path.exists(ckpt_file_path):
            log_rank0(f"Warning: Checkpoint file {ckpt_file_path} not found. Skipping.")
            load_index += len(model_state_dict)  # 更新加载计数
            continue
            
        with safe_open(ckpt_file_path, framework="pt") as f:
            available_keys = set(f.keys())
            
            for k in model_state_dict:
                # 跳过不在当前流水线阶段的层参数
                if pp_size > 1:
                    # 获取层编号 (假设格式如 layers.10.xxx)
                    if 'layers.' in k:
                        layer_parts = k.split('layers.')[1].split('.')
                        if layer_parts and layer_parts[0].isdigit():
                            layer_id = int(layer_parts[0])
                            start_layer = model.transformer.start_layer_id
                            end_layer = model.transformer.end_layer_id
                            if layer_id < start_layer or layer_id >= end_layer:
                                load_index += 1
                                continue
                    
                    # 如果不是第一阶段，跳过embedding参数
                    if pp_stage != 0 and 'embed' in k:
                        load_index += 1
                        continue
                        
                    # 如果不是最后阶段，跳过norm和head参数
                    if pp_stage != pp_size - 1 and ('norm' in k or 'head' in k):
                        load_index += 1
                        continue
                
                # 加载参数
                # 1. 专家模型权重 (.experts.) 从所有分片加载
                # 2. 其他权重只从对应的张量并行分片加载
                if k in available_keys and (".experts." in k or i == tp_rank):
                    try:
                        param = f.get_tensor(k)
                        if param.shape != model_state_dict[k].shape:
                            log_rank0(f"Warning: Shape mismatch for {k}: checkpoint {param.shape} vs model {model_state_dict[k].shape}")
                            # 尝试调整形状
                            if len(param.shape) == len(model_state_dict[k].shape):
                                # 如果维度相同但大小不同，尝试裁剪或填充
                                model_state_dict[k].copy_(param.reshape(model_state_dict[k].shape))
                            else:
                                log_rank0(f"Error: Cannot load parameter {k} due to incompatible shapes")
                        else:
                            model_state_dict[k].copy_(param)
                    except Exception as e:
                        log_rank0(f"Error loading parameter {k}: {e}")
                        
                load_index += 1
    
    # 确保所有进程同步结束
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
        
    stop_progress(thread_tokens)
    
    # 报告内存使用情况
    if torch.cuda.is_available():
        hbm = torch.cuda.memory_cached() / 1024.0 / 1024.0 / 1024.0
        log_rank0(f"{hbm:.2f}GB Memory in Used")
    else:
        log_rank0("CUDA not available, memory usage cannot be reported")

def main(ckpt_path: str, config: str, model_args: list, model_name: str, startup_type: str = "online") -> None:
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(local_rank)
    torch.set_default_device("npu")
    torch.set_num_threads(8)
    torch.manual_seed(1234)

    # 解析模型参数
    pp_stages = None
    tp_size = 1
    pp_size = 1
    args_index = 0

    offload_cpu = False

    while args_index < len(model_args):
        arg = model_args[args_index].replace("--", "")
        if arg == "pp-layers":
            if args_index < len(model_args) - 1:
                pp_layers_str = model_args[args_index + 1]
                pp_stages = [int(x) for x in pp_layers_str.split(",")]
                pp_size = len(pp_stages)
                args_index += 2
            else:
                args_index += 1
        elif arg == "tp-size":
            if args_index < len(model_args) - 1:
                tp_size = int(model_args[args_index + 1])
                args_index += 2
            else:
                args_index += 1
        elif arg == "offload-cpu":
            if args_index < len(model_args) - 1:
                offload_cpu = int(model_args[args_index + 1])
                args_index += 2
            else:
                args_index += 1
        else:
            args_index += 1

    # 设置张量并行组和流水线并行组
    if dist.is_initialized() and world_size > 1:
        global_rank = dist.get_rank()
        # 计算当前进程的pp_stage
        pp_stage = global_rank // tp_size if pp_size > 1 else 0
        # 创建张量并行组
        tp_ranks = [global_rank // tp_size * tp_size + i for i in range(tp_size)]
        tp_group = dist.new_group(tp_ranks)
    else:
        pp_stage = 0
        tp_group = None

    # 根据模型名字导入模型
    model_path = f"model/{model_name}.py"
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    module_spec = importlib.util.spec_from_file_location("DeepSeekMP", model_path)
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
            if args_index < len(model_args) - 1:
                value = model_args[args_index + 1]
                args_index += 2
            else:
                args_index += 1
                continue
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            log_rank0(f"Unknow args: {key} = {value}")
    log_rank0(args)

    with torch.device(default_device):
        # 使用DeepSeekMP模型，并传递流水线和张量并行参数
        model = module.DeepSeekMP(args, default_device, tp_group, pp_stage, pp_stages)
        # model = torch.compile(model)

    # 只有第一个流水线阶段的第一个进程需要进行预热
    # 对于多阶段流水线，只有最后一个阶段会返回结果
    if (pp_stages is None) or (pp_stage == 0 and rank % tp_size == 0) or (pp_stage == len(pp_stages) - 1 and rank % tp_size == 0):
        # 模型预热
        t0 = datetime.now()
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        tokenizer.decode(batch_generate(model, [tokenizer.encode("DeepSeek")], tokenizer.eos_token_id, warmup=True)[0])
        dur = (datetime.now() - t0).total_seconds()
        log_rank0(f"Prepare DeepSeek Model in {dur:.2f} sec")
    else:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

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
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--pp-layers", type=str, default=None, help="Pipeline parallelism layer distribution, comma separated")
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