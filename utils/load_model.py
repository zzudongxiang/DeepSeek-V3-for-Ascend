import os
import torch
import torch.distributed as dist
from safetensors import safe_open
from utils.tools.logger import log_rank0
from utils.tools.progress import start_progress, stop_progress

def load_model_weight(model, ckpt_path, tp_group):
    model_state_dict = model.state_dict()
    total_num = len(model_state_dict)
    load_index = 0
    tp_rank = dist.get_rank(tp_group) if dist.is_initialized() else 0
    tp_num = dist.get_world_size(tp_group) if dist.is_initialized() else 1
    thread_tokens = start_progress(lambda: load_index / (total_num * tp_num),
                                   description="Load Model Weight",
                                   interval=30)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
    for i in range(tp_num):
        ckpt_file_path = os.path.join(ckpt_path, f"model{i}-mp{tp_num}.safetensors")
        with safe_open(ckpt_file_path, framework="pt") as f:
            for k in model_state_dict:
                if k in f.keys() and (".experts." in k or i == tp_rank):
                    model_state_dict[k].copy_(f.get_tensor(k))
                load_index += 1
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
    stop_progress(thread_tokens)
    hbm = torch.cuda.memory_cached() / 1024.0 / 1024.0 / 1024.0
    log_rank0(f"{hbm:.2f}GB Memory in Used")
