import os
import torch
from datetime import datetime
import torch.distributed as dist

try:
    import torch_npu
    import mindspeed.megatron_adaptor
except:
    print(f"[{datetime.now()}] torch_npu not found, use torch instead")

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
if world_size > 1 and not dist.is_initialized():
    dist.init_process_group("nccl")

def log_rank0(*args, **kwargs):
    if rank != 0:
        return
    print(f"[{datetime.now()}]", end=" ")
    print(*args, **kwargs)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"
