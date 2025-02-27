import torch
from typing import List
import torch.distributed as dist
from utils.generate.v3 import batch_generate_v3
from utils.generate.pp import batch_generate_pp
from utils.tools.progress import start_progress, stop_progress

@torch.inference_mode()
def batch_generate(model, prompt_tokens, eos_id, warmup=False) -> List[List[int]]:
    if model.pp_stage_num > 1:
        from utils.generate.pp import reset_generate_progress, get_generate_progress
    else:
        from utils.generate.v3 import reset_generate_progress, get_generate_progress
    thread_tokens = start_progress(
        get_generate_progress,
        reset_generate_progress,
        description="Generate Progress")
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
    if model.pp_stage_num > 1:
        ret = batch_generate_pp(model, prompt_tokens, eos_id, warmup)
    else:
        ret = batch_generate_v3(model, prompt_tokens, eos_id, warmup)
    stop_progress(thread_tokens)
    return ret
