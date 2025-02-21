import torch.distributed as dist
from utils.logger import log_rank0
from utils.generate import batch_generate

def run(model, tokenizer):
    messages = []
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    while True:
        if world_size == 1:
            prompt = input(">>> ")
        elif rank == 0:
            prompt = input(">>> ")
            objects = [prompt]
            dist.broadcast_object_list(objects, 0)
        else:
            objects = [None]
            dist.broadcast_object_list(objects, 0)
            prompt = objects[0]
        if prompt == "/exit":
            break
        elif prompt == "/clear":
            messages.clear()
            continue
        messages.append({"role": "user", "content": prompt})
        prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        completion_tokens = batch_generate(model, [prompt_tokens], tokenizer.eos_token_id)
        completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
        log_rank0(completion)
        messages.append({"role": "assistant", "content": completion})
