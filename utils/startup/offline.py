import torch.distributed as dist
from utils.tools.logger import log
from utils.generate.entry import batch_generate

def run(model, tokenizer, file_path):
    with open(file_path) as f:
        prompts = [line.strip() for line in f.readlines()]
    if len(prompts) > model.args.max_batch_size:
        prompts = prompts[:model.args.max_batch_size]
    prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
    completion_tokens = batch_generate(model, prompt_tokens, tokenizer.eos_token_id)
    if not dist.is_initialized() or dist.get_rank() == 0:
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            log(f"Prompt >>>\n{prompt}")
            log(f"Completion >>>\n{completion}")
