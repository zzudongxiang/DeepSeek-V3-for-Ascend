#!/bin/python

import os
import json
import time
import flask
import torch
import random
import threading
from typing import List
from queue import Queue
from datetime import datetime
import torch.distributed as dist
from argparse import ArgumentParser
from concurrent.futures import Future
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model.utils.tools import sample_cpu
from model.deepseek_int8 import Transformer, ModelArgs

default_device = "cuda"

try:
    import torch_npu
    import mindspeed.megatron_adaptor
    default_device = "npu"
except:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch_npu not found, using {default_device}")

app = flask.Flask(__name__)

# 请求队列和批处理配置
max_reponse_delay = 1.0 # 将{max_reponse_delay}秒内的请求合并成一个batch进行处理
max_new_tokens = 1024
temperature = 1.0
world_size = 1
local_rank = 0
rank = 0
request_queue = Queue()
condition = threading.Condition()
model, tokenizer, args = None, None, None

@torch.inference_mode()
def generate_tokens(prompt_tokens: List[List[int]]) -> List[List[int]]:
    global max_new_tokens, temperature
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device=default_device)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=default_device)
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device=default_device)
    prompt_mask = tokens != -1
    t0 = datetime.now()
    ttft_flag = True
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if ttft_flag:
            ttft_flag = False
            ttft = datetime.now() - t0
            print(f"TTFT: {ttft.total_seconds():.4f} seconds ({cur_pos} tokens)")
            t0 = datetime.now()
        if temperature > 0:
            next_token = sample_cpu(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == tokenizer.eos_token_id)
        prev_pos = cur_pos
        if finished.all():
            break
    output_tokens = 0
    dur = (datetime.now() - t0).total_seconds()
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if tokenizer.eos_token_id in toks:
            toks = toks[:toks.index(tokenizer.eos_token_id)]
            output_tokens += toks.index(tokenizer.eos_token_id)
        else:
            output_tokens += len(toks)
        completion_tokens.append(toks)
    tpot = output_tokens / dur
    print(f"Throughput: {tpot:.4f} tokens/s ({output_tokens} tokens for {dur} sec)")
    return completion_tokens

def generate_response(prompt, completion):
    return {
        "id": f"cmpl-{random.randint(100000, 999999)}",
        "object": "text.completion",
        "created": int(time.time()),
        "model": "text-deepseek",
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(completion),
            "total_tokens": len(prompt) + len(completion)
        },
        "choices": [{
            "message": {
                "role": "assistant",
                "content": completion
            },
            "finish_reason": "stop",
            "index": 0
        }]
    }

def batch_processor():
    batch = []
    timeout = max_reponse_delay
    t0 = datetime.now()
    while True:
        with condition:
            # 等待直到满足批处理条件
            condition.wait(timeout=timeout)
            timeout = max_reponse_delay - (datetime.now() - t0).total_seconds()
            timeout = timeout if timeout > 0 else max_reponse_delay
            # 收集队列中的请求
            while not request_queue.empty():
                if not batch:
                    t0 = datetime.now()
                batch.append(request_queue.get())
        if batch and (datetime.now() - t0).total_seconds() >= max_reponse_delay:
            try:
                sub_batches = [batch[i:i + args.max_batch_size] for i in range(0, len(batch), args.max_batch_size)]
                for sub_batch in sub_batches:
                    # 提取prompts并生成响应
                    prompts = [item['prompt'] for item in sub_batch]
                    prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
                    completions = tokenizer.batch_decode(generate_tokens(prompt_tokens), skip_special_tokens=True)
                    responses = [generate_response(prompt, completion) for prompt, completion in zip(prompts, completions)]
                    # 将响应分配给对应的future
                    for item, response in zip(sub_batch, responses):
                        item['future'].set_result(response)
            except Exception as e:
                for item in batch:
                    item['future'].set_exception(e)
            finally:
                batch.clear()

# 启动后台批处理线程
processor_thread = threading.Thread(target=batch_processor, daemon=True)
processor_thread.start()

@app.route('/chat', methods=['POST'])
def generate_text():
    data = flask.request.json
    prompt = data.get('prompt')
    if not prompt:
        return flask.jsonify({'error': 'Prompt is required'}), 400

    # 创建Future并加入队列
    future = Future()
    request_queue.put({'prompt': prompt, 'future': future})
    
    # 通知批处理线程
    with condition:
        condition.notify_all()
    
    # 等待并返回结果
    try:
        response = future.result()
        return flask.jsonify(response)
    except Exception as e:
        return flask.jsonify({'error': str(e)}), 500

def main(ckpt_path: str, config: str) -> None:
    global model, tokenizer, args
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None

    try:
        # 设置torch参数
        if default_device == "npu":
            torch_npu.npu.set_device(local_rank)
        elif default_device == "cuda":
            torch.cuda.set_device(local_rank)
        torch.set_default_dtype(torch.bfloat16)
        torch.set_num_threads(8)
        torch.manual_seed(965)

        # 构造模型
        with open(config) as f:
            args = ModelArgs(**json.load(f))
        print(args)
        with torch.device(default_device):
            model = Transformer(args)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

        # 验证模型
        now = datetime.now()
        tokenizer.decode(generate_tokens(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
        dur = (datetime.now() - now).total_seconds()
        print(datetime.now(), f"Prepare DeepSeek Model in {dur:.2f} sec")

        # 加载模型权重
        now = datetime.now()
        load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
        dur = (datetime.now() - now).total_seconds() / 60.0
        print(datetime.now(), f"Load DeepSeek Weight in {dur:.2f} min")

        # 启动服务
        # curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"prompt": "Hello World!"}'
        print("curl -X POST http://127.0.0.1:5000/chat -H \"Content-Type: application/json\" -d '{\"prompt\": \"Hello World!\"}'")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        raise e
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)

    args = parser.parse_args()
    assert args.input_file or args.interactive
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature

    main(args.ckpt_path, args.config)
