import os
import time
import flask
import random
import threading
from queue import Queue
from datetime import datetime
from concurrent.futures import Future
from utils.startup.tcpip import Server, Client
from utils.generate.entry import batch_generate

app = flask.Flask(__name__)

# 将{max_reponse_delay}秒内的请求合并成一个batch进行处理
max_reponse_delay = 1.0

request_queue = Queue()
condition = threading.Condition()

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

def batch_generate_response(model, tokenizer, prompts):
    prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
    completions = tokenizer.batch_decode(batch_generate(model, prompt_tokens, tokenizer.eos_token_id), skip_special_tokens=True)
    responses = [generate_response(prompt, completion) for prompt, completion in zip(prompts, completions)]
    return responses

def master_worker(model, tokenizer, server):
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
                sub_batches = [batch[i: i + model.args.max_batch_size] for i in range(0, len(batch), model.args.max_batch_size)]
                for sub_batch in sub_batches:
                    # 广播给其他节点
                    prompts = [item['prompt'] for item in sub_batch]
                    server.broadcast(prompts)
                    responses = batch_generate_response(model, tokenizer, prompts)
                    # 将响应分配给对应的future
                    for item, response in zip(sub_batch, responses):
                        item['future'].set_result(response)
            except Exception as e:
                for item in batch:
                    item['future'].set_exception(e)
            finally:
                batch.clear()

@app.route('/chat', methods=['POST'])
def generate_text():
    data = flask.request.json
    prompt = data.get('prompt')
    if not prompt:
        return flask.jsonify({'error': 'Prompt is required'}), 400
    future = Future()
    request_queue.put({'prompt': prompt, 'future': future})
    with condition:
        condition.notify_all()
    try:
        response = future.result()
        return flask.jsonify(response)
    except Exception as e:
        return flask.jsonify({'error': str(e)}), 500

def run_app(api_port):
    print(f"[{datetime.now()}]", end=" ")
    app.run(host="0.0.0.0", port=api_port)

def run_master(model, tokenizer, api_port, service_port) -> None:
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    server = Server(master_addr, service_port)
    processor_thread = threading.Thread(target=run_app, args=(api_port,), daemon=True)
    processor_thread.start()
    server.start()
    master_worker(model, tokenizer, server)

def run_slaver(model, tokenizer, service_port) -> None:
    time.sleep(5)
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    client = Client(master_addr, service_port)
    client.connect()
    client.receive_messages(lambda batch_prompts: batch_generate_response(model, tokenizer, batch_prompts))
    input()
