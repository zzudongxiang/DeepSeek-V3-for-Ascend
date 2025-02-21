import time
import flask
import random
import threading
from queue import Queue
from datetime import datetime
from utils.logger import log_rank0
from concurrent.futures import Future
from utils.generate import batch_generate

app = flask.Flask(__name__)

# 将{max_reponse_delay}秒内的请求合并成一个batch进行处理
max_reponse_delay = 1.0

request_queue = Queue()
condition = threading.Condition()
model, tokenizer = None, None

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
                sub_batches = [batch[i:i + model.args.max_batch_size] for i in range(0, len(batch), model.args.max_batch_size)]
                for sub_batch in sub_batches:
                    # 提取prompts并生成响应
                    prompts = [item['prompt'] for item in sub_batch]
                    prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
                    completions = tokenizer.batch_decode(batch_generate(model, prompt_tokens, tokenizer.eos_token_id), skip_special_tokens=True)
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

def run(local_model, local_tokenizer) -> None:
    # 传入参数
    global model, tokenizer
    tokenizer = local_tokenizer
    model = local_model

    # 启动服务
    # curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"prompt": "Hello World!"}'
    log_rank0("curl -X POST http://127.0.0.1:5000/chat -H \"Content-Type: application/json\" -d '{\"prompt\": \"Hello World!\"}'")
    app.run(host='0.0.0.0', port=5000, debug=True)
