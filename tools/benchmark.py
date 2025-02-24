import requests
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, Future

# 默认发送请求的目标地址
post_url = "http://127.0.0.1:{port}/chat"

def get_response(prompt: str) -> str:
    data = {
        "prompt": prompt,
    }
    result = None
    try:
        response = requests.post(post_url, json=data, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            resp = response.json()
            result = resp["choices"][0]["message"]["content"]
        else:
            print(f"Request Error ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"Request Error: {e}")
    finally:
        return result

def get_batch_response(prompts: list) -> list:
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(get_response, prompt) for prompt in prompts]
    results = []
    for future in futures:
        results.append(future.result())
    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--input-file", type=str, default="scripts/inputs.txt")
    args = parser.parse_args()

    # 更新服务器的请求地址
    post_url = post_url.replace("{port}", str(args.port))

    while True:
        # 暂时不支持上下文多轮对话
        prompt = input(">>> ")
        if prompt == "/all":
            # 测试场景1: 并行对话请求
            with open(args.input_file) as f:
                prompts = [line.strip() for line in f.readlines()]
            results = get_batch_response(prompts)
            for prompt, result in zip(prompts, results):
                # TODO: 对响应的内容进行评分
                print(prompt, result)
        else:
            # 测试场景2: 测试单轮对话
            completion = get_response(prompt)
            # TODO: 对响应的内容进行评分
            print(completion)
