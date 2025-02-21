#!/bin/python

import os
import re
import gc
import sys
import torch
import shutil
from glob import glob
from tqdm import tqdm, trange
from argparse import ArgumentParser
from safetensors.torch import safe_open, save_file

n_experts = 256
state_dicts = []
quant_type = None
model_parallel = 0
mmap_dir = "log/mmap_tmp"
sys.path.append(os.getcwd())

mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}

quant_list = [
    r".*layers\.\d+\.attn\.wkv_a.*",
    r".*layers\.\d+\.attn\.wkv_b.*",
    r".*layers\.\d+\.attn\.wq_a.*",
    r".*layers\.\d+\.attn\.wq_b.*",
    r".*layers\.\d+\.attn\.wq.*",
    r".*layers\.\d+\.attn\.wo.*",
    r".*layers\.\d+\.ffn\.w\d+.*",
    r".*layers\.\d+\.ffn\.experts\.\d+\.w\d+.*",
    r".*layers\.\d+\.ffn\.shared_experts\.w\d+.*"
]

def quant_func(name, weight):
    quant_flag = False
    for item in quant_list:
        if quant_flag:
            break
        quant_flag = re.match(item, name, re.IGNORECASE)
    if quant_flag:
        assert ".scale" not in name
        quantized_weight_path = f"{mmap_dir}/{name}.pt"
        weight_scale_path = f"{mmap_dir}/{name}.scale.pt"
        if "int8" in quant_type:
            from utils.quantization.int8 import int8_quant
            quantized_weight, weight_scale = int8_quant(weight.T)
        elif "int4" in quant_type:
            import torch_npu
            from utils.quantization.int4 import int4_quant
            quantized_weight, weight_scale = int4_quant(weight.T.npu())
            quantized_weight = quantized_weight.cpu()
            weight_scale = weight_scale.cpu()
        else:
            raise NotImplementedError(f"quant type {quant_type} not supported")
        quantized_weight = quantized_weight.contiguous()
        weight_scale = weight_scale.contiguous()
        torch.save(weight_scale, weight_scale_path)
        torch.save(quantized_weight, quantized_weight_path)
        weight_scale = torch.load(weight_scale_path, map_location="cpu")
        quantized_weight = torch.load(quantized_weight_path, map_location="cpu")
        return quantized_weight, weight_scale
    else:
        return weight, None

def process_file(file_path):
    global state_dicts
    n_local_experts = n_experts // model_parallel
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            if "model.layers.61" in name:
                continue
            param: torch.Tensor = f.get_tensor(name)
            if name.startswith("model."):
                name = name[len("model."):]
            name = name.replace("self_attn", "attn")
            name = name.replace("mlp", "ffn")
            name = name.replace("weight_scale_inv", "scale")
            name = name.replace("e_score_correction_bias", "bias")
            key = name.split(".")[-2]
            assert key in mapping
            new_key, dim = mapping[key]
            name = name.replace(key, new_key)
            for i in range(model_parallel):
                new_param = param
                if "experts" in name and "shared_experts" not in name:
                    idx = int(name.split(".")[-3])
                    if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                        continue
                elif dim is not None:
                    assert param.size(dim) % model_parallel == 0
                    shard_size = param.size(dim) // model_parallel
                    new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                if quant_type is None:
                    state_dicts[i][name] = new_param
                else:
                    quantized_weight, weight_scale = quant_func(name, new_param)
                    state_dicts[i][name] = quantized_weight
                    if weight_scale is not None:
                        state_dicts[i][name.replace(".weight", ".scale")] = weight_scale
                    del quantized_weight
                    del weight_scale
                del new_param
                gc.collect()
            del param
            gc.collect()
    gc.collect()


def main(hf_path, save_path):
    global state_dicts
    torch.set_num_threads(96)
    os.makedirs(mmap_dir, exist_ok=True)
    state_dicts = [{} for _ in range(model_parallel)]

    # 读取文件并处理
    for file_path in tqdm(glob(os.path.join(hf_path, "*.safetensors"))):
        process_file(file_path)

    # 将处理后的数据保存到对应的文件中
    os.makedirs(save_path, exist_ok=True)
    for i in trange(model_parallel):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{model_parallel}.safetensors"))

    # 复制其他的必须文件到目标文件夹
    for file_path in glob(os.path.join(hf_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)
    os.system(f"rm -rf {mmap_dir}/*.pt")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--quant-type", type=str, default=None)
    parser.add_argument("--model-parallel", type=int, default=0)
    args = parser.parse_args()

    # 获取量化的数据类型
    if args.quant_type is not None:
        quant_type = args.quant_type
    else:
        if "int8" in args.save_path:
            quant_type = "int8"
        elif "int4" in args.save_path:
            quant_type = "int4"

    # 获取并行的数量
    if args.model_parallel > 0:
        model_parallel = args.model_parallel
    else:
        match = re.search(r"mp(\d+)", args.save_path)
        if match:
            model_parallel = int(match.group(1))
        else:
            raise Exception("save folder format: ./v3-int8-mp16/")

    assert n_experts % model_parallel == 0
    quant_type_str = quant_type if quant_type is not None else "bf16"
    mmap_dir = f"{mmap_dir}/{os.path.basename(args.save_path)}"

    print(f"quant_type: {quant_type_str}, model_parallel: {model_parallel}")
    main(args.hf_path, args.save_path)
