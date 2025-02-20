import os, re
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import gc
import torch
from safetensors.torch import safe_open, save_file

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


def main(hf_ckpt_path, save_path, n_experts, mp):
    torch.set_num_threads(96)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]

    # 创建tensor映射的磁盘文件夹
    mmap_dir = "log/mmap_tmp"
    if os.path.exists(mmap_dir):
        os.system(f"rm -rf {mmap_dir}")
    os.makedirs(f"{mmap_dir}", exist_ok=True)

    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
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
                for i in range(mp):
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    elif dim is not None:
                        assert param.size(dim) % mp == 0
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    # 手动将bf16数据类型的权重量化到int8
                    re_list = [
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
                    quant_flag = False # ".weight" in name and "norm." not in name
                    for item in re_list:
                        if quant_flag:
                            break
                        quant_flag = re.match(item, name, re.IGNORECASE)
                    if quant_flag:
                        assert ".scale" not in name
                        weight_fp32 = new_param.float().T # [7168, 2048]
                        max_vals_per_row = torch.max(torch.abs(weight_fp32), dim=0).values
                        assert torch.all(max_vals_per_row > 0), "Some rows have zero max values"

                        scales_per_row = max_vals_per_row / 127.0 # [2048,]
                        quantized_weight = torch.clamp(torch.round(weight_fp32 / scales_per_row), min=-128, max=127)
                        torch.save(quantized_weight.to(torch.int8).contiguous(), f"{mmap_dir}/{name}.pt")
                        torch.save(scales_per_row.to(torch.bfloat16).contiguous(), f"{mmap_dir}/{name}.scale.pt")
                        state_dicts[i][name] = torch.load(f"{mmap_dir}/{name}.pt", map_location="cpu")
                        state_dicts[i][name.replace(".weight", ".scale")] = torch.load(f"{mmap_dir}/{name}.scale.pt", map_location="cpu")

                        # if ".w1." in name: # 为了方便对比数据，选定Expert的w1权重作为参考 (量化前后的误差大约在1%左右)
                        #     import torch_npu
                        #     import torch.nn.functional as F

                        #     w = (state_dicts[i][name].to(torch.bfloat16) * state_dicts[i][name.replace(".weight", ".scale")]).T
                        #     diff = w - new_param
                        #     diff_error = diff.abs().sum() / new_param.abs().sum()
                        #     print(f"[{name}] w >> min: {diff.min().item():.8f}, max: {diff.max().item():.8f}, mean: {diff.mean().item():.8f}, error: {diff_error.item() * 100:.2f}%")

                        #     x = torch.rand(1024, w.shape[1], dtype=torch.bfloat16).to("npu")
                        #     y1 = F.linear(x, new_param.to("npu"))
                        #     y2 = F.linear(x, w.to("npu"))
                        #     diff = y1 - y2
                        #     diff_error = diff.abs().sum() / y1.abs().sum()
                        #     print(f"[{name}] y >> min: {diff.min().item():.8f}, max: {diff.max().item():.8f}, mean: {diff.mean().item():.8f}, error: {diff_error.item() * 100:.2f}%")
                        #     print("-" * 50)

                        del quantized_weight
                        del scales_per_row
                        del weight_fp32
                        del new_param
                        gc.collect()
                    else:
                        state_dicts[i][name] = new_param
                del param
                gc.collect()
        gc.collect()

    # os.makedirs(save_path, exist_ok=True)

    # for i in trange(mp):
    #     save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    # for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
    #     new_file_path = os.path.join(save_path, os.path.basename(file_path))
    #     shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, default="../weights-from-hf/DeepSeek-V3-BF16")
    parser.add_argument("--save-path", type=str, default="../ckpt/v3-int8-mp16")
    parser.add_argument("--n-experts", type=int, default="256")
    parser.add_argument("--model-parallel", type=int, default=16)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
