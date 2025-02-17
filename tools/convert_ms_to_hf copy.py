import os, re
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

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

name_map = {
    "tok_embeddings.embedding_weight": "embed_tokens.weight",
    "lm_head.weight": "lm_head.weight",
    "model.norm_out.weight": "model.norm.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attention.lkv_norm.weight": "self_attn.kv_a_layernorm.weight",
    "attention.kv2l.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attention.lkv2kv.weight": "self_attn.kv_b_proj.weight",
    "attention.wo.weight": "self_attn.o_proj.weight",
    "attention.lq_norm.weight": "self_attn.q_a_layernorm.weight",
    "attention.q2l_proj.weight": "self_attn.q_a_proj.weight",
    "attention.l2q_proj.weight": "self_attn.q_b_proj.weight",
    "feed_forward.routed_experts.router.e_score_correction_bias": "mlp.gate.e_score_correction_bias",
    "feed_forward.routed_experts.router.dense.weight": "mlp.gate.weight",
    "feed_forward.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
    "feed_forward.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
    "feed_forward.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
    "attention.lkv_norm.weight": "self_attn.kv_a_layernorm.weight",
    "attention.kv2l.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attention.lkv2kv.weight": "self_attn.kv_b_proj.weight",
    "attention.wo.weight": "self_attn.o_proj.weight",
    "attention.lq_norm.weight": "self_attn.q_a_layernorm.weight",
    "attention.q2l_proj.weight": "self_attn.q_a_proj.weight",
    "attention.l2q_proj.weight": "self_attn.q_b_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "feed_forward.routed_experts.ffn.w1.weight": "mlp.experts.*.gate_proj.weight",
    "feed_forward.routed_experts.ffn.w2.weight": "mlp.experts.*.down_proj.weight",
    "feed_forward.routed_experts.ffn.w3.weight": "mlp.experts.*.up_proj.weight",
}

def copy_expert(hf_ckpt_path, save_path):
    torch.set_num_threads(92)
    os.makedirs(save_path, exist_ok=True)
    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        state_dict = {}
        if re.match(r"model\.layers\.\d+\.feed_forward\.routed_experts\.ffn\.w\d+\.weight", os.path.basename(file_path)):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    layer_id = name.split('.')[2]
                    w = name.split('.')[-2]
                    w_map = {
                        "w1": "gate_proj",
                        "w2": "down_proj",
                        "w3": "up_proj",
                    }
                    for i in range(256):
                        state_dict[f"model.layers.{layer_id}.mlp.experts.{i}.{w_map[w]}.weight"] = tensor[i, :, :]
            save_file(state_dict, os.path.join(save_path, os.path.basename(file_path)))
        else:
            os.system(f"cp {file_path} {os.path.join(save_path, os.path.basename(file_path))}")

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)

def main(hf_ckpt_path, save_path, n_experts, mp):
    torch.set_num_threads(92)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]

    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                # 原有逻辑
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                # 替换变量名
                for pattern in name_map:
                    if pattern in name:
                        name = name.replace(pattern, name_map[pattern])
                        break
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
                    state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    for i in trange(mp):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, default="/home/zhangdx/mnt/deepseek-v3/mindspore-dsv3-hf")
    parser.add_argument("--save-path", type=str, default="/home/zhangdx/mnt/deepseek-v3/ckpt/v3-bf16-mp32-ms")
    parser.add_argument("--n-experts", type=int, default=256)
    parser.add_argument("--model-parallel", type=int, default=32)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0

    # 复制专家，将ms中的专家拆分成多个专家对象
    # copy_expert(args.hf_ckpt_path, args.save_path)

    # 按照模型并行的方法将权重拆分到多个节点上
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
