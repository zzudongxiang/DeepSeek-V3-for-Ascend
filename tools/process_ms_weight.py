import os, re
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


def copy_expert(hf_ckpt_path, save_path):
    torch.set_num_threads(92)
    os.makedirs(save_path, exist_ok=True)
    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        state_dict = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "feed_forward.routed_experts.ffn" in name:
                    tensor = f.get_tensor(name)
                    layer_id = name.split('.')[2]
                    w = name.split('.')[-2]
                    w_map = {
                        "w1": "gate_proj",
                        "w2": "down_proj",
                        "w3": "up_proj",
                    }
                    for i in range(256):
                        state_dict[f"model.layers.{layer_id}.mlp.experts.{i}.{w_map[w]}.weight"] = tensor[i, :, :].T.contiguous()
                    save_file(state_dict, os.path.join(save_path, os.path.basename(file_path)))
                else:
                    os.system(f"cp {file_path} {os.path.join(save_path, os.path.basename(file_path))}")

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ms-ckpt-path", type=str, default="/home/zhangdx/mnt/deepseek-v3/mindspore-dsv3")
    parser.add_argument("--hf-ckpt-path", type=str, default="/home/zhangdx/mnt/deepseek-v3/mindspore-dsv3-hf")
    args = parser.parse_args()

    # 复制专家，将ms中的专家拆分成多个专家对象
    copy_expert(args.ms_ckpt_path, args.hf_ckpt_path)

