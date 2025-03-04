import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_pt_files_with_rank_0(folder):
    pt_files = {}
    for file in os.listdir(folder):
        if file.endswith('.pt'):
            file_path = os.path.join(folder, file)
            pt_files[file] = torch.load(file_path, map_location=torch.device('cpu'))
    return pt_files

def compare_pt_files(pt_files1, pt_files2):
    differences = {}
    for file, tensor1 in pt_files1.items():
        if file in pt_files2:
            tensor2 = pt_files2[file]
            if tensor1.shape == tensor2.shape:
                diff = tensor1 - tensor2

                tensor1_num = tensor1.float().flatten().numpy()
                tensor2_num = tensor2.float().flatten().numpy()
                diff_num = diff.float().flatten().numpy()
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
                ax1.hist(tensor1_num, bins=30, color='blue', alpha=0.7)
                ax1.set_title(f"GPU - {file}")
                ax2.hist(tensor2_num, bins=30, color='red', alpha=0.7)
                ax2.set_title(f"NPU - {file}")
                ax3.hist(diff_num, bins=30, color='green', alpha=0.7)
                ax3.set_title(f"diff")
                if not os.path.exists(f"png"):
                    os.makedirs(f"png")
                plt.savefig(f"png/{file}.png")
                plt.close(fig)

                if "indices" in file:
                    np.savetxt(f"txt/{file}-tensor1.csv", tensor1.float().numpy(), delimiter=',', fmt='%03d')
                    np.savetxt(f"txt/{file}-tensor2.csv", tensor2.float().numpy(), delimiter=',', fmt='%03d')
                    np.savetxt(f"txt/{file}-diff.csv", diff.float().numpy(), delimiter=',', fmt='%04d')

                mean_diff = torch.mean(torch.abs(diff).float())
                var_diff = torch.var(torch.abs(diff).float())
                min_diff = torch.min(diff)
                max_diff = torch.max(diff)
                differences[file] = {
                    'mean': mean_diff.item(),
                    'variance': var_diff.item(),
                    'min': min_diff.item(),
                    'max': max_diff.item(),
                    'score': (torch.abs(diff).sum() / torch.abs(tensor1).sum()).item()
                }
            else:
                print(f"Shapes of tensors in {file} do not match: {tensor1.shape} vs {tensor2.shape}")
                # raise ValueError()
        else:
            print(f"File {file} not found in the second folder.")
    return differences


def main():
    folder1 = 'dump-gpu'
    folder2 = 'dump-npu'

    pt_files1 = load_pt_files_with_rank_0(folder1)
    pt_files2 = load_pt_files_with_rank_0(folder2)
    
    differences = compare_pt_files(pt_files1, pt_files2)

    print()
    print("name layer mean variance min max score")
    for file, diff in differences.items():
        layer = file.split('_')[0].replace("layer", "")
        name = "_".join(file.split("_")[1:]).split('.')[0].replace("_rank_0", "")
        print(name, layer, diff['mean'], diff['variance'], diff['min'], diff['max'], diff['score'])

if __name__ == "__main__":
    main()
