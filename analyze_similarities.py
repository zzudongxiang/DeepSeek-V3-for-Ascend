import torch
import os
from pathlib import Path
import numpy as np
import json
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def visualize_gate_scores(save_dir="saves", output_dir="analysis_results"):
    save_dir = Path(save_dir)
    output_dir = Path(f"{output_dir}/{timestamp}/gate_scores")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有层的编号
    layer_dirs = [d for d in save_dir.iterdir() if d.name.startswith('layer_')]
    layer_nums = sorted([int(d.name.split('_')[1]) for d in layer_dirs])
    
    # 创建总结文件
    summary_before_bias = output_dir / "gate_scores_before_bias_summary.txt"
    summary_after_bias = output_dir / "gate_scores_after_bias_summary.txt"
    summary_lines_before = ["层号\t最大值\t最小值\t平均值\t标准差"]
    summary_lines_after = ["层号\t最大值\t最小值\t平均值\t标准差"]
    
    # 设置图表样式
    plt.style.use('seaborn')
    
    for layer_num in layer_nums:
        scores_before_path = save_dir / f"layer_{layer_num}/moe/gate/scores_before_bias.pt"
        scores_after_path = save_dir / f"layer_{layer_num}/moe/gate/scores_after_bias.pt"
        
        # 检查文件是否存在
        if not scores_after_path.exists() or not scores_before_path.exists():
            continue
            
        # 加载scores并转换为float32
        scores_before = torch.load(scores_before_path).to(torch.float32).cpu()[0]
        scores_after = torch.load(scores_after_path).to(torch.float32).cpu()[0]
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 绘制before bias的柱状图
        bars1 = ax1.bar(range(len(scores_before)), scores_before)
        ax1.set_title(f'Layer {layer_num} Gate Scores (Before Bias)', fontsize=14)
        ax1.set_xlabel('Expert Index', fontsize=12)
        ax1.set_ylabel('Activated Score', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 计算before bias的统计信息
        stats_before = {
            'max': scores_before.max().item(),
            'min': scores_before.min().item(),
            'mean': scores_before.mean().item(),
            'std': scores_before.std().item()
        }
        
        # 添加before bias的统计信息
        stats_text_before = f'Max: {stats_before["max"]:.4f}\nMin: {stats_before["min"]:.4f}\nMean: {stats_before["mean"]:.4f}\nStd: {stats_before["std"]:.4f}'
        ax1.text(0.95, 0.95, stats_text_before,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 绘制after bias的柱状图
        bars2 = ax2.bar(range(len(scores_after)), scores_after)
        ax2.set_title(f'Layer {layer_num} Gate Scores (After Bias)', fontsize=14)
        ax2.set_xlabel('Expert Index', fontsize=12)
        ax2.set_ylabel('Activated Score', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 计算after bias的统计信息
        stats_after = {
            'max': scores_after.max().item(),
            'min': scores_after.min().item(),
            'mean': scores_after.mean().item(),
            'std': scores_after.std().item()
        }
        
        # 添加after bias的统计信息
        stats_text_after = f'Max: {stats_after["max"]:.4f}\nMin: {stats_after["min"]:.4f}\nMean: {stats_after["mean"]:.4f}\nStd: {stats_after["std"]:.4f}'
        ax2.text(0.95, 0.95, stats_text_after,
                transform=ax2.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 调整布局并保存图表
        plt.tight_layout()
        plt.savefig(output_dir / f"layer_{layer_num}_gate_scores_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 添加到总结文件
        summary_lines_before.append(f"{layer_num}\t{stats_before['max']:.4f}\t{stats_before['min']:.4f}\t{stats_before['mean']:.4f}\t{stats_before['std']:.4f}")
        summary_lines_after.append(f"{layer_num}\t{stats_after['max']:.4f}\t{stats_after['min']:.4f}\t{stats_after['mean']:.4f}\t{stats_after['std']:.4f}")
        
        # 打印进度
        print(f"已处理第 {layer_num} 层")
    
    # 保存总结文件
    with open(summary_before_bias, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines_before))
    with open(summary_after_bias, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines_after))
    
    print(f"可视化结果已保存到：{output_dir}")
    print(f"总结文件已保存到：")
    print(f"1. Before bias: {summary_before_bias}")
    print(f"2. After bias: {summary_after_bias}")


def cosine_similarity_batch_mean(tensor1, tensor2):
    """计算两个tensor在batch维度上的余弦相似度，并取平均"""
    # 将张量展平到2D: [batch_size, -1]
    t1_flat = tensor1.reshape(tensor1.shape[0], -1)
    t2_flat = tensor2.reshape(tensor2.shape[0], -1)
    
    # 计算余弦相似度
    norm1 = torch.norm(t1_flat, dim=1, keepdim=True)
    norm2 = torch.norm(t2_flat, dim=1, keepdim=True)
    dot_products = torch.sum(t1_flat * t2_flat, dim=1)
    similarities = dot_products / (norm1.squeeze() * norm2.squeeze())
    
    # 返回batch维度的平均值
    return similarities.mean().item()

def analyze_similarities(save_dir="saves", output_dir="analysis_results"):
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(save_dir)
    output_dir = Path(f"{output_dir}/{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有层的编号
    layer_dirs = [d for d in save_dir.iterdir() if d.name.startswith('layer_')]
    layer_nums = sorted([int(d.name.split('_')[1]) for d in layer_dirs])
    
    # 定义要分析的相似度组合
    similarity_configs = [
        {
            "name": "attn_i-1_ffn_i",
            "description": "前一层attn的input与当前层ffn的input的相似度",
            "path1": lambda i: f"layer_{i}/attn/input.pt",
            "path2": lambda i: f"layer_{i+1}/ffn/input.pt"
        },
        {
            "name": "attn_norm_i-1_ffn_i",
            "description": "前一层attn_norm的input与当前层ffn的input的相似度",
            "path1": lambda i: f"layer_{i}/attn_norm/input.pt",
            "path2": lambda i: f"layer_{i+1}/ffn/input.pt"
        },
        {
            "name": "ffn_i-1_ffn_i",
            "description": "前一层ffn的input与当前层ffn的input的相似度",
            "path1": lambda i: f"layer_{i}/ffn/input.pt",
            "path2": lambda i: f"layer_{i+1}/ffn/input.pt"
        },
        {
            "name": "attn_norm_i_ffn_i",
            "description": "当前层attn_norm的input与当前层ffn的input的相似度",
            "path1": lambda i: f"layer_{i}/attn_norm/input.pt",
            "path2": lambda i: f"layer_{i}/ffn/input.pt"
        },
        {
            "name": "attn_i_ffn_i",
            "description": "当前层attn的input与当前层ffn的input的相似度",
            "path1": lambda i: f"layer_{i}/attn/input.pt",
            "path2": lambda i: f"layer_{i}/ffn/input.pt"
        },
        {
            "name": "ffn_norm_i-1_ffn_i",
            "description": "前一层ffn_norm的input与当前层ffn的input的相似度",
            "path1": lambda i: f"layer_{i}/ffn_norm/input.pt",
            "path2": lambda i: f"layer_{i+1}/ffn/input.pt"
        },
        {
            "name": "attn_output_i-1_ffn_input_i",
            "description": "前一层attn的output与当前层ffn的input的相似度",
            "path1": lambda i: f"layer_{i}/attn/output.pt",
            "path2": lambda i: f"layer_{i+1}/ffn/input.pt"
        }

    ]
    
    for config in similarity_configs:
        output_lines = [
            f"层间余弦相似度分析（{config['description']}）：",
            "-" * 60,
            "层对比\t\t余弦相似度",
            "-" * 60
        ]
        
        results = {
            "timestamp": timestamp,
            "description": config['description'],
            "similarities": [],
            "layer_pairs": [],
            "statistics": {}
        }
        
        similarities = []
        
        # 确定要比较的层范围
        start_layer = 0
        end_layer = len(layer_nums) - (1 if "i-1" in config["name"] else 0)
        
        for i in range(start_layer, end_layer):
            if "i-1" in config["name"]:
                layer_pair = f"Layer {i}->{i+1}"
            else:
                layer_pair = f"Layer {i}"
            
            # 加载张量
            tensor1 = torch.load(save_dir / config["path1"](i))
            tensor2 = torch.load(save_dir / config["path2"](i))
            similarity = cosine_similarity_batch_mean(tensor1, tensor2)
            
            similarities.append(similarity)
            output_lines.append(f"{layer_pair}\t{similarity:.6f}")
            
            results["similarities"].append(similarity)
            results["layer_pairs"].append(layer_pair)
        
        # 计算统计信息
        stats = {
            "mean": np.mean(similarities),
            "max": np.max(similarities),
            "min": np.min(similarities),
            "std": np.std(similarities)
        }
        results["statistics"] = stats
        
        # 添加统计信息到输出
        output_lines.extend([
            "-" * 60,
            f"平均相似度:\t{stats['mean']:.6f}",
            f"最大相似度:\t{stats['max']:.6f}",
            f"最小相似度:\t{stats['min']:.6f}",
            f"标准差:\t\t{stats['std']:.6f}"
        ])
        
        # 保存文本输出
        output_file = output_dir / f"{config['name']}_similarities.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        # 保存JSON结果
        results_file = output_dir / f"{config['name']}_similarities.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\n分析结果 ({config['name']}) 已保存到：")
        print(f"1. 文本文件：{output_file}")
        print(f"2. JSON文件：{results_file}")
        print("\n".join(output_lines))
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    visualize_gate_scores()
    analyze_similarities()