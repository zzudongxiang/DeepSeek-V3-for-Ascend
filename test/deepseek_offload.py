import os
import math
import time
import torch
import torch_npu
from torch import nn
from datetime import datetime
from tqdm import tqdm, trange
import torch.nn.functional as F
import mindspeed.megatron_adaptor
from safetensors import safe_open
from argparse import ArgumentParser
from transformers import AutoTokenizer
from model.deepseek.rms_norm import RMSNorm
from model.deepseek.args import get_model_args
from utils.quantization.fp8 import fp8_dequant
from utils.quantization.int4 import int4_dequant
from utils.quantization.int8 import int8_dequant
from model.deepseek.linear import set_linear_args, get_linear, Linear
from model.deepseek.rope import precompute_freqs_cis, apply_rotary_emb


expert_hits = 0
expert_misses = 0
layer_misses = {}
n_prefetch_experts = 24
prefetch_experts_pool = []
routed_experts_stream = []
shared_experts_stream = torch_npu.npu.Stream()
prefetch_experts_stream = torch_npu.npu.Stream()


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim))

    def forward(self, x):
        return F.embedding(x, self.weight)


class MLA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        if self.args.q_lora_rank == 0:
            self.wq = Linear(self.args.dim, self.args.n_heads * self.qk_head_dim)
            self.q_linear = lambda x: self.wq(x)
        else:
            self.wq_a = Linear(self.args.dim, self.args.q_lora_rank, x_shape="NCL")
            self.q_norm = RMSNorm(self.args.q_lora_rank)
            self.wq_b = Linear(self.args.q_lora_rank, self.args.n_heads * self.qk_head_dim, x_shape="NCL")
            self.q_linear = lambda x: self.wq_b(self.q_norm(self.wq_a(x)))
        self.wkv_a = Linear(self.args.dim, self.args.kv_lora_rank + self.args.qk_rope_head_dim, x_shape="NCL")
        self.kv_norm = RMSNorm(self.args.kv_lora_rank)
        self.wkv_b = Linear(self.args.kv_lora_rank, self.args.n_heads * (self.args.qk_nope_head_dim + self.args.v_head_dim), x_shape="NCL")
        self.wo = Linear(self.args.n_heads * self.args.v_head_dim, self.args.dim, x_shape="NCL")
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        if args.attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.args.n_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.args.n_heads, self.args.v_head_dim), persistent=False)
            self.get_score_part1 = self.get_naive_score_part1
            self.get_score_part2 = self.get_naive_score_part2
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.args.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.args.qk_rope_head_dim), persistent=False)
            self.get_score_part1 = self.get_absorb_score_part1
            self.get_score_part2 = self.get_absorb_score_part2
        self.update_wkv_b_cache()

    def update_wkv_b_cache(self):
        if self.args.attn_impl == "absorb":
            if self.wkv_b.weight.dtype == torch.bfloat16:
                self.get_wkv_b_weight = self.wkv_b.weight
            elif self.wkv_b.weight.dtype == torch.float8_e4m3fn:
                self.get_wkv_b_weight = fp8_dequant(self.wkv_b.weight, self.wkv_b.scale)
            elif self.wkv_b.weight.dtype == torch.int8:
                self.get_wkv_b_weight = int8_dequant(self.wkv_b.weight, self.wkv_b.scale).T
            elif self.wkv_b.weight.dtype == torch.int32:
                self.get_wkv_b_weight = int4_dequant(self.wkv_b.weight, self.wkv_b.scale).T
            else:
                raise NotImplementedError(f"Unsupported dtype: {self.wkv_b.weight.dtype}")

    def get_naive_score_part1(self, kv, q_nope, q_pe, k_pe, bsz, seqlen, start_pos, end_pos):
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.args.n_heads, self.args.qk_nope_head_dim + self.args.v_head_dim)
        k_nope, v = torch.split(kv, [self.args.qk_nope_head_dim, self.args.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.args.n_heads, -1)], dim=-1)
        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v
        scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        return scores, q, q_nope, kv, None

    def get_absorb_score_part1(self, kv, q_nope, q_pe, k_pe, bsz, seqlen, start_pos, end_pos):
        wkv_b = self.get_wkv_b_weight
        wkv_b = wkv_b.view(self.args.n_heads, -1, self.args.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.args.qk_nope_head_dim])
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                  torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        return scores, None, q_nope, kv, wkv_b

    def get_naive_score_part2(self, scores, wkv_b, bsz, end_pos):
        return torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])

    def get_absorb_score_part2(self, scores, wkv_b, bsz, end_pos):
        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.args.v_head_dim:])
        return x

    def forward(self, x, start_pos, freqs_cis, mask_func):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.q_linear(x)
        q = q.view(bsz, seqlen, self.args.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.args.qk_nope_head_dim, self.args.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.args.kv_lora_rank, self.args.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        scores, q, q_nope, kv, wkv_b = self.get_score_part1(kv, q_nope, q_pe, k_pe, bsz, seqlen, start_pos, end_pos)
        scores = mask_func(scores)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = self.get_score_part2(scores, wkv_b, bsz, end_pos)
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    def __init__(self, dim, inter_dim, x_shape="ND"):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, x_shape=x_shape)
        self.w2 = Linear(inter_dim, dim, x_shape=x_shape)
        self.w3 = Linear(dim, inter_dim, x_shape=x_shape)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.args.dim == 7168 else None
        self.linear = get_linear(self.weight)
        if self.args.score_func == "softmax":
            self.get_score = lambda x: x.softmax(dim=-1, dtype=torch.float32)
            self.get_weight = lambda x: x
        elif self.args.score_func == "sigmoid":
            self.get_score = lambda x: x.sigmoid()
            self.get_weight = lambda x: x / x.sum(dim=-1, keepdim=True)
        else:
            raise NotImplementedError("score_func must be softmax or sigmoid")
        if self.bias is None:
            self.add_bias = lambda x: x
            self.group_scores_func = lambda x: x.amax(dim=-1)
        else:
            self.add_bias = lambda x: x + self.bias
            self.group_scores_func = lambda x: x.topk(2, dim=-1)[0].sum(dim=-1)
        self.calc_group_score = self.calc_group_score if self.args.n_expert_groups > 1 else lambda x, scores: scores

    def calc_group_score(self, x, scores):
        scores = scores.view(x.size(0), self.args.n_expert_groups, -1)
        group_scores = self.group_scores_func(scores)
        indices = group_scores.topk(self.args.n_limited_groups, dim=-1)[1]
        mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
        scores = (scores * mask.unsqueeze(-1)).flatten(1)
        return scores

    def get_indices(self, x, topk):
        scores = self.get_score(self.linear(x, self.weight))
        original_scores = scores
        scores = self.add_bias(scores)
        scores = self.calc_group_score(x, scores)
        indices = torch.topk(scores, topk, dim=-1)[1]
        return indices, original_scores

    def forward(self, x):
        indices, original_scores = self.get_indices(x, self.args.n_activated_experts)
        weights = original_scores.gather(1, indices)
        weights = self.get_weight(weights)
        weights *= self.args.route_scale
        return weights.type_as(x), indices


class MoE(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.gate = Gate(args)
        self.layer_id = layer_id
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
        self.experts = nn.ModuleList([
            MLP(args.dim, args.moe_inter_dim)
            for _ in range(self.args.n_routed_experts)
        ])
        if self.layer_id > self.args.n_dense_layers:
            for expert in self.experts:
                expert.to('cpu')
            self.prefetch_experts = nn.ModuleList([
                nn.Identity()
                for _ in range(self.args.n_routed_experts)
            ])
            self.experts_on_cpu = True
        else:
            self.experts_on_cpu = False
            self.prefetch_experts = self.experts
            self.prefetch_next_moe = lambda x: None
            self.clear_prefetched_experts = lambda: None
        self.prefetch_moe = None

    def prefetch_next_moe(self, x):
        global prefetch_experts_pool
        with torch_npu.npu.stream(prefetch_experts_stream):
            if self.prefetch_moe is None:
                return
            offset = self.args.n_routed_experts if self.layer_id % 2 == 0 else 0
            prefetch_indices = self.prefetch_moe.gate.get_indices(x, n_prefetch_experts)[0]
            for indices in torch.unique(prefetch_indices.flatten()):
                npu_expert = prefetch_experts_pool[indices + offset].state_dict()
                cpu_expert = self.prefetch_moe.experts[indices].state_dict()
                for k in npu_expert:
                    npu_expert[k].copy_(cpu_expert[k], non_blocking=True)
                self.prefetch_moe.prefetch_experts[indices] = prefetch_experts_pool[indices + offset]

    def clear_prefetched_experts(self):
        for i in range(len(self.prefetch_experts)):
            self.prefetch_experts[i] = nn.Identity()

    def forward(self, x):
        global expert_hits, expert_misses, layer_misses, prefetch_experts_pool
        shape = x.size()
        x = x.view(-1, self.args.dim)
        self.prefetch_next_moe(x)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.args.n_routed_experts).tolist()
        offset = self.args.n_routed_experts if self.layer_id % 2 == 1 else 0
        for i in range(self.args.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.prefetch_experts[i]
            if self.experts_on_cpu and isinstance(expert, nn.Identity):
                npu_expert = prefetch_experts_pool[i + offset].state_dict()
                cpu_expert = self.experts[i].state_dict()
                for k in npu_expert:
                    npu_expert[k].copy_(cpu_expert[k], non_blocking=True)
                expert = prefetch_experts_pool[i + offset]
                expert_misses += 1
                layer_misses[self.layer_id] = layer_misses.get(self.layer_id, 0) + 1
            elif self.layer_id > self.args.n_dense_layers:
                expert_hits += 1
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        self.clear_prefetched_experts()
        return (y + z).view(shape)


class Block(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim, x_shape="NCL") if layer_id < args.n_dense_layers else MoE(args, layer_id)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x, start_pos, freqs_cis, mask_func):
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask_func)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args):
        global routed_experts_stream, prefetch_experts_pool
        self.args = args
        if args.dtype == "fp8":
            Linear.dtype = torch.float8_e4m3fn
        elif args.dtype == "int8":
            Linear.dtype = torch.int8
        elif args.dtype == "int4":
            Linear.dtype = torch.int32
        else:
            Linear.dtype = torch.bfloat16
        set_linear_args(args.gemm_impl, args.fp8_quant_block_size, args.offload_cpu)
        super().__init__()
        self.embed = Embedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = Linear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        routed_experts_stream = [torch_npu.npu.Stream() for _ in range(args.n_routed_experts)]
        prefetch_experts_pool = [MLP(args.dim, args.moe_inter_dim) for _ in range(args.n_routed_experts * 2)]
        self.update_cache()

    def update_cache(self):
        for layer in self.layers:
            layer.attn.update_wkv_b_cache()
        for layer_id in range(self.args.n_dense_layers, self.args.n_layers - 1):
            self.layers[layer_id].ffn.prefetch_moe = self.layers[layer_id + 1].ffn

    @torch.inference_mode()
    def forward(self, tokens, start_pos=0):
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
            mask_func = lambda x: x + mask.unsqueeze(1)
        else:
            mask_func = lambda x: x
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask_func)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        return logits


def load_model_weight(model, ckpt_path):
    model_state_dict = model.state_dict()
    ckpt_file_path = os.path.join(ckpt_path, f"model0-mp1.safetensors")
    start_time = datetime.now()
    with safe_open(ckpt_file_path, framework="pt") as f:
        for k in tqdm(model_state_dict.keys()):
            if ".prefetch_experts." in k or ".prefetch_moe." in k:
                continue
            model_state_dict[k].copy_(f.get_tensor(k))
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    model.update_cache()
    hbm = torch_npu.npu.memory_reserved() / 1024.0 / 1024.0 / 1024.0
    print(f"{hbm:.2f}GB Memory in Used")


@torch.inference_mode()
def generate(model, prompt_tokens, max_new_tokens, eos_id):
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.args.max_seq_len
    total_len = min(model.args.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1

    # 初始化时间和token计数
    start_time = time.time()
    first_token_time = None
    total_new_tokens = 0

    for cur_pos in trange(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos

        # 记录第一个token的时间
        if first_token_time is None and cur_pos >= min(prompt_lens):
            first_token_time = time.time() - start_time

        # 统计生成的新token数量
        total_new_tokens += torch.sum(~prompt_mask[:, cur_pos]).item()

        if finished.all():
            break

    # 计算总生成时间和吞吐量
    total_time = time.time() - start_time
    throughput = total_new_tokens / total_time if total_time > 0 else 0

    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    torch.cuda.empty_cache()
    return completion_tokens, first_token_time, throughput


def main(ckpt_path, config, model_args):
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(1234)

    # 构建模型
    args = get_model_args(config, model_args)
    with torch.device("npu"):
        model = Transformer(args)

    # 模型预热
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1)[0][0])

    # 加载模型权重
    if not args.use_random_weights:
        load_model_weight(model, ckpt_path)

    # 重置性能指标
    global expert_hits, expert_misses, layer_misses
    expert_hits = 0
    expert_misses = 0
    layer_misses = {}

    with open("scripts/inputs.txt") as f:
        prompts = [line.strip() for line in f.readlines()]
        prompts = prompts[: args.max_batch_size]

    print("\n开始生成...")
    start_time = time.time()
    prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
    completion_tokens, ttft, throughput = generate(model, prompt_tokens, 2000, tokenizer.eos_token_id)
    total_time = time.time() - start_time

    completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
    for prompt, completion in zip(prompts, completions):
        print("Prompt:", prompt)
        print("Completion:", completion)
        print()

    # 显示性能指标
    print("\n======== 性能指标 ========")
    print(f"TTFT (首词生成延迟): {ttft*1000:.2f} ms")
    print(f"生成吞吐量: {throughput:.2f} tokens/s")
    print(f"总生成时间: {total_time:.2f} s")

    # 打印平均每个提示的生成速度
    avg_tokens_per_prompt = sum(len(tokens) for tokens in completion_tokens) / len(completion_tokens)
    print(f"平均每个提示生成: {avg_tokens_per_prompt:.1f} tokens")
    print(f"提示数量: {len(prompts)}")

    # 显示专家预取指标
    print("\n------- 专家预取指标 -------")
    total_experts = expert_hits + expert_misses
    if total_experts > 0:
        hit_rate = expert_hits / total_experts * 100
        print(f"专家缓存命中率: {hit_rate:.1f}% ({expert_hits}/{total_experts})")
        print(f"缓存未命中: {expert_misses} 次")

        # 打印每层的miss次数
        print("\n每层专家预取Miss次数:")
        dense_layers = args.n_dense_layers
        for layer_id in sorted(layer_misses.keys()):
            print(f"  层 {layer_id} (MoE {layer_id - dense_layers}): {layer_misses[layer_id]} 次")

    print("==========================\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, default="../ckpt/v3-int4-mp1")
    parser.add_argument("--config-path", type=str, default="configs/config_offload.json")
    args, model_args = parser.parse_known_args()
    main(args.ckpt_path, args.config_path, model_args)
