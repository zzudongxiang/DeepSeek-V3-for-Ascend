import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

from model.kernel import act_quant, weight_dequant, fp8_gemm
from model.utils.writer import FLOPs_Writer, XCCL_Writer, Memory_Writer, Weights_Writer, tensor_hist

rank = 0
world_size = 1
block_size = 128
xccl_writer_enabled = False
flops_writer_enabled = False
memory_writer_enabled = False
weights_writer_enabled = False
tensor_hist_writer_enabled = False
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

xccl_writer = flops_writer = memory_writer = weights_writer = None
xccl_recoder = lambda *_, **__: (xccl_writer.recoder(*_, **__) if xccl_writer_enabled else None)
flops_recoder = lambda *_, **__: (flops_writer.recoder(*_, **__) if flops_writer_enabled else None)
memory_recoder = lambda *_, **__: (memory_writer.recoder(*_, **__) if memory_writer_enabled else None)
weights_recoder = lambda *_, **__: (weights_writer.recoder(*_, **__) if weights_writer_enabled else None)

tensor_hist_recoder = lambda *_, **__: (tensor_hist(*_, **__) if tensor_hist_writer_enabled else None)
print_flops = lambda flush_only = False: (flops_writer.print(flush_only) if flops_writer_enabled else 0, 0, 0, 0, 0)

def writer_finished():
    if memory_writer_enabled and memory_writer is not None:
        memory_writer.finished("a+")
    if flops_writer_enabled and flops_writer is not None:
        flops_writer.finished()
    if xccl_writer_enabled and xccl_writer is not None:
        xccl_writer.finished()

def writer_split():
    if flops_writer_enabled and flops_writer is not None:
        flops_writer.write("-" * 50)
    if xccl_writer_enabled and xccl_writer is not None:
        xccl_writer.write("-" * 50)

@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))
        memory_recoder("ParallelEmbedding", "init", "malloc", "weight", self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        flops_recoder("ParallelEmbedding", "Embedding", x, y)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        xccl_recoder("ParallelEmbedding", "all_reduce", y)
        tensor_hist_recoder(rank, "ParallelEmbedding", "y", y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight.element_size() > 1:
        flops_recoder("linear", "GEMM", x, weight, bias)
        y = F.linear(x, weight, bias)
        tensor_hist_recoder(rank, "linear", "y", y)
        return y
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        flops_recoder("linear", "GEMM", x, weight, bias, weight.scale)
        y = F.linear(x, weight, bias)
        tensor_hist_recoder(rank, "linear", "y", y)
        return y
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        flops_recoder("linear", "GEMM", x, weight, bias, weight.scale)
        tensor_hist_recoder(rank, "linear", "y", y)
        return y


class Linear(nn.Module):
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        memory_recoder("Linear", "init", "malloc", "weight", self.weight)
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
            memory_recoder("Linear", "init", "malloc", "scale", self.scale)
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.part_out_features))
            memory_recoder("Linear", "init", "malloc", "bias", self.bias)
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        xccl_recoder("ParallelEmbedding", "all_reduce", y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        memory_recoder("RMSNorm", "init", "malloc", "weight", self.weight)

    def forward(self, x: torch.Tensor):
        x = x.float()
        flops_recoder("RMSNorm", "Vector(pow)", x)
        tensor_a = x.pow(2)
        flops_recoder("RMSNorm", "Vector(mean)", tensor_a)
        tensor_b = tensor_a.mean(-1, keepdim=True)
        flops_recoder("RMSNorm", "Vector(+)", tensor_a, self.eps)
        tensor_c = tensor_b + self.eps
        flops_recoder("RMSNorm", "Vector(rsqrt)", tensor_c)
        tensor_d = torch.rsqrt(tensor_c)
        flops_recoder("RMSNorm", "Vector(*)", x, tensor_c)
        y = x * tensor_d
        flops_recoder("RMSNorm", "Vector(*)", y, self.weight)
        y = y.type_as(self.weight) * self.weight
        tensor_hist_recoder(rank, "RMSNorm", "y", y)
        return y

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    flops_recoder("apply_rotary_emb", "GEMM", x, freqs_cis)
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    tensor_hist_recoder(rank, "apply_rotary_emb", "y", y)
    return y.to(dtype)


class MLA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
            memory_recoder("MLA", "init", "malloc", "k_cache", self.k_cache)
            memory_recoder("MLA", "init", "malloc", "v_cache", self.v_cache)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
            memory_recoder("MLA", "init", "malloc", "kv_cache", self.kv_cache)
            memory_recoder("MLA", "init", "malloc", "pe_cache", self.pe_cache)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            memory_recoder("Transformer", "cache", "malloc", "k", k)
            memory_recoder("Transformer", "cache", "malloc", "v", v)
            flops_recoder("MLA", "einsum(bshd_bthd->bsht)", q, self.k_cache[:bsz, :end_pos])
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos])
            flops_recoder("MLA", "Vector(*)", scores, self.softmax_scale)
            scores = scores * self.softmax_scale
            tensor_hist_recoder(rank, "MLA", "scores", scores)
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            kv_cache = self.kv_norm(kv)
            k_pe_cache = k_pe.squeeze(2)
            self.kv_cache[:bsz, start_pos:end_pos] = kv_cache
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe_cache
            memory_recoder("Transformer", "cache", "malloc", "kv_cache", kv_cache)
            memory_recoder("Transformer", "cache", "malloc", "k_pe_cache", k_pe_cache)
            flops_recoder("MLA", "einsum(bshc_btc->bsht)", q_nope, self.kv_cache[:bsz, :end_pos])
            scores_a = torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
            flops_recoder("MLA", "einsum(bshr_btr->bsht)", q_pe, self.pe_cache[:bsz, :end_pos])
            scores_b = torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            flops_recoder("MLA", "Vector(+)", q_pe, self.pe_cache[:bsz, :end_pos])
            scores = scores_a + scores_b
            flops_recoder("MLA", "Vector(*)", scores, self.softmax_scale)
            scores = scores * self.softmax_scale
            tensor_hist_recoder(rank, "MLA", "scores", scores)
        if mask is not None:
            tensor = mask.unsqueeze(1)
            flops_recoder("MLA", "Vector(+)", scores, tensor)
            scores += tensor
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            flops_recoder("MLA", "einsum(bsht_bthd->bshd)", scores, self.v_cache[:bsz, :end_pos])
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            flops_recoder("MLA", "einsum(bsht_btc->bshc)", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            flops_recoder("MLA", "einsum(bshc_hdc->bshd)", x, wkv_b[:, -self.v_head_dim:])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        tensor_hist_recoder(rank, "MLA", "x", x)
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensor = self.w1(x)
        flops_recoder("MLP", "silu", tensor)
        y = self.w2(F.silu(tensor) * self.w3(x))
        tensor_hist_recoder(rank, "MLP", "y", y)
        return y


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None
        memory_recoder("Gate", "init", "malloc", "weight", self.weight)
        memory_recoder("Gate", "init", "malloc", "bias", self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            flops_recoder("Gate", "softmax(dim=-1)", scores)
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            flops_recoder("Gate", "sigmoid", scores)
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            flops_recoder("Gate", "Vector(+)", scores, self.bias)
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            flops_recoder("Gate", "Sum(dim=-1)", weights)
            tensor = weights.sum(dim=-1, keepdim=True)
            flops_recoder("Gate", "Vector(/)", weights, tensor)
            weights /= tensor
        flops_recoder("Gate", "Vector(*)", weights, self.route_scale)
        weights *= self.route_scale
        tensor_hist_recoder(rank, "Gate", "weights", weights)
        return weights.type_as(x), indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensor = self.w1(x)
        flops_recoder("Expert", "silu", tensor)
        y = self.w2(F.silu(tensor) * self.w3(x))
        tensor_hist_recoder(rank, "Expert", "y", y)
        return y


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            tensor_a = expert(x[idx])
            flops_recoder("MoE", "Vector(*)", tensor_a, weights[idx, top, None])
            tensor_b = tensor_a * weights[idx, top, None]
            flops_recoder("MoE", "Vector(+)", y[idx], tensor_b)
            y[idx] += tensor_b
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        xccl_recoder("MoE", "all_reduce", y)
        flops_recoder("MoE", "Vector(+)", y, z)
        out = (y + z).view(shape)
        tensor_hist_recoder(rank, "MoE", "out", out)
        return out


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        tensor_a = self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        flops_recoder("Block", "Vector(+)", tensor_a, x)
        x += tensor_a
        tensor_b = self.ffn(self.ffn_norm(x))
        flops_recoder("Block", "Vector(+)", tensor_b, x)
        x += tensor_b
        tensor_hist_recoder(rank, "Block", "x", x)
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        global world_size, rank, weights_writer, memory_writer, flops_writer, xccl_writer
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        weights_writer = Weights_Writer(rank) if weights_writer_enabled else None
        memory_writer = Memory_Writer(rank) if memory_writer_enabled else None
        flops_writer = FLOPs_Writer(rank) if flops_writer_enabled else None
        xccl_writer = XCCL_Writer(rank) if xccl_writer_enabled else None
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        weights_recoder(self)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, targets: torch.Tensor = None):
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
            memory_recoder("Transformer", "forward", "malloc", "mask", mask)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        logits = self.head(h) if targets is not None else self.head(h[:, -1])
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        xccl_recoder("Transformer", "all_gather", logits)
        if targets is not None:
            return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            return logits, None
