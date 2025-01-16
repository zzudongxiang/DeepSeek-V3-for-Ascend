import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from kernel import act_quant, weight_dequant, fp8_gemm
from writer_utils import FLOPs_Writer, XCCL_Writer, Memory_Writer, Weights_Writer

rank = 0
world_size = 1
block_size = 128
disable_writer = False
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

xccl_writer = XCCL_Writer()
flops_writer = FLOPs_Writer()
memory_writer = Memory_Writer()

def writer_finished():
    memory_writer.finished("a+")
    flops_writer.finished()
    xccl_writer.finished()

def split_writer():
    flops_writer.write("-" * 50)
    xccl_writer.write("-" * 50)

def print_flops(flush_only = False):
    return flops_writer.print(flush_only)

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
        memory_writer.recoder("ParallelEmbedding", "init", "malloc", "weight", self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        flops_writer.recoder("ParallelEmbedding", "Embedding", x, y)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        xccl_writer.recoder("ParallelEmbedding", "all_reduce", y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weight.element_size() > 1:
        flops_writer.recoder("linear", "GEMM", x, weight, bias)
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        flops_writer.recoder("linear", "GEMM", x, weight, bias, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        flops_writer.recoder("linear", "GEMM", x, weight, bias, weight.scale)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        memory_writer.recoder("Linear", "init", "malloc", "weight", self.weight)
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
            memory_writer.recoder("Linear", "init", "malloc", "scale", self.scale)
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.part_out_features))
            memory_writer.recoder("Linear", "init", "malloc", "bias", self.bias)
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
        xccl_writer.recoder("RowParallelLinear", "all_reduce", y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        memory_writer.recoder("RMSNorm", "init", "malloc", "weight", self.weight)


    def forward(self, x: torch.Tensor):
        x = x.float()
        flops_writer.recoder("RMSNorm", "Vector(pow)", x)
        tensor_a = x.pow(2)
        flops_writer.recoder("RMSNorm", "Vector(mean)", tensor_a)
        tensor_b = tensor_a.mean(-1, keepdim=True)
        flops_writer.recoder("RMSNorm", "Vector(+)", tensor_a, self.eps)
        tensor_c = tensor_b + self.eps
        flops_writer.recoder("RMSNorm", "Vector(rsqrt)", tensor_c)
        tensor_d = torch.rsqrt(tensor_c)
        flops_writer.recoder("RMSNorm", "Vector(*)", x, tensor_c)
        y = x * tensor_c
        flops_writer.recoder("RMSNorm", "Vector(*)", y, self.weight)
        return y.type_as(self.weight) * self.weight


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
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    flops_writer.recoder("apply_rotary_emb", "GEMM", x, freqs_cis)
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
            memory_writer.recoder("MLA", "init", "malloc", "k_cache", self.k_cache)
            memory_writer.recoder("MLA", "init", "malloc", "v_cache", self.v_cache)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
            memory_writer.recoder("MLA", "init", "malloc", "kv_cache", self.kv_cache)
            memory_writer.recoder("MLA", "init", "malloc", "pe_cache", self.pe_cache)

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
            memory_writer.recoder("Transformer", "cache", "malloc", "k", k)
            self.v_cache[:bsz, start_pos:end_pos] = v
            memory_writer.recoder("Transformer", "cache", "malloc", "v", v)
            flops_writer.recoder("MLA", "einsum(bshd_bthd->bsht)", q, self.k_cache[:bsz, :end_pos])
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos])
            flops_writer.recoder("MLA", "Vector(*)", scores, self.softmax_scale)
            scores = scores * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            kv_cache = self.kv_norm(kv)
            self.kv_cache[:bsz, start_pos:end_pos] = kv_cache
            memory_writer.recoder("Transformer", "cache", "malloc", "kv_cache", kv_cache)
            k_pe_cache = k_pe.squeeze(2)
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe_cache
            memory_writer.recoder("Transformer", "cache", "malloc", "k_pe_cache", k_pe_cache)
            flops_writer.recoder("MLA", "einsum(bshc_btc->bsht)", q_nope, self.kv_cache[:bsz, :end_pos])
            scores_a = torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
            flops_writer.recoder("MLA", "einsum(bshr_btr->bsht)", q_pe, self.pe_cache[:bsz, :end_pos])
            scores_b = torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            flops_writer.recoder("MLA", "Vector(+)", q_pe, self.pe_cache[:bsz, :end_pos])
            scores = scores_a + scores_b
            flops_writer.recoder("MLA", "Vector(*)", scores, self.softmax_scale)
            scores = scores * self.softmax_scale
        if mask is not None:
            tensor = mask.unsqueeze(1)
            flops_writer.recoder("MLA", "Vector(+)", scores, tensor)
            scores = scores + tensor
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        if attn_impl == "naive":
            flops_writer.recoder("MLA", "einsum(bsht_bthd->bshd)", scores, self.v_cache[:bsz, :end_pos])
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            flops_writer.recoder("MLA", "einsum(bsht_btc->bshc)", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            flops_writer.recoder("MLA", "einsum(bshc_hdc->bshd)", x, wkv_b[:, -self.v_head_dim:])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensor = self.w1(x)
        flops_writer.recoder("MLP", "silu", tensor)
        return self.w2(F.silu(tensor) * self.w3(x))


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
        memory_writer.recoder("Gate", "init", "malloc", "weight", self.weight)
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None
        memory_writer.recoder("Gate", "init", "malloc", "bias", self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            flops_writer.recoder("Gate", "softmax(dim=-1)", scores)
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            flops_writer.recoder("Gate", "sigmoid", scores)
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            flops_writer.recoder("Gate", "Vector(+)", scores, self.bias)
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
            flops_writer.recoder("Gate", "Sum(dim=-1)", weights)
            tensor = weights.sum(dim=-1, keepdim=True)
            flops_writer.recoder("Gate", "Vector(/)", weights, tensor)
            weights /= weights.sum(dim=-1, keepdim=True)
        flops_writer.recoder("Gate", "Vector(*)", weights, self.route_scale)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tensor = self.w1(x)
        flops_writer.recoder("Expert", "silu", tensor)
        return self.w2(F.silu(tensor) * self.w3(x))


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
            flops_writer.recoder("MoE", "Vector(*)", tensor_a, weights[idx, top, None])
            tensor_b = tensor_a * weights[idx, top, None]
            flops_writer.recoder("MoE", "Vector(+)", y[idx], tensor_b)
            y[idx] = y[idx] + tensor_b
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        flops_writer.recoder("MoE", "Vector(+)", y, z)
        xccl_writer.recoder("MoE", "all_reduce", y)
        return (y + z).view(shape)


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        tensor_a = self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        flops_writer.recoder("Block", "Vector(+)", tensor_a, x)
        x = x + tensor_a
        tensor_b = self.ffn(self.ffn_norm(x))
        flops_writer.recoder("Block", "Vector(+)", tensor_b, x)
        x = x + tensor_b
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        global world_size, rank, flops_writer, xccl_writer, memory_writer
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        if not disable_writer:
            memory_writer = Memory_Writer(rank)
            flops_writer = FLOPs_Writer(rank)
            xccl_writer = XCCL_Writer(rank)
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
        if not disable_writer:
            weights_writer = Weights_Writer(rank)
            weights_writer.recoder(self)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        seqlen = tokens.size(1)
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
            memory_writer.recoder("Transformer", "forward", "malloc", "mask", mask)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        xccl_writer.recoder("Transformer", "all_gather", logits)
        return logits


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())
