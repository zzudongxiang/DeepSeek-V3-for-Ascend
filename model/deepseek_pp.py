import math
import torch
import torch_npu
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.quantization.fp8 import fp8_dequant
from utils.quantization.int4 import int4_dequant
from utils.quantization.int8 import int8_dequant
from model.deepseek.rms_norm import RMSNorm_NPU as RMSNorm
from model.deepseek.linear import set_linear_args, get_linear, Linear
from model.deepseek.rope import precompute_freqs_cis, apply_rotary_emb


class ParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, group=None):
        super().__init__()
        self.dim = dim
        self.group = group
        self.vocab_size = vocab_size
        self.local_rank = dist.get_rank(group=self.group) if dist.is_initialized() else 0
        self.group_size = dist.get_world_size(group=self.group) if dist.is_initialized() else 1
        assert vocab_size % self.group_size == 0
        self.part_vocab_size = (vocab_size // self.group_size)
        self.vocab_start_idx = self.local_rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))
        if self.group_size > 1:
            self.parallel_split = self.parallel_split
            self.parallel_merge = self.parallel_merge
        else:
            self.parallel_split = lambda x: (x, None)
            self.parallel_merge = lambda x, mask: x

    def parallel_split(self, x):
        mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
        x = x - self.vocab_start_idx
        x[mask] = 0
        return x, mask

    def parallel_merge(self, x, mask):
        x[mask] = 0
        dist.all_reduce(x, group=self.group)
        return x

    def forward(self, x):
        x, mask = self.parallel_split(x)
        y = F.embedding(x, self.weight)
        return self.parallel_merge(y, mask)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features, out_features, bias=False, dtype=None, x_shape="ND", group=None):
        self.group = group
        self.local_rank = dist.get_rank(group=group) if dist.is_initialized() else 0
        self.group_size =  dist.get_world_size(group=group) if dist.is_initialized() else 1
        assert out_features % self.group_size == 0
        self.part_out_features = out_features // self.group_size
        super().__init__(in_features, self.part_out_features, bias, dtype, x_shape)
        self.linear = get_linear(self.weight, x_shape)

    def forward(self, x):
        return self.linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    def __init__(self, in_features, out_features, bias=False, dtype=None, x_shape="ND", group=None):
        self.group = group
        self.group_size = dist.get_world_size(group=group) if dist.is_initialized() else 1
        assert in_features % self.group_size == 0
        self.part_in_features = in_features // self.group_size
        super().__init__(self.part_in_features, out_features, bias, dtype, x_shape)
        self.data_reduce = self.data_reduce if self.group_size > 1 else lambda x: x
        self.add_bias = lambda x: x if self.bias is None else lambda x: x + self.bias
        self.linear = get_linear(self.weight, x_shape)

    def data_reduce(self, x):
        dist.all_reduce(x, group=self.group)
        return x

    def forward(self, x):
        return self.add_bias(self.data_reduce(self.linear(x, self.weight)))


class MLA(nn.Module):
    def __init__(self, args, layer_id=0, group=None):
        self.group_size = dist.get_world_size(group=group) if dist.is_initialized() else 1
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // self.group_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.layer_id = layer_id
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim, group=group)
            self.q_linear = lambda x: self.wq(x)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank, x_shape="NCL")
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim, x_shape="NCL", group=group)
            self.q_linear = lambda x: self.wq_b(self.q_norm(self.wq_a(x)))
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, x_shape="NCL")
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), x_shape="NCL", group=group)
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim, x_shape="NCL", group=group)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        if args.attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
            self.get_score_part1 = self.get_naive_score_part1
            self.get_score_part2 = self.get_naive_score_part2
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
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

    def get_naive_score_part1(self, kv, q_nope, q_pe, k_pe, bsz, seqlen, bsz_index, start_pos, end_pos):
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
        bsz_start_index = bsz_index * self.args.mini_batch_size
        bsz_end_index = bsz_start_index + bsz
        self.k_cache[bsz_start_index: bsz_end_index, start_pos:end_pos] = k
        self.v_cache[bsz_start_index: bsz_end_index, start_pos:end_pos] = v
        scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[bsz_start_index: bsz_end_index, :end_pos]) * self.softmax_scale
        return scores, q, q_nope, kv, None

    def get_absorb_score_part1(self, kv, q_nope, q_pe, k_pe, bsz, seqlen, bsz_index, start_pos, end_pos):
        wkv_b = self.get_wkv_b_weight
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        bsz_start_index = bsz_index * self.args.mini_batch_size
        bsz_end_index = bsz_start_index + bsz
        self.kv_cache[bsz_start_index: bsz_end_index, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[bsz_start_index: bsz_end_index, start_pos:end_pos] = k_pe.squeeze(2)
        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[bsz_start_index: bsz_end_index, :end_pos]) +
                  torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[bsz_start_index: bsz_end_index, :end_pos])) * self.softmax_scale
        return scores, None, q_nope, kv, wkv_b

    def get_naive_score_part2(self, scores, wkv_b, bsz, bsz_index, end_pos):
        bsz_start_index = bsz_index * self.args.mini_batch_size
        bsz_end_index = bsz_start_index + bsz
        return torch.einsum("bsht,bthd->bshd", scores, self.v_cache[bsz_start_index: bsz_end_index, :end_pos])

    def get_absorb_score_part2(self, scores, wkv_b, bsz, bsz_index, end_pos):
        bsz_start_index = bsz_index * self.args.mini_batch_size
        bsz_end_index = bsz_start_index + bsz
        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[bsz_start_index: bsz_end_index, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        return x

    def forward(self, x, bsz_index, start_pos, freqs_cis, mask_func):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.q_linear(x)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        scores, q, q_nope, kv, wkv_b = self.get_score_part1(kv, q_nope, q_pe, k_pe, bsz, seqlen, bsz_index, start_pos, end_pos)
        scores = mask_func(scores)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = self.get_score_part2(scores, wkv_b, bsz, bsz_index, end_pos)
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    def __init__(self, dim, inter_dim, x_shape="ND", group=None):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim, x_shape=x_shape, group=group)
        self.w2 = RowParallelLinear(inter_dim, dim, x_shape=x_shape, group=group)
        self.w3 = ColumnParallelLinear(dim, inter_dim, x_shape=x_shape, group=group)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None
        self.linear = get_linear(self.weight)
        if self.score_func == "softmax":
            self.get_score = lambda x: x.softmax(dim=-1, dtype=torch.float32)
            self.get_weight = lambda x: x
        elif self.score_func == "sigmoid":
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
        self.calc_group_score = self.calc_group_score if self.n_groups > 1 else lambda x, scores: scores

    def calc_group_score(self, x, scores):
        scores = scores.view(x.size(0), self.n_groups, -1)
        group_scores = self.group_scores_func(scores)
        indices = group_scores.topk(self.topk_groups, dim=-1)[1]
        # mask = torch.zeros_like(scores[..., 0])
        # for i in range(scores.shape[0]):
        #     mask[i, indices[i]] = True
        mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
        scores = (scores * mask.unsqueeze(-1)).flatten(1)
        return scores

    def forward(self, x):
        scores = self.get_score(self.linear(x, self.weight))
        original_scores = scores
        scores = self.add_bias(scores)
        scores = self.calc_group_score(x, scores)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        weights = self.get_weight(weights)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    def __init__(self, dim, inter_dim):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args, device, layer_id=0, group=None):
        self.group = group
        self.local_rank = dist.get_rank(group=group) if dist.is_initialized() else 0
        self.group_size = dist.get_world_size(group=group) if dist.is_initialized() else 1
        super().__init__()
        self.device = device
        self.dim = args.dim
        self.layer_id = layer_id
        assert args.n_routed_experts % self.group_size == 0
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // self.group_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = self.local_rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim).to("cpu" if args.offload_cpu else device)
            if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim, group=group)
        self.data_reduce = self.data_reduce if self.group_size > 1 else lambda x: x
        self.calc = self.calc_in_cpu if args.offload_cpu else self.calc_in_xpu

    def data_reduce(self, x):
        dist.all_reduce(x, group=self.group)
        return x

    def calc_in_cpu(self, x, y, idx, expert, weights):
        x_cpu = x[idx].cpu()
        y_cpu = expert(x_cpu).to(self.device) * weights
        y[idx] += y_cpu.to(x.device)
        return y

    def calc_in_xpu(self, x, y, idx, expert, weights):
        y[idx] += expert(x[idx]) * weights
        return y

    def forward(self, x):
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
            y = self.calc(x, y, idx, expert, weights[idx, top, None])
        z = self.shared_experts(x)
        y = self.data_reduce(y)
        return (y + z).view(shape)


class Block(nn.Module):
    def __init__(self, layer_id, args, device, group=None):
        super().__init__()
        self.device = device
        self.attn = MLA(args, layer_id, group)
        self.ffn = MLP(args.dim, args.inter_dim, "NCL", group) if layer_id < args.n_dense_layers else MoE(args, device, layer_id, group)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x, bsz_index, start_pos, freqs_cis, mask_func):
        x = x + self.attn(self.attn_norm(x), bsz_index, start_pos, freqs_cis, mask_func)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args, device, tp_group, pp_stage):
        self.tp_group = tp_group
        self.pp_stage = pp_stage
        self.pp_stage_num = 1 if args.pp_layer_list is None else len(args.pp_layer_list)
        self.local_size = dist.get_world_size(group=tp_group) if dist.is_initialized() else 1
        self.local_rank = dist.get_rank(group=tp_group) if dist.is_initialized() else 0
        self.global_size = dist.get_world_size() if dist.is_initialized() else 1
        self.global_rank = dist.get_rank() if dist.is_initialized() else 0
        self.args = args
        self.device = device
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
        self.max_seq_len = args.max_seq_len
        assert args.max_batch_size % args.mini_batch_size == 0
        assert self.pp_stage_num < 2 or sum(args.pp_layer_list) == args.n_layers
        assert self.pp_stage_num < 2 or self.pp_stage < self.pp_stage_num
        self.layers = torch.nn.ModuleList()
        if self.pp_stage_num < 2 or self.pp_stage == 0:
            self.embed = ParallelEmbedding(args.vocab_size, args.dim, self.tp_group)
        self.start_layer_id = sum(args.pp_layer_list[:pp_stage]) if self.pp_stage_num > 1 else 0
        self.end_layer_id = self.start_layer_id + (args.pp_layer_list[pp_stage] if self.pp_stage_num > 1 else args.n_layers)
        for layer_id in range(args.n_layers):
            if self.start_layer_id <= layer_id < self.end_layer_id:
                self.layers.append(Block(layer_id, self.args, self.device, self.tp_group))
            else:
                self.layers.append(torch.nn.Identity())
        if self.pp_stage_num < 2 or self.pp_stage == self.pp_stage_num - 1:
            self.norm = RMSNorm(args.dim)
            self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype(), group=self.tp_group)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        self.gather_logits = self.gather_logits if self.local_size > 1 else lambda x: x
        self.recv_h = self.recv_h_local if self.pp_stage_num < 2 or self.pp_stage == 0 else self.recv_h_remote
        self.send_h = self.send_h_local if self.pp_stage_num < 2 or self.pp_stage == self.pp_stage_num - 1 else self.send_h_remote
        self.dst_rank = (self.pp_stage + 1) * self.local_size + (self.global_rank % self.local_size)
        self.src_rank = (self.pp_stage - 1) * self.local_size + (self.global_rank % self.local_size)
        self.dst_rank = self.dst_rank if self.dst_rank < self.global_size else self.dst_rank - self.global_size
        self.src_rank = self.global_size + self.src_rank if self.src_rank < 0 else self.src_rank

    def gather_logits(self, logits):
        all_logits = [torch.empty_like(logits) for _ in range(self.local_size)]
        dist.all_gather(all_logits, logits, group=self.tp_group)
        return torch.cat(all_logits, dim=-1)

    def recv_h_local(self, tokens, tag):
        yield from self.embed(tokens)

    def recv_h_remote(self, tokens, tag):
        batch_size, seqlen = tokens.shape
        h = torch.zeros(batch_size, seqlen, self.args.dim, device=self.device)
        dist.recv(h, src=self.src_rank, tag=tag)
        return h

    def send_h_local(self, h, tag):
        h = self.norm(h)[:, -1]
        logits = self.gather_logits(self.head(h))
        return logits

    def send_h_remote(self, h, tag):
        dist.send(h, dst=self.dst_rank, tag=tag)
        return None

    def update_wkv_b_cache(self):
        for layer_id in range(self.start_layer_id, self.end_layer_id):
            self.layers[layer_id].attn.update_wkv_b_cache()

    @torch.inference_mode()
    def forward(self, tokens, bsz_index=0, start_pos=0):
        seqlen = tokens.size(1)
        h = self.recv_h(tokens, bsz_index)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
            mask_func = lambda x: x + mask.unsqueeze(1)
        else:
            mask_func = lambda x: x
        for layer_id in range(self.start_layer_id, self.end_layer_id):
            h = self.layers[layer_id](h, bsz_index, start_pos, freqs_cis, mask_func)
        return self.send_h(h, bsz_index)
