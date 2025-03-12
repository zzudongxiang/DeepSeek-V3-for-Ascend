import math
import torch
from torch import nn
from typing import Tuple
import torch.nn.functional as F
import torch.distributed as dist
from model.deepseek.args import ModelArgs
from utils.quantization.fp8 import fp8_dequant
from utils.quantization.int4 import int4_dequant
from utils.quantization.int8 import int8_dequant
from utils.logger import log_moe_prefetch_rank0
from model.deepseek.linear import set_linear_args, get_linear
from model.deepseek.rope import precompute_freqs_cis, apply_rotary_emb
import os
from typing import Optional

rank, world_size = 0, 1  # world_size表示参与分布式计算的进程总数（或者说设备总数）

def save_tensor(tensor: torch.Tensor, path: str, filename: str):
    os.makedirs(path, exist_ok=True)
    torch.save(tensor.detach().cpu(), os.path.join(path, filename))

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
        if world_size > 1:
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
        dist.all_reduce(x)
        return x

    def forward(self, x):
        x, mask = self.parallel_split(x)
        y = F.embedding(x, self.weight)
        return self.parallel_merge(y, mask)


class Linear(nn.Module):
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, x_shape="ND"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if dtype is None and Linear.dtype != torch.bfloat16:
            # 由于使用torch_npu的MatMul算子，需要对Weight转置，所以提前转置Weight矩阵
            if Linear.dtype == torch.int8:
                self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=Linear.dtype), requires_grad=False)
            elif Linear.dtype == torch.int32:
                assert out_features % 8 == 0
                self.weight = nn.Parameter(torch.empty(in_features, out_features // 8, dtype=Linear.dtype), requires_grad=False)
            else:
                raise ValueError(f"Unsupported dtype: {Linear.dtype}")
        else:
            # embed层不需要量化处理，所以不需要转置Weight
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype), requires_grad=False)
        if self.weight.dtype != torch.bfloat16:
            self.weight.scale = self.scale = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.part_out_features))
        else:
            self.register_parameter("bias", None)
        self.linear = get_linear(self.weight, x_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, x_shape="ND"):
        assert out_features % world_size == 0
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype, x_shape)
        self.linear = get_linear(self.weight, x_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, x_shape="ND"):
        assert in_features % world_size == 0
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype, x_shape)
        self.data_reduce = self.data_reduce if world_size > 1 else lambda x: x
        self.add_bias = lambda x: x if self.bias is None else lambda x: x + self.bias
        self.linear = get_linear(self.weight, x_shape)

    def data_reduce(self, x):
        dist.all_reduce(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.add_bias(self.data_reduce(self.linear(x, self.weight)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x = x.float()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(self.weight) * self.weight


class MLA(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int = 0):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads # 将注意力头在进程间平均分配
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.layer_id = layer_id

        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
            self.q_linear = lambda x: self.wq(x)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank, x_shape="NCL")
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim, x_shape="NCL")
            self.q_linear = lambda x: self.wq_b(self.q_norm(self.wq_a(x)))
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, x_shape="NCL")
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), x_shape="NCL")
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim, x_shape="NCL")
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        if args.attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
            self.get_score_part1 = self.get_naive_score_part1
            self.get_score_part2 = self.get_naive_score_part2
            self.get_wkv_b_weight = lambda: self.wkv_b.weight
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
            self.get_score_part1 = self.get_absorb_score_part1
            self.get_score_part2 = self.get_absorb_score_part2
            # TODO: 这里的wkv可以提前计算并储存，但是储存所需的显存容量较大
            if self.wkv_b.weight.dtype == torch.bfloat16:
                self.get_wkv_b_weight = lambda: self.wkv_b.weight
            elif self.wkv_b.weight.dtype == torch.float8_e4m3fn:
                self.get_wkv_b_weight = lambda: fp8_dequant(self.wkv_b.weight, self.wkv_b.scale)
            elif self.wkv_b.weight.dtype == torch.int8:
                self.get_wkv_b_weight = lambda: int8_dequant(self.wkv_b.weight, self.wkv_b.scale).T
            elif self.wkv_b.weight.dtype == torch.int32:
                self.get_wkv_b_weight = lambda: int4_dequant(self.wkv_b.weight, self.wkv_b.scale).T
            else:
                raise NotImplementedError(f"Unsupported dtype: {self.wkv_b.weight.dtype}")

    def get_naive_score_part1(self, kv, q_nope, q_pe, k_pe, bsz, seqlen, start_pos, end_pos):
        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v
        scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        return scores, q, q_nope, kv, None

    def get_absorb_score_part1(self, kv, q_nope, q_pe, k_pe, bsz, seqlen, start_pos, end_pos):
        wkv_b = self.get_wkv_b_weight()
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
                  torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        return scores, None, q_nope, kv, wkv_b

    def get_naive_score_part2(self, scores, wkv_b, bsz, end_pos):
        return torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])

    def get_absorb_score_part2(self, scores, wkv_b, bsz, end_pos):
        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        return x

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask_func):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.q_linear(x)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        scores, q, q_nope, kv, wkv_b = self.get_score_part1(kv, q_nope, q_pe, k_pe, bsz, seqlen, start_pos, end_pos)
        scores = mask_func(scores)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = self.get_score_part2(scores, wkv_b, bsz, end_pos)
        x = self.wo(x.flatten(2))
        return x


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int, x_shape: str="ND"):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim, x_shape=x_shape)
        self.w2 = RowParallelLinear(inter_dim, dim, x_shape=x_shape)
        self.w3 = ColumnParallelLinear(dim, inter_dim, x_shape=x_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts # 8
        self.n_groups = args.n_expert_groups # 8
        self.topk_groups = args.n_limited_groups # 4
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
            self.group_scores_func = lambda x: x.topk(2, dim=-1)[0].sum(dim=-1) # topk # 返回一个元组 (values, indices), values 表示每个 group 中得分最高的 topk_groups 个专家的得分, indices 表示每个 group 中得分最高的 topk_groups 个专家的索引
        self.calc_group_score = self.calc_group_score if self.n_groups > 1 else lambda x, scores: scores
        self.should_save = False  # 添加控制保存的标志
        self.layer_id = None  # 添加layer_id用于保存路径
        self.prefetch_mode = "decode"  # 默认为decode模式

    def calc_group_score(self, x, scores):
        scores = scores.view(x.size(0), self.n_groups, -1)
        group_scores = self.group_scores_func(scores) # [batch_size, n_groups]，表示每个 group 的重要性得分
        indices = group_scores.topk(self.topk_groups, dim=-1)[1] # [batch_size, topk_groups]，表示每个 group 中得分最高的 topk_groups 个专家的索引
        mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True) # [batch_size, n_groups]，表示每个 group 中得分最高的 topk_groups 个专家的掩码
        scores = (scores * mask.unsqueeze(-1)).flatten(1)
        return scores   # [batch_size, n_routed_experts]，但其中仅 topk_groups * experts_per_group 个分数有效，其他值为 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.get_score(self.linear(x, self.weight))    # [batch_size, n_routed_experts]
        original_scores = scores

        # 保存加上bias前的gate scores
        if self.should_save and rank == 0 and self.layer_id is not None:
            save_path = f"saves/layer_{self.layer_id}/moe/gate"
            save_tensor(original_scores, save_path, "scores_before_bias.pt")

        scores = self.add_bias(scores)
        
        # 保存加上bias后的gate scores
        if self.should_save and rank == 0 and self.layer_id is not None:
            save_path = f"saves/layer_{self.layer_id}/moe/gate"
            save_tensor(scores, save_path, "scores_after_bias.pt")
            
        scores = self.calc_group_score(x, scores)   # [batch_size, n_groups, experts_per_group]
        indices = torch.topk(scores, self.topk, dim=-1)[1]   # [batch_size, topk]
        weights = original_scores.gather(1, indices)   # [batch_size, topk]
        weights = self.get_weight(weights)
        weights *= self.route_scale
        return weights.type_as(x), indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, layer_id: int = 0, expert_id: int = 0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)
        self.expert_id = expert_id
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, args: ModelArgs, device: str, layer_id: int = 0):
        super().__init__()
        self.device = device
        self.dim = args.dim
        self.layer_id = layer_id
        assert args.n_routed_experts % world_size == 0
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.gate.layer_id = layer_id  # 设置gate的layer_id
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim, layer_id, i).to("cpu" if args.offload_cpu else device)
            if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
        self.data_reduce = self.data_reduce if world_size > 1 else lambda x: x
        self.calc = self.calc_in_cpu if args.offload_cpu else self.calc_in_xpu
        self.should_save = False  # 添加控制保存的标志
        
        # 预取相关属性 - 设置为非持久化参数
        self.register_buffer("_prefetch_indices", None, persistent=False)
        self.register_buffer("_actual_indices", None, persistent=False)
        self._prefetch_hit_count = 0   # 预取命中计数
        self._prefetch_total_count = 0 # 预取总次数
        self._prefetched_experts = set()  # 已预取的专家集合
        self.offload_cpu = args.offload_cpu  # 是否将专家卸载到CPU
        
        # 添加推理步骤计数器
        self.inference_step = 0
        self._file_initialized = False  # 添加文件初始化标志
        self.prefetch_mode = "decode"  # 默认为decode模式
        
    # 为预取相关属性添加属性访问器
    @property
    def prefetch_indices(self):
        return self._prefetch_indices
        
    @prefetch_indices.setter
    def prefetch_indices(self, value):
        self._prefetch_indices = value
        
    @property
    def actual_indices(self):
        return self._actual_indices
        
    @actual_indices.setter
    def actual_indices(self, value):
        self._actual_indices = value
        
    @property
    def prefetch_hit_count(self):
        return self._prefetch_hit_count
        
    @prefetch_hit_count.setter
    def prefetch_hit_count(self, value):
        self._prefetch_hit_count = value
        
    @property
    def prefetch_total_count(self):
        return self._prefetch_total_count
        
    @prefetch_total_count.setter
    def prefetch_total_count(self, value):
        self._prefetch_total_count = value
        
    @property
    def prefetched_experts(self):
        return self._prefetched_experts
        
    @prefetched_experts.setter
    def prefetched_experts(self, value):
        self._prefetched_experts = value

    def data_reduce(self, x):
        dist.all_reduce(x)
        return x

    def calc_in_cpu(self, x, y, idx, expert, weights):
        x_cpu = x[idx].cpu()
        y_cpu = expert(x_cpu).to(self.device) * weights
        y[idx] += y_cpu.to(x.device)
        return y

    def calc_in_xpu(self, x, y, idx, expert, weights):
        y[idx] += expert(x[idx]) * weights
        return y

    def prefetch_experts(self, indices: torch.Tensor):
        """预取专家模型
        
        Args:
            indices: 预测的专家索引
        """
        if indices is None:
            return
        
        # 获取需要预取的专家索引
        expert_indices = set(indices.flatten().tolist())
        
        # 只预取本节点负责的专家
        local_expert_indices = [i for i in expert_indices 
                               if self.experts_start_idx <= i < self.experts_end_idx]
        
        # 记录已预取的专家
        for i in local_expert_indices:
            if i not in self.prefetched_experts:
                # 如果专家被卸载到CPU，则预取到设备内存
                if self.offload_cpu and i >= self.experts_start_idx and i < self.experts_end_idx:
                    expert = self.experts[i]
                    if expert is not None and next(expert.parameters()).device != self.device:
                        # 打印预取信息
                        log_moe_prefetch_rank0(f"Prefetching expert {i} for layer {self.layer_id}")
                        # 将专家移动到设备
                        self.experts[i] = expert.to(self.device)
                self.prefetched_experts.add(i)

    def predict_next_layer_experts(self, x: torch.Tensor) -> torch.Tensor:
        """预测下一层可能需要的专家索引"""
        shape = x.size()
        x = x.view(-1, self.dim)
        with torch.no_grad():
            # 只使用gate进行预测，不执行实际的专家计算
            _, indices = self.gate(x)
        return indices

    def record_actual_indices(self, indices: torch.Tensor):
        """记录实际使用的专家索引"""
        self.actual_indices = indices

    def calculate_prefetch_hit_rate(self, mode="decode"):
        """计算预取命中率"""
        if self.prefetch_indices is None or self.actual_indices is None:
            return 0.0
        
        # 将预测的索引和实际索引转换为集合
        prefetch_set = set(self.prefetch_indices.flatten().tolist())
        actual_set = set(self.actual_indices.flatten().tolist())
        
        # 保存第一个batch的预取索引和实际索引数据
        if rank == 0:  # 只在rank 0上保存
            save_path = f"saves/layer_{self.layer_id}/moe/prefetch"
            os.makedirs(save_path, exist_ok=True)
            
            # 获取第一个batch的数据
            first_batch_prefetch = self.prefetch_indices[0].detach().cpu()
            first_batch_actual = self.actual_indices[0].detach().cpu()
            
            # 根据模式选择文件后缀
            suffix = "_prefill" if mode == "prefill" else "_decode"
            
            # 保存张量数据
            torch.save(first_batch_prefetch, os.path.join(save_path, f"prefetch_indices{suffix}.pt"))
            torch.save(first_batch_actual, os.path.join(save_path, f"actual_indices{suffix}.pt"))
            
            # 文本文件路径
            txt_file = os.path.join(save_path, f"indices_info{suffix}.txt")
            
            # 如果是第一次调用，清空文件
            if not hasattr(self, f'_file_initialized_{mode}'):
                with open(txt_file, "w") as f:
                    f.write("")  # 清空文件
                setattr(self, f'_file_initialized_{mode}', True)
            
            # 追加写入新的数据
            with open(txt_file, "a") as f:
                f.write(f"\n=== Inference Step {self.inference_step} ===\n")
                f.write(f"Prefetch indices: {first_batch_prefetch.tolist()}\n")
                f.write(f"Actual indices: {first_batch_actual.tolist()}\n")
                f.write(f"Prefetch set: {sorted(list(prefetch_set))}\n")
                f.write(f"Actual set: {sorted(list(actual_set))}\n")
                f.write(f"Common indices: {sorted(list(prefetch_set.intersection(actual_set)))}\n")
                f.write(f"Hit rate: {len(prefetch_set.intersection(actual_set)) / len(actual_set) if len(actual_set) > 0 else 0:.4f}\n")
        
        # 计算交集大小
        hit_count = len(prefetch_set.intersection(actual_set))
        self.prefetch_hit_count += hit_count
        self.prefetch_total_count += len(actual_set)
        
        # 返回命中率
        if len(actual_set) == 0:
            return 0.0
        return hit_count / len(actual_set)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 传递should_save标志给gate
        self.gate.should_save = self.should_save
        
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        
        # 记录实际使用的专家索引
        self.record_actual_indices(indices)
        
        # 计算命中率时传入当前模式
        hit_rate = self.calculate_prefetch_hit_rate(self.prefetch_mode)
        if rank == 0 and self.prefetch_total_count > 0:
            log_moe_prefetch_rank0(f"Layer {self.layer_id} MoE prefetch hit rate: {hit_rate:.4f}, "
                         f"Total hit rate: {self.prefetch_hit_count / self.prefetch_total_count:.4f}")
        
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y = self.calc(x, y, idx, expert, weights[idx, top, None])
            
            # 使用完专家后，如果需要卸载到CPU，则进行卸载
            if self.offload_cpu and i in self.prefetched_experts:
                self.experts[i] = expert.to("cpu")
                self.prefetched_experts.remove(i)
                
        z = self.shared_experts(x)
        y = self.data_reduce(y)
        
        # 更新推理步骤计数器
        self.inference_step += 1
        
        return (y + z).view(shape)


class Block(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs, device: str):
        super().__init__()
        self.device = device
        self.layer_id = layer_id
        self.attn = MLA(args, layer_id)
        self.ffn = MLP(args.dim, args.inter_dim, x_shape="NCL") if layer_id < args.n_dense_layers else MoE(args, device, layer_id)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)
        self.should_save = False
        # 预取相关属性 - 将next_layer_moe设置为非持久化参数
        self.register_buffer("_next_layer_moe", None, persistent=False)

    def set_next_layer_moe(self, next_layer_moe):
        """设置下一层的MoE引用"""
        self._next_layer_moe = next_layer_moe
        
    @property
    def next_layer_moe(self):
        """获取下一层的MoE引用"""
        return self._next_layer_moe

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask_func) -> torch.Tensor:
        if self.should_save and rank == 0:
            base_path = f"saves/layer_{self.layer_id}"
            
            # 保存注意力层标准化的输入输出
            save_tensor(x, f"{base_path}/attn_norm", "input.pt")
            attn_norm_out = self.attn_norm(x)
            save_tensor(attn_norm_out, f"{base_path}/attn_norm", "output.pt")
            
            # 保存注意力层的输入输出
            save_tensor(attn_norm_out, f"{base_path}/attn", "input.pt")
            attn_out = self.attn(attn_norm_out, start_pos, freqs_cis, mask_func)
            save_tensor(attn_out, f"{base_path}/attn", "output.pt")
            
            x = x + attn_out
            
            # 保存前馈层标准化的输入输出
            save_tensor(x, f"{base_path}/ffn_norm", "input.pt")
            ffn_norm_out = self.ffn_norm(x)
            save_tensor(ffn_norm_out, f"{base_path}/ffn_norm", "output.pt")
            
            # 如果当前层是MoE层且下一层也是MoE层，进行预取
            if isinstance(self.ffn, MoE) and self.next_layer_moe is not None:
                # 使用当前层的MoE输入预测下一层可能需要的专家
                next_layer_indices = self.next_layer_moe.predict_next_layer_experts(ffn_norm_out)
                # 存储预测的专家索引到下一层MoE
                self.next_layer_moe.prefetch_indices = next_layer_indices
                # 执行实际的预取逻辑
                self.next_layer_moe.prefetch_experts(next_layer_indices)
            
            # 如果是MoE层，传递should_save标志
            if isinstance(self.ffn, MoE):
                self.ffn.should_save = True
                # 设置预取模式为"prefetch"
                self.ffn.prefetch_mode = "prefill"
                print(f"Layer {self.layer_id} MoE prefill mode")

            # 保存前馈层的输入输出
            save_tensor(ffn_norm_out, f"{base_path}/ffn", "input.pt")
            ffn_out = self.ffn(ffn_norm_out)
            save_tensor(ffn_out, f"{base_path}/ffn", "output.pt")
            
            x = x + ffn_out
            
            # 重置MoE的should_save标志和预取模式
            if isinstance(self.ffn, MoE):
                self.ffn.should_save = False

        else:
            # 注意力层处理
            attn_norm_out = self.attn_norm(x)
            attn_out = self.attn(attn_norm_out, start_pos, freqs_cis, mask_func)
            x = x + attn_out
            
            # 前馈层标准化
            ffn_norm_out = self.ffn_norm(x)
            
            # 如果当前层是MoE层且下一层也是MoE层，进行预取
            if isinstance(self.ffn, MoE):
                self.ffn.prefetch_mode = "decode"
                if self.next_layer_moe is not None:
                    next_layer_indices = self.next_layer_moe.predict_next_layer_experts(ffn_norm_out)
                    self.next_layer_moe.prefetch_indices = next_layer_indices
                    self.next_layer_moe.prefetch_experts(next_layer_indices)
            
            # 执行前馈层计算
            ffn_out = self.ffn(ffn_norm_out)
            x = x + ffn_out
            
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, device: str):
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
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
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args, self.device))
        
        # 设置每一层的next_layer_moe引用
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            
            # 如果当前层和下一层都是MoE层，设置next_layer_moe引用
            if (isinstance(current_layer.ffn, MoE) and 
                isinstance(next_layer.ffn, MoE)):
                current_layer.set_next_layer_moe(next_layer.ffn)
        
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        self.gather_logits = self.gather_logits if world_size > 1 else lambda x: x
        self.save_done = False  # 保存完成标记
        
        # 将统计相关属性设置为非持久化
        self._inference_count = 0  # 添加推理计数器
        self._total_prefetch_hit_count = 0  # 总预取命中计数
        self._total_prefetch_total_count = 0  # 总预取总次数
        
    # 为统计相关属性添加属性访问器
    @property
    def inference_count(self):
        return self._inference_count
        
    @inference_count.setter
    def inference_count(self, value):
        self._inference_count = value
        
    @property
    def total_prefetch_hit_count(self):
        return self._total_prefetch_hit_count
        
    @total_prefetch_hit_count.setter
    def total_prefetch_hit_count(self, value):
        self._total_prefetch_hit_count = value
        
    @property
    def total_prefetch_total_count(self):
        return self._total_prefetch_total_count
        
    @total_prefetch_total_count.setter
    def total_prefetch_total_count(self, value):
        self._total_prefetch_total_count = value

    def gather_logits(self, logits):
        all_logits = [torch.empty_like(logits) for _ in range(world_size)]
        dist.all_gather(all_logits, logits)
        return torch.cat(all_logits, dim=-1)
        
    def get_prefetch_statistics(self):
        """获取所有MoE层的预取统计信息"""
        total_hit_count = 0
        total_count = 0
        
        for layer in self.layers:
            if isinstance(layer.ffn, MoE):
                moe = layer.ffn
                total_hit_count += moe.prefetch_hit_count
                total_count += moe.prefetch_total_count
                
        return total_hit_count, total_count

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        seqlen = tokens.size(1)
        
        self.inference_count += 1
        
        # self.inference_count += 1
        # should_save_this_time = (self.inference_count == 800) and (not self.save_done) and (rank == 0)
        if seqlen > 1 and rank == 0 :
            should_save_this_time = True
        else:
            should_save_this_time = False
        
        if should_save_this_time:
            # 保存embedding层的输入输出
            save_tensor(tokens, "saves/embedding", "input.pt")
            h = self.embed(tokens)
            save_tensor(h, "saves/embedding", "output.pt")
        else:
            h = self.embed(tokens)
            
        freqs_cis = self.freqs_cis[start_pos:start_pos + tokens.size(1)]
        
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
            mask_func = lambda x: x + mask.unsqueeze(1)
        else:
            mask_func = lambda x: x
            
        if should_save_this_time:
            print(f"Saving layer time: {self.inference_count}")
            # 设置所有层保存中间结果
            for layer in self.layers:
                layer.should_save = True
                
        for layer in self.layers:
            # if should_save_this_time:
            #     print(f"input:{torch.sum(h)}")
            h = layer(h, start_pos, freqs_cis, mask_func)
            
        if should_save_this_time:
            # 关闭保存并标记已完成
            for layer in self.layers:
                layer.should_save = False
            self.save_done = True
            
        # 每100次推理输出一次预取统计信息
        if self.inference_count % 50 == 0 and rank == 0:
            hit_count, total_count = self.get_prefetch_statistics()
            if total_count > 0:
                hit_rate = hit_count / total_count
                log_moe_prefetch_rank0(f"Inference {self.inference_count}, Total MoE prefetch hit rate: {hit_rate:.4f} ({hit_count}/{total_count})")
            
        h = self.norm(h)[:, -1]
        logits = self.gather_logits(self.head(h))
        return logits
