import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from model.deepseek.linear import Linear
from model.deepseek.rope import apply_rotary_emb

rank, world_size = 0, 1

class ParallelEmbedding(nn.Module):
    def forward(self, x):
        # 1 -> 230: 1.239ms
        mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
        x = x - self.vocab_start_idx
        x[mask] = 0
        y = F.embedding(x, self.weight)
        y[mask] = 0
        dist.all_reduce(y) # 109: 602us (14kB -> 22.7 MB/s)
        return self.parallel_merge(y, mask)


class ColumnParallelLinear(Linear):
    def forward(self, x):
        return self.linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    def forward(self, x):
        y = self.linear(x, self.weight)
        dist.all_reduce(y)
        return self.add_bias(y)


class MLA(nn.Module):
    def forward(self, x, start_pos, freqs_cis):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(self.q_norm(self.wq_a(x)))                                                # Layer 0 [Dense] 253 -> 285: 1.044ms
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)   # Layer 0 [Dense] 285 -> 288: 0.099ms
        q_pe = apply_rotary_emb(q_pe, freqs_cis)                                                # Layer 0 [Dense] 288 -> 300: 0.295ms
        kv = self.wkv_a(x)                                                                      # Layer 0 [Dense] 300 -> 303: 0.134ms
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)                                   # Layer 0 [Dense] 303 -> 312: 0.304ms
        wkv_b = self.get_wkv_b_weight
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])       # Layer 0 [Dense] 312 -> 325: 0.446ms
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)                               # Layer 0 [Dense] 325 -> 348: 0.820ms
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)                                # Layer 0 [Dense] 348 -> 350: 0.001ms (2次异步memcpy)
        scores1 = torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])         # Layer 0 [Dense] 350 -> 356: 0.248ms
        scores2 = torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])           # Layer 0 [Dense] 356 -> 363: 0.226ms
        scores = (scores1 + scores2) * self.softmax_scale                                       # Layer 0 [Dense] 363 -> 369: 0.199ms
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)                         # Layer 0 [Dense] 369 -> 378: 0.303ms
        x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])               # Layer 0 [Dense] 378 -> 384: 0.208ms
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])                      # Layer 0 [Dense] 384 -> 391: 0.208ms
        x = self.wo(x.flatten(2))                                                               # Layer 0 [Dense] 391 -> 447: 0.242ms (14KB -> 179.43MB/s)
        return x


class MLP(nn.Module):
    def __init__(self, dim, inter_dim, x_shape="ND"):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim, x_shape=x_shape)
        self.w2 = RowParallelLinear(inter_dim, dim, x_shape=x_shape)
        self.w3 = ColumnParallelLinear(dim, inter_dim, x_shape=x_shape)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    def calc_group_score(self, x, scores):
        scores = scores.view(x.size(0), self.n_groups, -1)
        group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
        indices = group_scores.topk(self.topk_groups, dim=-1)[1]
        mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
        scores = (scores * mask.unsqueeze(-1)).flatten(1)
        return scores

    def forward(self, x):
        scores = self.linear(x, self.weight).sigmoid()          # Layer 3 [MoE] 2064 -> 2071: 0.231ms
        original_scores = scores
        scores += self.bias                                     # Layer 3 [MoE] 2071 -> 2075: 0.092ms
        scores = self.calc_group_score(x, scores)               # Layer 3 [MoE] 2075 -> 2105: 0.974ms
        indices = torch.topk(scores, self.topk, dim=-1)[1]      # Layer 3 [MoE] 2105 -> 2111: 0.197ms
        weights = original_scores.gather(1, indices)            # Layer 3 [MoE] 2111 -> 2114: 0.110ms
        weights /= weights.sum(dim=-1, keepdim=True)            # Layer 3 [MoE] 2114 -> 2120: 0.208ms
        weights *= self.route_scale                             # Layer 3 [MoE] 2120 -> 2122: 0.312ms
        return weights.type_as(x), indices


class Expert(nn.Module):
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def forward(self, x):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)                                                         # Layer 3 [MoE] 2064 -> 2122: 1.921ms
        y = torch.zeros_like(x)                                                                 # Layer 3 [MoE] 2122 -> 2126: 0.096ms
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()    # Layer 3 [MoE] 2126 -> 2141: 0.760ms
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)                                                # Layer 3 [MoE] 2141 -> 2149: 0.462ms
                                                                                                # Layer 3 [MoE] 2186 -> 2195: 0.443ms
                                                                                                # Layer 3 [MoE] 2230 -> 2239: 0.461ms

            y[idx] += expert(x[idx]) * weights[idx, top, None]                                  # Layer 3 [MoE] 2149 -> 2186: 1.737ms
                                                                                                # Layer 3 [MoE] 2195 -> 2230: 1.351ms
                                                                                                # Layer 3 [MoE] 2239 -> 2321: 1.420ms

        z = self.shared_experts(x)                                                              # Layer 3 [MoE] 2321 -> 2417: 0.597ms (14KB -> 1.0GB/s) !!!偶尔会特别慢(1.27MB/s)!!!
        dist.all_reduce(y)                                                                      # Layer 3 [MoE] 2417 -> 2467: 0.030ms (14KB -> 0.95GB/s)
        return (y + z).view(shape)                                                              # Layer 3 [MoE] 2467 -> 2486: 0.104ms


class Block(nn.Module):
    def forward(self, x, start_pos, freqs_cis, mask_func):
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask_func)   # layer 0 [Dense]: 230 -> 451: 5.470ms
                                                                                #    |- attn_norm: 230 -> 253: 0.843ms
                                                                                #    |- attn: 253 -> 447: 5.625ms (14KB -> 179.43MB/s)
                                                                                #    |- add: 447 -> 451: 0.115ms

        x = x + self.ffn(self.ffn_norm(x))                                      # layer 0 [Dense]: 451 -> 546: 1.659ms
                                                                                #    |- ffn_norm: 451 -> 475: 0.825ms
                                                                                #    |- ffn: 475 -> 544: 0.737ms (14KB -> 174.04MB/s)
                                                                                #    |- add: 544 -> 546: 0.076ms
                                                                                # layer 3 [MoE]: 1890 -> 2494: 9.954ms
                                                                                #    |- ffn_norm: 1890 -> 2064: 0.844ms
                                                                                #    |- ffn: 2064 -> 2486: 9.001ms
                                                                                #    |- add: 2486 -> 2494: 0.109ms
        return x


class Transformer(nn.Module):
    def forward(self, tokens, bsz_index=0, start_pos=0):
        seqlen = tokens.size(1)
        h = self.embed(tokens) # 1 -> 230: 1.239ms
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
            mask_func = lambda x: x + mask.unsqueeze(1)
        else:
            mask_func = lambda x: x
        for layer in self.layers:
            # layer 0 [Dense]: 230 -> 546: 7.399ms
            # layer 1 [Dense]: 546 -> 858: 7.334ms
            # layer 2 [Dense]: 858 -> 1273: 7.259ms
            # layer 3 [MoE]: 1273 -> 2494: 15.585ms
            # layer 4 [MoE]: 2494 -> 3871: 17.575ms
            # layer 5 [MoE]: 3871 -> 4976: 15.569ms
            h = layer(h, start_pos, freqs_cis, mask_func)
            # layer 58 [MoE]: 60251 -> 61073: 15.509ms
            # layer 59 [MoE]: 61073 -> 62318: 15.527ms
            # layer 61 [MoE]: 62318 -> 63444: 17.485ms
        h = self.norm(h)[:, -1]                                             # 63444 -> 63579: 0.822ms
        logits = self.head(h)                                               # 63579 -> 63630: 0.360ms
        all_logits = [torch.empty_like(logits) for _ in range(world_size)]  # 63630 -> 63922: 1.897ms
        dist.all_gather(all_logits, logits)                                 # 53924 -> 64149: 0.083ms (32kB -> 376 MB/s)
        return torch.cat(all_logits, dim=-1)                                # 64149 -> 64169: 0.121ms

# RMSNorm: 0.8~1.0ms
# Embedding (with AllReduce): 1.24ms
# Linear (with AllReduce): 0.2~0.3ms
# Linear (wo AllReduce): 0.15 ~ 0.2ms

# MLA: 5.47ms
# MLP: 1.6ms
# MoE: 10ms
#  |- Routed_Expert: 1.5ms (Rank 0 Layer 3 共计3个Expert选中，每个耗时1.5ms左右)
#  |- Shared_Expert: 0.6ms
#  |- Shared_Expert的AllReduce大概率会特别慢(1.27MB/s), 但是也有正常的时候(1GB/s)
#  |- Bincount: 0.7ms (降级为CPU Kernel)


# Decode Head Tag: aclnnEmbedding_GatherV2AiCore_GatherV2

# Layer Tags
# --------------------
# Layer 0 [Dense] 232
# Layer 1 [Dense] 551
# Layer 2 [Dense] 860
# Layer 3 [MoE] 1275
# Layer 4 [MoE] 2513
# Layer 5 [MoE] 3896
# Layer 6 [MoE] 4986
# Layer 7 [MoE] 5800
# Layer 8 [MoE] 7259
# Layer 9 [MoE] 8501
# Layer 10 [MoE] 9633
# Layer 11 [MoE] 11099
# Layer 12 [MoE] 12234
# Layer 13 [MoE] 13583
# Layer 14 [MoE] 14824
# Layer 15 [MoE] 16044
# Layer 16 [MoE] 16965
# Layer 17 [MoE] 17862
# Layer 18 [MoE] 18780
# Layer 19 [MoE] 19679
# Layer 20 [MoE] 20601
# Layer 21 [MoE] 21506
# Layer 22 [MoE] 22323
# Layer 23 [MoE] 23432
# Layer 24 [MoE] 24386
# Layer 25 [MoE] 25470
# Layer 26 [MoE] 26384
# Layer 27 [MoE] 27214
# Layer 28 [MoE] 28109
# Layer 29 [MoE] 29240
# Layer 30 [MoE] 30467
# Layer 31 [MoE] 31288
# Layer 32 [MoE] 32302
# Layer 33 [MoE] 33570
# Layer 34 [MoE] 34659
# Layer 35 [MoE] 35569
# Layer 36 [MoE] 36477
# Layer 37 [MoE] 37382
# Layer 38 [MoE] 38295
# Layer 39 [MoE] 39109
# Layer 40 [MoE] 40239
# Layer 41 [MoE] 41684
# Layer 42 [MoE] 42502
# Layer 43 [MoE] 43850
# Layer 44 [MoE] 44986
# Layer 45 [MoE] 46117
# Layer 46 [MoE] 47335
# Layer 47 [MoE] 48157
# Layer 48 [MoE] 49999
# Layer 49 [MoE] 50896
# Layer 50 [MoE] 51895
# Layer 51 [MoE] 53027
# Layer 52 [MoE] 54264
# Layer 53 [MoE] 55488
# Layer 54 [MoE] 56401
# Layer 55 [MoE] 57218
# Layer 56 [MoE] 58440
# Layer 57 [MoE] 59259
# Layer 58 [MoE] 60263
# Layer 59 [MoE] 61076
# Layer 60 [MoE] 62337
