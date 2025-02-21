import functools
from typing import Literal
from dataclasses import dataclass
from typing_extensions import get_origin
from typing import get_type_hints, Literal, get_args

def auto_convert_types(func):
    @functools.wraps(func)
    def wrapper(obj, name, value):
        hints = get_type_hints(obj.__class__)
        if name in hints:
            target_type = hints[name]
            origin = get_origin(target_type)
            if origin is Literal:
                literal_values = get_args(target_type)
                if value in literal_values:
                    return func(obj, name, value)
                else:
                    raise ValueError(f"Value '{value}' must be one of {literal_values}")
            try:
                converted_value = target_type(value)
                return func(obj, name, converted_value)
            except (ValueError, TypeError):
                raise TypeError(f"Cannot convert {value} to {target_type}")
        return func(obj, name, value)
    return wrapper


@dataclass
class ModelArgs:
    max_batch_size: int = 128
    max_seq_len: int = 2048
    dtype: Literal["bf16", "fp8", "int8", "int4"] = "bf16"
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
    # others
    gemm_impl: Literal["naive", "bf16"] = "bf16"
    attn_impl: Literal["naive", "absorb"] = "absorb"
    use_random_weights: bool = False
    max_new_tokens: int = 100
    temperature: float = 0.2
    fp8_quant_block_size: int = 128
    offload_cpu: bool = False

    @auto_convert_types
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
