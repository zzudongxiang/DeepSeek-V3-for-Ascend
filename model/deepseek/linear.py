import torch
from typing import Optional
import torch.nn.functional as F
from utils.quantization.fp8 import fp8_dequant
from utils.quantization.int4 import int4_dequant
from utils.quantization.int8 import int8_dequant

try:
    import torch_npu
    default_device = "npu"
except:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

# aarch架构的torch2.1在cpu上执行gemm时出现异常
# https://www.hiascend.com/doc_center/source/zh/Pytorch/60RC1/comref/comaq/commonqa_0021.html
cpu_gemm_support = not torch.version.__version__ == "2.1.0"
fp8_quant_block_size = 128
gemm_impl = "naive"

def set_linear_args(local_gemm_impl: str, local_fp8_quant_block_size: int):
    global gemm_impl, fp8_quant_block_size
    gemm_impl = local_gemm_impl
    fp8_quant_block_size = local_fp8_quant_block_size

def fp8_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if gemm_impl == "bf16":
        weight = fp8_dequant(weight, weight.scale, fp8_quant_block_size)
        return F.linear(x, weight, bias)
    else:
        raise NotImplementedError(f"Unsupported gemm_impl: {gemm_impl} for torch.float8_e4m3fn")

def int8_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if gemm_impl == "bf16":
        weight = int8_dequant(weight, weight.scale).T
        return F.linear(x, weight, bias)
    elif gemm_impl == "naive":
        if len(x.shape) > 2:
            batch_size, seq_len, hidden_dim = x.shape
            x_reshaped = x.view(-1, hidden_dim)
            y = torch_npu.npu_weight_quant_batchmatmul(x_reshaped, weight, weight.scale, bias=bias)
            return y.view(batch_size, seq_len, -1)
        else:
            return torch_npu.npu_weight_quant_batchmatmul(x, weight, weight.scale, bias=bias)
    else:
        raise NotImplementedError(f"Unsupported gemm_impl: {gemm_impl} for torch.int8")
    
def int4_linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if gemm_impl == "bf16":
        weight = int4_dequant(weight, weight.scale).T
        return F.linear(x, weight, bias)
    elif gemm_impl == "naive":
        # TODO
        pass
    else:
        raise NotImplementedError(f"Unsupported gemm_impl: {gemm_impl} for torch.int4")

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if (x.is_cpu or weight.is_cpu) and not cpu_gemm_support:
        x = x.to(default_device)
        if hasattr(weight, "scale"):
            scale = weight.scale.to(default_device)
            weight = weight.to(default_device)
            weight.scale = scale
        else:
            weight = weight.to(default_device)
        bias = bias.to(default_device) if bias is not None else None
    if weight.dtype == torch.bfloat16:
        y = F.linear(x, weight, bias)
    elif weight.dtype == torch.float8_e4m3fn:
        y = fp8_linear(x, weight, bias)
    elif weight.dtype == torch.int8:
        y = int8_linear(x, weight, bias)
    elif weight.dtype == torch.int32:
        y = int4_linear(x, weight, bias)
    else:
        raise NotImplementedError(f"Unsupported dtype: {weight.dtype} for linear")
    return y.to(default_device)