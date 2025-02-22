import torch
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
offload_cpu = False
gemm_impl = "naive"


def set_linear_args(local_gemm_impl: str, local_fp8_quant_block_size: int, local_offload_cpu: bool):
    global pre_process, post_process
    global gemm_impl, fp8_quant_block_size, offload_cpu
    fp8_quant_block_size = local_fp8_quant_block_size
    offload_cpu = local_offload_cpu
    gemm_impl = local_gemm_impl
    if not offload_cpu or cpu_gemm_support:
        pre_process = lambda x, weight, bias: (x, weight, bias)
        post_process = lambda x: x

def pre_process(x, weight, bias):
    x = x.to(default_device)
    if hasattr(weight, "scale"):
        scale = weight.scale.to(default_device)
        weight = weight.to(default_device)
        weight.scale = scale
    else:
        weight = weight.to(default_device)
    bias = bias.to(default_device) if bias is not None else None
    return x, weight, bias

def post_process(x):
    return x.to(default_device)

def fp8_bf16_linear(x, weight, bias=None):
    weight = fp8_dequant(weight, weight.scale, fp8_quant_block_size)
    return F.linear(x, weight, bias)

def int8_bf16_linear(x, weight, bias=None):
    weight = int8_dequant(weight, weight.scale).T
    return F.linear(x, weight, bias)

def int4_bf16_linear(x, weight, bias=None):
    weight = int4_dequant(weight, weight.scale).T
    return F.linear(x, weight, bias)

def nd_quant_naive_linear(x, weight, bias=None):
    return torch_npu.npu_weight_quant_batchmatmul(x, weight, weight.scale, bias=bias)

def ncl_quant_naive_linear(x, weight, bias=None):
    batch_size, seq_len, hidden_dim = x.shape
    x_reshaped = x.view(-1, hidden_dim)
    y = torch_npu.npu_weight_quant_batchmatmul(x_reshaped, weight, weight.scale, bias=bias)
    return y.view(batch_size, seq_len, -1)

def get_linear(weight, x_shape="ND"):
    if weight.dtype == torch.bfloat16:
        # 如果权重是bf16的，则不需要进行额外的判断
        return F.linear
    if gemm_impl == "bf16":
        # 如果权重是量化的，则需要判断矩阵乘的方法
        if weight.dtype == torch.float8_e4m3fn:
            return fp8_bf16_linear
        elif weight.dtype == torch.int8:
            return int8_bf16_linear
        elif weight.dtype == torch.int32:
            if offload_cpu:
                raise NotImplementedError(f"Unsupported int4 gemm_impl on cpu")
            return int4_bf16_linear
        else:
            raise NotImplementedError(f"Unsupported dtype: {weight.dtype} for bf16 gemm_impl")
    elif gemm_impl == "naive" and not offload_cpu:
        # 如果使用量化的矩阵乘法
        if weight.dtype == torch.int8 or weight.dtype == torch.int32:
            if x_shape == "ND":
                return nd_quant_naive_linear
            elif x_shape == "NCL":
                return ncl_quant_naive_linear
            else:
                raise NotImplementedError(f"Unsupported x_shape: {x_shape} for {gemm_impl}")
    else:
        raise NotImplementedError(f"Unsupported gemm_impl: {gemm_impl} for {gemm_impl} with cpu_offload: {offload_cpu}")
