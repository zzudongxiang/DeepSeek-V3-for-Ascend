import torch
import torch_npu


def int4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.float()
    max_vals_per_dim = torch.max(torch.abs(x_fp32), dim=0).values
    scales_per_dim = max_vals_per_dim / 7.0 if torch.all(max_vals_per_dim > 0) else torch.ones_like(max_vals_per_dim)
    quantized_x = torch.clamp(torch.round(x_fp32 / scales_per_dim), min=-8, max=7)
    quantized_x = torch_npu.npu_convert_weight_to_int4pack(quantized_x.to(torch.int32))
    return quantized_x.to(torch.int32), scales_per_dim.to(torch.bfloat16)

def int4_dequant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.shape[0], x.shape[0], device=x.device, dtype=torch.bfloat16)
    x = torch_npu.npu_weight_quant_batchmatmul(ones, x, scale)
    return x.to(torch.bfloat16)
