import torch

def int8_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp32 = x.float()
    max_vals_per_dim = torch.max(torch.abs(x_fp32), dim=0).values
    scales_per_dim = max_vals_per_dim / 127.0 if torch.all(max_vals_per_dim > 0) else torch.ones_like(max_vals_per_dim)
    quantized_x = torch.clamp(torch.round(x_fp32 / scales_per_dim), min=-128, max=127)
    return quantized_x.to(torch.int8), scales_per_dim.to(torch.bfloat16)

def int8_dequant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16) * scale
