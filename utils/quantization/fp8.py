import torch

def fp8_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert block_size > 0, "block_size must be a positive integer."
    x = x.cpu().to(torch.float32)
    M, N = x.shape
    scale_m = (M + block_size - 1) // block_size
    scale_n = (N + block_size - 1) // block_size
    scale = torch.zeros((scale_m, scale_n), dtype=torch.float32)
    max_float8 = torch.finfo(torch.float8_e4m3fn).max
    for i in range(scale_m):
        row_start = i * block_size
        row_end = min(row_start + block_size, M)
        for j in range(scale_n):
            col_start = j * block_size
            col_end = min(col_start + block_size, N)
            block = x[row_start:row_end, col_start:col_end]
            max_val = torch.max(torch.abs(block))
            scale_val = max_val / max_float8 if max_val != 0 else 1.0
            scale[i, j] = scale_val
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[:M, :N]
    quantized_weight = (x / scale_expanded).to(torch.float8_e4m3fn)
    return quantized_weight, scale

def fp8_dequant(x: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    x = x.cpu()
    scale = scale.cpu()
    M, N = x.shape
    scale_m, scale_n = scale.shape
    assert scale_m == (M + block_size - 1) // block_size, "Mismatch in scale rows and weight rows."
    assert scale_n == (N + block_size - 1) // block_size, "Mismatch in scale columns and weight columns."
    x = x.to(torch.float32)
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[:M, :N]
    dequantized_weight = x * scale_expanded
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())

    return dequantized_weight
