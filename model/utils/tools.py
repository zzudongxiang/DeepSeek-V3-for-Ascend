import torch

def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

def sample_cpu(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    sample_result = torch.empty(list(probs.shape), dtype=probs.dtype, device="cpu").exponential_(1)
    sample_result = sample_result.to(device=logits.device)
    return probs.div_(sample_result).argmax(dim=-1)

def weight_dequant_cpu(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor, efficiently handling cases where
    `weight` is not a multiple of `block_size` by broadcasting `scale`.

    Args:
        weight (torch.Tensor): The quantized weight tensor of shape(M, N).
        scale (torch.Tensor): The scale tensor of shape (M // block_size, N // block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.

    Raises:
        AssertionError: If `scale` dimensions do not align with `weight` shape after scaling.
    """
    weight = weight.cpu()
    scale = scale.cpu()

    #Get the original dimensions of weight
    M, N = weight.shape

    # Compute the effective block dimensions for scale
    scale_m, scale_n = scale.shape
    assert scale_m == (M + block_size - 1) // block_size, "Mismatch in scale rows and weight rows."
    assert scale_n == (N + block_size - 1) // block_size, "Mismatch in scale columns and weight columns."

    # Convert weight to float32 for calculations
    weight = weight.to(torch.float32)

    # Expand scale to match the weight tensor's shape
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:M, :N]

    # Perform element-wise multiplication
    dequantized_weight = weight * scale_expanded

    # Convert the output to the default dtype
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())

    return dequantized_weight

def weight_quant_int8_cpu(weight: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the given weight tensor using block-wise scaling, handling cases where
    `weight` is not a multiple of `block_size` by properly aligning scale dimensions.

    Args:
        weight (torch.Tensor): The float weight tensor of shape (M, N) to be quantized.
        block_size (int, optional): The block size to use for quantization. Defaults to 128.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - Quantized int8 weight tensor of same shape as input
            - Scale tensor of shape (M // block_size, N // block_size)
    
    Raises:
        AssertionError: If weight dimensions are smaller than block_size
    """
    weight = weight.cpu().float()
    M, N = weight.shape
    assert M >= block_size and N >= block_size, "Weight dimensions must be >= block_size"

    # 计算分块后的scale矩阵维度
    scale_m = (M + block_size - 1) // block_size
    scale_n = (N + block_size - 1) // block_size

    # 分块并计算每个块的绝对值最大值
    blocks = weight.abs().unfold(0, block_size, block_size).unfold(1, block_size, block_size)
    blocks = blocks.contiguous().view(scale_m, scale_n, -1)
    max_val, _ = blocks.max(dim=-1)

    # 计算scale并处理零值
    scale = max_val / 127.0
    scale = torch.where(max_val == 0, torch.ones_like(scale), scale)

    # 扩展scale矩阵以匹配原始权重形状
    scale_expanded = scale.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)[:M, :N]

    # 执行量化并转换类型
    quantized = torch.round(weight / scale_expanded).to(torch.int8)

    return quantized, scale

def weight_quant_fp8_cpu(weight: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the given weight tensor into blocks of size block_size x block_size, computing scales for each block.
    The quantized weight is stored in torch.float8_e4m3fn format, and the scale tensor is returned for dequantization.

    Args:
        weight (torch.Tensor): The input weight tensor of shape (M, N).
        block_size (int, optional): The block size for quantization. Defaults to 128.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The quantized weight tensor in torch.float8_e4m3fn format and the scale tensor.

    Raises:
        AssertionError: If block_size is not a positive integer.
    """
    assert block_size > 0, "block_size must be a positive integer."
    
    weight = weight.cpu().to(torch.float32)  # Ensure computation in float32
    M, N = weight.shape

    # Calculate scale dimensions with ceiling division
    scale_m = (M + block_size - 1) // block_size
    scale_n = (N + block_size - 1) // block_size
    scale = torch.zeros((scale_m, scale_n), dtype=torch.float32)

    # Get dynamic range of float8_e4m3fn format
    max_float8 = torch.finfo(torch.float8_e4m3fn).max

    # Calculate scale for each block
    for i in range(scale_m):
        row_start = i * block_size
        row_end = min(row_start + block_size, M)
        for j in range(scale_n):
            col_start = j * block_size
            col_end = min(col_start + block_size, N)
            
            block = weight[row_start:row_end, col_start:col_end]
            max_val = torch.max(torch.abs(block))
            
            # Handle zero-initialized blocks
            scale_val = max_val / max_float8 if max_val != 0 else 1.0
            scale[i, j] = scale_val

    # Expand scale tensor to match weight dimensions
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    scale_expanded = scale_expanded[:M, :N]  # Trim to original dimensions

    # Quantize and cast to float8_e4m3fn
    quantized_weight = (weight / scale_expanded).to(torch.float8_e4m3fn)

    return quantized_weight, scale
