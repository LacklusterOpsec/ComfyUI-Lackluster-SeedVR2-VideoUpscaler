import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if triton is not None:
    @triton.jit
    def _fused_adaln_fwd(
        X_ptr, Y_ptr, 
        ScaleA_ptr, ScaleB_ptr, 
        ShiftA_ptr, ShiftB_ptr,
        stride_x_row, stride_x_col,
        stride_y_row, stride_y_col,
        stride_scaleA_row, stride_scaleA_col,
        stride_shiftA_row, stride_shiftA_col,
        N, eps,
        BLOCK_N: tl.constexpr
    ):
        # Map the program id to the row of X it should compute.
        row_idx = tl.program_id(0)
        
        # Pointers to the start of the row
        X_row_ptr = X_ptr + row_idx * stride_x_row
        Y_row_ptr = Y_ptr + row_idx * stride_y_row
        ScaleA_row_ptr = ScaleA_ptr + row_idx * stride_scaleA_row
        ShiftA_row_ptr = ShiftA_ptr + row_idx * stride_shiftA_row
        
        # Generate N offsets
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < N
        
        # Load X
        x = tl.load(X_row_ptr + offsets * stride_x_col, mask=mask, other=0.0).to(tl.float32)
        
        # Compute mean
        mean = tl.sum(x, axis=0) / N
        
        # Compute variance
        x_centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(x_centered * x_centered, axis=0) / N
        rstd = tl.math.rsqrt(var + eps)
        
        # Normalize
        x_hat = x_centered * rstd
        
        # Load scales and shifts
        scaleA = tl.load(ScaleA_row_ptr + offsets * stride_scaleA_col, mask=mask, other=0.0).to(tl.float32)
        shiftA = tl.load(ShiftA_row_ptr + offsets * stride_shiftA_col, mask=mask, other=0.0).to(tl.float32)
        
        if ScaleB_ptr is not None:
            scaleB = tl.load(ScaleB_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            scale = scaleA + scaleB
        else:
            scale = scaleA
            
        if ShiftB_ptr is not None:
            shiftB = tl.load(ShiftB_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            shift = shiftA + shiftB
        else:
            shift = shiftA
            
        # Apply AdaLN
        y = x_hat * scale + shift
        
        # Write back
        tl.store(Y_row_ptr + offsets * stride_y_col, y.to(X_ptr.dtype.element_ty), mask=mask)

def fused_adaln_forward(x: torch.Tensor, scaleA: torch.Tensor, shiftA: torch.Tensor, 
                        scaleB: torch.Tensor = None, shiftB: torch.Tensor = None, eps: float = 1e-5):
    """
    Fused Adaptive Layer Norm forward pass using Triton.
    Computes: y = layer_norm(x) * (scaleA + scaleB) + (shiftA + shiftB)
    """
    if triton is None:
        # Fallback to PyTorch eager if Triton is not available
        import torch.nn.functional as F
        x_norm = F.layer_norm(x, (x.size(-1),), eps=eps)
        scale = scaleA + scaleB if scaleB is not None else scaleA
        shift = shiftA + shiftB if shiftB is not None else shiftA
        return x_norm * scale + shift
        
    # We flatten the batch and sequence dimensions
    x_shape = x.shape
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    
    x_2d = x.view(M, N)
    # scaleA/shiftA might be broadcastable or have shape (M, N)
    # If they are (M, 1, N), we can view them as (M, N)
    scaleA_2d = scaleA.expand_as(x).contiguous().view(M, N)
    shiftA_2d = shiftA.expand_as(x).contiguous().view(M, N)
    
    y_2d = torch.empty_like(x_2d)
    
    # We assume N is less than or equal to 8192 for the block size
    # Find the next power of 2
    MAX_FUSED_SIZE = 65536
    assert N <= MAX_FUSED_SIZE, "Hidden size too large for fused AdaLN"
    BLOCK_N = triton.next_power_of_2(N)
    
    # Make sure inputs are contiguous where needed
    if not x_2d.is_contiguous(): x_2d = x_2d.contiguous()
    
    def get_ptr(tensor):
        return tensor if tensor is not None else None

    # Launch kernel
    grid = (M,)
    _fused_adaln_fwd[grid](
        x_2d, y_2d,
        scaleA_2d, get_ptr(scaleB),
        shiftA_2d, get_ptr(shiftB),
        x_2d.stride(0), x_2d.stride(1),
        y_2d.stride(0), y_2d.stride(1),
        scaleA_2d.stride(0), scaleA_2d.stride(1),
        shiftA_2d.stride(0), shiftA_2d.stride(1),
        N, eps,
        BLOCK_N=BLOCK_N,
        num_warps=min(max(BLOCK_N // 256, 1), 8)
    )
    
    return y_2d.view(x_shape)
