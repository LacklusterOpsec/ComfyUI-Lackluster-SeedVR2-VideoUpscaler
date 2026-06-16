import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

if triton is not None:
    @triton.jit
    def _fused_index_select_fwd(
        x_ptr, y_ptr, idx_ptr,
        stride_x_row, stride_x_col,
        stride_y_row, stride_y_col,
        N_cols: tl.constexpr, BLOCK_SIZE_COL: tl.constexpr
    ):
        # program_id(0) is the row index in the output tensor Y
        row_idx = tl.program_id(0)
        
        # Load the source row index from the idx tensor
        src_row = tl.load(idx_ptr + row_idx)
        
        # Pointers to the start of the respective rows
        X_row_ptr = x_ptr + src_row * stride_x_row
        Y_row_ptr = y_ptr + row_idx * stride_y_row
        
        # Generate N_cols offsets
        offsets = tl.arange(0, BLOCK_SIZE_COL)
        mask = offsets < N_cols
        
        # Load the entire row from X and store into Y
        x_vals = tl.load(X_row_ptr + offsets * stride_x_col, mask=mask)
        tl.store(Y_row_ptr + offsets * stride_y_col, x_vals, mask=mask)


def fused_index_select(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Fused Triton implementation of torch.index_select(x, 0, idx).
    Assumes x is a 2D tensor of shape (M, N).
    """
    if triton is None:
        return torch.index_select(x, 0, idx)
        
    # We flatten the tensor to 2D: (batch*seq, hidden_dim)
    orig_shape = list(x.shape)
    M = orig_shape[0]
    
    if len(orig_shape) > 1:
        N = x.numel() // M
        x_2d = x.view(M, N)
    else:
        N = 1
        x_2d = x.view(M, 1)
        
    M_out = idx.numel()
    
    y_2d = torch.empty((M_out, N), device=x.device, dtype=x.dtype)
    
    MAX_FUSED_SIZE = 65536
    assert N <= MAX_FUSED_SIZE, "Hidden size too large for fused window partition"
    BLOCK_SIZE_COL = triton.next_power_of_2(N)
    
    if not x_2d.is_contiguous(): x_2d = x_2d.contiguous()
    if not idx.is_contiguous(): idx = idx.contiguous()
    
    grid = (M_out,)
    _fused_index_select_fwd[grid](
        x_2d, y_2d, idx,
        x_2d.stride(0), x_2d.stride(1),
        y_2d.stride(0), y_2d.stride(1),
        N, BLOCK_SIZE_COL=BLOCK_SIZE_COL,
        num_warps=min(max(BLOCK_SIZE_COL // 256, 1), 8)
    )
    
    out_shape = [M_out] + orig_shape[1:]
    return y_2d.view(out_shape)
