import numpy as np
import torch
import torch.nn.functional as F

def multi_head_attention_no_tiling(Q, K, V, num_heads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):
    """
    Multi-head attention without tiling for efficient computation.

    Parameters:
    Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model)
    K (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model)
    V (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model)
    num_heads (int): Number of attention heads
    in_proj_weight (torch.Tensor): Input projection weight
    in_proj_bias (torch.Tensor): Input projection bias
    out_proj_weight (torch.Tensor): Output projection weight
    out_proj_bias (torch.Tensor): Output projection bias

    Returns:
    torch.Tensor: Output of the multi-head attention mechanism
    """
    batch_size, seq_len, d_model = Q.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    head_dim = d_model // num_heads

    # Apply input projection
    Q = torch.matmul(Q, in_proj_weight.T) + in_proj_bias
    K = torch.matmul(K, in_proj_weight.T) + in_proj_bias
    V = torch.matmul(V, in_proj_weight.T) + in_proj_bias

    # Split Q, K, V into multiple heads
    Q_heads = Q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    K_heads = K.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    V_heads = V.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

    # Initialize output
    output_heads = torch.zeros_like(Q_heads)

    # Compute attention for each head
    for h in range(num_heads):
        Q_h = Q_heads[:, h]
        K_h = K_heads[:, h]
        V_h = V_heads[:, h]
        
        # Compute attention scores
        scores = torch.matmul(Q_h, K_h.transpose(-1, -2)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=Q.device))
        attention_weights = F.softmax(scores, dim=-1)

        # Compute weighted sum of values
        output_heads[:, h] = torch.matmul(attention_weights, V_h)

    # Concatenate heads
    output = output_heads.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, d_model)

    # Apply output projection
    output = torch.matmul(output, out_proj_weight.T) + out_proj_bias

    return output


def multi_head_attention_tiling_without_kernel(Q, K, V, num_heads, tile_size, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):
    """
    Multi-head attention with tiling for efficient computation.

    Parameters:
    Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model)
    K (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model)
    V (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model)
    num_heads (int): Number of attention heads
    tile_size (int): Tile size for computation
    in_proj_weight (torch.Tensor): Input projection weight
    in_proj_bias (torch.Tensor): Input projection bias
    out_proj_weight (torch.Tensor): Output projection weight
    out_proj_bias (torch.Tensor): Output projection bias

    Returns:
    torch.Tensor: Output of the multi-head attention mechanism
    """
    batch_size, seq_len, d_model = Q.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    head_dim = d_model // num_heads

    # Apply input projection
    Q = torch.matmul(Q, in_proj_weight.T) + in_proj_bias
    K = torch.matmul(K, in_proj_weight.T) + in_proj_bias
    V = torch.matmul(V, in_proj_weight.T) + in_proj_bias

    # Split Q, K, V into multiple heads
    Q_heads = Q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    K_heads = K.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    V_heads = V.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

    # Initialize output
    output_heads = torch.zeros_like(Q_heads)

    # Compute attention for each head
    for h in range(num_heads):
        Q_h = Q_heads[:, h]
        K_h = K_heads[:, h]
        V_h = V_heads[:, h]

        # Initialize tiled attention weights and output
        attention_weights = torch.zeros(batch_size, seq_len, seq_len, device=Q.device)
        tiled_output = torch.zeros(batch_size, seq_len, head_dim, device=Q.device)

        # Tiling over the sequence length for attention weights calculation
        for i in range(0, seq_len, tile_size):
            max_scores_tile = torch.zeros(batch_size, tile_size, device=Q.device)
            for j in range(0, seq_len, tile_size):
                # Compute partial attention scores for the current tile
                scores_tile = torch.matmul(Q_h[:, i:i+tile_size], K_h[:, j:j+tile_size].transpose(-1, -2)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=Q.device))
                max_scores_tile = torch.maximum(max_scores_tile, torch.max(scores_tile, dim=-1).values)
                if j == 0 and i == 0 and h == 0:
                    print(scores_tile.shape, max_scores_tile.shape)
                    print("scores_tile", scores_tile)
                    print("max_scores_tile", max_scores_tile)
            for j in range(0, seq_len, tile_size):
                # Compute partial attention scores for the current tile
                scores_tile = torch.matmul(Q_h[:, i:i+tile_size], K_h[:, j:j+tile_size].transpose(-1, -2)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=Q.device))
                attention_weights[:, i:i+tile_size, j:j+tile_size] = torch.exp(scores_tile - max_scores_tile.unsqueeze(-1))
                # TODO: Add running sum for softmax normalization
                if j == 0 and i == 0 and h == 0:
                    print("scores_tile - max_scores_tile.unsqueeze(-1)", scores_tile - max_scores_tile.unsqueeze(-1))
                    print("attention_weights[:, i:i+tile_size, j:j+tile_size]", attention_weights[:, i:i+tile_size, j:j+tile_size])
            # Normalize attention weights across the entire row
            attention_weights[:, i:i+tile_size] /= torch.sum(attention_weights[:, i:i+tile_size], dim=-1, keepdim=True)
            # Compute weighted sum of values using tiled attention weights
            tiled_output[:, i:i+tile_size] = torch.matmul(attention_weights[:, i:i+tile_size], V_h)
            if i == 0 and h == 0:
                print("tiled_output[:, i:i+tile_size]", tiled_output[:, i:i+tile_size])
        output_heads[:, h] = tiled_output

    # Concatenate heads
    output = output_heads.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, d_model)

    # Apply output projection
    output = torch.matmul(output, out_proj_weight.T) + out_proj_bias

    return output

def multi_head_attention_tiling_with_kernel(Q, K, V, num_heads, tile_size, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):
    """
    Multi-head attention with tiling for efficient computation.

    Parameters:
    Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model)
    K (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model)
    V (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model)
    num_heads (int): Number of attention heads
    tile_size (int): Tile size for computation
    in_proj_weight (torch.Tensor): Input projection weight
    in_proj_bias (torch.Tensor): Input projection bias
    out_proj_weight (torch.Tensor): Output projection weight
    out_proj_bias (torch.Tensor): Output projection bias

    Returns:
    torch.Tensor: Output of the multi-head attention mechanism
    """
    def tiled_matmul(A, B, m=32, k=32, n=32):
        """
        Perform tiled matrix multiplication with tiles of size m x k x n.

        Parameters:
        A (torch.Tensor): Left matrix of shape (batch_size, seq_len, k)
        B (torch.Tensor): Right matrix of shape (batch_size, k, seq_len)
        m (int): Tile size for rows of A
        k (int): Tile size for shared dimension
        n (int): Tile size for columns of B

        Returns:
        torch.Tensor: Result of tiled matrix multiplication
        """
        batch_size, a_rows, a_cols = A.shape
        if B.dim() == 2:
            B = B.unsqueeze(0)  # Add a batch dimension if B is 2D
        _, b_rows, b_cols = B.shape
        assert a_cols == b_rows, "Matrix dimensions do not match for multiplication"
        result = torch.zeros(batch_size, a_rows, b_cols, device=A.device)

        for i in range(0, a_rows, m):
            for j in range(0, b_cols, n):
                for p in range(0, a_cols, k):
                    A_tile = A[:, i:i+m, p:p+k]
                    B_tile = B[:, p:p+k, j:j+n]
                    result[:, i:i+m, j:j+n] += torch.matmul(A_tile, B_tile)

        return result

    batch_size, seq_len, d_model = Q.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    head_dim = d_model // num_heads

    # Apply input projection
    Q = tiled_matmul(Q, in_proj_weight.T) + in_proj_bias
    K = tiled_matmul(K, in_proj_weight.T) + in_proj_bias
    V = tiled_matmul(V, in_proj_weight.T) + in_proj_bias

    # Split Q, K, V into multiple heads
    Q_heads = Q.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    K_heads = K.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
    V_heads = V.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

    # Initialize output
    output_heads = torch.zeros_like(Q_heads)

    # Compute attention for each head
    for h in range(num_heads):
        Q_h = Q_heads[:, h]
        K_h = K_heads[:, h]
        V_h = V_heads[:, h]

        # Initialize tiled attention weights and output
        attention_weights = torch.zeros(batch_size, seq_len, seq_len, device=Q.device)
        tiled_output = torch.zeros(batch_size, seq_len, head_dim, device=Q.device)

        # Tiling over the sequence length for attention weights calculation
        for i in range(0, seq_len, tile_size):
            max_scores_tile = torch.zeros(batch_size, tile_size, device=Q.device)
            for j in range(0, seq_len, tile_size):
                # Compute partial attention scores for the current tile
                scores_tile = tiled_matmul(Q_h[:, i:i+tile_size], K_h[:, j:j+tile_size].transpose(-1, -2)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=Q.device))
                max_scores_tile = torch.maximum(max_scores_tile, torch.max(scores_tile, dim=-1).values)
            for j in range(0, seq_len, tile_size):
                # Compute partial attention scores for the current tile
                scores_tile = tiled_matmul(Q_h[:, i:i+tile_size], K_h[:, j:j+tile_size].transpose(-1, -2)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=Q.device))
                attention_weights[:, i:i+tile_size, j:j+tile_size] = torch.exp(scores_tile - max_scores_tile.unsqueeze(-1))
            # Normalize attention weights across the entire row
            attention_weights[:, i:i+tile_size] /= torch.sum(attention_weights[:, i:i+tile_size], dim=-1, keepdim=True)
            # Compute weighted sum of values using tiled attention weights
            tiled_output[:, i:i+tile_size] = tiled_matmul(attention_weights[:, i:i+tile_size], V_h)
        output_heads[:, h] = tiled_output

    # Concatenate heads
    output = output_heads.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, d_model)

    # Apply output projection
    output = tiled_matmul(output, out_proj_weight.T) + out_proj_bias

    return output

def verify_multi_head_attention():
    tiling = True
    matmul_kernel = True

    # Parameters
    batch_size = 1
    seq_len = 256
    d_model = 768
    num_heads = 12
    tile_size = 64

    # Random input tensors
    Q = np.random.rand(batch_size, seq_len, d_model)
    K = np.random.rand(batch_size, seq_len, d_model)
    V = np.random.rand(batch_size, seq_len, d_model)

    # Convert to PyTorch tensors
    Q_torch = torch.tensor(Q, dtype=torch.float32)
    K_torch = torch.tensor(K, dtype=torch.float32)
    V_torch = torch.tensor(V, dtype=torch.float32)

    # Initialize weights and biases with 1's
    in_proj_weight = torch.ones((d_model, d_model), dtype=torch.float32)
    in_proj_bias = torch.ones((d_model,), dtype=torch.float32)
    out_proj_weight = torch.ones((d_model, d_model), dtype=torch.float32)
    out_proj_bias = torch.ones((d_model,), dtype=torch.float32)

    # PyTorch multi-head attention
    mha = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
    # Set weights and biases to 1
    with torch.no_grad():
        mha.in_proj_weight.fill_(1.0)
        mha.in_proj_bias.fill_(1.0)
        mha.out_proj.weight.fill_(1.0)
        mha.out_proj.bias.fill_(1.0)
    torch_output, _ = mha(Q_torch, K_torch, V_torch)

    # Custom multi-head attention
    if tiling:
        if matmul_kernel:
            custom_output = multi_head_attention_tiling_with_kernel(Q_torch, K_torch, V_torch, num_heads, tile_size, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias)
        else:
            custom_output = multi_head_attention_tiling_without_kernel(Q_torch, K_torch, V_torch, num_heads, tile_size, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias)
    else:
        custom_output = multi_head_attention_no_tiling(Q_torch, K_torch, V_torch, num_heads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias)

    # Compare outputs
    torch_output_np = torch_output.detach().numpy()
    custom_output_np = custom_output.detach().numpy()
    print(custom_output_np.shape, torch_output_np.shape)
    print("custom_output_np", custom_output_np)
    print("torch_output_np", torch_output_np)
    assert np.allclose(custom_output_np, torch_output_np, atol=1e-5), "Outputs do not match!"

    print("Custom multi-head attention matches PyTorch implementation!")


if __name__ == "__main__":
    verify_multi_head_attention()
