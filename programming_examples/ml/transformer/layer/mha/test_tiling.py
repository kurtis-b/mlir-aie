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

def tiled_matmul(A, B, m, k, n):
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

def get_tile_bytes(tile_size, dtype):
    """
    Get the number of bytes for a tile of given size and data type.

    Parameters:
    tile_size (int): Size of the tile
    dtype (torch.dtype): Data type of the tensor

    Returns:
    int: Number of bytes for the tile
    """
    if dtype == torch.float32:
        return tile_size * 4  # 4 bytes for float32
    elif dtype == torch.float16:
        return tile_size * 2  # 2 bytes for float16
    elif dtype == torch.int16:
        return tile_size * 2 # 2 bytes for int16
    elif dtype == torch.int8:
        return tile_size  # 1 byte for int8
    else:
        raise ValueError("Unsupported data type")

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
    m=32
    k=32
    n=32
    
    batch_size, seq_len, d_model = Q.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    head_dim = d_model // num_heads

    # Apply input projection
    Q = tiled_matmul(Q, in_proj_weight.T, m, k, n) + in_proj_bias
    K = tiled_matmul(K, in_proj_weight.T, m, k, n) + in_proj_bias
    V = tiled_matmul(V, in_proj_weight.T, m, k, n) + in_proj_bias

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
                scores_tile = tiled_matmul(Q_h[:, i:i+tile_size], K_h[:, j:j+tile_size].transpose(-1, -2), m, k, n) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=Q.device))
                max_scores_tile = torch.maximum(max_scores_tile, torch.max(scores_tile, dim=-1).values)
            for j in range(0, seq_len, tile_size):
                # Compute partial attention scores for the current tile
                scores_tile = tiled_matmul(Q_h[:, i:i+tile_size], K_h[:, j:j+tile_size].transpose(-1, -2), m, k, n) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=Q.device))
                # print("scores_tile - max_scores_tile.unsqueeze(-1)", scores_tile - max_scores_tile.unsqueeze(-1))
                attention_weights[:, i:i+tile_size, j:j+tile_size] = torch.exp(scores_tile - max_scores_tile.unsqueeze(-1))
            # Normalize attention weights across the entire row
            print("attention_weights[:, i:i+tile_size]", attention_weights[:, i:i+tile_size])
            attention_weights[:, i:i+tile_size] /= torch.sum(attention_weights[:, i:i+tile_size], dim=-1, keepdim=True)
            print("attention_weights[:, i:i+tile_size]", attention_weights[:, i:i+tile_size])
            max_values, max_indices = torch.max(attention_weights[:, i:i+tile_size], dim=-1)
            print("Max attention_weights[:, i:i+tile_size]:", max_values)
            print("Indices of max attention_weights[:, i:i+tile_size]:", max_indices)
            # Compute weighted sum of values using tiled attention weights
            tiled_output[:, i:i+tile_size] = tiled_matmul(attention_weights[:, i:i+tile_size], V_h, m, k, n)
            print("tiled_output[:, i:i+tile_size]", tiled_output[:, i:i+tile_size])
        output_heads[:, h] = tiled_output

    # Concatenate heads
    output = output_heads.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, d_model)

    # Apply output projection
    output = tiled_matmul(output, out_proj_weight.T, m, k, n) + out_proj_bias

    return output

def multi_head_attention_tiling_with_kernel_unrolled(Q, K, V, num_heads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias):
    """
    Multi-head attention with tiling for efficient computation.

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
    m_o=16
    n_o=16
    n_s=16
    k_o=16
    k_ot=16
    k_q=16
    k_k=16
    k_v=16
    k_s=16
    total_bytes_local_mem = 0

    batch_size, seq_len, d_model = Q.shape
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    head_dim = d_model // num_heads
    # Projections below will be used to check that some of the tiling is correct
    Q_proj = torch.matmul(Q, in_proj_weight.T) + in_proj_bias
    K_proj = torch.matmul(K, in_proj_weight.T) + in_proj_bias
    V_proj = torch.matmul(V, in_proj_weight.T) + in_proj_bias

    output = torch.zeros(batch_size, seq_len, d_model, device=Q.device)
    for out_row in range(0, seq_len, m_o):
        for out_col in range(0, d_model, n_o):
            for head in range(num_heads):
                for out_reduce in range(head * head_dim, (head + 1) * head_dim, k_o):
                    # Accumulate a tile of output matrix via reduction dimension
                    out_tile = torch.zeros(batch_size, m_o, k_o, device=Q.device)
                    for out_tile_reduce in range(0, seq_len, k_ot):
                        # Get the V tile
                        v_h_tile = torch.zeros(batch_size, k_ot, k_o, device=Q.device)
                        for v_tile_reduce in range(0, d_model, k_v):
                            v_h_tile += torch.matmul(V[:, out_tile_reduce:out_tile_reduce+k_ot, v_tile_reduce:v_tile_reduce+k_v], in_proj_weight[v_tile_reduce:v_tile_reduce+k_v, out_reduce:out_reduce+k_o]) 
                        v_h_tile += in_proj_bias[out_reduce:out_reduce+k_o].unsqueeze(0)
                        assert torch.allclose(v_h_tile, V_proj[:, out_tile_reduce:out_tile_reduce+k_ot, out_reduce:out_reduce+k_o], atol=1e-5), "v_h_tile does not match the corresponding tile in V_proj"
                        # Get the max scores of each row across all columns of the attention weights. This is to prevent
                        # overflow when computing the exponential of the scores
                        max_scores_tile_vector = torch.zeros(batch_size, m_o, device=Q.device)
                        for scores_tile_col in range(0, seq_len, n_s):
                            scores_tile = torch.zeros(batch_size, m_o, k_ot, device=Q.device)
                            for scores_tile_reduce in range(0, head_dim, k_s):
                                # Get the Q tile
                                q_h_tile = torch.zeros(batch_size, m_o, k_s, device=Q.device)
                                for q_tile_reduce in range(0, d_model, k_q):
                                    q_h_tile += torch.matmul(Q[:, out_row:out_row+m_o, q_tile_reduce:q_tile_reduce+k_q], in_proj_weight[q_tile_reduce:q_tile_reduce+k_q, scores_tile_reduce:scores_tile_reduce+k_s])
                                q_h_tile += in_proj_bias[scores_tile_reduce:scores_tile_reduce+k_s].unsqueeze(0)
                                assert torch.allclose(q_h_tile, Q_proj[:, out_row:out_row+m_o, scores_tile_reduce:scores_tile_reduce+k_s], atol=1e-5), "q_h_tile does not match the corresponding tile in Q_proj"
                                # Get the K tile
                                k_h_tile = torch.zeros(batch_size, k_s, n_s, device=Q.device)
                                for k_tile_reduce in range(0, d_model, k_k):
                                    k_h_tile += torch.matmul(in_proj_weight.T[scores_tile_reduce:scores_tile_reduce+k_s, k_tile_reduce:k_tile_reduce+k_k], K[:, scores_tile_col:scores_tile_col+n_s, k_tile_reduce:k_tile_reduce+k_k].transpose(-1, -2))
                                k_h_tile += in_proj_bias[scores_tile_col:scores_tile_col+n_s].unsqueeze(0)
                                assert torch.allclose(k_h_tile, K_proj[:, scores_tile_col:scores_tile_col+n_s, scores_tile_reduce:scores_tile_reduce+k_s].transpose(-1, -2), atol=1e-5), "k_h_tile does not match the corresponding tile in K_proj"
                                scores_tile += torch.matmul(q_h_tile, k_h_tile)
                                max_scores_tile_vector = torch.maximum(max_scores_tile_vector, torch.max(scores_tile, dim=-1).values)
                        # Get the sum of each row across all columns of the attention weights
                        sums_tile_vector = torch.zeros(batch_size, m_o, device=Q.device)
                        for scores_tile_col in range(0, seq_len, n_s):
                            scores_tile = torch.zeros(batch_size, m_o, k_ot, device=Q.device)
                            for scores_tile_reduce in range(0, head_dim, k_s):
                                q_h_tile = torch.zeros(batch_size, m_o, k_s, device=Q.device)
                                for q_tile_reduce in range(0, d_model, k_q):
                                    q_h_tile += torch.matmul(Q[:, out_row:out_row+m_o, q_tile_reduce:q_tile_reduce+k_q], in_proj_weight[q_tile_reduce:q_tile_reduce+k_q, scores_tile_reduce:scores_tile_reduce+k_s])
                                q_h_tile += in_proj_bias[scores_tile_reduce:scores_tile_reduce+k_s].unsqueeze(0)
                                assert torch.allclose(q_h_tile, Q_proj[:, out_row:out_row+m_o, scores_tile_reduce:scores_tile_reduce+k_s], atol=1e-5), "q_h_tile does not match the corresponding tile in Q_proj"
                                k_h_tile = torch.zeros(batch_size, k_s, n_s, device=Q.device)
                                for k_tile_reduce in range(0, d_model, k_k):
                                    k_h_tile += torch.matmul(in_proj_weight.T[scores_tile_reduce:scores_tile_reduce+k_s, k_tile_reduce:k_tile_reduce+k_k], K[:, scores_tile_col:scores_tile_col+n_s, k_tile_reduce:k_tile_reduce+k_k].transpose(-1, -2))
                                k_h_tile += in_proj_bias[scores_tile_col:scores_tile_col+n_s].unsqueeze(0)
                                assert torch.allclose(k_h_tile, K_proj[:, scores_tile_col:scores_tile_col+n_s, scores_tile_reduce:scores_tile_reduce+k_s].transpose(-1, -2), atol=1e-5), "k_h_tile does not match the corresponding tile in K_proj"
                                scores_tile += torch.matmul(q_h_tile, k_h_tile)
                            # Subtract the max scores from the scores to prevent overflow
                            att_weights_tile = torch.exp(scores_tile - max_scores_tile_vector.unsqueeze(-1))
                            sums_tile_vector += torch.sum(att_weights_tile, dim=-1)
                        # Get the relevant scores tile for the output tile reduction
                        scores_tile = torch.zeros(batch_size, m_o, k_ot, device=Q.device)
                        for scores_tile_reduce in range(0, head_dim, k_s):
                            q_h_tile = torch.zeros(batch_size, m_o, k_s, device=Q.device)
                            for q_tile_reduce in range(0, d_model, k_q):
                                q_h_tile += torch.matmul(Q[:, out_row:out_row+m_o, q_tile_reduce:q_tile_reduce+k_q], in_proj_weight[q_tile_reduce:q_tile_reduce+k_q, scores_tile_reduce:scores_tile_reduce+k_s])
                            q_h_tile += in_proj_bias[scores_tile_reduce:scores_tile_reduce+k_s].unsqueeze(0)
                            assert torch.allclose(q_h_tile, Q_proj[:, out_row:out_row+m_o, scores_tile_reduce:scores_tile_reduce+k_s], atol=1e-5), "q_h_tile does not match the corresponding tile in Q_proj"
                            k_h_tile = torch.zeros(batch_size, k_s, n_s, device=Q.device)
                            for k_tile_reduce in range(0, d_model, k_k):
                                k_h_tile += torch.matmul(in_proj_weight.T[scores_tile_reduce:scores_tile_reduce+k_s, k_tile_reduce:k_tile_reduce+k_k], K[:, out_tile_reduce:out_tile_reduce+k_ot, k_tile_reduce:k_tile_reduce+k_k].transpose(-1, -2))
                            k_h_tile += in_proj_bias[out_tile_reduce:out_tile_reduce+k_ot].unsqueeze(0)
                            assert torch.allclose(k_h_tile, K_proj[:, out_tile_reduce:out_tile_reduce+k_ot, scores_tile_reduce:scores_tile_reduce+k_s].transpose(-1, -2), atol=1e-5), "k_h_tile does not match the corresponding tile in K_proj"
                            scores_tile += torch.matmul(q_h_tile, k_h_tile)
                        # print("scores_tile", scores_tile)
                        # print("max_scores_tile_vector", max_scores_tile_vector)
                        # print("sums_tile_vector", sums_tile_vector)
                        # print("scores_tile - max_scores_tile_vector.unsqueeze(-1)", scores_tile - max_scores_tile_vector.unsqueeze(-1))
                        # Compute the attention weights for the current tile
                        att_weights_tile = torch.exp(scores_tile - max_scores_tile_vector.unsqueeze(-1))
                        # print("att_weights_tile", att_weights_tile)
                        att_weights_tile /= sums_tile_vector.unsqueeze(-1)
                        # print("att_weights_tile", att_weights_tile)
                        # Acumulate for the out tile in one head
                        out_tile += torch.matmul(att_weights_tile, v_h_tile)
                        # print("out_tile", out_tile)
                    out_weight_tile = out_proj_weight[out_reduce:out_reduce+k_o, out_col:out_col+n_o]
                    output[:, out_row:out_row+m_o, out_col:out_col+n_o] += torch.matmul(out_tile, out_weight_tile)
                    # print(f"output[:, {out_row}:{out_row+m_o}, {out_col}:{out_col+n_o}]", output[:, out_row:out_row+m_o, out_col:out_col+n_o])
            output[:, out_row:out_row+m_o, out_col:out_col+n_o] += out_proj_bias[out_col:out_col+n_o].unsqueeze(0)
            # print(f"output[:, {out_row}:{out_row+m_o}, {out_col}:{out_col+n_o}]", output[:, out_row:out_row+m_o, out_col:out_col+n_o])
    return output

def verify_multi_head_attention():
    tiling = True
    matmul_kernel = True
    unroll = True

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Parameters
    batch_size = 1
    seq_len = 64
    d_model = 192
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
    in_proj_weight = torch.ones((d_model, d_model), dtype=torch.float32) / 100
    in_proj_bias = torch.ones((d_model,), dtype=torch.float32) / 100
    out_proj_weight = torch.ones((d_model, d_model), dtype=torch.float32) / 100
    out_proj_bias = torch.ones((d_model,), dtype=torch.float32) / 100

    # PyTorch multi-head attention
    mha = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
    # Set weights and biases to 1
    with torch.no_grad():
        mha.in_proj_weight.fill_(0.01)
        mha.in_proj_bias.fill_(0.01)
        mha.out_proj.weight.fill_(0.01)
        mha.out_proj.bias.fill_(0.01)
    torch_output, _ = mha(Q_torch, K_torch, V_torch)

    # Custom multi-head attention
    if tiling:
        if matmul_kernel:
            if unroll:
                custom_output = multi_head_attention_tiling_with_kernel_unrolled(Q_torch, K_torch, V_torch, num_heads, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias)
            else:
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
    assert np.allclose(custom_output_np, torch_output_np, atol=1e-2), "Outputs do not match!"

    print("Custom multi-head attention matches PyTorch implementation!")


if __name__ == "__main__":
    verify_multi_head_attention()
