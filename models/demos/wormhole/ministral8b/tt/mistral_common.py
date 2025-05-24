# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def generate_cos_sin_cache_ttnn(
    tt_devices,
    head_dim,
    max_position_embeddings=2048,
    base=10000,
    dtype=None,
):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    t = torch.arange(
        max_position_embeddings,
        device=inv_freq.device,
        dtype=inv_freq.dtype,
    )
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    emb_cos = emb.cos()[None, None, :, :]
    emb_sin = emb.sin()[None, None, :, :]

    tt_cos_cached = [
        ttnn.from_torch(
            emb_cos,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        for tt_device in tt_devices
    ]

    tt_sin_cached = [
        ttnn.from_torch(
            emb_sin,
            device=tt_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
        )
        for tt_device in tt_devices
    ]

    return tt_cos_cached, tt_sin_cached


def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def freqs_to_rotation_matrix(cos_freqs, sin_freqs):
    """
    Transform cos/sin frequencies to a rotation matrix.
    """
    emb_size, emb_dim = cos_freqs.shape
    dhead = emb_dim * 2
    rot_emb_matrix = torch.zeros(emb_size, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()

    rot_emb_matrix = rot_emb_matrix.transpose(-1, -2)  # Necessary for correct rotation when applied as (x @ R)
    return rot_emb_matrix


def gather_rotary_emb(rot_emb_matrix, position_ids):
    """
    Gather the rotary embeddings for a given position_ids
    """
    batch_size, seqlen = position_ids.shape
    emb_size, _, dhead = rot_emb_matrix.shape
    position_ids = position_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, dhead, dhead)
    rot_emb = rot_emb_matrix.gather(0, position_ids).view(batch_size, seqlen, dhead, dhead)
    return rot_emb


def apply_rotary_emb(x, rot_emb_matrix, transformation_matrix=None):
    """
    Apply rotary embeddings to input tensor using rotation matrix

    Args:
        x: Input tensor [batch, seq, heads, head_dim]
        rot_emb_matrix: Rotary embedding matrices
        transformation_matrix: Optional transformation matrix

    Returns:
        Rotated tensor with same shape as input
    """
    orig_shape = x.shape
    x = ttnn.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
    
    if transformation_matrix is not None:
        # Apply transformation matrix first for RoPE
        x = ttnn.matmul(transformation_matrix, x)
        
    # Apply rotary embeddings using matrix multiplication
    x_out = ttnn.matmul(x, rot_emb_matrix[0])
    
    # Reshape back to original shape
    x_out = ttnn.reshape(x_out, orig_shape)
    
    return x_out


def get_rot_transformation_mat(head_dim):
    """
    Get transformation matrix for rotary embeddings

    Args:
        head_dim: Dimension of each attention head

    Returns:
        Transformation matrix for RoPE
    """
    # Create transformation matrix for RoPE
    arange = torch.arange(head_dim) // 2 * 2
    indices1 = torch.arange(head_dim)
    indices2 = torch.zeros(head_dim)
    indices2[1::2] = indices1[::2]
    indices2[::2] = indices1[1::2]
    
    all_indices = torch.stack([indices1, indices2], dim=0)
    transition_indices = all_indices.T.reshape(-1).long()
    
    # Create permutation matrix
    transform_mat = torch.zeros(head_dim, head_dim)
    transform_mat[indices1.long(), transition_indices] = 1
    
    # Reshape for batch processing
    transform_mat = transform_mat.unsqueeze(0)
    
    return transform_mat


def get_prefill_rot_mat(head_dim, max_seq_len, device, seq_len=128, base=10000.0):
    """
    Get rotary embeddings for prefill mode

    Args:
        head_dim: Hidden dimension of attention heads
        max_seq_len: Maximum sequence length
        device: Device to place tensors on
        seq_len: Current sequence length
        base: Base for frequency computation

    Returns:
        Rotary embedding matrices
    """
    # Precompute frequencies
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    
    # Compute positions
    t = torch.arange(seq_len)
    
    # Compute frequency matrix
    freqs = torch.outer(t, freqs).float()
    
    # Compute sin and cos values
    cos = torch.cos(freqs)
    sin = -torch.sin(freqs)
    
    # Format for matrix application
    cos = cos.unsqueeze(0)  # [1, seq, head_dim/2]
    sin = sin.unsqueeze(0)  # [1, seq, head_dim/2]
    
    # Move to device and convert to TTNN tensors
    rot_mats = []
    
    # Convert to TTNN format
    cos_ttnn = ttnn.from_torch(cos, device=device, dtype=ttnn.bfloat16)
    sin_ttnn = ttnn.from_torch(sin, device=device, dtype=ttnn.bfloat16)
    rot_mats.append([cos_ttnn, sin_ttnn])
    
    return rot_mats


def cache_attention(device, state_dict, args, rot_mat, dtype, max_seq_len):
    """Cache attention weights and rotary embeddings for faster inference"""
    pass  # Placeholder - this function will be implemented as needed for the model


def prepare_inputs_ttnn(pt_encoded_input, current_pos, hidden_dim, window_size, device):
    """
    Prepare inputs for TTNN in decode mode
    
    Args:
        pt_encoded_input: PyTorch encoded input tensor
        current_pos: Current position in sequence
        hidden_dim: Hidden dimension size
        window_size: Attention window size
        device: Device to place tensors on
        
    Returns:
        TTNN input tensor and current position
    """
    import torch
    
    # Extract the batch size and sequence length
    batch_size = pt_encoded_input.shape[0]
    
    # Move to TTNN format
    decode_input = ttnn.from_torch(
        pt_encoded_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16
    )
    
    return decode_input, current_pos


def prepare_inputs_ttnn_prefill(pt_encoded_input, device):
    """
    Prepare inputs for TTNN in prefill mode
    
    Args:
        pt_encoded_input: PyTorch encoded input tensor
        device: Device to place tensors on
        
    Returns:
        TTNN input tensor and attention mask
    """
    import torch
    
    # Create causal attention mask (lower triangular)
    seq_len = pt_encoded_input.shape[0]
    mask = torch.triu(torch.ones(seq_len, seq_len) * -1e9, diagonal=1)
    
    # Convert inputs to TTNN format
    prefill_input = ttnn.from_torch(
        pt_encoded_input,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16
    )
    
    # Convert mask to TTNN format
    attn_mask = ttnn.from_torch(
        mask.unsqueeze(0).unsqueeze(0),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16
    )
    
    return prefill_input, attn_mask, seq_len


def sample(logits, temperature=0.8, top_p=0.95):
    """
    Sample from logits with temperature and top-p sampling
    
    Args:
        logits: Output logits from model [batch, seq, vocab]
        temperature: Sampling temperature (0 for greedy)
        top_p: Top-p sampling parameter
        
    Returns:
        Sampled token IDs
    """
    import torch
    import torch.nn.functional as F
    
    batch_size = logits.size(0)
    
    if temperature == 0:
        # Greedy decoding - take argmax
        return torch.argmax(logits, dim=-1)
    
    # Apply temperature scaling
    logits = logits / temperature
    
    # Get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Apply top-p sampling
    if top_p < 1.0:
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Filter tokens below threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create a scatter mask to zero out removed tokens
        indices = torch.arange(sorted_probs.size(-1))
        indices = indices.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        scatter_mask = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        
        # Zero out all probs below threshold
        probs = probs.masked_fill(scatter_mask, 0.0)
        
        # Renormalize probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Sample from the distribution
    next_token = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
    
    # Reshape to match expected output
    next_token = next_token.view(batch_size, -1)
    
    return next_token
