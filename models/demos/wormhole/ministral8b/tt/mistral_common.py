# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

# Import shared functionality from tt-transformers framework
from models.tt_transformers.tt.rope import RotarySetup
from models.tt_transformers.tt.common import (
    precompute_freqs,
    freqs_to_rotation_matrix,
    gather_cos_sin,
    get_rot_transformation_mat,
    get_prefill_rot_mat,
    sample_host,
    copy_host_to_device,
)

# Migration comments for removed functions:
# - generate_cos_sin_cache_ttnn: Use RotarySetup from models.tt_transformers.tt.rope
# - precompute_freqs: Use shared implementation from models.tt_transformers.tt.common
# - freqs_to_rotation_matrix: Use shared implementation from models.tt_transformers.tt.common
# - apply_rotary_emb: Use RotarySetup.get_rot_mats() and ttnn.experimental.rotary_embedding
# - get_rot_transformation_mat: Use shared implementation from models.tt_transformers.tt.common
# - get_prefill_rot_mat: Use shared implementation from models.tt_transformers.tt.common
# - prepare_inputs_ttnn and prepare_inputs_ttnn_prefill: Use shared input preparation utilities
# - sample: Use sample_host from models.tt_transformers.tt.common for standard sampling


def cache_attention(device, state_dict, args, rot_mat, dtype, max_seq_len):
    """Cache attention weights and rotary embeddings for faster inference"""
    # Ministral-specific attention caching logic can be implemented here if needed
    # Currently placeholder - will be implemented based on specific Ministral-8B requirements
    pass


def gather_rotary_emb(rot_emb_matrix, position_ids):
    """
    Gather the rotary embeddings for a given position_ids
    This function is kept as it may have Ministral-specific behavior
    """
    batch_size, seqlen = position_ids.shape
    emb_size, _, dhead = rot_emb_matrix.shape
    position_ids = position_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, dhead, dhead)
    rot_emb = rot_emb_matrix.gather(0, position_ids).view(batch_size, seqlen, dhead, dhead)
    return rot_emb


# Ministral-specific sampling function if different from standard implementation
def sample_ministral(logits, temperature=0.8, top_p=0.95):
    """
    Ministral-8B specific sampling logic
    
    Args:
        logits: Output logits from model [batch, seq, vocab]
        temperature: Sampling temperature (0 for greedy)
        top_p: Top-p sampling parameter
        
    Returns:
        Sampled token IDs
    """
    # For now, use the standard sampling from tt-transformers
    # This can be customized if Ministral-8B requires specific sampling behavior
    return sample_host(logits, temperature=temperature, top_p=top_p)
