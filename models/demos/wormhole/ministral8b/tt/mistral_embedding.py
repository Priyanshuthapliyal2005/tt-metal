# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import ttnn
from ttnn.layer import LightweightModule


class TtMistralEmbedding(LightweightModule):
    """Embedding module for Ministral-8B-Instruct-2410 model"""
    
    def __init__(self, device, args, weight_cache_path, state_dict, dtype=ttnn.bfloat16):
        super().__init__()
        self.device = device
        self.args = args
        self.dtype = dtype
        self.vocab_size = args.vocab_size
        self.hidden_size = args.dim
        
        # Create embedding cache path
        embedding_cache_path = os.path.join(weight_cache_path, "embedding")
        os.makedirs(embedding_cache_path, exist_ok=True)
        
        # Load embedding weights
        # We use row major layout for token embedding for efficiency
        self.embedding_weights = ttnn.as_tensor(
            state_dict["tok_embeddings.weight"],
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            cache_path=embedding_cache_path,
            memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    
    def forward(self, input_ids):
        """Embed input tokens
        
        Args:
            input_ids: Token ids to embed [batch_size, seq_len]
            
        Returns:
            Embedded representation [batch_size, seq_len, hidden_size]
        """
        # Use TTNN embedding lookup operation
        embeddings = ttnn.nn.functional.embedding(
            input_ids, 
            self.embedding_weights,
            dtype=self.dtype
        )
        
        return embeddings
