# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
TT-metal implementation of Ministral-8B-Instruct-2410
"""
import os
import torch
import ttnn
from ttnn.layer import LightweightModule
from loguru import logger

class TtTransformer(LightweightModule):
    """Ministral-8B transformer model adapted for Tenstorrent hardware"""
    
    def __init__(self, args, device, dtype, state_dict=None, weight_cache_path=None, layers=None, rot_mat=None, start_pos=0):
        super().__init__()
        self.args = args
        self.device = device
        self.dtype = dtype
        self.layers = []
        
        # Initialize caching parameters
        self.current_pos = start_pos
        self.rot_mat = rot_mat  # Save rotation matrices for position embeddings
        
        # Set up model configuration parameters
        # These are specific to Ministral-8B-Instruct-2410
        self.dim = args.dim  # Hidden dimension size
        self.n_heads = args.n_heads  # Number of attention heads
        self.n_kv_heads = args.n_kv_heads  # Number of key/value heads (grouped-query attention)
        self.head_dim = args.head_dim  # Dimension per attention head
        self.n_layers = args.n_layers  # Number of transformer layers
        self.vocab_size = args.vocab_size  # Vocabulary size
        self.sliding_window = args.sliding_window  # Size of sliding window for attention
        
        # Load weights if provided
        if state_dict is not None and weight_cache_path is not None:
            logger.info(f"Loading weights for Ministral-8B-Instruct-2410 model...")
            os.makedirs(weight_cache_path, exist_ok=True)
            self._load_weights(state_dict, weight_cache_path, layers)
            logger.info(f"Weights loaded successfully!")
        else:
            logger.warning("No weights provided. Model will use random initialization.")
            # Initialize empty model components
            self.embedding = None
            self.norm = None
            self.lm_head = None
    
    def _load_weights(self, state_dict, weight_cache_path, layers):
        """Load model weights from state dictionary
        
        Args:
            state_dict: Model state dictionary from PyTorch checkpoint
            weight_cache_path: Path to cache converted weights
            layers: List of layer indices to load (None for all)
        """
        # Initialize component weights
        # 1. Initialize embedding
        self.embedding = self._create_embedding_layer(state_dict, weight_cache_path)
        
        # 2. Initialize transformer layers (attention + MLP)
        if layers is None:
            layers = list(range(self.args.n_layers))
            
        self.layers = self._create_transformer_layers(state_dict, weight_cache_path, layers)
        
        # 3. Initialize final layer norm
        self.norm = self._create_layernorm(
            state_dict["norm.weight"],
            weight_cache_path,
            name="final_layernorm"
        )
        
        # 4. Initialize LM head
        self.lm_head = self._create_lm_head(state_dict, weight_cache_path)
        
        logger.info(f"Loaded {len(layers)} transformer layers")
    
    def _create_embedding_layer(self, state_dict, weight_cache_path):
        """Create token embedding layer
        
        Args:
            state_dict: Model state dictionary
            weight_cache_path: Path to cache converted weights
            
        Returns:
            Token embedding module
        """
        from ttnn.module import Embedding
        
        # Get embedding weights from state dict
        embedding_weights = state_dict["tok_embeddings.weight"]
        
        # Create cached weight path
        cache_path = os.path.join(weight_cache_path, "embedding")
        os.makedirs(cache_path, exist_ok=True)
        
        # Create embedding module
        embedding_module = Embedding(
            num_embeddings=self.args.vocab_size,
            embedding_dim=self.args.dim,
            dtype=self.dtype,
            device=self.device,
            weight=embedding_weights,
            weight_cache_path=cache_path,
        )
        
        return embedding_module
    
    def _create_transformer_layers(self, state_dict, weight_cache_path, layers):
        """Create transformer layers (attention + MLP)"""
        # Create a list of transformer layers
        transformer_layers = []
        
        for i in layers:
            # Extract layer-specific weights
            layer_prefix = f"layers.{i}."
            layer_state_dict = {
                k[len(layer_prefix):]: v 
                for k, v in state_dict.items() 
                if k.startswith(layer_prefix)
            }
            
            # Create TransformerLayer instance
            layer = self._create_transformer_layer(
                layer_state_dict,
                weight_cache_path,
                layer_idx=i
            )
            transformer_layers.append(layer)
        
        return transformer_layers
    
    def _create_transformer_layer(self, layer_state_dict, weight_cache_path, layer_idx):
        """Create individual transformer layer
        
        Args:
            layer_state_dict: State dict for this layer
            weight_cache_path: Path to cache weights
            layer_idx: Layer index
            
        Returns:
            Transformer layer module
        """
        from ttnn.layer import Linear, LightweightModule
        
        # Create layer cache path
        layer_cache_path = os.path.join(weight_cache_path, f"layer_{layer_idx}")
        os.makedirs(layer_cache_path, exist_ok=True)
        
        # Create attention layer normalization
        attention_norm = self._create_layernorm(
            weight=layer_state_dict["attention_norm.weight"],
            weight_cache_path=layer_cache_path,
            name=f"attention_norm_{layer_idx}"
        )
        
        # Create MLP layer normalization
        mlp_norm = self._create_layernorm(
            weight=layer_state_dict["ffn_norm.weight"],
            weight_cache_path=layer_cache_path,
            name=f"ffn_norm_{layer_idx}"
        )
        
        # Create attention components
        attention_qkv_cache_path = os.path.join(layer_cache_path, "attention_qkv")
        os.makedirs(attention_qkv_cache_path, exist_ok=True)
        
        # Query projection
        wq = Linear(
            in_features=self.args.dim,
            out_features=self.args.dim,
            bias=False,  # Ministral doesn't use bias in attention projections
            dtype=self.dtype,
            device=self.device,
            weight=layer_state_dict["attention.wq.weight"],
            weight_cache_path=os.path.join(attention_qkv_cache_path, "wq"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Key projection
        wk = Linear(
            in_features=self.args.dim,
            out_features=self.args.n_kv_heads * self.args.head_dim,
            bias=False,
            dtype=self.dtype,
            device=self.device,
            weight=layer_state_dict["attention.wk.weight"],
            weight_cache_path=os.path.join(attention_qkv_cache_path, "wk"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Value projection
        wv = Linear(
            in_features=self.args.dim,
            out_features=self.args.n_kv_heads * self.args.head_dim,
            bias=False,
            dtype=self.dtype,
            device=self.device,
            weight=layer_state_dict["attention.wv.weight"],
            weight_cache_path=os.path.join(attention_qkv_cache_path, "wv"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Output projection
        wo = Linear(
            in_features=self.args.dim,
            out_features=self.args.dim,
            bias=False,
            dtype=self.dtype,
            device=self.device,
            weight=layer_state_dict["attention.wo.weight"],
            weight_cache_path=os.path.join(layer_cache_path, "attention_wo"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # Create MLP components
        mlp_cache_path = os.path.join(layer_cache_path, "mlp")
        os.makedirs(mlp_cache_path, exist_ok=True)
        
        # MLP W1 (gate projection)
        w1 = Linear(
            in_features=self.args.dim,
            out_features=self.args.hidden_dim,  # Expanded size for Ministral-8B
            bias=False,
            dtype=self.dtype,
            device=self.device,
            weight=layer_state_dict["feed_forward.w1.weight"],
            weight_cache_path=os.path.join(mlp_cache_path, "w1"),
            program_config=self.args.model_config.get("PREFILL_MLP_W1_PRG_CONFIG"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # MLP W2 (up projection)
        w2 = Linear(
            in_features=self.args.dim,
            out_features=self.args.hidden_dim,
            bias=False,
            dtype=self.dtype,
            device=self.device,
            weight=layer_state_dict["feed_forward.w2.weight"],
            weight_cache_path=os.path.join(mlp_cache_path, "w2"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        # MLP W3 (down projection)
        w3 = Linear(
            in_features=self.args.hidden_dim,
            out_features=self.args.dim,
            bias=False,
            dtype=self.dtype,
            device=self.device,
            weight=layer_state_dict["feed_forward.w3.weight"],
            weight_cache_path=os.path.join(mlp_cache_path, "w3"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        
        from models.demos.wormhole.ministral8b.tt.mistral_common import (
            apply_rotary_emb,
        )
        
        # Create a transformer layer that combines all these components
        class TransformerLayer(LightweightModule):
            def __init__(self):
                super().__init__()
                self.attention_norm = attention_norm
                self.attention = AttentionBlock(wq, wk, wv, wo)
                self.mlp_norm = mlp_norm
                self.mlp = MlpBlock(w1, w2, w3)
                
                # Initialize KV cache for this layer
                self.attention.layer_past_list = []
                
            def forward(self, x, position, attn_mask=None, rot_mats=None, transformation_matrix=None, user_id=0, mode="decode"):
                # Apply pre-normalization
                residual = x
                x = self.attention_norm(x)
                
                # Apply attention block
                if mode == "prefill":
                    x = self.attention(x, position, attn_mask, rot_mats, transformation_matrix, user_id, mode)
                else:
                    x = self.attention(x, position, None, self.parent.rot_mat, None, user_id, mode)
                
                # Apply residual connection
                x = x + residual
                
                # Apply MLP with pre-normalization
                residual = x
                x = self.mlp_norm(x)
                x = self.mlp(x)
                
                # Apply residual connection
                x = x + residual
                
                return x
        
        # Inner attention block class to handle attention operations
        class AttentionBlock(LightweightModule):
            def __init__(self, wq, wk, wv, wo):
                super().__init__()
                self.wq = wq
                self.wk = wk
                self.wv = wv
                self.wo = wo
                self.layer_past_list = []  # For KV caching
            
            def forward(self, x, position, attn_mask=None, rot_mats=None, transformation_matrix=None, user_id=0, mode="decode"):
                import math
                import ttnn
                
                batch_size, seq_len, hidden_dim = x.shape
                head_dim = self.parent.parent.head_dim
                num_heads = self.parent.parent.n_heads
                num_kv_heads = self.parent.parent.n_kv_heads
                
                # Project query, key, value
                q = self.wq(x)  # [batch, seq, hidden]
                k = self.wk(x)  # [batch, seq, kv_heads*head_dim]
                v = self.wv(x)  # [batch, seq, kv_heads*head_dim]
                
                # Reshape to [batch, seq, num_heads, head_dim]
                q = ttnn.reshape(q, (batch_size, seq_len, num_heads, head_dim))
                k = ttnn.reshape(k, (batch_size, seq_len, num_kv_heads, head_dim))
                v = ttnn.reshape(v, (batch_size, seq_len, num_kv_heads, head_dim))
                
                # Apply rotary embeddings
                if mode == "prefill":
                    q = apply_rotary_emb(q, rot_mats, transformation_matrix)
                    k = apply_rotary_emb(k, rot_mats, transformation_matrix)
                    
                    # Compute attention score and apply mask
                    scale = 1.0 / math.sqrt(head_dim)
                    q = q * scale
                    
                    # Modify for grouped-query attention - repeat k,v if needed
                    if num_heads > num_kv_heads:
                        repeats = num_heads // num_kv_heads
                        k = ttnn.repeat(k, repeats, dim=2)
                        v = ttnn.repeat(v, repeats, dim=2)
                    
                    # Transpose for attention computation
                    q = ttnn.transpose(q, 1, 2)  # [batch, heads, seq, head_dim]
                    k = ttnn.transpose(k, 1, 2)  # [batch, heads, seq, head_dim]
                    v = ttnn.transpose(v, 1, 2)  # [batch, heads, seq, head_dim]
                    
                    # Compute attention scores
                    attn_scores = ttnn.matmul(q, ttnn.transpose(k, 2, 3))  # [batch, heads, seq, seq]
                    
                    # Apply attention mask if provided
                    if attn_mask is not None:
                        attn_scores = attn_scores + attn_mask
                    
                    # Softmax along last dimension
                    attn_probs = ttnn.softmax(attn_scores, dim=-1)
                    
                    # Apply attention to values
                    context = ttnn.matmul(attn_probs, v)  # [batch, heads, seq, head_dim]
                    context = ttnn.transpose(context, 1, 2)  # [batch, seq, heads, head_dim]
                    context = ttnn.reshape(context, (batch_size, seq_len, hidden_dim))
                    
                    # Project to output dimension
                    output = self.wo(context)
                    
                    return output
                else:
                    # For decode mode (autoregressive generation)
                    # Apply rotary embeddings
                    q = apply_rotary_emb(q, rot_mats, None)
                    k = apply_rotary_emb(k, rot_mats, None)
                    
                    # Ensure we have KV cache for this user
                    while len(self.layer_past_list) <= user_id:
                        device = self.parent.parent.device
                        kv_seq_len = self.parent.parent.args.kv_seq_len
                        
                        k_cache = ttnn.zeros(
                            (batch_size, kv_seq_len, num_kv_heads, head_dim),
                            dtype=self.parent.parent.dtype,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            memory_config=ttnn.L1_MEMORY_CONFIG
                        )
                        
                        v_cache = ttnn.zeros(
                            (batch_size, kv_seq_len, num_kv_heads, head_dim),
                            dtype=self.parent.parent.dtype,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            memory_config=ttnn.L1_MEMORY_CONFIG
                        )
                        
                        self.layer_past_list.append([k_cache, v_cache])
                    
                    # Get KV cache for this user
                    k_cache, v_cache = self.layer_past_list[user_id]
                    
                    # Update KV cache at current position
                    pos_idx = ttnn.to_layout(
                        ttnn.from_torch(
                            torch.tensor([[position]]), device=ttnn.get_device(k_cache)
                        ),
                        ttnn.ROW_MAJOR_LAYOUT
                    )
                    
                    # Update k_cache and v_cache with new key and value
                    k_cache = ttnn.tensor_scatter_update(k_cache, pos_idx, k)
                    v_cache = ttnn.tensor_scatter_update(v_cache, pos_idx, v)
                    
                    # Update layer past
                    self.layer_past_list[user_id] = [k_cache, v_cache]
                    
                    # Get cached keys and values
                    if position < self.parent.parent.sliding_window:
                        k_to_use = ttnn.slice(k_cache, (0, 0, 0, 0), (batch_size, position + 1, num_kv_heads, head_dim))
                        v_to_use = ttnn.slice(v_cache, (0, 0, 0, 0), (batch_size, position + 1, num_kv_heads, head_dim))
                    else:
                        window_size = self.parent.parent.sliding_window
                        start_idx = position - window_size + 1
                        k_to_use = ttnn.slice(k_cache, (0, start_idx, 0, 0), (batch_size, window_size, num_kv_heads, head_dim))
                        v_to_use = ttnn.slice(v_cache, (0, start_idx, 0, 0), (batch_size, window_size, num_kv_heads, head_dim))
                    
                    # Modify for grouped-query attention - repeat k,v if needed
                    if num_heads > num_kv_heads:
                        repeats = num_heads // num_kv_heads
                        k_to_use = ttnn.repeat(k_to_use, repeats, dim=2)
                        v_to_use = ttnn.repeat(v_to_use, repeats, dim=2)
                    
                    # Prepare for attention computation
                    scale = 1.0 / math.sqrt(head_dim)
                    q = q * scale
                    
                    # Transpose for attention computation
                    q = ttnn.transpose(q, 1, 2)  # [batch, heads, 1, head_dim]
                    k_to_use = ttnn.transpose(k_to_use, 1, 2)  # [batch, heads, window, head_dim]
                    v_to_use = ttnn.transpose(v_to_use, 1, 2)  # [batch, heads, window, head_dim]
                    
                    # Compute attention scores
                    attn_scores = ttnn.matmul(q, ttnn.transpose(k_to_use, 2, 3))  # [batch, heads, 1, window]
                    
                    # Softmax along last dimension
                    attn_probs = ttnn.softmax(attn_scores, dim=-1)
                    
                    # Apply attention to values
                    context = ttnn.matmul(attn_probs, v_to_use)  # [batch, heads, 1, head_dim]
                    context = ttnn.transpose(context, 1, 2)  # [batch, 1, heads, head_dim]
                    context = ttnn.reshape(context, (batch_size, seq_len, hidden_dim))
                    
                    # Project to output dimension
                    output = self.wo(context)
                    
                    return output
        
        # Inner MLP block class to handle feedforward operations
        class MlpBlock(LightweightModule):
            def __init__(self, w1, w2, w3):
                super().__init__()
                self.w1 = w1  # Gate projection
                self.w2 = w2  # Up projection
                self.w3 = w3  # Down projection
            
            def forward(self, x):
                # SwiGLU activation as used in Ministral-8B
                gate = self.w1(x)
                gate = ttnn.silu(gate)
                
                up = self.w2(x)
                
                # Element-wise multiplication
                activations = gate * up
                
                # Down projection
                output = self.w3(activations)
                
                return output
        
        # Create transformer layer instance
        layer = TransformerLayer()
        layer.parent = self  # Set parent reference for access to model parameters
        layer.attention.parent = layer  # Set parent reference for access to layer
        layer.mlp.parent = layer  # Set parent reference for access to layer
        
        return layer
    
    def _create_layernorm(self, weight, weight_cache_path, name):
        """Create layer normalization module
        
        Args:
            weight: Layer norm weight
            weight_cache_path: Path to cache weights
            name: Name for the layernorm module
            
        Returns:
            Layer normalization module
        """
        from ttnn.module import RMSNorm
        
        # Create cache path
        cache_path = os.path.join(weight_cache_path, name)
        os.makedirs(cache_path, exist_ok=True)
        
        # Create RMSNorm module
        rms_norm = RMSNorm(
            dim=self.args.dim,
            eps=self.args.norm_eps,
            device=self.device,
            dtype=self.dtype,
            weight=weight,
            weight_cache_path=cache_path,
        )
        
        return rms_norm
    
    def _create_lm_head(self, state_dict, weight_cache_path):
        """Create language model head
        
        Args:
            state_dict: Model state dictionary
            weight_cache_path: Path to cache weights
            
        Returns:
            Linear layer for LM head
        """
        from ttnn.layer import Linear
        
        # Create cache path
        cache_path = os.path.join(weight_cache_path, "lm_head")
        os.makedirs(cache_path, exist_ok=True)
        
        # Create LM head module
        lm_head = Linear(
            in_features=self.args.dim,
            out_features=self.args.vocab_size,
            bias=False,
            dtype=self.dtype,
            device=self.device,
            weight=state_dict["output.weight"],
            weight_cache_path=cache_path,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.args.model_config.get("OUTPUT_MM_PROGCFG"),
        )
        
        return lm_head
    
    def forward(self, hidden_states, current_pos, attn_mask=None, rot_mats=None, transformation_matrix=None, user_id=0, mode="decode"):
        """Forward pass for the Ministral-8B transformer model
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            current_pos: Current position for decoding
            attn_mask: Optional attention mask [batch_size, 1, seq_len, seq_len]
            rot_mats: Optional rotary embeddings for position
            transformation_matrix: Optional transformation matrix for rotary embeddings
            user_id: User ID for batched generation
            mode: "decode" for autoregressive generation, "prefill" for prefill mode
            
        Returns:
            Output logits [batch_size, seq_len, vocab_size]
        """
        # Update current position
        self.current_pos = current_pos
        
        # Process through all transformer layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, current_pos, attn_mask, rot_mats, transformation_matrix, user_id, mode)
        
        # Apply final layer normalization
        hidden_states = self.norm(hidden_states)
        
        # Project to vocab size with LM head
        logits = self.lm_head(hidden_states)
        
        return logits
