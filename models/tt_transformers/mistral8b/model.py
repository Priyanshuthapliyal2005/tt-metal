# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.model import Transformer
from models.tt_transformers.tt.model_config import ModelArgs


class MistralModelArgs(ModelArgs):
    """Ministral-8B specific model configuration extending the base ModelArgs"""
    
    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=32768,
        optimizations=None,
    ):
        # Initialize base ModelArgs
        super().__init__(
            mesh_device=mesh_device,
            instruct=instruct,
            dummy_weights=dummy_weights,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
        )
        
        # Override with Ministral-8B specific parameters
        self._set_ministral_params()
        
        # Update model name for proper identification
        self.model_name = "Ministral-8B-Instruct-2410" if instruct else "Ministral-8B-2410"
        
        # Set checkpoint paths for Ministral-8B
        self._set_ministral_paths()
        
        # Recalculate configurations with Ministral-specific parameters
        self._update_ministral_configs()
    
    def _set_ministral_params(self):
        """Set Ministral-8B specific model parameters"""
        # Core model dimensions
        self.dim = 4096
        self.n_layers = 32
        self.n_heads = 32
        self.n_kv_heads = 8  # Grouped-query attention
        self.head_dim = 128
        self.vocab_size = 131072  # Ministral-8B uses larger vocabulary
        
        # MLP configuration
        self.hidden_dim = 14336  # Intermediate size for MLP
        self.ffn_dim_multiplier = None
        self.multiple_of = None
        
        # Normalization
        self.norm_eps = 1e-5
        
        # RoPE configuration
        self.rope_theta = 1000000.0
        self.rope_scaling_factor = None  # Ministral doesn't use scaled RoPE by default
        self.orig_context_len = None
        
        # Sliding window attention (Ministral-specific)
        self.sliding_window = 4096
        
        # Context length
        self.max_context_len = 32768
        
        # Update derived parameters
        self.is_70b = False
        self.is_90b = False
        
        logger.info(f"Configured Ministral-8B parameters: dim={self.dim}, n_layers={self.n_layers}, "
                   f"n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}, vocab_size={self.vocab_size}")
    
    def _set_ministral_paths(self):
        """Set Ministral-8B specific checkpoint and cache paths"""
        # Check for environment variables first
        ministral_dir = os.getenv("MINISTRAL_CKPT_DIR")
        hf_model = os.getenv("HF_MODEL")
        
        if hf_model and "ministral" in hf_model.lower():
            # Use HF model path
            self.CKPT_DIR = hf_model
            self.TOKENIZER_PATH = hf_model
            self.model_name = hf_model.strip("/").split("/")[-1]
            self.from_hf_url = True
            
            if not self.CACHE_PATH:
                self.CACHE_PATH = os.path.join("model_cache", hf_model, self.device_name)
            else:
                self.CACHE_PATH = os.path.join(self.CACHE_PATH, self.device_name)
                
        elif ministral_dir:
            # Use Ministral-specific directory
            self.CKPT_DIR = ministral_dir
            self.TOKENIZER_PATH = ministral_dir
            
            if not self.CACHE_PATH:
                self.CACHE_PATH = os.path.join(ministral_dir, self.device_name)
                
        else:
            # Default Ministral paths
            default_ministral_dir = "/mnt/MLPerf/tt_dnn-models/Mistral/ministral-8b-instruct-2410/"
            self.CKPT_DIR = default_ministral_dir
            self.TOKENIZER_PATH = default_ministral_dir
            
            if not self.CACHE_PATH:
                self.CACHE_PATH = os.path.join(default_ministral_dir, self.device_name)
        
        # Update paths
        self.model_base_path = self.CKPT_DIR
        self.model_cache_path = self.CACHE_PATH
        self.consolidated_weights_path = os.path.join(self.CKPT_DIR, "consolidated.00.pth")
        self.tokenizer_path = os.path.join(self.TOKENIZER_PATH, "tokenizer.model")
        
        logger.info(f"Ministral-8B checkpoint directory: {self.CKPT_DIR}")
        logger.info(f"Ministral-8B tokenizer path: {self.tokenizer_path}")
        logger.info(f"Ministral-8B cache directory: {self.CACHE_PATH}")
    
    def _update_ministral_configs(self):
        """Update model configurations for Ministral-8B specific requirements"""
        if self.mesh_device is None:
            return
            
        # Update vocabulary size configurations
        self.padded_vocab_size = 128 * 1024 if self.is_galaxy else None
        
        # Update LM head configuration for larger vocabulary
        if hasattr(self, 'lm_head_core_grid'):
            # Recalculate LM head grid for larger vocabulary
            lm_head_num_rows = 8
            lm_head_cores_per_row = 8
            
            # Find optimal grid for Ministral's vocabulary size
            while self.vocab_size % (32 * lm_head_num_rows * lm_head_cores_per_row) != 0:
                lm_head_num_rows -= 1
                if lm_head_num_rows == 0:
                    lm_head_cores_per_row -= 1
                    if lm_head_cores_per_row == 0:
                        raise ValueError(
                            f"Could not find LM head grid for vocab_size={self.vocab_size}"
                        )
                    lm_head_num_rows = 8
            
            self.lm_head_core_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=lm_head_cores_per_row)
            
            # Update max columns per device for larger vocabulary
            self.max_columns_per_device_lm_head = 668 * self.lm_head_core_grid.num_cores
            
            # Update LM head input memory config
            self.model_config["LM_HEAD_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                (
                    self.tile_padded_batch_rows,
                    self.nearest_32((self.dim // (4 if self.is_galaxy else 1)) // self.lm_head_core_grid.num_cores),
                ),
                self.lm_head_core_grid,
                ttnn.ShardStrategy.WIDTH,
                ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        
        # Update QKV size for grouped-query attention
        self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_heads)
        
        # Update sliding window attention configurations if needed
        if hasattr(self, 'model_config'):
            # Add Ministral-specific SDPA configuration with sliding window support
            self.model_config["MINISTRAL_SLIDING_WINDOW"] = self.sliding_window
            
            # Update SDPA program config for sliding window
            original_sdpa_config = self.model_config.get("SDPA_PROGCFG")
            if original_sdpa_config:
                def ministral_sdpa_config(seqlen):
                    config = original_sdpa_config(seqlen)
                    # Adjust chunk sizes for sliding window if sequence is longer than window
                    if seqlen > self.sliding_window:
                        config.q_chunk_size = min(config.q_chunk_size, self.sliding_window // 4)
                        config.k_chunk_size = min(config.k_chunk_size, self.sliding_window // 4)
                    return config
                
                self.model_config["SDPA_PROGCFG"] = ministral_sdpa_config
    
    @property
    def base_model_name(self):
        """Return base model name for Ministral-8B"""
        return "Ministral-8B"
    
    def get_state_dict_prefix(self, module_name, layer_num):
        """Get state dict prefix for Ministral-8B model structure"""
        # Ministral uses standard transformer structure
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""
        module_map = {
            "MLP": "feed_forward",
            "Attention": "attention", 
            "TransformerBlock": "",
            "": "",
        }
        return layer_prefix + module_map[module_name]


class MistralTransformer(Transformer):
    """Ministral-8B transformer model using shared tt-transformers framework"""
    
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        # Ensure we're using MistralModelArgs
        if not isinstance(args, MistralModelArgs):
            raise TypeError("MistralTransformer requires MistralModelArgs")
        
        logger.info("Initializing Ministral-8B transformer with shared tt-transformers framework")
        
        # Initialize the base Transformer with Ministral-specific args
        super().__init__(
            args=args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )
        
        # Store Ministral-specific parameters
        self.sliding_window = args.sliding_window
        
        logger.info(f"Ministral-8B transformer initialized with {args.n_layers} layers, "
                   f"sliding window size: {self.sliding_window}")
    
    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
    ):
        """
        Forward pass with Ministral-8B specific handling
        
        Delegates to the base Transformer implementation while maintaining
        compatibility with the existing server interface.
        """
        try:
            # Use the shared transformer implementation
            return super().forward(
                x=x,
                current_pos=current_pos,
                rot_mats=rot_mats,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                get_last_token=get_last_token,
                kv_cache=kv_cache,
            )
        except Exception as e:
            logger.error(f"Error in Ministral-8B forward pass: {e}")
            raise
    
    def prepare_inputs_prefill(self, tokens, start_pos=0, page_table=None, chunk_page_table=None):
        """
        Prepare inputs for prefill mode with Ministral-8B specific handling
        """
        try:
            return super().prepare_inputs_prefill(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
            )
        except Exception as e:
            logger.error(f"Error preparing Ministral-8B prefill inputs: {e}")
            raise
    
    def prepare_inputs_decode(self, *inputs):
        """
        Prepare inputs for decode mode with Ministral-8B specific handling
        """
        try:
            return super().prepare_inputs_decode(*inputs)
        except Exception as e:
            logger.error(f"Error preparing Ministral-8B decode inputs: {e}")
            raise
    
    def process_output_prefill(self, tt_out, last_token_idx):
        """
        Process prefill output with Ministral-8B specific handling
        """
        try:
            return super().process_output_prefill(tt_out, last_token_idx)
        except Exception as e:
            logger.error(f"Error processing Ministral-8B prefill output: {e}")
            raise
    
    def process_output_decode(self, tt_out, B, S=1, is_tokens=False):
        """
        Process decode output with Ministral-8B specific handling
        """
        try:
            return super().process_output_decode(tt_out, B, S, is_tokens)
        except Exception as e:
            logger.error(f"Error processing Ministral-8B decode output: {e}")
            raise


def create_ministral_model(
    mesh_device,
    instruct=True,
    max_batch_size=1,
    max_seq_len=32768,
    optimizations=None,
    dtype=ttnn.bfloat8_b,
    state_dict=None,
    paged_attention_config=None,
    use_paged_kv_cache=False,
):
    """
    Create a Ministral-8B model using the shared tt-transformers framework
    
    Args:
        mesh_device: TT mesh device
        instruct: Whether to use instruct model
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length
        optimizations: Model optimizations configuration
        dtype: Model data type
        state_dict: Pre-loaded state dictionary
        paged_attention_config: Paged attention configuration
        use_paged_kv_cache: Whether to use paged KV cache
        
    Returns:
        Tuple of (model_args, model, kv_cache, state_dict)
    """
    try:
        logger.info("Creating Ministral-8B model with tt-transformers framework")
        
        # Create Ministral-specific model arguments
        model_args = MistralModelArgs(
            mesh_device=mesh_device,
            instruct=instruct,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
        )
        
        # Load state dict if not provided
        if state_dict is None:
            logger.info("Loading Ministral-8B state dictionary")
            state_dict = model_args.load_state_dict()
        
        # Create the Ministral transformer model
        model = MistralTransformer(
            args=model_args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype),
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )
        
        # Create KV cache if using paged attention
        kv_cache = None
        if paged_attention_config:
            kv_cache = [layer.attention.layer_past for layer in model.layers]
        
        logger.info("Ministral-8B model created successfully")
        
        return model_args, model, kv_cache, state_dict
        
    except Exception as e:
        logger.error(f"Failed to create Ministral-8B model: {e}")
        raise


# Backward compatibility aliases for existing server interface
TtTransformer = MistralTransformer
TtModelArgs = MistralModelArgs

# Export the main classes and functions
__all__ = [
    "MistralModelArgs",
    "MistralTransformer", 
    "create_ministral_model",
    "TtTransformer",  # Backward compatibility
    "TtModelArgs",    # Backward compatibility
]