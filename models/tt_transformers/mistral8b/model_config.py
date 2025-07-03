# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import json
from pathlib import Path
from loguru import logger

import ttnn
from models.tt_transformers.tt.model_config import ModelArgs, ModelOptimizations


class MistralModelArgs(ModelArgs):
    """Model configuration for Ministral-8B extending the base ModelArgs class"""
    
    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=32768,
        optimizations=None,
    ):
        # Set Ministral-8B specific parameters before calling parent init
        self._set_ministral_params()
        
        # Call parent constructor
        super().__init__(
            mesh_device=mesh_device,
            instruct=instruct,
            dummy_weights=dummy_weights,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
        )
        
        # Override model name and paths for Ministral-8B
        self._override_ministral_paths()
        
        # Set Ministral-specific configurations
        self._configure_ministral_specifics()

    def _set_ministral_params(self):
        """Set Ministral-8B specific model parameters"""
        # Core model dimensions
        self.dim = 4096
        self.n_layers = 32
        self.n_heads = 32
        self.n_kv_heads = 8
        self.head_dim = 128
        
        # Vocabulary and sequence configuration
        self.vocab_size = 131072
        self.max_context_len = 32768
        
        # MLP configuration
        self.intermediate_size = 14336
        self.hidden_dim = 14336
        self.ffn_dim_multiplier = None
        self.multiple_of = None
        
        # Normalization
        self.norm_eps = 1e-5
        
        # RoPE configuration
        self.rope_theta = 1000000.0
        self.rope_scaling_factor = None
        self.orig_context_len = None
        
        # Ministral-specific: Sliding window attention
        self.sliding_window = 4096
        
        # Model identification
        self.model_name = "Ministral-8B-Instruct-2410"
        self.base_model_name = "Ministral-8B"

    def _override_ministral_paths(self):
        """Override default paths for Ministral-8B"""
        # Check for environment variables first
        ministral_dir = os.getenv("MINISTRAL_DIR")
        ministral_ckpt_dir = os.getenv("MINISTRAL_CKPT_DIR")
        ministral_tokenizer_path = os.getenv("MINISTRAL_TOKENIZER_PATH")
        ministral_cache_path = os.getenv("MINISTRAL_CACHE_PATH")
        
        if ministral_dir:
            self.CKPT_DIR = ministral_dir
            self.TOKENIZER_PATH = ministral_dir
            if not self.CACHE_PATH:
                self.CACHE_PATH = os.path.join(ministral_dir, self.device_name)
        elif ministral_ckpt_dir:
            self.CKPT_DIR = ministral_ckpt_dir
            if ministral_tokenizer_path:
                self.TOKENIZER_PATH = ministral_tokenizer_path
            else:
                self.TOKENIZER_PATH = ministral_ckpt_dir
            if ministral_cache_path:
                self.CACHE_PATH = ministral_cache_path
            else:
                self.CACHE_PATH = os.path.join(ministral_ckpt_dir, self.device_name)
        
        # Update paths
        self.model_base_path = Path(self.CKPT_DIR)
        self.model_cache_path = Path(self.CACHE_PATH)
        
        # Update weight and tokenizer paths
        self.consolidated_weights_path = os.path.join(self.CKPT_DIR, "consolidated.00.pth")
        self.tokenizer_path = os.path.join(self.TOKENIZER_PATH, "tokenizer.model")
        
        logger.info(f"Ministral-8B Checkpoint directory: {self.CKPT_DIR}")
        logger.info(f"Ministral-8B Tokenizer path: {self.tokenizer_path}")
        logger.info(f"Ministral-8B Cache directory: {self.CACHE_PATH}")

    def _configure_ministral_specifics(self):
        """Configure Ministral-8B specific settings"""
        # Set optimizations if not provided
        if self.optimizations is None:
            # Use performance optimizations by default for Ministral-8B
            self.optimizations = ModelOptimizations.performance(self.model_name)
        
        # Override prefill length cutoff for sliding window attention
        self.prefill_len_cutoff = min(self.sliding_window, 1024)
        
        # Ministral-8B specific memory and compute configurations
        if self.mesh_device is not None:
            self._configure_ministral_memory_configs()

    def _configure_ministral_memory_configs(self):
        """Configure Ministral-8B specific memory and compute settings"""
        # Override specific configurations for Ministral-8B
        
        # Sliding window attention configuration
        self.model_config["SLIDING_WINDOW"] = self.sliding_window
        
        # Adjust SDPA configuration for sliding window
        original_sdpa_progcfg = self.model_config.get("SDPA_PROGCFG")
        if original_sdpa_progcfg:
            def ministral_sdpa_progcfg(seqlen):
                # Use smaller chunks for sliding window attention
                effective_seqlen = min(seqlen, self.sliding_window)
                return ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8),
                    exp_approx_mode=False,
                    q_chunk_size=128 if effective_seqlen >= 2048 else 64,
                    k_chunk_size=128 if effective_seqlen >= 2048 else 64,
                )
            self.model_config["SDPA_PROGCFG"] = ministral_sdpa_progcfg
        
        # Adjust prefill configurations for Ministral-8B dimensions
        if hasattr(self, 'max_grid_size'):
            # MLP configurations optimized for Ministral-8B hidden_dim=14336
            mlp_grid = self.find_prefill_grid(8, self.hidden_dim // self.tile_size)
            
            self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, self.prefill_len_cutoff),
                k=self.dim,
                n=self.hidden_dim // self.num_devices,
                grid_size=mlp_grid,
            )
            
            self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = lambda seq_len: self.matmul_config(
                m=min(seq_len, self.prefill_len_cutoff),
                k=self.hidden_dim // self.num_devices,
                n=self.dim,
                grid_size=mlp_grid,
            )

    def is_sliding_window_attention(self):
        """Check if sliding window attention is enabled"""
        return hasattr(self, 'sliding_window') and self.sliding_window is not None

    def get_effective_seq_len(self, seq_len):
        """Get effective sequence length considering sliding window"""
        if self.is_sliding_window_attention():
            return min(seq_len, self.sliding_window)
        return seq_len

    def _set_params_from_dict(self, config, is_hf=False):
        """Override to handle Ministral-8B specific config parsing"""
        # Call parent method first
        super()._set_params_from_dict(config, is_hf)
        
        # Override with Ministral-8B specific values
        self._set_ministral_params()
        
        # Handle sliding window from config if present
        if "sliding_window" in config:
            self.sliding_window = config["sliding_window"]

    def __repr__(self):
        return f"""MistralModelArgs(
    dim={self.dim},
    n_layers={self.n_layers},
    n_heads={self.n_heads},
    n_kv_heads={self.n_kv_heads},
    head_dim={self.head_dim},
    vocab_size={self.vocab_size},
    intermediate_size={self.intermediate_size},
    norm_eps={self.norm_eps},
    rope_theta={self.rope_theta},
    sliding_window={self.sliding_window},
    max_batch_size={self.max_batch_size},
    max_seq_len={self.max_seq_len},
    device_name={self.device_name},
    num_devices={self.num_devices}
)"""


def create_mistral_model_args(
    mesh_device,
    instruct=True,
    dummy_weights=False,
    max_batch_size=1,
    max_seq_len=32768,
    optimizations=None,
):
    """
    Factory function to create MistralModelArgs with sensible defaults
    
    Args:
        mesh_device: TTNN mesh device
        instruct: Whether to use instruct model variant
        dummy_weights: Whether to use dummy weights for testing
        max_batch_size: Maximum batch size
        max_seq_len: Maximum sequence length
        optimizations: Model optimization settings
    
    Returns:
        MistralModelArgs: Configured model arguments
    """
    return MistralModelArgs(
        mesh_device=mesh_device,
        instruct=instruct,
        dummy_weights=dummy_weights,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        optimizations=optimizations,
    )