#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight conversion utility for Ministral-8B following tt-transformers pattern.
Converts Hugging Face weights to TTNN format with proper sharding and caching.
"""

import os
import json
import math
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
from loguru import logger
import ttnn

from models.tt_transformers.tt.model_config import ModelArgs


class MistralWeightConverter:
    """
    Converts Ministral-8B weights from Hugging Face format to tt-transformers format.
    Handles weight naming conventions, sharding, and caching following tt-transformers patterns.
    """
    
    def __init__(self, model_args: ModelArgs, dtype: ttnn.DataType = ttnn.bfloat16):
        self.model_args = model_args
        self.dtype = dtype
        self.mesh_device = model_args.mesh_device
        self.num_devices = model_args.num_devices
        self.cache_path = model_args.weight_cache_path(dtype)
        
        # Ministral-8B specific weight mappings
        self.weight_mappings = self._create_weight_mappings()
        
        # Create cache directory
        os.makedirs(self.cache_path, exist_ok=True)
        
    def _create_weight_mappings(self) -> Dict[str, str]:
        """Create mapping from HF weight names to tt-transformers format."""
        mappings = {}
        
        # Embedding weights
        mappings["model.embed_tokens.weight"] = "tok_embeddings.weight"
        
        # Final layer norm and output
        mappings["model.norm.weight"] = "norm.weight"
        mappings["lm_head.weight"] = "output.weight"
        
        # Layer-specific weights
        for i in range(self.model_args.n_layers):
            layer_prefix = f"model.layers.{i}"
            tt_prefix = f"layers.{i}"
            
            # Attention weights
            mappings[f"{layer_prefix}.input_layernorm.weight"] = f"{tt_prefix}.attention_norm.weight"
            mappings[f"{layer_prefix}.self_attn.q_proj.weight"] = f"{tt_prefix}.attention.wq.weight"
            mappings[f"{layer_prefix}.self_attn.k_proj.weight"] = f"{tt_prefix}.attention.wk.weight"
            mappings[f"{layer_prefix}.self_attn.v_proj.weight"] = f"{tt_prefix}.attention.wv.weight"
            mappings[f"{layer_prefix}.self_attn.o_proj.weight"] = f"{tt_prefix}.attention.wo.weight"
            
            # MLP weights
            mappings[f"{layer_prefix}.post_attention_layernorm.weight"] = f"{tt_prefix}.ffn_norm.weight"
            mappings[f"{layer_prefix}.mlp.gate_proj.weight"] = f"{tt_prefix}.feed_forward.w1.weight"
            mappings[f"{layer_prefix}.mlp.up_proj.weight"] = f"{tt_prefix}.feed_forward.w3.weight"
            mappings[f"{layer_prefix}.mlp.down_proj.weight"] = f"{tt_prefix}.feed_forward.w2.weight"
            
        return mappings
    
    def _calculate_checksum(self, tensor: torch.Tensor) -> str:
        """Calculate checksum for weight integrity validation."""
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()
    
    def _apply_sharding(self, weight: torch.Tensor, weight_name: str) -> torch.Tensor:
        """Apply proper sharding for multi-device deployment."""
        if self.num_devices <= 1:
            return weight
            
        # Determine sharding strategy based on weight type
        if "attention.wq" in weight_name or "attention.wk" in weight_name or "attention.wv" in weight_name:
            # Shard attention weights along head dimension
            if weight.dim() == 2:
                # Split along output dimension (heads)
                chunk_size = weight.shape[0] // self.num_devices
                return weight[:chunk_size * self.num_devices].chunk(self.num_devices, dim=0)
        elif "attention.wo" in weight_name:
            # Shard output projection along input dimension
            if weight.dim() == 2:
                chunk_size = weight.shape[1] // self.num_devices
                return weight[:, :chunk_size * self.num_devices].chunk(self.num_devices, dim=1)
        elif "feed_forward.w1" in weight_name or "feed_forward.w3" in weight_name:
            # Shard MLP gate/up projections along output dimension
            if weight.dim() == 2:
                chunk_size = weight.shape[0] // self.num_devices
                return weight[:chunk_size * self.num_devices].chunk(self.num_devices, dim=0)
        elif "feed_forward.w2" in weight_name:
            # Shard MLP down projection along input dimension
            if weight.dim() == 2:
                chunk_size = weight.shape[1] // self.num_devices
                return weight[:, :chunk_size * self.num_devices].chunk(self.num_devices, dim=1)
        elif "output.weight" in weight_name:
            # Shard LM head along vocabulary dimension
            if weight.dim() == 2:
                chunk_size = weight.shape[0] // self.num_devices
                return weight[:chunk_size * self.num_devices].chunk(self.num_devices, dim=0)
                
        return weight
    
    def _convert_to_ttnn(self, weight: torch.Tensor, weight_name: str) -> ttnn.Tensor:
        """Convert PyTorch tensor to TTNN format with proper memory layout."""
        # Apply sharding if needed
        sharded_weight = self._apply_sharding(weight, weight_name)
        
        if isinstance(sharded_weight, (list, tuple)):
            # Handle sharded weights
            ttnn_weights = []
            for shard in sharded_weight:
                ttnn_weight = ttnn.from_torch(
                    shard,
                    dtype=self.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if self.num_devices > 1 else None,
                )
                ttnn_weights.append(ttnn_weight)
            return ttnn_weights
        else:
            # Handle non-sharded weights
            mesh_mapper = None
            if self.num_devices > 1:
                if "norm" in weight_name or "tok_embeddings" in weight_name:
                    mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
                else:
                    mesh_mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=-1)
            
            return ttnn.from_torch(
                sharded_weight,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=mesh_mapper,
            )
    
    def _save_weight_cache(self, weight_name: str, ttnn_weight: ttnn.Tensor, checksum: str):
        """Save weight to cache with metadata."""
        cache_file = self.cache_path / f"{weight_name.replace('.', '_')}.bin"
        metadata_file = self.cache_path / f"{weight_name.replace('.', '_')}_metadata.json"
        
        # Save weight
        ttnn.dump_tensor(cache_file, ttnn_weight)
        
        # Save metadata
        metadata = {
            "weight_name": weight_name,
            "checksum": checksum,
            "dtype": str(self.dtype),
            "shape": list(ttnn_weight.shape) if hasattr(ttnn_weight, 'shape') else None,
            "num_devices": self.num_devices,
            "model_name": self.model_args.model_name,
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_weight_cache(self, weight_name: str) -> Optional[ttnn.Tensor]:
        """Load weight from cache if available and valid."""
        cache_file = self.cache_path / f"{weight_name.replace('.', '_')}.bin"
        metadata_file = self.cache_path / f"{weight_name.replace('.', '_')}_metadata.json"
        
        if not (cache_file.exists() and metadata_file.exists()):
            return None
            
        try:
            # Load and validate metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            if (metadata.get("dtype") != str(self.dtype) or 
                metadata.get("num_devices") != self.num_devices or
                metadata.get("model_name") != self.model_args.model_name):
                logger.warning(f"Cache metadata mismatch for {weight_name}, regenerating")
                return None
                
            # Load weight
            return ttnn.load_tensor(cache_file)
            
        except Exception as e:
            logger.warning(f"Failed to load cached weight {weight_name}: {e}")
            return None
    
    def convert_weights(self, state_dict: Dict[str, torch.Tensor], 
                       use_cache: bool = True) -> Dict[str, ttnn.Tensor]:
        """
        Convert all weights from HuggingFace format to tt-transformers format.
        
        Args:
            state_dict: HuggingFace model state dict
            use_cache: Whether to use cached weights if available
            
        Returns:
            Dictionary of converted weights in tt-transformers format
        """
        converted_weights = {}
        cache_hits = 0
        total_weights = len(self.weight_mappings)
        
        logger.info(f"Converting {total_weights} weights for Ministral-8B")
        
        for hf_name, tt_name in self.weight_mappings.items():
            if hf_name not in state_dict:
                logger.warning(f"Weight {hf_name} not found in state dict, skipping")
                continue
                
            # Try to load from cache first
            if use_cache:
                cached_weight = self._load_weight_cache(tt_name)
                if cached_weight is not None:
                    converted_weights[tt_name] = cached_weight
                    cache_hits += 1
                    continue
            
            # Convert weight
            logger.debug(f"Converting {hf_name} -> {tt_name}")
            original_weight = state_dict[hf_name]
            
            # Calculate checksum for integrity
            checksum = self._calculate_checksum(original_weight)
            
            # Convert to TTNN format
            ttnn_weight = self._convert_to_ttnn(original_weight, tt_name)
            converted_weights[tt_name] = ttnn_weight
            
            # Save to cache
            if use_cache:
                try:
                    self._save_weight_cache(tt_name, ttnn_weight, checksum)
                except Exception as e:
                    logger.warning(f"Failed to cache weight {tt_name}: {e}")
        
        logger.info(f"Weight conversion complete: {cache_hits}/{total_weights} from cache, "
                   f"{total_weights - cache_hits} converted")
        
        return converted_weights
    
    def validate_weights(self, converted_weights: Dict[str, ttnn.Tensor]) -> bool:
        """Validate converted weights for completeness and correctness."""
        required_weights = set(self.weight_mappings.values())
        converted_names = set(converted_weights.keys())
        
        missing_weights = required_weights - converted_names
        if missing_weights:
            logger.error(f"Missing weights: {missing_weights}")
            return False
            
        extra_weights = converted_names - required_weights
        if extra_weights:
            logger.warning(f"Extra weights found: {extra_weights}")
        
        # Validate weight shapes
        for weight_name, weight in converted_weights.items():
            if hasattr(weight, 'shape'):
                if any(dim == 0 for dim in weight.shape):
                    logger.error(f"Invalid shape for {weight_name}: {weight.shape}")
                    return False
        
        logger.info("Weight validation passed")
        return True


def convert_hf_mistral_weights(model_args: ModelArgs, 
                              state_dict: Optional[Dict[str, torch.Tensor]] = None,
                              dtype: ttnn.DataType = ttnn.bfloat16,
                              use_cache: bool = True) -> Dict[str, ttnn.Tensor]:
    """
    Main function to convert Ministral-8B weights from HuggingFace to tt-transformers format.
    
    Args:
        model_args: Model configuration
        state_dict: HuggingFace state dict (if None, will load from model_args.CKPT_DIR)
        dtype: Target TTNN data type
        use_cache: Whether to use weight caching
        
    Returns:
        Dictionary of converted weights
    """
    # Load state dict if not provided
    if state_dict is None:
        logger.info(f"Loading weights from {model_args.CKPT_DIR}")
        if model_args.from_hf_url:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(model_args.CKPT_DIR)
            state_dict = model.state_dict()
        else:
            # Load from local checkpoint
            checkpoint_path = Path(model_args.CKPT_DIR) / "consolidated.00.pth"
            if checkpoint_path.exists():
                state_dict = torch.load(checkpoint_path, map_location="cpu")
            else:
                # Try PyTorch format
                checkpoint_path = Path(model_args.CKPT_DIR) / "pytorch_model.bin"
                if checkpoint_path.exists():
                    state_dict = torch.load(checkpoint_path, map_location="cpu")
                else:
                    # Try safetensors format
                    try:
                        from safetensors.torch import load_file
                        checkpoint_path = Path(model_args.CKPT_DIR) / "model.safetensors"
                        if checkpoint_path.exists():
                            state_dict = load_file(checkpoint_path)
                        else:
                            raise FileNotFoundError(f"No valid checkpoint found in {model_args.CKPT_DIR}")
                    except ImportError:
                        raise ImportError("safetensors not available, install with: pip install safetensors")
    
    # Create converter and convert weights
    converter = MistralWeightConverter(model_args, dtype)
    converted_weights = converter.convert_weights(state_dict, use_cache)
    
    # Validate converted weights
    if not converter.validate_weights(converted_weights):
        raise RuntimeError("Weight validation failed")
    
    logger.info(f"Successfully converted {len(converted_weights)} weights for Ministral-8B")
    return converted_weights


def main():
    """CLI interface for weight conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Ministral-8B weights to tt-transformers format")
    parser.add_argument("--model-dir", type=str, required=True,
                       help="Path to HuggingFace model directory")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Path to weight cache directory")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "bfloat8_b"],
                       help="Target data type")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable weight caching")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing cached weights")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["HF_MODEL"] = args.model_dir
    if args.cache_dir:
        os.environ["TT_CACHE_PATH"] = args.cache_dir
    
    # Create model args (without device for conversion)
    model_args = ModelArgs(
        mesh_device=None,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=32768,
    )
    
    # Set dtype
    dtype = ttnn.bfloat16 if args.dtype == "bfloat16" else ttnn.bfloat8_b
    
    if args.validate_only:
        # Only validate cached weights
        converter = MistralWeightConverter(model_args, dtype)
        cache_path = converter.cache_path
        if not cache_path.exists():
            logger.error(f"Cache directory {cache_path} does not exist")
            return 1
            
        # Load and validate cached weights
        cached_weights = {}
        for weight_file in cache_path.glob("*.bin"):
            weight_name = weight_file.stem.replace("_", ".")
            try:
                cached_weights[weight_name] = converter._load_weight_cache(weight_name)
            except Exception as e:
                logger.error(f"Failed to load cached weight {weight_name}: {e}")
                return 1
        
        if converter.validate_weights(cached_weights):
            logger.info("All cached weights are valid")
            return 0
        else:
            logger.error("Cached weight validation failed")
            return 1
    else:
        # Convert weights
        try:
            converted_weights = convert_hf_mistral_weights(
                model_args=model_args,
                dtype=dtype,
                use_cache=not args.no_cache
            )
            logger.info(f"Conversion successful: {len(converted_weights)} weights converted")
            return 0
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return 1


if __name__ == "__main__":
    exit(main())