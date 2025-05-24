#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Convert Ministral-8B-Instruct-2410 weights from Hugging Face format to Tenstorrent format
"""

import os
import torch
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

def convert_hf_mistral_to_tenstorrent(model_dir, output_dir=None):
    """
    Convert Hugging Face Ministral-8B-Instruct-2410 model to Tenstorrent format
    
    Args:
        model_dir: Path to Hugging Face model directory
        output_dir: Path to output directory (if None, use model_dir)
    """
    logger.info(f"Loading model from {model_dir}")
    
    # Use the same directory if output_dir not specified
    if output_dir is None:
        output_dir = model_dir

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model from HuggingFace
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )
    
    # Save tokenizer in the format expected by TT-Metal
    tokenizer_path = os.path.join(output_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_path):
        logger.info(f"Saving tokenizer to {tokenizer_path}")
        with open(tokenizer_path, "wb") as f:
            f.write(tokenizer.sp_model.serialized_model_proto())
    
    # Create consolidated state dict
    logger.info("Creating consolidated state dict")
    consolidated_state_dict = {}
    
    # Token embeddings
    consolidated_state_dict['tok_embeddings.weight'] = model.model.embed_tokens.weight
    
    # Norm
    consolidated_state_dict['norm.weight'] = model.model.norm.weight
    
    # Output
    consolidated_state_dict['output.weight'] = model.lm_head.weight
    
    # Process all layers
    for i in range(model.config.num_hidden_layers):
        prefix = f'layers.{i}.'
        
        # Attention norms
        consolidated_state_dict[f'{prefix}attention_norm.weight'] = model.model.layers[i].input_layernorm.weight
        
        # Attention weights
        consolidated_state_dict[f'{prefix}attention.wq.weight'] = model.model.layers[i].self_attn.q_proj.weight
        consolidated_state_dict[f'{prefix}attention.wk.weight'] = model.model.layers[i].self_attn.k_proj.weight
        consolidated_state_dict[f'{prefix}attention.wv.weight'] = model.model.layers[i].self_attn.v_proj.weight
        consolidated_state_dict[f'{prefix}attention.wo.weight'] = model.model.layers[i].self_attn.o_proj.weight
        
        # FFN norms
        consolidated_state_dict[f'{prefix}ffn_norm.weight'] = model.model.layers[i].post_attention_layernorm.weight
        
        # Feed forward weights
        consolidated_state_dict[f'{prefix}feed_forward.w1.weight'] = model.model.layers[i].mlp.gate_proj.weight
        consolidated_state_dict[f'{prefix}feed_forward.w2.weight'] = model.model.layers[i].mlp.up_proj.weight
        consolidated_state_dict[f'{prefix}feed_forward.w3.weight'] = model.model.layers[i].mlp.down_proj.weight
    
    # Save consolidated weights
    consolidated_path = os.path.join(output_dir, "consolidated.00.pth")
    logger.info(f"Saving consolidated weights to {consolidated_path}")
    torch.save(consolidated_state_dict, consolidated_path)
    
    # Create and save model configuration
    config = {
        'dim': model.config.hidden_size,
        'n_layers': model.config.num_hidden_layers,
        'n_heads': model.config.num_attention_heads,
        'n_kv_heads': model.config.num_key_value_heads,
        'hidden_dim': model.config.intermediate_size,
        'vocab_size': model.config.vocab_size,
        'max_seq_len': model.config.max_position_embeddings,
    }
    
    logger.info(f"Model configuration: {config}")
    logger.info("Conversion complete!")
    
    return consolidated_path

def main():
    parser = argparse.ArgumentParser(description="Convert Ministral-8B-Instruct-2410 model from HuggingFace to Tenstorrent format")
    parser.add_argument("--model-dir", type=str, default=os.environ.get("MINISTRAL_CKPT_DIR", "/mnt/MLPerf/tt_dnn-models/Mistral/ministral-8b-instruct-2410/"), 
                        help="Directory containing Hugging Face model")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Directory to save converted model (defaults to model-dir)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        raise ValueError(f"Model directory {args.model_dir} does not exist")
    
    convert_hf_mistral_to_tenstorrent(args.model_dir, args.output_dir)

if __name__ == "__main__":
    main()
