#!/usr/bin/env python3
"""
Download utility for Ministral-8B model.
Automatically detects environment and handles download for both development and Koyeb environments.
"""

import os
import sys
import json
import logging
from pathlib import Path
from huggingface_hub import snapshot_download, login
import torch

# Configure logging
logging.basicConfig(
    level=os.environ.get('TT_METAL_LOGGER_LEVEL', 'INFO'),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("download-model")

def download_ministral_model():
    """Download Ministral-8B-Instruct-2410 model from Hugging Face."""
    
    # Detect environment and set up paths
    # Check various possible paths based on environment
    possible_roots = [
        Path(os.environ.get("MODEL_CACHE_PATH", "")),  # Use environment variable if set
        Path("/workspace/tt_models/ministral8b"),      # Koyeb runtime path
        Path("/builder/workspace/tt_models/ministral8b"),  # Koyeb build path
        Path("/app/tt_models/ministral8b"),           # Container path
        Path("/workspaces/tt-metal/models/demos/wormhole/ministral8b/weights"),  # Dev environment
    ]
    
    # Use the first valid path or create a default one
    weights_dir = next((p for p in possible_roots if p and str(p) != ""), 
                    Path(os.getcwd()) / "weights")
    
    # Create directory if it doesn't exist
    weights_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Using model cache path: {weights_dir}")
    
    # Hugging Face token - use environment variable only
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable is required")
        return False
    
    logger.info("Logging into Hugging Face...")
    try:
        login(token=hf_token)
        logger.info("✓ Successfully logged into Hugging Face")
    except Exception as e:
        logger.error(f"✗ Failed to login to Hugging Face: {e}")
        return False
    
    # Download model
    model_name = os.environ.get("MODEL_NAME", "mistralai/Ministral-8B-Instruct-2410")
    logger.info(f"Downloading {model_name} to {weights_dir}...")
    
    try:
        # Download the model files
        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_dir=weights_dir,
            token=hf_token,
            cache_dir=None,  # Don't use cache, download directly to target
            local_files_only=False,  # Force download
            ignore_patterns=["*.md", "*.pt"],  # Skip unnecessary files
        )
        logger.info(f"✓ Model downloaded successfully to {downloaded_path}")
        
        # Verify key files exist
        required_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "pytorch_model.bin"
        ]
        
        # Show available files for debugging
        logger.info(f"Files in model directory: {[f.name for f in weights_dir.iterdir()]}")
        
        missing_files = []
        for file in required_files:
            if not (weights_dir / file).exists() and not list(weights_dir.glob(f"*{file}")):
                # Check for safetensors format
                if file == "pytorch_model.bin":
                    safetensor_files = list(weights_dir.glob("*.safetensors"))
                    if not safetensor_files:
                        missing_files.append(file)
                else:
                    missing_files.append(file)
        
        if missing_files:
            logger.warning(f"⚠ Warning: Missing files: {missing_files}")
        else:
            print("✓ All required files downloaded successfully")
            
        # Convert to consolidated format if needed
        convert_to_consolidated_format(weights_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        return False

def convert_to_consolidated_format(weights_dir):
    """Convert model weights to consolidated format expected by tt-metal."""
    
    print("Converting to consolidated format...")
    
    try:
        # Check if we have safetensors files
        safetensor_files = list(weights_dir.glob("*.safetensors"))
        pytorch_files = list(weights_dir.glob("pytorch_model*.bin"))
        
        if safetensor_files:
            print("Found safetensors files, converting...")
            from safetensors.torch import load_file
            
            # Load all safetensor files
            state_dict = {}
            for file in safetensor_files:
                print(f"Loading {file.name}...")
                file_state = load_file(file)
                state_dict.update(file_state)
                
        elif pytorch_files:
            print("Found PyTorch files, converting...")
            
            # Load all pytorch files
            state_dict = {}
            for file in pytorch_files:
                print(f"Loading {file.name}...")
                file_state = torch.load(file, map_location="cpu")
                if isinstance(file_state, dict):
                    state_dict.update(file_state)
                else:
                    # Single file case
                    state_dict = file_state
                    break
        else:
            print("No model weight files found!")
            return False
            
        # Save as consolidated.bin
        consolidated_path = weights_dir / "consolidated.bin"
        print(f"Saving consolidated weights to {consolidated_path}...")
        torch.save(state_dict, consolidated_path)
        
        # Also save a params.json file with model configuration
        config_path = weights_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create params.json in the format expected by tt-metal
            params = {
                "dim": config.get("hidden_size", 4096),
                "n_layers": config.get("num_hidden_layers", 32),
                "n_heads": config.get("num_attention_heads", 32),
                "n_kv_heads": config.get("num_key_value_heads", 8),
                "vocab_size": config.get("vocab_size", 131072),
                "multiple_of": 256,
                "ffn_dim_multiplier": None,
                "norm_eps": config.get("rms_norm_eps", 1e-5),
                "rope_theta": config.get("rope_theta", 1000000.0),
                "sliding_window": config.get("sliding_window", 4096),
                "max_seq_len": config.get("max_position_embeddings", 32768)
            }
            
            params_path = weights_dir / "params.json"
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2)
            print(f"✓ Saved model parameters to {params_path}")
        
        print("✓ Model conversion completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Failed to convert model: {e}")
        return False

if __name__ == "__main__":
    print("=== Ministral-8B Model Download Script ===")
    success = download_ministral_model()
    
    if success:
        print("\n✓ Model download and setup completed successfully!")
        print("You can now run the demo with: python demo/demo_with_prefill.py")
    else:
        print("\n✗ Model download failed!")
        sys.exit(1)
