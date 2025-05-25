#!/usr/bin/env python3
"""
Download utility for Ministral-8B model with resumable downloads.
Automatically detects environment and handles download for both development and Koyeb environments.
"""

import os
import sys
import json
import logging
import time
import requests
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import login, HfApi, HfFolder, snapshot_download
import torch

# Configure logging
logging.basicConfig(
    level=os.environ.get('TT_METAL_LOGGER_LEVEL', 'INFO'),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("download-model")

class DownloadProgressBar(tqdm):
    """Progress bar for downloads with speed and ETA"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file_with_resume(url, destination, token=None):
    """Download a file with resumable downloads and progress bar"""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    # Check if file exists and get its size
    if destination.exists():
        downloaded_size = destination.stat().st_size
        headers["Range"] = f"bytes={downloaded_size}-"
    else:
        downloaded_size = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
    
    with requests.get(url, headers=headers, stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        
        # Get total size for progress bar
        total_size = int(response.headers.get('content-length', 0)) + downloaded_size
        
        # Check if download is partial
        mode = 'ab' if downloaded_size > 0 else 'wb'
        
        with open(destination, mode) as f, \
             DownloadProgressBar(unit='B', unit_scale=True, unit_divisor=1024,
                              miniters=1, desc=destination.name,
                              total=total_size, initial=downloaded_size) as pbar:
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    size = f.write(chunk)
                    pbar.update(size)

# Configure logging
logging.basicConfig(
    level=os.environ.get('TT_METAL_LOGGER_LEVEL', 'INFO'),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("download-model")

def download_ministral_model():
    """Download Ministral-8B-Instruct-2410 model from Hugging Face with resumable downloads."""
    
    # Detect environment and set up paths
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
    
    # Create a lock file to indicate download is in progress
    lock_file = weights_dir / "downloading.lock"
    if lock_file.exists():
        logger.info("Previous download in progress, checking status...")
    else:
        try:
            with open(lock_file, 'w') as f:
                f.write(f"Download started at: {time.ctime()}")
            logger.info("Created download lock file")
        except Exception as e:
            logger.warning(f"Could not create lock file: {e}")
    
    # Hugging Face token - use environment variable only
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable is required")
        if lock_file.exists():
            try:
                lock_file.unlink()
            except:
                pass
        return False
    
    logger.info("Logging into Hugging Face...")
    try:
        login(token=hf_token)
        logger.info("✓ Successfully logged into Hugging Face")
    except Exception as e:
        logger.error(f"✗ Failed to login to Hugging Face: {e}")
        return False
    
    # Initialize HF API
    api = HfApi(token=hf_token)
    model_name = os.environ.get("MODEL_NAME", "mistralai/Ministral-8B-Instruct-2410")
    
    try:
        logger.info(f"Fetching model files list for {model_name}...")
        
        # Get all files in the model repo
        model_info = api.model_info(model_name, token=hf_token)
        files_to_download = []
        
        # Filter out unnecessary files and patterns
        ignore_patterns = ["*.md", "*.bin.index.json", "*.h5", "*.ot", "*.msgpack"]
        
        for file_info in model_info.siblings:
            filename = file_info.rfilename
            if not any(filename.endswith(p.replace("*", "")) for p in ignore_patterns):
                files_to_download.append(filename)
        
        if not files_to_download:
            logger.error("No files to download found in the model repository")
            return False
        
        logger.info(f"Found {len(files_to_download)} files to download")
        
        # Download each file with resume support
        for filename in files_to_download:
            file_url = f"https://huggingface.co/{model_name}/resolve/main/{filename}"
            dest_file = weights_dir / filename
            
            # Create parent directories if they don't exist
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading {filename}...")
            try:
                download_file_with_resume(file_url, dest_file, hf_token)
                logger.info(f"✓ Downloaded {filename}")
            except Exception as e:
                logger.error(f"✗ Failed to download {filename}: {e}")
                # Continue with other files even if one fails
                continue
        
        # Verify key files exist
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        
        # Check for either safetensors or pytorch model files
        has_weights = any(weights_dir.glob("*.safetensors")) or any(weights_dir.glob("pytorch_model*.bin"))
        
        missing_files = []
        for file in required_files:
            if not (weights_dir / file).exists() and not list(weights_dir.glob(f"*{file}")):
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"⚠ Warning: Missing required files: {missing_files}")
        if not has_weights:
            logger.warning("⚠ Warning: No model weight files found (expected .safetensors or .bin files)")
        
        if not missing_files and has_weights:
            logger.info("✓ All required files downloaded successfully")
            
            # Convert to consolidated format if needed
            conversion_success = convert_to_consolidated_format(weights_dir)
            
            # Clean up lock file if download was successful
            if lock_file.exists():
                try:
                    lock_file.unlink()
                    logger.info("Removed download lock file")
                except Exception as e:
                    logger.warning(f"Could not remove lock file: {e}")
            
            return conversion_success
            
        return False
        
    except Exception as e:
        logger.error(f"✗ Error during download: {str(e)}", exc_info=True)
        # Clean up lock file on error
        if lock_file.exists():
            try:
                lock_file.unlink()
                logger.info("Removed download lock file after error")
            except Exception as e2:
                logger.warning(f"Could not remove lock file after error: {e2}")
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
