#!/usr/bin/env python3
"""
Memory-optimized download script for Ministral-8B model.
Uses streaming downloads and chunked processing to minimize RAM usage.
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from memory_efficient_loader import MemoryOptimizedLoader, check_system_resources

# Configure logging
logging.basicConfig(
    level=os.environ.get('TT_METAL_LOGGER_LEVEL', 'INFO'),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("memory-optimized-download")

def get_model_cache_path():
    """Get the model cache path with environment detection."""
    possible_paths = [
        Path(os.environ.get("MODEL_CACHE_PATH", "")),
        Path("/workspace/tt_models/ministral8b"),      # Koyeb runtime
        Path("/builder/workspace/tt_models/ministral8b"),  # Koyeb build
        Path("/app/tt_models/ministral8b"),           # Container
        Path("/workspaces/tt-metal/models/demos/wormhole/ministral8b/weights"),  # Dev
    ]
    
    # Use the first valid path or create default
    cache_path = next((p for p in possible_paths if p and str(p) != ""), 
                     Path(os.getcwd()) / "weights")
    
    cache_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Using model cache path: {cache_path}")
    return cache_path

def download_ministral_model_optimized():
    """Download Ministral-8B model with memory optimization."""
    
    # Check system resources first
    resources = check_system_resources()
    available_ram = resources.get('available_ram_gb', 0)
    
    if available_ram < 8:
        logger.error(f"Insufficient RAM: {available_ram:.2f}GB available, minimum 8GB required")
        return False
    elif available_ram < 16:
        logger.warning(f"Low RAM: {available_ram:.2f}GB available, consider using minimal mode")
        chunk_size_mb = 128  # Smaller chunks for low memory
    else:
        chunk_size_mb = 512  # Normal chunk size
    
    cache_path = get_model_cache_path()
    
    # Initialize memory-efficient loader
    loader = MemoryOptimizedLoader(str(cache_path), chunk_size_mb=chunk_size_mb)
    
    # Atomic lock file creation with wait/retry logic
    lock_file = cache_path / "downloading.lock"
    max_wait_time = 1800  # 30 minutes max wait
    retry_interval = 30   # Check every 30 seconds
    start_time = time.time()
    
    while True:
        try:
            # Atomic lock creation - fails if file already exists
            with open(lock_file, 'x') as f:
                f.write(f"Memory-optimized download started at: {time.ctime()}\n")
                f.write(f"PID: {os.getpid()}\n")
            logger.info("Created download lock file atomically")
            break
        except FileExistsError:
            # Lock file exists, check if it's stale
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                logger.error(f"Download lock timeout after {max_wait_time}s, removing stale lock")
                try:
                    lock_file.unlink()
                    continue  # Try to create lock again
                except Exception as e:
                    logger.error(f"Could not remove stale lock: {e}")
                    return False
            
            # Check if lock file is stale (older than 2 hours)
            try:
                lock_age = time.time() - lock_file.stat().st_mtime
                if lock_age > 7200:  # 2 hours
                    logger.warning(f"Removing stale lock file (age: {lock_age/3600:.1f}h)")
                    lock_file.unlink()
                    continue  # Try to create lock again
            except Exception:
                pass  # Lock file might have been removed by another process
            
            logger.info(f"Another download in progress, waiting {retry_interval}s... (elapsed: {elapsed:.0f}s)")
            time.sleep(retry_interval)
        except Exception as e:
            logger.error(f"Could not create lock file: {e}")
            return False
    
    model_name = "mistralai/Ministral-8B-Instruct-2410"
    hf_token = os.environ.get("HF_TOKEN")
    
    try:
        # Files to download (prioritized by importance)
        essential_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        
        weight_files = [
            "model.safetensors.index.json",  # Check if sharded
            "model.safetensors",             # Single file
            "pytorch_model.bin.index.json", # Check if sharded  
            "pytorch_model.bin",             # Single file
        ]
        
        # Download essential files first (small, low memory impact)
        logger.info("=== Downloading essential configuration files ===")
        for filename in essential_files:
            file_url = f"https://huggingface.co/{model_name}/resolve/main/{filename}"
            dest_file = cache_path / filename
            
            try:
                loader.stream_download_file(file_url, dest_file)
                logger.info(f"✓ Downloaded {filename}")
            except Exception as e:
                logger.warning(f"⚠ Could not download {filename}: {e}")
                continue
        
        # Determine weight file structure
        logger.info("=== Determining model weight structure ===")
        weight_files_to_download = []
        
        # Check for safetensors index
        safetensors_index = cache_path / "model.safetensors.index.json"
        if safetensors_index.exists():
            with open(safetensors_index, 'r') as f:
                index_data = json.load(f)
                weight_files_to_download = list(set(index_data.get("weight_map", {}).values()))
                logger.info(f"Found safetensors sharded model with {len(weight_files_to_download)} files")
        else:
            # Check for single safetensors file
            single_safetensors_url = f"https://huggingface.co/{model_name}/resolve/main/model.safetensors"
            try:
                import requests
                response = requests.head(single_safetensors_url, timeout=10)
                if response.status_code == 200:
                    weight_files_to_download = ["model.safetensors"]
                    logger.info("Found single safetensors file")
            except:
                pass
        
        # Fallback to PyTorch files if no safetensors
        if not weight_files_to_download:
            pytorch_index = cache_path / "pytorch_model.bin.index.json"
            if pytorch_index.exists():
                with open(pytorch_index, 'r') as f:
                    index_data = json.load(f)
                    weight_files_to_download = list(set(index_data.get("weight_map", {}).values()))
                    logger.info(f"Found PyTorch sharded model with {len(weight_files_to_download)} files")
            else:
                weight_files_to_download = ["pytorch_model.bin"]
                logger.info("Defaulting to single PyTorch file")
        
        # Download weight files with memory monitoring
        logger.info(f"=== Downloading {len(weight_files_to_download)} weight files ===")
        downloaded_weight_files = []
        
        for i, filename in enumerate(weight_files_to_download, 1):
            logger.info(f"Downloading weight file {i}/{len(weight_files_to_download)}: {filename}")
            
            file_url = f"https://huggingface.co/{model_name}/resolve/main/{filename}"
            dest_file = cache_path / filename
            
            try:
                # Check available memory before each download
                current_resources = check_system_resources()
                if current_resources.get('available_ram_gb', 0) < 4:
                    logger.warning("Low memory detected, forcing garbage collection")
                    import gc
                    gc.collect()
                
                loader.stream_download_file(file_url, dest_file)
                downloaded_weight_files.append(dest_file)
                logger.info(f"✓ Downloaded {filename} ({i}/{len(weight_files_to_download)})")
                
            except Exception as e:
                logger.error(f"✗ Failed to download {filename}: {e}")
                # Continue with other files even if one fails
                continue
        
        if not downloaded_weight_files:
            logger.error("✗ No weight files were downloaded successfully")
            return False
        
        # Process weights with memory optimization
        logger.info("=== Processing weights with memory optimization ===")
        consolidated_path = cache_path / "consolidated.bin"
        
        # Estimate memory requirements
        total_size_gb = sum(f.stat().st_size for f in downloaded_weight_files) / (1024**3)
        estimates = {
            'total_weights_gb': total_size_gb,
            'estimated_processing_gb': total_size_gb * 1.5
        }
        
        logger.info(f"Total weight files: {total_size_gb:.2f}GB")
        logger.info(f"Estimated processing memory: {estimates['estimated_processing_gb']:.2f}GB")
        
        # Check if we have enough memory for processing
        if available_ram < estimates['estimated_processing_gb']:
            logger.warning("Insufficient RAM for optimal processing, using minimal chunked mode")
            # Use smaller chunks and more aggressive cleanup
            loader.chunk_size_bytes = 128 * 1024 * 1024  # 128MB chunks
        
        success = loader.process_weights_chunked(
            model_files=[str(f) for f in downloaded_weight_files],
            output_path=consolidated_path
        )
        
        if not success:
            logger.error("✗ Failed to process weights")
            return False
        
        # Create model configuration
        logger.info("=== Creating model configuration ===")
        config_path = cache_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create params.json for tt-metal compatibility
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
            
            params_path = cache_path / "params.json"
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2)
            logger.info(f"✓ Created params.json")
        
        # Verify essential files exist
        required_files = [
            "consolidated.bin",
            "config.json", 
            "tokenizer.json",
            "params.json"
        ]
        
        missing_files = [f for f in required_files if not (cache_path / f).exists()]
        if missing_files:
            logger.error(f"✗ Missing required files: {missing_files}")
            return False
        
        # Clean up temporary and original weight files if consolidation succeeded
        if consolidated_path.exists():
            logger.info("=== Cleaning up temporary files ===")
            for weight_file in downloaded_weight_files:
                try:
                    if weight_file.name != "consolidated.bin":
                        weight_file.unlink()
                        logger.info(f"Cleaned up {weight_file.name}")
                except Exception as e:
                    logger.warning(f"Could not clean up {weight_file.name}: {e}")
        
        # Final memory and disk usage report
        final_resources = check_system_resources()
        final_size_gb = sum(f.stat().st_size for f in cache_path.glob("*") if f.is_file()) / (1024**3)
        
        logger.info("=== Download Summary ===")
        logger.info(f"✓ Model downloaded successfully to {cache_path}")
        logger.info(f"✓ Total cache size: {final_size_gb:.2f}GB")
        logger.info(f"✓ Final available RAM: {final_resources.get('available_ram_gb', 0):.2f}GB")
        logger.info(f"✓ Ready for TTNN inference")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Download failed: {e}", exc_info=True)
        return False
    finally:
        # Always clean up lock file in finally block
        if 'lock_file' in locals() and lock_file.exists():
            try:
                lock_file.unlink()
                logger.info("Removed download lock file in cleanup")
            except Exception as e:
                logger.warning(f"Could not remove lock file in cleanup: {e}")

if __name__ == "__main__":
    print("=== Memory-Optimized Ministral-8B Download ===")
    
    # Check initial system state
    initial_resources = check_system_resources()
    logger.info(f"Starting with {initial_resources.get('available_ram_gb', 0):.2f}GB available RAM")
    
    # Run optimized download
    success = download_ministral_model_optimized()
    
    if success:
        print("\n✓ Memory-optimized download completed successfully!")
        print("You can now run the server with optimized model loading")
    else:
        print("\n✗ Memory-optimized download failed!")
        sys.exit(1)

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from memory_efficient_loader import MemoryOptimizedLoader, check_system_resources

# Configure logging
logging.basicConfig(
    level=os.environ.get('TT_METAL_LOGGER_LEVEL', 'INFO'),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("memory-optimized-download")

def get_model_cache_path():
    """Get the model cache path with environment detection."""
    possible_paths = [
        Path(os.environ.get("MODEL_CACHE_PATH", "")),
        Path("/workspace/tt_models/ministral8b"),      # Koyeb runtime
        Path("/builder/workspace/tt_models/ministral8b"),  # Koyeb build
        Path("/app/tt_models/ministral8b"),           # Container
        Path("/workspaces/tt-metal/models/demos/wormhole/ministral8b/weights"),  # Dev
    ]
    
    # Use the first valid path or create default
    cache_path = next((p for p in possible_paths if p and str(p) != ""), 
                     Path(os.getcwd()) / "weights")
    
    cache_path.mkdir(exist_ok=True, parents=True)
    logger.info(f"Using model cache path: {cache_path}")
    return cache_path

def download_ministral_model_optimized():
    """Download Ministral-8B model with memory optimization."""
    
    # Check system resources first
    resources = check_system_resources()
    available_ram = resources.get('available_ram_gb', 0)
    
    if available_ram < 8:
        logger.error(f"Insufficient RAM: {available_ram:.2f}GB available, minimum 8GB required")
        return False
    elif available_ram < 16:
        logger.warning(f"Low RAM: {available_ram:.2f}GB available, consider using minimal mode")
        chunk_size_mb = 128  # Smaller chunks for low memory
    else:
        chunk_size_mb = 512  # Normal chunk size
    
    cache_path = get_model_cache_path()
    
    # Initialize memory-efficient loader
    try:
        loader = MemoryOptimizedLoader(str(cache_path), chunk_size_mb=chunk_size_mb)
    except TypeError:
        logger.warning("chunk_size_mb not supported—falling back to default loader args")
        loader = MemoryOptimizedLoader(str(cache_path))
    
    # Atomic lock file creation with wait/retry logic
    lock_file = cache_path / "downloading.lock"
    max_wait_time = 1800  # 30 minutes max wait
    retry_interval = 30   # Check every 30 seconds
    start_time = time.time()
    
    while True:
        try:
            # Atomic lock creation - fails if file already exists
            with open(lock_file, 'x') as f:
                f.write(f"Memory-optimized download started at: {time.ctime()}\n")
                f.write(f"PID: {os.getpid()}\n")
            logger.info("Created download lock file atomically")
            break
        except FileExistsError:
            # Lock file exists, check if it's stale
            elapsed = time.time() - start_time
            if elapsed > max_wait_time:
                logger.error(f"Download lock timeout after {max_wait_time}s, removing stale lock")
                try:
                    lock_file.unlink()
                    continue  # Try to create lock again
                except Exception as e:
                    logger.error(f"Could not remove stale lock: {e}")
                    return False
            
            # Check if lock file is stale (older than 2 hours)
            try:
                lock_age = time.time() - lock_file.stat().st_mtime
                if lock_age > 7200:  # 2 hours
                    logger.warning(f"Removing stale lock file (age: {lock_age/3600:.1f}h)")
                    lock_file.unlink()
                    continue  # Try to create lock again
            except Exception:
                pass  # Lock file might have been removed by another process
            
            logger.info(f"Another download in progress, waiting {retry_interval}s... (elapsed: {elapsed:.0f}s)")
            time.sleep(retry_interval)
        except Exception as e:
            logger.error(f"Could not create lock file: {e}")
            return False
    
    model_name = "mistralai/Ministral-8B-Instruct-2410"
    hf_token = os.environ.get("HF_TOKEN")
    
    try:
        # Files to download (prioritized by importance)
        essential_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        
        weight_files = [
            "model.safetensors.index.json",  # Check if sharded
            "model.safetensors",             # Single file
            "pytorch_model.bin.index.json", # Check if sharded  
            "pytorch_model.bin",             # Single file
        ]
        
        # Download essential files first (small, low memory impact)
        logger.info("=== Downloading essential configuration files ===")
        for filename in essential_files:
            file_url = f"https://huggingface.co/{model_name}/resolve/main/{filename}"
            dest_file = cache_path / filename
            
            try:
                loader.stream_download_file(file_url, dest_file)
                logger.info(f"✓ Downloaded {filename}")
            except Exception as e:
                logger.warning(f"⚠ Could not download {filename}: {e}")
                continue
        
        # Determine weight file structure
        logger.info("=== Determining model weight structure ===")
        weight_files_to_download = []
        
        # Check for safetensors index
        safetensors_index = cache_path / "model.safetensors.index.json"
        if safetensors_index.exists():
            with open(safetensors_index, 'r') as f:
                index_data = json.load(f)
                weight_files_to_download = list(set(index_data.get("weight_map", {}).values()))
                logger.info(f"Found safetensors sharded model with {len(weight_files_to_download)} files")
        else:
            # Check for single safetensors file
            single_safetensors_url = f"https://huggingface.co/{model_name}/resolve/main/model.safetensors"
            try:
                import requests
                response = requests.head(single_safetensors_url, timeout=10)
                if response.status_code == 200:
                    weight_files_to_download = ["model.safetensors"]
                    logger.info("Found single safetensors file")
            except:
                pass
        
        # Fallback to PyTorch files if no safetensors
        if not weight_files_to_download:
            pytorch_index = cache_path / "pytorch_model.bin.index.json"
            if pytorch_index.exists():
                with open(pytorch_index, 'r') as f:
                    index_data = json.load(f)
                    weight_files_to_download = list(set(index_data.get("weight_map", {}).values()))
                    logger.info(f"Found PyTorch sharded model with {len(weight_files_to_download)} files")
            else:
                weight_files_to_download = ["pytorch_model.bin"]
                logger.info("Defaulting to single PyTorch file")
        
        # Download weight files with memory monitoring
        logger.info(f"=== Downloading {len(weight_files_to_download)} weight files ===")
        downloaded_weight_files = []
        
        for i, filename in enumerate(weight_files_to_download, 1):
            logger.info(f"Downloading weight file {i}/{len(weight_files_to_download)}: {filename}")
            
            file_url = f"https://huggingface.co/{model_name}/resolve/main/{filename}"
            dest_file = cache_path / filename
            
            try:
                # Check available memory before each download
                current_resources = check_system_resources()
                if current_resources.get('available_ram_gb', 0) < 4:
                    logger.warning("Low memory detected, forcing garbage collection")
                    import gc
                    gc.collect()
                
                loader.stream_download_file(file_url, dest_file)
                downloaded_weight_files.append(dest_file)
                logger.info(f"✓ Downloaded {filename} ({i}/{len(weight_files_to_download)})")
                
            except Exception as e:
                logger.error(f"✗ Failed to download {filename}: {e}")
                # Continue with other files even if one fails
                continue
        
        if not downloaded_weight_files:
            logger.error("✗ No weight files were downloaded successfully")
            return False
        
        # Process weights with memory optimization
        logger.info("=== Processing weights with memory optimization ===")
        consolidated_path = cache_path / "consolidated.bin"
        
        # Estimate memory requirements
        total_size_gb = sum(f.stat().st_size for f in downloaded_weight_files) / (1024**3)
        estimates = {
            'total_weights_gb': total_size_gb,
            'estimated_processing_gb': total_size_gb * 1.5
        }
        
        logger.info(f"Total weight files: {total_size_gb:.2f}GB")
        logger.info(f"Estimated processing memory: {estimates['estimated_processing_gb']:.2f}GB")
        
        # Check if we have enough memory for processing
        if available_ram < estimates['estimated_processing_gb']:
            logger.warning("Insufficient RAM for optimal processing, using minimal chunked mode")
            # Use smaller chunks and more aggressive cleanup
            loader.chunk_size_bytes = 128 * 1024 * 1024  # 128MB chunks
        
        success = loader.process_weights_chunked(
            model_files=[str(f) for f in downloaded_weight_files],
            output_path=consolidated_path
        )
        
        if not success:
            logger.error("✗ Failed to process weights")
            return False
        
        # Create model configuration
        logger.info("=== Creating model configuration ===")
        config_path = cache_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Create params.json for tt-metal compatibility
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
            
            params_path = cache_path / "params.json"
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2)
            logger.info(f"✓ Created params.json")
        
        # Verify essential files exist
        required_files = [
            "consolidated.bin",
            "config.json", 
            "tokenizer.json",
            "params.json"
        ]
        
        missing_files = [f for f in required_files if not (cache_path / f).exists()]
        if missing_files:
            logger.error(f"✗ Missing required files: {missing_files}")
            return False
        
        # Clean up temporary and original weight files if consolidation succeeded
        if consolidated_path.exists():
            logger.info("=== Cleaning up temporary files ===")
            for weight_file in downloaded_weight_files:
                try:
                    if weight_file.name != "consolidated.bin":
                        weight_file.unlink()
                        logger.info(f"Cleaned up {weight_file.name}")
                except Exception as e:
                    logger.warning(f"Could not clean up {weight_file.name}: {e}")
        
        # Final memory and disk usage report
        final_resources = check_system_resources()
        final_size_gb = sum(f.stat().st_size for f in cache_path.glob("*") if f.is_file()) / (1024**3)
        
        logger.info("=== Download Summary ===")
        logger.info(f"✓ Model downloaded successfully to {cache_path}")
        logger.info(f"✓ Total cache size: {final_size_gb:.2f}GB")
        logger.info(f"✓ Final available RAM: {final_resources.get('available_ram_gb', 0):.2f}GB")
        logger.info(f"✓ Ready for TTNN inference")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Download failed: {e}", exc_info=True)
        return False
    finally:
        # Always clean up lock file in finally block
        if 'lock_file' in locals() and lock_file.exists():
            try:
                lock_file.unlink()
                logger.info("Removed download lock file in cleanup")
            except Exception as e:
                logger.warning(f"Could not remove lock file in cleanup: {e}")

if __name__ == "__main__":
    print("=== Memory-Optimized Ministral-8B Download ===")
    
    # Check initial system state
    initial_resources = check_system_resources()
    logger.info(f"Starting with {initial_resources.get('available_ram_gb', 0):.2f}GB available RAM")
    
    # Run optimized download
    success = download_ministral_model_optimized()
    
    if success:
        print("\n✓ Memory-optimized download completed successfully!")
        print("You can now run the server with optimized model loading")
    else:
        print("\n✗ Memory-optimized download failed!")
        sys.exit(1)
