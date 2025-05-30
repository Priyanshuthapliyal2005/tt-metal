#!/usr/bin/env python3
"""
Memory-efficient model loader for Ministral-8B with TTNN hardware optimization.
Implements streaming downloads, lazy loading, and chunked processing to minimize RAM usage.
"""

import os
import gc
import json
import time
import torch
import logging
import psutil
from pathlib import Path
from typing import Dict, Optional, Any, Generator, Tuple
from contextlib import contextmanager
import tempfile
import safetensors.torch
from huggingface_hub import hf_hub_download, snapshot_download
import requests
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryOptimizedLoader:
    """Memory-efficient model loader optimized for TTNN hardware and large models."""
    
    def __init__(self, cache_dir: str = "./model_cache", max_memory_gb: float = 16.0, chunk_size_mb: int = 256):
        """
        Initialize the memory-optimized loader.
        
        Args:
            cache_dir: Directory to store model cache
            max_memory_gb: Maximum memory limit in GB (default 16GB)
            chunk_size_mb: Chunk size in MB for downloads and processing (default 256MB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_memory_gb = max_memory_gb
        self.chunk_size_mb = chunk_size_mb
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.metrics = {
            'download_speed': [],
            'loading_time': [],
            'memory_peak': 0,
            'device_utilization': {}
        }
        
        # Multi-device optimization
        self.device_config = self._detect_ttnn_devices()
    
    def _detect_ttnn_devices(self) -> Dict[str, Any]:
        """Detect and configure TTNN devices for optimal utilization."""
        try:
            import ttnn
            num_devices = ttnn.get_num_devices()
            device_config = {
                'num_devices': num_devices,
                'devices': [],
                'memory_per_device': {},
                'optimal_sharding': num_devices > 1
            }
            
            for i in range(num_devices):
                try:
                    device = ttnn.open_device(device_id=i)
                    device_config['devices'].append(device)
                    # Get device memory info if available
                    device_config['memory_per_device'][i] = self._get_device_memory(device)
                except Exception as e:
                    self.logger.warning(f"Could not initialize TTNN device {i}: {e}")
            
            self.logger.info(f"TTNN Multi-device setup: {num_devices} devices detected")
            return device_config
            
        except ImportError:
            self.logger.warning("TTNN not available, using CPU fallback")
            return {'num_devices': 0, 'devices': [], 'memory_per_device': {}, 'optimal_sharding': False}
    
    def _get_device_memory(self, device) -> Dict[str, float]:
        """Get memory information for a TTNN device."""
        try:
            # This is a placeholder - replace with actual TTNN memory query
            return {'total_gb': 8.0, 'available_gb': 7.5}  # Default values
        except:
            return {'total_gb': 8.0, 'available_gb': 7.5}
    
    @contextmanager
    def performance_monitor(self, operation_name: str):
        """Enhanced performance monitoring with multi-device tracking."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)
        
        # Monitor device utilization if TTNN available
        device_start_metrics = {}
        for i, device in enumerate(self.device_config['devices']):
            device_start_metrics[i] = self._get_device_memory(device)
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**3)
            peak_memory = max(start_memory, end_memory)
            
            # Update metrics
            duration = end_time - start_time
            self.metrics['loading_time'].append(duration)
            self.metrics['memory_peak'] = max(self.metrics['memory_peak'], peak_memory)
            
            # Device utilization tracking
            for i, device in enumerate(self.device_config['devices']):
                end_metrics = self._get_device_memory(device)
                if i not in self.metrics['device_utilization']:
                    self.metrics['device_utilization'][i] = []
                
                memory_used = device_start_metrics.get(i, {}).get('available_gb', 0) - end_metrics.get('available_gb', 0)
                self.metrics['device_utilization'][i].append(memory_used)
            
            self.logger.info(f"{operation_name} completed in {duration:.2f}s, peak memory: {peak_memory:.2f}GB")
    
    def stream_download_file(self, url: str, destination: Path, resume: bool = True):
        """
        Stream download a file with resumable capability and memory efficiency.
        
        Args:
            url: URL to download from
            destination: Local file path
            resume: Whether to resume partial downloads
        """
        import requests
        
        # Check if file already exists and is complete
        if destination.exists():
            try:
                response = requests.head(url, timeout=10)
                expected_size = int(response.headers.get('content-length', 0))
                if destination.stat().st_size == expected_size:
                    logger.info(f"File {destination.name} already complete, skipping")
                    return
            except Exception as e:
                logger.warning(f"Could not verify file completeness: {e}")
        
        downloaded_size = destination.stat().st_size if destination.exists() and resume else 0
        
        headers = {}
        if downloaded_size > 0:
            headers['Range'] = f'bytes={downloaded_size}-'
            logger.info(f"Resuming download from byte {downloaded_size}")
        
        with self.performance_monitor(f"downloading {destination.name}"):
            try:
                response = requests.get(url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0)) + downloaded_size
                
                mode = 'ab' if downloaded_size > 0 else 'wb'
                chunk_size_bytes = self.chunk_size_mb * 1024 * 1024
                with open(destination, mode) as f:
                    downloaded = downloaded_size
                    for chunk in response.iter_content(chunk_size=min(8192, chunk_size_bytes)):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Log progress every 100MB
                            if downloaded % (100 * 1024 * 1024) < 8192:
                                progress = (downloaded / total_size * 100) if total_size > 0 else 0
                                logger.info(f"Downloaded {downloaded / (1024**3):.2f}GB / "
                                          f"{total_size / (1024**3):.2f}GB ({progress:.1f}%)")
                
                logger.info(f"✓ Downloaded {destination.name} successfully")
                
            except Exception as e:
                logger.error(f"✗ Failed to download {destination.name}: {e}")
                raise
    
    def load_state_dict_streaming(self, file_path: Path) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """
        Stream load state dict tensors one by one to minimize memory usage.
        
        Args:
            file_path: Path to the model file (.bin or .safetensors)
            
        Yields:
            Tuple of (key, tensor) for each parameter
        """
        with self.performance_monitor(f"streaming load {file_path.name}"):
            if file_path.suffix == '.safetensors':
                from safetensors.torch import safe_open
                
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor = f.get_tensor(key)
                        yield key, tensor
                        # Clear reference to help with memory management
                        del tensor
                        gc.collect()
                        
            else:  # .bin file
                # Load the entire state dict - unfortunately PyTorch doesn't support streaming
                # But we can immediately yield and delete to minimize peak memory
                state_dict = torch.load(file_path, map_location="cpu")
                for key, tensor in state_dict.items():
                    yield key, tensor
                    # Remove from dict to free memory
                    del state_dict[key]
                    gc.collect()
                del state_dict
    
    def process_weights_chunked(self, model_files: list, output_path: Path, 
                              target_layers: Optional[list] = None) -> bool:
        """
        Process model weights in chunks to minimize memory usage.
        
        Args:
            model_files: List of model weight files to process
            output_path: Path to save consolidated weights
            target_layers: Specific layers to load (None for all)
            
        Returns:
            True if successful, False otherwise
        """
        with self.performance_monitor("chunked weight processing"):
            try:
                # Create temporary files for different parameter types
                temp_files = {
                    'embeddings': self.cache_dir / "temp" / "embeddings.bin",
                    'layers': self.cache_dir / "temp" / "layers.bin", 
                    'output': self.cache_dir / "temp" / "output.bin",
                    'norm': self.cache_dir / "temp" / "norm.bin"
                }
                
                # Process each file and categorize parameters
                for model_file in model_files:
                    logger.info(f"Processing {model_file}")
                    
                    for key, tensor in self.load_state_dict_streaming(Path(model_file)):
                        # Filter by target layers if specified
                        if target_layers and "layers." in key:
                            layer_num = int(key.split("layers.")[1].split(".")[0])
                            if layer_num not in target_layers:
                                continue
                        
                        # Categorize and save to appropriate temp file
                        if "tok_embeddings" in key:
                            self._append_to_temp_file(temp_files['embeddings'], key, tensor)
                        elif "layers." in key:
                            self._append_to_temp_file(temp_files['layers'], key, tensor)
                        elif "output." in key:
                            self._append_to_temp_file(temp_files['output'], key, tensor)
                        elif "norm." in key:
                            self._append_to_temp_file(temp_files['norm'], key, tensor)
                        
                        # Clear tensor reference
                        del tensor
                        gc.collect()
                
                # Combine temp files into final consolidated file
                self._combine_temp_files(temp_files, output_path)
                
                # Cleanup temp files
                for temp_file in temp_files.values():
                    if temp_file.exists():
                        temp_file.unlink()
                
                logger.info(f"✓ Created consolidated weights at {output_path}")
                return True
                
            except Exception as e:
                logger.error(f"✗ Failed to process weights: {e}")
                return False
    
    def _append_to_temp_file(self, temp_file: Path, key: str, tensor: torch.Tensor):
        """Append a tensor to a temporary file."""
        # Load existing data if file exists
        data = {}
        if temp_file.exists():
            data = torch.load(temp_file, map_location="cpu")
        
        # Add new tensor
        data[key] = tensor
        
        # Save back
        torch.save(data, temp_file)
        del data
    
    def _combine_temp_files(self, temp_files: Dict[str, Path], output_path: Path):
        """Combine temporary files into final consolidated file."""
        logger.info("Combining temporary files...")
        
        final_state_dict = {}
        
        for category, temp_file in temp_files.items():
            if temp_file.exists():
                logger.info(f"Loading {category} weights...")
                category_data = torch.load(temp_file, map_location="cpu")
                final_state_dict.update(category_data)
                del category_data
                gc.collect()
        
        logger.info(f"Saving consolidated state dict with {len(final_state_dict)} parameters...")
        torch.save(final_state_dict, output_path)
        del final_state_dict
        gc.collect()
    
    def lazy_load_for_ttnn(self, weights_path: Path, device_id: int = 0, 
                          batch_size: int = 1, max_layers: Optional[int] = None) -> Dict[str, Any]:
        """
        Lazy load model for TTNN with minimal memory footprint.
        
        Args:
            weights_path: Path to consolidated weights
            device_id: TTNN device ID
            batch_size: Batch size for inference
            max_layers: Maximum number of layers to load
            
        Returns:
            Dictionary containing model components
        """
        with self.performance_monitor("TTNN lazy loading"):
            try:
                import ttnn
                
                # Open TTNN device
                device = ttnn.open_device(device_id=device_id)
                logger.info(f"Opened TTNN device {device_id}")
                
                # Load only essential components first
                essential_keys = [
                    "tok_embeddings.weight",
                    "norm.weight", 
                    "output.weight"
                ]
                
                essential_weights = {}
                state_dict = torch.load(weights_path, map_location="cpu")
                
                # Extract essential weights
                for key in essential_keys:
                    if key in state_dict:
                        essential_weights[key] = state_dict[key]
                        del state_dict[key]  # Free memory immediately
                
                # Process layers in batches
                layer_weights = {}
                max_layer = max_layers or 32  # Default for Ministral-8B
                
                for layer_idx in range(max_layer):
                    layer_prefix = f"layers.{layer_idx}."
                    layer_data = {}
                    
                    # Extract this layer's weights
                    keys_to_remove = []
                    for key in state_dict:
                        if key.startswith(layer_prefix):
                            layer_data[key] = state_dict[key]
                            keys_to_remove.append(key)
                    
                    # Remove from main dict to free memory
                    for key in keys_to_remove:
                        del state_dict[key]
                    
                    if layer_data:
                        layer_weights[layer_idx] = layer_data
                    
                    # Log progress and force garbage collection
                    if (layer_idx + 1) % 8 == 0:
                        logger.info(f"Loaded {layer_idx + 1}/{max_layer} layers")
                        gc.collect()
                
                del state_dict  # Final cleanup
                gc.collect()
                
                return {
                    'device': device,
                    'essential_weights': essential_weights,
                    'layer_weights': layer_weights,
                    'device_id': device_id
                }
                
            except Exception as e:
                logger.error(f"✗ Failed to lazy load for TTNN: {e}")
                raise
    
    def create_minimal_tokenizer(self, tokenizer_path: Path) -> Any:
        """Create a minimal tokenizer with reduced memory footprint."""
        try:
            from models.demos.wormhole.ministral8b.reference.tokenizer import Tokenizer
            
            with self.memory_monitor("tokenizer creation"):
                # Initialize with minimal configuration
                tokenizer = Tokenizer(str(tokenizer_path))
                logger.info(f"✓ Created minimal tokenizer from {tokenizer_path}")
                return tokenizer
                
        except Exception as e:
            logger.error(f"✗ Failed to create tokenizer: {e}")
            raise
    
    def estimate_memory_usage(self, model_path: Path) -> Dict[str, float]:
        """
        Estimate memory usage for different loading strategies.
        
        Returns:
            Dictionary with memory estimates in GB
        """
        try:
            # Get file size
            file_size_gb = model_path.stat().st_size / (1024**3)
            
            estimates = {
                'file_size_gb': file_size_gb,
                'traditional_load_gb': file_size_gb * 2.5,  # File + state_dict + filtered + TTNN
                'streaming_load_gb': file_size_gb * 1.2,    # Minimal overhead
                'chunked_load_gb': file_size_gb * 1.1,      # Even less overhead
                'recommended_ram_gb': max(file_size_gb * 1.5, 16)  # Minimum recommended
            }
            
            logger.info("Memory usage estimates:")
            for method, memory in estimates.items():
                logger.info(f"  {method}: {memory:.2f}GB")
            
            return estimates
            
        except Exception as e:
            logger.error(f"Failed to estimate memory usage: {e}")
            return {}

def check_system_resources() -> Dict[str, float]:
    """Check available system resources."""
    try:
        import psutil
        
        # Memory info
        memory = psutil.virtual_memory()
        available_ram_gb = memory.available / (1024**3)
        total_ram_gb = memory.total / (1024**3)
        
        # Disk space
        disk = psutil.disk_usage('/')
        available_disk_gb = disk.free / (1024**3)
        
        resources = {
            'available_ram_gb': available_ram_gb,
            'total_ram_gb': total_ram_gb,
            'ram_usage_percent': memory.percent,
            'available_disk_gb': available_disk_gb
        }
        
        logger.info("System resources:")
        for resource, value in resources.items():
            logger.info(f"  {resource}: {value:.2f}{'GB' if 'gb' in resource else '%'}")
        
        return resources
        
    except Exception as e:
        logger.error(f"Failed to check system resources: {e}")
        return {}

# Example usage and testing
if __name__ == "__main__":
    # Check system resources
    resources = check_system_resources()
    
    # Initialize memory-efficient loader
    cache_dir = os.environ.get('MODEL_CACHE_PATH', '/tmp/ministral8b_cache')
    loader = MemoryOptimizedLoader(cache_dir)
    
    # Example model file (replace with actual path)
    model_file = Path(cache_dir) / "consolidated.bin"
    
    if model_file.exists():
        # Estimate memory usage
        estimates = loader.estimate_memory_usage(model_file)
        
        # Check if we have enough RAM
        if resources.get('available_ram_gb', 0) < estimates.get('recommended_ram_gb', 32):
            logger.warning("⚠️ Insufficient RAM for optimal model loading")
            logger.warning("Consider using chunked processing or increasing available memory")
        else:
            logger.info("✓ Sufficient RAM available for efficient model loading")
