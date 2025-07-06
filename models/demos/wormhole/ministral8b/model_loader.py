#!/usr/bin/env python3
"""
Unified model loader module for Ministral-8B deployment.
Consolidates all model loading strategies into a clean, maintainable architecture.
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import torch
import gc

# Setup logging
logger = logging.getLogger(__name__)

# Import performance monitoring
try:
    from performance_optimizer import performance_optimizer
    from memory_efficient_loader import MemoryOptimizedLoader, check_system_resources
    PERFORMANCE_MONITORING_ENABLED = True
    logger.info("🔥 Performance monitoring enabled")
except ImportError as e:
    logger.warning(f"Performance monitoring disabled: {e}")
    PERFORMANCE_MONITORING_ENABLED = False

# Import tt-transformers framework
try:
    from models.tt_transformers.mistral8b.model import create_ministral_model, MistralModelArgs, MistralTransformer
    from models.tt_transformers.tt.common import create_tt_model, sample_host, PagedAttentionConfig
    TT_TRANSFORMERS_AVAILABLE = True
    logger.info("✅ tt-transformers framework imported successfully")
except ImportError as e:
    logger.warning(f"tt-transformers framework not available: {e}")
    TT_TRANSFORMERS_AVAILABLE = False

# Import TTNN and related modules
try:
    import ttnn
    from models.demos.wormhole.ministral8b.reference.tokenizer import Tokenizer
    from models.demos.wormhole.ministral8b.tt.model_config import TtModelArgs
    from models.demos.wormhole.ministral8b.tt.mistral_model import TtTransformer
    from models.demos.wormhole.ministral8b.tt.mistral_embedding import TtMistralEmbedding
    from models.demos.wormhole.ministral8b.tt.mistral_common import (
        cache_attention,
        freqs_to_rotation_matrix,
        precompute_freqs,
    )
    TTNN_AVAILABLE = True
    logger.info("✅ TTNN modules imported successfully")
except ImportError as e:
    logger.warning(f"TTNN modules not available: {e}")
    TTNN_AVAILABLE = False

# Import hardware utilities
try:
    from hw_utils import detect_hardware_capabilities, initialize_tt_device, get_environment_type
    HW_UTILS_AVAILABLE = True
except ImportError:
    logger.warning("hw_utils not available, using fallback hardware detection")
    HW_UTILS_AVAILABLE = False

class UnifiedModelLoader:
    """
    Unified model loader that consolidates all loading strategies.
    Provides a clean strategy pattern for different loading approaches.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_path = os.environ.get('MODEL_CACHE_PATH', '/workspace/model_cache')
        self.hardware_capabilities = None
        self.device = None
        self.model_args = None
        self._lock = threading.Lock()
        
        # Initialize hardware capabilities
        self._detect_hardware()
    
    def _detect_hardware(self):
        """Detect hardware capabilities using hw_utils or fallback."""
        try:
            if HW_UTILS_AVAILABLE:
                self.hardware_capabilities = detect_hardware_capabilities()
            else:
                # Fallback hardware detection
                self.hardware_capabilities = self._fallback_hardware_detection()
            
            self.logger.info(f"Hardware capabilities: {self.hardware_capabilities}")
        except Exception as e:
            self.logger.error(f"Hardware detection failed: {e}")
            self.hardware_capabilities = {
                'ttnn_available': False,
                'hardware_available': False,
                'firmware_available': False,
                'environment_type': 'unknown',
                'devices': [],
                'error': str(e)
            }
    
    def _fallback_hardware_detection(self) -> Dict[str, Any]:
        """Fallback hardware detection when hw_utils is not available."""
        capabilities = {
            'ttnn_available': TTNN_AVAILABLE,
            'hardware_available': False,
            'firmware_available': False,
            'environment_type': get_environment_type() if HW_UTILS_AVAILABLE else 'unknown',
            'devices': [],
            'error': None
        }
        
        if not TTNN_AVAILABLE:
            capabilities['error'] = "TTNN not available"
            return capabilities
        
        try:
            # Try to detect devices using current TTNN API
            num_devices = ttnn.GetNumAvailableDevices()
            if num_devices > 0:
                capabilities['hardware_available'] = True
                # Generate device IDs list based on available devices
                capabilities['devices'] = [str(i) for i in range(num_devices)]
                
                # Check firmware files
                firmware_path = "/workspace/runtime/hw/lib/wormhole"
                firmware_files = ["tmu-crt0.o", "noc.o", "substitutes.o", "idle_erisc.elf"]
                missing_files = []
                
                for file in firmware_files:
                    file_path = Path(f"{firmware_path}/{file}")
                    if not file_path.exists():
                        missing_files.append(file)
                
                if not missing_files:
                    capabilities['firmware_available'] = True
                else:
                    capabilities['error'] = f"Missing firmware files: {missing_files}"
                    
        except Exception as e:
            capabilities['error'] = f"Hardware detection failed: {e}"
        
        return capabilities
    
    def load_model(self, strategy='auto', device_id=0, batch_size=1, max_seq_len=512, instruct=True) -> Tuple[Any, Any]:
        """
        Main entry point that selects appropriate loading strategy.
        
        Args:
            strategy: Loading strategy ('auto', 'optimized', 'standard', 'legacy', 'mock')
            device_id: TT device ID to use
            batch_size: Batch size for inference
            max_seq_len: Maximum sequence length
            instruct: Whether to use instruct mode
            
        Returns:
            Tuple of (model, tokenizer)
        """
        with self._lock:
            self.logger.info(f"🚀 Loading Ministral-8B model with strategy='{strategy}', device_id={device_id}, batch_size={batch_size}, max_seq_len={max_seq_len}")
            
            # Auto-select strategy based on hardware capabilities
            if strategy == 'auto':
                strategy = self._select_optimal_strategy()
                self.logger.info(f"Auto-selected strategy: {strategy}")
            
            # Store parameters for use in loading methods
            self.device_id = device_id
            self.batch_size = batch_size
            self.max_seq_len = max_seq_len
            self.instruct = instruct
            
            # Execute selected strategy
            try:
                if strategy == 'optimized':
                    return self._load_optimized()
                elif strategy == 'standard':
                    return self._load_standard()
                elif strategy == 'legacy':
                    return self._load_legacy()
                elif strategy == 'mock':
                    return self._load_mock()
                else:
                    raise ValueError(f"Unknown loading strategy: {strategy}")
                    
            except Exception as e:
                self.logger.error(f"Loading strategy '{strategy}' failed: {e}")
                # Try fallback strategies
                return self._try_fallback_strategies(strategy, e)
    
    def _select_optimal_strategy(self) -> str:
        """Select optimal loading strategy based on hardware capabilities."""
        caps = self.hardware_capabilities
        
        # Check environment type
        env_type = caps.get('environment_type', 'unknown')
        if env_type in ['koyeb', 'docker'] and not caps.get('hardware_available', False):
            return 'mock'
        
        # Check hardware and firmware availability
        if caps.get('hardware_available', False) and caps.get('firmware_available', False):
            # Full hardware acceleration available
            if PERFORMANCE_MONITORING_ENABLED and TT_TRANSFORMERS_AVAILABLE:
                return 'optimized'
            elif TT_TRANSFORMERS_AVAILABLE:
                return 'standard'
            else:
                return 'legacy'
        elif caps.get('ttnn_available', False):
            # TTNN available but hardware/firmware issues
            if TT_TRANSFORMERS_AVAILABLE:
                return 'standard'
            else:
                return 'legacy'
        else:
            # No TTNN available
            return 'mock'
    
    def _try_fallback_strategies(self, failed_strategy: str, error: Exception) -> Tuple[Any, Any]:
        """Try fallback strategies when primary strategy fails."""
        fallback_order = ['optimized', 'standard', 'legacy', 'mock']
        
        # Remove the failed strategy and try others
        if failed_strategy in fallback_order:
            fallback_order.remove(failed_strategy)
        
        for fallback_strategy in fallback_order:
            try:
                self.logger.warning(f"Trying fallback strategy: {fallback_strategy}")
                
                if fallback_strategy == 'optimized':
                    return self._load_optimized()
                elif fallback_strategy == 'standard':
                    return self._load_standard()
                elif fallback_strategy == 'legacy':
                    return self._load_legacy()
                elif fallback_strategy == 'mock':
                    return self._load_mock()
                    
            except Exception as fallback_error:
                self.logger.error(f"Fallback strategy '{fallback_strategy}' also failed: {fallback_error}")
                continue
        
        # All strategies failed
        self.logger.error("All loading strategies failed, returning mock model")
        return self._load_mock()
    
    def _load_optimized(self) -> Tuple[Any, Any]:
        """Memory-efficient loading using performance monitoring."""
        if not PERFORMANCE_MONITORING_ENABLED:
            raise ImportError("Performance monitoring not available for optimized loading")
        
        # Check system resources
        resources = check_system_resources()
        available_ram = resources.get('available_ram_gb', 0)
        self.logger.info(f"Available RAM: {available_ram:.2f}GB")
        
        if available_ram < 8:
            raise RuntimeError(f"Insufficient RAM for optimized loading: {available_ram:.2f}GB available, minimum 8GB required")
        
        # Performance monitoring context
        monitor_context = performance_optimizer.performance_monitor("Optimized Model Loading")
        
        with monitor_context:
            # Setup device
            device = self._setup_device()
            
            if TT_TRANSFORMERS_AVAILABLE:
                return self._load_optimized_tt_transformers(device, available_ram)
            else:
                return self._load_optimized_legacy(device, available_ram)
    
    def _load_optimized_tt_transformers(self, device, available_ram: float) -> Tuple[Any, Any]:
        """Optimized loading using tt-transformers framework."""
        self.logger.info("Using optimized tt-transformers loading")
        
        # Set up optimizations based on available memory
        optimizations = {
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "enable_async": True,
            "memory_efficient": True,
            "chunk_loading": True,
            "lazy_init": True
        }
        
        if available_ram < 16:
            optimizations["reduced_precision"] = True
            optimizations["smaller_cache"] = True
            self.max_seq_len = min(self.max_seq_len, 1024)
            self.logger.info(f"Low memory detected, reducing max_seq_len to {self.max_seq_len}")
        
        # Create model using tt-transformers
        model_args, model, kv_cache, state_dict = create_ministral_model(
            mesh_device=device,
            instruct=self.instruct,
            max_batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            optimizations=optimizations,
            dtype=ttnn.bfloat8_b,
            state_dict=None,
            paged_attention_config=None,
            use_paged_kv_cache=False
        )
        
        # Initialize tokenizer
        tokenizer = Tokenizer(model_args.tokenizer_path)
        
        # Store additional components
        model._device = device
        model._args = model_args
        model._kv_cache = kv_cache
        model._state_dict = state_dict
        
        # Final memory cleanup
        gc.collect()
        
        self.logger.info("Optimized tt-transformers model loaded successfully")
        return model, tokenizer
    
    def _load_optimized_legacy(self, device, available_ram: float) -> Tuple[Any, Any]:
        """Optimized loading using legacy implementation with memory optimization."""
        self.logger.info("Using optimized legacy loading")
        
        # Initialize memory-efficient loader
        chunk_size_mb = 256 if available_ram < 16 else 512
        loader = MemoryOptimizedLoader(self.cache_path, chunk_size_mb=chunk_size_mb)
        
        # Initialize model args
        model_args = TtModelArgs(device, instruct=self.instruct)
        
        # Initialize tokenizer using memory-efficient method
        tokenizer = loader.create_minimal_tokenizer(Path(model_args.tokenizer_path))
        
        # Load weights using lazy loading
        weights_path = Path(model_args.consolidated_weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Consolidated weights not found at {weights_path}")
        
        # Use lazy loading for TTNN
        model_components = loader.lazy_load_for_ttnn(
            weights_path=weights_path,
            device_id=self.device_id,
            batch_size=self.batch_size,
            max_layers=model_args.n_layers
        )
        
        # Build model with optimized components
        model = self._build_legacy_model(model_components, model_args)
        
        # Final cleanup
        gc.collect()
        
        self.logger.info("Optimized legacy model loaded successfully")
        return model, tokenizer
    
    def _load_standard(self) -> Tuple[Any, Any]:
        """Standard tt-transformers loading."""
        if not TT_TRANSFORMERS_AVAILABLE:
            raise ImportError("tt-transformers not available for standard loading")
        
        self.logger.info("Using standard tt-transformers loading")
        
        # Setup device
        device = self._setup_device()
        
        # Set up standard optimizations
        optimizations = {
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "enable_async": True,
            "memory_efficient": True
        }
        
        # Create model using tt-transformers
        model_args, model, kv_cache, state_dict = create_ministral_model(
            mesh_device=device,
            instruct=self.instruct,
            max_batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            optimizations=optimizations,
            dtype=ttnn.bfloat8_b,
            state_dict=None,
            paged_attention_config=None,
            use_paged_kv_cache=False
        )
        
        # Initialize tokenizer
        tokenizer = Tokenizer(model_args.tokenizer_path)
        
        # Store additional components
        model._device = device
        model._args = model_args
        model._kv_cache = kv_cache
        model._state_dict = state_dict
        
        self.logger.info("Standard tt-transformers model loaded successfully")
        return model, tokenizer
    
    def _load_legacy(self) -> Tuple[Any, Any]:
        """Direct TTNN fallback loading."""
        if not TTNN_AVAILABLE:
            raise ImportError("TTNN not available for legacy loading")
        
        self.logger.info("Using legacy TTNN loading")
        
        # Setup device
        device = self._setup_device()
        
        # Initialize model args
        model_args = TtModelArgs(device, instruct=self.instruct)
        
        # Initialize tokenizer
        tokenizer = Tokenizer(model_args.tokenizer_path)
        
        # Load and filter weights
        state_dict = self._load_weights(model_args.consolidated_weights_path, model_args.n_layers)
        
        # Setup embeddings
        embd = self._setup_embeddings(model_args, state_dict)
        
        # Setup attention cache
        rot_emb_matrix_list = self._setup_attention_cache(device, model_args, state_dict)
        
        # Create transformer model
        model = TtTransformer(
            args=model_args,
            device=device,
            dtype=ttnn.bfloat8_b,
            state_dict=state_dict,
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
            layers=list(range(model_args.n_layers)),
            rot_mat=rot_emb_matrix_list,
            start_pos=0,
        )
        
        # Create TT embedding layer
        tt_embd = TtMistralEmbedding(
            device=device,
            args=model_args,
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,
        )
        
        # Store components in model
        model._embd = embd
        model._tt_embd = tt_embd
        model._rot_emb_matrix_list = rot_emb_matrix_list
        model._device = device
        model._args = model_args
        
        self.logger.info("Legacy TTNN model loaded successfully")
        return model, tokenizer
    
    def _load_mock(self) -> Tuple[Any, Any]:
        """Mock model for cloud environments."""
        self.logger.info("Using mock model for cloud environment")
        
        # Create mock model and tokenizer
        class MockModel:
            def __init__(self):
                self._device = None
                self._args = None
                
            def __call__(self, *args, **kwargs):
                return "Mock model response"
        
        class MockTokenizer:
            def __init__(self):
                self.eos_id = 2
                
            def encode(self, text):
                # Simple word-based tokenization for mock
                return list(range(len(text.split())))
                
            def decode(self, tokens):
                return f"Mock decoded text from {len(tokens)} tokens"
        
        model = MockModel()
        tokenizer = MockTokenizer()
        
        self.logger.info("Mock model created successfully")
        return model, tokenizer
    
    def _setup_device(self) -> Any:
        """Common device initialization logic."""
        if self.device is not None:
            return self.device
        
        if not TTNN_AVAILABLE:
            raise ImportError("TTNN not available for device setup")
        
        try:
            if HW_UTILS_AVAILABLE:
                self.device = initialize_tt_device(self.device_id)
            else:
                self.device = ttnn.open_device(device_id=self.device_id)
            
            self.logger.info(f"TT device {self.device_id} initialized successfully")
            return self.device
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TT device {self.device_id}: {e}")
            raise
    
    def _load_weights(self, weights_path: str, n_layers: int) -> Dict[str, torch.Tensor]:
        """Common weight loading and filtering logic."""
        self.logger.info(f"Loading model weights from {weights_path}")
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        # Load state dict
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Filter state dict to only include relevant layers
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if (
                any([f"layers.{i}." in k for i in range(n_layers)])
                or k in ["tok_embeddings.weight", "norm.weight", "output.weight"]
            )
        }
        
        self.logger.info(f"Filtered state dict with {len(filtered_state_dict)} keys")
        return filtered_state_dict
    
    def _setup_embeddings(self, model_args, state_dict: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """Common embedding layer creation."""
        class Emb(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(model_args.vocab_size, model_args.dim)

            def forward(self, x):
                return self.emb(x)

        embd = Emb()
        if "tok_embeddings.weight" in state_dict:
            embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
            self.logger.info("Loaded embedding weights")
        
        return embd
    
    def _setup_attention_cache(self, device, model_args, state_dict: Dict[str, torch.Tensor]) -> list:
        """Common attention caching logic."""
        self.logger.info("Setting up attention cache and rotation matrices")
        
        # Precompute rotation matrices
        cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
        rot_emb_matrix = freqs_to_rotation_matrix(cos, sin)
        
        rot_emb_matrix_list = []
        for i in range(rot_emb_matrix.shape[0]):
            rot_emb_matrix_list.append(
                ttnn.from_torch(
                    rot_emb_matrix[i, :, :].unsqueeze(0).unsqueeze(0), 
                    device=device, 
                    dtype=ttnn.bfloat8_b, 
                    layout=ttnn.TILE_LAYOUT
                )
            )
        
        self.logger.info(f"Created {len(rot_emb_matrix_list)} rotation matrices")
        
        # Cache attention
        max_generated_tokens = 120
        try:
            cache_attention(device, state_dict, model_args, rot_emb_matrix_list, ttnn.bfloat8_b, max_generated_tokens)
            self.logger.info(f"Cached attention for {max_generated_tokens} tokens")
        except Exception as e:
            self.logger.warning(f"Attention caching failed, proceeding without cache: {e}")
        
        return rot_emb_matrix_list
    
    def _build_legacy_model(self, model_components: Dict[str, Any], model_args) -> Any:
        """Build legacy model from loaded components."""
        device = model_components['device']
        essential_weights = model_components['essential_weights']
        layer_weights = model_components['layer_weights']
        
        # Create filtered state dict from loaded components
        filtered_state_dict = essential_weights.copy()
        
        # Add layer weights progressively
        for layer_idx, layer_data in layer_weights.items():
            filtered_state_dict.update(layer_data)
            
            # Force garbage collection every few layers
            if layer_idx % 4 == 0:
                gc.collect()
        
        # Setup embeddings and attention cache
        embd = self._setup_embeddings(model_args, filtered_state_dict)
        rot_emb_matrix_list = self._setup_attention_cache(device, model_args, filtered_state_dict)
        
        # Create transformer model
        model = TtTransformer(
            args=model_args,
            device=device,
            dtype=ttnn.bfloat8_b,
            state_dict=filtered_state_dict,
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
            layers=list(range(model_args.n_layers)),
            rot_mat=rot_emb_matrix_list,
            start_pos=0,
        )
        
        # Create TT embedding layer
        tt_embd = TtMistralEmbedding(
            device=device,
            args=model_args,
            weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
            state_dict=filtered_state_dict,
            dtype=ttnn.bfloat16,
        )
        
        # Store components
        model._embd = embd
        model._tt_embd = tt_embd
        model._rot_emb_matrix_list = rot_emb_matrix_list
        model._device = device
        model._args = model_args
        
        return model

# Global instance for easy access
unified_model_loader = UnifiedModelLoader()

# Convenience functions for backward compatibility
def load_ministral_model_and_tokenizer(device_id=0, batch_size=1, max_seq_len=512, instruct=True):
    """Backward compatibility function for standard loading."""
    return unified_model_loader.load_model('standard', device_id, batch_size, max_seq_len, instruct)

def load_ministral_model_and_tokenizer_optimized(device_id=0, batch_size=1, max_seq_len=512, instruct=True):
    """Backward compatibility function for optimized loading."""
    return unified_model_loader.load_model('optimized', device_id, batch_size, max_seq_len, instruct)

def load_ministral_model_and_tokenizer_legacy(device_id=0, batch_size=1, max_seq_len=512, instruct=True):
    """Backward compatibility function for legacy loading."""
    return unified_model_loader.load_model('legacy', device_id, batch_size, max_seq_len, instruct)
