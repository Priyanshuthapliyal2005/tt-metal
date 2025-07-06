#!/usr/bin/env python3
"""
Centralized configuration module for Ministral-8B server.
Manages all environment variables, hardware settings, and deployment configurations.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class HardwareConfig:
    """Hardware-specific configuration settings."""
    device_id: int = 0
    batch_size: int = 1
    max_seq_len: int = 512
    instruct_mode: bool = True
    max_generated_tokens: int = 120
    temperature: float = 0.7
    top_p: float = 0.9
    dtype: str = "bfloat8_b"
    enable_async: bool = True
    memory_efficient: bool = True
    chunk_loading: bool = True
    lazy_init: bool = True
    reduced_precision: bool = False
    smaller_cache: bool = False
    
    def __post_init__(self):
        """Validate hardware configuration after initialization."""
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        if self.device_id < 0:
            raise ValueError("device_id must be >= 0")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")

@dataclass
class EnvironmentConfig:
    """Environment detection and cloud mode configuration."""
    environment_type: str = "unknown"
    is_docker: bool = False
    is_koyeb: bool = False
    is_cloud: bool = False
    is_local: bool = True
    cloud_mode: bool = False
    cache_path: str = "/workspace/model_cache"
    working_dir: str = field(default_factory=lambda: os.getcwd())
    log_level: str = "INFO"
    
    # Firmware and hardware paths
    firmware_path: str = "/workspace/runtime/hw/lib/wormhole"
    soc_descriptor_path: str = "/workspace/tt-metal/tt_metal/soc_descriptors"
    tt_metal_home: str = "/workspace/tt-metal"
    
    # Performance monitoring
    performance_monitoring_enabled: bool = False
    memory_monitoring_enabled: bool = True
    
    def __post_init__(self):
        """Auto-detect environment type and set flags."""
        # Detect environment from environment variables
        self.is_docker = os.environ.get('IS_DOCKER_ENVIRONMENT') == 'true'
        self.is_koyeb = os.environ.get('IS_KOYEB_ENVIRONMENT') == 'true'
        
        # Determine environment type
        if self.is_docker:
            self.environment_type = 'docker'
        elif self.is_koyeb:
            self.environment_type = 'koyeb'
        else:
            self.environment_type = 'local'
        
        # Set cloud flags
        self.is_cloud = self.is_docker or self.is_koyeb
        self.is_local = not self.is_cloud
        self.cloud_mode = self.is_cloud
        
        # Ensure cache path exists
        os.makedirs(self.cache_path, exist_ok=True)

@dataclass
class ModelConfig:
    """Model-specific configuration settings."""
    model_name: str = "mistralai/Ministral-8B-Instruct-2410"
    model_id: str = "ministral-8b-instruct-2410"
    tokenizer_path: Optional[str] = None
    consolidated_weights_path: Optional[str] = None
    weight_cache_path: Optional[str] = None
    loading_strategy: str = "auto"  # auto, optimized, standard, legacy, mock
    
    # Model architecture parameters
    vocab_size: int = 32768
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 14336
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    
    # Loading preferences
    prefer_consolidated: bool = True
    prefer_safetensors: bool = True
    enable_weight_filtering: bool = True
    
    # Optimization settings
    use_paged_kv_cache: bool = False
    enable_chunked_loading: bool = True
    chunk_size_mb: int = 256
    
    def __post_init__(self):
        """Set derived paths and validate configuration."""
        if self.tokenizer_path is None:
            cache_path = os.environ.get('MODEL_CACHE_PATH', '/workspace/model_cache')
            self.tokenizer_path = os.path.join(cache_path, "tokenizer.model")
        
        if self.consolidated_weights_path is None:
            cache_path = os.environ.get('MODEL_CACHE_PATH', '/workspace/model_cache')
            self.consolidated_weights_path = os.path.join(cache_path, "consolidated.bin")
        
        if self.weight_cache_path is None:
            cache_path = os.environ.get('MODEL_CACHE_PATH', '/workspace/model_cache')
            self.weight_cache_path = os.path.join(cache_path, "tt_cache")

@dataclass
class ServerConfig:
    """HTTP server configuration settings."""
    port: int = 8000
    host: str = ""
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: str = "*"
    cors_methods: str = "GET, POST, OPTIONS"
    cors_headers: str = "Content-Type"
    
    # Health check settings
    health_check_interval: int = 30
    health_check_timeout: int = 5
    
    # Request handling
    max_request_size: int = 1024 * 1024  # 1MB
    request_timeout: int = 300  # 5 minutes
    max_tokens_per_request: int = 1024
    
    # Background processing
    model_loading_check_interval: int = 30
    background_download_enabled: bool = True
    preload_model: bool = True
    
    # Threading
    use_threading_server: bool = True
    max_worker_threads: int = 4
    
    def __post_init__(self):
        """Validate server configuration."""
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.health_check_interval < 1:
            raise ValueError("health_check_interval must be >= 1")
        if self.max_tokens_per_request < 1:
            raise ValueError("max_tokens_per_request must be >= 1")

@dataclass
class TTNNConfig:
    """TTNN-specific configuration and status."""
    ttnn_available: bool = False
    hardware_available: bool = False
    firmware_available: bool = False
    devices: List[str] = field(default_factory=list)
    error: Optional[str] = None
    soc_descriptor_status: str = "unknown"
    yaml_error_details: Optional[Dict[str, Any]] = None
    
    # Initialization settings
    force_hw_detect: bool = False
    initialize_bus_table: bool = False  # Deprecated in current TTNN API
    
    # Firmware files to check
    required_firmware_files: List[str] = field(default_factory=lambda: [
        "idle_erisc.elf",
        "active_erisc.elf", 
        "tmu-crt0.o",
        "noc.o",
        "substitutes.o"
    ])
    
    # SOC descriptor files to check
    soc_descriptor_files: List[str] = field(default_factory=lambda: [
        "wormhole_b0_80_arch.yaml",
        "wormhole_b0_versim.yaml",
        "blackhole_140_arch.yaml"
    ])

@dataclass
class Config:
    """Main configuration container."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    ttnn: TTNNConfig = field(default_factory=TTNNConfig)
    
    def __post_init__(self):
        """Apply cross-configuration adjustments."""
        # Adjust hardware config based on environment
        if self.environment.is_cloud:
            self.hardware.memory_efficient = True
            self.hardware.chunk_loading = True
            self.hardware.reduced_precision = True
            self.hardware.smaller_cache = True
            self.model.loading_strategy = "mock" if self.environment.is_koyeb else "optimized"
        
        # Adjust model config based on hardware constraints
        if self.hardware.memory_efficient:
            self.model.chunk_size_mb = min(self.model.chunk_size_mb, 256)
            self.model.enable_chunked_loading = True
        
        # Adjust server config based on environment
        if self.environment.is_cloud:
            self.server.background_download_enabled = True
            self.server.preload_model = True

def load_config() -> Config:
    """
    Load configuration from environment variables with sensible defaults.
    
    Returns:
        Config: Complete configuration object
    """
    logger.info("Loading configuration from environment variables")
    
    # Hardware configuration from environment
    hardware_config = HardwareConfig(
        device_id=int(os.environ.get('DEVICE_ID', '0')),
        batch_size=int(os.environ.get('BATCH_SIZE', '1')),
        max_seq_len=int(os.environ.get('MAX_SEQ_LEN', '512')),
        instruct_mode=os.environ.get('INSTRUCT_MODE', 'true').lower() == 'true',
        max_generated_tokens=int(os.environ.get('MAX_GENERATED_TOKENS', '120')),
        temperature=float(os.environ.get('TEMPERATURE', '0.7')),
        top_p=float(os.environ.get('TOP_P', '0.9')),
        dtype=os.environ.get('TTNN_DTYPE', 'bfloat8_b'),
        enable_async=os.environ.get('ENABLE_ASYNC', 'true').lower() == 'true',
        memory_efficient=os.environ.get('MEMORY_EFFICIENT', 'true').lower() == 'true',
        chunk_loading=os.environ.get('CHUNK_LOADING', 'true').lower() == 'true',
        lazy_init=os.environ.get('LAZY_INIT', 'true').lower() == 'true'
    )
    
    # Environment configuration
    environment_config = EnvironmentConfig(
        cache_path=os.environ.get('MODEL_CACHE_PATH', '/workspace/model_cache'),
        log_level=os.environ.get('LOG_LEVEL', 'INFO'),
        firmware_path=os.environ.get('FIRMWARE_PATH', '/workspace/runtime/hw/lib/wormhole'),
        soc_descriptor_path=os.environ.get('SOC_DESCRIPTOR_PATH', '/workspace/tt-metal/tt_metal/soc_descriptors'),
        tt_metal_home=os.environ.get('TT_METAL_HOME', '/workspace/tt-metal'),
        performance_monitoring_enabled=os.environ.get('PERFORMANCE_MONITORING', 'false').lower() == 'true',
        memory_monitoring_enabled=os.environ.get('MEMORY_MONITORING', 'true').lower() == 'true'
    )
    
    # Model configuration
    model_config = ModelConfig(
        model_name=os.environ.get('MODEL_NAME', 'mistralai/Ministral-8B-Instruct-2410'),
        loading_strategy=os.environ.get('LOADING_STRATEGY', 'auto'),
        prefer_consolidated=os.environ.get('PREFER_CONSOLIDATED', 'true').lower() == 'true',
        prefer_safetensors=os.environ.get('PREFER_SAFETENSORS', 'true').lower() == 'true',
        enable_weight_filtering=os.environ.get('ENABLE_WEIGHT_FILTERING', 'true').lower() == 'true',
        use_paged_kv_cache=os.environ.get('USE_PAGED_KV_CACHE', 'false').lower() == 'true',
        enable_chunked_loading=os.environ.get('ENABLE_CHUNKED_LOADING', 'true').lower() == 'true',
        chunk_size_mb=int(os.environ.get('CHUNK_SIZE_MB', '256'))
    )
    
    # Server configuration
    server_config = ServerConfig(
        port=int(os.environ.get('PORT', '8000')),
        host=os.environ.get('HOST', ''),
        cors_enabled=os.environ.get('CORS_ENABLED', 'true').lower() == 'true',
        cors_origins=os.environ.get('CORS_ORIGINS', '*'),
        cors_methods=os.environ.get('CORS_METHODS', 'GET, POST, OPTIONS'),
        cors_headers=os.environ.get('CORS_HEADERS', 'Content-Type'),
        health_check_interval=int(os.environ.get('HEALTH_CHECK_INTERVAL', '30')),
        health_check_timeout=int(os.environ.get('HEALTH_CHECK_TIMEOUT', '5')),
        max_request_size=int(os.environ.get('MAX_REQUEST_SIZE', str(1024 * 1024))),
        request_timeout=int(os.environ.get('REQUEST_TIMEOUT', '300')),
        max_tokens_per_request=int(os.environ.get('MAX_TOKENS_PER_REQUEST', '1024')),
        model_loading_check_interval=int(os.environ.get('MODEL_LOADING_CHECK_INTERVAL', '30')),
        background_download_enabled=os.environ.get('BACKGROUND_DOWNLOAD', 'true').lower() == 'true',
        preload_model=os.environ.get('PRELOAD_MODEL', 'true').lower() == 'true',
        use_threading_server=os.environ.get('USE_THREADING_SERVER', 'true').lower() == 'true',
        max_worker_threads=int(os.environ.get('MAX_WORKER_THREADS', '4'))
    )
    
    # TTNN configuration
    ttnn_config = TTNNConfig(
        force_hw_detect=os.environ.get('FORCE_HW_DETECT', 'false').lower() == 'true'
    )
    
    # Create main config
    config = Config(
        hardware=hardware_config,
        environment=environment_config,
        model=model_config,
        server=server_config,
        ttnn=ttnn_config
    )
    
    logger.info(f"Configuration loaded for {config.environment.environment_type} environment")
    logger.info(f"Hardware: device_id={config.hardware.device_id}, batch_size={config.hardware.batch_size}")
    logger.info(f"Model: {config.model.model_name}, strategy={config.model.loading_strategy}")
    logger.info(f"Server: port={config.server.port}, preload={config.server.preload_model}")
    
    return config

def validate_config(config: Config) -> Dict[str, List[str]]:
    """
    Validate configuration consistency and hardware compatibility.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        Dict with 'errors' and 'warnings' lists
    """
    errors = []
    warnings = []
    
    # Validate paths exist
    if not os.path.exists(config.environment.cache_path):
        try:
            os.makedirs(config.environment.cache_path, exist_ok=True)
            warnings.append(f"Created missing cache directory: {config.environment.cache_path}")
        except Exception as e:
            errors.append(f"Cannot create cache directory {config.environment.cache_path}: {e}")
    
    # Validate firmware path in non-cloud environments
    if not config.environment.is_cloud:
        if not os.path.exists(config.environment.firmware_path):
            warnings.append(f"Firmware path does not exist: {config.environment.firmware_path}")
        
        # Check for required firmware files
        missing_firmware = []
        for fw_file in config.ttnn.required_firmware_files:
            fw_path = os.path.join(config.environment.firmware_path, fw_file)
            if not os.path.exists(fw_path):
                missing_firmware.append(fw_file)
        
        if missing_firmware:
            warnings.append(f"Missing firmware files: {missing_firmware}")
    
    # Validate SOC descriptor path
    if not os.path.exists(config.environment.soc_descriptor_path):
        warnings.append(f"SOC descriptor path does not exist: {config.environment.soc_descriptor_path}")
    else:
        # Check for SOC descriptor files
        missing_soc = []
        for soc_file in config.ttnn.soc_descriptor_files:
            soc_path = os.path.join(config.environment.soc_descriptor_path, soc_file)
            if not os.path.exists(soc_path):
                missing_soc.append(soc_file)
        
        if missing_soc:
            warnings.append(f"Missing SOC descriptor files: {missing_soc}")
    
    # Validate model paths
    if not os.path.exists(config.model.tokenizer_path):
        warnings.append(f"Tokenizer path does not exist: {config.model.tokenizer_path}")
    
    if not os.path.exists(config.model.consolidated_weights_path):
        warnings.append(f"Consolidated weights path does not exist: {config.model.consolidated_weights_path}")
    
    # Validate hardware configuration consistency
    if config.hardware.batch_size > 1 and config.environment.is_cloud:
        warnings.append("Batch size > 1 may cause memory issues in cloud environments")
    
    if config.hardware.max_seq_len > 2048 and config.hardware.memory_efficient:
        warnings.append("Large max_seq_len with memory_efficient=True may cause performance issues")
    
    # Validate server configuration
    if config.server.port < 1024 and os.getuid() != 0:
        errors.append(f"Port {config.server.port} requires root privileges")
    
    # Validate loading strategy
    valid_strategies = ["auto", "optimized", "standard", "legacy", "mock"]
    if config.model.loading_strategy not in valid_strategies:
        errors.append(f"Invalid loading strategy: {config.model.loading_strategy}. Must be one of {valid_strategies}")
    
    # Environment-specific validations
    if config.environment.is_cloud:
        if config.model.loading_strategy not in ["mock", "optimized", "auto"]:
            warnings.append(f"Loading strategy '{config.model.loading_strategy}' may not work in cloud environment")
        
        if not config.hardware.memory_efficient:
            warnings.append("memory_efficient=False may cause OOM in cloud environments")
    
    # Memory consistency checks
    available_ram_gb = 0
    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
    except ImportError:
        warnings.append("psutil not available - cannot check memory constraints")
    
    if available_ram_gb > 0:
        if available_ram_gb < 8 and not config.hardware.memory_efficient:
            warnings.append(f"Low RAM ({available_ram_gb:.1f}GB) detected but memory_efficient=False")
        
        if available_ram_gb < 4:
            errors.append(f"Insufficient RAM for model loading: {available_ram_gb:.1f}GB available, minimum 4GB required")
    
    # Log validation results
    if errors:
        logger.error(f"Configuration validation errors: {errors}")
    if warnings:
        logger.warning(f"Configuration validation warnings: {warnings}")
    else:
        logger.info("Configuration validation passed")
    
    return {"errors": errors, "warnings": warnings}

def get_optimizations_dict(config: Config) -> Dict[str, Any]:
    """
    Get optimizations dictionary for model loading based on configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary of optimization settings
    """
    return {
        "batch_size": config.hardware.batch_size,
        "max_seq_len": config.hardware.max_seq_len,
        "enable_async": config.hardware.enable_async,
        "memory_efficient": config.hardware.memory_efficient,
        "chunk_loading": config.hardware.chunk_loading,
        "lazy_init": config.hardware.lazy_init,
        "reduced_precision": config.hardware.reduced_precision,
        "smaller_cache": config.hardware.smaller_cache,
        "use_paged_kv_cache": config.model.use_paged_kv_cache,
        "enable_chunked_loading": config.model.enable_chunked_loading,
        "chunk_size_mb": config.model.chunk_size_mb
    }

def update_config_from_args(config: Config, args) -> Config:
    """
    Update configuration from command line arguments.
    
    Args:
        config: Base configuration object
        args: Parsed command line arguments
        
    Returns:
        Updated configuration object
    """
    if hasattr(args, 'port') and args.port:
        config.server.port = args.port
    
    if hasattr(args, 'device_id') and args.device_id is not None:
        config.hardware.device_id = args.device_id
    
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config.hardware.batch_size = args.batch_size
    
    if hasattr(args, 'max_seq_len') and args.max_seq_len is not None:
        config.hardware.max_seq_len = args.max_seq_len
    
    if hasattr(args, 'instruct') and args.instruct is not None:
        config.hardware.instruct_mode = args.instruct
    
    if hasattr(args, 'no_preload') and args.no_preload:
        config.server.preload_model = False
    
    if hasattr(args, 'cloud_mode') and args.cloud_mode:
        config.environment.cloud_mode = True
        config.model.loading_strategy = "mock"
    
    if hasattr(args, 'use_legacy') and args.use_legacy:
        config.model.loading_strategy = "legacy"
    
    return config

# Global configuration instance
_global_config: Optional[Config] = None

def get_config() -> Config:
    """Get the global configuration instance, loading it if necessary."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config

def set_config(config: Config):
    """Set the global configuration instance."""
    global _global_config
    _global_config = config

def reload_config() -> Config:
    """Reload configuration from environment variables."""
    global _global_config
    _global_config = load_config()
    return _global_config