# Ministral-8B Deployment Guide

This guide covers deployment of Ministral-8B using the tt-transformers framework, which provides shared components and eliminates code duplication while maintaining high performance.

## Architecture Overview

Ministral-8B now uses the shared tt-transformers framework instead of custom modules:
- **Model Implementation**: Uses shared `Transformer` class from tt-transformers
- **Weight Conversion**: Leverages shared weight conversion utilities
- **Performance Optimization**: Built-in performance monitoring and optimization
- **Device Management**: Enhanced device detection with better error handling

## Hardware Requirements

### Supported Hardware
- **Wormhole B0**: Primary target hardware for Ministral-8B
- **N150/N300**: Single and multi-chip configurations
- **T3K (LoudBox/QuietBox)**: Multi-chip tensor parallelism
- **TG (Galaxy)**: Large-scale deployments

### Hardware Detection Process
The system automatically detects available hardware through:
1. TTNN device enumeration via `ttnn.get_num_devices()`
2. SOC descriptor YAML parsing for hardware configuration
3. Environment variable validation for hardware-specific settings
4. Fallback to CPU-only mode if hardware detection fails

## Environment Variables

### Required Variables
```bash
# Core TT-Metal configuration
export TT_METAL_HOME=/opt/tt-metal
export PYTHONPATH=/opt/tt-metal
export ARCH_NAME=wormhole_b0

# Model configuration (choose one)
export HF_MODEL=mistralai/Ministral-8B-Instruct-2410  # HuggingFace model
# OR
export LLAMA_DIR=/path/to/downloaded/weights  # Local weights directory
```

### Hardware-Specific Variables
```bash
# For N150, N300, and multi-chip systems
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

# Optional: Control device mesh size
export MESH_DEVICE=N150  # Use single chip even on multi-chip systems

# Optional: Custom cache location
export TT_CACHE_PATH=/path/to/cache/directory
```

### Performance Tuning Variables
```bash
# Prefill chunk size (powers of 2: 4, 8, 16, 32, 64, 128)
export MAX_PREFILL_CHUNK_SIZE=32  # Thousands of tokens per chunk

# MLP core padding for non-power-of-2 hidden dimensions
export PAD_MLP_CORES=32  # Multiple of 8, between 8-64

# Optimization level
export MODEL_OPTIMIZATION=performance  # or 'accuracy'
```

## Quick Deployment Commands

### Option 1: Direct Koyeb Deployment (Recommended)
```bash
# Deploy directly using the enhanced deployment script
./koyeb_deploy_ttnn.sh
```

### Option 2: Manual Koyeb Deployment
```bash
# Create the service with tt-transformers support
koyeb service create ministral-8b-app \
  --docker ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc \
  --docker-command "bash" \
  --docker-args "-c,pip install -r requirements.txt && python server.py" \
  --ports 8000:http \
  --regions fra \
  --instance-type nano \
  --env TT_METAL_HOME=/opt/tt-metal \
  --env PYTHONPATH=/opt/tt-metal \
  --env ARCH_NAME=wormhole_b0 \
  --env HF_MODEL=mistralai/Ministral-8B-Instruct-2410 \
  --env WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml \
  --env MODEL_OPTIMIZATION=performance
```

### Option 3: Local Testing
```bash
# Pull and run the official TT-Metalium Docker image
docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

# Run locally for testing with tt-transformers
docker run -p 8000:8000 \
  -e TT_METAL_HOME=/opt/tt-metal \
  -e PYTHONPATH=/opt/tt-metal \
  -e ARCH_NAME=wormhole_b0 \
  -e HF_MODEL=mistralai/Ministral-8B-Instruct-2410 \
  -e WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml \
  -e MODEL_OPTIMIZATION=performance \
  -v $(pwd):/app \
  -w /app \
  ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc \
  bash -c "pip install -r requirements.txt && python server.py"
```

## Troubleshooting

### YAML Parsing Issues

**Problem**: `ttnn.get_num_devices()` fails with "bad conversion" error at line 29, column 21 in SOC descriptor YAML.

**Root Cause**: Type mismatch in SOC descriptor where `eth_endpoint: [0, 0]` (list format) conflicts with expected integer format.

**Solutions**:

1. **Check SOC Descriptor Path**:
   ```bash
   echo $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml
   ls -la $TT_METAL_HOME/tt_metal/soc_descriptors/
   ```

2. **Verify YAML Format**:
   ```bash
   # Check for list vs integer format inconsistencies
   grep -n "eth_endpoint\|worker_endpoint" $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml
   ```

3. **Use Alternative SOC Descriptor**:
   ```bash
   # Try the versim descriptor which uses integer format
   export WH_ARCH_YAML=wormhole_b0_versim.yaml
   ```

4. **Environment Variable Override**:
   ```bash
   # Force specific SOC descriptor
   export TT_METAL_SOC_DESC_PATH=$TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_versim.yaml
   ```

### Device Initialization Failures

**Problem**: Hardware detection fails or returns 0 devices.

**Diagnostic Steps**:

1. **Check Hardware Availability**:
   ```bash
   # Verify TT hardware is detected by the system
   lspci | grep -i tenstorrent
   dmesg | grep -i tenstorrent
   ```

2. **Test TTNN Import**:
   ```python
   import ttnn
   print(f"TTNN version: {ttnn.__version__}")
   try:
       num_devices = ttnn.get_num_devices()
       print(f"Detected devices: {num_devices}")
   except Exception as e:
       print(f"Device detection failed: {e}")
   ```

3. **Check Environment Setup**:
   ```bash
   # Verify all required environment variables
   env | grep -E "TT_METAL|ARCH_NAME|PYTHONPATH"
   ```

4. **Fallback to CPU Mode**:
   ```bash
   # Force CPU-only operation for testing
   export TT_METAL_FORCE_CPU=1
   ```

### Performance Issues

**Problem**: Token generation is slower than expected targets.

**Optimization Steps**:

1. **Enable Performance Mode**:
   ```bash
   export MODEL_OPTIMIZATION=performance
   export MAX_PREFILL_CHUNK_SIZE=128  # Increase if memory allows
   ```

2. **Check Device Utilization**:
   ```bash
   # Monitor device usage during inference
   tt-smi  # If available
   ```

3. **Adjust Batch Size**:
   ```python
   # In server configuration
   BATCH_SIZE = 32  # Increase for better throughput
   ```

4. **Use Tensor Parallelism**:
   ```bash
   # Enable multi-device parallelism
   unset MESH_DEVICE  # Use all available devices
   ```

### Memory Issues

**Problem**: Out of memory errors during model loading or inference.

**Solutions**:

1. **Reduce Prefill Chunk Size**:
   ```bash
   export MAX_PREFILL_CHUNK_SIZE=16  # Reduce from default 32
   ```

2. **Enable Weight Caching**:
   ```bash
   export TT_CACHE_PATH=/tmp/tt_cache  # Use fast storage
   ```

3. **Use Lower Precision**:
   ```bash
   export MODEL_OPTIMIZATION=accuracy  # Uses more conservative settings
   ```

## Health Check
After deployment, verify the service is running:
```bash
curl https://your-app-url.koyeb.app/health
```

Expected response with TTNN detection:
```json
{
  "status": "healthy",
  "ttnn_available": true,
  "tt_hardware_detected": false,
  "framework": "tt-transformers",
  "model_config": "mistral8b",
  "optimization_level": "performance",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Performance Targets

### Token Generation Rates
- **Easy**: ≥6 tokens/sec/user (CPU fallback mode)
- **Medium**: ≥12 tokens/sec/user (Single device)
- **Hard**: ≥16 tokens/sec/user (Multi-device tensor parallelism)

### Latency Targets
- **First Token Latency**: <2 seconds for 128 token prompts
- **Prefill Throughput**: >1000 tokens/sec for batch processing
- **Memory Usage**: <8GB per device for standard configurations

### Optimization Levels
- **Accuracy Mode**: Prioritizes output quality, slower performance
- **Performance Mode**: Optimized for speed, maintains acceptable quality
- **Custom Config**: Fine-grained control via decoder configuration files

## Key Files

### Core Implementation
- `server.py` - Main HTTP API server with tt-transformers integration
- `/models/tt_transformers/mistral8b/model.py` - Model implementation using shared framework
- `/models/tt_transformers/mistral8b/model_config.py` - Ministral-8B specific configuration

### Deployment Scripts
- `koyeb_deploy_ttnn.sh` - Enhanced deployment script with tt-transformers support
- `requirements.txt` - Updated Python dependencies including tt-transformers
- `Dockerfile.ttnn` - Multi-stage Docker configuration (optional)

### Configuration Files
- `/models/tt_transformers/mistral8b/demo/demo.py` - Reference implementation
- `performance_optimizer.py` - Enhanced performance monitoring with better error handling

## Migration Notes

This deployment now uses the tt-transformers framework, which provides:

1. **Shared Components**: Eliminates code duplication by using shared transformer, attention, and MLP modules
2. **Better Performance**: Optimized implementations with tensor parallelism support
3. **Improved Reliability**: Enhanced error handling and device detection
4. **Easier Maintenance**: Follows established patterns used by other models

### Breaking Changes
- Custom transformer modules have been replaced with shared tt-transformers components
- Weight conversion now uses shared utilities instead of custom logic
- Configuration format updated to match tt-transformers standards

### Backward Compatibility
- HTTP API endpoints remain unchanged
- Environment variables are backward compatible with additional options
- Performance targets are maintained or improved

## Cloud Deployment Notes

### Koyeb Platform
- Uses official TT-Metalium Docker image with pre-built TTNN and tt-transformers
- No local Docker building required
- Automatic scaling based on demand
- Health endpoint reports framework status and hardware detection

### Environment Detection
The system automatically detects deployment environment:
- **Koyeb Cloud**: Uses CPU fallback mode with performance optimization
- **Local Hardware**: Attempts hardware detection with graceful fallback
- **Development**: Supports both hardware and CPU-only testing

### Monitoring and Logging
- Enhanced logging for device initialization and YAML parsing
- Performance metrics collection for optimization
- Error reporting with specific troubleshooting guidance
