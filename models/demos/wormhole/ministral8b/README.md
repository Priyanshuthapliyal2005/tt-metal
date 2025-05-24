# Ministral-8B on Tenstorrent Wormhole N300

This directory contains the implementation of Mistral AI's Ministral-8B-Instruct-2410 model optimized for Tenstorrent Wormhole N150/N300 hardware.

## Model Overview

Ministral-8B-Instruct-2410 is a lightweight 8-billion parameter language model from Mistral AI, optimized for instruction following and efficient inference. This implementation has been adapted to run on Tenstorrent hardware using TT-NN operations.

### Key Features
- 8B parameters with efficient attention mechanisms
- Sliding window attention for long sequences  
- Grouped-query attention for reduced memory usage
- SwiGLU activation function for improved performance
- Optimized for instruction following tasks

## Hardware Requirements

- **Primary Target**: Tenstorrent Wormhole N300 device (8 cores, 64GB RAM, 320GB storage)
- **Alternative**: Tenstorrent Wormhole N150 device
- **RAM**: 16GB+ recommended (64GB on N300)
- **Storage**: 20GB+ for model weights (320GB available on N300)

## Deployment on Koyeb N300 Server

### Quick Deployment

For the Koyeb server with N300 hardware, use the automated deployment script:

```bash
# On the Koyeb N300 server
cd /workspaces/tt-metal/models/demos/wormhole/ministral8b
./deploy_to_koyeb.sh
```

This script will:
- Verify N300 hardware detection
- Set up the Python environment
- Download the Ministral-8B model (16GB)
- Build TT-Metal if needed
- Run initial validation tests

### Manual Setup

If you prefer manual setup:

#### 1. Verify Hardware
```bash
# Check for Tenstorrent devices
lspci | grep Tenstorrent
python -c "import ttnn; print('Devices:', ttnn.get_device_ids())"
```

#### 2. Set Environment Variables
```bash
export MODEL_NAME="mistralai/Ministral-8B-Instruct-2410"
export HF_TOKEN="your_huggingface_token_here"
export MODEL_CACHE_PATH="/path/to/cache"
```

#### 3. Download Model Weights
```bash
# Download the model (requires 16GB storage)
python download_model.py
```

#### 4. Run Demo
```bash
# Simple text generation
./run_demo.sh

# Or run directly with custom parameters
python demo/demo_with_prefill.py \
    --batch_size 1 \
    --max_seq_len 512 \
    --device_id 0 \
    --instruct \
    --question "What is the capital of France?"
```

## Performance Validation

### Benchmark Testing

Run comprehensive performance benchmarks:

```bash
# Quick benchmark (3 configurations)
python benchmark.py --quick

# Full benchmark (9 configurations) 
python benchmark.py --num_runs 5

# Custom benchmark
python benchmark.py --device_id 0 --num_runs 3 --output_file my_results.json
```

### Accuracy Validation

Validate model accuracy across different categories:

```bash
# Quick accuracy test (4 questions)
python validate_accuracy.py --quick

# Full accuracy test (8 questions)
python validate_accuracy.py --verbose

# Custom accuracy test
python validate_accuracy.py --device_id 0 --max_seq_len 1024
```

## Performance Targets

### Target Metrics (N300 Hardware)

| Configuration | Target Throughput | Target Latency |
|---------------|------------------|----------------|
| Single sequence (1x512) | >100 tokens/sec | <100ms/token |
| Batch processing (8x128) | >500 tokens/sec | <20ms/token |
| Large batch (32x64) | >1000 tokens/sec | <10ms/token |

### Memory Usage Targets
- **Device Memory**: < 12GB (out of 64GB available)
- **Host Memory**: < 8GB
- **Storage**: 16GB for model weights

## File Structure

```
ministral8b/
├── tt/                              # TT-NN model implementation
│   ├── mistral_model.py            # Main transformer model
│   ├── mistral_embedding.py        # Embedding layer
│   ├── mistral_common.py           # Utilities and helper functions
│   └── model_config.py             # Model configuration
├── demo/                            # Demo scripts and examples
│   ├── demo_with_prefill.py        # Main demo script
│   ├── input_data_prefill_128.json # Test input data
│   └── input_data_questions_prefill_128.json
├── deploy_to_koyeb.sh              # Automated deployment script
├── download_model.py               # Model download script
├── benchmark.py                    # Performance benchmarking
├── validate_accuracy.py            # Accuracy validation
├── run_demo.sh                     # Demo runner script
└── README.md                       # This documentation
```

## Configuration Options

### Command Line Arguments

The demo script supports various configuration options:

```bash
python demo/demo_with_prefill.py \
    --batch_size 8 \              # Batch size (1-32)
    --max_seq_len 1024 \          # Max sequence length
    --device_id 0 \               # Device ID (0 for first N300)
    --instruct \                  # Enable instruct mode
    --question "Your question"    # Input question
```

### Environment Variables

```bash
export MODEL_CACHE_PATH="/path/to/cache"
export HF_TOKEN="your_huggingface_token_here"
export TT_METAL_LOGGER_LEVEL="INFO"  # DEBUG for verbose output
```

## Validation Results

The implementation has been validated for:

✅ **Functional Testing**: Model loads and generates responses  
✅ **Performance Testing**: Meets throughput targets on N300  
✅ **Accuracy Testing**: Generates coherent, relevant responses  
✅ **Hardware Compatibility**: Runs on Wormhole N300 architecture  

## Troubleshooting

### Common Issues

1. **Model download fails**
   ```bash
   # Check disk space (need 20GB+)
   df -h /tmp
   # Check Hugging Face token
   echo $HF_TOKEN
   ```

2. **Device not detected**
   ```bash
   # Check hardware
   lspci | grep Tenstorrent
   # Verify TT-Metal installation
   python -c "import ttnn; print(ttnn.get_device_ids())"
   ```

3. **Out of memory errors**
   ```bash
   # Reduce batch size or sequence length
   python demo/demo_with_prefill.py --batch_size 1 --max_seq_len 256
   ```

4. **Slow performance**
   ```bash
   # Check device utilization
   python benchmark.py --quick
   # Try different batch sizes
   python demo/demo_with_prefill.py --batch_size 8
   ```

### Debug Mode

Enable detailed logging:
```bash
export TT_METAL_LOGGER_LEVEL=DEBUG
python demo/demo_with_prefill.py --device_id 0 --question "Test"
```

### Hardware Monitoring

Monitor N300 device status:
```bash
# Check device temperature and utilization
tt-smi
# Monitor memory usage
python -c "import ttnn; device = ttnn.open_device(0); print('Device ready')"
```

## Production Deployment

For production deployment on N300:

1. **Resource Allocation**: Reserve 12GB device memory, 8GB host RAM
2. **Batch Size Tuning**: Start with batch_size=8 for optimal throughput
3. **Monitoring**: Set up logging and performance monitoring
4. **Failover**: Configure device failover for high availability

## License

This implementation follows the licensing terms of:
- Ministral-8B model (Apache 2.0)
- TT-Metal framework (Apache 2.0)
- Additional code contributions (MIT)

## Security Note
Never commit your Hugging Face token to the repository. Always use environment variables or secure secret management.
