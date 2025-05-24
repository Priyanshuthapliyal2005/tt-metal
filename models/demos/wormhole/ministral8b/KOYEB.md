# Ministral-8B Koyeb Deployment Guide

## Quick Start

1. Deploy using the universal script:
   ```bash
   ./koyeb_deploy.sh
   ```
   The script automatically detects build vs runtime phase.

2. For manual deployment:
   - Copy `koyeb_config_fixed.json` to your Koyeb configuration
   - Ensure all required files are present

## Deployment Process

### 1. Build Phase
- Repository is cloned to `/builder/workspace`
- Dependencies are installed from `requirements-fixed.txt`
- Code is compiled and validated
- Script exits cleanly to avoid build timeout

### 2. Runtime Phase
- Model server starts on N300 hardware
- Loads model (or uses mock responses for health checks)
- API available on configured port (default: 8000)

## Recent Fixes & Improvements

1. **Python Version Compatibility**
   - Build: Python 3.11
   - Runtime: Python 3.10
   - Uses version-compatible dependencies
   - Automatic version detection and adaptation

2. **Build Timeout Prevention**
   - Detects Koyeb builder environment
   - Skips server startup during build
   - Creates placeholder files for testing
   - Validates configuration without hanging

3. **Environment Configuration**
   - `MINISTRAL_CKPT_DIR`: Model checkpoints
   - `MINISTRAL_TOKENIZER_PATH`: Tokenizer files
   - `MINISTRAL_CACHE_PATH`: Cache directory
   - `KOYEB_SKIP_MODEL_LOAD`: Testing mode
   - `PORT`: Server port (default: 8000)

4. **Error Handling**
   - Graceful handling of missing files
   - Mock responses for health checks
   - Automatic recovery from common issues
   - Clear error messages and logging

## API Endpoints

### Health Check
```http
GET /health
```
Returns: Server status, available devices, uptime

### Text Generation
```http
POST /generate
Content-Type: application/json

{
    "prompt": "Your text here",
    "max_tokens": 128,
    "temperature": 0.7
}
```

## Troubleshooting

### Common Issues

1. **Build Times Out**
   - Symptom: Build phase exceeds time limit
   - Fix: Script now detects build phase and exits cleanly

2. **Python Version Mismatch**
   - Symptom: Module import errors
   - Fix: Using version-compatible dependencies

3. **Missing Model Files**
   - Symptom: File not found errors
   - Fix: Script creates placeholders for testing

4. **Server Won't Start**
   - Symptom: Import or runtime errors
   - Fix: Check logs for specific error messages
   - Solution: Use `KOYEB_SKIP_MODEL_LOAD=true` for testing

### Validation

Use the health check endpoint to verify deployment:
```bash
curl http://your-app.koyeb.app/health
```

## Directory Structure

```
ministral8b/
├── demo/               # Model demo scripts
├── tt/                 # Model implementation
├── tests/             # Test files
├── koyeb_deploy.sh    # Universal deployment script
├── server.py          # API server
├── requirements.txt   # Standard dependencies
└── requirements-fixed.txt  # Version-specific deps
```
