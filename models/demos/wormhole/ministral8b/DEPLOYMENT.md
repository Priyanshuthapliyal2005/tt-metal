# Ministral-8B Deployment Guide

## Quick Deployment Commands

### Option 1: Direct Koyeb Deployment (Recommended)
```bash
# Deploy directly using the enhanced deployment script
./koyeb_deploy_ttnn.sh
```

### Option 2: Manual Koyeb Deployment
```bash
# Create the service
koyeb service create ministral-8b-app \
  --docker ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc \
  --docker-command "bash" \
  --docker-args "-c,pip install flask requests transformers torch && python server.py" \
  --ports 8000:http \
  --regions fra \
  --instance-type nano \
  --env TT_METAL_HOME=/opt/tt-metal \
  --env PYTHONPATH=/opt/tt-metal \
  --env ARCH_NAME=wormhole_b0
```

### Option 3: Local Testing
```bash
# Pull and run the official TT-Metalium Docker image
docker pull ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc

# Run locally for testing
docker run -p 8000:8000 \
  -e TT_METAL_HOME=/opt/tt-metal \
  -e PYTHONPATH=/opt/tt-metal \
  -e ARCH_NAME=wormhole_b0 \
  -v $(pwd):/app \
  -w /app \
  ghcr.io/tenstorrent/tt-metal/tt-metalium-ubuntu-22.04-release-amd64:latest-rc \
  bash -c "pip install -r requirements-docker.txt && python server.py"
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
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Key Files
- `server.py` - Main HTTP API server with TTNN detection
- `koyeb_deploy_ttnn.sh` - Enhanced deployment script
- `requirements-docker.txt` - Python dependencies for Docker
- `Dockerfile.ttnn` - Multi-stage Docker configuration (optional)

## Performance Targets
- **Easy**: ≥6 tokens/sec/user
- **Medium**: ≥12 tokens/sec/user  
- **Hard**: ≥16 tokens/sec/user

## Notes
- Uses official TT-Metalium Docker image with pre-built TTNN
- No local Docker building required
- Deployment targets Koyeb cloud platform
- Health endpoint reports TTNN availability and hardware status
