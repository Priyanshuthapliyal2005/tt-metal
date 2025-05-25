#!/bin/bash
# Build script for TT-Metal Ministral-8B deployment

set -e

echo "🔧 Building TT-Metal Ministral-8B Docker image..."

# Build the Docker image
docker build -f models/demos/wormhole/ministral8b/Dockerfile.ttnn -t ministral8b:latest .

echo "✅ Build completed successfully!"
echo "🚀 To run locally: docker run -p 8000:8000 -e ARCH_NAME=wormhole_b0 ministral8b:latest"
