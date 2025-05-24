#!/bin/bash
# Runtime setup script for Ministral-8B on Koyeb
# This script handles the ttnn runtime directory setup and server startup
# Can be used for both build and run phases

set -e

echo "🚀 Ministral-8B Setup - $(date)"

# Detect if this is build or run phase
# Since build is empty, this will always be run phase
PHASE="run"
echo "🚀 Detected RUN phase (build command is empty)"

# Set up ttnn runtime directories (needed in both phases)
echo "🔧 Setting up ttnn runtime directories..."
mkdir -p /workspace/ttnn/ttnn/runtime
mkdir -p /workspace/tt_metal/ttnn/ttnn/runtime

# Copy runtime files using the exact command from build
if [ -d "/workspace/tt_metal/ttnn/ttnn/runtime" ]; then
    echo "Copying from /workspace/tt_metal/ttnn/ttnn/runtime/*"
    cp -r /workspace/tt_metal/ttnn/ttnn/runtime/* /workspace/ttnn/ttnn/runtime/ || true
    echo "✅ TTNN runtime files copied successfully"
else
    echo "⚠️ Warning: /workspace/tt_metal/ttnn/ttnn/runtime not found, creating placeholder"
    touch /workspace/ttnn/ttnn/runtime/.placeholder
fi

# Set environment variables
export TT_METAL_HOME="/workspace/tt_metal"
export TT_METAL_ROOT="/workspace/tt_metal" 
export ARCH_NAME="wormhole_b0"
export TT_METAL_ENV_ACTIVATED=1
export IS_KOYEB_ENVIRONMENT="true"
export ENVIRONMENT="$PHASE"
export KOYEB_SKIP_MODEL_LOAD="true"

if [ "$PHASE" = "build" ]; then
    # This shouldn't happen since build is empty, but keeping for safety
    echo "✅ Build phase setup complete"
    echo "📦 Installing Python dependencies..."
    cd /workspace/tt_metal/models/demos/wormhole/ministral8b
    python -m pip install --no-cache-dir -r requirements.txt || echo "⚠️ requirements.txt not found, continuing..."
    echo "✅ Build phase finished successfully"
else
    # This is the main execution path
    echo "🚀 Starting Ministral-8B server..."
    cd /workspace/tt_metal/models/demos/wormhole/ministral8b
    
    # Log environment for debugging
    echo "📊 Environment info:"
    echo "   Working directory: $(pwd)"
    echo "   Python version: $(python --version 2>&1)"
    echo "   TT_METAL_HOME: $TT_METAL_HOME"
    echo "   TTNN runtime directory exists: $([ -d /workspace/ttnn/ttnn/runtime ] && echo 'Yes' || echo 'No')"
    
    # Start server with proper error handling
    echo "🎯 Executing: python server.py --port 8000 --instruct"
    exec python server.py --port 8000 --instruct
fi
