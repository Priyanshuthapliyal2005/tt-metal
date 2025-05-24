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

# Detect the correct workspace structure
echo "🔍 Detecting workspace structure..."
echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

# Find the correct paths
if [ -d "/workspace" ]; then
    WORKSPACE_ROOT="/workspace"
elif [ -d "/app" ]; then
    WORKSPACE_ROOT="/app"
else
    WORKSPACE_ROOT="$(pwd)"
fi

echo "Using workspace root: $WORKSPACE_ROOT"

# Find tt-metal directory
if [ -d "$WORKSPACE_ROOT/tt-metal" ]; then
    TT_METAL_PATH="$WORKSPACE_ROOT/tt-metal"
elif [ -d "$WORKSPACE_ROOT" ] && [ -f "$WORKSPACE_ROOT/build_metal.sh" ]; then
    TT_METAL_PATH="$WORKSPACE_ROOT"
elif [ -d "$(pwd)" ] && [ -f "$(pwd)/build_metal.sh" ]; then
    TT_METAL_PATH="$(pwd)"
else
    echo "❌ Error: Could not find tt-metal directory"
    echo "Searching for tt-metal..."
    find $WORKSPACE_ROOT -name "build_metal.sh" -type f 2>/dev/null | head -5
    TT_METAL_PATH="$WORKSPACE_ROOT"
fi

echo "Using TT-Metal path: $TT_METAL_PATH"

# Set up ttnn runtime directories
echo "🔧 Setting up ttnn runtime directories..."
mkdir -p $WORKSPACE_ROOT/ttnn/ttnn/runtime

# Try to find and copy ttnn runtime files from various possible locations
TTNN_RUNTIME_FOUND=false
for possible_path in \
    "$TT_METAL_PATH/ttnn/ttnn/runtime" \
    "$WORKSPACE_ROOT/ttnn/ttnn/runtime" \
    "$TT_METAL_PATH/tt_metal/ttnn/ttnn/runtime" \
    "$(pwd)/ttnn/ttnn/runtime"; do
    
    if [ -d "$possible_path" ] && [ "$(ls -A $possible_path 2>/dev/null)" ]; then
        echo "Found TTNN runtime at: $possible_path"
        cp -r $possible_path/* $WORKSPACE_ROOT/ttnn/ttnn/runtime/ || true
        TTNN_RUNTIME_FOUND=true
        break
    fi
done

if [ "$TTNN_RUNTIME_FOUND" = "true" ]; then
    echo "✅ TTNN runtime files copied successfully"
else
    echo "⚠️ Warning: TTNN runtime directory not found, creating placeholder"
    touch $WORKSPACE_ROOT/ttnn/ttnn/runtime/.placeholder
fi

# Set environment variables
export TT_METAL_HOME="$TT_METAL_PATH"
export TT_METAL_ROOT="$TT_METAL_PATH" 
export ARCH_NAME="wormhole_b0"
export TT_METAL_ENV_ACTIVATED=1
export IS_KOYEB_ENVIRONMENT="true"
export ENVIRONMENT="$PHASE"
export KOYEB_SKIP_MODEL_LOAD="true"

# Find the server script
SERVER_SCRIPT=""
for possible_server in \
    "$TT_METAL_PATH/models/demos/wormhole/ministral8b/server.py" \
    "$WORKSPACE_ROOT/models/demos/wormhole/ministral8b/server.py" \
    "$(pwd)/models/demos/wormhole/ministral8b/server.py" \
    "$(pwd)/server.py"; do
    
    if [ -f "$possible_server" ]; then
        SERVER_SCRIPT="$possible_server"
        break
    fi
done

if [ -z "$SERVER_SCRIPT" ]; then
    echo "❌ Error: Could not find server.py script"
    echo "Searching for server.py..."
    find $WORKSPACE_ROOT -name "server.py" -type f 2>/dev/null | head -5
    exit 1
fi

echo "Found server script at: $SERVER_SCRIPT"
SERVER_DIR="$(dirname $SERVER_SCRIPT)"

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
    cd "$SERVER_DIR"
    
    # Log environment for debugging
    echo "📊 Environment info:"
    echo "   Working directory: $(pwd)"
    echo "   Python version: $(python --version 2>&1)"
    echo "   TT_METAL_HOME: $TT_METAL_HOME"
    echo "   TT_METAL_ROOT: $TT_METAL_ROOT"
    echo "   TTNN runtime directory exists: $([ -d $WORKSPACE_ROOT/ttnn/ttnn/runtime ] && echo 'Yes' || echo 'No')"
    echo "   Server script: $SERVER_SCRIPT"
    
    # Start server with proper error handling
    echo "🎯 Executing: python server.py --port 8000 --instruct"
    exec python server.py --port 8000 --instruct
fi
