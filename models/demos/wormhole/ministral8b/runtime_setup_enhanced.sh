#!/bin/bash
# Enhanced runtime setup script for Ministral-8B on Koyeb
# This script handles progressive setup based on available resources

set -e

echo "🚀 Ministral-8B Enhanced Setup - $(date)"

# Detect runtime phase
PHASE="run"
echo "🚀 Detected RUN phase (build command is empty)"

# Detect the correct workspace structure with improved logic
echo "🔍 Detecting workspace structure..."
echo "Current directory: $(pwd)"
echo "Environment variables:"
echo "  HOME: $HOME"
echo "  USER: $USER"
echo "  PWD: $PWD"

# Enhanced workspace detection
if [ -d "/workspace" ]; then
    WORKSPACE_ROOT="/workspace"
    echo "✅ Using Koyeb workspace: /workspace"
elif [ -d "/app" ]; then
    WORKSPACE_ROOT="/app"
    echo "✅ Using container workspace: /app"
else
    WORKSPACE_ROOT="$(pwd)"
    echo "✅ Using current directory: $WORKSPACE_ROOT"
fi

# Find tt-metal directory with improved search
echo "🔍 Searching for tt-metal directory..."
TT_METAL_PATH=""
for possible_path in \
    "$WORKSPACE_ROOT/tt-metal" \
    "$WORKSPACE_ROOT" \
    "$(pwd)" \
    "/workspace" \
    "/app"; do
    
    if [ -f "$possible_path/build_metal.sh" ]; then
        TT_METAL_PATH="$possible_path"
        echo "✅ Found tt-metal at: $TT_METAL_PATH"
        break
    fi
done

if [ -z "$TT_METAL_PATH" ]; then
    echo "⚠️ Warning: Could not find tt-metal directory with build_metal.sh"
    echo "🔍 Searching for any tt-metal related files..."
    find "$WORKSPACE_ROOT" -name "*.py" -path "*/ministral8b/*" | head -3
    TT_METAL_PATH="$WORKSPACE_ROOT"
fi

# Set up essential environment variables
export TT_METAL_HOME="$TT_METAL_PATH"
export TT_METAL_ROOT="$TT_METAL_PATH" 
export ARCH_NAME="wormhole_b0"
export TT_METAL_ENV_ACTIVATED=1
export IS_KOYEB_ENVIRONMENT="true"
export ENVIRONMENT="$PHASE"

# Set up TTNN directories (improved)
echo "🔧 Setting up TTNN runtime environment..."
mkdir -p "$WORKSPACE_ROOT/ttnn/ttnn/runtime"

# Create a comprehensive setup for TTNN runtime
TTNN_RUNTIME_DIRS=(
    "$TT_METAL_PATH/ttnn/ttnn/runtime"
    "$WORKSPACE_ROOT/ttnn/ttnn/runtime" 
    "$TT_METAL_PATH/tt_metal/ttnn/ttnn/runtime"
)

TTNN_RUNTIME_FOUND=false
for runtime_dir in "${TTNN_RUNTIME_DIRS[@]}"; do
    if [ -d "$runtime_dir" ] && [ "$(ls -A $runtime_dir 2>/dev/null)" ]; then
        echo "✅ Found TTNN runtime at: $runtime_dir"
        cp -r "$runtime_dir"/* "$WORKSPACE_ROOT/ttnn/ttnn/runtime/" 2>/dev/null || true
        TTNN_RUNTIME_FOUND=true
        break
    fi
done

if [ "$TTNN_RUNTIME_FOUND" = "false" ]; then
    echo "⚠️ TTNN runtime not found, creating minimal setup"
    touch "$WORKSPACE_ROOT/ttnn/ttnn/runtime/.placeholder"
    # Create basic Python module structure
    echo "# Placeholder ttnn module for cloud environment" > "$WORKSPACE_ROOT/ttnn/__init__.py"
    echo "# Placeholder ttnn.ttnn module" > "$WORKSPACE_ROOT/ttnn/ttnn/__init__.py"
    echo "# Placeholder ttnn.ttnn.runtime module" > "$WORKSPACE_ROOT/ttnn/ttnn/runtime/__init__.py"
fi

# Enhanced Python path setup
export PYTHONPATH="$TT_METAL_PATH:$WORKSPACE_ROOT:$PYTHONPATH"

# Find server script with comprehensive search
echo "🔍 Locating server script..."
SERVER_SCRIPT=""
for possible_server in \
    "$TT_METAL_PATH/models/demos/wormhole/ministral8b/server.py" \
    "$WORKSPACE_ROOT/models/demos/wormhole/ministral8b/server.py" \
    "$(pwd)/models/demos/wormhole/ministral8b/server.py" \
    "$(pwd)/server.py" \
    "$(find $WORKSPACE_ROOT -name "server.py" -path "*/ministral8b/*" 2>/dev/null | head -1)"; do
    
    if [ -f "$possible_server" ]; then
        SERVER_SCRIPT="$possible_server"
        echo "✅ Found server script: $SERVER_SCRIPT"
        break
    fi
done

if [ -z "$SERVER_SCRIPT" ]; then
    echo "❌ Error: Could not find server.py script"
    echo "🔍 Available Python files:"
    find "$WORKSPACE_ROOT" -name "*.py" | grep -E "(server|ministral)" | head -10
    exit 1
fi

SERVER_DIR="$(dirname $SERVER_SCRIPT)"

# Model configuration (enhanced)
export MODEL_NAME="mistralai/Ministral-8B-Instruct-2410"
export HF_TOKEN=${HF_TOKEN:-""}
export MODEL_CACHE_PATH=${MODEL_CACHE_PATH:-"$WORKSPACE_ROOT/tt_models/ministral8b"}

# Set Ministral-specific environment variables
export MINISTRAL_CKPT_DIR="$MODEL_CACHE_PATH"
export MINISTRAL_TOKENIZER_PATH="$MODEL_CACHE_PATH"
export MINISTRAL_CACHE_PATH="$MODEL_CACHE_PATH"

# Port configuration
export PORT=${PORT:-8000}

# Enhanced debugging setup
export KOYEB_SKIP_MODEL_LOAD="true"  # Skip actual TT model loading in cloud
export TT_METAL_LOGGER_LEVEL="INFO"

echo "📊 Environment Summary:"
echo "   TT_METAL_ROOT: $TT_METAL_ROOT"
echo "   WORKSPACE_ROOT: $WORKSPACE_ROOT"
echo "   SERVER_SCRIPT: $SERVER_SCRIPT"
echo "   MODEL_CACHE_PATH: $MODEL_CACHE_PATH"
echo "   PYTHONPATH: $(echo $PYTHONPATH | cut -c1-100)..."
echo "   IS_KOYEB_ENVIRONMENT: $IS_KOYEB_ENVIRONMENT"
echo "   KOYEB_SKIP_MODEL_LOAD: $KOYEB_SKIP_MODEL_LOAD"

# Start the server
echo "🚀 Starting Ministral-8B server..."
cd "$SERVER_DIR"

echo "📍 Working directory: $(pwd)"
echo "🐍 Python version: $(python --version 2>&1)"
echo "📦 Python executable: $(which python)"

# Check if we can import basic modules
echo "🧪 Testing Python imports..."
python -c "import sys; print('Python path entries:', len(sys.path))" || echo "⚠️ Python import test failed"
python -c "import os; print('Working dir accessible:', os.path.exists('.'))" || echo "⚠️ OS import test failed"

# Final server startup with enhanced error handling
echo "🎯 Executing: python server.py --port $PORT --instruct"
exec python server.py --port "$PORT" --instruct
