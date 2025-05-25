#!/bin/bash
# Enhanced Koyeb deployment script using official TT-Metalium Docker image
# Based on official documentation: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/installing.html

set -e

echo "🚀 Starting Koyeb deployment with official TT-Metalium Docker image..."

# Configuration
APP_NAME="ministral-8b-ttnn"
SERVICE_NAME="ministral-8b-service"
DOCKER_IMAGE_TAG="ministral-8b:ttnn-$(date +%s)"

echo "📋 Deployment Configuration:"
echo "  App Name: $APP_NAME"
echo "  Service Name: $SERVICE_NAME"
echo "  Docker Tag: $DOCKER_IMAGE_TAG"

# Step 1: Build Docker image using official TT-Metalium base
echo "🐋 Building Docker image with official TT-Metalium base..."
docker build -f Dockerfile.ttnn -t $DOCKER_IMAGE_TAG .

# Step 2: Test the image locally (optional)
echo "🧪 Testing Docker image locally..."
echo "Starting container for health check..."
CONTAINER_ID=$(docker run -d -p 8000:8000 \
    -e IS_DOCKER_ENVIRONMENT=true \
    -e ENVIRONMENT=test \
    $DOCKER_IMAGE_TAG)

# Wait for container to start
sleep 10

# Health check
echo "Performing health check..."
if curl -f http://localhost:8000/health; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    docker logs $CONTAINER_ID
    docker stop $CONTAINER_ID
    exit 1
fi

# Stop test container
docker stop $CONTAINER_ID
echo "🧹 Cleaned up test container"

# Step 3: Deploy to Koyeb using Docker
echo "☁️ Deploying to Koyeb..."

# Check if koyeb CLI is installed
if ! command -v koyeb &> /dev/null; then
    echo "❌ Koyeb CLI not found. Please install it first:"
    echo "   curl -s https://binaries.koyeb.com/install.sh | bash"
    exit 1
fi

# Deploy using Docker
koyeb app init $APP_NAME --docker $DOCKER_IMAGE_TAG || echo "App may already exist"

# Create or update service
koyeb service create $SERVICE_NAME \
    --app $APP_NAME \
    --docker $DOCKER_IMAGE_TAG \
    --ports 8000:http \
    --regions fra \
    --instance-type nano \
    --env IS_DOCKER_ENVIRONMENT=true \
    --env ENVIRONMENT=production \
    --env TT_METAL_HOME=/workspace \
    --env PYTHONPATH=/workspace:/workspace/models \
    --env ARCH_NAME=wormhole_b0 \
    --env MODEL_CACHE_PATH=/workspace/model_cache \
    --health-check-http-port 8000 \
    --health-check-http-path /health \
    --min-scale 1 \
    --max-scale 3 || \
koyeb service update $SERVICE_NAME \
    --docker $DOCKER_IMAGE_TAG \
    --env IS_DOCKER_ENVIRONMENT=true \
    --env ENVIRONMENT=production \
    --env TT_METAL_HOME=/workspace \
    --env PYTHONPATH=/workspace:/workspace/models \
    --env ARCH_NAME=wormhole_b0 \
    --env MODEL_CACHE_PATH=/workspace/model_cache

echo "⏳ Waiting for deployment to complete..."
sleep 30

# Get the deployment URL
DEPLOYMENT_URL=$(koyeb service get $SERVICE_NAME --output json | jq -r '.public_domain')

if [ "$DEPLOYMENT_URL" != "null" ] && [ -n "$DEPLOYMENT_URL" ]; then
    echo "🎉 Deployment successful!"
    echo "🌐 Application URL: https://$DEPLOYMENT_URL"
    
    # Test the deployed application
    echo "🧪 Testing deployed application..."
    sleep 10
    
    if curl -f "https://$DEPLOYMENT_URL/health"; then
        echo "✅ Deployment health check passed"
    else
        echo "⚠️ Deployment health check failed, but deployment is complete"
    fi
    
    echo ""
    echo "🔗 API Endpoints:"
    echo "  Health: https://$DEPLOYMENT_URL/health"
    echo "  Generate: https://$DEPLOYMENT_URL/generate"
    echo ""
    echo "📝 Example usage:"
    echo "curl -X POST https://$DEPLOYMENT_URL/generate \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"prompt\": \"What is artificial intelligence?\", \"max_tokens\": 100}'"
    
else
    echo "❌ Failed to get deployment URL"
    echo "Check deployment status with: koyeb service get $SERVICE_NAME"
    exit 1
fi

echo "✨ Deployment completed successfully with official TT-Metalium Docker image!"
