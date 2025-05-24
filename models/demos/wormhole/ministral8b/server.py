#!/usr/bin/env python3
"""
Server script for running Ministral-8B as an API endpoint.
When deployed on Koyeb, this provides a REST API for model inference.
"""

import argparse
import json
import logging
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List
import time
import threading

# Add the necessary paths to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=os.environ.get('TT_METAL_LOGGER_LEVEL', 'INFO'),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("ministral-server")

# Log environment information for diagnostic purposes
logger.info(f"Starting Ministral-8B server in environment: {os.environ.get('ENVIRONMENT', 'unknown')}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"MODEL_CACHE_PATH: {os.environ.get('MODEL_CACHE_PATH', 'not set')}")

# Global variables
MODEL = None
TOKENIZER = None
DEVICE_ID = 0
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
INSTRUCT_MODE = True

class ModelRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Ministral-8B model API."""
    
    def _set_headers(self, status_code=200, content_type="application/json"):
        self.send_response(status_code)
        self.send_header("Content-type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        
    def do_OPTIONS(self):
        self._set_headers()
        
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            # Health check endpoint
            try:
                import ttnn
                devices = ttnn.get_device_ids()
                health_status = {
                    "status": "ok" if len(devices) > 0 else "warning",
                    "model": "ministral8b",
                    "devices": list(map(str, devices)),
                    "uptime": time.time() - SERVER_START_TIME,
                    "memory": self._get_memory_usage(),
                    "batch_size": BATCH_SIZE,
                    "max_seq_len": MAX_SEQ_LEN,
                    "instruct_mode": INSTRUCT_MODE
                }
                self._set_headers()
                self.wfile.write(json.dumps(health_status).encode())
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                error_response = {
                    "status": "error", 
                    "error": str(e)
                }
                self._set_headers(status_code=500)
                self.wfile.write(json.dumps(error_response).encode())
        else:
            # Invalid endpoint
            self._set_headers(status_code=404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/generate":
            # Text generation endpoint
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            
            try:
                request = json.loads(post_data.decode("utf-8"))
                response = self._process_generation_request(request)
                self._set_headers()
                self.wfile.write(json.dumps(response).encode())
            except json.JSONDecodeError:
                self._set_headers(status_code=400)
                self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            except Exception as e:
                logger.error(f"Generation error: {e}")
                self._set_headers(status_code=500)
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            # Invalid endpoint
            self._set_headers(status_code=404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())
    
    def _process_generation_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a text generation request."""
        # Get request parameters
        prompt = request.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided"}
            
        max_tokens = min(int(request.get("max_tokens", 128)), 1024)
        temperature = float(request.get("temperature", 0.7))
        
        logger.info(f"Generating response for prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        start_time = time.time()
        
        # Check if model is loaded
        global MODEL, TOKENIZER
        if MODEL is None or TOKENIZER is None:
            # In Koyeb environment, return a mock response for health checks
            if os.environ.get('ENVIRONMENT') == 'runtime':
                logger.warning("Model not loaded. Returning mock response for health check.")
                mock_response = "[HEALTH CHECK] Server is running but model is not loaded. This is a mock response."
                return {
                    "text": mock_response,
                    "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(mock_response.split()), "total_tokens": len(prompt.split()) + len(mock_response.split())},
                    "model": "ministral-8b-koyeb-mock",
                    "status": "ok-mock",
                    "generation_time": 0.1
                }
            else:
                raise Exception("Model or tokenizer not loaded. Please wait for initialization to complete.")
                
        try:
            # Import here to avoid loading model until needed
            from demo.demo_with_prefill import process_question
            
            # Process the question using our demo script
            response = process_question(
                question=prompt,
                batch_size=BATCH_SIZE,
                max_seq_len=min(max_tokens, MAX_SEQ_LEN),
                device_id=DEVICE_ID,
                instruct=INSTRUCT_MODE,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            # Return a graceful error in Koyeb environment
            if os.environ.get('ENVIRONMENT') == 'runtime':
                return {
                    "text": f"[ERROR] Failed to generate response: {str(e)}",
                    "error": str(e),
                    "model": "ministral-8b-koyeb",
                    "status": "error"
                }
            else:
                raise
        
        generation_time = time.time() - start_time
        tokens_generated = len(response.split()) # Approximate
        
        logger.info(f"Generated {tokens_generated} tokens in {generation_time:.2f} seconds")
        
        return {
            "text": response,
            "usage": {
                "prompt_tokens": len(prompt.split()),  # Approximate
                "completion_tokens": tokens_generated,
                "total_tokens": len(prompt.split()) + tokens_generated
            },
            "model": "ministral-8b-instruct-2410",
            "generation_time": generation_time
        }
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024)
            }
        except Exception:
            return {"error": "Could not get memory usage"}

def preload_model():
    """Preload model into memory."""
    logger.info("Preloading model into memory...")
    try:
        global MODEL, TOKENIZER
        
        # Check if we're in Koyeb environment
        is_koyeb = os.environ.get('ENVIRONMENT') == 'runtime'
        
        if is_koyeb and os.environ.get('KOYEB_SKIP_MODEL_LOAD') == 'true':
            logger.warning("Skipping model load in Koyeb environment (KOYEB_SKIP_MODEL_LOAD=true)")
            logger.info("This is useful for testing server startup without model loading")
            return
            
        try:
            from demo.demo_with_prefill import load_model_and_tokenizer
            
            logger.info(f"Loading model with device_id={DEVICE_ID}, batch_size={BATCH_SIZE}, max_seq_len={MAX_SEQ_LEN}")
            MODEL, TOKENIZER = load_model_and_tokenizer(
                device_id=DEVICE_ID,
                batch_size=BATCH_SIZE,
                max_seq_len=MAX_SEQ_LEN,
                instruct=INSTRUCT_MODE
            )
            logger.info("Model loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import load_model_and_tokenizer: {e}")
            if is_koyeb:
                logger.warning("Continuing server startup for health checks in Koyeb environment")
            else:
                raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Log the detailed error for debugging
        import traceback
        logger.error(f"Detailed error: {traceback.format_exc()}")
        # Don't exit, let the server start anyway for health checks

def run_server(port=8080, preload=True):
    """Run the HTTP server."""
    global SERVER_START_TIME
    SERVER_START_TIME = time.time()
    
    if preload:
        # Preload model in a separate thread
        threading.Thread(target=preload_model).start()
    
    server_address = ("", port)
    httpd = HTTPServer(server_address, ModelRequestHandler)
    logger.info(f"Starting Ministral-8B server on port {port}")
    httpd.serve_forever()

def main():
    parser = argparse.ArgumentParser(description="Ministral-8B API Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--instruct", action="store_true", help="Use instruct mode")
    parser.add_argument("--no-preload", action="store_true", help="Don't preload model at startup")
    
    args = parser.parse_args()
    
    global DEVICE_ID, BATCH_SIZE, MAX_SEQ_LEN, INSTRUCT_MODE
    DEVICE_ID = args.device_id
    BATCH_SIZE = args.batch_size
    MAX_SEQ_LEN = args.max_seq_len
    INSTRUCT_MODE = args.instruct
    
    run_server(port=args.port, preload=not args.no_preload)

if __name__ == "__main__":
    main()
