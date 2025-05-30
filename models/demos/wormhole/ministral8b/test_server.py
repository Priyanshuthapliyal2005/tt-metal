#!/usr/bin/env python3
"""
Test script for the Ministral-8B server.
This tests the server endpoints locally.
"""

import requests
import json
import time
import os
import sys

def test_health_endpoint(base_url):
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Health endpoint working!")
            print(f"   Status: {data.get('status')}")
            print(f"   Environment: {data.get('environment')}")
            print(f"   TTNN Available: {data.get('ttnn_available')}")
            print(f"   Model Loaded: {data.get('model_loaded')}")
            if data.get('import_error'):
                print(f"   Import Error: {data.get('import_error')}")
            return True
        else:
            print(f"Health endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_root_endpoint(base_url):
    """Test the root endpoint."""
    print("🏠 Testing root endpoint...")
    try:
        response = requests.get(base_url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Root endpoint working!")
            print(f"   Content length: {len(response.text)} characters")
            return True
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_generate_endpoint(base_url):
    """Test the generate endpoint."""
    print("🤖 Testing generate endpoint...")
    try:
        payload = {
            "prompt": "What is artificial intelligence?",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Generate endpoint working!")
            print(f"   Generated text: {data.get('text', '')[:100]}...")
            print(f"   Model: {data.get('model')}")
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print(f"❌ Generate endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Generate endpoint error: {e}")
        return False

def main():
    print("🧪 Ministral-8B Server Test Suite")
    print("=" * 50)
    
    # Test both local and deployed URLs
    urls_to_test = [
        "http://localhost:8000",  # Local development
        "https://ministral-8b-priyanshuthapliyal2005-40bb59f6.koyeb.app"  # Deployed
    ]
    
    for base_url in urls_to_test:
        print(f"\n🌐 Testing server at: {base_url}")
        print("-" * 50)
        
        # Test endpoints
        results = []
        results.append(test_root_endpoint(base_url))
        results.append(test_health_endpoint(base_url))
        results.append(test_generate_endpoint(base_url))
        
        # Summary
        passed = sum(results)
        total = len(results)
        print(f"\n📊 Results for {base_url}:")
        print(f"   Passed: {passed}/{total}")
        
        if passed == total:
            print("   ✅ All tests passed!")
        else:
            print("   ⚠️ Some tests failed")

if __name__ == "__main__":
    main()

import requests
import json
import time
import os
import sys
import tempfile
import pytest

def test_health_endpoint(base_url):
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Health endpoint working!")
            print(f"   Status: {data.get('status')}")
            print(f"   Environment: {data.get('environment')}")
            print(f"   TTNN Available: {data.get('ttnn_available')}")
            print(f"   Model Loaded: {data.get('model_loaded')}")
            if data.get('import_error'):
                print(f"   Import Error: {data.get('import_error')}")
            return True
        else:
            print(f"Health endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_root_endpoint(base_url):
    """Test the root endpoint."""
    print("🏠 Testing root endpoint...")
    try:
        response = requests.get(base_url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Root endpoint working!")
            print(f"   Content length: {len(response.text)} characters")
            return True
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_generate_endpoint(base_url):
    """Test the generate endpoint."""
    print("🤖 Testing generate endpoint...")
    try:
        payload = {
            "prompt": "What is artificial intelligence?",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Generate endpoint working!")
            print(f"   Generated text: {data.get('text', '')[:100]}...")
            print(f"   Model: {data.get('model')}")
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print(f"❌ Generate endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Generate endpoint error: {e}")
        return False

def main():
    print("🧪 Ministral-8B Server Test Suite")
    print("=" * 50)
    
    # Test both local and deployed URLs
    urls_to_test = [
        "http://localhost:8000",  # Local development
        "https://ministral-8b-priyanshuthapliyal2005-40bb59f6.koyeb.app"  # Deployed
    ]
    
    for base_url in urls_to_test:
        print(f"\n🌐 Testing server at: {base_url}")
        print("-" * 50)
        
        # Test endpoints
        results = []
        results.append(test_root_endpoint(base_url))
        results.append(test_health_endpoint(base_url))
        results.append(test_generate_endpoint(base_url))
        
        # Summary
        passed = sum(results)
        total = len(results)
        print(f"\n📊 Results for {base_url}:")
        print(f"   Passed: {passed}/{total}")
        
        if passed == total:
            print("   ✅ All tests passed!")
        else:
            print("   ⚠️ Some tests failed")

def test_loader_and_firmware_precompile(tmp_path=None):
    """
    Smoke test to verify loader signature and firmware availability.
    Tests that MemoryOptimizedLoader accepts chunk_size_mb and TTNN can open device.
    """
    print("🔧 Testing loader and firmware precompile...")
    
    # Use provided tmp_path or create temporary directory
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    
    try:
        # Test 1: Verify MemoryOptimizedLoader constructor with chunk_size_mb works
        print("   Testing MemoryOptimizedLoader with chunk_size_mb...")
        from memory_efficient_loader import MemoryOptimizedLoader
        
        # This should not raise TypeError
        loader = MemoryOptimizedLoader(str(tmp_path), chunk_size_mb=128)
        print("   ✅ MemoryOptimizedLoader accepts chunk_size_mb parameter")
        
        # Test 2: Verify TTNN can open device (firmware compilation)
        print("   Testing TTNN device opening...")
        import ttnn
        
        # Should build and open device successfully
        dev = ttnn.open_device(device_id=0)
        assert dev is not None, "Failed to open TTNN device"
        print("   ✅ TTNN device opened successfully")
        
        # Clean up device
        ttnn.close_device(dev)
        print("   ✅ TTNN device closed successfully")
        
        print("   🎉 All loader and firmware tests passed!")
        return True
        
    except ImportError as e:
        print(f"   ⚠️ Import error (expected in some environments): {e}")
        return False
    except Exception as e:
        print(f"   ❌ Loader/firmware test failed: {e}")
        return False
    finally:
        # Clean up temporary directory if we created it
        if tmp_path and tmp_path != str(tmp_path):
            import shutil
            try:
                shutil.rmtree(tmp_path)
            except:
                pass

if __name__ == "__main__":
    # Run main server tests
    main()
    
    # Run smoke test for loader and firmware
    print("\n" + "=" * 50)
    print("🧪 Running Smoke Tests")
    print("=" * 50)
    test_loader_and_firmware_precompile()

import requests
import json
import time
import os
import sys

def test_health_endpoint(base_url):
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Health endpoint working!")
            print(f"   Status: {data.get('status')}")
            print(f"   Environment: {data.get('environment')}")
            print(f"   TTNN Available: {data.get('ttnn_available')}")
            print(f"   Model Loaded: {data.get('model_loaded')}")
            if data.get('import_error'):
                print(f"   Import Error: {data.get('import_error')}")
            return True
        else:
            print(f"Health endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_root_endpoint(base_url):
    """Test the root endpoint."""
    print("🏠 Testing root endpoint...")
    try:
        response = requests.get(base_url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Root endpoint working!")
            print(f"   Content length: {len(response.text)} characters")
            return True
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_generate_endpoint(base_url):
    """Test the generate endpoint."""
    print("🤖 Testing generate endpoint...")
    try:
        payload = {
            "prompt": "What is artificial intelligence?",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Generate endpoint working!")
            print(f"   Generated text: {data.get('text', '')[:100]}...")
            print(f"   Model: {data.get('model')}")
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print(f"❌ Generate endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Generate endpoint error: {e}")
        return False

def main():
    print("🧪 Ministral-8B Server Test Suite")
    print("=" * 50)
    
    # Test both local and deployed URLs
    urls_to_test = [
        "http://localhost:8000",  # Local development
        "https://ministral-8b-priyanshuthapliyal2005-40bb59f6.koyeb.app"  # Deployed
    ]
    
    for base_url in urls_to_test:
        print(f"\n🌐 Testing server at: {base_url}")
        print("-" * 50)
        
        # Test endpoints
        results = []
        results.append(test_root_endpoint(base_url))
        results.append(test_health_endpoint(base_url))
        results.append(test_generate_endpoint(base_url))
        
        # Summary
        passed = sum(results)
        total = len(results)
        print(f"\n📊 Results for {base_url}:")
        print(f"   Passed: {passed}/{total}")
        
        if passed == total:
            print("   ✅ All tests passed!")
        else:
            print("   ⚠️ Some tests failed")

if __name__ == "__main__":
    main()

import requests
import json
import time
import os
import sys
import tempfile
import pytest
import threading
import subprocess
import socket
from contextlib import closing

def test_health_endpoint(base_url):
    """Test the health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Health endpoint working!")
            print(f"   Status: {data.get('status')}")
            print(f"   Environment: {data.get('environment')}")
            print(f"   TTNN Available: {data.get('ttnn_available')}")
            print(f"   Model Loaded: {data.get('model_loaded')}")
            if data.get('import_error'):
                print(f"   Import Error: {data.get('import_error')}")
            return True
        else:
            print(f"Health endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Health endpoint error: {e}")
        return False

def test_root_endpoint(base_url):
    """Test the root endpoint."""
    print("🏠 Testing root endpoint...")
    try:
        response = requests.get(base_url, timeout=10)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Root endpoint working!")
            print(f"   Content length: {len(response.text)} characters")
            return True
        else:
            print(f"❌ Root endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_generate_endpoint(base_url):
    """Test the generate endpoint."""
    print("🤖 Testing generate endpoint...")
    try:
        payload = {
            "prompt": "What is artificial intelligence?",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Generate endpoint working!")
            print(f"   Generated text: {data.get('text', '')[:100]}...")
            print(f"   Model: {data.get('model')}")
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print(f"❌ Generate endpoint failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Generate endpoint error: {e}")
        return False

def main():
    print("🧪 Ministral-8B Server Test Suite")
    print("=" * 50)
    
    # Test both local and deployed URLs
    urls_to_test = [
        "http://localhost:8000",  # Local development
        "https://ministral-8b-priyanshuthapliyal2005-40bb59f6.koyeb.app"  # Deployed
    ]
    
    for base_url in urls_to_test:
        print(f"\n🌐 Testing server at: {base_url}")
        print("-" * 50)
        
        # Test endpoints
        results = []
        results.append(test_root_endpoint(base_url))
        results.append(test_health_endpoint(base_url))
        results.append(test_generate_endpoint(base_url))
        
        # Summary
        passed = sum(results)
        total = len(results)
        print(f"\n📊 Results for {base_url}:")
        print(f"   Passed: {passed}/{total}")
        
        if passed == total:
            print("   ✅ All tests passed!")
        else:
            print("   ⚠️ Some tests failed")

def find_free_port():
    """Find a free port for testing."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def test_health_endpoint_immediate_response():
    """
    Test that /health endpoint responds immediately with 200 status,
    even when server is just starting up and model is not loaded.
    This catches regressions where health checks might be blocked.
    """
    print("🚀 Testing immediate health endpoint response...")
    
    # Find a free port for testing
    test_port = find_free_port()
    base_url = f"http://localhost:{test_port}"
    
    # Start server in background thread
    server_process = None
    server_thread = None
    
    try:
        # Use subprocess to start server to avoid import conflicts
        server_cmd = [
            sys.executable, 
            "server.py", 
            "--port", str(test_port),
            "--no-preload"  # Don't preload model to test immediate response
        ]
        
        print(f"   Starting server on port {test_port}...")
        server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Give server a moment to bind to port
        time.sleep(0.5)
        
        # Test immediate response
        print("   Testing immediate health response...")
        start_time = time.time()
        
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            print(f"   Response time: {response_time:.1f}ms")
            
            # Assert response code is 200
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            print("   ✅ Health endpoint returned 200 status")
            
            # Assert response time is reasonable (under 200ms)
            assert response_time < 200, f"Response too slow: {response_time:.1f}ms > 200ms"
            print(f"   ✅ Response time under 200ms: {response_time:.1f}ms")
            
            # Parse JSON response
            data = response.json()
            status = data.get('status')
            
            # Assert status is valid
            valid_statuses = ['initializing', 'ready', 'downloading', 'loading']
            assert status in valid_statuses, f"Invalid status: {status}, expected one of {valid_statuses}"
            print(f"   ✅ Valid status: {status}")
            
            # Assert no network/socket errors in response
            error_msg = data.get('error', '')
            network_errors = ['connection', 'socket', 'network', 'timeout', 'refused']
            has_network_error = any(err in error_msg.lower() for err in network_errors)
            assert not has_network_error, f"Network error detected in response: {error_msg}"
            print("   ✅ No network/socket errors detected")
            
            # Log additional useful info
            print(f"   Model loaded: {data.get('model_loaded', False)}")
            print(f"   TTNN available: {data.get('ttnn_available', False)}")
            print(f"   Environment: {data.get('environment', 'unknown')}")
            
            print("   🎉 Immediate health endpoint test passed!")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request failed: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Test setup failed: {e}")
        return False
        
    finally:
        # Clean up server process
        if server_process:
            try:
                server_process.terminate()
                server_process.wait(timeout=5)
                print("   Server process terminated")
            except subprocess.TimeoutExpired:
                server_process.kill()
                print("   Server process killed")
            except Exception as e:
                print(f"   Error cleaning up server: {e}")

def test_loader_and_firmware_precompile(tmp_path=None):
    """
    Smoke test to verify loader signature and firmware availability.
    Tests that MemoryOptimizedLoader accepts chunk_size_mb and TTNN can open device.
    """
    print("🔧 Testing loader and firmware precompile...")
    
    # Use provided tmp_path or create temporary directory
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    
    try:
        # Test 1: Verify MemoryOptimizedLoader constructor with chunk_size_mb works
        print("   Testing MemoryOptimizedLoader with chunk_size_mb...")
        from memory_efficient_loader import MemoryOptimizedLoader
        
        # This should not raise TypeError
        loader = MemoryOptimizedLoader(str(tmp_path), chunk_size_mb=128)
        print("   ✅ MemoryOptimizedLoader accepts chunk_size_mb parameter")
        
        # Test 2: Verify TTNN can open device (firmware compilation)
        print("   Testing TTNN device opening...")
        import ttnn
        
        # Should build and open device successfully
        dev = ttnn.open_device(device_id=0)
        assert dev is not None, "Failed to open TTNN device"
        print("   ✅ TTNN device opened successfully")
        
        # Clean up device
        ttnn.close_device(dev)
        print("   ✅ TTNN device closed successfully")
        
        print("   🎉 All loader and firmware tests passed!")
        return True
        
    except ImportError as e:
        print(f"   ⚠️ Import error (expected in some environments): {e}")
        return False
    except Exception as e:
        print(f"   ❌ Loader/firmware test failed: {e}")
        return False
    finally:
        # Clean up temporary directory if we created it
        if tmp_path and tmp_path != str(tmp_path):
            import shutil
            try:
                shutil.rmtree(tmp_path)
            except:
                pass

if __name__ == "__main__":
    # Run main server tests
    main()
    
    # Run smoke test for loader and firmware
    print("\n" + "=" * 50)
    print("🧪 Running Smoke Tests")
    print("=" * 50)
    test_loader_and_firmware_precompile()
    
    # Run immediate health endpoint test
    print("\n" + "=" * 50)
    print("🚀 Testing Immediate Health Response")
    print("=" * 50)
    test_health_endpoint_immediate_response()
