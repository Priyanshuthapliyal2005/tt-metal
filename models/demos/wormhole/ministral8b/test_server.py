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
