#!/usr/bin/env python3
"""
Comprehensive deployment status checker for Ministral-8B on Koyeb.
This script checks the current deployment status and identifies issues.
"""

import requests
import json
import time
import os
import sys
from datetime import datetime

class DeploymentChecker:
    def __init__(self, base_url="https://ministral-8b-priyanshuthapliyal2005-40bb59f6.koyeb.app"):
        self.base_url = base_url
        self.start_time = time.time()
        
    def test_endpoint(self, endpoint, method="GET", json_data=None, timeout=10):
        """Test a specific endpoint and return detailed results."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=timeout)
            elif method == "POST":
                response = requests.post(url, json=json_data, timeout=timeout)
            else:
                return {"error": f"Unsupported method: {method}"}
                
            result = {
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_time": response.elapsed.total_seconds(),
                "headers": dict(response.headers),
                "url": url
            }
            
            # Try to parse JSON response
            try:
                result["json"] = response.json()
            except:
                result["text"] = response.text[:500]  # First 500 chars
                
            return result
            
        except requests.exceptions.Timeout:
            return {"error": "Timeout", "url": url}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection failed", "url": url}
        except Exception as e:
            return {"error": str(e), "url": url}
    
    def check_all_endpoints(self):
        """Check all available endpoints."""
        print("🧪 Ministral-8B Deployment Status Check")
        print("=" * 60)
        print(f"🌐 Base URL: {self.base_url}")
        print(f"🕐 Check time: {datetime.now()}")
        print()
        
        endpoints = [
            {"path": "/", "method": "GET", "name": "Root endpoint"},
            {"path": "/health", "method": "GET", "name": "Health check"},
            {"path": "/generate", "method": "POST", "name": "Text generation", 
             "data": {"prompt": "What is AI?", "max_tokens": 50}},
        ]
        
        results = {}
        overall_status = "✅ HEALTHY"
        
        for endpoint in endpoints:
            print(f"🔍 Testing {endpoint['name']}...")
            
            result = self.test_endpoint(
                endpoint["path"], 
                endpoint["method"], 
                endpoint.get("data")
            )
            
            results[endpoint["name"]] = result
            
            if "error" in result:
                print(f"   ❌ ERROR: {result['error']}")
                overall_status = "🔥 CRITICAL"
            elif not result.get("success", False):
                print(f"   ⚠️  FAILED: Status {result.get('status_code', 'unknown')}")
                if overall_status == "✅ HEALTHY":
                    overall_status = "⚠️ DEGRADED"
            else:
                print(f"   ✅ SUCCESS: {result['response_time']:.2f}s")
                
            # Show response details for important endpoints
            if endpoint["name"] == "Health check" and "json" in result:
                health_data = result["json"]
                print(f"      Status: {health_data.get('status', 'unknown')}")
                print(f"      TTNN Available: {health_data.get('ttnn_available', 'unknown')}")
                print(f"      Model Loaded: {health_data.get('model_loaded', 'unknown')}")
                if health_data.get('import_error'):
                    print(f"      Import Error: {health_data.get('import_error')}")
                    
            elif endpoint["name"] == "Text generation" and "json" in result:
                gen_data = result["json"]
                if "text" in gen_data:
                    print(f"      Generated: {gen_data['text'][:100]}...")
                    print(f"      Model: {gen_data.get('model', 'unknown')}")
                elif "error" in gen_data:
                    print(f"      Error: {gen_data['error']}")
            
            print()
        
        # Overall assessment
        print("=" * 60)
        print(f"🎯 OVERALL STATUS: {overall_status}")
        print(f"⏱️  Total check time: {time.time() - self.start_time:.2f}s")
        
        # Provide recommendations
        self.provide_recommendations(results)
        
        return results, overall_status
    
    def provide_recommendations(self, results):
        """Provide specific recommendations based on test results."""
        print("\n💡 RECOMMENDATIONS:")
        
        health_result = results.get("Health check", {})
        if "json" in health_result:
            health_data = health_result["json"]
            
            if not health_data.get("ttnn_available", False):
                print("   🔧 TTNN is not available - this is expected in cloud environments")
                print("      This indicates the server is running in mock mode")
                
            if not health_data.get("model_loaded", False):
                print("   📦 Model is not loaded - check KOYEB_SKIP_MODEL_LOAD setting")
                print("      In production, set KOYEB_SKIP_MODEL_LOAD=false to enable model loading")
                
            if health_data.get("import_error"):
                error = health_data["import_error"]
                if "library_tweaks" in error:
                    print("   🛠️  library_tweaks error detected - add ttnn path to PYTHONPATH")
                else:
                    print(f"   🚨 Import error: {error}")
        
        gen_result = results.get("Text generation", {})
        if gen_result.get("status_code") == 500:
            print("   🔄 Generation endpoint failing - likely due to model not being loaded")
            print("      This will be resolved once model loading is enabled")
        elif "json" in gen_result and gen_result["json"].get("status") == "ok-mock":
            print("   🎭 Generation endpoint returning mock responses - this is expected in test mode")
            
        print("\n🚀 NEXT STEPS:")
        print("   1. Deploy with enhanced runtime_setup_enhanced.sh")
        print("   2. Monitor server logs for detailed error information")
        print("   3. Test with actual TT hardware for full functionality")
        print("   4. Enable model loading by setting KOYEB_SKIP_MODEL_LOAD=false")

def main():
    # Test both local and deployed URLs
    urls_to_test = [
        "https://ministral-8b-priyanshuthapliyal2005-40bb59f6.koyeb.app",
        "http://localhost:8000"  # If testing locally
    ]
    
    for url in urls_to_test:
        print(f"\n🌐 Testing deployment at: {url}")
        checker = DeploymentChecker(url)
        try:
            results, status = checker.check_all_endpoints()
            if "HEALTHY" in status:
                print("🎉 Deployment is working well!")
                break
        except Exception as e:
            print(f"❌ Failed to check {url}: {e}")
            continue

if __name__ == "__main__":
    main()
