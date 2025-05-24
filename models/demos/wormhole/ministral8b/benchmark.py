#!/usr/bin/env python3
"""
Performance benchmarking script for Ministral-8B on Tenstorrent N300 hardware.
Tests various configurations to validate performance targets.
"""

import argparse
import time
import json
import statistics
from typing import List, Dict, Any
import os
import sys

# Add the parent directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def run_benchmark_test(
    batch_size: int, 
    max_seq_len: int, 
    device_id: int,
    num_runs: int = 3,
    warmup_runs: int = 1
) -> Dict[str, Any]:
    """Run a benchmark test with specified parameters."""
    
    print(f"🔥 Running benchmark: batch_size={batch_size}, max_seq_len={max_seq_len}, device_id={device_id}")
    
    # Import here to avoid issues if TT-Metal isn't available
    from demo.demo_with_prefill import main as run_demo
    
    times = []
    
    # Warmup runs
    for i in range(warmup_runs):
        print(f"  Warmup run {i+1}/{warmup_runs}")
        try:
            start_time = time.time()
            # Run the demo with a simple question
            os.system(f"cd /workspaces/tt-metal/models/demos/wormhole/ministral8b && "
                     f"python demo/demo_with_prefill.py "
                     f"--batch_size {batch_size} --max_seq_len {max_seq_len} "
                     f"--device_id {device_id} --instruct "
                     f"--question 'Hello' > /dev/null 2>&1")
            warmup_time = time.time() - start_time
            print(f"    Warmup time: {warmup_time:.2f}s")
        except Exception as e:
            print(f"    Warmup failed: {e}")
    
    # Actual benchmark runs
    for i in range(num_runs):
        print(f"  Benchmark run {i+1}/{num_runs}")
        try:
            start_time = time.time()
            
            # Run the demo
            result = os.system(f"cd /workspaces/tt-metal/models/demos/wormhole/ministral8b && "
                              f"python demo/demo_with_prefill.py "
                              f"--batch_size {batch_size} --max_seq_len {max_seq_len} "
                              f"--device_id {device_id} --instruct "
                              f"--question 'What is artificial intelligence?' > /dev/null 2>&1")
            
            run_time = time.time() - start_time
            
            if result == 0:  # Success
                times.append(run_time)
                print(f"    Run time: {run_time:.2f}s")
            else:
                print(f"    Run failed with code {result}")
                
        except Exception as e:
            print(f"    Run failed: {e}")
    
    if not times:
        return {
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "device_id": device_id,
            "status": "failed",
            "error": "All runs failed"
        }
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    tokens_per_second = (batch_size * max_seq_len) / avg_time
    
    result = {
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "device_id": device_id,
        "status": "success",
        "num_successful_runs": len(times),
        "times": times,
        "avg_time": avg_time,
        "min_time": min(times),
        "max_time": max(times),
        "std_time": statistics.stdev(times) if len(times) > 1 else 0,
        "tokens_per_second": tokens_per_second,
        "throughput_tokens_per_sec": tokens_per_second
    }
    
    print(f"  ✅ Average time: {avg_time:.2f}s, Tokens/sec: {tokens_per_second:.1f}")
    return result

def check_hardware():
    """Check if N300 hardware is available and detected."""
    try:
        import ttnn
        devices = ttnn.get_device_ids()
        print(f"✅ Found {len(devices)} Tenstorrent device(s): {devices}")
        return len(devices) > 0
    except Exception as e:
        print(f"❌ Error accessing Tenstorrent devices: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Benchmark Ministral-8B on N300 hardware")
    parser.add_argument("--device_id", type=int, default=0, help="Device ID to use")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of benchmark runs per configuration")
    parser.add_argument("--warmup_runs", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--output_file", type=str, default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with limited configurations")
    
    args = parser.parse_args()
    
    print("🚀 Starting Ministral-8B Performance Benchmark on N300 Hardware")
    print("=" * 60)
    
    # Check hardware
    if not check_hardware():
        print("❌ No Tenstorrent hardware detected. Exiting.")
        return 1
    
    # Define test configurations
    if args.quick:
        test_configs = [
            {"batch_size": 1, "max_seq_len": 128},
            {"batch_size": 1, "max_seq_len": 512},
            {"batch_size": 4, "max_seq_len": 256},
        ]
    else:
        test_configs = [
            {"batch_size": 1, "max_seq_len": 128},
            {"batch_size": 1, "max_seq_len": 256},
            {"batch_size": 1, "max_seq_len": 512},
            {"batch_size": 1, "max_seq_len": 1024},
            {"batch_size": 2, "max_seq_len": 256},
            {"batch_size": 4, "max_seq_len": 256},
            {"batch_size": 8, "max_seq_len": 128},
            {"batch_size": 16, "max_seq_len": 128},
            {"batch_size": 32, "max_seq_len": 64},
        ]
    
    results = []
    
    print(f"\n📊 Running {len(test_configs)} benchmark configurations...")
    
    for i, config in enumerate(test_configs):
        print(f"\n[{i+1}/{len(test_configs)}] Testing configuration: {config}")
        
        result = run_benchmark_test(
            batch_size=config["batch_size"],
            max_seq_len=config["max_seq_len"],
            device_id=args.device_id,
            num_runs=args.num_runs,
            warmup_runs=args.warmup_runs
        )
        
        results.append(result)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 60)
    
    successful_results = [r for r in results if r["status"] == "success"]
    failed_results = [r for r in results if r["status"] == "failed"]
    
    print(f"✅ Successful runs: {len(successful_results)}")
    print(f"❌ Failed runs: {len(failed_results)}")
    
    if successful_results:
        print("\n🏆 TOP PERFORMING CONFIGURATIONS:")
        
        # Sort by tokens per second
        top_throughput = sorted(successful_results, key=lambda x: x["throughput_tokens_per_sec"], reverse=True)[:3]
        
        for i, result in enumerate(top_throughput):
            print(f"{i+1}. Batch={result['batch_size']}, SeqLen={result['max_seq_len']}: "
                  f"{result['throughput_tokens_per_sec']:.1f} tokens/sec "
                  f"({result['avg_time']:.2f}s avg)")
        
        # Performance targets validation
        print("\n🎯 PERFORMANCE TARGETS VALIDATION:")
        
        # Target: >100 tokens/sec for batch_size=1, seq_len=512
        target_config = next((r for r in successful_results 
                            if r["batch_size"] == 1 and r["max_seq_len"] == 512), None)
        if target_config:
            target_met = target_config["throughput_tokens_per_sec"] > 100
            status = "✅ PASSED" if target_met else "❌ FAILED"
            print(f"  Single sequence (1x512): {target_config['throughput_tokens_per_sec']:.1f} tokens/sec {status}")
        
        # Target: >500 tokens/sec for batch_size=8, seq_len=128
        batch_config = next((r for r in successful_results 
                           if r["batch_size"] == 8 and r["max_seq_len"] == 128), None)
        if batch_config:
            target_met = batch_config["throughput_tokens_per_sec"] > 500
            status = "✅ PASSED" if target_met else "❌ FAILED"
            print(f"  Batch processing (8x128): {batch_config['throughput_tokens_per_sec']:.1f} tokens/sec {status}")
        
        # Overall max throughput
        max_throughput = max(r["throughput_tokens_per_sec"] for r in successful_results)
        print(f"  Peak throughput: {max_throughput:.1f} tokens/sec")
    
    print(f"\n💾 Full results saved to: {args.output_file}")
    
    return 0 if len(failed_results) == 0 else 1

if __name__ == "__main__":
    exit(main())
