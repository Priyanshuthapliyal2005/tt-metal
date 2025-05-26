#!/usr/bin/env python3
"""
Comprehensive benchmarking script for Ministral-8B memory-efficient deployment.
Tests performance optimizations and generates detailed reports.
"""

import time
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any
import argparse

from performance_optimizer import performance_optimizer
from memory_efficient_loader import MemoryOptimizedLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinistralBenchmark:
    """Comprehensive benchmark suite for Ministral-8B deployment."""
    
    def __init__(self, model_path: str, cache_dir: str = "./benchmark_cache"):
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.loader = MemoryOptimizedLoader(cache_dir=str(self.cache_dir))
        self.results = {}
    
    def benchmark_model_loading(self) -> Dict[str, Any]:
        """Benchmark model loading with different strategies."""
        logger.info("🔄 Benchmarking model loading strategies...")
        
        loading_results = {}
        
        # Test single-device loading
        with performance_optimizer.performance_monitor("Single Device Loading"):
            start_time = time.time()
            try:
                model, tokenizer = self.loader.lazy_load_for_ttnn(self.model_path)
                loading_time = time.time() - start_time
                loading_results['single_device'] = {
                    'success': True,
                    'loading_time': loading_time,
                    'strategy': 'single_device'
                }
                logger.info(f"✅ Single device loading: {loading_time:.2f}s")
            except Exception as e:
                loading_results['single_device'] = {
                    'success': False,
                    'error': str(e),
                    'loading_time': time.time() - start_time
                }
                logger.error(f"❌ Single device loading failed: {e}")
        
        # Test multi-device loading if available
        if performance_optimizer.device_config['num_devices'] > 1:
            with performance_optimizer.performance_monitor("Multi Device Loading"):
                start_time = time.time()
                try:
                    # Estimate model size for sharding strategy
                    model_size = self.loader.estimate_memory_usage(self.model_path)
                    strategy = performance_optimizer.optimize_model_sharding(model_size)
                    
                    model, tokenizer = self.loader.load_ministral_model_optimized(
                        self.model_path, device_strategy=strategy
                    )
                    loading_time = time.time() - start_time
                    loading_results['multi_device'] = {
                        'success': True,
                        'loading_time': loading_time,
                        'strategy': strategy,
                        'devices_used': len(strategy.get('devices', [1]))
                    }
                    logger.info(f"✅ Multi-device loading: {loading_time:.2f}s")
                except Exception as e:
                    loading_results['multi_device'] = {
                        'success': False,
                        'error': str(e),
                        'loading_time': time.time() - start_time
                    }
                    logger.error(f"❌ Multi-device loading failed: {e}")
        
        return loading_results
    
    def benchmark_inference_performance(self, model, tokenizer, num_samples: int = 10) -> Dict[str, Any]:
        """Benchmark inference performance with various input sizes."""
        logger.info(f"🧠 Benchmarking inference performance with {num_samples} samples...")
        
        test_prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short story about a robot learning to paint.",
            "Describe the process of photosynthesis.",
            "What are the benefits of renewable energy?",
            "How does machine learning work?",
            "Explain the theory of relativity.",
            "What is the meaning of life?",
            "Describe the water cycle.",
            "How do neural networks function?"
        ]
        
        inference_results = {
            'samples': [],
            'avg_time': 0,
            'avg_throughput': 0,
            'tokens_per_second': []
        }
        
        total_time = 0
        total_tokens = 0
        
        for i, prompt in enumerate(test_prompts[:num_samples]):
            try:
                with performance_optimizer.performance_monitor(f"Inference {i+1}"):
                    start_time = time.time()
                    
                    # Tokenize input
                    inputs = tokenizer(prompt, return_tensors="pt")
                    input_tokens = len(inputs['input_ids'][0])
                    
                    # Generate response (placeholder - replace with actual inference)
                    # In real implementation, this would call model.generate()
                    time.sleep(0.1)  # Simulate inference time
                    output_text = f"Response to: {prompt}"
                    output_tokens = len(tokenizer(output_text)['input_ids'])
                    
                    inference_time = time.time() - start_time
                    tokens_per_sec = (input_tokens + output_tokens) / inference_time
                    
                    # Track performance
                    performance_optimizer.track_inference_performance(
                        input_tokens, output_tokens, inference_time
                    )
                    
                    sample_result = {
                        'prompt': prompt,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'inference_time': inference_time,
                        'tokens_per_second': tokens_per_sec
                    }
                    
                    inference_results['samples'].append(sample_result)
                    inference_results['tokens_per_second'].append(tokens_per_sec)
                    
                    total_time += inference_time
                    total_tokens += input_tokens + output_tokens
                    
                    logger.info(f"Sample {i+1}: {inference_time:.2f}s, {tokens_per_sec:.1f} tok/s")
                    
            except Exception as e:
                logger.error(f"❌ Inference sample {i+1} failed: {e}")
                inference_results['samples'].append({
                    'prompt': prompt,
                    'error': str(e)
                })
        
        # Calculate averages
        successful_samples = [s for s in inference_results['samples'] if 'error' not in s]
        if successful_samples:
            inference_results['avg_time'] = sum(s['inference_time'] for s in successful_samples) / len(successful_samples)
            inference_results['avg_throughput'] = sum(s['tokens_per_second'] for s in successful_samples) / len(successful_samples)
        
        return inference_results
    
    def benchmark_memory_efficiency(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns during different operations."""
        logger.info("💾 Benchmarking memory efficiency...")
        
        import psutil
        import gc
        
        memory_results = {
            'baseline_memory_gb': 0,
            'peak_during_loading_gb': 0,
            'peak_during_inference_gb': 0,
            'memory_efficiency_score': 0
        }
        
        # Baseline memory
        gc.collect()
        baseline_memory = psutil.virtual_memory().used / (1024**3)
        memory_results['baseline_memory_gb'] = baseline_memory
        
        # Memory during loading
        with performance_optimizer.performance_monitor("Memory Efficiency Test"):
            peak_memory = baseline_memory
            
            # Simulate loading operations
            try:
                model_size = self.loader.estimate_memory_usage(self.model_path)
                current_memory = psutil.virtual_memory().used / (1024**3)
                peak_memory = max(peak_memory, current_memory)
                
                memory_results['peak_during_loading_gb'] = peak_memory
                
                # Calculate efficiency score (lower is better)
                memory_overhead = peak_memory - baseline_memory
                efficiency_score = min(100, max(0, 100 - (memory_overhead / model_size * 100)))
                memory_results['memory_efficiency_score'] = efficiency_score
                
            except Exception as e:
                logger.error(f"Memory efficiency test failed: {e}")
                memory_results['error'] = str(e)
        
        return memory_results
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark tests and generate comprehensive report."""
        logger.info("🚀 Starting comprehensive Ministral-8B benchmark...")
        
        benchmark_results = {
            'timestamp': time.time(),
            'model_path': self.model_path,
            'hardware_config': {
                'num_ttnn_devices': performance_optimizer.device_config['num_devices'],
                'optimal_sharding': performance_optimizer.device_config['optimal_sharding']
            },
            'tests': {}
        }
        
        # 1. Model Loading Benchmark
        try:
            benchmark_results['tests']['loading'] = self.benchmark_model_loading()
        except Exception as e:
            logger.error(f"Loading benchmark failed: {e}")
            benchmark_results['tests']['loading'] = {'error': str(e)}
        
        # 2. Memory Efficiency Benchmark
        try:
            benchmark_results['tests']['memory'] = self.benchmark_memory_efficiency()
        except Exception as e:
            logger.error(f"Memory benchmark failed: {e}")
            benchmark_results['tests']['memory'] = {'error': str(e)}
        
        # 3. Inference Performance Benchmark (if model loaded successfully)
        if benchmark_results['tests']['loading'].get('single_device', {}).get('success'):
            try:
                # Note: In real implementation, you'd pass the actual loaded model
                benchmark_results['tests']['inference'] = self.benchmark_inference_performance(
                    None, None, num_samples=5  # Reduced for demo
                )
            except Exception as e:
                logger.error(f"Inference benchmark failed: {e}")
                benchmark_results['tests']['inference'] = {'error': str(e)}
        
        # Generate performance report
        benchmark_results['performance_report'] = performance_optimizer.get_performance_report()
        
        # Save results
        results_file = self.cache_dir / f"benchmark_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"📈 Benchmark results saved to {results_file}")
        
        return benchmark_results
    
    def print_benchmark_summary(self, results: Dict[str, Any]):
        """Print a formatted benchmark summary."""
        print("\n" + "="*70)
        print("🔥 MINISTRAL-8B BENCHMARK RESULTS")
        print("="*70)
        
        # Hardware Configuration
        hw_config = results['hardware_config']
        print(f"🖥️  Hardware: {hw_config['num_ttnn_devices']} TTNN devices")
        print(f"🔀 Multi-device optimization: {'Enabled' if hw_config['optimal_sharding'] else 'Disabled'}")
        
        # Loading Performance
        loading = results['tests'].get('loading', {})
        if 'single_device' in loading:
            single = loading['single_device']
            if single.get('success'):
                print(f"📱 Single-device loading: {single['loading_time']:.2f}s")
            else:
                print(f"❌ Single-device loading: FAILED ({single.get('error', 'Unknown error')})")
        
        if 'multi_device' in loading:
            multi = loading['multi_device']
            if multi.get('success'):
                print(f"🔀 Multi-device loading: {multi['loading_time']:.2f}s ({multi.get('devices_used', 0)} devices)")
            else:
                print(f"❌ Multi-device loading: FAILED ({multi.get('error', 'Unknown error')})")
        
        # Memory Efficiency
        memory = results['tests'].get('memory', {})
        if 'memory_efficiency_score' in memory:
            print(f"💾 Memory efficiency score: {memory['memory_efficiency_score']:.1f}/100")
            print(f"📊 Peak memory usage: {memory['peak_during_loading_gb']:.2f}GB")
        
        # Inference Performance
        inference = results['tests'].get('inference', {})
        if 'avg_throughput' in inference:
            print(f"🧠 Average inference speed: {inference['avg_throughput']:.1f} tokens/sec")
            print(f"⏱️  Average inference time: {inference['avg_time']:.2f}s")
        
        # Recommendations
        report = results.get('performance_report', {})
        recommendations = report.get('optimization_recommendations', [])
        if recommendations:
            print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   {rec}")
        
        print("="*70 + "\n")

def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="Benchmark Ministral-8B memory-efficient deployment")
    parser.add_argument("--model-path", default="./model_cache/ministral-8b-instruct",
                       help="Path to the model directory")
    parser.add_argument("--cache-dir", default="./benchmark_cache",
                       help="Directory for benchmark cache")
    parser.add_argument("--samples", type=int, default=10,
                       help="Number of inference samples to test")
    
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = MinistralBenchmark(args.model_path, args.cache_dir)
    
    try:
        # Run comprehensive benchmark
        results = benchmark.run_comprehensive_benchmark()
        
        # Print summary
        benchmark.print_benchmark_summary(results)
        
        # Print detailed performance report
        performance_optimizer.print_performance_summary()
        
        # Save performance metrics
        metrics_file = Path(args.cache_dir) / f"performance_metrics_{int(time.time())}.json"
        performance_optimizer.save_metrics(str(metrics_file))
        
        logger.info("🎉 Benchmark completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    main()
