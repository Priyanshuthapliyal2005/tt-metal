#!/usr/bin/env python3
"""
Performance optimizer for memory-efficient Ministral-8B deployment.
Provides real-time monitoring, multi-device optimization, and performance analytics.
"""

import time
import json
import logging
import psutil
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Enhanced performance optimization for TTNN multi-device deployments."""
    
    def __init__(self):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.metrics = {
            'download_speeds': [],
            'loading_times': [],
            'memory_peaks': [],
            'device_utilization': {},
            'inference_times': [],
            'throughput_tokens_per_sec': []
        }
        
        # Detect TTNN devices after logger is initialized
        self.device_config = self._detect_ttnn_devices()
    
    def _detect_ttnn_devices(self) -> Dict[str, Any]:
        """Detect and configure TTNN devices for optimal utilization."""
        try:
            import ttnn
            num_devices = ttnn.get_num_devices()
            device_config = {
                'num_devices': num_devices,
                'devices': [],
                'memory_per_device': {},
                'optimal_sharding': num_devices > 1
            }
            
            for i in range(num_devices):
                try:
                    device = ttnn.open_device(device_id=i)
                    device_config['devices'].append(device)
                    device_config['memory_per_device'][i] = self._get_device_memory(device)
                    self.logger.info(f"TTNN Device {i} initialized successfully")
                except Exception as e:
                    self.logger.warning(f"Could not initialize TTNN device {i}: {e}")
            
            self.logger.info(f"🔥 TTNN Multi-device setup: {num_devices} devices detected")
            return device_config
            
        except ImportError:
            self.logger.warning("TTNN not available, using CPU fallback")
            return {'num_devices': 0, 'devices': [], 'memory_per_device': {}, 'optimal_sharding': False}
    
    def _get_device_memory(self, device) -> Dict[str, float]:
        """Get memory information for a TTNN device."""
        try:
            # Placeholder for actual TTNN memory query
            # In real implementation, this would query device memory status
            return {'total_gb': 8.0, 'available_gb': 7.5}
        except:
            return {'total_gb': 8.0, 'available_gb': 7.5}
    
    @contextmanager
    def performance_monitor(self, operation_name: str):
        """Enhanced performance monitoring with real-time tracking."""
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)
        start_cpu = psutil.cpu_percent()
        
        # Monitor device utilization if TTNN available
        device_start_metrics = {}
        for i, device in enumerate(self.device_config['devices']):
            device_start_metrics[i] = self._get_device_memory(device)
        
        self.logger.info(f"🚀 Starting {operation_name}...")
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**3)
            end_cpu = psutil.cpu_percent()
            peak_memory = max(start_memory, end_memory)
            
            # Update metrics
            duration = end_time - start_time
            self.metrics['loading_times'].append(duration)
            self.metrics['memory_peaks'].append(peak_memory)
            
            # Device utilization tracking
            for i, device in enumerate(self.device_config['devices']):
                end_metrics = self._get_device_memory(device)
                if i not in self.metrics['device_utilization']:
                    self.metrics['device_utilization'][i] = []
                
                start_available = device_start_metrics.get(i, {}).get('available_gb', 0)
                end_available = end_metrics.get('available_gb', 0)
                memory_used = start_available - end_available
                self.metrics['device_utilization'][i].append(max(0, memory_used))
            
            self.logger.info(f"✅ {operation_name} completed in {duration:.2f}s")
            self.logger.info(f"📊 Memory: {peak_memory:.2f}GB peak, CPU: {end_cpu:.1f}%")
    
    def optimize_model_sharding(self, model_size_gb: float) -> Dict[str, Any]:
        """Determine optimal model sharding strategy based on available devices."""
        if not self.device_config['optimal_sharding'] or self.device_config['num_devices'] < 2:
            return {'strategy': 'single_device', 'shards': 1}
        
        num_devices = self.device_config['num_devices']
        memory_per_device = min(
            [info.get('available_gb', 8.0) for info in self.device_config['memory_per_device'].values()]
        )
        
        self.logger.info(f"🔧 Model size: {model_size_gb:.1f}GB, Available memory per device: {memory_per_device:.1f}GB")
        
        # Calculate optimal sharding strategy
        if model_size_gb <= memory_per_device * 0.8:  # 80% utilization max
            strategy = {'strategy': 'single_device', 'shards': 1, 'device_id': 0}
            self.logger.info("📱 Using single-device strategy")
        elif model_size_gb <= memory_per_device * num_devices * 0.8:
            shards = min(num_devices, int(model_size_gb / (memory_per_device * 0.8)) + 1)
            strategy = {'strategy': 'multi_device', 'shards': shards, 'devices': list(range(shards))}
            self.logger.info(f"🔀 Using multi-device strategy with {shards} shards")
        else:
            strategy = {'strategy': 'streaming', 'shards': num_devices, 'devices': list(range(num_devices))}
            self.logger.info("💿 Using streaming strategy for large model")
        
        return strategy
    
    def track_download_performance(self, url: str, total_size: int, downloaded_size: int, 
                                 elapsed_time: float):
        """Track and log download performance metrics."""
        if elapsed_time > 0:
            speed_mbps = (downloaded_size / (1024 * 1024)) / elapsed_time
            self.metrics['download_speeds'].append(speed_mbps)
            
            progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
            eta = ((total_size - downloaded_size) / (downloaded_size / elapsed_time)) if downloaded_size > 0 else 0
            
            self.logger.info(f"📥 Download: {progress:.1f}% complete, "
                           f"Speed: {speed_mbps:.1f} MB/s, ETA: {eta:.0f}s")
    
    def track_inference_performance(self, input_tokens: int, output_tokens: int, 
                                  inference_time: float):
        """Track inference performance metrics."""
        total_tokens = input_tokens + output_tokens
        tokens_per_second = total_tokens / inference_time if inference_time > 0 else 0
        
        self.metrics['inference_times'].append(inference_time)
        self.metrics['throughput_tokens_per_sec'].append(tokens_per_second)
        
        self.logger.info(f"🤖 Inference: {input_tokens} → {output_tokens} tokens "
                        f"in {inference_time:.2f}s ({tokens_per_second:.1f} tok/s)")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report with insights."""
        download_speeds = self.metrics['download_speeds']
        loading_times = self.metrics['loading_times']
        memory_peaks = self.metrics['memory_peaks']
        inference_times = self.metrics['inference_times']
        throughput = self.metrics['throughput_tokens_per_sec']
        
        report = {
            'deployment_summary': {
                'total_operations': len(loading_times),
                'avg_loading_time_sec': sum(loading_times) / max(len(loading_times), 1),
                'peak_memory_gb': max(memory_peaks) if memory_peaks else 0,
                'avg_download_speed_mbps': sum(download_speeds) / max(len(download_speeds), 1),
                'avg_inference_time_sec': sum(inference_times) / max(len(inference_times), 1),
                'avg_throughput_tokens_per_sec': sum(throughput) / max(len(throughput), 1)
            },
            'hardware_utilization': {},
            'optimization_recommendations': [],
            'device_efficiency': {}
        }
        
        # Analyze device utilization
        for device_id, utilization_history in self.metrics['device_utilization'].items():
            if utilization_history:
                avg_usage = sum(utilization_history) / len(utilization_history)
                peak_usage = max(utilization_history)
                efficiency = avg_usage / 8.0  # Assuming 8GB per device
                
                report['hardware_utilization'][f'device_{device_id}'] = {
                    'avg_memory_used_gb': avg_usage,
                    'peak_memory_used_gb': peak_usage,
                    'efficiency_percent': efficiency * 100
                }
                
                report['device_efficiency'][device_id] = efficiency
        
        # Generate optimization recommendations
        recommendations = []
        
        if report['deployment_summary']['peak_memory_gb'] > 14:
            recommendations.append("⚠️  High memory usage detected - consider more aggressive streaming")
        
        if len(self.device_config['devices']) > 1:
            avg_efficiency = sum(report['device_efficiency'].values()) / max(len(report['device_efficiency']), 1)
            if avg_efficiency < 0.5:
                recommendations.append("🔧 Low device utilization - optimize multi-device sharding")
            elif len(report['device_efficiency']) == 1:
                recommendations.append("📱 Multiple devices available but only one utilized - enable multi-device sharding")
        
        if report['deployment_summary']['avg_throughput_tokens_per_sec'] < 50:
            recommendations.append("🚀 Low inference throughput - optimize model loading and caching")
        
        if report['deployment_summary']['avg_download_speed_mbps'] < 50:
            recommendations.append("📡 Slow download speeds - check network connectivity and caching")
        
        if not recommendations:
            recommendations.append("✅ System performing optimally!")
        
        report['optimization_recommendations'] = recommendations
        
        return report
    
    def print_performance_summary(self):
        """Print a formatted performance summary to console."""
        report = self.get_performance_report()
        
        print("\n" + "="*60)
        print("🔥 MINISTRAL-8B PERFORMANCE REPORT")
        print("="*60)
        
        summary = report['deployment_summary']
        print(f"📊 Operations Completed: {summary['total_operations']}")
        print(f"⏱️  Average Loading Time: {summary['avg_loading_time_sec']:.2f}s")
        print(f"💾 Peak Memory Usage: {summary['peak_memory_gb']:.2f}GB")
        print(f"📥 Average Download Speed: {summary['avg_download_speed_mbps']:.1f} MB/s")
        print(f"🤖 Average Inference Time: {summary['avg_inference_time_sec']:.2f}s")
        print(f"🚀 Average Throughput: {summary['avg_throughput_tokens_per_sec']:.1f} tokens/sec")
        
        print(f"\n🔧 TTNN Hardware: {self.device_config['num_devices']} devices")
        for device_id, metrics in report['hardware_utilization'].items():
            print(f"   • {device_id}: {metrics['efficiency_percent']:.1f}% utilized "
                  f"({metrics['avg_memory_used_gb']:.1f}GB avg)")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in report['optimization_recommendations']:
            print(f"   {rec}")
        
        print("="*60 + "\n")
    
    def save_metrics(self, filepath: str):
        """Save performance metrics to JSON file."""
        metrics_with_report = {
            'raw_metrics': self.metrics,
            'performance_report': self.get_performance_report(),
            'device_config': {
                'num_devices': self.device_config['num_devices'],
                'optimal_sharding': self.device_config['optimal_sharding']
            },
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_with_report, f, indent=2)
        
        self.logger.info(f"📈 Performance metrics saved to {filepath}")

# Global instance for easy access
performance_optimizer = PerformanceOptimizer()
