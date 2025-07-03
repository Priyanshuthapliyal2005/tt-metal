#!/usr/bin/env python3
"""
Performance optimizer for memory-efficient Ministral-8B deployment.
Provides real-time monitoring, multi-device optimization, and performance analytics.
"""

import time
import json
import logging
import psutil
import os
import yaml
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
        """Detect and configure TTNN devices for optimal utilization with enhanced YAML error handling."""
        # Check environment variables for SOC descriptor paths
        soc_descriptor_path = os.environ.get('TT_METAL_SOC_DESCRIPTOR_PATH')
        arch_yaml_path = os.environ.get('TT_METAL_ARCH_YAML_PATH')
        
        if soc_descriptor_path:
            self.logger.info(f"🔧 Using SOC descriptor from environment: {soc_descriptor_path}")
        if arch_yaml_path:
            self.logger.info(f"🔧 Using arch YAML from environment: {arch_yaml_path}")
        
        # Detect hardware type from environment or system
        hardware_type = os.environ.get('TT_METAL_ARCH_NAME', 'wormhole_b0')
        self.logger.info(f"🔍 Detected hardware type: {hardware_type}")
        
        # Define fallback SOC descriptors to try
        fallback_descriptors = [
            '/workspaces/tt-metal/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml',
            '/workspaces/tt-metal/tt_metal/soc_descriptors/wormhole_b0_versim.yaml',
            '/workspaces/tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml'
        ]
        
        # Add environment-specified descriptor to the front of the list
        if soc_descriptor_path and soc_descriptor_path not in fallback_descriptors:
            fallback_descriptors.insert(0, soc_descriptor_path)
        
        try:
            import ttnn
            self.logger.info("✅ TTNN module imported successfully")
            
            # Try to get number of devices with enhanced error handling
            num_devices = None
            yaml_error_details = None
            successful_descriptor = None
            
            for descriptor_path in fallback_descriptors:
                try:
                    self.logger.info(f"🔍 Attempting to use SOC descriptor: {descriptor_path}")
                    
                    # Validate YAML file exists and is readable
                    if not os.path.exists(descriptor_path):
                        self.logger.warning(f"❌ SOC descriptor not found: {descriptor_path}")
                        continue
                    
                    # Try to parse the YAML file to check for syntax errors
                    try:
                        with open(descriptor_path, 'r') as f:
                            yaml_content = yaml.safe_load(f)
                        self.logger.info(f"✅ YAML file parsed successfully: {descriptor_path}")
                        
                        # Validate critical YAML structure
                        if not self._validate_soc_descriptor(yaml_content, descriptor_path):
                            continue
                            
                    except yaml.YAMLError as yaml_err:
                        line_num = getattr(yaml_err, 'problem_mark', None)
                        if line_num:
                            error_msg = f"YAML parsing error at line {line_num.line + 1}, column {line_num.column + 1}: {yaml_err}"
                        else:
                            error_msg = f"YAML parsing error: {yaml_err}"
                        
                        self.logger.error(f"❌ {error_msg} in {descriptor_path}")
                        yaml_error_details = error_msg
                        continue
                    except Exception as file_err:
                        self.logger.error(f"❌ Failed to read YAML file {descriptor_path}: {file_err}")
                        continue
                    
                    # Set environment variable to use this descriptor
                    os.environ['TT_METAL_SOC_DESCRIPTOR_PATH'] = descriptor_path
                    
                    # Try to get device count with this descriptor
                    num_devices = ttnn.get_num_devices()
                    successful_descriptor = descriptor_path
                    self.logger.info(f"✅ Successfully detected {num_devices} devices using {descriptor_path}")
                    break
                    
                except Exception as device_err:
                    error_str = str(device_err)
                    
                    # Check for specific YAML-related errors
                    if any(keyword in error_str.lower() for keyword in ['yaml', 'bad conversion', 'parse', 'syntax']):
                        self.logger.error(f"❌ YAML parsing error with {descriptor_path}: {device_err}")
                        yaml_error_details = f"Device detection failed due to YAML error: {device_err}"
                    else:
                        self.logger.warning(f"⚠️ Device detection failed with {descriptor_path}: {device_err}")
                    
                    continue
            
            # If all descriptors failed, provide detailed error information
            if num_devices is None:
                error_msg = "Failed to detect TTNN devices with all available SOC descriptors."
                if yaml_error_details:
                    error_msg += f" Last YAML error: {yaml_error_details}"
                
                self.logger.error(f"❌ {error_msg}")
                self.logger.info("🔧 Troubleshooting tips:")
                self.logger.info("   1. Check if SOC descriptor YAML files have correct format")
                self.logger.info("   2. Verify eth_endpoint and worker_endpoint use consistent types")
                self.logger.info("   3. Ensure TT hardware is properly connected and drivers loaded")
                self.logger.info("   4. Try setting TT_METAL_SOC_DESCRIPTOR_PATH environment variable")
                
                return {
                    'num_devices': 0, 
                    'devices': [], 
                    'memory_per_device': {}, 
                    'optimal_sharding': False,
                    'error': error_msg,
                    'yaml_error': yaml_error_details
                }
            
            # Log successful configuration
            self.logger.info(f"🔧 Using SOC descriptor: {successful_descriptor}")
            self.logger.info(f"🔧 Hardware type: {hardware_type}")
            
            # Initialize device configuration
            device_config = {
                'num_devices': num_devices,
                'devices': [],
                'memory_per_device': {},
                'optimal_sharding': num_devices > 1,
                'soc_descriptor': successful_descriptor,
                'hardware_type': hardware_type
            }
            
            # Validate device configuration before proceeding
            if not self._validate_device_config(device_config):
                self.logger.error("❌ Device configuration validation failed")
                return {
                    'num_devices': 0, 
                    'devices': [], 
                    'memory_per_device': {}, 
                    'optimal_sharding': False,
                    'error': 'Device configuration validation failed'
                }
            
            # Initialize individual devices
            for i in range(num_devices):
                try:
                    device = ttnn.open_device(device_id=i)
                    device_config['devices'].append(device)
                    device_config['memory_per_device'][i] = self._get_device_memory(device)
                    self.logger.info(f"✅ TTNN Device {i} initialized successfully")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not initialize TTNN device {i}: {e}")
                    # Continue with other devices even if one fails
            
            self.logger.info(f"🔥 TTNN Multi-device setup: {num_devices} devices detected")
            return device_config
            
        except ImportError as import_err:
            self.logger.warning(f"❌ TTNN not available: {import_err}")
            self.logger.info("🔧 Using CPU fallback mode")
            return {
                'num_devices': 0, 
                'devices': [], 
                'memory_per_device': {}, 
                'optimal_sharding': False,
                'error': f'TTNN import failed: {import_err}'
            }
        except Exception as general_err:
            self.logger.error(f"❌ Unexpected error in device detection: {general_err}")
            return {
                'num_devices': 0, 
                'devices': [], 
                'memory_per_device': {}, 
                'optimal_sharding': False,
                'error': f'Unexpected error: {general_err}'
            }
    
    def _validate_soc_descriptor(self, yaml_content: Dict[str, Any], descriptor_path: str) -> bool:
        """Validate SOC descriptor YAML structure for common issues."""
        try:
            # Check for required top-level keys
            required_keys = ['grid', 'arc', 'dram', 'eth']
            missing_keys = [key for key in required_keys if key not in yaml_content]
            
            if missing_keys:
                self.logger.warning(f"⚠️ Missing required keys in {descriptor_path}: {missing_keys}")
                return False
            
            # Check dram_views for type consistency issues
            if 'dram_views' in yaml_content:
                dram_views = yaml_content['dram_views']
                for i, view in enumerate(dram_views):
                    if 'eth_endpoint' in view:
                        eth_endpoint = view['eth_endpoint']
                        # Check for type consistency - should be either all integers or all lists
                        if isinstance(eth_endpoint, list) and len(eth_endpoint) == 2:
                            # List format like [0, 0] - check if this is consistent
                            if not all(isinstance(x, int) for x in eth_endpoint):
                                self.logger.warning(f"⚠️ Invalid eth_endpoint format in dram_view {i}: {eth_endpoint}")
                                return False
                        elif isinstance(eth_endpoint, int):
                            # Integer format like 0 - this is also valid
                            pass
                        else:
                            self.logger.warning(f"⚠️ Invalid eth_endpoint type in dram_view {i}: {type(eth_endpoint)}")
                            return False
                    
                    if 'worker_endpoint' in view:
                        worker_endpoint = view['worker_endpoint']
                        # Similar validation for worker_endpoint
                        if isinstance(worker_endpoint, list) and len(worker_endpoint) == 2:
                            if not all(isinstance(x, int) for x in worker_endpoint):
                                self.logger.warning(f"⚠️ Invalid worker_endpoint format in dram_view {i}: {worker_endpoint}")
                                return False
                        elif isinstance(worker_endpoint, int):
                            pass
                        else:
                            self.logger.warning(f"⚠️ Invalid worker_endpoint type in dram_view {i}: {type(worker_endpoint)}")
                            return False
            
            self.logger.info(f"✅ SOC descriptor validation passed: {descriptor_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating SOC descriptor {descriptor_path}: {e}")
            return False
    
    def _validate_device_config(self, device_config: Dict[str, Any]) -> bool:
        """Validate device configuration before proceeding."""
        try:
            # Check if we have a reasonable number of devices
            num_devices = device_config.get('num_devices', 0)
            if num_devices < 0 or num_devices > 16:  # Reasonable bounds
                self.logger.error(f"❌ Invalid number of devices: {num_devices}")
                return False
            
            # Check if configuration is internally consistent
            if num_devices > 1 and not device_config.get('optimal_sharding', False):
                self.logger.warning("⚠️ Multiple devices detected but sharding not enabled")
            
            # Validate SOC descriptor path if provided
            soc_descriptor = device_config.get('soc_descriptor')
            if soc_descriptor and not os.path.exists(soc_descriptor):
                self.logger.error(f"❌ SOC descriptor path does not exist: {soc_descriptor}")
                return False
            
            self.logger.info("✅ Device configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating device configuration: {e}")
            return False
    
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
