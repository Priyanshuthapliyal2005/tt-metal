#!/usr/bin/env python3
"""
Real-time deployment monitoring for Ministral-8B in production.
Tracks system metrics, model performance, and provides alerts.
"""

import time
import json
import asyncio
import logging
import psutil
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeploymentMonitor:
    """Real-time monitoring for Ministral-8B deployment."""
    
    def __init__(self, 
                 server_url: str = "http://localhost:8080",
                 check_interval: int = 30,
                 alert_thresholds: Optional[Dict] = None):
        self.server_url = server_url
        self.check_interval = check_interval
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': 80,
            'memory_percent': 85,
            'response_time_ms': 5000,
            'error_rate_percent': 5
        }
        
        self.metrics_history = []
        self.alert_history = []
        self.is_monitoring = False
        self.start_time = time.time()
        
        # Performance tracking
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0,
            'response_times': [],
            'uptime_seconds': 0
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network (if available)
            network = psutil.net_io_counters()
            
            # TTNN Device metrics (placeholder)
            ttnn_metrics = self._get_ttnn_metrics()
            
            return {
                'timestamp': time.time(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count()
                },
                'memory': {
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'percent': memory.percent
                },
                'disk': {
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                },
                'ttnn': ttnn_metrics
            }
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def _get_ttnn_metrics(self) -> Dict[str, Any]:
        """Get TTNN device metrics (placeholder for actual implementation)."""
        try:
            # In real implementation, this would query TTNN device status
            import ttnn
            num_devices = ttnn.get_num_devices()
            
            return {
                'num_devices': num_devices,
                'devices_active': num_devices,  # Placeholder
                'memory_utilization': [75.0] * num_devices,  # Placeholder
                'temperature': [65.0] * num_devices  # Placeholder
            }
        except:
            return {
                'num_devices': 0,
                'devices_active': 0,
                'memory_utilization': [],
                'temperature': []
            }
    
    def check_server_health(self) -> Dict[str, Any]:
        """Check server health and responsiveness."""
        try:
            start_time = time.time()
            response = requests.get(f"{self.server_url}/health", timeout=10)
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            health_data = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time_ms': response_time,
                'status_code': response.status_code,
                'timestamp': time.time()
            }
            
            # Try to get server stats if available
            try:
                stats_response = requests.get(f"{self.server_url}/stats", timeout=5)
                if stats_response.status_code == 200:
                    health_data['server_stats'] = stats_response.json()
            except:
                pass
            
            # Update performance stats
            self.performance_stats['total_requests'] += 1
            if response.status_code == 200:
                self.performance_stats['successful_requests'] += 1
            else:
                self.performance_stats['failed_requests'] += 1
            
            self.performance_stats['response_times'].append(response_time)
            if len(self.performance_stats['response_times']) > 100:
                self.performance_stats['response_times'] = self.performance_stats['response_times'][-100:]
            
            self.performance_stats['avg_response_time'] = sum(self.performance_stats['response_times']) / len(self.performance_stats['response_times'])
            
            return health_data
            
        except requests.RequestException as e:
            logger.error(f"Server health check failed: {e}")
            self.performance_stats['total_requests'] += 1
            self.performance_stats['failed_requests'] += 1
            return {
                'status': 'unreachable',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def test_inference_performance(self) -> Dict[str, Any]:
        """Test model inference performance with a sample request."""
        try:
            test_payload = {
                "prompt": "What is artificial intelligence?",
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/generate",
                json=test_payload,
                timeout=30
            )
            inference_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'status': 'success',
                    'inference_time_ms': inference_time * 1000,
                    'input_tokens': len(test_payload['prompt'].split()),
                    'output_tokens': len(result.get('response', '').split()),
                    'tokens_per_second': (len(test_payload['prompt'].split()) + len(result.get('response', '').split())) / inference_time,
                    'timestamp': time.time()
                }
            else:
                return {
                    'status': 'failed',
                    'error': f"HTTP {response.status_code}",
                    'inference_time_ms': inference_time * 1000,
                    'timestamp': time.time()
                }
                
        except Exception as e:
            logger.error(f"Inference test failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        
        # CPU alert
        if metrics.get('cpu', {}).get('percent', 0) > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'type': 'cpu_high',
                'message': f"High CPU usage: {metrics['cpu']['percent']:.1f}%",
                'severity': 'warning',
                'timestamp': time.time()
            })
        
        # Memory alert
        if metrics.get('memory', {}).get('percent', 0) > self.alert_thresholds['memory_percent']:
            alerts.append({
                'type': 'memory_high',
                'message': f"High memory usage: {metrics['memory']['percent']:.1f}%",
                'severity': 'warning',
                'timestamp': time.time()
            })
        
        # Disk space alert
        if metrics.get('disk', {}).get('used_percent', 0) > 90:
            alerts.append({
                'type': 'disk_space_low',
                'message': f"Low disk space: {metrics['disk']['used_percent']:.1f}% used",
                'severity': 'critical',
                'timestamp': time.time()
            })
        
        # Response time alert
        if self.performance_stats['avg_response_time'] > self.alert_thresholds['response_time_ms']:
            alerts.append({
                'type': 'slow_response',
                'message': f"Slow response time: {self.performance_stats['avg_response_time']:.1f}ms avg",
                'severity': 'warning',
                'timestamp': time.time()
            })
        
        # Error rate alert
        if self.performance_stats['total_requests'] > 0:
            error_rate = (self.performance_stats['failed_requests'] / self.performance_stats['total_requests']) * 100
            if error_rate > self.alert_thresholds['error_rate_percent']:
                alerts.append({
                    'type': 'high_error_rate',
                    'message': f"High error rate: {error_rate:.1f}%",
                    'severity': 'critical',
                    'timestamp': time.time()
                })
        
        return alerts
    
    def monitor_loop(self):
        """Main monitoring loop."""
        logger.info(f"🔍 Starting deployment monitoring (check interval: {self.check_interval}s)")
        
        while self.is_monitoring:
            try:
                # Collect metrics
                system_metrics = self.get_system_metrics()
                health_check = self.check_server_health()
                inference_test = self.test_inference_performance()
                
                # Update uptime
                self.performance_stats['uptime_seconds'] = time.time() - self.start_time
                
                # Combine all metrics
                current_metrics = {
                    'timestamp': time.time(),
                    'system': system_metrics,
                    'health': health_check,
                    'inference': inference_test,
                    'performance': self.performance_stats.copy()
                }
                
                # Add to history
                self.metrics_history.append(current_metrics)
                if len(self.metrics_history) > 1000:  # Keep last 1000 entries
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Check for alerts
                alerts = self.check_alerts(system_metrics)
                if alerts:
                    for alert in alerts:
                        self.alert_history.append(alert)
                        logger.warning(f"🚨 ALERT: {alert['message']}")
                
                # Log status
                self.log_status(current_metrics)
                
                # Wait for next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def log_status(self, metrics: Dict[str, Any]):
        """Log current status summary."""
        system = metrics.get('system', {})
        health = metrics.get('health', {})
        inference = metrics.get('inference', {})
        perf = metrics.get('performance', {})
        
        # System status
        cpu_percent = system.get('cpu', {}).get('percent', 0)
        memory_percent = system.get('memory', {}).get('percent', 0)
        
        # Server status
        server_status = health.get('status', 'unknown')
        response_time = health.get('response_time_ms', 0)
        
        # Inference status
        inference_status = inference.get('status', 'unknown')
        tokens_per_sec = inference.get('tokens_per_second', 0)
        
        # Performance stats
        uptime = perf.get('uptime_seconds', 0)
        total_requests = perf.get('total_requests', 0)
        error_rate = (perf.get('failed_requests', 0) / max(total_requests, 1)) * 100
        
        logger.info(f"💻 System: CPU {cpu_percent:.1f}%, RAM {memory_percent:.1f}%")
        logger.info(f"🌐 Server: {server_status} ({response_time:.1f}ms)")
        logger.info(f"🧠 Inference: {inference_status} ({tokens_per_sec:.1f} tok/s)")
        logger.info(f"📊 Stats: {total_requests} requests, {error_rate:.1f}% errors, {uptime/3600:.1f}h uptime")
    
    def start_monitoring(self):
        """Start the monitoring process."""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔍 Deployment monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring process."""
        self.is_monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Deployment monitoring stopped")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics."""
        if not self.metrics_history:
            return {'error': 'No metrics collected yet'}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 entries
        
        # Calculate averages
        avg_cpu = sum(m.get('system', {}).get('cpu', {}).get('percent', 0) for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.get('system', {}).get('memory', {}).get('percent', 0) for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.get('health', {}).get('response_time_ms', 0) for m in recent_metrics) / len(recent_metrics)
        
        # Server health status
        healthy_checks = sum(1 for m in recent_metrics if m.get('health', {}).get('status') == 'healthy')
        health_percentage = (healthy_checks / len(recent_metrics)) * 100
        
        return {
            'monitoring_duration_hours': (time.time() - self.start_time) / 3600,
            'total_metrics_collected': len(self.metrics_history),
            'recent_averages': {
                'cpu_percent': avg_cpu,
                'memory_percent': avg_memory,
                'response_time_ms': avg_response_time,
                'health_percentage': health_percentage
            },
            'performance_stats': self.performance_stats.copy(),
            'active_alerts': len(self.alert_history),
            'last_check': self.metrics_history[-1]['timestamp'] if self.metrics_history else None
        }
    
    def save_metrics(self, filepath: str):
        """Save collected metrics to file."""
        data = {
            'monitoring_session': {
                'start_time': self.start_time,
                'duration_seconds': time.time() - self.start_time,
                'server_url': self.server_url,
                'check_interval': self.check_interval
            },
            'metrics_history': self.metrics_history,
            'alert_history': self.alert_history,
            'performance_stats': self.performance_stats,
            'summary': self.get_metrics_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📈 Metrics saved to {filepath}")

def main():
    """Main function for standalone monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Ministral-8B deployment")
    parser.add_argument("--server-url", default="http://localhost:8080",
                       help="Server URL to monitor")
    parser.add_argument("--interval", type=int, default=30,
                       help="Check interval in seconds")
    parser.add_argument("--duration", type=int, default=3600,
                       help="Monitoring duration in seconds (default: 1 hour)")
    parser.add_argument("--output", default="monitoring_results.json",
                       help="Output file for metrics")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = DeploymentMonitor(
        server_url=args.server_url,
        check_interval=args.interval
    )
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Monitor for specified duration
        logger.info(f"🕐 Monitoring for {args.duration} seconds...")
        time.sleep(args.duration)
        
        # Stop monitoring and save results
        monitor.stop_monitoring()
        monitor.save_metrics(args.output)
        
        # Print summary
        summary = monitor.get_metrics_summary()
        print("\n" + "="*60)
        print("🔍 DEPLOYMENT MONITORING SUMMARY")
        print("="*60)
        print(f"Duration: {summary['monitoring_duration_hours']:.2f} hours")
        print(f"Metrics Collected: {summary['total_metrics_collected']}")
        print(f"Average CPU: {summary['recent_averages']['cpu_percent']:.1f}%")
        print(f"Average Memory: {summary['recent_averages']['memory_percent']:.1f}%")
        print(f"Average Response Time: {summary['recent_averages']['response_time_ms']:.1f}ms")
        print(f"Server Health: {summary['recent_averages']['health_percentage']:.1f}%")
        print(f"Total Requests: {summary['performance_stats']['total_requests']}")
        print(f"Active Alerts: {summary['active_alerts']}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
        monitor.stop_monitoring()
        monitor.save_metrics(args.output)

if __name__ == "__main__":
    main()
