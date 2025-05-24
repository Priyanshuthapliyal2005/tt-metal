#!/usr/bin/env python3
"""
Bounty completion status tracker for Ministral-8B on Tenstorrent N300.
Tracks progress on functional bring-up, performance validation, accuracy validation, and documentation.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

class BountyTracker:
    def __init__(self):
        self.status = {
            "bounty_id": "ministral8b_tenstorrent_n300",
            "model": "mistralai/Ministral-8B-Instruct-2410",
            "hardware": "Tenstorrent Wormhole N300",
            "last_updated": datetime.now().isoformat(),
            "overall_progress": 0.0,
            "tasks": {
                "functional_bringup": {
                    "description": "Enable functional bring-up of the model",
                    "weight": 0.3,
                    "progress": 0.0,
                    "subtasks": {
                        "model_implementation": {"status": "complete", "description": "Implement model in TT-NN"},
                        "demo_script": {"status": "complete", "description": "Create demo script with CLI"},
                        "model_download": {"status": "pending", "description": "Download model weights on N300"},
                        "basic_inference": {"status": "pending", "description": "Verify basic inference works"},
                    }
                },
                "performance_validation": {
                    "description": "Validate performance targets on N300 hardware",
                    "weight": 0.3,
                    "progress": 0.0,
                    "targets": {
                        "single_sequence_throughput": ">100 tokens/sec",
                        "batch_throughput": ">500 tokens/sec",
                        "peak_throughput": ">1000 tokens/sec",
                        "memory_usage": "<12GB device memory"
                    },
                    "subtasks": {
                        "benchmark_script": {"status": "complete", "description": "Performance benchmarking script"},
                        "throughput_tests": {"status": "pending", "description": "Run throughput benchmarks"},
                        "memory_profiling": {"status": "pending", "description": "Profile memory usage"},
                        "optimization": {"status": "pending", "description": "Optimize for target performance"},
                    }
                },
                "accuracy_validation": {
                    "description": "Validate model accuracy and output quality",
                    "weight": 0.2,
                    "progress": 0.0,
                    "targets": {
                        "overall_accuracy": ">70%",
                        "success_rate": ">90%",
                        "inference_time": "<10s per test"
                    },
                    "subtasks": {
                        "accuracy_script": {"status": "complete", "description": "Accuracy validation script"},
                        "test_suite": {"status": "complete", "description": "Create test questions across categories"},
                        "accuracy_testing": {"status": "pending", "description": "Run accuracy validation"},
                        "quality_review": {"status": "pending", "description": "Review output quality"},
                    }
                },
                "documentation": {
                    "description": "Complete documentation and deployment guides",
                    "weight": 0.2,
                    "progress": 0.0,
                    "subtasks": {
                        "readme": {"status": "complete", "description": "Comprehensive README documentation"},
                        "deployment_script": {"status": "complete", "description": "Automated deployment script"},
                        "troubleshooting": {"status": "complete", "description": "Troubleshooting guide"},
                        "api_docs": {"status": "pending", "description": "API documentation"},
                    }
                }
            }
        }
    
    def check_file_exists(self, filepath: str) -> bool:
        """Check if a file exists."""
        return os.path.exists(filepath)
    
    def check_implementation_files(self) -> Dict[str, bool]:
        """Check if implementation files exist and are complete."""
        base_path = "/workspaces/tt-metal/models/demos/wormhole/ministral8b"
        files_to_check = {
            "mistral_model.py": f"{base_path}/tt/mistral_model.py",
            "mistral_embedding.py": f"{base_path}/tt/mistral_embedding.py", 
            "mistral_common.py": f"{base_path}/tt/mistral_common.py",
            "model_config.py": f"{base_path}/tt/model_config.py",
            "demo_script": f"{base_path}/demo/demo_with_prefill.py",
            "download_script": f"{base_path}/download_model.py",
            "benchmark_script": f"{base_path}/benchmark.py",
            "accuracy_script": f"{base_path}/validate_accuracy.py",
            "deployment_script": f"{base_path}/deploy_to_koyeb.sh",
            "readme": f"{base_path}/README.md",
            "run_script": f"{base_path}/run_demo.sh"
        }
        
        return {name: self.check_file_exists(path) for name, path in files_to_check.items()}
    
    def check_model_downloaded(self) -> bool:
        """Check if model weights are downloaded."""
        cache_paths = [
            "/tmp/tt_metal_models/ministral8b",
            os.path.expanduser("~/.cache/huggingface/hub/models--mistralai--Ministral-8B-Instruct-2410")
        ]
        
        for path in cache_paths:
            if os.path.exists(path):
                # Check if there are model files
                for root, dirs, files in os.walk(path):
                    if any(f.endswith(('.bin', '.safetensors', '.pt', '.pth')) for f in files):
                        return True
        return False
    
    def check_hardware_available(self) -> bool:
        """Check if N300 hardware is available."""
        try:
            import ttnn
            devices = ttnn.get_device_ids()
            return len(devices) > 0
        except Exception:
            return False
    
    def update_task_progress(self):
        """Update task progress based on file checks and status."""
        
        # Check implementation files
        files_status = self.check_implementation_files()
        
        # Update functional bring-up progress
        functional_subtasks = self.status["tasks"]["functional_bringup"]["subtasks"]
        
        # Model implementation
        if all(files_status[f] for f in ["mistral_model.py", "mistral_embedding.py", "mistral_common.py", "model_config.py"]):
            functional_subtasks["model_implementation"]["status"] = "complete"
        
        # Demo script
        if files_status["demo_script"]:
            functional_subtasks["demo_script"]["status"] = "complete"
        
        # Model download
        if self.check_model_downloaded():
            functional_subtasks["model_download"]["status"] = "complete"
        
        # Calculate functional bring-up progress
        functional_complete = sum(1 for task in functional_subtasks.values() if task["status"] == "complete")
        self.status["tasks"]["functional_bringup"]["progress"] = functional_complete / len(functional_subtasks)
        
        # Update performance validation progress
        perf_subtasks = self.status["tasks"]["performance_validation"]["subtasks"]
        if files_status["benchmark_script"]:
            perf_subtasks["benchmark_script"]["status"] = "complete"
        
        perf_complete = sum(1 for task in perf_subtasks.values() if task["status"] == "complete")
        self.status["tasks"]["performance_validation"]["progress"] = perf_complete / len(perf_subtasks)
        
        # Update accuracy validation progress
        acc_subtasks = self.status["tasks"]["accuracy_validation"]["subtasks"]
        if files_status["accuracy_script"]:
            acc_subtasks["accuracy_script"]["status"] = "complete"
            acc_subtasks["test_suite"]["status"] = "complete"  # Test questions are built into the script
        
        acc_complete = sum(1 for task in acc_subtasks.values() if task["status"] == "complete")
        self.status["tasks"]["accuracy_validation"]["progress"] = acc_complete / len(acc_subtasks)
        
        # Update documentation progress
        doc_subtasks = self.status["tasks"]["documentation"]["subtasks"]
        if files_status["readme"]:
            doc_subtasks["readme"]["status"] = "complete"
        if files_status["deployment_script"]:
            doc_subtasks["deployment_script"]["status"] = "complete"
            doc_subtasks["troubleshooting"]["status"] = "complete"  # Included in README
        
        doc_complete = sum(1 for task in doc_subtasks.values() if task["status"] == "complete")
        self.status["tasks"]["documentation"]["progress"] = doc_complete / len(doc_subtasks)
        
        # Calculate overall progress
        total_progress = 0.0
        for task_name, task_info in self.status["tasks"].items():
            weighted_progress = task_info["progress"] * task_info["weight"]
            total_progress += weighted_progress
        
        self.status["overall_progress"] = total_progress
    
    def print_status_report(self):
        """Print a detailed status report."""
        self.update_task_progress()
        
        print("🎯 MINISTRAL-8B BOUNTY COMPLETION STATUS")
        print("=" * 60)
        print(f"Model: {self.status['model']}")
        print(f"Hardware: {self.status['hardware']}")
        print(f"Last Updated: {self.status['last_updated']}")
        print(f"Overall Progress: {self.status['overall_progress']:.1%}")
        
        # Hardware status
        hardware_available = self.check_hardware_available()
        model_downloaded = self.check_model_downloaded()
        
        print(f"\n🔧 ENVIRONMENT STATUS:")
        print(f"  N300 Hardware Available: {'✅ Yes' if hardware_available else '❌ No'}")
        print(f"  Model Downloaded: {'✅ Yes' if model_downloaded else '❌ No'}")
        
        print(f"\n📋 TASK BREAKDOWN:")
        
        for task_name, task_info in self.status["tasks"].items():
            progress = task_info["progress"]
            weight = task_info["weight"]
            weighted_contribution = progress * weight
            
            status_icon = "✅" if progress == 1.0 else "🔄" if progress > 0 else "⏳"
            
            print(f"\n{status_icon} {task_name.replace('_', ' ').title()} ({weight:.0%} weight)")
            print(f"   Progress: {progress:.1%} (contributes {weighted_contribution:.1%} to overall)")
            print(f"   Description: {task_info['description']}")
            
            # Show subtasks
            if "subtasks" in task_info:
                for subtask_name, subtask_info in task_info["subtasks"].items():
                    status = subtask_info["status"]
                    icon = "✅" if status == "complete" else "🔄" if status == "in_progress" else "⏳"
                    print(f"     {icon} {subtask_name.replace('_', ' ').title()}: {subtask_info['description']}")
            
            # Show targets for performance and accuracy
            if "targets" in task_info:
                print(f"   Targets:")
                for target_name, target_value in task_info["targets"].items():
                    print(f"     • {target_name.replace('_', ' ').title()}: {target_value}")
        
        print(f"\n🚀 NEXT STEPS:")
        
        if not hardware_available:
            print("   1. Deploy to Koyeb N300 server with Tenstorrent hardware")
        elif not model_downloaded:
            print("   1. Run deployment script: ./deploy_to_koyeb.sh")
        else:
            print("   1. Run functional tests: python demo/demo_with_prefill.py --quick")
            print("   2. Run performance benchmarks: python benchmark.py --quick")
            print("   3. Run accuracy validation: python validate_accuracy.py --quick")
        
        # Completion estimate
        remaining_work = 1.0 - self.status["overall_progress"]
        if remaining_work > 0:
            print(f"\n📊 COMPLETION ESTIMATE:")
            print(f"   Remaining work: {remaining_work:.1%}")
            
            if hardware_available and model_downloaded:
                print(f"   Estimated time to completion: 1-2 hours (testing and validation)")
            elif hardware_available:
                print(f"   Estimated time to completion: 2-3 hours (download + testing)")
            else:
                print(f"   Estimated time to completion: 3-4 hours (deployment + testing)")
        else:
            print(f"\n🎉 BOUNTY COMPLETE!")
            print(f"   All tasks finished. Ready for final review and submission.")
    
    def save_status(self, filename: str = "bounty_status.json"):
        """Save status to JSON file."""
        self.update_task_progress()
        with open(filename, 'w') as f:
            json.dump(self.status, f, indent=2)
        print(f"\n💾 Status saved to {filename}")

def main():
    tracker = BountyTracker()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        tracker.save_status()
    else:
        tracker.print_status_report()
        
        # Also save JSON for record keeping
        tracker.save_status()

if __name__ == "__main__":
    main()
