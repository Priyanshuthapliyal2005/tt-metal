#!/usr/bin/env python3
"""
Hardware utilities module for TTNN hardware detection and device management.
Centralizes all hardware detection logic for the Ministral-8B server.
"""

import os
import logging
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Import compatibility layer for TTNN API migration
from . import ttnn_compat

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class HardwareCapabilities:
    """Structured dataclass containing hardware detection results."""
    ttnn_available: bool
    devices: List[str]
    firmware_status: Dict[str, str]
    soc_descriptor_status: Dict[str, str]
    environment_type: str
    hardware_available: bool = False
    error: Optional[str] = None
    yaml_error_details: Optional[Dict[str, str]] = None
    device_handle: Optional[Any] = None  # Store device handle for cleanup

def get_environment_type() -> str:
    """
    Detect the current environment type consistently.
    
    Returns:
        str: Environment type ('docker', 'koyeb', or 'local')
    """
    is_docker = os.environ.get('IS_DOCKER_ENVIRONMENT') == 'true'
    is_koyeb = os.environ.get('IS_KOYEB_ENVIRONMENT') == 'true'
    
    if is_docker:
        return 'docker'
    elif is_koyeb:
        return 'koyeb'
    else:
        return 'local'

def validate_firmware_files() -> Dict[str, str]:
    """
    Check for required firmware binaries in expected locations.
    
    Returns:
        Dict[str, str]: Mapping of firmware file paths to their status
    """
    firmware_status = {}
    firmware_path = "/workspace/runtime/hw/lib/wormhole"
    
    # Essential firmware files for Wormhole architecture
    firmware_files = [
        "tmu-crt0.o",
        "noc.o", 
        "substitutes.o",
        "idle_erisc.elf",
        "active_erisc.elf"
    ]
    
    try:
        for file in firmware_files:
            file_path = Path(firmware_path) / file
            if file_path.exists():
                try:
                    # Check file size to ensure it's not a stub
                    file_size = file_path.stat().st_size
                    if file_size > 0:
                        firmware_status[str(file_path)] = "present"
                        logger.debug(f"✅ Firmware file found: {file} ({file_size} bytes)")
                    else:
                        firmware_status[str(file_path)] = "empty"
                        logger.warning(f"⚠️ Firmware file is empty: {file}")
                except Exception as e:
                    firmware_status[str(file_path)] = f"error: {e}"
                    logger.error(f"❌ Error checking firmware file {file}: {e}")
            else:
                firmware_status[str(file_path)] = "missing"
                logger.warning(f"⚠️ Missing firmware file: {file}")
    
    except Exception as e:
        logger.error(f"Failed to check firmware directory {firmware_path}: {e}")
        firmware_status[firmware_path] = f"directory_error: {e}"
    
    return firmware_status

def validate_soc_descriptors() -> Dict[str, str]:
    """
    Validate SOC descriptor YAML files and mesh configuration.
    
    Returns:
        Dict[str, str]: Mapping of SOC descriptor paths to their validation status
    """
    soc_status = {}
    
    # SOC descriptor files for different architectures
    soc_descriptor_paths = [
        "/workspace/tt-metal/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml",
        "/workspace/tt-metal/tt_metal/soc_descriptors/wormhole_b0_versim.yaml", 
        "/workspace/tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml"
    ]
    
    for soc_path in soc_descriptor_paths:
        if os.path.exists(soc_path):
            try:
                # Try to parse the YAML file to check for syntax errors
                import yaml
                with open(soc_path, 'r') as f:
                    yaml_content = yaml.safe_load(f)
                
                # Basic validation of required fields
                if isinstance(yaml_content, dict):
                    # Check for essential fields
                    required_fields = ['arch_name', 'grid', 'functional_workers']
                    missing_fields = [field for field in required_fields if field not in yaml_content]
                    
                    if missing_fields:
                        soc_status[soc_path] = f"missing_fields: {missing_fields}"
                        logger.warning(f"⚠️ SOC descriptor missing fields in {soc_path}: {missing_fields}")
                    else:
                        soc_status[soc_path] = "valid"
                        logger.debug(f"✅ SOC descriptor valid: {soc_path}")
                else:
                    soc_status[soc_path] = "invalid_format"
                    logger.error(f"❌ SOC descriptor has invalid format: {soc_path}")
                    
            except yaml.YAMLError as yaml_error:
                soc_status[soc_path] = f"yaml_error: {yaml_error}"
                logger.error(f"❌ YAML parsing error in {soc_path}: {yaml_error}")
                
                # Extract line and column information if available
                if hasattr(yaml_error, 'problem_mark'):
                    mark = yaml_error.problem_mark
                    logger.error(f"   Error at line {mark.line + 1}, column {mark.column + 1}")
                    
            except Exception as e:
                soc_status[soc_path] = f"read_error: {e}"
                logger.error(f"❌ Error reading {soc_path}: {e}")
        else:
            soc_status[soc_path] = "missing"
            logger.warning(f"⚠️ SOC descriptor missing: {soc_path}")
    
    return soc_status

def initialize_tt_device(device_id: int = 0) -> Tuple[Any, Optional[str]]:
    """
    Handle device opening with proper error handling and fallback logic.
    
    Args:
        device_id: TT device ID to open
        
    Returns:
        Tuple[device, error]: Device object and error message if any
    """
    try:
        import ttnn
        
        # Check environment for cloud deployment
        environment = get_environment_type()
        is_cloud = environment in ['docker', 'koyeb']
        
        logger.info(f"Attempting to open TT device {device_id} in {environment} environment")
        
        # Open device with appropriate error handling
        try:
            device = ttnn.open_device(device_id=device_id)
            logger.info(f"✅ Successfully opened TT device {device_id}")
            return device, None
            
        except Exception as device_error:
            error_str = str(device_error)
            
            # Enhanced error classification and handling
            if "YAML" in error_str or "bad conversion" in error_str:
                logger.error(f"❌ YAML parsing error during device initialization: {device_error}")
                
                # Provide specific guidance for YAML errors
                if "line" in error_str and "column" in error_str:
                    line_match = re.search(r'line (\d+)', error_str)
                    col_match = re.search(r'column (\d+)', error_str)
                    if line_match and col_match:
                        line_num = int(line_match.group(1))
                        col_num = int(col_match.group(1))
                        logger.error(f"YAML error at line {line_num}, column {col_num}")
                        
                        # Provide specific guidance for common YAML errors
                        if col_num == 21 and line_num == 29:
                            logger.error("This appears to be the known eth_endpoint format issue:")
                            logger.error("  - Expected: eth_endpoint: 0")
                            logger.error("  - Found: eth_endpoint: [0, 0]")
                            logger.error("  - Solution: Change list format to integer format in SOC descriptor")
                
                return None, f"YAML parsing error: {device_error}"
                
            elif "build failed" in error_str or "link failure" in error_str or "cannot find" in error_str:
                logger.warning(f"Device initialization failed due to firmware build errors: {device_error}")
                return None, f"Firmware build errors: {device_error}"
                
            elif "library_tweaks" in error_str:
                if is_cloud:
                    logger.info("library_tweaks error detected - expected in cloud environments without TT hardware")
                    return None, f"Cloud environment (no hardware): {device_error}"
                else:
                    logger.error(f"Unexpected library_tweaks error in {environment} environment: {device_error}")
                    return None, f"Library configuration error: {device_error}"
                    
            else:
                logger.error(f"Device initialization failed: {device_error}")
                return None, f"Device initialization failed: {device_error}"
                
    except ImportError as e:
        logger.error(f"TTNN import failed: {e}")
        return None, f"TTNN import failed: {e}"
    except Exception as e:
        logger.error(f"Unexpected error during device initialization: {e}")
        return None, f"Unexpected error: {e}"

def detect_hardware_capabilities() -> HardwareCapabilities:
    """
    Detect TTNN availability and hardware status with comprehensive validation.
    
    Returns:
        HardwareCapabilities: Structured hardware detection results
    """
    logger.info("Starting comprehensive hardware capabilities detection")
    
    # Initialize capabilities structure
    capabilities = HardwareCapabilities(
        ttnn_available=False,
        devices=[],
        firmware_status={},
        soc_descriptor_status={},
        environment_type=get_environment_type()
    )
    
    # Validate SOC descriptors first (before TTNN import)
    logger.debug("Validating SOC descriptors...")
    capabilities.soc_descriptor_status = validate_soc_descriptors()
    
    # Check for YAML errors that would prevent device initialization
    yaml_errors = [status for status in capabilities.soc_descriptor_status.values() 
                   if "yaml_error" in status]
    if yaml_errors:
        logger.warning(f"Found YAML errors in SOC descriptors: {yaml_errors}")
        capabilities.yaml_error_details = {
            'file': 'Multiple SOC descriptors',
            'error': '; '.join(yaml_errors),
            'line': 'unknown',
            'column': 'unknown'
        }
    
    # Validate firmware files
    logger.debug("Validating firmware files...")
    capabilities.firmware_status = validate_firmware_files()
    
    # Check firmware availability
    firmware_available = any(status == "present" for status in capabilities.firmware_status.values())
    missing_firmware = [path for path, status in capabilities.firmware_status.items() 
                       if status == "missing"]
    
    if missing_firmware:
        logger.warning(f"Missing firmware files: {missing_firmware}")
        if "idle_erisc.elf" in str(missing_firmware):
            logger.error("idle_erisc.elf missing - this indicates firmware compilation failed")
            logger.error("Try running: ttnn_compat.init_tt_metal(device_id=0) to trigger firmware compilation")
    
    # Try to import and initialize TTNN
    try:
        import ttnn
        capabilities.ttnn_available = True
        logger.info("✅ TTNN module imported successfully")
        
        # Early initialization for better hardware detection
        try:
            # Initialize TT Metal core manager using compatibility layer
            device = ttnn_compat.init_tt_metal(device_id=0)
            logger.debug("✅ TT Metal initialized successfully")
            capabilities.device_handle = device  # Store for cleanup
            
            # Initialize bus table if needed (no-op in new API)
            try:
                ttnn_compat.init_bus_table()
                logger.debug("✅ Bus table initialization completed")
            except Exception as bus_error:
                logger.warning(f"Bus table initialization failed (non-critical): {bus_error}")
                
        except Exception as init_error:
            logger.warning(f"Early TT initialization failed: {init_error}")
        
        # Try to detect hardware devices
        try:
            # Use device enumeration through compatibility layer
            if hasattr(ttnn, 'get_device_ids'):
                devices = ttnn.get_device_ids()
            else:
                # Fallback: if device was successfully opened, assume device 0 exists
                devices = [0] if capabilities.device_handle else []
                
            if devices and len(devices) > 0:
                capabilities.hardware_available = True
                capabilities.devices = [str(device_id) for device_id in devices]
                logger.info(f"✅ TT Hardware detected: {capabilities.devices}")
                
                # Warn if hardware detected but firmware missing
                if missing_firmware:
                    logger.warning(f"Hardware detected but firmware files missing: {missing_firmware}")
                    logger.warning("This will prevent model loading.")
                    capabilities.error = f"Hardware available but missing firmware: {missing_firmware}"
            else:
                logger.info("⚠️ TTNN available but no TT hardware detected")
                capabilities.error = "No TT hardware devices detected"
                
        except Exception as device_error:
            error_str = str(device_error)
            
            # Enhanced error classification
            if "YAML" in error_str or "bad conversion" in error_str:
                logger.error(f"❌ YAML parsing error during device detection: {device_error}")
                capabilities.error = f"YAML parsing error: {device_error}"
                
                # Extract detailed YAML error information
                if "line" in error_str and "column" in error_str:
                    line_match = re.search(r'line (\d+)', error_str)
                    col_match = re.search(r'column (\d+)', error_str)
                    if line_match and col_match:
                        capabilities.yaml_error_details = {
                            'file': 'SOC descriptor (detected during device init)',
                            'error': error_str,
                            'line': line_match.group(1),
                            'column': col_match.group(1)
                        }
                        
            elif "build failed" in error_str or "link failure" in error_str or "cannot find" in error_str:
                logger.warning(f"Hardware detection failed due to firmware build errors: {device_error}")
                capabilities.error = f"Firmware build errors: {device_error}"
                # Still report hardware as available since PCI device was likely detected
                capabilities.hardware_available = True
                capabilities.devices = ['0']  # Assume at least one device
                logger.info("Hardware reported as available despite firmware issues")
                
            elif "library_tweaks" in error_str:
                if capabilities.environment_type in ['docker', 'koyeb']:
                    logger.info("library_tweaks error detected - expected in cloud environments without TT hardware")
                    capabilities.error = f"Cloud environment (no hardware): {device_error}"
                else:
                    logger.error(f"Unexpected library_tweaks error in {capabilities.environment_type} environment")
                    capabilities.error = f"Library configuration error: {device_error}"
                    
            else:
                logger.warning(f"Hardware detection failed: {device_error}")
                capabilities.error = f"Hardware detection failed: {device_error}"
            
    except ImportError as e:
        capabilities.error = f"TTNN import failed: {e}"
        logger.warning(f"❌ TTNN import failed: {e}")
    except Exception as e:
        capabilities.error = f"TTNN initialization failed: {e}"
        logger.warning(f"❌ TTNN initialization failed: {e}")
    finally:
        # Ensure cleanup in error paths
        if capabilities.device_handle and capabilities.error:
            try:
                ttnn_compat.finalize_tt_metal(capabilities.device_handle)
                capabilities.device_handle = None
                logger.debug("Device cleaned up after error")
            except Exception as cleanup_error:
                logger.warning(f"Error during cleanup: {cleanup_error}")
    
    # Log final capabilities summary
    logger.info(f"Hardware detection complete:")
    logger.info(f"  Environment: {capabilities.environment_type}")
    logger.info(f"  TTNN Available: {capabilities.ttnn_available}")
    logger.info(f"  Hardware Available: {capabilities.hardware_available}")
    logger.info(f"  Devices: {capabilities.devices}")
    logger.info(f"  Firmware Status: {len([s for s in capabilities.firmware_status.values() if s == 'present'])} present, {len([s for s in capabilities.firmware_status.values() if s == 'missing'])} missing")
    logger.info(f"  SOC Descriptors: {len([s for s in capabilities.soc_descriptor_status.values() if s == 'valid'])} valid, {len([s for s in capabilities.soc_descriptor_status.values() if 'error' in s])} with errors")
    
    if capabilities.error:
        logger.warning(f"  Error: {capabilities.error}")
    
    return capabilities

def cleanup_tt_resources(device=None):
    """Clean up TT resources on shutdown."""
    try:
        ttnn_compat.finalize_tt_metal(device)
        logger.info("✅ TT Metal resources cleaned up")
    except Exception as e:
        logger.warning(f"Error during TT cleanup: {e}")

# Convenience functions for backward compatibility
def get_ttnn_status() -> Dict[str, Any]:
    """
    Get TTNN status in the legacy format for backward compatibility.
    
    Returns:
        Dict[str, Any]: Legacy TTNN status dictionary
    """
    capabilities = detect_hardware_capabilities()
    
    # Ensure cleanup of device handle if it was opened during detection
    if capabilities.device_handle:
        try:
            ttnn_compat.finalize_tt_metal(capabilities.device_handle)
        except Exception as e:
            logger.warning(f"Error cleaning up device during status check: {e}")
    
    return {
        'ttnn_available': capabilities.ttnn_available,
        'hardware_available': capabilities.hardware_available,
        'devices': capabilities.devices,
        'error': capabilities.error,
        'environment_type': capabilities.environment_type,
        'firmware_available': any(status == "present" for status in capabilities.firmware_status.values()),
        'soc_descriptor_status': capabilities.soc_descriptor_status,
        'yaml_error_details': capabilities.yaml_error_details
    }

def is_hardware_ready() -> bool:
    """
    Quick check if hardware is ready for model loading.
    
    Returns:
        bool: True if hardware and firmware are available
    """
    capabilities = detect_hardware_capabilities()
    firmware_ready = any(status == "present" for status in capabilities.firmware_status.values())
    return capabilities.ttnn_available and capabilities.hardware_available and firmware_ready

def get_recommended_device_id() -> int:
    """
    Get the recommended device ID for model loading.
    
    Returns:
        int: Recommended device ID (0 if no specific recommendation)
    """
    capabilities = detect_hardware_capabilities()
    if capabilities.devices:
        return int(capabilities.devices[0])
    return 0
