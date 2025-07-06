"""
SOC descriptor configuration module for Ministral-8B.

This module handles SOC descriptor selection, mesh configuration validation,
and device topology detection to ensure proper single-device vs multi-device setup.
Addresses the "Did not find edge with expected number of East-West chips 2" error
by ensuring SOC descriptor and mesh configuration match actual hardware topology.
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DeviceTopology(Enum):
    """Device topology types."""
    SINGLE_DEVICE = "single_device"
    MULTI_DEVICE = "multi_device"
    UNKNOWN = "unknown"

class EnvironmentType(Enum):
    """Environment types."""
    LOCAL = "local"
    DOCKER = "docker"
    KOYEB = "koyeb"
    CLOUD = "cloud"

@dataclass
class HardwareCapabilities:
    """Hardware capabilities and status."""
    ttnn_available: bool
    devices: List[str]
    firmware_available: bool
    environment_type: EnvironmentType
    topology: DeviceTopology
    error: Optional[str] = None

@dataclass
class SOCDescriptorInfo:
    """SOC descriptor file information."""
    path: str
    exists: bool
    valid: bool
    topology_type: DeviceTopology
    error: Optional[str] = None
    mesh_config: Optional[Dict[str, Any]] = None

def detect_device_topology() -> Tuple[DeviceTopology, HardwareCapabilities]:
    """
    Determines if setup is single-device or multi-device based on available hardware.
    
    Returns:
        Tuple of (topology, hardware_capabilities)
    """
    logger.info("Detecting device topology...")
    
    # Initialize hardware capabilities
    capabilities = HardwareCapabilities(
        ttnn_available=False,
        devices=[],
        firmware_available=False,
        environment_type=_detect_environment_type(),
        topology=DeviceTopology.UNKNOWN
    )
    
    # Check if we're in a cloud environment where hardware detection is limited
    if capabilities.environment_type in [EnvironmentType.KOYEB, EnvironmentType.CLOUD]:
        logger.info("Cloud environment detected - assuming single device topology")
        capabilities.topology = DeviceTopology.SINGLE_DEVICE
        return DeviceTopology.SINGLE_DEVICE, capabilities
    
    # Try to detect TTNN and hardware
    try:
        import ttnn
        capabilities.ttnn_available = True
        logger.info("✅ TTNN module available")
        
        # Check firmware availability
        capabilities.firmware_available = _check_firmware_availability()
        
        # Try to get device IDs using current TTNN API
        try:
            # Get number of available PCIe devices
            num_devices = ttnn.GetNumPCIeDevices()
            device_ids = []
            
            # Enumerate device IDs
            for i in range(num_devices):
                try:
                    device_id = ttnn.GetPCIeDeviceID(i)
                    device_ids.append(device_id)
                except Exception as device_enum_error:
                    logger.warning(f"Failed to get device ID for index {i}: {device_enum_error}")
                    continue
            
            if device_ids:
                capabilities.devices = [str(device_id) for device_id in device_ids]
                logger.info(f"Detected {len(capabilities.devices)} device(s): {capabilities.devices}")
                
                # Determine topology based on device count
                if len(capabilities.devices) == 1:
                    capabilities.topology = DeviceTopology.SINGLE_DEVICE
                    logger.info("Single device topology detected")
                elif len(capabilities.devices) > 1:
                    capabilities.topology = DeviceTopology.MULTI_DEVICE
                    logger.info(f"Multi-device topology detected with {len(capabilities.devices)} devices")
                else:
                    capabilities.topology = DeviceTopology.UNKNOWN
                    logger.warning("No devices detected")
            else:
                logger.warning("No TT devices found")
                capabilities.topology = DeviceTopology.SINGLE_DEVICE  # Assume single for fallback
                
        except Exception as device_error:
            error_str = str(device_error)
            capabilities.error = error_str
            
            # Check for specific error patterns
            if "YAML" in error_str or "bad conversion" in error_str:
                logger.error(f"YAML parsing error during device detection: {device_error}")
                logger.error("This indicates SOC descriptor configuration issues")
                # Still assume single device for configuration purposes
                capabilities.topology = DeviceTopology.SINGLE_DEVICE
            elif "East-West chips" in error_str:
                logger.error(f"Mesh configuration error: {device_error}")
                logger.error("This indicates multi-device SOC descriptor used with single device")
                capabilities.topology = DeviceTopology.SINGLE_DEVICE
            else:
                logger.warning(f"Device detection failed: {device_error}")
                capabilities.topology = DeviceTopology.SINGLE_DEVICE  # Safe fallback
                
    except ImportError as e:
        logger.warning(f"TTNN not available: {e}")
        capabilities.error = f"TTNN import failed: {e}"
        capabilities.topology = DeviceTopology.SINGLE_DEVICE  # Safe fallback
        
    return capabilities.topology, capabilities

def select_soc_descriptor(topology: DeviceTopology, environment: EnvironmentType) -> SOCDescriptorInfo:
    """
    Chooses appropriate SOC descriptor YAML file based on detected topology.
    
    Args:
        topology: Detected device topology
        environment: Environment type
        
    Returns:
        SOCDescriptorInfo with selected descriptor information
    """
    logger.info(f"Selecting SOC descriptor for topology: {topology.value}, environment: {environment.value}")
    
    # Define SOC descriptor candidates in order of preference
    base_path = "/workspace/tt-metal/tt_metal/soc_descriptors"
    
    if topology == DeviceTopology.SINGLE_DEVICE:
        # Single device descriptors (prefer versim for single device)
        candidates = [
            f"{base_path}/wormhole_b0_versim.yaml",  # Single device simulator
            f"{base_path}/wormhole_b0_80_arch.yaml",  # Full architecture (may work for single)
        ]
    elif topology == DeviceTopology.MULTI_DEVICE:
        # Multi-device descriptors
        candidates = [
            f"{base_path}/wormhole_b0_80_arch.yaml",  # Full multi-device architecture
            f"{base_path}/blackhole_140_arch.yaml",   # Alternative multi-device
        ]
    else:
        # Unknown topology - try safe single device first
        candidates = [
            f"{base_path}/wormhole_b0_versim.yaml",
            f"{base_path}/wormhole_b0_80_arch.yaml",
        ]
    
    # Cloud environments should prefer versim
    if environment in [EnvironmentType.KOYEB, EnvironmentType.CLOUD]:
        candidates = [f"{base_path}/wormhole_b0_versim.yaml"] + candidates
    
    # Try each candidate
    for candidate_path in candidates:
        descriptor_info = _analyze_soc_descriptor(candidate_path, topology)
        if descriptor_info.exists and descriptor_info.valid:
            logger.info(f"Selected SOC descriptor: {candidate_path}")
            return descriptor_info
        elif descriptor_info.exists:
            logger.warning(f"SOC descriptor exists but invalid: {candidate_path} - {descriptor_info.error}")
        else:
            logger.debug(f"SOC descriptor not found: {candidate_path}")
    
    # No valid descriptor found - return the first candidate with error info
    first_candidate = candidates[0] if candidates else f"{base_path}/wormhole_b0_versim.yaml"
    descriptor_info = _analyze_soc_descriptor(first_candidate, topology)
    descriptor_info.error = f"No valid SOC descriptor found for topology {topology.value}"
    logger.error(descriptor_info.error)
    
    return descriptor_info

def validate_mesh_config(descriptor_info: SOCDescriptorInfo, topology: DeviceTopology) -> bool:
    """
    Validates that mesh configuration matches actual hardware setup.
    
    Args:
        descriptor_info: SOC descriptor information
        topology: Detected device topology
        
    Returns:
        True if mesh configuration is valid, False otherwise
    """
    logger.info(f"Validating mesh configuration for topology: {topology.value}")
    
    if not descriptor_info.exists or not descriptor_info.valid:
        logger.error("Cannot validate mesh config - SOC descriptor is invalid")
        return False
    
    try:
        # Load and parse the SOC descriptor
        with open(descriptor_info.path, 'r') as f:
            soc_config = yaml.safe_load(f)
        
        # Store mesh config for reference
        descriptor_info.mesh_config = soc_config
        
        # Check for multi-device indicators in single-device setup
        if topology == DeviceTopology.SINGLE_DEVICE:
            # Look for multi-device patterns that would cause "East-West chips" errors
            multi_device_indicators = []
            
            # Check for galaxy configuration (multi-device)
            if 'galaxy' in soc_config:
                galaxy_config = soc_config['galaxy']
                if 'row' in galaxy_config and len(galaxy_config['row']) > 1:
                    multi_device_indicators.append(f"Galaxy config has {len(galaxy_config['row'])} rows")
                if 'col' in galaxy_config and len(galaxy_config['col']) > 1:
                    multi_device_indicators.append(f"Galaxy config has {len(galaxy_config['col'])} columns")
            
            # Check for nebula_x2 configuration (multi-device)
            if 'nebula_x2' in soc_config:
                multi_device_indicators.append("nebula_x2 configuration detected")
            
            # Check eth configuration for multi-device patterns
            if 'eth' in soc_config and soc_config['eth']:
                multi_device_indicators.append("Ethernet configuration present (multi-device)")
            
            # Check dram_views for multi-device patterns
            if 'dram_views' in soc_config:
                dram_views = soc_config['dram_views']
                if isinstance(dram_views, list) and len(dram_views) > 1:
                    multi_device_indicators.append(f"Multiple DRAM views: {len(dram_views)}")
                elif isinstance(dram_views, list) and len(dram_views) == 1:
                    # Check for multi-device eth_endpoint format
                    dram_view = dram_views[0]
                    if 'eth_endpoint' in dram_view:
                        eth_endpoint = dram_view['eth_endpoint']
                        if isinstance(eth_endpoint, list):
                            multi_device_indicators.append(f"eth_endpoint is list format: {eth_endpoint}")
            
            if multi_device_indicators:
                logger.warning(f"Multi-device indicators found in single-device setup:")
                for indicator in multi_device_indicators:
                    logger.warning(f"  - {indicator}")
                logger.warning("This may cause 'East-West chips' errors")
                return False
            else:
                logger.info("✅ Single-device mesh configuration validated")
                return True
                
        elif topology == DeviceTopology.MULTI_DEVICE:
            # For multi-device, ensure we have proper mesh configuration
            required_multi_configs = []
            
            if 'galaxy' not in soc_config and 'nebula_x2' not in soc_config:
                required_multi_configs.append("No multi-device configuration (galaxy/nebula_x2) found")
            
            if 'eth' in soc_config and not soc_config['eth']:
                required_multi_configs.append("Empty ethernet configuration for multi-device")
            
            if required_multi_configs:
                logger.warning(f"Multi-device setup issues:")
                for issue in required_multi_configs:
                    logger.warning(f"  - {issue}")
                return False
            else:
                logger.info("✅ Multi-device mesh configuration validated")
                return True
        else:
            logger.warning("Unknown topology - cannot validate mesh configuration")
            return False
            
    except Exception as e:
        logger.error(f"Error validating mesh configuration: {e}")
        descriptor_info.error = f"Mesh validation error: {e}"
        return False

def fix_single_device_config(descriptor_path: str) -> bool:
    """
    Ensures single-device setups don't use multi-device mesh configurations.
    Creates a corrected version if needed.
    
    Args:
        descriptor_path: Path to SOC descriptor file
        
    Returns:
        True if configuration was fixed or is already correct, False on error
    """
    logger.info(f"Fixing single-device configuration: {descriptor_path}")
    
    try:
        if not os.path.exists(descriptor_path):
            logger.error(f"SOC descriptor not found: {descriptor_path}")
            return False
        
        # Load current configuration
        with open(descriptor_path, 'r') as f:
            soc_config = yaml.safe_load(f)
        
        # Check if fixes are needed
        fixes_applied = []
        
        # Fix eth_endpoint format if it's a list
        if 'dram_views' in soc_config and isinstance(soc_config['dram_views'], list):
            for i, dram_view in enumerate(soc_config['dram_views']):
                if 'eth_endpoint' in dram_view and isinstance(dram_view['eth_endpoint'], list):
                    # Convert list to single integer (take first element)
                    old_value = dram_view['eth_endpoint']
                    new_value = old_value[0] if old_value else 0
                    soc_config['dram_views'][i]['eth_endpoint'] = new_value
                    fixes_applied.append(f"eth_endpoint: {old_value} -> {new_value}")
        
        # Remove multi-device configurations for single device
        multi_device_keys = ['galaxy', 'nebula_x2']
        for key in multi_device_keys:
            if key in soc_config:
                # Only remove if it has multi-device indicators
                config_section = soc_config[key]
                if isinstance(config_section, dict):
                    # Check for multiple rows/cols
                    has_multi_rows = 'row' in config_section and len(config_section['row']) > 1
                    has_multi_cols = 'col' in config_section and len(config_section['col']) > 1
                    
                    if has_multi_rows or has_multi_cols:
                        # Keep only first row/col for single device
                        if has_multi_rows:
                            first_row = {1: config_section['row'][1]} if 1 in config_section['row'] else {}
                            soc_config[key]['row'] = first_row
                            fixes_applied.append(f"Reduced {key} rows to single device")
                        
                        if has_multi_cols:
                            first_col = {1: config_section['col'][1]} if 1 in config_section['col'] else {}
                            soc_config[key]['col'] = first_col
                            fixes_applied.append(f"Reduced {key} cols to single device")
        
        # Clear ethernet configuration for single device
        if 'eth' in soc_config and soc_config['eth']:
            soc_config['eth'] = []
            fixes_applied.append("Cleared ethernet configuration for single device")
        
        # Apply fixes if any were needed
        if fixes_applied:
            # Create backup
            backup_path = f"{descriptor_path}.backup"
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(descriptor_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Write corrected configuration
            with open(descriptor_path, 'w') as f:
                yaml.dump(soc_config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Applied {len(fixes_applied)} fixes to SOC descriptor:")
            for fix in fixes_applied:
                logger.info(f"  - {fix}")
            
            return True
        else:
            logger.info("SOC descriptor is already configured for single device")
            return True
            
    except Exception as e:
        logger.error(f"Error fixing single-device configuration: {e}")
        return False

def get_soc_descriptor_path(topology: DeviceTopology = None, environment: EnvironmentType = None) -> str:
    """
    Returns correct path to SOC descriptor file, with fallback options.
    
    Args:
        topology: Device topology (auto-detected if None)
        environment: Environment type (auto-detected if None)
        
    Returns:
        Path to the best available SOC descriptor file
    """
    logger.info("Getting SOC descriptor path...")
    
    # Auto-detect if not provided
    if topology is None or environment is None:
        detected_topology, capabilities = detect_device_topology()
        if topology is None:
            topology = detected_topology
        if environment is None:
            environment = capabilities.environment_type
    
    # Select appropriate descriptor
    descriptor_info = select_soc_descriptor(topology, environment)
    
    # Validate and fix if needed for single device
    if topology == DeviceTopology.SINGLE_DEVICE:
        if descriptor_info.exists:
            # Try to fix configuration issues
            if not validate_mesh_config(descriptor_info, topology):
                logger.info("Attempting to fix single-device configuration...")
                fix_single_device_config(descriptor_info.path)
                # Re-validate after fix
                validate_mesh_config(descriptor_info, topology)
    
    # Return the path (even if file doesn't exist - caller should handle)
    logger.info(f"Selected SOC descriptor path: {descriptor_info.path}")
    return descriptor_info.path

def _detect_environment_type() -> EnvironmentType:
    """Detect the current environment type."""
    if os.environ.get('IS_KOYEB_ENVIRONMENT') == 'true':
        return EnvironmentType.KOYEB
    elif os.environ.get('IS_DOCKER_ENVIRONMENT') == 'true':
        return EnvironmentType.DOCKER
    elif any(cloud_indicator in os.environ.get('HOSTNAME', '').lower() 
             for cloud_indicator in ['koyeb', 'heroku', 'vercel', 'netlify']):
        return EnvironmentType.CLOUD
    else:
        return EnvironmentType.LOCAL

def _check_firmware_availability() -> bool:
    """Check if required firmware files are available."""
    firmware_path = "/workspace/runtime/hw/lib/wormhole"
    required_files = ["idle_erisc.elf", "tmu-crt0.o", "noc.o", "substitutes.o"]
    
    missing_files = []
    for filename in required_files:
        file_path = os.path.join(firmware_path, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    if missing_files:
        logger.warning(f"Missing firmware files: {missing_files}")
        return False
    else:
        logger.info("✅ All required firmware files found")
        return True

def _analyze_soc_descriptor(path: str, expected_topology: DeviceTopology) -> SOCDescriptorInfo:
    """Analyze a SOC descriptor file for validity and topology compatibility."""
    descriptor_info = SOCDescriptorInfo(
        path=path,
        exists=os.path.exists(path),
        valid=False,
        topology_type=DeviceTopology.UNKNOWN
    )
    
    if not descriptor_info.exists:
        descriptor_info.error = "File does not exist"
        return descriptor_info
    
    try:
        # Try to parse YAML
        with open(path, 'r') as f:
            soc_config = yaml.safe_load(f)
        
        # Determine topology from configuration
        if 'galaxy' in soc_config:
            galaxy_config = soc_config['galaxy']
            row_count = len(galaxy_config.get('row', {}))
            col_count = len(galaxy_config.get('col', {}))
            if row_count > 1 or col_count > 1:
                descriptor_info.topology_type = DeviceTopology.MULTI_DEVICE
            else:
                descriptor_info.topology_type = DeviceTopology.SINGLE_DEVICE
        elif 'nebula_x2' in soc_config:
            descriptor_info.topology_type = DeviceTopology.MULTI_DEVICE
        elif 'grid' in soc_config:
            # Simple grid configuration (like versim) - single device
            descriptor_info.topology_type = DeviceTopology.SINGLE_DEVICE
        else:
            descriptor_info.topology_type = DeviceTopology.UNKNOWN
        
        # Check for common YAML issues
        if 'dram_views' in soc_config:
            dram_views = soc_config['dram_views']
            if isinstance(dram_views, list):
                for dram_view in dram_views:
                    if 'eth_endpoint' in dram_view:
                        eth_endpoint = dram_view['eth_endpoint']
                        if isinstance(eth_endpoint, list) and expected_topology == DeviceTopology.SINGLE_DEVICE:
                            descriptor_info.error = f"eth_endpoint list format incompatible with single device: {eth_endpoint}"
                            return descriptor_info
        
        descriptor_info.valid = True
        logger.debug(f"SOC descriptor analysis: {path} - topology: {descriptor_info.topology_type.value}")
        
    except yaml.YAMLError as e:
        descriptor_info.error = f"YAML parsing error: {e}"
        logger.error(f"YAML error in {path}: {e}")
    except Exception as e:
        descriptor_info.error = f"Analysis error: {e}"
        logger.error(f"Error analyzing {path}: {e}")
    
    return descriptor_info

# Convenience functions for backward compatibility
def get_hardware_status() -> Dict[str, Any]:
    """Get comprehensive hardware status for debugging."""
    topology, capabilities = detect_device_topology()
    descriptor_info = select_soc_descriptor(topology, capabilities.environment_type)
    mesh_valid = validate_mesh_config(descriptor_info, topology)
    
    return {
        'topology': topology.value,
        'environment': capabilities.environment_type.value,
        'ttnn_available': capabilities.ttnn_available,
        'devices': capabilities.devices,
        'firmware_available': capabilities.firmware_available,
        'soc_descriptor_path': descriptor_info.path,
        'soc_descriptor_exists': descriptor_info.exists,
        'soc_descriptor_valid': descriptor_info.valid,
        'mesh_config_valid': mesh_valid,
        'errors': [
            capabilities.error,
            descriptor_info.error
        ] if capabilities.error or descriptor_info.error else []
    }

def ensure_compatible_configuration() -> bool:
    """
    Ensure SOC descriptor configuration is compatible with detected hardware.
    This is the main function to call for automatic configuration.
    
    Returns:
        True if configuration is compatible or was successfully fixed
    """
    logger.info("Ensuring compatible SOC descriptor configuration...")
    
    try:
        # Detect current setup
        topology, capabilities = detect_device_topology()
        
        # Get appropriate descriptor
        descriptor_path = get_soc_descriptor_path(topology, capabilities.environment_type)
        
        # Validate and fix if needed
        descriptor_info = SOCDescriptorInfo(
            path=descriptor_path,
            exists=os.path.exists(descriptor_path),
            valid=False,
            topology_type=topology
        )
        
        if descriptor_info.exists:
            mesh_valid = validate_mesh_config(descriptor_info, topology)
            if not mesh_valid and topology == DeviceTopology.SINGLE_DEVICE:
                logger.info("Fixing single-device configuration...")
                fix_success = fix_single_device_config(descriptor_path)
                if fix_success:
                    # Re-validate
                    mesh_valid = validate_mesh_config(descriptor_info, topology)
                    if mesh_valid:
                        logger.info("✅ SOC descriptor configuration fixed and validated")
                        return True
                    else:
                        logger.error("Configuration fix failed validation")
                        return False
                else:
                    logger.error("Failed to fix SOC descriptor configuration")
                    return False
            elif mesh_valid:
                logger.info("✅ SOC descriptor configuration is already compatible")
                return True
            else:
                logger.error("SOC descriptor configuration is incompatible and cannot be automatically fixed")
                return False
        else:
            logger.error(f"SOC descriptor file not found: {descriptor_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error ensuring compatible configuration: {e}")
        return False
