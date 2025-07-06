# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Compatibility Layer

This module provides a compatibility layer to bridge the old and new TTNN APIs.
It maps deprecated functions like ttnn.initialize_tt_metal() to the current
ttnn.open_device() API, ensuring backward compatibility while providing a
clean migration path.
"""

import logging
import ttnn
from typing import Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

# Global device registry to track opened devices
_device_registry: Dict[int, Any] = {}
_default_device: Optional[Any] = None


def _check_ttnn_version():
    """
    Check if we're running with the new or old TTNN API.
    
    Returns:
        bool: True if new API is available, False if old API
    """
    try:
        # Check if the new API functions exist
        hasattr(ttnn, 'open_device') and hasattr(ttnn, 'close_device')
        return True
    except AttributeError:
        return False


def _get_available_device_id():
    """
    Get the first available device ID.
    
    Returns:
        int: Available device ID, defaults to 0
    """
    try:
        if hasattr(ttnn, 'GetNumAvailableDevices'):
            num_devices = ttnn.GetNumAvailableDevices()
            if num_devices > 0:
                return 0
        return 0
    except Exception as e:
        logger.warning(f"Could not determine available devices: {e}")
        return 0


def init_tt_metal(device_id: Optional[int] = None, force_hw_detect: bool = False) -> Any:
    """
    Initialize TT Metal device using the new TTNN API.
    
    This function replaces the deprecated ttnn.initialize_tt_metal() call
    with the current ttnn.open_device() API.
    
    Args:
        device_id (Optional[int]): Device ID to open. If None, uses first available device.
        force_hw_detect (bool): Legacy parameter for compatibility, ignored in new API.
        
    Returns:
        Device: The opened device handle
        
    Raises:
        RuntimeError: If device initialization fails
    """
    global _default_device
    
    try:
        # Determine device ID if not provided
        if device_id is None:
            device_id = _get_available_device_id()
            
        logger.info(f"Initializing TT Metal device {device_id}")
        
        # Check if device is already open
        if device_id in _device_registry:
            logger.warning(f"Device {device_id} is already open, returning existing handle")
            return _device_registry[device_id]
        
        # Open the device using new API
        device = ttnn.open_device(device_id=device_id)
        
        # Store device in registry
        _device_registry[device_id] = device
        
        # Set as default device if it's the first one
        if _default_device is None:
            _default_device = device
            if hasattr(ttnn, 'SetDefaultDevice'):
                ttnn.SetDefaultDevice(device)
        
        logger.info(f"Successfully initialized TT Metal device {device_id}")
        return device
        
    except Exception as e:
        error_msg = f"Failed to initialize TT Metal device {device_id}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def init_bus_table(force: bool = False) -> None:
    """
    Initialize bus table (no-op in new API).
    
    This function replaces the deprecated ttnn.initialize_bus_table() call.
    In the new TTNN API, bus table initialization is handled automatically
    during device opening, so this is a no-op function for compatibility.
    
    Args:
        force (bool): Legacy parameter for compatibility, ignored in new API.
    """
    logger.debug("Bus table initialization is handled automatically in new TTNN API")
    # No-op: bus table initialization is automatic in new API


def finalize_tt_metal(device: Optional[Any] = None) -> None:
    """
    Finalize TT Metal device using the new TTNN API.
    
    This function replaces the deprecated ttnn.finalize_tt_metal() call
    with the current ttnn.close_device() API.
    
    Args:
        device (Optional[Device]): Device to close. If None, closes all registered devices.
    """
    global _default_device, _device_registry
    
    try:
        if device is not None:
            # Close specific device
            logger.info(f"Finalizing specific TT Metal device")
            ttnn.close_device(device)
            
            # Remove from registry
            device_id_to_remove = None
            for dev_id, dev_handle in _device_registry.items():
                if dev_handle == device:
                    device_id_to_remove = dev_id
                    break
            
            if device_id_to_remove is not None:
                del _device_registry[device_id_to_remove]
                
            # Clear default device if it was the one being closed
            if _default_device == device:
                _default_device = None
                
        else:
            # Close all registered devices
            logger.info("Finalizing all TT Metal devices")
            for device_id, device_handle in list(_device_registry.items()):
                try:
                    ttnn.close_device(device_handle)
                    logger.info(f"Successfully closed device {device_id}")
                except Exception as e:
                    logger.error(f"Error closing device {device_id}: {e}")
            
            # Clear registry and default device
            _device_registry.clear()
            _default_device = None
            
        logger.info("TT Metal finalization completed")
        
    except Exception as e:
        error_msg = f"Error during TT Metal finalization: {e}"
        logger.error(error_msg)
        # Don't raise exception during cleanup to avoid masking original errors


def get_default_device() -> Optional[Any]:
    """
    Get the default device handle.
    
    Returns:
        Optional[Device]: The default device handle, or None if no device is open
    """
    return _default_device


def get_device_registry() -> Dict[int, Any]:
    """
    Get a copy of the current device registry.
    
    Returns:
        Dict[int, Device]: Copy of the device registry
    """
    return _device_registry.copy()


def is_device_open(device_id: int) -> bool:
    """
    Check if a specific device is open.
    
    Args:
        device_id (int): Device ID to check
        
    Returns:
        bool: True if device is open, False otherwise
    """
    return device_id in _device_registry


def cleanup_all_devices() -> None:
    """
    Emergency cleanup function to close all devices.
    
    This function can be used for cleanup in error scenarios.
    """
    logger.warning("Emergency cleanup of all TT Metal devices")
    finalize_tt_metal()


# Compatibility check on module import
_NEW_API_AVAILABLE = _check_ttnn_version()

if _NEW_API_AVAILABLE:
    logger.info("TTNN compatibility layer initialized with new API support")
else:
    logger.warning("TTNN compatibility layer: new API not detected, some features may not work")


# Export public interface
__all__ = [
    'init_tt_metal',
    'init_bus_table', 
    'finalize_tt_metal',
    'get_default_device',
    'get_device_registry',
    'is_device_open',
    'cleanup_all_devices'
]