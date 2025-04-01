import pytest
import sys
from functools import wraps

def pytest_configure(config):
    """This hook is called before test collection, allowing us to patch the registry system"""
    # Import the registry module
    from xinnovation.src.core.registry import Registry, LIGHTNING
    
    # Save the original method
    original_register_module = Registry.register_module
    
    # Create a patched version that uses force=True only for real registries (not test ones)
    @wraps(original_register_module)
    def patched_register_module(self, name=None, module=None, force=None, group=None):
        # If force is explicitly set, use that value
        if force is not None:
            return original_register_module(self, name, module, force=force, group=group)
        
        # For test registries, don't force (preserve test behavior)
        if self._name.startswith("test") or self._name.startswith("force_test"):
            return original_register_module(self, name, module, force=False, group=group)
        
        # For real registries, always force
        return original_register_module(self, name, module, force=True, group=group)
    
    # Apply the patch
    Registry.register_module = patched_register_module
    
    print("Registry system patched to use force=True for non-test registries") 