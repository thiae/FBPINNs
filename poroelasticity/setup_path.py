#!/usr/bin/env python3
"""
Quick setup script for poroelasticity directory

This script sets up the Python path to make FBPINNs available when running
from the poroelasticity directory.

Usage:
    %cd poroelasticity
    !python setup_path.py
    !python run_all_validations.py 
"""

import sys
import os
from pathlib import Path

def setup_fbpinns_path():
    """Add FBPINNs to Python path from poroelasticity directory."""
    
    # Get the FBPINNs root directory (parent of current directory)
    current_dir = Path.cwd()
    fbpinns_root = current_dir.parent
    
    print(f"Current directory: {current_dir}")
    print(f"FBPINNs root: {fbpinns_root}")
    
    # Check if fbpinns directory exists
    if not (fbpinns_root / "fbpinns").exists():
        print(f"ERROR: fbpinns directory not found at {fbpinns_root}")
        print("   Make sure you're in the poroelasticity subdirectory of FBPINNs")
        return False
    
    # Add to Python path
    if str(fbpinns_root) not in sys.path:
        sys.path.insert(0, str(fbpinns_root))
        print(f"SUCCESS: Added {fbpinns_root} to Python path")
    else:
        print(f"INFO: {fbpinns_root} already in Python path")
    
    # Also add current directory for local imports
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        print(f"SUCCESS: Added {current_dir} to Python path")
    
    return True

def test_imports():
    """Test that FBPINNs and local modules can be imported."""
    print("\nTesting imports...")
    
    try:
        import fbpinns
        print("  SUCCESS: FBPINNs imported")
    except ImportError as e:
        print(f"  ERROR: FBPINNs import failed: {e}")
        return False
    
    try:
        from trainers.base_model import BiotCoupledTrainer
        print("  SUCCESS: Physics trainer imported")
    except ImportError as e:
        print(f"  ERROR: Physics trainer import failed: {e}")
        return False
    
    try:
        from trainers.biot_trainer_2d_data import BiotCoupledDataTrainer
        print("  SUCCESS: Data trainer imported")
    except ImportError as e:
        print(f"  WARNING: Data trainer import failed: {e} (optional)")
    
    return True

def main():
    print("Setting up Python path for poroelasticity...")
    print("=" * 50)
    
    if setup_fbpinns_path():
        if test_imports():
            print("\nSUCCESS: All required modules are now available!")
            print("\nYou can now run:")
            print("  !python simple_validation.py")
        else:
            print("\nERROR: Some imports failed. Check the errors above.")
    else:
        print("\nERROR: Failed to setup Python path.")

if __name__ == "__main__":
    main()
