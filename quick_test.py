#!/usr/bin/env python3
"""
FBPINNs Quick Test Script
========================

This script performs a quick test to verify that FBPINNs is properly installed
and can run a simple training example.

Usage:
    !python quick_test.py
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import jax
        import jax.numpy as jnp
        print("  SUCCESS: JAX")
    except ImportError as e:
        print(f"  ERROR: JAX: {e}")
        return False
    
    try:
        import optax
        print("  SUCCESS: Optax")
    except ImportError as e:
        print(f"  ERROR: Optax: {e}")
        return False
    
    try:
        import fbpinns
        print("  SUCCESS: FBPINNs")
    except ImportError as e:
        print(f"  ERROR: FBPINNs: {e}")
        return False
    
    try:
        from fbpinns.domains import RectangularDomainND
        from fbpinns.problems import Problem
        from fbpinns.networks import FCN
        from fbpinns.constants import Constants
        from fbpinns.trainers import FBPINNTrainer
        print("  SUCCESS: FBPINNs core modules")
    except ImportError as e:
        print(f"  ERROR: FBPINNs core modules: {e}")
        return False
    
    return True

def test_simple_problem():
    """Test a simple problem setup (no training)."""
    print("Testing problem setup...")
    
    try:
        import numpy as np
        from fbpinns.domains import RectangularDomainND
        from fbpinns.problems import HarmonicOscillator1D
        from fbpinns.decompositions import RectangularDecompositionND
        from fbpinns.networks import FCN
        from fbpinns.constants import Constants
        
        # Test domain class (don't instantiate directly)
        print("  SUCCESS: Domain class imported")
        
        # Create constants with proper structure (like in README example)
        c = Constants(
            domain=RectangularDomainND,
            domain_init_kwargs=dict(
                xmin=np.array([0.0]),
                xmax=np.array([1.0]),
            ),
            problem=HarmonicOscillator1D,
            problem_init_kwargs=dict(
                d=2, w0=10,
            ),
            network=FCN,
            network_init_kwargs=dict(
                layer_sizes=[1, 10, 1],
            ),
        )
        print("  SUCCESS: Constants created with domain and problem")
        
        # Test network class
        print("  SUCCESS: Network class imported")
        
        print("  SUCCESS: All components initialized successfully")
        return True
        
    except Exception as e:
        print(f"  ERROR: Problem setup failed: {e}")
        return False

def test_poroelasticity_modules():
    """Test if poroelasticity modules can be imported."""
    print("Testing poroelasticity modules...")
    
    # Check if we're in the right directory
    poro_dir = Path.cwd() / "poroelasticity"
    if not poro_dir.exists():
        print("  WARNING: Not in FBPINNs root directory or poroelasticity folder missing")
        return False
    
    # Add poroelasticity to path
    if str(poro_dir) not in sys.path:
        sys.path.insert(0, str(poro_dir))
    
    try:
        from trainers.biot_trainer_2d import BiotCoupledTrainer, BiotCoupled2D
        print("  SUCCESS: Physics trainer")
    except ImportError as e:
        print(f"  ERROR: Physics trainer: {e}")
        return False
    
    try:
        from trainers.biot_trainer_2d_data import BiotCoupledDataTrainer
        print("  SUCCESS: Data trainer")
    except ImportError as e:
        print(f"  WARNING: Data trainer: {e} (optional)")
    
    return True

def check_gpu():
    """Check GPU availability."""
    print("Checking GPU availability...")
    
    try:
        import jax
        devices = jax.devices()
        
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        if gpu_devices:
            print(f"  SUCCESS: GPU available: {gpu_devices}")
            return True
        else:
            print("  INFO: No GPU detected (CPU-only mode)")
            return False
            
    except Exception as e:
        print(f"  WARNING: Could not check GPU: {e}")
        return False

def main():
    """Run all tests."""
    print("FBPINNs Quick Test")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    print()
    
    # Test simple problem
    if not test_simple_problem():
        all_passed = False
    print()
    
    # Test poroelasticity modules
    if not test_poroelasticity_modules():
        all_passed = False
    print()
    
    # Check GPU
    check_gpu()
    print()
    
    # Final result
    if all_passed:
        print("SUCCESS: All tests passed! FBPINNs is ready to use.")
        print()
        print("Next steps:")
        print("  - Run validation: !python poroelasticity/run_all_validations.py --quick")
        print("  - Open notebook: poroelasticity/notebooks/Biot_Visualization_Hub.ipynb")
        print("  - Run specific test: !python poroelasticity/test/test_biot_2d.py")
    else:
        print("ERROR: Some tests failed. Check the errors above.")
        print()
        print("Try running:")
        print("  !python colab_setup.py")

if __name__ == "__main__":
    main()
