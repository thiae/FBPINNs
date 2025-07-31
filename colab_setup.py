#!/usr/bin/env python3
"""
FBPINNs Colab Setup Script
==========================

This script handles the installation of FBPINNs and its dependencies on Google Colab
in a way that minimizes restart issues and ensures all components work properly.

Usage:
    !python colab_setup.py [--method pip|path|auto]
    
Methods:
    - pip: Install FBPINNs using pip install -e . (may trigger restart)
    - path: Add FBPINNs to Python path (no pip install, restart-free)
    - auto: Try path method first, fallback to pip if needed
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path

def check_package(package_name):
    """Check if a package is available for import."""
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except ImportError:
        return False

def install_dependencies():
    """Install core dependencies needed for FBPINNs."""
    print("Installing dependencies...")
    
    # Check if we're on Colab (has GPU by default) or need CPU-only
    try:
        import google.colab
        # On Colab, use GPU version of JAX
        jax_package = "jax[cuda12]>=0.4.8"
        print("  Detected Google Colab - using GPU JAX")
    except ImportError:
        # Not on Colab, use CPU version
        jax_package = "jax[cpu]>=0.4.8"
        print("  Using CPU JAX")
    
    # Core dependencies from pyproject.toml
    dependencies = [
        jax_package,
        "optax>=0.1.4", 
        "numpy>=1.24.2",
        "scipy>=1.10.1",
        "matplotlib>=3.7.1",
        "tensorboardX>=2.6",
        "ipython>=8.12.0",
        "seaborn"  # For visualization
    ]
    
    
    for dep in dependencies:
        print(f"  Installing {dep}...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  capture_output=True, text=True, check=True)
            print(f"  SUCCESS: {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: {dep} installation failed: {e}")
            print(f"     Output: {e.stdout}")
            print(f"     Error: {e.stderr}")

def setup_fbpinns_path():
    """Add FBPINNs to Python path without pip install."""
    print("Setting up FBPINNs via Python path...")
    
    # Get the FBPINNs root directory
    fbpinns_root = Path.cwd()
    if not (fbpinns_root / "fbpinns").exists():
        print(f"ERROR: fbpinns directory not found at {fbpinns_root}")
        print("   Make sure you're running this from the FBPINNs root directory")
        return False
    
    # Add to Python path
    if str(fbpinns_root) not in sys.path:
        sys.path.insert(0, str(fbpinns_root))
        print(f"SUCCESS: Added {fbpinns_root} to Python path")
    
    return True

def setup_fbpinns_pip():
    """Install FBPINNs using pip install -e ."""
    print("Installing FBPINNs via pip...")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                              capture_output=True, text=True, check=True)
        print("SUCCESS: FBPINNs installed successfully via pip")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: FBPINNs pip installation failed: {e}")
        print(f"   Output: {e.stdout}")
        print(f"   Error: {e.stderr}")
        return False

def test_fbpinns_import():
    """Test if FBPINNs can be imported successfully."""
    print("Testing FBPINNs import...")
    
    try:
        import fbpinns
        print("SUCCESS: FBPINNs imported successfully")
        
        # Test core components
        test_imports = [
            "fbpinns.domains",
            "fbpinns.problems", 
            "fbpinns.trainers",
            "fbpinns.networks",
            "fbpinns.constants"
        ]
        
        for module in test_imports:
            try:
                importlib.import_module(module)
                print(f"  SUCCESS: {module}")
            except ImportError as e:
                print(f"  WARNING: {module}: {e}")
        
        return True
        
    except ImportError as e:
        print(f"ERROR: FBPINNs import failed: {e}")
        return False

def check_jax_gpu():
    """Check if JAX can see GPU devices."""
    try:
        import jax
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        
        gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
        if gpu_devices:
            print(f"GPU available: {gpu_devices}")
        else:
            print("No GPU detected (CPU-only mode)")
            
    except ImportError:
        print("JAX not available for device check")

def main():
    """Main setup function."""
    print("FBPINNs Colab Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("ERROR: pyproject.toml not found")
        print("   Make sure you're in the FBPINNs root directory")
        print("   Run: %cd /content/FBPINNs")
        return
    
    # Parse command line arguments
    method = "auto"
    if len(sys.argv) > 1:
        if "--method" in sys.argv:
            idx = sys.argv.index("--method")
            if idx + 1 < len(sys.argv):
                method = sys.argv[idx + 1]
    
    print(f"Setup method: {method}")
    print()
    
    # Install dependencies first
    install_dependencies()
    print()
    
    # Setup FBPINNs based on method
    success = False
    
    if method == "path":
        success = setup_fbpinns_path()
    elif method == "pip":
        success = setup_fbpinns_pip()
    elif method == "auto":
        # Try path method first (restart-free)
        success = setup_fbpinns_path()
        if not success:
            print("Path method failed, trying pip method...")
            success = setup_fbpinns_pip()
    
    print()
    
    # Test the installation
    if success:
        test_success = test_fbpinns_import()
        if test_success:
            print()
            check_jax_gpu()
            print()
            print("Setup completed successfully!")
            print()
            print("Next steps:")
            print("1. Navigate to poroelasticity folder: %cd poroelasticity")
            print("2. Run validation: !python run_all_validations.py --quick")
            print("3. Or open notebook: poroelasticity/notebooks/Biot_Visualization_Hub.ipynb")
        else:
            print("Setup completed but import test failed")
    else:
        print("Setup failed")

if __name__ == "__main__":
    main()
