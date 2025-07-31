#!/usr/bin/env python3
"""
Test runner with automatic path setup
=====================================

Runs test files with FBPINNs properly accessible.

Usage:
    python run_test.py test/test_biot_2d.py
    python run_test.py validation_scripts/validate_2d_physics.py --quick
"""

import sys
import os
from pathlib import Path
import subprocess

def setup_python_path():
    """Setup Python path to include FBPINNs."""
    current_dir = Path.cwd()
    fbpinns_root = current_dir.parent
    
    if str(fbpinns_root) not in sys.path:
        sys.path.insert(0, str(fbpinns_root))
    
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_test.py <test_file> [args...]")
        sys.exit(1)
    
    # Setup path
    setup_python_path()
    
    # Get test file and arguments
    test_file = sys.argv[1]
    test_args = sys.argv[2:] if len(sys.argv) > 2 else []
    
    print(f"Running: {test_file} {' '.join(test_args)}")
    print("=" * 50)
    
    # Import and run the test directly in the same Python process
    # to preserve the path setup
    try:
        if test_file == "test/test_biot_2d.py":
            # Run the test by importing and executing
            import test.test_biot_2d
            print("Test completed successfully!")
        else:
            print(f"Running {test_file} as subprocess...")
            result = subprocess.run([sys.executable, test_file] + test_args, 
                                  env={**os.environ, 'PYTHONPATH': ':'.join(sys.path)})
            if result.returncode == 0:
                print("Test completed successfully!")
            else:
                print(f"Test failed with exit code {result.returncode}")
                sys.exit(result.returncode)
            
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
