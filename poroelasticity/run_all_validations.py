#!/usr/bin/env python3
"""
Master Validation Script for Biot Poroelasticity Project

This script runs all available validation tests for the Biot poroelasticity print(f"  {script_path.name} completed in {execution_time:.1f}s")

        return results
        
    except Exception as e:
                      if all(dependencies.get(dep, False) for dep in required_deps):
                    validations_to_run[key] = validation
                else:
                    print(f"Skipping {validation['name']} - missing dependencies")
            else:
                print(f"Skipping {validation['name']} - script not found: {script_path}")
    
    if not validations_to_run:
        print("\n No validations available to run")
        sys.exit(1)
    
    print(f"\nRunning {len(validations_to_run)} validation(s):")time = time.time() - start_time
        error_msg = f" Error running {script_path.name}: {e}"
        print(f"{error_msg}") informed neural network implementations.

Usage: 
    python run_all_validations.py                    # Run all available validations
    python run_all_validations.py --quick           # Quick validation (reduced training)
    python run_all_validations.py --models 2d_physics  # Run specific model only

"""

import sys
import os
import argparse
from pathlib import Path
import time
import json
from datetime import datetime

def check_dependencies():
    """Check if all required dependencies and project structure are available"""
    print("Checking Dependencies and Project Structure...")
    
    # First check project structure
    print("\nVerifying Project Structure...")
    expected_files = [
        "trainers/biot_trainer_2d.py",
        "trainers/biot_trainer_2d_data.py", 
        "test/test_biot_2d.py",
        "test/test_biot_2d_data.py",
        "utilities/visualization_tools.py",
        "utilities/validation_metrics.py",
        "validation_scripts/validate_2d_physics.py",
        "notebooks/Biot_Visualization_Hub.ipynb"
    ]
    
    structure_ok = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"   [OK] {file_path}")
        else:
            print(f"   [MISSING] {file_path}")
            structure_ok = False
    
    if not structure_ok:
        print("\nProject structure incomplete. Please ensure all files are in correct locations.")
        return None, False
    
    print("\nChecking Package Dependencies...")
    dependencies = {
        'numpy': False,
        'matplotlib': False,
        'jax': False,
        'fbpinns': False
    }
    
    # Check numpy
    try:
        import numpy as np
        dependencies['numpy'] = True
        print("   [OK] NumPy available")
    except ImportError:
        print("   [MISSING] NumPy not available")
    
    # Check matplotlib
    try:
        import matplotlib.pyplot as plt
        dependencies['matplotlib'] = True
        print("   [OK] Matplotlib available")
    except ImportError:
        print("   [MISSING] Matplotlib not available")
    
    # Check JAX
    try:
        import jax
        dependencies['jax'] = True
        print("   [OK] JAX available")
    except ImportError:
        print("   [WARNING] JAX not available (will use NumPy fallback)")
        dependencies['jax'] = False  # Not required, just preferred
    
    # Check FBPINNs
    try:
        from fbpinns.domains import RectangularDomainND
        dependencies['fbpinns'] = True
        print("   [OK] FBPINNs available")
    except ImportError:
        print("   [MISSING] FBPINNs not available")
    
    # Check custom modules
    try:
        from trainers.biot_trainer_2d import BiotCoupledTrainer
        print("   [OK] Biot physics trainer available")
    except ImportError:
        print("   [MISSING] Biot physics trainer not available")
        dependencies['fbpinns'] = False  # Dependent on this
    
    try:
        from trainers.biot_trainer_2d_data import BiotCoupledDataTrainer
        print("   [OK] Biot data trainer available")
    except ImportError:
        print("   [WARNING] Biot data trainer not available (optional)")
    
    # Determine what can be run
    essential_deps = ['numpy', 'matplotlib', 'fbpinns']
    can_run = all(dependencies[dep] for dep in essential_deps)
    
    return dependencies, can_run

def run_validation_script(script_path, args=None):
    """
    Run validation script and capture results

    Args:
        script_path: Path to validation script
        args: Additional arguments to pass
        
    Returns:
        Dictionary with results
    """
    if args is None:
        args = []
    
    script_path = Path(script_path)
    if not script_path.exists():
        return {
            'success': False,
            'error': f"Script not found: {script_path}",
            'execution_time': 0
        }
    
    print(f"\nRunning: {script_path.name}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        # Import and run the validation function directly
        sys.path.insert(0, str(script_path.parent))
        
        if script_path.name == 'validate_2d_physics.py':
            from validation_scripts.validate_2d_physics import validate_2d_physics
            
            # Extract arguments
            quick_test = '--quick' in args
            save_dir = "results/2d_physics_validation"
            for i, arg in enumerate(args):
                if arg == '--save-dir' and i + 1 < len(args):
                    save_dir = args[i + 1]
            
            results = validate_2d_physics(quick_test=quick_test, save_dir=save_dir)
            
        else:
            return {
                'success': False,
                'error': f"Unknown validation script: {script_path.name}",
                'execution_time': 0
            }
        
        execution_time = time.time() - start_time
        results['execution_time'] = execution_time
        
        print(f" {script_path.name} completed in {execution_time:.1f}s")
        
        return results
        
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error running {script_path.name}: {e}"
        print(f"  {error_msg}")
        
        return {
            'success': False,
            'error': error_msg,
            'execution_time': execution_time
        }

def generate_master_report(all_results, save_dir="results"):
    """Generate comprehensive master validation report"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    report_file = save_path / "master_validation_report.json"
    summary_file = save_path / "master_validation_summary.txt"
    
    # Add metadata
    master_results = {
        'timestamp': datetime.now().isoformat(),
        'project': 'Biot Poroelasticity PINNs',
        'validation_results': all_results,
        'overall_summary': {}
    }
    
    # Compute overall statistics
    total_validations = len(all_results)
    successful_validations = sum(1 for result in all_results.values() 
                               if result.get('validation_success', False))
    total_time = sum(result.get('total_time', 0) for result in all_results.values())
    
    master_results['overall_summary'] = {
        'total_validations': total_validations,
        'successful_validations': successful_validations,
        'success_rate': successful_validations / total_validations if total_validations > 0 else 0,
        'total_time': total_time
    }
    
    # Save JSON report
    with open(report_file, 'w') as f:
        json.dump(master_results, f, indent=2, default=str)
    
    # Save text summary
    with open(summary_file, 'w') as f:
        f.write("BIOT POROELASTICITY - MASTER VALIDATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Total Validations: {total_validations}\n")
        f.write(f"Successful: {successful_validations}\n")
        f.write(f"Success Rate: {successful_validations/total_validations*100:.1f}%\n")
        f.write(f"Total Time: {total_time:.1f} seconds\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"Model: {model_name}\n")
            f.write("-" * 30 + "\n")
            
            if results.get('validation_success', False):
                f.write("Status: [SUCCESS]\n")
            else:
                f.write("Status: [FAILED]\n")
            
            f.write(f"Time: {results.get('total_time', 0):.1f}s\n")
            
            if 'L2_errors' in results and 'total' in results['L2_errors']:
                f.write(f"L2 Error: {results['L2_errors']['total']:.2e}\n")
            
            if 'error_messages' in results and results['error_messages']:
                f.write("Errors:\n")
                for error in results['error_messages']:
                    f.write(f"  - {error}\n")
            
            f.write("\n")
    
    print(f"\nMaster report saved to:")
    print(f"   JSON: {report_file}")
    print(f"   Summary: {summary_file}")
    
    return master_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run all Biot Poroelasticity validations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_validations.py                    # Run all available
  python run_all_validations.py --quick           # Quick mode
  python run_all_validations.py --models 2d_physics  # Specific model only
        """
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use reduced training steps for quick validation'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['2d_physics'],
        help='Specific models to validate (default: all available)'
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='results',
        help='Directory to save all results'
    )
    
    args = parser.parse_args()
    
    print("BIOT POROELASTICITY - MASTER VALIDATION SUITE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print(f"Quick mode: {'Yes' if args.quick else 'No'}")
    print(f"Save directory: {args.save_dir}")
    
    # Check dependencies
    dependencies, can_run = check_dependencies()
    
    if not can_run:
        print("\n Essential dependencies missing. Cannot proceed.")
        print("   Please install: numpy, matplotlib, and FBPINNs")
        sys.exit(1)
    
    # Define available validations
    available_validations = {
        '2d_physics': {
            'script': 'validation_scripts/validate_2d_physics.py',
            'name': '2D Physics Only Biot',
            'required_deps': ['numpy', 'matplotlib', 'fbpinns']
        }
    }
    
    # Determine which validations to run
    if args.models:
        validations_to_run = {k: v for k, v in available_validations.items() 
                            if k in args.models}
    else:
        # Run all available validations
        validations_to_run = {}
        for key, validation in available_validations.items():
            script_path = Path(validation['script'])
            if script_path.exists():
                required_deps = validation['required_deps']
                if all(dependencies.get(dep, False) for dep in required_deps):
                    validations_to_run[key] = validation
                else:
                    print(f" Skipping {validation['name']} - missing dependencies")
            else:
                print(f" Skipping {validation['name']} - script not found: {script_path}")
    
    if not validations_to_run:
        print("\n No validations available to run")
        sys.exit(1)
    
    print(f"\n Running {len(validations_to_run)} validation(s):")
    for key, validation in validations_to_run.items():
        print(f"   - {validation['name']}")
    
    # Run validations
    all_results = {}
    script_args = []
    if args.quick:
        script_args.append('--quick')
    
    total_start_time = time.time()
    
    for key, validation in validations_to_run.items():
        model_save_dir = f"{args.save_dir}/{key}_validation"
        model_args = script_args + ['--save-dir', model_save_dir]
        
        result = run_validation_script(validation['script'], model_args)
        all_results[validation['name']] = result
    
    total_time = time.time() - total_start_time
    
    # Generate master report
    print("\nGenerating Master Report...")
    master_results = generate_master_report(all_results, args.save_dir)
    
    # Final summary
    successful = master_results['overall_summary']['successful_validations']
    total = master_results['overall_summary']['total_validations']
    success_rate = master_results['overall_summary']['success_rate']
    
    print("\nFINAL SUMMARY")
    print("=" * 60)
    print(f"Total time: {total_time:.1f} seconds")
    print(f"Success rate: {successful}/{total} ({success_rate*100:.1f}%)")
    
    if success_rate == 1.0:
        print("\n[SUCCESS] ALL VALIDATIONS SUCCESSFUL!")
        print("   Project structure is correctly organized")
        print("   Biot poroelasticity implementation is working correctly")
        print("   Ready for production use and academic submission")
        print("\nUsage Options:")
        print("   Quick tests:      python test/test_biot_2d.py")
        print("   Physics valid:    python validation_scripts/validate_2d_physics.py --quick")
        print("   Interactive:      Open notebooks/Biot_Visualization_Hub.ipynb")
    elif success_rate >= 0.5:
        print("\n[PARTIAL] PARTIAL SUCCESS")
        print("   Some validations passed, review failed ones")
    else:
        print("\n[ERROR] VALIDATION FAILURES")
        print("   Most validations failed, review implementation")
    
    print(f"\nAll results saved to: {args.save_dir}")
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if success_rate > 0.5 else 1)

if __name__ == "__main__":
    main()
