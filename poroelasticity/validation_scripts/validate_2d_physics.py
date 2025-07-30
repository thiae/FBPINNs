#!/usr/bin/env python3
"""
2D Biot Poroelasticity Physics-Only Validation Script

This script validates the physics-only implementation of 2D Biot poroelasticity
using FBPINNs. It trains the model and performs comprehensive validation including:
- Solution field visualization
- Error analysis vs exact solution
- Boundary condition verification  
- Physics conservation checks

Usage: python validate_2d_physics.py [--quick] [--save-dir results/2d_physics]
"""

import sys
import os
import argparse
from pathlib import Path
import time

# Add current directory to path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

# Add FBPINNs root directory to path if needed
fbpinns_root = current_dir.parent
if str(fbpinns_root) not in sys.path:
    sys.path.insert(0, str(fbpinns_root))

# Test FBPINNs import
try:
    from fbpinns.domains import RectangularDomainND
    print("✅ FBPINNs framework available")
except ImportError as e:
    print(f"❌ FBPINNs framework not available: {e}")
    print("   Make sure FBPINNs is installed: pip install -e . from FBPINNs root")
    sys.exit(1)

# Import utilities
try:
    from utilities.visualization_tools import BiotVisualizationTools
    from utilities.validation_metrics import ValidationMetrics
    print("✅ Utilities loaded successfully")
except ImportError as e:
    print(f"❌ Error importing utilities: {e}")
    print("   Make sure you're running from the poroelasticity directory")
    sys.exit(1)

# Import Biot trainer
try:
    from trainers.biot_trainer_2d import BiotCoupledTrainer, BiotCoupled2D
    print("✅ Biot trainer loaded successfully")
except ImportError as e:
    print(f"❌ Error importing Biot trainer: {e}")
    print("   Make sure trainers/biot_trainer_2d.py exists")
    sys.exit(1)

def validate_2d_physics(quick_test=False, save_dir="results/2d_physics_validation"):
    """
    Complete validation of 2D physics-only Biot trainer
    
    Args:
        quick_test: If True, use reduced training for fast validation
        save_dir: Directory to save results
        
    Returns:
        Dictionary with validation results and success status
    """
    print("🏗️ 2D BIOT POROELASTICITY PHYSICS-ONLY VALIDATION")
    print("=" * 60)
    
    start_time = time.time()
    
    # Create results directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualization tools
    viz = BiotVisualizationTools()
    metrics_calculator = ValidationMetrics()
    
    results = {
        'model_type': '2D Physics-Only',
        'validation_success': False,
        'error_messages': []
    }
    
    try:
        # Step 1: Create and train the physics-only trainer
        print("\n🚂 Step 1: Training Physics-Only Model")
        print("-" * 40)
        
        trainer = BiotCoupledTrainer(auto_balance=True)
        
        if quick_test:
            print("⚡ Quick test mode - reduced training steps")
            n_steps_pre = 50
            n_steps_coupled = 200
        else:
            print("🎯 Full training mode")
            n_steps_pre = 100
            n_steps_coupled = 1000
        
        print(f"   Pre-training steps: {n_steps_pre}")
        print(f"   Coupled training steps: {n_steps_coupled}")
        
        # Train with timing
        train_start = time.time()
        all_params = trainer.train_gradual_coupling(
            n_steps_pre=n_steps_pre, 
            n_steps_coupled=n_steps_coupled
        )
        train_time = time.time() - train_start
        
        print(f"✅ Training completed in {train_time:.1f} seconds")
        results['training_time'] = train_time
        
        # Step 2: Error analysis vs exact solution
        print("\n📊 Step 2: Error Analysis vs Exact Solution")
        print("-" * 40)
        
        # Create test points
        nx, ny = 50, 50
        X, Y, test_points = viz.create_mesh_grid(nx, ny)
        
        # Get predictions and exact solution
        try:
            import jax.numpy as jnp
            test_points_jax = jnp.array(test_points)
        except ImportError:
            test_points_jax = test_points
        
        pred = trainer.predict(test_points_jax)
        exact = trainer.trainer.c.problem.exact_solution(trainer.all_params, test_points_jax)
        
        # Compute error metrics
        error_metrics = metrics_calculator.compute_error_metrics(pred, exact)
        results.update(error_metrics)
        
        # Print validation summary
        validation_status = metrics_calculator.print_validation_summary(
            error_metrics, "2D Physics-Only Biot"
        )
        results['validation_status'] = validation_status
        
        # Step 3: Boundary condition verification
        print("\n🔍 Step 3: Boundary Condition Verification")
        print("-" * 40)
        
        bc_results = metrics_calculator.verify_boundary_conditions(trainer)
        results['boundary_conditions'] = bc_results
        
        if 'overall' in bc_results:
            print(f"   {bc_results['overall']['summary']}")
            if bc_results['overall']['all_boundaries_satisfied']:
                print("   ✅ All boundary conditions satisfied")
            else:
                print("   ⚠️ Some boundary conditions may be violated")
                for boundary, data in bc_results.items():
                    if boundary != 'overall' and isinstance(data, dict):
                        for condition, satisfied in data.items():
                            if condition.endswith('_satisfied') and not satisfied:
                                print(f"      ❌ {boundary} {condition}")
        
        # Step 4: Physics conservation checks
        print("\n⚖️ Step 4: Physics Conservation Checks")
        print("-" * 40)
        
        physics_results = metrics_calculator.physics_conservation_check(trainer)
        results['physics_conservation'] = physics_results
        
        if 'overall' in physics_results:
            print(f"   {physics_results['overall']['summary']}")
            if physics_results['overall']['physics_reasonable']:
                print("   ✅ Physics values are reasonable")
            else:
                print("   ⚠️ Physics values may need review")
        
        # Step 5: Generate visualizations
        print("\n🎨 Step 5: Generating Visualizations")
        print("-" * 40)
        
        # Solution fields
        solution_path = save_path / "solution_fields.png"
        viz.plot_solution_fields(
            trainer, 
            save_path=solution_path,
            title_prefix="2D Physics-Only"
        )
        
        # Error fields
        error_path = save_path / "error_fields.png"
        viz.plot_error_fields(
            trainer,
            save_path=error_path, 
            title_prefix="2D Physics-Only"
        )
        
        print(f"   ✅ Visualizations saved to: {save_path}")
        
        # Step 6: Save comprehensive results
        print("\n💾 Step 6: Saving Results")
        print("-" * 40)
        
        metrics_file, summary_file = viz.save_training_summary(
            trainer, error_metrics, save_path, "2D_Physics_Only"
        )
        
        results['output_files'] = {
            'metrics': str(metrics_file),
            'summary': str(summary_file),
            'solution_plot': str(solution_path),
            'error_plot': str(error_path)
        }
        
        # Determine overall success
        success_criteria = [
            validation_status in ['EXCELLENT', 'GOOD'],
            bc_results.get('overall', {}).get('all_boundaries_satisfied', False),
            physics_results.get('overall', {}).get('physics_reasonable', False)
        ]
        
        results['validation_success'] = all(success_criteria)
        
        # Final assessment
        total_time = time.time() - start_time
        results['total_time'] = total_time
        
        print("\n🎯 FINAL ASSESSMENT")
        print("=" * 60)
        print(f"⏱️ Total validation time: {total_time:.1f} seconds")
        print(f"📊 Error status: {validation_status}")
        print(f"🔍 Boundary conditions: {'✅ PASS' if bc_results.get('overall', {}).get('all_boundaries_satisfied', False) else '⚠️ ISSUES'}")
        print(f"⚖️ Physics conservation: {'✅ PASS' if physics_results.get('overall', {}).get('physics_reasonable', False) else '⚠️ ISSUES'}")
        
        if results['validation_success']:
            print("\n🎉 VALIDATION SUCCESSFUL!")
            print("   ✅ 2D Physics-Only Biot implementation is working correctly")
            print("   ✅ Ready to proceed with data-enhanced training")
        else:
            print("\n⚠️ VALIDATION ISSUES DETECTED")
            print("   📈 Consider:")
            print("   - Increasing training steps")
            print("   - Adjusting network architecture")
            print("   - Checking physics implementation")
        
        print("=" * 60)
        
        return results
        
    except Exception as e:
        error_msg = f"❌ Validation failed with error: {e}"
        print(error_msg)
        results['error_messages'].append(error_msg)
        
        import traceback
        traceback.print_exc()
        
        return results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Validate 2D Biot Poroelasticity Physics-Only Implementation"
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Use reduced training steps for quick validation'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='results/2d_physics_validation',
        help='Directory to save validation results'
    )
    
    args = parser.parse_args()
    
    # Run validation
    results = validate_2d_physics(
        quick_test=args.quick,
        save_dir=args.save_dir
    )
    
    # Exit with appropriate code
    if results['validation_success']:
        print(f"\n✅ Validation completed successfully!")
        print(f"📁 Results saved to: {args.save_dir}")
        sys.exit(0)
    else:
        print(f"\n❌ Validation completed with issues")
        print(f"📁 Results saved to: {args.save_dir}")
        sys.exit(1)

if __name__ == "__main__":
    main()
