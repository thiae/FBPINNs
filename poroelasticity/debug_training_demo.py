#!/usr/bin/env python3
"""
🔍 DEBUGGING DEMO: How to diagnose and fix PINN training issues

This script demonstrates how to use the new diagnostic tools to solve
the exact problem you're experiencing: loss decreases but model doesn't
learn the physics correctly.

Run this step-by-step to identify and fix training issues.
"""

import numpy as np
import jax.numpy as jnp
from trainers.biot_trainer_2d import BiotCoupledTrainer

def debug_training_protocol():
    """
    Step-by-step debugging protocol for PINN training issues
    """
    print("🔍 PINN TRAINING DEBUGGING PROTOCOL")
    print("=" * 60)
    
    # Create trainer with improved configuration
    trainer = BiotCoupledTrainer(w_mech=1.0, w_flow=1.0, w_bc=1.0, auto_balance=True)
    
    print("\n📋 STEP 1: Start with simplified debugging training")
    print("-" * 50)
    
    # Start with debug training to identify core issues
    trainer.train_simple_debug(n_steps=500)
    
    # Diagnose what happened
    diagnostics = trainer.diagnose_training_issues()
    
    # Based on diagnostics, choose next step
    max_violation = max(diagnostics['boundary_violations'].values()) if diagnostics['boundary_violations'] else 0
    
    if max_violation > 0.01:
        print("\n🚨 BOUNDARY CONDITIONS NOT SATISFIED")
        print("   Recommended: Extreme boundary enforcement")
        trainer.train_extreme_bc_enforcement(n_steps=800)
        
        # Re-diagnose
        print("\n📋 Re-diagnosing after extreme BC enforcement...")
        diagnostics = trainer.diagnose_training_issues()
        
    elif diagnostics['field_statistics']['has_nan'] or diagnostics['field_statistics']['has_inf']:
        print("\n🚨 NUMERICAL INSTABILITY DETECTED")
        print("   Recommended: Physics-first training with lower learning rate")
        # Note: You'd need to modify the trainer to support learning rate changes
        trainer.train_physics_first(n_steps=1000)
        
    else:
        print("\n✅ BOUNDARY CONDITIONS OK")
        print("   Proceeding with physics-first training...")
        trainer.train_physics_first(n_steps=1700)
    
    print("\n📋 STEP 2: Final diagnosis and visualization")
    print("-" * 50)
    
    # Final comprehensive diagnosis
    final_diagnostics = trainer.diagnose_training_issues()
    
    # Test on a grid for visualization
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)
    test_points = np.column_stack([X.flatten(), Y.flatten()])
    
    # Get predictions
    predictions = trainer.predict(test_points)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Displacement X range: [{predictions[:, 0].min():.6f}, {predictions[:, 0].max():.6f}]")
    print(f"   Displacement Y range: [{predictions[:, 1].min():.6f}, {predictions[:, 1].max():.6f}]")
    print(f"   Pressure range: [{predictions[:, 2].min():.6f}, {predictions[:, 2].max():.6f}]")
    
    # Check if pressure is in reasonable bounds
    p_min, p_max = predictions[:, 2].min(), predictions[:, 2].max()
    if p_min >= -0.05 and p_max <= 1.05:
        print("   ✅ Pressure field in reasonable bounds [0, 1]")
    else:
        print(f"   ⚠️ Pressure field outside expected bounds [0, 1]")
    
    # Save the model if training looks successful
    if max(final_diagnostics['boundary_violations'].values() if final_diagnostics['boundary_violations'] else [0]) < 0.01:
        trainer.save_model("biot_model_debugged.jax")
        print("   ✅ Model saved as 'biot_model_debugged.jax'")
    
    return trainer, final_diagnostics

def compare_training_methods():
    """
    Compare different training approaches to see which works best
    """
    print("\n🔬 TRAINING METHOD COMPARISON")
    print("=" * 60)
    
    methods = [
        ("Simple Debug", lambda t: t.train_simple_debug(500)),
        ("Physics First", lambda t: t.train_physics_first(800)),
        ("Extreme BC", lambda t: t.train_extreme_bc_enforcement(800)),
        ("Standard Auto", lambda t: t.train_coupled(800))
    ]
    
    results = {}
    
    for method_name, train_func in methods:
        print(f"\n🧪 Testing: {method_name}")
        print("-" * 30)
        
        trainer = BiotCoupledTrainer(auto_balance=True)
        
        try:
            train_func(trainer)
            diagnostics = trainer.diagnose_training_issues(print_details=False)
            
            max_violation = max(diagnostics['boundary_violations'].values()) if diagnostics['boundary_violations'] else 0
            p_range = diagnostics['field_statistics']['p_range']
            p_in_bounds = p_range[0] >= -0.1 and p_range[1] <= 1.1
            
            results[method_name] = {
                'max_bc_violation': max_violation,
                'pressure_in_bounds': p_in_bounds,
                'success': max_violation < 0.01 and p_in_bounds
            }
            
            print(f"   Max BC violation: {max_violation:.6f}")
            print(f"   Pressure in bounds: {p_in_bounds}")
            print(f"   Overall success: {results[method_name]['success']}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[method_name] = {'success': False, 'error': str(e)}
    
    print(f"\n📊 COMPARISON SUMMARY:")
    print("-" * 30)
    for method, result in results.items():
        status = "✅" if result.get('success', False) else "❌"
        print(f"   {method}: {status}")
    
    return results

def create_test_points():
    """Create test points for consistent evaluation"""
    # Boundary points
    left = np.array([[0.0, y] for y in np.linspace(0, 1, 10)])      # x=0
    right = np.array([[1.0, y] for y in np.linspace(0, 1, 10)])     # x=1  
    bottom = np.array([[x, 0.0] for x in np.linspace(0, 1, 10)])    # y=0
    top = np.array([[x, 1.0] for x in np.linspace(0, 1, 10)])       # y=1
    
    # Interior points
    x_int = np.linspace(0.1, 0.9, 8)
    y_int = np.linspace(0.1, 0.9, 8)
    X_int, Y_int = np.meshgrid(x_int, y_int)
    interior = np.column_stack([X_int.flatten(), Y_int.flatten()])
    
    return np.vstack([left, right, bottom, top, interior])

if __name__ == "__main__":
    print("🚀 Starting PINN debugging protocol...")
    print("This will help identify why your loss decreases but physics isn't learned.")
    
    # Run the main debugging protocol
    trainer, diagnostics = debug_training_protocol()
    
    print(f"\n🎯 KEY RECOMMENDATIONS:")
    for rec in diagnostics['recommendations']:
        print(f"   {rec}")
    
    # Optionally run comparison
    run_comparison = input("\n❓ Run training method comparison? (y/n): ").lower() == 'y'
    if run_comparison:
        compare_training_methods()
    
    print(f"\n✅ Debugging complete! Check the diagnostics above for next steps.")