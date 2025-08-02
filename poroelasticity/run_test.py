#!/usr/bin/env python3
"""
Quick test script to run diagnostics and improved training for Biot poroelasticity
This script will help identify why the model isn't learning and provide solutions.
"""

import sys
import os
from pathlib import Path

# Add paths for importing modules
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    import jax.numpy as jnp
    import jax
    print("✅ JAX libraries loaded")
except ImportError as e:
    print(f"❌ JAX not available: {e}")
    import numpy as np
    print("✅ NumPy fallback loaded")

try:
    from trainers.biot_trainer_2d import BiotCoupledTrainer
    print("✅ BiotCoupledTrainer loaded")
except ImportError as e:
    print(f"❌ Failed to import BiotCoupledTrainer: {e}")
    sys.exit(1)

def diagnose_trainer(trainer):
    """Run comprehensive diagnostics on a trainer"""
    print("\n🔍 RUNNING DIAGNOSTICS...")
    print("="*50)
    
    if trainer is None:
        print("❌ No trainer provided")
        return
    
    # 1. Check trainer structure
    print(f"Trainer type: {type(trainer).__name__}")
    print(f"Has underlying trainer: {hasattr(trainer, 'trainer')}")
    
    if hasattr(trainer, 'trainer'):
        base_trainer = trainer.trainer
        print(f"Underlying trainer: {type(base_trainer).__name__}")
        
        # 2. Test predictions
        print("\n🧪 Testing predictions...")
        test_points = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]])
        
        try:
            pred = trainer.predict(jnp.array(test_points))
            if hasattr(pred, 'numpy'):
                pred = pred.numpy()
            else:
                pred = np.array(pred)
            
            print("Sample predictions:")
            for i, (point, prediction) in enumerate(zip(test_points, pred)):
                print(f"  Point {point}: ux={prediction[0]:.6f}, uy={prediction[1]:.6f}, p={prediction[2]:.6f}")
            
            # Check for problematic predictions
            if np.allclose(pred, 0.0, atol=1e-10):
                print("⚠️ WARNING: All predictions are essentially zero!")
                return "zero_predictions"
            elif np.std(pred) < 1e-8:
                print("⚠️ WARNING: Very low prediction variance!")
                return "low_variance"
            else:
                print("✅ Predictions have reasonable variation")
                
        except Exception as e:
            print(f"❌ Error in prediction test: {e}")
            return "prediction_error"
    
    # 3. Check loss
    try:
        if hasattr(trainer, 'trainer') and hasattr(trainer.trainer, 'test_loss'):
            loss = trainer.trainer.test_loss()
            print(f"\nCurrent loss: {loss:.6e}")
            
            if loss > 1e-1:
                print("⚠️ WARNING: Very high loss - model not learning well")
                return "high_loss"
            else:
                print("✅ Loss seems reasonable")
                return "ok"
                
    except Exception as e:
        print(f"Cannot get loss: {e}")
    
    return "unknown"

def test_original_approach():
    """Test the original approach that's failing"""
    print("\n🧪 TESTING ORIGINAL APPROACH")
    print("="*40)
    
    try:
        trainer = BiotCoupledTrainer(
            w_mech=1.0,
            w_flow=1.0, 
            w_bc=1.0,
            auto_balance=True
        )
        print("✅ Original trainer created")
        
        trainer.train_gradual_coupling(n_steps_pre=25, n_steps_coupled=50)
        print("✅ Original training completed")
        
        diagnosis = diagnose_trainer(trainer)
        return trainer, diagnosis
        
    except Exception as e:
        print(f"❌ Original approach failed: {e}")
        return None, "failed"

def test_improved_approach():
    """Test improved approach with higher boundary weights"""
    print("\n🚀 TESTING IMPROVED APPROACH")
    print("="*40)
    
    try:
        trainer = BiotCoupledTrainer(
            w_mech=1.0,
            w_flow=1.0, 
            w_bc=50.0,  # Much higher boundary weight
            auto_balance=False  # Disable auto-balance to force high BC weight
        )
        print("✅ Improved trainer created (w_bc=50.0)")
        
        # More training steps
        trainer.train_gradual_coupling(n_steps_pre=100, n_steps_coupled=200)
        print("✅ Improved training completed (300 total steps)")
        
        diagnosis = diagnose_trainer(trainer)
        return trainer, diagnosis
        
    except Exception as e:
        print(f"❌ Improved approach failed: {e}")
        return None, "failed"

def test_extreme_approach():
    """Test extreme approach for stubborn models"""
    print("\n💥 TESTING EXTREME APPROACH")
    print("="*40)
    
    try:
        trainer = BiotCoupledTrainer(
            w_mech=0.1,   # Lower mechanics weight
            w_flow=0.1,   # Lower flow weight
            w_bc=100.0,   # Extremely high boundary weight
            auto_balance=False
        )
        print("✅ Extreme trainer created (w_bc=100.0)")
        
        # Even more training
        trainer.train_gradual_coupling(n_steps_pre=150, n_steps_coupled=350)
        print("✅ Extreme training completed (500 total steps)")
        
        diagnosis = diagnose_trainer(trainer)
        return trainer, diagnosis
        
    except Exception as e:
        print(f"❌ Extreme approach failed: {e}")
        return None, "failed"

def main():
    """Run comprehensive testing to fix the learning issue"""
    print("🎯 BIOT POROELASTICITY LEARNING DIAGNOSTICS")
    print("="*60)
    print("This script will test different approaches to fix your learning issue\n")
    
    results = {}
    
    # Test 1: Original approach (what's currently failing)
    original_trainer, original_diagnosis = test_original_approach()
    results['original'] = (original_trainer, original_diagnosis)
    
    # Test 2: Improved approach
    improved_trainer, improved_diagnosis = test_improved_approach()
    results['improved'] = (improved_trainer, improved_diagnosis)
    
    # Test 3: Extreme approach (if improved still fails)
    if improved_diagnosis in ['zero_predictions', 'low_variance', 'high_loss']:
        extreme_trainer, extreme_diagnosis = test_extreme_approach()
        results['extreme'] = (extreme_trainer, extreme_diagnosis)
    
    # Summary
    print("\n" + "="*60)
    print("📊 RESULTS SUMMARY")
    print("="*60)
    
    best_trainer = None
    best_approach = None
    
    for approach, (trainer, diagnosis) in results.items():
        print(f"{approach.upper()}: {diagnosis}")
        if diagnosis == "ok" and best_trainer is None:
            best_trainer = trainer
            best_approach = approach
    
    if best_trainer is not None:
        print(f"\n🎉 SUCCESS: {best_approach.upper()} approach worked!")
        print("\nTo use this trainer in your notebook:")
        if best_approach == 'improved':
            print("trainer = BiotCoupledTrainer(w_mech=1.0, w_flow=1.0, w_bc=50.0, auto_balance=False)")
            print("trainer.train_gradual_coupling(n_steps_pre=100, n_steps_coupled=200)")
        elif best_approach == 'extreme':
            print("trainer = BiotCoupledTrainer(w_mech=0.1, w_flow=0.1, w_bc=100.0, auto_balance=False)")
            print("trainer.train_gradual_coupling(n_steps_pre=150, n_steps_coupled=350)")
        
        # Quick visualization test
        print("\n🎨 Testing visualization with successful trainer...")
        try:
            test_points = np.linspace(0, 1, 10)
            X, Y = np.meshgrid(test_points, test_points)
            points = np.column_stack([X.flatten(), Y.flatten()])
            
            pred = best_trainer.predict(jnp.array(points))
            if hasattr(pred, 'numpy'):
                pred = pred.numpy()
            else:
                pred = np.array(pred)
            
            ux = pred[:, 0].reshape(X.shape)
            uy = pred[:, 1].reshape(X.shape)
            p = pred[:, 2].reshape(X.shape)
            
            print(f"✅ Visualization test passed!")
            print(f"   ux range: [{ux.min():.4f}, {ux.max():.4f}]")
            print(f"   uy range: [{uy.min():.4f}, {uy.max():.4f}]")
            print(f"   p range: [{p.min():.4f}, {p.max():.4f}]")
            
            # Check if it's actually learning something meaningful
            if not (np.allclose(ux, 0, atol=1e-10) and np.allclose(uy, 0, atol=1e-10)):
                print("🎉 SUCCESS: Model is actually learning displacement fields!")
            else:
                print("⚠️ Model still producing mostly zero displacements")
                
        except Exception as e:
            print(f"❌ Visualization test failed: {e}")
        
    else:
        print("\n❌ NONE OF THE APPROACHES WORKED")
        print("\nPossible issues:")
        print("1. Problem setup is incorrect")
        print("2. Boundary conditions are not well-defined")
        print("3. Physics implementation has bugs")
        print("4. Need even more training steps")
        print("5. Network architecture needs adjustment")
        
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
