"""
Simple test for the data enhanced Biot trainer
"""

import numpy as np
import jax
import jax.numpy as jnp
import sys
import os

# Add the parent directory to Python path to find fbpinns
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from trainers.biot_trainer_2d_data import VTKDataLoader, BiotCoupledDataTrainer, DataEnhancedTrainer


def test_vtk_loader():
    """Test VTK data loading"""
    print("Testing VTK Data Loader")
    try:
        loader = VTKDataLoader("Data_2D")
        data = loader.load_experimental_data()
        
        if data:
            print(f" Loaded data types: {list(data.keys())}")
            for data_type in data:
                print(f"  {data_type}: {list(data[data_type].keys())}")
        else:
            print(" No VTK files found (this is OK for testing)")
            
        return True
    except Exception as e:
        print(f" VTK loader error: {e}")
        return False


def test_data_trainer():
    """Test the data enhanced trainer creation"""
    print("\nTesting Data Enhanced Trainer ")
    try:
        # Create trainer (will work even without data files)
        trainer = DataEnhancedTrainer(
            data_dir="Data_2D",
            data_weight=0.5
        )
        
        print("  Created data-enhanced trainer")
        print(f"  Data weight: {trainer.data_weight}")
        print(f"  Batch size: {trainer.data_batch_size}")
        print(f"  Auto balance: {trainer.auto_balance}")
        
        # Test basic training methods exist
        assert hasattr(trainer, 'train_with_data'), "Missing train_with_data method"
        assert hasattr(trainer, '_sample_data_points'), "Missing sampling method"
        assert hasattr(trainer, '_compute_data_loss'), "Missing data loss method"
        
        print(" All required methods present")
        return True
        
    except Exception as e:
        print(f" Trainer creation error: {e}")
        return False


def test_fallback_training():
    """Test that trainer falls back to physics only when no data available"""
    print("\nTesting Fallback Training...")
    try:
        trainer = DataEnhancedTrainer(
            data_dir="NonExistent_Dir"  # Intentional wrong directory
        )
        
        # Should not crash even with no data
        print(" Trainer created without data files")
        
        # Test data sampling with no data
        key = jax.random.PRNGKey(42)
        sampled = trainer._sample_data_points(key, 10)
        assert sampled is None, "Should return None when no data available"
        
        # Test data loss with no data  
        data_loss = trainer._compute_data_loss({}, None)
        assert data_loss == 0.0, "Should return 0.0 when no data available"
        
        print(" Proper fallback behavior when no data available")
        return True
        
    except Exception as e:
        print(f" Fallback test error: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("="*50)
    print("Testing Data Enhanced Biot Trainer")
    print("="*50)
    
    tests = [
        test_vtk_loader,
        test_data_trainer, 
        test_fallback_training
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*50)
    print("Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print(" All tests passed!")
        print("\nThe data enhanced trainer is ready to use!")
        print("Usage example:")
        print("  from trainers.biot_trainer_2d_data import Data EnhancedTrainer")
        print("  trainer = DataEnhancedTrainer(data_dir='Data_2D', data_weight=0.5)")
        print("  params = trainer.train_with_data(n_steps=1000)")
    else:
        print(" Some tests failed")

    return all(results)


if __name__ == "__main__":
    run_all_tests()
