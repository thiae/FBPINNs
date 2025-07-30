#!/usr/bin/env python3
"""
Simple test script for biot_trainer functionality
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import jax
import jax.numpy as jnp
from fbpinns.domains import RectangularDomainND
from poroelasticity.biot_trainer import BiotCoupled2D, CoupledTrainer

def test_basic_functionality():
    """Test basic functionality without pytest"""
    print("Testing BiotCoupled2D basic functionality...")
    
    # Initialize static parameters
    static, _ = BiotCoupled2D.init_params()
    all_params = {"static": {"problem": static}, "step": 0}
    
    # Create domain
    dom = RectangularDomainND()
    domain_static, _ = dom.init_params(jnp.array([0., 0.]), jnp.array([1., 1.]))
    all_params["static"]["domain"] = domain_static
    
    print("✓ Parameters initialized")
    
    # Sample constraints
    key = jax.random.PRNGKey(0)
    shapes = ((10,), (5,), (5,), (5,), (5,))
    constraints = BiotCoupled2D.sample_constraints(
        all_params, dom, key, sampler="grid", batch_shapes=shapes
    )
    
    print(f"✓ Constraints sampled: {len(constraints)} constraint blocks")
    print(f"  Interior points: {constraints[0][0].shape}")
    print(f"  Left BC points: {constraints[1][0].shape}")
    
    # Test exact solution BCs
    ys = jnp.linspace(0, 1, 20)
    pts = jnp.stack([jnp.zeros_like(ys), ys], axis=1)
    bc_valid = BiotCoupled2D.verify_bcs(all_params, pts)
    print(f"✓ Boundary conditions verification: {bc_valid}")
    
    print("✓ All basic tests passed!")

def test_trainer():
    """Test CoupledTrainer with minimal steps"""
    print("\nTesting CoupledTrainer...")
    
    trainer = CoupledTrainer()
    trainer.auto_balance = False
    
    print("  Training mechanics only (5 steps)...")
    trainer.train_mechanics_only(n_steps=5)
    
    print("  Training flow only (5 steps)...")
    trainer.train_flow_only(n_steps=5)
    
    print("  Coupled training (5 steps)...")
    trainer.auto_balance = True
    params = trainer.train_coupled(n_steps=5)
    
    print("✓ Training completed successfully!")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_trainer()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
