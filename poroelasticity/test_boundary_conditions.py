#!/usr/bin/env python3
"""
Boundary Condition Verification for Biot Poroelasticity

This script verifies that the exact solution satisfies all boundary conditions
including the critical traction-free condition at the right boundary.
"""

import sys
import os
from pathlib import Path
import numpy as np
import jax.numpy as jnp

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Add FBPINNs root directory to path if needed
fbpinns_root = current_dir.parent
if str(fbpinns_root) not in sys.path:
    sys.path.insert(0, str(fbpinns_root))

try:
    from trainers.biot_trainer_2d import BiotCoupled2D
    print("✅ Biot trainer loaded successfully")
except ImportError as e:
    print(f"❌ Error importing Biot trainer: {e}")
    sys.exit(1)

def verify_boundary_conditions():
    """
    Verify that the exact solution satisfies all boundary conditions
    """
    print("🔍 BOUNDARY CONDITION VERIFICATION")
    print("=" * 50)
    
    # Initialize problem parameters
    static_params, trainable_params = BiotCoupled2D.init_params(
        E=5000.0, nu=0.25, alpha=0.8, k=1.0, mu=1.0
    )
    
    all_params = {
        "static": {
            "problem": static_params
        },
        "trainable": trainable_params
    }
    
    # Extract material parameters
    G = static_params["G"]
    lam = static_params["lam"]
    alpha = static_params["alpha"]
    
    print(f"Material parameters:")
    print(f"  G (shear modulus) = {G:.2f}")
    print(f"  λ (Lamé parameter) = {lam:.2f}")
    print(f"  α (Biot coefficient) = {alpha:.2f}")
    print()
    
    # Test points at boundaries
    tol = 1e-6
    
    # Left boundary (x=0)
    x_left = jnp.array([[0.0, 0.5]])
    
    # Right boundary (x=1)
    x_right = jnp.array([[1.0, 0.5]])
    
    # Bottom boundary (y=0)
    x_bottom = jnp.array([[0.5, 0.0]])
    
    # Top boundary (y=1)
    x_top = jnp.array([[0.5, 1.0]])
    
    # Get exact solutions
    sol_left = BiotCoupled2D.exact_solution(all_params, x_left)
    sol_right = BiotCoupled2D.exact_solution(all_params, x_right)
    sol_bottom = BiotCoupled2D.exact_solution(all_params, x_bottom)
    sol_top = BiotCoupled2D.exact_solution(all_params, x_top)
    
    print("📊 BOUNDARY CONDITION CHECKS:")
    print("-" * 30)
    
    # Left boundary: u_x=0, u_y=0, p=1
    ux_left, uy_left, p_left = sol_left[0, 0], sol_left[0, 1], sol_left[0, 2]
    print(f"Left boundary (x=0):")
    print(f"  u_x = {ux_left:.6f} (should be 0)")
    print(f"  u_y = {uy_left:.6f} (should be 0)")
    print(f"  p = {p_left:.6f} (should be 1)")
    left_ok = abs(ux_left) < 1e-6 and abs(uy_left) < 1e-6 and abs(p_left - 1.0) < 1e-6
    print(f"  ✓ All conditions satisfied" if left_ok else f"  ❌ Conditions violated")
    print()
    
    # Right boundary: p=0, traction-free
    ux_right, uy_right, p_right = sol_right[0, 0], sol_right[0, 1], sol_right[0, 2]
    print(f"Right boundary (x=1):")
    print(f"  p = {p_right:.6f} (should be 0)")
    print(f"  u_x = {ux_right:.6f}")
    print(f"  u_y = {uy_right:.6f}")
    right_p_ok = abs(p_right) < 1e-6
    print(f"  ✓ Pressure condition satisfied" if right_p_ok else f"  ❌ Pressure condition violated")
    print()
    
    # Bottom boundary: u_y=0
    ux_bottom, uy_bottom, p_bottom = sol_bottom[0, 0], sol_bottom[0, 1], sol_bottom[0, 2]
    print(f"Bottom boundary (y=0):")
    print(f"  u_y = {uy_bottom:.6f} (should be 0)")
    print(f"  u_x = {ux_bottom:.6f}")
    print(f"  p = {p_bottom:.6f}")
    bottom_ok = abs(uy_bottom) < 1e-6
    print(f"  ✓ Displacement condition satisfied" if bottom_ok else f"  ❌ Displacement condition violated")
    print()
    
    # Top boundary: ∂p/∂y=0 (check numerically)
    x_top_plus = jnp.array([[0.5, 1.0 + 1e-6]])
    x_top_minus = jnp.array([[0.5, 1.0 - 1e-6]])
    sol_top_plus = BiotCoupled2D.exact_solution(all_params, x_top_plus)
    sol_top_minus = BiotCoupled2D.exact_solution(all_params, x_top_minus)
    dpdy_top = (sol_top_plus[0, 2] - sol_top_minus[0, 2]) / (2e-6)
    
    print(f"Top boundary (y=1):")
    print(f"  ∂p/∂y = {dpdy_top:.6f} (should be 0)")
    top_ok = abs(dpdy_top) < 1e-6
    print(f"  ✓ Pressure gradient condition satisfied" if top_ok else f"  ❌ Pressure gradient condition violated")
    print()
    
    # CRITICAL: Check traction boundary conditions
    print("🔧 TRACTION BOUNDARY CONDITION VERIFICATION:")
    print("-" * 45)
    
    # Right boundary traction: σxx = 0, σxy = 0
    # We need to compute derivatives at x=1
    x_right_plus = jnp.array([[1.0 + 1e-6, 0.5]])
    x_right_minus = jnp.array([[1.0 - 1e-6, 0.5]])
    x_right_y_plus = jnp.array([[1.0, 0.5 + 1e-6]])
    x_right_y_minus = jnp.array([[1.0, 0.5 - 1e-6]])
    
    sol_right_plus = BiotCoupled2D.exact_solution(all_params, x_right_plus)
    sol_right_minus = BiotCoupled2D.exact_solution(all_params, x_right_minus)
    sol_right_y_plus = BiotCoupled2D.exact_solution(all_params, x_right_y_plus)
    sol_right_y_minus = BiotCoupled2D.exact_solution(all_params, x_right_y_minus)
    
    # Compute derivatives at x=1
    duxdx_right = (sol_right_plus[0, 0] - sol_right_minus[0, 0]) / (2e-6)
    duydy_right = (sol_right_y_plus[0, 1] - sol_right_y_minus[0, 1]) / (2e-6)
    duxdy_right = (sol_right_y_plus[0, 0] - sol_right_y_minus[0, 0]) / (2e-6)
    duydx_right = (sol_right_plus[0, 1] - sol_right_minus[0, 1]) / (2e-6)
    
    # Compute stress components
    sigma_xx_right = (2*G + lam) * duxdx_right + lam * duydy_right - alpha * p_right
    sigma_xy_right = G * (duxdy_right + duydx_right)
    
    print(f"Right boundary traction (x=1):")
    print(f"  ∂u_x/∂x = {duxdx_right:.6f}")
    print(f"  ∂u_y/∂y = {duydy_right:.6f}")
    print(f"  σ_xx = {sigma_xx_right:.6f} (should be 0)")
    print(f"  σ_xy = {sigma_xy_right:.6f} (should be 0)")
    
    traction_ok = abs(sigma_xx_right) < 1e-6 and abs(sigma_xy_right) < 1e-6
    print(f"  ✓ Traction conditions satisfied" if traction_ok else f"  ❌ Traction conditions violated")
    print()
    
    # Summary
    print("📋 SUMMARY:")
    print("-" * 20)
    all_ok = left_ok and right_p_ok and bottom_ok and top_ok and traction_ok
    print(f"All boundary conditions satisfied: {'✓ YES' if all_ok else '❌ NO'}")
    
    if all_ok:
        print("🎉 EXACT SOLUTION IS MATHEMATICALLY CONSISTENT!")
        print("   The neural network should now be able to learn properly.")
    else:
        print("⚠️  BOUNDARY CONDITIONS STILL VIOLATED!")
        print("   Further mathematical corrections needed.")
    
    return all_ok

if __name__ == "__main__":
    verify_boundary_conditions()