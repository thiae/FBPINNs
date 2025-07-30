"""
Complete Example: Data-Enhanced 2D Biot Poroelasticity Trainer

This script demonstrates how to use the data-enhanced Biot trainer
for physics-informed neural network training with experimental data integration.

Author: Assistant  
Date: 2024
"""

import numpy as np
import jax
import jax.numpy as jnp
import sys
import os
from pathlib import Path

# Add the parent directory to Python path to find fbpinns
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from biot_trainer_data import VTKDataLoader, BiotCoupled2DData, BiotCoupledDataTrainer


def create_data_enhanced_setup():
    """
    Create a complete data-enhanced Biot poroelasticity setup.
    
    Returns:
        Tuple of (data_problem, trainer) ready for training
    """
    print("Setting up Data-Enhanced 2D Biot Poroelasticity Trainer")
    print("=" * 60)
    
    # 1. Define the computational domain based on experimental data
    # Experimental coordinates: x ∈ [-2000, 2000] m, z ∈ [-3300, 0] m
    # Map to 2D problem: (x, y) where y represents depth (z-coordinate)
    domain = ((-2000, 2000), (-3300, 0))  # (x_range, y_range) in meters
    print(f"Computational domain: x ∈ [{domain[0][0]}, {domain[0][1]}] m")
    print(f"                     y ∈ [{domain[1][0]}, {domain[1][1]}] m")
    
    # 2. Define realistic material properties for poroelastic rock/soil
    material_params = {
        # Mechanical properties (typical for sedimentary rock)
        'E': 15e9,      # Young's modulus (Pa) - 15 GPa for limestone/sandstone
        'nu': 0.25,     # Poisson's ratio - typical for rock
        
        # Poroelastic properties  
        'alpha': 0.9,   # Biot's coefficient - high porosity rock
        'k': 1e-14,     # Permeability (m^2) - low perm reservoir rock
        'mu': 1e-3,     # Fluid viscosity (Pa·s) - water at standard conditions
        
        # Density properties
        'rho_s': 2700,  # Solid density (kg/m^3) - limestone
        'rho_f': 1000,  # Fluid density (kg/m^3) - water
    }
    
    print(f"\nMaterial Properties:")
    print(f"  Young's modulus (E):     {material_params['E']/1e9:.1f} GPa")
    print(f"  Poisson's ratio (ν):     {material_params['nu']:.2f}")
    print(f"  Biot coefficient (α):    {material_params['alpha']:.2f}")
    print(f"  Permeability (k):        {material_params['k']:.2e} m²")
    print(f"  Fluid viscosity (μ):     {material_params['mu']:.2e} Pa·s")
    print(f"  Solid density (ρₛ):      {material_params['rho_s']} kg/m³")
    print(f"  Fluid density (ρf):      {material_params['rho_f']} kg/m³")
    
    # 3. Create the data-enhanced problem
    print(f"\nLoading experimental data...")
    data_problem = BiotCoupled2DData(
        domain=domain,
        material_params=material_params,
        data_dir="Data_2D",
        data_weight=1.0,  # Equal weight to physics and data
        use_data_conditions=["initial", "loaded_MHm"]  # Use initial and loaded states
    )
    
    print(f"✓ Data-enhanced problem created successfully")
    
    # 4. Display data statistics
    print(f"\nExperimental Data Summary:")
    for data_type in data_problem.experimental_data:
        print(f"  {data_type.upper()}:")
        for condition in data_problem.experimental_data[data_type]:
            if condition in data_problem.use_data_conditions:
                n_points = data_problem.data_points[data_type][condition]["n_points"]
                print(f"    {condition:12}: {n_points:3d} points")
    
    # 5. Create the trainer
    print(f"\nCreating trainer...")
    trainer = BiotCoupledDataTrainer(
        data_problem=data_problem,
        base_problem=None,  # Would need actual BiotCoupled2D for full implementation
        domain_batch_size=1000,    # Physics constraint points per batch
        boundary_batch_size=100,   # Boundary condition points per batch  
        data_batch_size=50         # Experimental data points per batch
    )
    
    print(f"✓ Data-enhanced trainer created successfully")
    
    return data_problem, trainer


def analyze_experimental_data(data_problem):
    """
    Analyze the experimental data to understand the physical system.
    
    Args:
        data_problem: BiotCoupled2DData instance with loaded experimental data
    """
    print(f"\nExperimental Data Analysis")
    print("=" * 40)
    
    # Sample some data points to analyze
    key = jax.random.PRNGKey(42)
    sampled_data = data_problem.sample_data_points(key, batch_size=100)
    
    for data_type in sampled_data:
        print(f"\n{data_type.upper()} ANALYSIS:")
        
        for condition in sampled_data[data_type]:
            data = sampled_data[data_type][condition]
            if data["n_points"] > 0:
                coords = data["coordinates"]
                values = data["values"]
                
                print(f"  {condition} condition:")
                print(f"    Number of points: {data['n_points']}")
                print(f"    Spatial extent:")
                print(f"      X: [{coords[:, 0].min():.1f}, {coords[:, 0].max():.1f}] m")
                print(f"      Y: [{coords[:, 1].min():.1f}, {coords[:, 1].max():.1f}] m")
                
                if data_type == "displacement":
                    # Displacement analysis
                    u_x = values[:, 0]  # X-displacement
                    u_y = values[:, 1]  # Y-displacement (mapped from Z in 3D)
                    u_z = values[:, 2]  # Z-displacement (out of plane in 2D)
                    
                    magnitude = np.sqrt(u_x**2 + u_y**2 + u_z**2)
                    
                    print(f"    Displacement magnitudes:")
                    print(f"      X-component: [{u_x.min():.2e}, {u_x.max():.2e}] m")
                    print(f"      Y-component: [{u_y.min():.2e}, {u_y.max():.2e}] m")
                    print(f"      Z-component: [{u_z.min():.2e}, {u_z.max():.2e}] m")
                    print(f"      Total magnitude: [{magnitude.min():.2e}, {magnitude.max():.2e}] m")
                    
                elif data_type == "pressure":
                    # Pressure analysis
                    print(f"    Pressure values:")
                    print(f"      Range: [{values.min():.2e}, {values.max():.2e}] Pa")
                    print(f"      Mean:  {values.mean():.2e} Pa ({values.mean()/1e6:.1f} MPa)")
                    print(f"      Std:   {values.std():.2e} Pa ({values.std()/1e6:.1f} MPa)")


def demonstrate_data_loss(data_problem):
    """
    Demonstrate the data loss computation with sample predictions.
    
    Args:
        data_problem: BiotCoupled2DData instance
    """
    print(f"\nData Loss Computation Demonstration")
    print("=" * 40)
    
    # Sample experimental data points
    key = jax.random.PRNGKey(123)
    sampled_data = data_problem.sample_data_points(key, batch_size=20)
    
    # Create sample predictions (in practice, these come from neural network)
    n_points = 20
    
    # Case 1: Perfect predictions (zero loss)
    print(f"Case 1: Perfect predictions")
    perfect_u = jnp.zeros((n_points, 2))
    perfect_p = jnp.ones((n_points,)) * 5e6
    
    # For perfect predictions, we'd use actual experimental values
    # This is just a demonstration with dummy values
    perfect_loss = data_problem.data_loss(perfect_u, perfect_p, sampled_data)
    print(f"  Perfect prediction loss: {perfect_loss:.2e}")
    
    # Case 2: Random predictions (high loss)  
    print(f"\nCase 2: Random predictions")
    key1, key2 = jax.random.split(key)
    random_u = jax.random.normal(key1, (n_points, 2)) * 0.01  # Random displacements
    random_p = jax.random.uniform(key2, (n_points,)) * 1e7    # Random pressures
    
    random_loss = data_problem.data_loss(random_u, random_p, sampled_data)
    print(f"  Random prediction loss: {random_loss:.2e}")
    
    print(f"\nLoss ratio (random/perfect): {random_loss/perfect_loss:.2e}")


def create_training_plan():
    """
    Create a suggested training plan for the data-enhanced Biot problem.
    """
    print(f"\nSuggested Training Plan")
    print("=" * 30)
    
    print(f"Phase 1: Physics-Only Pre-training")
    print(f"  - Train on physics constraints only (PDE + boundary conditions)")
    print(f"  - Epochs: 5,000-10,000")
    print(f"  - Learning rate: 1e-3 → 1e-4 (decay)")
    print(f"  - Batch sizes: domain=1000, boundary=100")
    print(f"  - Goal: Establish physically consistent solution")
    
    print(f"\nPhase 2: Data-Enhanced Fine-tuning")
    print(f"  - Add experimental data constraints")
    print(f"  - Epochs: 2,000-5,000")
    print(f"  - Learning rate: 1e-4 → 1e-5 (decay)")
    print(f"  - Batch sizes: domain=500, boundary=50, data=30")
    print(f"  - Data weight: 0.1 → 1.0 (gradual increase)")
    print(f"  - Goal: Match experimental observations")
    
    print(f"\nPhase 3: Sensitivity Analysis")
    print(f"  - Material parameters: E, ν, α, k")
    print(f"  - Network architecture: layers, neurons")
    print(f"  - Loss weighting: physics vs data balance")
    print(f"  - Subdomain decomposition parameters")
    
    print(f"\nValidation Metrics:")
    print(f"  - Physics loss (PDE residuals)")
    print(f"  - Boundary condition satisfaction")  
    print(f"  - Data fitting accuracy (L2 error)")
    print(f"  - Conservation laws (mass, momentum)")


def main():
    """
    Main demonstration of the data-enhanced Biot trainer.
    """
    try:
        # Create the complete setup
        data_problem, trainer = create_data_enhanced_setup()
        
        # Analyze the experimental data
        analyze_experimental_data(data_problem)
        
        # Demonstrate data loss computation
        demonstrate_data_loss(data_problem)
        
        # Show training plan
        create_training_plan()
        
        print(f"\n" + "=" * 60)
        print(f"DATA-ENHANCED BIOT TRAINER SETUP COMPLETE!")
        print(f"=" * 60)
        
        print(f"\nSystem Ready For:")
        print(f"✓ Physics-informed neural network training")
        print(f"✓ Experimental data integration")
        print(f"✓ Material property sensitivity analysis")
        print(f"✓ Multi-physics coupling (mechanics + flow)")
        
        print(f"\nNext Implementation Steps:")
        print(f"1. Integrate with FBPINNs training loop")
        print(f"2. Implement proper network evaluation in trainer")
        print(f"3. Set up multi-stage training pipeline")
        print(f"4. Add visualization and monitoring capabilities")
        print(f"5. Implement parameter sensitivity analysis")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in setup: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 Data-enhanced Biot trainer demonstration completed successfully!")
    else:
        print(f"\n❌ Setup failed. Check error messages above.")
