"""
Comprehensive test for the data-enhanced Biot trainer
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

# Now we can import from the data trainer
from biot_trainer_data import VTKDataLoader, BiotCoupled2DData


def test_vtk_data_loader():
    """Test the VTK data loader class."""
    print("Testing VTK Data Loader...")
    print("-" * 30)
    
    try:
        loader = VTKDataLoader("Data_2D")
        experimental_data = loader.load_experimental_data()
        
        print(f"✓ Loaded experimental data successfully")
        print(f"Data types available: {list(experimental_data.keys())}")
        
        for data_type in experimental_data:
            print(f"\n{data_type.capitalize()} data:")
            for condition in experimental_data[data_type]:
                data = experimental_data[data_type][condition]
                print(f"  {condition}: {len(data['coordinates'])} points")
        
        return True, experimental_data
        
    except Exception as e:
        print(f"✗ Error in VTK data loader: {e}")
        return False, None


def test_data_enhanced_problem():
    """Test the data-enhanced Biot problem class."""
    print("\nTesting Data-Enhanced Biot Problem...")
    print("-" * 40)
    
    # Define domain based on experimental data range
    # Convert from 3D (x,y,z) to 2D (x,y) coordinates
    # Experimental data: x ∈ [-2000, 2000], z ∈ [-3300, 0]
    # Map z → y for 2D problem
    domain = ((-2000, 2000), (-3300, 0))  # (x_range, y_range)
    
    # Realistic material parameters for rock/soil
    material_params = {
        'E': 15e9,      # Young's modulus (Pa) - typical for rock
        'nu': 0.25,     # Poisson's ratio - typical for rock
        'alpha': 0.9,   # Biot's coefficient - high for porous rock
        'k': 1e-14,     # Permeability (m^2) - low permeability rock
        'mu': 1e-3,     # Fluid viscosity (Pa·s) - water
        'rho_s': 2700,  # Solid density (kg/m^3) - rock
        'rho_f': 1000,  # Fluid density (kg/m^3) - water
    }
    
    try:
        # Create data-enhanced problem
        data_problem = BiotCoupled2DData(
            domain=domain,
            material_params=material_params,
            data_dir="Data_2D",
            data_weight=1.0,
            use_data_conditions=["initial", "loaded_MHm"]
        )
        
        print(f"✓ Created data-enhanced problem successfully")
        print(f"Domain: {data_problem.domain}")
        print(f"Material parameters: {len(data_problem.material_params)} parameters")
        print(f"Data weight: {data_problem.data_weight}")
        print(f"Using conditions: {data_problem.use_data_conditions}")
        
        # Test data point sampling
        key = jax.random.PRNGKey(42)
        sampled_data = data_problem.sample_data_points(key, batch_size=50)
        
        print(f"\nSampled data for training:")
        for data_type in sampled_data:
            print(f"  {data_type}:")
            for condition in sampled_data[data_type]:
                n_points = sampled_data[data_type][condition]["n_points"]
                coords_shape = sampled_data[data_type][condition]["coordinates"].shape
                values_shape = sampled_data[data_type][condition]["values"].shape
                print(f"    {condition}: {n_points} points, coords {coords_shape}, values {values_shape}")
        
        # Test data loss computation (with dummy predictions)
        n_test_points = 50
        u_dummy = jnp.zeros((n_test_points, 2))  # 2D displacement
        p_dummy = jnp.ones((n_test_points,)) * 5e6  # 5 MPa pressure
        
        # Filter sampled data to match dummy prediction size
        filtered_data = {}
        for data_type in sampled_data:
            filtered_data[data_type] = {}
            for condition in sampled_data[data_type]:
                data = sampled_data[data_type][condition]
                if data["n_points"] > 0:
                    n_take = min(n_test_points, data["n_points"])
                    filtered_data[data_type][condition] = {
                        "coordinates": data["coordinates"][:n_take],
                        "values": data["values"][:n_take],
                        "n_points": n_take
                    }
        
        data_loss = data_problem.data_loss(u_dummy, p_dummy, filtered_data)
        print(f"\nData loss computation: {data_loss:.6f}")
        
        return True, data_problem
        
    except Exception as e:
        print(f"✗ Error in data-enhanced problem: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_coordinate_mapping():
    """Test coordinate mapping from experimental 3D to problem 2D."""
    print("\nTesting Coordinate Mapping...")
    print("-" * 30)
    
    try:
        loader = VTKDataLoader("Data_2D")
        
        # Load one file to check coordinates
        sample_data = loader.parse_vtk_file("displacement_MSAMPLE2D_RES_S0_M.vtk")
        coords_3d = sample_data["coordinates"]
        
        print(f"Original 3D coordinates shape: {coords_3d.shape}")
        print(f"X range: [{coords_3d[:, 0].min():.1f}, {coords_3d[:, 0].max():.1f}]")
        print(f"Y range: [{coords_3d[:, 1].min():.1f}, {coords_3d[:, 1].max():.1f}]")
        print(f"Z range: [{coords_3d[:, 2].min():.1f}, {coords_3d[:, 2].max():.1f}]")
        
        # Convert to 2D by taking (x, z) → (x, y)
        coords_2d = coords_3d[:, [0, 2]]  # Take x and z columns
        
        print(f"\nMapped 2D coordinates shape: {coords_2d.shape}")
        print(f"X range: [{coords_2d[:, 0].min():.1f}, {coords_2d[:, 0].max():.1f}]")
        print(f"Y range: [{coords_2d[:, 1].min():.1f}, {coords_2d[:, 1].max():.1f}]")
        
        # Check if coordinates are within a reasonable domain
        domain_x = (coords_2d[:, 0].min(), coords_2d[:, 0].max())
        domain_y = (coords_2d[:, 1].min(), coords_2d[:, 1].max())
        
        print(f"\nSuggested domain:")
        print(f"  x_range: {domain_x}")
        print(f"  y_range: {domain_y}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in coordinate mapping: {e}")
        return False


def test_data_statistics():
    """Analyze the experimental data statistics."""
    print("\nAnalyzing Data Statistics...")
    print("-" * 30)
    
    try:
        loader = VTKDataLoader("Data_2D")
        experimental_data = loader.load_experimental_data()
        
        for data_type in experimental_data:
            print(f"\n{data_type.upper()} DATA:")
            
            for condition in experimental_data[data_type]:
                data = experimental_data[data_type][condition]
                values = data["data"]
                
                print(f"  {condition}:")
                
                if data["data_type"] == "vectors":
                    # Displacement vectors
                    print(f"    Shape: {values.shape}")
                    print(f"    X-displacement: [{values[:, 0].min():.6f}, {values[:, 0].max():.6f}] m")
                    print(f"    Y-displacement: [{values[:, 1].min():.6f}, {values[:, 1].max():.6f}] m") 
                    print(f"    Z-displacement: [{values[:, 2].min():.6f}, {values[:, 2].max():.6f}] m")
                    
                    # Compute displacement magnitudes
                    magnitude = np.sqrt(np.sum(values**2, axis=1))
                    print(f"    Magnitude: [{magnitude.min():.6f}, {magnitude.max():.6f}] m")
                    
                else:
                    # Scalar pressure
                    print(f"    Shape: {values.shape}")
                    print(f"    Range: [{values.min():.2e}, {values.max():.2e}] Pa")
                    print(f"    Mean: {values.mean():.2e} Pa")
                    print(f"    Std:  {values.std():.2e} Pa")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in data statistics: {e}")
        return False


def main():
    """Run comprehensive tests for the data-enhanced Biot trainer."""
    print("Data-Enhanced Biot Trainer - Comprehensive Test")
    print("=" * 50)
    
    # Test 1: VTK Data Loader
    success1, exp_data = test_vtk_data_loader()
    
    # Test 2: Coordinate Mapping
    success2 = test_coordinate_mapping()
    
    # Test 3: Data Statistics
    success3 = test_data_statistics()
    
    # Test 4: Data-Enhanced Problem
    success4, data_problem = test_data_enhanced_problem()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print(f"VTK Data Loader:          {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Coordinate Mapping:       {'✓ PASS' if success2 else '✗ FAIL'}")
    print(f"Data Statistics:          {'✓ PASS' if success3 else '✗ FAIL'}")
    print(f"Data-Enhanced Problem:    {'✓ PASS' if success4 else '✗ FAIL'}")
    
    overall_success = all([success1, success2, success3, success4])
    print(f"\nOverall Result:           {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\nThe data-enhanced Biot trainer is ready for training!")
        print("\nNext steps:")
        print("1. Implement network evaluation in _predict_at_points method")
        print("2. Set up FBPINNs training loop with data constraints")
        print("3. Run sensitivity analysis on material parameters")
        print("4. Compare physics-only vs data-enhanced results")
    
    return overall_success


if __name__ == "__main__":
    main()
