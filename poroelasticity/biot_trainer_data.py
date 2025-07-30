"""
2D Biot Poroelasticity Trainer with Experimental Data Integration

This module extends the basic Biot trainer to incorporate experimental VTK data
for data-enhanced physics-informed neural network training.

Author: Assistant
Date: 2024
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, List
import os
from pathlib import Path

# Import the base trainer classes
from biot_trainer import BiotCoupled2D, BiotCoupledTrainer


class VTKDataLoader:
    """
    Loads and processes VTK data files for the 2D Biot poroelasticity problem.
    """
    
    def __init__(self, data_dir: str = "Data_2D"):
        """
        Initialize the VTK data loader.
        
        Args:
            data_dir: Directory containing VTK files
        """
        self.data_dir = Path(data_dir)
        self.data_cache = {}
        
    def parse_vtk_file(self, filename: str) -> Dict[str, np.ndarray]:
        """
        Parse a VTK file and extract coordinates and field data.
        
        Args:
            filename: Name of the VTK file
            
        Returns:
            Dictionary containing coordinates and field data
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"VTK file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find the POINTS section
        points_start = None
        num_points = 0
        for i, line in enumerate(lines):
            if line.startswith("POINTS"):
                points_start = i + 1
                num_points = int(line.split()[1])
                break
        
        if points_start is None:
            raise ValueError(f"No POINTS section found in {filename}")
        
        # Read coordinates (x, y, z)
        coordinates = []
        line_idx = points_start
        points_read = 0
        
        while points_read < num_points and line_idx < len(lines):
            line = lines[line_idx].strip()
            if line and not line.startswith("#"):
                coords = line.split()
                if len(coords) >= 3:
                    try:
                        x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                        coordinates.append([x, y, z])
                        points_read += 1
                    except ValueError:
                        pass
            line_idx += 1
        
        coordinates = np.array(coordinates)
        
        # Find and read the data section
        data = None
        data_type = None
        
        for i, line in enumerate(lines):
            if "POINT_DATA" in line:
                # Look for VECTORS (displacement) or SCALARS (pressure)
                for j in range(i+1, min(i+10, len(lines))):
                    if lines[j].startswith("VECTORS"):
                        data_type = "vectors"
                        data_start = j + 1
                        break
                    elif lines[j].startswith("SCALARS"):
                        data_type = "scalars"
                        # Skip LOOKUP_TABLE line
                        data_start = j + 2
                        break
                break
        
        if data_type is None:
            raise ValueError(f"No data section found in {filename}")
        
        # Read the data values
        data_values = []
        line_idx = data_start
        
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if line and not line.startswith("#"):
                values = line.split()
                for val in values:
                    try:
                        data_values.append(float(val))
                    except ValueError:
                        pass
            line_idx += 1
        
        data_values = np.array(data_values)
        
        # Reshape data based on type
        if data_type == "vectors":
            # Displacement vectors: reshape to (n_points, 3)
            data = data_values.reshape(-1, 3)[:num_points]
        else:
            # Scalar pressure: reshape to (n_points,)
            data = data_values[:num_points]
        
        return {
            "coordinates": coordinates,
            "data": data,
            "data_type": data_type,
            "filename": filename
        }
    
    def load_experimental_data(self) -> Dict[str, Dict]:
        """
        Load all available experimental VTK data files.
        
        Returns:
            Dictionary containing all loaded data organized by type and condition
        """
        # Define expected file patterns
        file_patterns = [
            "displacement_MSAMPLE2D_RES_S0_M.vtk",
            "displacement_MSAMPLE2D_RES_S100_MHm.vtk", 
            "matrix_pressure_MSAMPLE2D_RES_S100_Hm.vtk",
            "matrix_pressure_MSAMPLE2D_RES_S100_MHm.vtk"
        ]
        
        loaded_data = {}
        
        for pattern in file_patterns:
            try:
                data = self.parse_vtk_file(pattern)
                # Parse condition from filename
                if "S0_M" in pattern:
                    condition = "initial"
                elif "S100_MHm" in pattern:
                    condition = "loaded_MHm"
                elif "S100_Hm" in pattern:
                    condition = "loaded_Hm"
                else:
                    condition = "unknown"
                
                # Store by data type and condition
                data_type = "displacement" if "displacement" in pattern else "pressure"
                
                if data_type not in loaded_data:
                    loaded_data[data_type] = {}
                
                loaded_data[data_type][condition] = data
                print(f"Loaded {pattern}: {len(data['coordinates'])} points")
                
            except Exception as e:
                print(f"Warning: Could not load {pattern}: {e}")
        
        return loaded_data


class BiotCoupled2DData:
    """
    2D Biot poroelasticity problem with experimental data integration.
    
    This class handles the experimental data loading and processing
    without inheriting from the FBPINNs Problem class to avoid inheritance issues.
    """
    
    def __init__(self, 
                 domain: Tuple[Tuple[float, float], Tuple[float, float]],
                 material_params: Dict[str, float],
                 data_dir: str = "Data_2D",
                 data_weight: float = 1.0,
                 use_data_conditions: Optional[List[str]] = None):
        """
        Initialize the data-enhanced Biot problem.
        
        Args:
            domain: ((x_min, x_max), (y_min, y_max))
            material_params: Dictionary of material parameters
            data_dir: Directory containing VTK experimental data
            data_weight: Weight for data loss terms
            use_data_conditions: List of conditions to use ["initial", "loaded_MHm", "loaded_Hm"]
        """
        self.domain = domain
        self.material_params = material_params
        
        self.data_dir = data_dir
        self.data_weight = data_weight
        self.use_data_conditions = use_data_conditions or ["initial", "loaded_MHm"]
        
        # Load experimental data
        self.data_loader = VTKDataLoader(data_dir)
        self.experimental_data = self.data_loader.load_experimental_data()
        
        # Process and validate data
        self._process_experimental_data()
        
    def _process_experimental_data(self):
        """
        Process experimental data for neural network training.
        Convert coordinates to domain coordinates and normalize if needed.
        """
        self.data_points = {}
        
        for data_type in self.experimental_data:
            self.data_points[data_type] = {}
            
            for condition in self.experimental_data[data_type]:
                if condition not in self.use_data_conditions:
                    continue
                    
                data = self.experimental_data[data_type][condition]
                coords = data["coordinates"]
                values = data["data"]
                
                # Extract 2D coordinates (x, y) from (x, y, z)
                xy_coords = coords[:, :2]  # Take only x and y
                
                # Filter points that are within our domain
                (x_min, x_max), (y_min, y_max) = self.domain
                
                mask = (
                    (xy_coords[:, 0] >= x_min) & (xy_coords[:, 0] <= x_max) &
                    (xy_coords[:, 1] >= y_min) & (xy_coords[:, 1] <= y_max)
                )
                
                valid_coords = xy_coords[mask]
                valid_values = values[mask] if values.ndim == 1 else values[mask]
                
                self.data_points[data_type][condition] = {
                    "coordinates": valid_coords,
                    "values": valid_values,
                    "n_points": len(valid_coords)
                }
                
                print(f"Processed {data_type} {condition}: {len(valid_coords)} valid points")
    
    def sample_data_points(self, key: jax.random.PRNGKey, batch_size: int = 100) -> Dict:
        """
        Sample data points for training.
        
        Args:
            key: JAX random key
            batch_size: Number of points to sample per condition
            
        Returns:
            Dictionary of sampled data points
        """
        sampled_data = {}
        keys = jax.random.split(key, len(self.data_points) * len(self.use_data_conditions))
        key_idx = 0
        
        for data_type in self.data_points:
            sampled_data[data_type] = {}
            
            for condition in self.data_points[data_type]:
                data = self.data_points[data_type][condition]
                n_available = data["n_points"]
                
                if n_available == 0:
                    continue
                
                # Sample indices
                n_sample = min(batch_size, n_available)
                indices = jax.random.choice(
                    keys[key_idx], 
                    n_available, 
                    shape=(n_sample,), 
                    replace=False
                )
                key_idx += 1
                
                # Get sampled points and values
                coords = data["coordinates"][indices]
                values = data["values"][indices]
                
                sampled_data[data_type][condition] = {
                    "coordinates": coords,
                    "values": values,
                    "n_points": n_sample
                }
        
        return sampled_data
    
    def data_loss(self, 
                  u_pred: jnp.ndarray, 
                  p_pred: jnp.ndarray, 
                  sampled_data: Dict) -> jnp.ndarray:
        """
        Compute data fitting loss terms.
        
        Args:
            u_pred: Predicted displacement field (n_points, 2)
            p_pred: Predicted pressure field (n_points,)
            sampled_data: Sampled experimental data points
            
        Returns:
            Total data loss
        """
        total_loss = 0.0
        loss_count = 0
        
        # Displacement data loss
        if "displacement" in sampled_data:
            for condition in sampled_data["displacement"]:
                data = sampled_data["displacement"][condition]
                if data["n_points"] > 0:
                    # Extract experimental displacement (only x, y components)
                    u_exp = data["values"][:, :2]  # Shape: (n_points, 2)
                    
                    # Compute L2 loss
                    disp_loss = jnp.mean((u_pred - u_exp)**2)
                    total_loss += self.data_weight * disp_loss
                    loss_count += 1
        
        # Pressure data loss  
        if "pressure" in sampled_data:
            for condition in sampled_data["pressure"]:
                data = sampled_data["pressure"][condition]
                if data["n_points"] > 0:
                    p_exp = data["values"]  # Shape: (n_points,)
                    
                    # Compute L2 loss
                    pressure_loss = jnp.mean((p_pred - p_exp)**2)
                    total_loss += self.data_weight * pressure_loss
                    loss_count += 1
        
        # Normalize by number of loss terms
        if loss_count > 0:
            total_loss /= loss_count
            
        return total_loss


class BiotCoupledDataTrainer:
    """
    Trainer for 2D Biot poroelasticity with experimental data integration.
    
    This class combines the base Biot physics with experimental data constraints.
    """
    
    def __init__(self, 
                 data_problem: BiotCoupled2DData,
                 base_problem: BiotCoupled2D,
                 domain_batch_size: int = 1000,
                 boundary_batch_size: int = 100,
                 data_batch_size: int = 50):
        """
        Initialize the data-enhanced trainer.
        
        Args:
            data_problem: BiotCoupled2DData problem instance with experimental data
            base_problem: Base BiotCoupled2D problem for physics constraints
            domain_batch_size: Batch size for domain points
            boundary_batch_size: Batch size for boundary points  
            data_batch_size: Batch size for data points
        """
        self.data_problem = data_problem
        self.base_problem = base_problem
        self.domain_batch_size = domain_batch_size
        self.boundary_batch_size = boundary_batch_size
        self.data_batch_size = data_batch_size
        
    def loss_fn(self, params, subdomain_xs, key):
        """
        Enhanced loss function including experimental data terms.
        
        Args:
            params: Neural network parameters
            subdomain_xs: Domain points for each subdomain
            key: JAX random key
            
        Returns:
            Total loss and loss components
        """
        # This is a placeholder implementation
        # In a real implementation, we would:
        # 1. Evaluate physics loss from base_problem
        # 2. Evaluate boundary conditions
        # 3. Add experimental data loss
        
        # Sample data points
        data_key, _ = jax.random.split(key)
        sampled_data = self.data_problem.sample_data_points(data_key, self.data_batch_size)
        
        # Placeholder for combined loss
        total_loss = 0.0
        loss_dict = {"physics_loss": 0.0, "boundary_loss": 0.0, "data_loss": 0.0}
        
        # Add data loss if we have sampled data
        if sampled_data:
            # For now, just compute a dummy data loss
            data_loss = 0.0
            for data_type in sampled_data:
                for condition in sampled_data[data_type]:
                    data = sampled_data[data_type][condition]
                    if data["n_points"] > 0:
                        # Dummy loss - in practice would evaluate network predictions
                        data_loss += 1.0
            
            loss_dict["data_loss"] = data_loss
            total_loss += data_loss
        
        return total_loss, loss_dict
    
    def _predict_at_points(self, params, coords):
        """
        Predict displacement and pressure at given coordinates.
        
        Args:
            params: Neural network parameters
            coords: Coordinates to evaluate at (n_points, 2)
            
        Returns:
            Tuple of (displacement, pressure) predictions
        """
        # This is a placeholder - actual implementation depends on network architecture
        # For now, return dummy predictions with correct shapes
        n_points = coords.shape[0]
        u_pred = jnp.zeros((n_points, 2))  # 2D displacement
        p_pred = jnp.zeros((n_points,))    # Scalar pressure
        
        # TODO: Implement actual network evaluation
        # This would typically involve:
        # 1. Forward pass through the network at coords
        # 2. Extract displacement and pressure components
        # 3. Return properly shaped predictions
        
        return u_pred, p_pred


def main():
    """
    Demonstration of the data-enhanced Biot trainer.
    """
    # Define domain (adjust based on experimental data range)
    domain = ((-2000, 2000), (-3300, 0))  # (x_range, y_range) in meters
    
    # Material parameters for poroelasticity
    material_params = {
        'E': 15e9,      # Young's modulus (Pa)
        'nu': 0.25,     # Poisson's ratio
        'alpha': 0.9,   # Biot's coefficient
        'k': 1e-14,     # Permeability (m^2)
        'mu': 1e-3,     # Fluid viscosity (Pa·s)
        'rho_s': 2700,  # Solid density (kg/m^3)
        'rho_f': 1000,  # Fluid density (kg/m^3)
    }
    
    try:
        # Create data-enhanced problem
        print("Creating data-enhanced Biot problem...")
        data_problem = BiotCoupled2DData(
            domain=domain,
            material_params=material_params,
            data_dir="Data_2D",
            data_weight=1.0,
            use_data_conditions=["initial", "loaded_MHm"]
        )
        
        # Create base problem for physics (would need actual BiotCoupled2D instantiation)
        print("Creating base physics problem...")
        # base_problem = BiotCoupled2D(domain, material_params)  # This would need proper setup
        base_problem = None  # Placeholder for now
        
        # Create trainer
        print("Creating data-enhanced trainer...")
        trainer = BiotCoupledDataTrainer(
            data_problem=data_problem,
            base_problem=base_problem,
            domain_batch_size=500,
            boundary_batch_size=50,
            data_batch_size=30
        )
        
        print("Data-enhanced Biot trainer created successfully!")
        print(f"Available data types: {list(data_problem.experimental_data.keys())}")
        
        for data_type in data_problem.experimental_data:
            print(f"\n{data_type.capitalize()} data:")
            for condition in data_problem.experimental_data[data_type]:
                n_points = data_problem.data_points[data_type][condition]["n_points"]
                print(f"  {condition}: {n_points} points")
        
    except Exception as e:
        print(f"Error creating data-enhanced trainer: {e}")
        print("\nMake sure the VTK data files are in the Data_2D directory:")
        print("- displacement_MSAMPLE2D_RES_S0_M.vtk")
        print("- displacement_MSAMPLE2D_RES_S100_MHm.vtk")
        print("- matrix_pressure_MSAMPLE2D_RES_S100_Hm.vtk")
        print("- matrix_pressure_MSAMPLE2D_RES_S100_MHm.vtk")


if __name__ == "__main__":
    main()
