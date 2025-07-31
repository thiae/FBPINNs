"""
2D Biot Poroelasticity Trainer with Experimental Data Integration

This module extends the basic Biot trainer to incorporate experimental VTK data
for data-enhanced physics-informed neural network training.

"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Optional, List
import os
from pathlib import Path

# Import the base trainer classes
from .biot_trainer_2d import BiotCoupled2D, BiotCoupledTrainer

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


class BiotCoupledDataTrainer(BiotCoupledTrainer):
    """
    Data-enhanced trainer for 2D Biot poroelasticity.
    
    This trainer extends the base BiotCoupledTrainer to incorporate experimental data
    from VTK files into the training process.
    """
    
    def __init__(self, 
                 data_dir: str = "Data_2D",
                 data_weight: float = 1.0,
                 data_batch_size: int = 32,
                 use_data_conditions: Optional[List[str]] = None,
                 w_mech=1.0, w_flow=1.0, w_bc=1.0, auto_balance=True):
        """
        Initialize the data-enhanced trainer.
        
        Args:
            data_dir: Directory containing VTK data files
            data_weight: Weight for data loss term
            data_batch_size: Batch size for data sampling
            use_data_conditions: List of conditions to use (if None, use all available)
            w_mech: Base weight for mechanics equation
            w_flow: Base weight for flow equation
            w_bc: Base weight for boundary conditions
            auto_balance: Whether to use automatic loss balancing
        """
        # Initialize base trainer
        super().__init__(w_mech, w_flow, w_bc, auto_balance)
        
        self.data_weight = data_weight
        self.data_batch_size = data_batch_size
        
        # Initialize VTK data loader
        self.data_loader = VTKDataLoader(data_dir)
        self.experimental_data = None
        self.use_data_conditions = use_data_conditions
        
        # Load experimental data
        self._load_experimental_data()
        
    def _load_experimental_data(self):
        """Load and process experimental VTK data"""
        try:
            self.experimental_data = self.data_loader.load_experimental_data()
            print(f"Loaded experimental data: {list(self.experimental_data.keys())}")
            
            # Filter by conditions if specified
            if self.use_data_conditions:
                filtered_data = {}
                for data_type in self.experimental_data:
                    filtered_data[data_type] = {}
                    for condition in self.experimental_data[data_type]:
                        if condition in self.use_data_conditions:
                            filtered_data[data_type][condition] = self.experimental_data[data_type][condition]
                self.experimental_data = filtered_data
                print(f"Filtered to conditions: {self.use_data_conditions}")
                
        except Exception as e:
            print(f"Warning: Could not load experimental data: {e}")
            self.experimental_data = None
    
    def _sample_data_points(self, key, batch_size):
        """Sample random points from experimental data"""
        if self.experimental_data is None:
            return None
            
        sampled_data = {}
        
        for data_type in self.experimental_data:
            sampled_data[data_type] = {}
            for condition in self.experimental_data[data_type]:
                data = self.experimental_data[data_type][condition]
                coords = data["coordinates"][:, :2]  # Extract x, y coordinates
                values = data["data"]
                
                # Sample random indices
                n_points = min(batch_size, len(coords))
                indices = jax.random.choice(key, len(coords), shape=(n_points,), replace=False)
                
                sampled_coords = coords[indices]
                sampled_values = values[indices] if values.ndim > 1 else values[indices]
                
                sampled_data[data_type][condition] = {
                    "coordinates": jnp.array(sampled_coords),
                    "values": jnp.array(sampled_values),
                    "n_points": n_points
                }
                
        return sampled_data
    
    def _compute_data_loss(self, all_params, sampled_data):
        """Compute loss against experimental data"""
        if sampled_data is None:
            return 0.0
            
        total_data_loss = 0.0
        
        for data_type in sampled_data:
            for condition in sampled_data[data_type]:
                data = sampled_data[data_type][condition]
                if data["n_points"] == 0:
                    continue
                    
                coords = data["coordinates"]
                target_values = data["values"]
                
                # Get network predictions at data coordinates
                predictions = self.predict(coords)
                
                if data_type == "displacement":
                    # Compare displacement components (first 2 outputs)
                    pred_displacement = predictions[:, :2]
                    if target_values.shape[1] >= 2:  # Has x,y components
                        target_displacement = target_values[:, :2]
                        displacement_loss = jnp.mean((pred_displacement - target_displacement)**2)
                        total_data_loss += displacement_loss
                        
                elif data_type == "pressure":
                    # Compare pressure (third output)
                    pred_pressure = predictions[:, 2]
                    if target_values.ndim == 1:  # Scalar pressure
                        pressure_loss = jnp.mean((pred_pressure - target_values)**2)
                        total_data_loss += pressure_loss
                        
        return total_data_loss
    
    def train_with_data(self, n_steps=1000, data_schedule=None):
        """
        Train with experimental data integration.
        
        Args:
            n_steps: Number of training steps
            data_schedule: Schedule for data weight (if None, use constant weight)
        """
        if self.experimental_data is None:
            print("No experimental data available, falling back to physics-only training")
            return self.train_coupled(n_steps)
            
        print(f"Training with experimental data integration (weight: {self.data_weight})")
        
        # Create data-enhanced loss function
        original_loss_fn = BiotCoupled2D.loss_fn
        
        def data_enhanced_loss_fn(all_params, constraints, w_mech=1.0, w_flow=1.0, w_bc=1.0, auto_balance=True):
            # Compute physics loss
            physics_loss = original_loss_fn(all_params, constraints, w_mech, w_flow, w_bc, auto_balance)
            
            # Sample and compute data loss
            key = jax.random.PRNGKey(all_params.get("step", 0))
            sampled_data = self._sample_data_points(key, self.data_batch_size)
            data_loss = self._compute_data_loss(all_params, sampled_data)
            
            # Apply data weight schedule if provided
            current_data_weight = self.data_weight
            if data_schedule is not None:
                step = all_params.get("step", 0)
                if step < len(data_schedule):
                    current_data_weight = data_schedule[step] * self.data_weight
            
            total_loss = physics_loss + current_data_weight * data_loss
            
            # Print loss components occasionally
            def _print_data_losses(step_val):
                jax.debug.print("Step {}: Physics: {:.2e}, Data: {:.2e} (weight: {:.2e})", 
                                step_val, physics_loss, data_loss, current_data_weight)
                return 0
                
            step = all_params.get("step", 0)
            jax.lax.cond(step % 100 == 0, lambda _: _print_data_losses(step), lambda _: 0, None)
            
            return total_loss
        
        # Replace loss function temporarily
        BiotCoupled2D.loss_fn = data_enhanced_loss_fn
        
        # Train with data-enhanced loss
        try:
            params = self._train_with_weights(n_steps, self.w_mech, self.w_flow, self.w_bc)
        finally:
            # Restore original loss function
            BiotCoupled2D.loss_fn = original_loss_fn
            
        return params


def DataEnhancedTrainer(data_dir="Data_2D", data_weight=1.0, use_data_conditions=None):
    """
    Create a data-enhanced trainer with experimental data integration.
    
    Args:
        data_dir: Directory containing VTK files
        data_weight: Weight for data loss term
        use_data_conditions: List of conditions to use (if None, use all available)
    
    Returns:
        BiotCoupledDataTrainer instance
    """
    return BiotCoupledDataTrainer(
        data_dir=data_dir,
        data_weight=data_weight, 
        use_data_conditions=use_data_conditions,
        auto_balance=True
    )
