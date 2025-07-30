"""
Test script to verify VTK data loading functionality
"""

import numpy as np
from pathlib import Path
import sys
import os

# Add the parent directory to Python path to find fbpinns
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def parse_vtk_file(filepath):
    """
    Simplified VTK parser for testing data loading.
    """
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
        raise ValueError(f"No POINTS section found")
    
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
        raise ValueError(f"No data section found")
    
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
        "num_points": num_points
    }


def test_vtk_loading():
    """
    Test loading of VTK experimental data files.
    """
    data_dir = Path("Data_2D")
    
    # Define expected file patterns
    file_patterns = [
        "displacement_MSAMPLE2D_RES_S0_M.vtk",
        "displacement_MSAMPLE2D_RES_S100_MHm.vtk", 
        "matrix_pressure_MSAMPLE2D_RES_S100_Hm.vtk",
        "matrix_pressure_MSAMPLE2D_RES_S100_MHm.vtk"
    ]
    
    loaded_data = {}
    
    for pattern in file_patterns:
        filepath = data_dir / pattern
        if filepath.exists():
            try:
                data = parse_vtk_file(filepath)
                loaded_data[pattern] = data
                
                print(f"✓ Loaded {pattern}:")
                print(f"  - Points: {data['num_points']}")
                print(f"  - Data type: {data['data_type']}")
                print(f"  - Coordinate range: ({data['coordinates'].min(axis=0)}, {data['coordinates'].max(axis=0)})")
                
                if data['data_type'] == 'vectors':
                    print(f"  - Displacement range: ({data['data'].min(axis=0)}, {data['data'].max(axis=0)})")
                else:
                    print(f"  - Pressure range: ({data['data'].min():.2e}, {data['data'].max():.2e})")
                print()
                
            except Exception as e:
                print(f"✗ Error loading {pattern}: {e}")
        else:
            print(f"✗ File not found: {filepath}")
    
    # Summary
    print(f"Successfully loaded {len(loaded_data)}/{len(file_patterns)} VTK files")
    
    # Check coordinate consistency
    if loaded_data:
        coords_arrays = [data['coordinates'] for data in loaded_data.values()]
        coords_consistent = all(
            np.allclose(coords_arrays[0], coords) 
            for coords in coords_arrays[1:]
        )
        print(f"Coordinate consistency across files: {'✓' if coords_consistent else '✗'}")
    
    return loaded_data


if __name__ == "__main__":
    print("Testing VTK data loading functionality...")
    print("=" * 50)
    
    loaded_data = test_vtk_loading()
    
    if loaded_data:
        print("\nData loading test completed successfully!")
        
        # Show some sample data points
        sample_file = list(loaded_data.keys())[0]
        sample_data = loaded_data[sample_file]
        
        print(f"\nSample from {sample_file}:")
        print("First 3 coordinates:")
        print(sample_data['coordinates'][:3])
        
        if sample_data['data_type'] == 'vectors':
            print("First 3 displacement vectors:")
            print(sample_data['data'][:3])
        else:
            print("First 3 pressure values:")
            print(sample_data['data'][:3])
    else:
        print("\nNo data files were loaded successfully.")
        print("Make sure the VTK files are in the Data_2D directory.")
