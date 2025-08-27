#  Independent Research Project: Coupled Geomechanics and Fluid Flow Simulation for CO₂ Storage in Heterogeneous Reservoirs using Finite Basis Pinns (FBPINNs)

## Project Description
This project investigates the application of Finite Basis Physics-Informed Neural Networks (PINNs) on Biot poroelasticity equations in heterogeneous reservoirs, with a focus on CO₂ storage applications. The aim is to adapt Dr. Ben Moseley’s FBPINN framework to predict pore pressure and displacement fields using a JAX-based implementation.

## Author
Ogechi Cynthia Eze 
Geo-Energy with Machine Learning and Data Science
Imperial College London
29th - August - 2025

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Git


## Installation and Setup

### Important Note on Package Installation
⚠️ **Do NOT use `pip install -e .` with this project.** The FBPINNs framework has compatibility issues with editable installs in Colab that cause runtime restarts. The path based setup below has been thoroughly tested and works reliably.

### Working Setup (Tested on Google Colab)

# Cell 1: Clone and initial setup
!git clone https://github.com/thiae/FBPINNs.git
%cd FBPINNs
!python colab_setup.py (Automatically uses method = 'auto' which defaults to 'path')

# Cell 2: Navigate to project and setup paths
%cd poroelasticity
!python setup_path.py

# Cell 3: Verify paths are set (optional but recommended)
import sys
from pathlib import Path
current_dir = Path.cwd()
fbpinns_root = current_dir.parent
print(f"Current: {current_dir}")
print(f"FBPINNs root: {fbpinns_root}")
print(f"Path configured: {str(fbpinns_root) in sys.path}")

# Cell 4: Validate
!python simple_validation.py

If validation passes, you're ready to run the notebooks.

# Running the Notebooks
After setup, the notebooks in Notebooks/ can be run directly:

base_model.ipynb - Single physics validation
coupled_model.ipynb - Full coupled simulation

Note: The notebooks contain additional path setup code for robustness. This is intentional redundancy to ensure imports work even if cells are run out of order.

## Project Structure
Despite using path based imports, this project maintains proper package structure:
poroelasticity/
├── Data_2D/                 # Data files for simulations
├── Notebooks/               # Jupyter notebooks with examples
│   ├── base_model.ipynb   # Single physics validation
│   └── coupled_model.ipynb # Coupled flow-mechanics simulation
├── trainers/                # Core implementation
│   ├── init.py
│   ├── base_model.py      # Base Biot trainer
│   └── coupled_model.py   # Coupled data trainer
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
├── setup_path.py          # Path setup for Colab
└── README.md              # This file


