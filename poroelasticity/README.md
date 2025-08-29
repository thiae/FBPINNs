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

# For reproducibility, run notebooks directly and change parameters for sensitivity analysis
- Open Notebooks/base_model.ipynb or Notebooks/coupled_model.ipynb
- Each notebook is self contained with setup code

# 1. Clone repository
!git clone https://github.com/thiae/FBPINNs.git

# 2. Navigate and setup
%cd FBPINNs
!python colab_setup.py

# 3. Go to project folder
%cd poroelasticity
!python setup_path.py
!python simple_validation.py

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


