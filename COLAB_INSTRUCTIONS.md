# 🚀 Running FBPINNs Biot Poroelasticity on Google Colab

## Quick Start (Clone and Run)

### Step 1: Clone the Repository
```python
# In a Colab cell:
!git clone https://github.com/thiae/FBPINNs.git
%cd FBPINNs
```

### Step 2: Install Dependencies and FBPINNs
```python
# Install required packages
!pip install jax[cpu] optax matplotlib seaborn

# Install FBPINNs in editable mode (CRITICAL!)
!pip install -e .

# Verify installation
import fbpinns
print("✅ FBPINNs installed successfully")
```

### Step 3: Choose Your Approach

#### Option A: Interactive Notebook (Recommended)
```python
# Navigate to the notebook
%cd poroelasticity/notebooks
# Then open Biot_Visualization_Hub.ipynb
```

#### Option B: Run Validation Scripts (Recommended for GPU)
```python
# Navigate to poroelasticity folder
%cd poroelasticity

# Run comprehensive validation (includes structure check + physics validation)
!python run_all_validations.py --quick

# Or run specific validation
!python validation_scripts/validate_2d_physics.py --quick
```

#### Option C: Run Individual Trainers (Quick Testing)
```python
# Navigate to poroelasticity folder
%cd poroelasticity

# Quick test of physics trainer
!python test/test_biot_2d.py

# Quick test of data trainer  
!python test/test_biot_2d_data.py
```

## What Each Approach Does

### 📓 Interactive Notebook (`Biot_Visualization_Hub.ipynb`)
- **Best for**: Learning, experimentation, parameter studies
- **Features**: Step-by-step visualization, interactive plots, parameter tuning
- **Time**: 10-30 minutes depending on training steps

### ⚡ Comprehensive Validation (`run_all_validations.py`)
- **Best for**: GPU-powered comprehensive testing (recommended)
- **Features**: Structure check + automated physics validation + error metrics + reports
- **Time**: 2-5 minutes with `--quick` flag, 20-30 minutes full training

### 🧪 Individual Tests
- **Best for**: Debugging specific components
- **Features**: Minimal examples, basic functionality checks
- **Time**: 1-2 minutes each

## Expected Results

✅ **Success indicators:**
- No import errors
- Training converges (loss decreases)
- Predictions match exact solution (low L2 error)
- Boundary conditions satisfied
- Physics residuals near zero

⚠️ **If something fails:**
- Check that all dependencies are installed
- Try the `--quick` option for faster testing
- Check the error messages in the output

## File Structure After Cloning
```
FBPINNs/
├── fbpinns/                    # Core framework
└── poroelasticity/            # 🎯 SELF-CONTAINED PROJECT
    ├── trainers/              # Our organized trainers
    │   ├── biot_trainer_2d.py
    │   └── biot_trainer_2d_data.py
    ├── notebooks/             # Interactive development
    ├── validation_scripts/    # Automated testing
    ├── test/                  # Quick functionality tests
    └── utilities/             # Visualization and metrics
```

## Next Steps After Validation

Once everything works on Colab:
1. **Experiment**: Modify parameters in the notebook
2. **Visualize**: Create publication-quality plots
3. **Extend**: Add your own physics variations
4. **Scale Up**: Remove `--quick` flags for full training

---
**That's it!** Your organized structure is ready to work directly when cloned on Colab. No additional setup files needed!
