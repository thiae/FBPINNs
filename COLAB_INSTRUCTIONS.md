# Running FBPINNs Biot Poroelasticity on Google Colab

## Quick Start (Clone and Run)

### Step 1: Clone the Repository
```python
# In a Colab cell:
!git clone https://github.com/thiae/FBPINNs.git
%cd FBPINNs
```

### Step 2: Install Dependencies and FBPINNs

**RECOMMENDED: Use the automated setup script**
```python
# Navigate to FBPINNs directory
%cd /content/FBPINNs

# Run the setup script (handles everything automatically)
!python colab_setup.py

# Quick test to verify everything works
!python quick_test.py

# That's it! If tests pass, you're ready to go!
```

**MANUAL METHOD (if you prefer step-by-step):**

**Cell 1 - Install JAX first (may trigger restart):**
```python
# For GPU support on Colab (recommended):
!pip install "jax[cuda12]" optax

# For CPU-only (if GPU issues):
# !pip install "jax[cpu]" optax

# If you see a restart warning, click "Restart session" and run Cell 2
# If no restart warning, continue directly to Cell 2
```

**Cell 2 - Install remaining dependencies and FBPINNs:**
```python
# Navigate to the repo (run this even after restart)
%cd /content/FBPINNs

# Install remaining packages (these rarely cause restarts)
!pip install matplotlib seaborn numpy scipy tensorboardX ipython

# Install FBPINNs in editable mode (CRITICAL!)
!pip install -e .

# Verify installation immediately
import fbpinns
print("FBPINNs installed successfully")
print("Ready to train models!")
```

**RESTART-FREE METHOD (if restarts are problematic):**
```python
# Navigate to repo
%cd /content/FBPINNs

# Use the path-based setup (no pip install of FBPINNs)
!python colab_setup.py --method path

# This avoids pip install -e . which can trigger restarts
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

### Interactive Notebook (`Biot_Visualization_Hub.ipynb`)
- **Best for**: Learning, experimentation, parameter studies
- **Features**: Step-by-step visualization, interactive plots, parameter tuning
- **Time**: 10-30 minutes depending on training steps

### Comprehensive Validation (`run_all_validations.py`)
- **Best for**: GPU-powered comprehensive testing (recommended)
- **Features**: Structure check + automated physics validation + error metrics + reports
- **Time**: 2-5 minutes with `--quick` flag, 20-30 minutes full training

### Individual Tests
- **Best for**: Debugging specific components
- **Features**: Minimal examples, basic functionality checks
- **Time**: 1-2 minutes each

## Expected Results

Success indicators:
- No import errors
- Training converges (loss decreases)
- Predictions match exact solution (low L2 error)
- Boundary conditions satisfied
- Physics residuals near zero

If something fails:
- Check that all dependencies are installed
- Try the `--quick` option for faster testing
- Check the error messages in the output

## Troubleshooting Common Issues

### "Restart session" Warning
This happens because JAX installation can conflict with pre-installed packages.

**Best Practice:**
1. Install JAX first in its own cell
2. If restart is required, restart immediately 
3. Navigate back to directory: `%cd /content/FBPINNs`
4. Install remaining packages and FBPINNs
5. Test import immediately

### Import Errors
If you see `ModuleNotFoundError: No module named 'fbpinns'`:

**Option 1 - Use the setup script:**
```python
%cd /content/FBPINNs
!python colab_setup.py --method auto
```

**Option 2 - Manual reinstall:**
```python
%cd /content/FBPINNs
!pip install -e .
import fbpinns  # Test immediately
```

**Option 3 - Use Python Path (no pip install):**
```python
%cd /content/FBPINNs
!python colab_setup.py --method path
```

### GPU Connection Issues
If GPU takes too long to reconnect after restart:
- Use the "Alternative Method" above (Python path instead of pip install)
- This avoids the need for `pip install -e .` which can trigger restarts

### Training Issues
If training fails or gives strange results:
- Try `--quick` flag first for faster debugging
- Check GPU is enabled: `Runtime > Change runtime type > GPU`
- Verify JAX can see GPU: `import jax; print(jax.devices())`

## File Structure After Cloning
```
FBPINNs/
├── fbpinns/                    # Core framework
└── poroelasticity/            # SELF-CONTAINED PROJECT
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

## Summary: Three Setup Approaches

### Automated Setup (RECOMMENDED)
```python
%cd /content/FBPINNs
!python colab_setup.py
!python quick_test.py
```
- **Pros**: Handles everything automatically, detects issues, restart-free fallback
- **Best for**: First-time users, when you want reliability

### Manual Step-by-Step  
```python
!pip install "jax[cuda12]" optax  # May trigger restart
%cd /content/FBPINNs             # Run after restart if needed
!pip install matplotlib seaborn numpy scipy tensorboardX ipython
!pip install -e .
```
- **Pros**: Full control, traditional pip install
- **Cons**: May require restart, GPU reconnection time
- **Best for**: When you understand the process and want pip install

### Path-Based (RESTART-FREE)
```python
%cd /content/FBPINNs
!python colab_setup.py --method path
```
- **Pros**: Never triggers restarts, fastest
- **Cons**: Non-standard installation method
- **Best for**: When restarts are problematic, quick testing
