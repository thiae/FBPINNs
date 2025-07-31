# FBPINNs Poroelasticity on Google Colab - Quick Start

This guide helps you get the FBPINNs poroelasticity models running on Google Colab without installation crashes.

## Method 1: Automated Setup (Recommended)

```python
# 1. Clone the repository
!git clone https://github.com/thiae/FBPINNs.git
%cd FBPINNs

# 2. Run automated setup
!python colab_setup.py

# 3. Navigate to poroelasticity and run validation
%cd poroelasticity
!python setup_path.py
!python simple_validation.py
```

## Method 2: Step-by-Step Manual Setup

```python
# 1. Clone and navigate
!git clone https://github.com/thiae/FBPINNs.git
%cd FBPINNs

# 2. Install dependencies manually
!pip install "jax[cuda12]>=0.4.8" optax numpy scipy matplotlib tensorboardX seaborn

# 3. Setup Python path (avoids restart issues)
import sys
from pathlib import Path
fbpinns_root = Path.cwd()
if str(fbpinns_root) not in sys.path:
    sys.path.insert(0, str(fbpinns_root))

# 4. Test the setup
import fbpinns
print("FBPINNs imported successfully!")

# 5. Navigate to poroelasticity
%cd poroelasticity

# 6. Run validation
!python simple_validation.py
```

## Next Steps

Once validation passes, you can:

1. **Run the visualization notebook:**
   ```python
   # Open notebooks/Biot_Visualization_Hub.ipynb
   ```

2. **Run individual tests:**
   ```python
   !python test/test_biot_2d.py
   ```

3. **Run comprehensive validation:**
   ```python
   !python run_all_validations.py --quick
   ```

## Common Issues

### GPU Not Available
If you see "No GPU detected", make sure your Colab runtime is set to GPU:
- Runtime → Change runtime type → Hardware accelerator → GPU

### Import Errors
If you get import errors, make sure you've run the path setup:
```python
%cd poroelasticity
!python setup_path.py
```

### JAX Version Conflicts
If you get JAX conflicts, use the restart-free method:
```python
!python colab_setup.py --method path
```

## Success Indicators

You should see:
- ✓ FBPINNs core modules imported successfully
- ✓ Physics trainer imported successfully  
- ✓ All basic functionality tests passed!
- 🎉 SUCCESS: All validation tests passed!

You're now ready to train poroelasticity models with FBPINNs!
