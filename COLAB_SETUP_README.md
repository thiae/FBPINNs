# FBPINNs Google Colab Setup

This repository includes automated setup scripts to make installing FBPINNs on Google Colab as smooth as possible, avoiding the common restart issues.

## Quick Start

```python
# In Google Colab:
!git clone https://github.com/thiae/FBPINNs.git
%cd FBPINNs
!python colab_setup.py
!python quick_test.py
```

That's it! If the test passes, you're ready to train models and run validations.

## Files Added for Colab

- **`colab_setup.py`** - Automated installation script with restart-free options
- **`quick_test.py`** - Verification script to test your installation
- **Updated `COLAB_INSTRUCTIONS.md`** - Comprehensive setup guide

## Why These Scripts?

The main issue with FBPINNs on Colab is that `!pip install -e .` often triggers a runtime restart because of JAX dependency conflicts. When this happens:

1. You lose your GPU connection
2. Colab takes time to reconnect to GPU  
3. You waste time waiting
4. Sometimes the installation still doesn't work after restart

Our solution provides multiple installation methods:
- **Automated**: Tries the best method for your situation
- **Path-based**: Adds FBPINNs to Python path (no pip install, restart-free)
- **Traditional**: Standard pip install with better dependency management

## Usage Examples

### Physics-Only Training
```python
%cd FBPINNs/poroelasticity
!python test/test_biot_2d.py
```

### Data-Enhanced Training  
```python
%cd FBPINNs/poroelasticity
!python test/test_biot_2d_data.py
```

### Comprehensive Validation
```python
%cd FBPINNs/poroelasticity
!python run_all_validations.py --quick
```

### Interactive Notebook
```python
# Open: FBPINNs/poroelasticity/notebooks/Biot_Visualization_Hub.ipynb
```

## Troubleshooting

If you encounter issues, the setup script provides detailed error messages and suggestions. You can also run:

```python
!python colab_setup.py --method path  # Force restart-free method
!python colab_setup.py --method pip   # Force traditional pip method  
!python quick_test.py                 # Test your installation
```

The scripts are designed to be robust and provide clear feedback about what's working and what isn't.
