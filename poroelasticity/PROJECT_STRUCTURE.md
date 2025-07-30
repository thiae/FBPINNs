# Biot Poroelasticity Project Structure

## 📁 Directory Organization

```
poroelasticity/                             # 🎯 SELF-CONTAINED PROJECT
├── 📓 notebooks/                           # Interactive development ✅
│   └── Biot_Visualization_Hub.ipynb       # Comprehensive exploration notebook
├── 🧪 validation_scripts/                   # Clean validation scripts
│   ├── validate_2d_physics.py              # 2D physics-only validation ✅
│   ├── validate_2d_data.py                 # 2D data-enhanced validation (TODO)
│   ├── validate_3d_physics.py              # 3D physics-only validation (FUTURE)
│   └── validate_3d_data.py                 # 3D data-enhanced validation (FUTURE)
├── 🏗️ trainers/                            # Core trainer implementations ✅
│   ├── __init__.py                          # Module initialization
│   ├── biot_trainer_2d.py                  # 2D physics-only ✅
│   ├── biot_trainer_2d_data.py             # 2D data-enhanced ✅
│   ├── biot_trainer_3d.py                  # 3D physics-only (FUTURE)
│   └── biot_trainer_3d_data.py             # 3D data-enhanced (FUTURE)
├── 🧪 test/                                # Development tests ✅
│   ├── test_biot_2d.py                     # 2D physics tests
│   └── test_biot_2d_data.py                # 2D data tests
│   └── test_biot_2d_data.py                # 2D data tests
├── 🔧 utilities/                           # Shared utilities ✅
│   ├── visualization_tools.py              # Common plotting functions
│   ├── validation_metrics.py               # Error analysis tools
│   └── data_loaders.py                     # VTK and other data loaders (TODO)
├── 📊 results/                             # Generated validation outputs
│   ├── 2d_physics_validation/
│   ├── 2d_data_validation/
│   └── comparative_analysis/
├── 📋 run_all_validations.py               # Master script for instructors ✅
└── 📄 This README file
```

## 🎯 Usage Instructions

### For Students (Interactive Development)
```bash
# Open the exploration notebook
jupyter notebook notebooks/Biot_Visualization_Hub.ipynb
```

### For Developers (Quick Testing)
```bash
# Run development tests
python test/test_biot_2d.py
python test/test_biot_2d_data.py
```

### For Instructors (Automated Validation)
```bash
# Run all available validations
python run_all_validations.py

# Quick validation (reduced training)
python run_all_validations.py --quick

# Run specific model only
python run_all_validations.py --models 2d_physics
```

### For Individual Validation
```bash
# Validate 2D physics-only model
python validation_scripts/validate_2d_physics.py

# Quick validation
python validation_scripts/validate_2d_physics.py --quick

# Custom save directory
python validation_scripts/validate_2d_physics.py --save-dir my_results
```

## 📊 Current Status

### ✅ Completed
- 2D physics-only trainer (`trainers/biot_trainer_2d.py`)
- 2D data-enhanced trainer (`trainers/biot_trainer_2d_data.py`)
- Shared visualization utilities
- Shared validation metrics
- 2D physics validation script
- Master validation script
- Interactive exploration notebook

### 🚧 In Progress
- 2D data-enhanced validation script
- VTK data loader utilities

### 🔮 Future Work
- 3D physics implementation
- 3D data-enhanced implementation
- Advanced parameter studies
- Real-time training monitoring

## 🏗️ Design Principles

1. **Separation of Concerns**: Interactive notebook for exploration, clean scripts for submission
2. **Modular Design**: Shared utilities avoid code duplication
3. **Scalable Structure**: Easy to add new dimensions (3D) or approaches
4. **Professional Output**: Clean validation reports for academic submission
5. **User-Friendly**: Clear documentation and usage instructions

## 🎓 Academic Submission

For course submission, instructors should run:
```bash
python run_all_validations.py
```

This will:
- Check all dependencies
- Run all available validations
- Generate comprehensive reports
- Save results with plots and metrics
- Provide clear pass/fail status

Results are saved in the `results/` directory with:
- Validation plots (solution fields, error analysis)
- Numerical metrics (JSON format)
- Text summaries
- Master validation report

## 🚀 Quick Start

**For GPU/Colab Users (Recommended):**
```bash
# One command does everything: structure + physics validation
python run_all_validations.py --quick      # Quick test (2-5 min)
python run_all_validations.py             # Full test (20-30 min)
```

**For Interactive Development:**
```bash
# Explore and experiment
jupyter notebook notebooks/Biot_Visualization_Hub.ipynb
```

**For Component Testing:**
```bash
# Test individual parts
python test/test_biot_2d.py                           # Basic functionality
python validation_scripts/validate_2d_physics.py --quick  # Physics only
```

The project is designed to be both powerful for development and clean for academic submission!
