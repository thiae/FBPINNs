"""
Biot Poroelasticity Trainers

This module contains trainer implementations for Biot poroelasticity problems:
- base_model.py: Main implementation with complete functionality
- biot_trainer_2d_data.py: 2D data-enhanced implementation
"""

# Primary import from base_model (main implementation)
from .base_model import BiotCoupled2D, BiotCoupledTrainer

try:
    from .biot_trainer_2d_data import BiotCoupledDataTrainer, VTKDataLoader, DataEnhancedTrainer
    __all__ = ['BiotCoupled2D', 'BiotCoupledTrainer', 
               'BiotCoupledDataTrainer', 'VTKDataLoader', 'DataEnhancedTrainer']
except ImportError:
    # Data trainer optional (requires additional dependencies)
    __all__ = ['BiotCoupled2D', 'BiotCoupledTrainer']
