"""
Biot Poroelasticity Trainers

This module contains trainer implementations for Biot poroelasticity problems:
- biot_trainer_2d.py: 2D physics-only implementation
- biot_trainer_2d_data.py: 2D data-enhanced implementation
"""

from .biot_trainer_2d import BiotCoupled2D, BiotCoupledTrainer, CoupledTrainer

try:
    from .biot_trainer_2d_data import BiotCoupledDataTrainer, VTKDataLoader, DataEnhancedTrainer
    __all__ = ['BiotCoupled2D', 'BiotCoupledTrainer', 'CoupledTrainer', 
               'BiotCoupledDataTrainer', 'VTKDataLoader', 'DataEnhancedTrainer']
except ImportError:
    # Data trainer optional (requires additional dependencies)
    __all__ = ['BiotCoupled2D', 'BiotCoupledTrainer', 'CoupledTrainer']
