"""
Biot Poroelasticity Trainers

This module contains trainer implementations for Biot poroelasticity problems:
- base_model.py: Main implementation with complete functionality
- biot_trainer_2d_data.py: 2D data-enhanced implementation
"""

# Primary imports from trainers package
from .base_model import BiotCoupled2D, BiotCoupledTrainer

__all__ = ['BiotCoupled2D', 'BiotCoupledTrainer']

# Optional extras 
try:
    from .coupled_model import (
        BiotCoupled2D_Heterogeneous,
        BiotCoupledTrainer_Heterogeneous,
        FixedTrainer,
    )
    __all__ += [
        'BiotCoupled2D_Heterogeneous',
        'BiotCoupledTrainer_Heterogeneous',
        'FixedTrainer',
    ]
except Exception:
    pass
