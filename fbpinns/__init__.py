"""
FBPINNs - Finite Basis Physics-Informed Neural Networks
========================================================

A framework for solving partial differential equations using physics-informed 
neural networks with finite basis methods.
"""

# Import main modules to make them available at package level
from . import domains
from . import networks  
from . import problems
from . import trainers
from . import constants

__version__ = "1.0.0"
__all__ = ['domains', 'networks', 'problems', 'trainers', 'constants']