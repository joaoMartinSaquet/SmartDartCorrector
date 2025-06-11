"""Classic RL Package

This package contains classic reinforcement learning components and correctors
for the SmartDartCorrector project.
"""

# Import main classes and functions for easier access
from .corrector import Corrector, LowPassCorrector
from .deep_stuff import networks

__all__ = [
    "Corrector",
    "LowPassCorrector", 
    "networks",
]

