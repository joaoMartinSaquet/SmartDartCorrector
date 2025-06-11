"""Classic RL Package

This package contains classic reinforcement learning components and correctors
for the SmartDartCorrector project.
"""

# Import main classes and functions for easier access
from .corrector import Corrector, LowPassCorrector
from .policy import REINFORCEnet

__all__ = [
    "Corrector",
    "LowPassCorrector", 
    "REINFORCEnet",
]

