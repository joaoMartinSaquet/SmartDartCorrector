"""Classic RL Package

This package contains classic reinforcement learning components and correctors
for the SmartDartCorrector project.
"""

# Import main classes and functions for easier access
from .rl_corrector import LowPassCorrector, ReinforceCorrector
from .policy import REINFORCEnet

__all__ = [
    "LowPassCorrector", 
    "REINFORCEnet",
]

