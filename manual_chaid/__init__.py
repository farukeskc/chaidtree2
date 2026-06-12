"""ManualChaidTree -- interactive manual CHAID decision-tree builder."""
from .node import ChaidNode, Condition
from .optimizer import ChaidTreeOptimizer, OptimizerSpec
from .tree import ManualChaidTree

__all__ = [
    "ManualChaidTree",
    "ChaidNode",
    "Condition",
    "ChaidTreeOptimizer",
    "OptimizerSpec",
]
__version__ = "0.1.0"
