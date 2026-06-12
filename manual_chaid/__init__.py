"""ManualChaidTree -- interactive manual CHAID decision-tree builder."""
from .node import ChaidNode, Condition
from .tree import ManualChaidTree

__all__ = ["ManualChaidTree", "ChaidNode", "Condition"]
__version__ = "0.1.0"
