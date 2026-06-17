"""Node and condition (rule) model for ManualChaidTree.

A node stores only the *rule* that carves it from its parent plus cached
statistics -- never the underlying data rows.  That keeps pickled trees small
and data-free, and lets a tree be reloaded either with data (editable) or
without (view / score only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# Operators offered in the cut-off editor.
OPERATORS = ["<=", "<", ">", ">=", "="]


@dataclass
class Condition:
    """A single-variable predicate describing how a child is split from a parent.

    kind:
      - "range"    : numeric interval (low, high] with explicit inclusivity.
                     low/high == None means -inf / +inf.
      - "category" : categorical membership in ``categories``.
      - "eq"       : exact equality ``variable == value`` (covers the "==0" and
                     pre-separated-value bins).
      - "isnull"   : ``variable`` is missing.
    """

    variable: str
    kind: str
    low: Optional[float] = None
    low_inclusive: bool = False
    high: Optional[float] = None
    high_inclusive: bool = True
    categories: Optional[tuple] = None
    value: Any = None
    members: Optional[list] = None  # for kind == "or": list[Condition]

    # -- evaluation -----------------------------------------------------------
    def evaluate(self, series: pd.Series) -> np.ndarray:
        """Boolean mask of rows in ``series`` that satisfy this condition.

        Missing values never satisfy range / category / eq conditions; they are
        captured only by an explicit ``isnull`` condition.
        """
        if self.kind == "or":
            mask = np.zeros(len(series), dtype=bool)
            for m in (self.members or ()):
                mask |= m.evaluate(series)
            return mask
        if self.kind == "isnull":
            return series.isna().to_numpy()

        notna = series.notna().to_numpy()
        if self.kind == "range":
            vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            mask = notna & ~np.isnan(vals)
            if self.low is not None:
                mask &= (vals >= self.low) if self.low_inclusive else (vals > self.low)
            if self.high is not None:
                mask &= (vals <= self.high) if self.high_inclusive else (vals < self.high)
            return mask
        if self.kind == "category":
            cats = set(self.categories or ())
            return notna & series.isin(cats).to_numpy()
        if self.kind == "eq":
            # Numeric-safe equality (handles 0 stored as 0.0 etc.).
            try:
                return notna & (series == self.value).to_numpy()
            except Exception:
                return notna & (series.astype(str) == str(self.value)).to_numpy()
        raise ValueError(f"Unknown condition kind: {self.kind!r}")

    # -- presentation ---------------------------------------------------------
    def describe(self) -> str:
        v = self.variable
        if self.kind == "or":
            return " OR ".join(m.describe() for m in (self.members or ()))
        if self.kind == "isnull":
            return f"{v} IS NULL"
        if self.kind == "eq":
            return f"{v} = {self.value}"
        if self.kind == "category":
            cats = ", ".join(str(c) for c in (self.categories or ()))
            return f"{v} in [{cats}]"
        # range
        lo_op = "<=" if self.low_inclusive else "<"
        hi_op = "<=" if self.high_inclusive else "<"
        if self.low is None and self.high is not None:
            return f"{v} {hi_op} {_fmt(self.high)}"
        if self.high is None and self.low is not None:
            inv = ">" if self.low_inclusive else ">="
            # low_inclusive means lower bound is inclusive -> var >= low
            inv = ">=" if self.low_inclusive else ">"
            return f"{v} {inv} {_fmt(self.low)}"
        if self.low is None and self.high is None:
            return f"{v}: any"
        return f"{_fmt(self.low)} {lo_op} {v} {hi_op} {_fmt(self.high)}"

    def describe_value(self) -> str:
        """The condition without the variable name (the parent already shows it).

        Used for child-node labels, e.g. "= -1", "< 1.5", "[1.5, 2.5)".
        """
        if self.kind == "or":
            return " OR ".join(m.describe_value() for m in (self.members or ()))
        if self.kind == "isnull":
            return "IS NULL"
        if self.kind == "eq":
            return f"= {self.value}"
        if self.kind == "category":
            return "in [" + ", ".join(str(c) for c in (self.categories or ())) + "]"
        # range
        if self.low is None and self.high is not None:
            return f"{'<=' if self.high_inclusive else '<'} {_fmt(self.high)}"
        if self.high is None and self.low is not None:
            return f"{'>=' if self.low_inclusive else '>'} {_fmt(self.low)}"
        if self.low is None and self.high is None:
            return "any"
        lo_br = "[" if self.low_inclusive else "("
        hi_br = "]" if self.high_inclusive else ")"
        return f"{lo_br}{_fmt(self.low)}, {_fmt(self.high)}{hi_br}"


def _fmt(x: float) -> str:
    if x is None:
        return ""
    if float(x).is_integer():
        return str(int(x))
    return repr(float(x))


@dataclass
class ChaidNode:
    """A single node of the manually-grown tree (structure + cached stats)."""

    node_id: int
    parent_id: Optional[int]
    depth: int
    condition: Optional[Condition] = None  # rule vs parent; None for root
    children: list = field(default_factory=list)  # list[int] of child node_ids

    # cached statistics (always present, survive a data-free reload)
    n_samples: int = 0
    n_target: int = 0
    target_rate: float = 0.0

    # split info -- only on internal (split) nodes
    split_variable: Optional[str] = None
    split_chi2: Optional[float] = None
    split_pvalue: Optional[float] = None
    split_conditions: Optional[list] = None  # list[Condition] used to split

    @property
    def is_leaf(self) -> bool:
        return not self.children
