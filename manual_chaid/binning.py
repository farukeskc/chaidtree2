"""optbinning wrapper.

Turns a (variable, node-subset) into a set of bin *conditions* and the Chi-square
statistic / p-value of the resulting split, honouring the special handling rules:

* ``pre_separate_zero``  -> rows where ``x == 0`` get a forced ``eq 0`` bin and are
  excluded from optbinning.
* ``pre_separate_values``-> rows where ``x == value`` get a forced ``eq value`` bin,
  excluded from optbinning.
* missing values         -> a forced ``IS NULL`` bin, excluded from optbinning.

The remaining rows are binned with :class:`optbinning.OptimalBinning`
(``monotonic_trend="auto_asc_desc"`` so bins are rank-ordered by event rate).
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .node import Condition, OPERATORS
from .stats import chi2_from_bins

DEFAULT_MIN_BIN_SIZE = 0.05
DEFAULT_MAX_N_BINS = 5


def infer_dtype(series: pd.Series) -> str:
    """"numerical" for numeric columns, otherwise "categorical"."""
    return "numerical" if pd.api.types.is_numeric_dtype(series) else "categorical"


def assign_bins(conditions: list, frame: pd.DataFrame) -> np.ndarray:
    """First-match-wins bin assignment.

    Returns an int array of bin indices (the position of the first matching
    condition); rows matching no condition get ``-1`` and are dropped from the
    Chi-square table.  Precedence (special/eq conditions placed first) cleanly
    resolves any overlap between an ``eq`` bin and a numeric range.
    """
    n = len(frame)
    out = np.full(n, -1, dtype=int)
    remaining = np.ones(n, dtype=bool)
    for i, cond in enumerate(conditions):
        if not remaining.any():
            break
        m = cond.evaluate(frame[cond.variable]) & remaining
        out[m] = i
        remaining &= ~m
    return out


def _precedence(cond) -> int:
    """Lower wins. Forced special bins (isnull/eq) must beat overlapping ranges."""
    if cond.kind == "isnull":
        return 0
    if cond.kind == "eq":
        return 1
    if cond.kind == "or":
        return min((_precedence(m) for m in (cond.members or ())), default=3)
    if cond.kind == "category":
        return 2
    return 3  # range


def assign_children_bins(conditions: list, frame: pd.DataFrame) -> np.ndarray:
    """Assign rows to child bins respecting forced-bin precedence.

    Children may be stored in target-rate order, but a row that satisfies a
    forced ``eq``/``isnull`` bin must always be claimed by it rather than by an
    overlapping numeric range.  We therefore evaluate conditions in precedence
    order, then map the result back to the original child positions so split-time
    and reload-time partitions are identical.
    """
    order = sorted(range(len(conditions)), key=lambda i: _precedence(conditions[i]))
    ordered = [conditions[i] for i in order]
    raw = assign_bins(ordered, frame)
    out = np.full(len(frame), -1, dtype=int)
    for ordered_idx, orig_idx in enumerate(order):
        out[raw == ordered_idx] = orig_idx
    return out


def numeric_ranges_from_thresholds(
    var: str, thresholds, inclusive_left: bool = True
) -> list:
    """Build consecutive numeric range Conditions from sorted thresholds.

    Default convention matches optbinning: left-closed, right-open
    -> ``x < t0`` | ``t0 <= x < t1`` | ... | ``x >= t_last``.
    """
    ts = sorted(float(t) for t in thresholds)
    conds: list[Condition] = []
    if not ts:
        conds.append(Condition(var, "range"))  # single bin: any
        return conds
    # first segment
    conds.append(
        Condition(var, "range", high=ts[0], high_inclusive=not inclusive_left)
    )
    for a, b in zip(ts[:-1], ts[1:]):
        conds.append(
            Condition(
                var,
                "range",
                low=a,
                low_inclusive=inclusive_left,
                high=b,
                high_inclusive=not inclusive_left,
            )
        )
    conds.append(Condition(var, "range", low=ts[-1], low_inclusive=inclusive_left))
    return conds


# operator -> whether the threshold value belongs to the LOWER segment
_OP_LOWER_INCLUSIVE = {"<=": True, ">": True, "<": False, ">=": False}


def conditions_from_cutpoints(
    var: str, cutpoints: list, specials: list, descending: bool = False
) -> list:
    """Translate the UI cut-off editor state into ordered bin Conditions.

    ``specials`` is a list of pre-built Condition objects (eq / isnull) that take
    precedence and are emitted first (so forced bins render left-most).
    ``cutpoints`` is a list of ``{"op": <one of OPERATORS>, "value": float}`` rows
    for the numeric thresholds; an ``"="`` row additionally emits a singleton
    ``eq`` bin.

    The numeric range bins are emitted smallest-to-largest by default; when
    ``descending`` is True they are emitted largest-to-smallest, matching the
    direction of the user's (validated monotonic) cut-off list.

    Only the configured ``specials`` (the ``pre_separate_*`` / null bins) are
    pinned left-most.  A manually-added ``"="`` cut-off is kept at its value
    position within the ordered bins -- it is NOT moved to the front.
    """
    conds: list[Condition] = list(specials)

    # Scan cut-offs in ascending value order, building the range bins between
    # consecutive inequality boundaries and inserting manual "=" singleton bins
    # at their value position (precedence in assign_children_bins resolves the
    # overlap between an "=" bin and the range that contains it).
    pts = sorted(cutpoints, key=lambda c: float(c["value"]))
    has_ineq = any(c.get("op") in _OP_LOWER_INCLUSIVE for c in pts)

    rest: list[Condition] = []
    prev_low = None
    prev_low_incl = False
    for c in pts:
        val = float(c["value"])
        op = c.get("op")
        if op == "=":
            rest.append(Condition(var, "eq", value=val))
            continue
        if op not in _OP_LOWER_INCLUSIVE:
            continue
        incl = _OP_LOWER_INCLUSIVE[op]  # does ``val`` belong to the lower segment?
        rest.append(
            Condition(
                var,
                "range",
                low=prev_low,
                low_inclusive=prev_low_incl,
                high=val,
                high_inclusive=incl,
            )
        )
        prev_low, prev_low_incl = val, not incl
    if has_ineq:
        rest.append(Condition(var, "range", low=prev_low, low_inclusive=prev_low_incl))
    else:
        # no inequality boundaries -> one catch-all bin for everything else
        rest.append(Condition(var, "range"))

    if descending:
        rest.reverse()
    conds.extend(rest)
    return conds


def categorical_conditions_from_groups(var: str, items: list, specials: list) -> list:
    """Build category bin Conditions from a (category, group-id) assignment.

    ``items`` is a list of ``(category, group_id)`` pairs from the cut-off editor;
    categories sharing a group id are merged into one bin.  Configured ``specials``
    stay left-most; the category groups follow, ordered by group id (numeric when
    possible, else lexicographic).
    """
    conds: list[Condition] = list(specials)
    groups: dict = {}
    for cat, gid in items:
        groups.setdefault(gid, []).append(cat)

    def key(g):
        try:
            return (0, float(g))
        except (TypeError, ValueError):
            return (1, str(g))

    for gid in sorted(groups, key=key):
        conds.append(Condition(var, "category", categories=tuple(groups[gid])))
    return conds


def _special_conditions(
    var: str, x: pd.Series, separate_zero: bool, separate_value, has_null: bool
) -> list:
    conds: list[Condition] = []
    if has_null and x.isna().any():
        conds.append(Condition(var, "isnull"))
    if separate_zero:
        conds.append(Condition(var, "eq", value=0))
    if separate_value is not None:
        conds.append(Condition(var, "eq", value=separate_value))
    return conds


def compute_candidate(
    x: pd.Series,
    y: np.ndarray,
    *,
    separate_zero: bool = False,
    separate_value=None,
    min_bin_size: float = DEFAULT_MIN_BIN_SIZE,
    max_n_bins: int = DEFAULT_MAX_N_BINS,
) -> dict:
    """Compute the optbinning-suggested split for one variable on a node subset.

    Returns a dict with keys: ``variable``, ``conditions`` (list[Condition],
    specials first then optbinning bins), ``chi2``, ``pvalue``, ``splittable``
    (p < 0.05), ``dtype`` and optional ``error``.
    """
    from optbinning import OptimalBinning  # imported lazily; heavy dependency

    var = x.name
    dtype = infer_dtype(x)
    frame = pd.DataFrame({var: x.values})

    specials = _special_conditions(
        var, x, separate_zero, separate_value, x.isna().any()
    )

    # Mask of rows handled by special bins (excluded from optbinning).
    special_mask = np.zeros(len(x), dtype=bool)
    for c in specials:
        special_mask |= c.evaluate(x)

    fit_mask = ~special_mask & x.notna().to_numpy()
    x_fit = x[fit_mask]
    y_fit = np.asarray(y)[fit_mask]

    opt_conditions: list[Condition] = []
    error: Optional[str] = None

    if x_fit.nunique(dropna=True) >= 2 and len(np.unique(y_fit)) == 2:
        try:
            optb = OptimalBinning(
                name=str(var),
                dtype=dtype,
                min_bin_size=min_bin_size,
                max_n_bins=max_n_bins,
                monotonic_trend="auto_asc_desc",
            )
            with warnings.catch_warnings():
                # optbinning calls a deprecated sklearn arg on 1.6/1.7.
                warnings.simplefilter("ignore", FutureWarning)
                if dtype == "numerical":
                    optb.fit(x_fit.to_numpy(dtype=float), y_fit)
                    opt_conditions = numeric_ranges_from_thresholds(
                        var, list(np.asarray(optb.splits, dtype=float))
                    )
                else:
                    optb.fit(x_fit.astype(object).to_numpy(), y_fit)
                    opt_conditions = _categorical_conditions(optb, var, x_fit)
        except Exception as exc:  # optbinning can raise on degenerate inputs
            error = f"optbinning failed: {exc}"
    else:
        error = "not enough distinct values / target classes to bin"

    conditions = specials + opt_conditions
    if len(conditions) < 2:
        return {
            "variable": var,
            "conditions": conditions,
            "chi2": float("nan"),
            "pvalue": float("nan"),
            "splittable": False,
            "dtype": dtype,
            "error": error or "only one resulting bin",
        }

    bin_ids = assign_bins(conditions, frame)
    chi2, pval, _ = chi2_from_bins(bin_ids, y)
    return {
        "variable": var,
        "conditions": conditions,
        "chi2": chi2,
        "pvalue": pval,
        "splittable": (not np.isnan(pval)) and pval < 0.05,
        "dtype": dtype,
        "error": error,
    }


def _categorical_conditions(optb, var: str, x_fit: pd.Series) -> list:
    """Group categories by their optbinning bin index into category Conditions."""
    cats = pd.unique(x_fit.dropna().astype(object))
    if len(cats) == 0:
        return []
    idx = optb.transform(np.asarray(cats, dtype=object), metric="indices")
    groups: dict[int, list] = {}
    for c, i in zip(cats, idx):
        groups.setdefault(int(i), []).append(c)
    return [
        Condition(var, "category", categories=tuple(groups[k]))
        for k in sorted(groups)
    ]
