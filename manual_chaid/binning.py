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


def conditions_from_cutpoints(var: str, cutpoints: list, specials: list) -> list:
    """Translate the UI cut-off editor state into ordered bin Conditions.

    ``specials`` is a list of pre-built Condition objects (eq / isnull) that take
    precedence and are emitted first.  ``cutpoints`` is a list of
    ``{"op": <one of OPERATORS>, "value": float}`` rows for the numeric thresholds;
    an ``"="`` row additionally emits a singleton ``eq`` bin.
    """
    conds: list[Condition] = list(specials)

    eq_rows = [c for c in cutpoints if c.get("op") == "="]
    ineq_rows = [c for c in cutpoints if c.get("op") in _OP_LOWER_INCLUSIVE]

    for c in eq_rows:
        conds.append(Condition(var, "eq", value=float(c["value"])))

    # Build boundaries with inclusivity from operators.
    boundaries = sorted(
        ((float(c["value"]), _OP_LOWER_INCLUSIVE[c["op"]]) for c in ineq_rows),
        key=lambda t: t[0],
    )
    if not boundaries:
        if not eq_rows:
            conds.append(Condition(var, "range"))
        return conds

    ts = [b[0] for b in boundaries]
    incl = [b[1] for b in boundaries]
    conds.append(Condition(var, "range", high=ts[0], high_inclusive=incl[0]))
    for k in range(1, len(ts)):
        conds.append(
            Condition(
                var,
                "range",
                low=ts[k - 1],
                low_inclusive=not incl[k - 1],
                high=ts[k],
                high_inclusive=incl[k],
            )
        )
    conds.append(Condition(var, "range", low=ts[-1], low_inclusive=not incl[-1]))
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
