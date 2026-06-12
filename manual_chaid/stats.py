"""Statistical helpers: Chi-square test for a split and Gini / AUC scoring."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2_contingency


def chi2_from_bins(bin_ids: np.ndarray, y: np.ndarray):
    """Chi-square test of independence between a binning and a binary target.

    Parameters
    ----------
    bin_ids : array of integer bin labels (one per row), values < 0 are dropped.
    y       : binary target (0/1), same length.

    Returns
    -------
    (chi2, pvalue, dof) ; (nan, nan, 0) when the table is degenerate
    (fewer than 2 non-empty bins or fewer than 2 target classes present).
    """
    bin_ids = np.asarray(bin_ids)
    y = np.asarray(y)
    keep = bin_ids >= 0
    bin_ids, y = bin_ids[keep], y[keep]
    if bin_ids.size == 0:
        return float("nan"), float("nan"), 0

    bins = np.unique(bin_ids)
    classes = np.unique(y)
    if bins.size < 2 or classes.size < 2:
        return float("nan"), float("nan"), 0

    table = np.zeros((bins.size, 2), dtype=float)
    bin_index = {b: i for i, b in enumerate(bins)}
    for b, t in zip(bin_ids, y):
        table[bin_index[b], int(t)] += 1

    # Drop empty rows (defensive; unique() should prevent these).
    table = table[table.sum(axis=1) > 0]
    if table.shape[0] < 2:
        return float("nan"), float("nan"), 0

    chi2, p, dof, _ = chi2_contingency(table, correction=False)
    return float(chi2), float(p), int(dof)


def auc_score(scores: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve for continuous ``scores`` vs binary ``y``.

    Computed via the rank-sum (Mann-Whitney U) identity so we depend only on
    numpy.  Returns nan when only one class is present.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y)
    valid = ~np.isnan(scores)
    scores, y = scores[valid], y[valid]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tie groups.
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1

    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def gini_from_scores(scores: np.ndarray, y: np.ndarray) -> float:
    """Gini coefficient = 2 * AUC - 1 (nan when AUC is undefined)."""
    auc = auc_score(scores, y)
    if np.isnan(auc):
        return float("nan")
    return 2.0 * auc - 1.0
