"""ManualChaidTree -- headless core logic for the manual CHAID tree builder.

All domain operations live here and work without any UI, so a tree can be
reloaded without data for viewing / scoring, or with data for further editing.
"""
from __future__ import annotations

import copy
import pickle
from typing import Optional

import numpy as np
import pandas as pd

from .binning import (
    DEFAULT_MAX_N_BINS,
    DEFAULT_MIN_BIN_SIZE,
    assign_bins,
    compute_candidate,
)
from .node import ChaidNode, Condition
from .stats import chi2_from_bins, gini_from_scores


class ManualChaidTree:
    """Manually-grown CHAID tree for a binary (0/1) target."""

    # ------------------------------------------------------------------ setup
    def __init__(self, X: pd.DataFrame, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        y_arr = np.asarray(pd.Series(y).values)
        uniq = set(pd.unique(y_arr[~pd.isna(y_arr)]))
        if not uniq.issubset({0, 1}) or len(uniq) < 2:
            raise ValueError(f"y must be binary 0/1 with both classes present; got {uniq}")

        self.X = X.reset_index(drop=True)
        self.y = y_arr.astype(int)
        self._has_data = True

        # config (populated by train / defaults)
        self.pre_separate_zero: list = []
        self.pre_separate_values: dict = {}
        self.min_samples: Optional[int] = None
        self.max_children: Optional[int] = None

        self.nodes: dict[int, ChaidNode] = {}
        self._masks: dict[int, np.ndarray] = {}
        self._next_id = 0
        self.history: list[str] = []
        self._undo_stack: list = []

        self._init_root()

    def _init_root(self):
        root = ChaidNode(node_id=0, parent_id=None, depth=0, condition=None)
        self.nodes = {0: root}
        self._masks = {0: np.ones(len(self.y), dtype=bool)}
        self._next_id = 1
        self._compute_stats(0)

    # ------------------------------------------------------------- public API
    def train(
        self,
        pre_separate_zero=None,
        pre_separate_values=None,
        min_samples=None,
        max_children=None,
        launch_ui: bool = True,
    ):
        """Store configuration and (optionally) launch the tkinter builder UI."""
        self.pre_separate_zero = list(pre_separate_zero or [])
        self.pre_separate_values = dict(pre_separate_values or {})
        self.min_samples = min_samples
        self.max_children = max_children
        if launch_ui:
            from .ui import TreeBuilderUI

            TreeBuilderUI(self).run()
        return self

    # -- node access ----------------------------------------------------------
    def get_node(self, node_id: int) -> ChaidNode:
        return self.nodes[node_id]

    def leaves(self) -> list:
        return [n for n in self.nodes.values() if n.is_leaf]

    def internal_nodes(self) -> list:
        return [n for n in self.nodes.values() if not n.is_leaf]

    def depth(self) -> int:
        return max((n.depth for n in self.nodes.values()), default=0)

    def _require_data(self):
        if not self._has_data:
            raise RuntimeError(
                "This tree was loaded without data; it is view/score-only. "
                "Reload with X and y to edit it."
            )

    # -- stats ----------------------------------------------------------------
    def _compute_stats(self, node_id: int):
        node = self.nodes[node_id]
        mask = self._masks[node_id]
        n = int(mask.sum())
        n_t = int(self.y[mask].sum()) if n else 0
        node.n_samples = n
        node.n_target = n_t
        node.target_rate = (n_t / n) if n else 0.0

    def _binning_params(self, n_rows: int):
        if self.min_samples:
            mbs = min(max(self.min_samples / max(n_rows, 1), 1e-6), 0.5)
        else:
            mbs = DEFAULT_MIN_BIN_SIZE
        mnb = self.max_children or DEFAULT_MAX_N_BINS
        return mbs, mnb

    # -- candidate splits -----------------------------------------------------
    def candidate_splits(self, node_id: int) -> list:
        """Chi2-sorted candidate splits for every X column on this leaf's rows."""
        self._require_data()
        mask = self._masks[node_id]
        idx = np.where(mask)[0]
        sub = self.X.iloc[idx]
        y_sub = self.y[idx]
        mbs, mnb = self._binning_params(len(idx))

        results = []
        for col in self.X.columns:
            res = compute_candidate(
                sub[col],
                y_sub,
                separate_zero=col in self.pre_separate_zero,
                separate_value=self.pre_separate_values.get(col),
                min_bin_size=mbs,
                max_n_bins=mnb,
            )
            results.append(res)

        def sort_key(r):
            c = r["chi2"]
            return -1e18 if (c is None or np.isnan(c)) else -c

        results.sort(key=sort_key)
        return results

    # -- split ----------------------------------------------------------------
    def split(self, node_id: int, variable: str, conditions: list) -> dict:
        """Split a leaf using the given (possibly hand-edited) bin conditions.

        Chi2/p are *recomputed from these conditions* before being stored.
        """
        self._require_data()
        node = self.nodes[node_id]
        if not node.is_leaf:
            raise ValueError(f"Node {node_id} is not a leaf")
        if len(conditions) < 2:
            raise ValueError("Need at least two bins to split")

        idx = np.where(self._masks[node_id])[0]
        sub = self.X.iloc[idx]
        y_sub = self.y[idx]
        bin_ids = assign_bins(conditions, sub)
        chi2, pval, _ = chi2_from_bins(bin_ids, y_sub)

        # Build child masks, dropping empty bins.
        built = []
        for i, cond in enumerate(conditions):
            sel = idx[bin_ids == i]
            if len(sel) == 0:
                continue
            full = np.zeros(len(self.y), dtype=bool)
            full[sel] = True
            rate = self.y[sel].mean()
            built.append((cond, full, rate))
        if len(built) < 2:
            raise ValueError("Split produced fewer than two non-empty bins")

        self._snapshot()
        built.sort(key=lambda t: t[2])  # rank-order children by target rate
        child_ids = []
        for cond, full, _ in built:
            cid = self._next_id
            self._next_id += 1
            child = ChaidNode(
                node_id=cid,
                parent_id=node_id,
                depth=node.depth + 1,
                condition=cond,
            )
            self.nodes[cid] = child
            self._masks[cid] = full
            self._compute_stats(cid)
            child_ids.append(cid)

        node.children = child_ids
        node.split_variable = variable
        node.split_chi2 = chi2
        node.split_pvalue = pval
        node.split_conditions = [c for c, _, _ in built]
        self._record(
            f"Split node {node_id} on '{variable}' -> {len(child_ids)} bins "
            f"(chi2={_fmt(chi2)}, p={_fmtp(pval)})"
        )
        return {"chi2": chi2, "pvalue": pval, "children": child_ids}

    # -- combine --------------------------------------------------------------
    def combine(self, parent_id: int, child_ids: list) -> dict:
        """Merge selected leaf children of a parent into one node; recompute chi2/p."""
        self._require_data()
        parent = self.nodes[parent_id]
        sel = [c for c in child_ids if c in parent.children]
        if len(sel) < 2:
            raise ValueError("Select at least two child nodes to combine")
        for c in sel:
            if not self.nodes[c].is_leaf:
                raise ValueError(f"Child {c} is not a leaf; only leaves can be merged")

        self._snapshot()
        members = [self.nodes[c].condition for c in sel]
        var = parent.split_variable
        merged_cond = Condition(var, "or", members=members)

        merged_mask = np.zeros(len(self.y), dtype=bool)
        for c in sel:
            merged_mask |= self._masks[c]

        mid = self._next_id
        self._next_id += 1
        merged = ChaidNode(
            node_id=mid,
            parent_id=parent_id,
            depth=parent.depth + 1,
            condition=merged_cond,
        )
        self.nodes[mid] = merged
        self._masks[mid] = merged_mask
        self._compute_stats(mid)

        # Replace the selected children with the merged node, keep the rest.
        new_children = [c for c in parent.children if c not in sel]
        new_children.append(mid)
        for c in sel:
            del self.nodes[c]
            del self._masks[c]

        # Rank-order remaining children by target rate.
        new_children.sort(key=lambda cid: self.nodes[cid].target_rate)
        parent.children = new_children
        parent.split_conditions = [self.nodes[c].condition for c in new_children]

        # Recompute parent chi2/p on the new (fewer) bins.
        idx = np.where(self._masks[parent_id])[0]
        sub = self.X.iloc[idx]
        bin_ids = assign_bins(parent.split_conditions, sub)
        chi2, pval, _ = chi2_from_bins(bin_ids, self.y[idx])
        parent.split_chi2 = chi2
        parent.split_pvalue = pval

        self._record(
            f"Combine {len(sel)} children of node {parent_id} "
            f"(chi2={_fmt(chi2)}, p={_fmtp(pval)})"
        )
        return {"merged_id": mid, "chi2": chi2, "pvalue": pval}

    # -- prune ----------------------------------------------------------------
    def prune(self, node_id: int) -> dict:
        """Delete all descendants of a node, turning it back into a leaf."""
        self._require_data()
        node = self.nodes[node_id]
        if node.is_leaf:
            raise ValueError(f"Node {node_id} is already a leaf")

        self._snapshot()
        descendants = self._descendants(node_id)
        for d in descendants:
            del self.nodes[d]
            del self._masks[d]
        node.children = []
        node.split_variable = None
        node.split_chi2 = None
        node.split_pvalue = None
        node.split_conditions = None
        self._record(f"Prune node {node_id} ({len(descendants)} nodes removed)")
        return {"removed": len(descendants)}

    def _descendants(self, node_id: int) -> list:
        out = []
        stack = list(self.nodes[node_id].children)
        while stack:
            cur = stack.pop()
            out.append(cur)
            stack.extend(self.nodes[cur].children)
        return out

    # -- undo -----------------------------------------------------------------
    def _snapshot(self):
        self._undo_stack.append(
            (
                copy.deepcopy(self.nodes),
                {k: v.copy() for k, v in self._masks.items()},
                self._next_id,
            )
        )

    def _record(self, msg: str):
        self.history.append(msg)

    def undo(self) -> bool:
        """Revert the last split/combine/prune; the undo itself is not recorded."""
        if not self._undo_stack:
            return False
        nodes, masks, next_id = self._undo_stack.pop()
        self.nodes = nodes
        self._masks = masks
        self._next_id = next_id
        if self.history:
            self.history.pop()
        return True

    # -- routing / scoring ----------------------------------------------------
    def _route(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the leaf node_id each row of ``frame`` falls into (-1 if none)."""
        result = np.full(len(frame), -1, dtype=int)
        positions = np.arange(len(frame))

        def rec(node_id, pos):
            node = self.nodes[node_id]
            if node.is_leaf:
                result[pos] = node_id
                return
            sub = frame.iloc[pos]
            conds = [self.nodes[c].condition for c in node.children]
            bin_ids = assign_bins(conds, sub)
            for ci, child_id in enumerate(node.children):
                child_pos = pos[bin_ids == ci]
                if len(child_pos):
                    rec(child_id, child_pos)

        rec(0, positions)
        return result

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Score new rows with the target rate of the leaf they fall into."""
        leaf_ids = self._route(X)
        scores = np.full(len(X), np.nan)
        for nid in np.unique(leaf_ids):
            if nid < 0:
                continue
            scores[leaf_ids == nid] = self.nodes[nid].target_rate
        return scores

    def gini(self, X: pd.DataFrame = None, y=None) -> float:
        """Gini of the current tree. No args -> training data; else score X vs y."""
        if X is None:
            self._require_data()
            scores = np.full(len(self.y), np.nan)
            for leaf in self.leaves():
                scores[self._masks[leaf.node_id]] = leaf.target_rate
            return gini_from_scores(scores, self.y)
        if y is None:
            raise ValueError("Provide y to compute Gini on a new dataset")
        scores = self.predict_proba(X)
        return gini_from_scores(scores, np.asarray(pd.Series(y).values).astype(int))

    # -- persistence ----------------------------------------------------------
    def save(self, path: str):
        """Pickle the tree structure + config only (no data, no masks)."""
        payload = {
            "version": 1,
            "nodes": self.nodes,
            "pre_separate_zero": self.pre_separate_zero,
            "pre_separate_values": self.pre_separate_values,
            "min_samples": self.min_samples,
            "max_children": self.max_children,
            "next_id": self._next_id,
            "history": self.history,
            "columns": list(self.X.columns) if self._has_data else None,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str, X: pd.DataFrame = None, y=None) -> "ManualChaidTree":
        """Load a saved tree. With X/y -> editable; without -> view/score-only."""
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        obj = cls.__new__(cls)
        obj.nodes = payload["nodes"]
        obj.pre_separate_zero = payload.get("pre_separate_zero", [])
        obj.pre_separate_values = payload.get("pre_separate_values", {})
        obj.min_samples = payload.get("min_samples")
        obj.max_children = payload.get("max_children")
        obj._next_id = payload.get("next_id", max(obj.nodes) + 1)
        obj.history = payload.get("history", [])
        obj._undo_stack = []
        obj._masks = {}

        if X is not None and y is not None:
            obj.X = X.reset_index(drop=True)
            obj.y = np.asarray(pd.Series(y).values).astype(int)
            obj._has_data = True
            obj._rebuild_masks()
        else:
            obj.X = None
            obj.y = None
            obj._has_data = False
        return obj

    def _rebuild_masks(self):
        """Recompute every node's mask from its rule chain (root -> node)."""
        self._masks = {0: np.ones(len(self.y), dtype=bool)}

        def rec(node_id):
            node = self.nodes[node_id]
            for cid in node.children:
                child = self.nodes[cid]
                parent_mask = self._masks[node_id]
                cond_mask = child.condition.evaluate(self.X[child.condition.variable])
                self._masks[cid] = parent_mask & cond_mask
                rec(cid)

        rec(0)
        for nid in self.nodes:
            self._compute_stats(nid)

    # -- rules export ---------------------------------------------------------
    def _rule_chain(self, node_id: int) -> list:
        chain = []
        cur = self.nodes[node_id]
        while cur.parent_id is not None:
            chain.append(cur.condition.describe())
            cur = self.nodes[cur.parent_id]
        chain.reverse()
        return chain

    def export_rules_excel(self, path: str):
        """Write one row per leaf with stats + per-level rule columns."""
        leaves = sorted(self.leaves(), key=lambda n: n.node_id)
        max_depth = max((n.depth for n in leaves), default=0)
        rows = []
        for leaf in leaves:
            chain = self._rule_chain(leaf.node_id)
            row = {
                "Node ID": leaf.node_id,
                "Samples": leaf.n_samples,
                "Targets": leaf.n_target,
                "Target Rate": round(leaf.target_rate, 6),
            }
            for i in range(max_depth):
                row[f"Rule{i + 1}"] = chain[i] if i < len(chain) else ""
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_excel(path, index=False)
        return df


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.2f}"


def _fmtp(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.3g}"
