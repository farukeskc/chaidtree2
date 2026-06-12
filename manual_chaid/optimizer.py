"""ChaidTreeOptimizer -- assign an optimal decision to each leaf of a tree.

Given a built :class:`ManualChaidTree` and a dataset ``(X, y)``, the user defines a
small optimisation problem through five layers:

    variables -> decisions -> metrics -> KPIs/constraints -> goal

and the optimiser picks exactly **one** decision per leaf to optimise the goal
subject to the KPI constraints.

The problem is a binary integer linear program: a 0/1 variable ``x[leaf, decision]``
with "exactly one decision per leaf", linear/ratio KPI constraints and a linear
metric-sum objective. It is solved with CBC via PuLP. The result is written to a
multi-sheet Excel report (Assignments / KPIs / Goal / Definition).

The maths
---------
Routing ``X`` through the tree gives a leaf id per sample. For a fixed decision ``d``
every user *variable* has a concrete per-decision expression, so every per-sample
*metric* is fully determined. We therefore precompute, per leaf ``l`` and decision
``d``::

    S[m][d][l] = sum over samples in leaf l of metric_m(sample, decision=d)

With ``x[l, d] in {0, 1}`` and ``sum_d x[l, d] = 1`` for each leaf:

* objective (goal metric ``g`` over all samples):
  ``min/max  sum_{l, d} x[l, d] * S[g][d][l]``
* ratio KPI ``num/den`` bounded by ``B`` on decision ``d``'s population:
  - max: ``sum_l x[l, d] * (S[num][d][l] - B * S[den][d][l]) <= 0``
  - min: ``sum_l x[l, d] * (S[num][d][l] - B * S[den][d][l]) >= 0``
* linear KPI (weighted metric sum) bounded by ``B`` on decision ``d``:
  ``sum_l x[l, d] * combo_l  </>= B``
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Sentinel decision for a constraint that spans the whole population (every leaf
# under its assigned decision) rather than a single decision's slice.
TOTAL = "TOTAL"

# Functions exposed to user expressions (no builtins beyond these).
_ALLOWED_FUNCS = {
    "np": np,
    "where": np.where,
    "log": np.log,
    "log1p": np.log1p,
    "exp": np.exp,
    "abs": np.abs,
    "sqrt": np.sqrt,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
}


# --- problem specification ---------------------------------------------------
@dataclass
class VariableDef:
    """A named quantity whose value depends on the decision taken."""

    name: str
    per_decision: dict = field(default_factory=dict)  # decision -> expression


@dataclass
class MetricDef:
    """A per-sample quantity computed from variables, X columns and ``y``."""

    name: str
    expr: str


@dataclass
class KpiDef:
    """A column-wise (population-level) quantity built from metric sums."""

    name: str
    kind: str = "ratio"  # "ratio" | "linear"
    num: str = ""  # ratio numerator metric name
    den: str = ""  # ratio denominator metric name
    weights: dict = field(default_factory=dict)  # linear: metric name -> weight


@dataclass
class ConstraintDef:
    """A min/max bound on a KPI evaluated over one decision's population."""

    kpi: str
    decision: str
    min: Optional[float] = None
    max: Optional[float] = None


@dataclass
class GoalDef:
    metric: str
    sense: str = "max"  # "min" | "max"


@dataclass
class OptimizerSpec:
    decisions: list = field(default_factory=list)
    variables: list = field(default_factory=list)  # VariableDef
    metrics: list = field(default_factory=list)  # MetricDef
    kpis: list = field(default_factory=list)  # KpiDef
    constraints: list = field(default_factory=list)  # ConstraintDef
    goal: Optional[GoalDef] = None


@dataclass
class SolveResult:
    status: str
    assignment: dict  # leaf_id -> decision
    objective: Optional[float]
    kpi_values: dict  # (kpi_name, decision) -> realised value or None


class ChaidTreeOptimizer:
    """Assign an optimal decision to every leaf of a ManualChaidTree."""

    def __init__(self, tree, X, y):
        self.tree = tree
        self.X = X.reset_index(drop=True)
        self.y = np.asarray(pd.Series(y).values)
        self.n = len(self.X)
        self.leaf_ids = np.asarray(tree.predict_node(self.X))
        # leaves that actually receive samples (skip unrouted -1)
        self.leaves = sorted(int(l) for l in np.unique(self.leaf_ids) if l >= 0)

    @property
    def columns(self) -> list:
        """X column names (for the UI variable picker)."""
        return list(self.X.columns)

    # ------------------------------------------------------------- entry point
    def optimize(self, spec: "OptimizerSpec" = None, launch_ui: bool = True):
        """Launch the UI, or solve a supplied spec headless.

        Mirrors :meth:`ManualChaidTree.train`: with ``launch_ui`` (default) the
        tkinter builder opens and drives ``solve``/``export_excel`` itself, and
        ``self`` is returned. With ``launch_ui=False`` a ``spec`` is required and
        the :class:`SolveResult` is returned.
        """
        if launch_ui:
            from .optimizer_ui import OptimizerUI

            OptimizerUI(self).run()
            return self
        if spec is None:
            raise ValueError("Provide a spec when launch_ui=False.")
        return self.solve(spec)

    # --------------------------------------------------------- expression eval
    def _base_namespace(self) -> dict:
        ns = dict(_ALLOWED_FUNCS)
        for col in self.X.columns:
            ns[str(col)] = self.X[col].to_numpy()
        ns["y"] = self.y
        return ns

    def _eval(self, expr: str, namespace: dict, what: str) -> np.ndarray:
        text = str(expr).strip()
        if text == "":
            raise ValueError(f"{what}: expression is empty.")
        try:
            val = eval(text, {"__builtins__": {}}, namespace)  # noqa: S307
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{what}: cannot evaluate '{text}'\n{exc}") from exc
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            arr = np.full(self.n, float(arr))
        if arr.shape[0] != self.n:
            raise ValueError(
                f"{what}: expression '{text}' produced {arr.shape[0]} values, "
                f"expected {self.n}."
            )
        return arr

    # ------------------------------------------------------------- aggregation
    def _aggregate(self, spec: "OptimizerSpec"):
        """Precompute ``S[metric][decision][leaf]`` sums and per-leaf counts."""
        leaf_series = pd.Series(self.leaf_ids)
        S = {m.name: {} for m in spec.metrics}
        for d in spec.decisions:
            ns = self._base_namespace()
            for v in spec.variables:
                ns[v.name] = self._eval(
                    v.per_decision.get(d, ""), ns, f"Variable '{v.name}' for '{d}'"
                )
            for m in spec.metrics:
                arr = self._eval(m.expr, ns, f"Metric '{m.name}'")
                sums = (
                    pd.Series(arr)
                    .groupby(leaf_series)
                    .sum()
                    .reindex(self.leaves, fill_value=0.0)
                )
                S[m.name][d] = {int(l): float(v) for l, v in sums.items()}
        counts = leaf_series.value_counts()
        n_samples = {l: int(counts.get(l, 0)) for l in self.leaves}
        return S, n_samples

    # ----------------------------------------------------------------- solving
    def _validate(self, spec: "OptimizerSpec"):
        if not spec.decisions:
            raise ValueError("Define at least one decision.")
        if not spec.metrics:
            raise ValueError("Define at least one metric.")
        if spec.goal is None or not spec.goal.metric:
            raise ValueError("Define a goal (metric + minimise/maximise).")
        metric_names = {m.name for m in spec.metrics}
        if spec.goal.metric not in metric_names:
            raise ValueError(f"Goal metric '{spec.goal.metric}' is not a defined metric.")
        kpi_names = {k.name for k in spec.kpis}
        for k in spec.kpis:
            if k.kind == "ratio":
                for role, mn in (("numerator", k.num), ("denominator", k.den)):
                    if mn not in metric_names:
                        raise ValueError(
                            f"KPI '{k.name}' {role} '{mn}' is not a defined metric."
                        )
            else:
                for mn in k.weights:
                    if mn not in metric_names:
                        raise ValueError(
                            f"KPI '{k.name}' references undefined metric '{mn}'."
                        )
        for c in spec.constraints:
            if c.kpi not in kpi_names:
                raise ValueError(f"Constraint references undefined KPI '{c.kpi}'.")
            if c.decision != TOTAL and c.decision not in spec.decisions:
                raise ValueError(
                    f"Constraint on '{c.kpi}' targets undefined decision '{c.decision}'."
                )

    def solve(self, spec: "OptimizerSpec") -> "SolveResult":
        import pulp

        self._validate(spec)
        if not self.leaves:
            raise ValueError("No samples were routed to any leaf.")
        S, _ = self._aggregate(spec)
        leaves, decisions = self.leaves, spec.decisions

        sense = pulp.LpMaximize if spec.goal.sense == "max" else pulp.LpMinimize
        prob = pulp.LpProblem("chaid_leaf_assignment", sense)
        x = pulp.LpVariable.dicts("x", (leaves, decisions), cat="Binary")

        # exactly one decision per leaf
        for l in leaves:
            prob += pulp.lpSum(x[l][d] for d in decisions) == 1

        # objective: goal metric summed over all samples under assigned decision
        g = spec.goal.metric
        prob += pulp.lpSum(x[l][d] * S[g][d][l] for l in leaves for d in decisions)

        for c in spec.constraints:
            self._add_constraint(prob, x, S, spec, c)

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[prob.status]

        assignment = {}
        if status == "Optimal":
            for l in leaves:
                for d in decisions:
                    v = x[l][d].value()
                    if v is not None and v > 0.5:
                        assignment[l] = d
        objective = pulp.value(prob.objective) if status == "Optimal" else None
        kpi_values = self._realised_kpis(spec, S, assignment)
        return SolveResult(status, assignment, objective, kpi_values)

    def _add_constraint(self, prob, x, S, spec, c):
        import pulp

        kpi = next((k for k in spec.kpis if k.name == c.kpi), None)
        if kpi is None:
            return
        # TOTAL spans every decision; summing the active x term over all decisions
        # makes each leaf contribute under whichever decision it is assigned.
        decisions = spec.decisions if c.decision == TOTAL else [c.decision]
        if kpi.kind == "ratio":
            # ratio = sum(num) / sum(den); linearise the bound (den >= 0).
            if c.max is not None:
                prob += (
                    pulp.lpSum(
                        x[l][d] * (S[kpi.num][d][l] - c.max * S[kpi.den][d][l])
                        for l in self.leaves
                        for d in decisions
                    )
                    <= 0
                )
            if c.min is not None:
                prob += (
                    pulp.lpSum(
                        x[l][d] * (S[kpi.num][d][l] - c.min * S[kpi.den][d][l])
                        for l in self.leaves
                        for d in decisions
                    )
                    >= 0
                )
        else:  # linear weighted sum of metric totals
            expr = pulp.lpSum(
                x[l][d] * sum(w * S[m][d][l] for m, w in kpi.weights.items())
                for l in self.leaves
                for d in decisions
            )
            if c.max is not None:
                prob += expr <= c.max
            if c.min is not None:
                prob += expr >= c.min

    # ---------------------------------------------------------- realised KPIs
    def _kpi_value(self, kpi, S, pairs):
        """Evaluate a KPI over a list of ``(leaf, decision)`` pairs."""
        if not pairs:
            return None
        if kpi.kind == "ratio":
            num = sum(S[kpi.num][d][l] for l, d in pairs)
            den = sum(S[kpi.den][d][l] for l, d in pairs)
            return num / den if den != 0 else None
        return sum(w * S[m][d][l] for l, d in pairs for m, w in kpi.weights.items())

    def _realised_kpis(self, spec, S, assignment) -> dict:
        out = {}
        all_pairs = list(assignment.items())
        for kpi in spec.kpis:
            for d in spec.decisions:
                pairs = [(l, dd) for l, dd in all_pairs if dd == d]
                out[(kpi.name, d)] = self._kpi_value(kpi, S, pairs)
            out[(kpi.name, TOTAL)] = self._kpi_value(kpi, S, all_pairs)
        return out

    @staticmethod
    def _constraint_status(val, cmin, cmax) -> str:
        if cmin is None and cmax is None:
            return ""
        if val is None:
            return "n/a (empty)"
        ok = True
        if cmin is not None and val < cmin - 1e-9:
            ok = False
        if cmax is not None and val > cmax + 1e-9:
            ok = False
        return "Pass" if ok else "Fail"

    # ----------------------------------------------------------- excel report
    def export_excel(self, spec: "OptimizerSpec", result: "SolveResult", path: str):
        S, n_samples = self._aggregate(spec)
        rule_chains = {l: self.tree._rule_chain(l) for l in self.leaves}
        max_depth = max((len(rc) for rc in rule_chains.values()), default=0)

        # -- Assignments
        rows = []
        for l in self.leaves:
            d = result.assignment.get(l, "")
            row = {"Node ID": l, "Samples": n_samples[l], "Assigned Decision": d}
            chain = rule_chains[l]
            for i in range(max_depth):
                row[f"Rule{i + 1}"] = chain[i] if i < len(chain) else ""
            for m in spec.metrics:
                row[m.name] = S[m.name][d][l] if d else None
            rows.append(row)
        assign_df = pd.DataFrame(rows)

        # -- KPIs
        bound = {(c.kpi, c.decision): c for c in spec.constraints}
        kpi_rows = []
        for kpi in spec.kpis:
            for d in [*spec.decisions, TOTAL]:
                val = result.kpi_values.get((kpi.name, d))
                c = bound.get((kpi.name, d))
                cmin = c.min if c else None
                cmax = c.max if c else None
                kpi_rows.append(
                    {
                        "KPI": kpi.name,
                        "Decision": d,
                        "Value": val,
                        "Min": cmin,
                        "Max": cmax,
                        "Status": self._constraint_status(val, cmin, cmax),
                    }
                )
        kpi_df = pd.DataFrame(kpi_rows)

        # -- Goal
        goal_df = pd.DataFrame(
            [
                {
                    "Goal Metric": spec.goal.metric if spec.goal else "",
                    "Sense": spec.goal.sense if spec.goal else "",
                    "Objective Value": result.objective,
                    "Solver Status": result.status,
                }
            ]
        )

        # -- Definition (traceability)
        def_rows = [{"Section": "Decisions", "Name": ", ".join(spec.decisions), "Detail": ""}]
        for v in spec.variables:
            for d, e in v.per_decision.items():
                def_rows.append({"Section": "Variable", "Name": v.name, "Detail": f"{d}: {e}"})
        for m in spec.metrics:
            def_rows.append({"Section": "Metric", "Name": m.name, "Detail": m.expr})
        for k in spec.kpis:
            detail = f"{k.num} / {k.den}" if k.kind == "ratio" else str(k.weights)
            def_rows.append({"Section": "KPI", "Name": k.name, "Detail": detail})
        for c in spec.constraints:
            def_rows.append(
                {
                    "Section": "Constraint",
                    "Name": c.kpi,
                    "Detail": f"decision={c.decision} min={c.min} max={c.max}",
                }
            )
        if spec.goal:
            def_rows.append({"Section": "Goal", "Name": spec.goal.metric, "Detail": spec.goal.sense})
        def_df = pd.DataFrame(def_rows)

        with pd.ExcelWriter(path, engine="openpyxl") as xl:
            assign_df.to_excel(xl, sheet_name="Assignments", index=False)
            kpi_df.to_excel(xl, sheet_name="KPIs", index=False)
            goal_df.to_excel(xl, sheet_name="Goal", index=False)
            def_df.to_excel(xl, sheet_name="Definition", index=False)
        return path
