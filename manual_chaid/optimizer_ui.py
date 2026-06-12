"""tkinter front-end for :class:`ChaidTreeOptimizer`.

A single desktop window: a read-only reference panel on the left (the available
X columns and ``y``, plus a result log) and a scrollable side menu on the right
with the five problem-definition sections -- variables, decisions, metrics,
KPIs/constraints and goal -- followed by a Solve button that writes the Excel
report. Layout and widget-interaction helpers mirror :mod:`manual_chaid.ui`.
"""
from __future__ import annotations

import json
import tkinter as tk
from dataclasses import asdict
from tkinter import filedialog, messagebox, ttk

from .optimizer import (
    TOTAL,
    ConstraintDef,
    GoalDef,
    KpiDef,
    MetricDef,
    OptimizerSpec,
    VariableDef,
)

MONO = ("Consolas", 9)


class OptimizerUI:
    def __init__(self, opt):
        self.opt = opt
        self._dec_rows = []  # (frame, name_var)
        self._var_rows = []  # (frame, name_var)
        self._metric_rows = []  # (frame, name_var, expr_var)
        self._kpi_rows = []  # (frame, name_var, kind_var, a_var, b_var)
        self._con_rows = []  # (frame, kpi_var, dec_var, min_var, max_var)
        self._grid_vars = {}  # (variable_name, decision) -> StringVar
        self._metric_combos = []  # comboboxes that list metric names
        self._kpi_combos = []  # comboboxes that list KPI names
        self._dec_combos = []  # comboboxes that list decision names
        self._focused_entry = None

        self.root = tk.Tk()
        self.root.title("ChaidTreeOptimizer")
        self.root.geometry("1180x800")
        self._build_layout()
        # start with one empty row in each list so the UI is usable immediately
        self._add_decision_row()
        self._add_variable_row()
        self._add_metric_row()
        self._add_kpi_row()
        self._add_constraint_row()
        self._rebuild_grid()

    def run(self):
        self.root.mainloop()

    # ------------------------------------------------------------- layout
    def _build_layout(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # --- left: reference columns + log
        left = ttk.Frame(paned)
        ttk.Label(
            left, text="Available variables (double-click to insert into focused box):"
        ).pack(anchor="w", padx=6, pady=(6, 0))
        self.ref_list = tk.Listbox(left, height=14, font=MONO, exportselection=False)
        self.ref_list.pack(fill=tk.X, padx=6)
        self.ref_list.bind("<Double-Button-1>", lambda e: self._insert_selected_ref())
        for col in self.opt.columns:
            self.ref_list.insert(tk.END, str(col))
        self.ref_list.insert(tk.END, "y")
        self.ref_list.itemconfig(tk.END, foreground="#a0522d")

        ttk.Label(left, text="Log:").pack(anchor="w", padx=6, pady=(8, 0))
        self.log = tk.Text(left, height=18, font=MONO, wrap="word")
        self.log.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.log.configure(state="disabled")
        paned.add(left, weight=1)

        # --- right: scrollable controls (fixed-width sidebar)
        sidebar_w = 540
        right_outer = ttk.Frame(paned, width=sidebar_w)
        right_outer.pack_propagate(False)
        self._menu_canvas = tk.Canvas(right_outer, width=sidebar_w, highlightthickness=0)
        rsb = ttk.Scrollbar(right_outer, orient=tk.VERTICAL, command=self._menu_canvas.yview)
        self.menu = ttk.Frame(self._menu_canvas)
        self._menu_window = self._menu_canvas.create_window((0, 0), window=self.menu, anchor="nw")
        self.menu.bind(
            "<Configure>",
            lambda e: self._menu_canvas.configure(scrollregion=self._menu_canvas.bbox("all")),
        )
        self._menu_canvas.bind(
            "<Configure>",
            lambda e: self._menu_canvas.itemconfigure(self._menu_window, width=e.width),
        )
        self._menu_canvas.configure(yscrollcommand=rsb.set)
        rsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._menu_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._menu_canvas.bind("<Enter>", self._bind_menu_wheel)
        self._menu_canvas.bind("<Leave>", self._unbind_menu_wheel)
        paned.add(right_outer, weight=0)

        self._build_variables_section()
        self._build_decisions_section()
        self._build_metrics_section()
        self._build_kpi_section()
        self._build_goal_section()
        self._build_actions_section()

    def _section(self, title) -> ttk.LabelFrame:
        lf = ttk.LabelFrame(self.menu, text=title)
        lf.pack(fill=tk.X, padx=6, pady=4)
        return lf

    # ------------------------------------------------------- 1. variables
    def _build_variables_section(self):
        f = self._section("1. Variables")
        ttk.Label(
            f,
            text="Name quantities used by metrics; set each one's per-decision\n"
            "value in section 2.",
            foreground="gray",
        ).pack(anchor="w", padx=4)
        self.var_container = ttk.Frame(f)
        self.var_container.pack(fill=tk.X, padx=4)
        ttk.Button(f, text="+ Add variable", command=self._add_variable_row).pack(
            anchor="w", padx=4, pady=4
        )

    def _add_variable_row(self, name=""):
        row = ttk.Frame(self.var_container)
        row.pack(fill=tk.X, pady=1)
        name_var = tk.StringVar(value=name)
        ttk.Label(row, text="name:").pack(side=tk.LEFT)
        self._entry(row, name_var, width=22).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            row, text="✕", width=3, command=lambda: self._del_variable_row(row)
        ).pack(side=tk.LEFT)
        self._var_rows.append((row, name_var))

    def _del_variable_row(self, row):
        self._var_rows = [r for r in self._var_rows if r[0] is not row]
        row.destroy()
        self._rebuild_grid()

    # ------------------------------------------------------- 2. decisions
    def _build_decisions_section(self):
        f = self._section("2. Decisions & per-decision values")
        self.dec_container = ttk.Frame(f)
        self.dec_container.pack(fill=tk.X, padx=4)
        ttk.Button(f, text="+ Add decision", command=self._add_decision_row).pack(
            anchor="w", padx=4, pady=4
        )
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=2)
        head = ttk.Frame(f)
        head.pack(fill=tk.X, padx=4)
        ttk.Label(head, text="Variable values per decision:").pack(side=tk.LEFT)
        ttk.Button(head, text="↻ Refresh grid", command=self._rebuild_grid).pack(side=tk.RIGHT)
        self.grid_frame = ttk.Frame(f)
        self.grid_frame.pack(fill=tk.X, padx=4, pady=2)

    def _add_decision_row(self, name=""):
        row = ttk.Frame(self.dec_container)
        row.pack(fill=tk.X, pady=1)
        name_var = tk.StringVar(value=name)
        ttk.Label(row, text="decision:").pack(side=tk.LEFT)
        self._entry(row, name_var, width=22).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            row, text="✕", width=3, command=lambda: self._del_decision_row(row)
        ).pack(side=tk.LEFT)
        self._dec_rows.append((row, name_var))

    def _del_decision_row(self, row):
        self._dec_rows = [r for r in self._dec_rows if r[0] is not row]
        row.destroy()
        self._rebuild_grid()

    def _rebuild_grid(self):
        for child in self.grid_frame.winfo_children():
            child.destroy()
        variables = self._variable_names()
        decisions = self._decision_names()
        ttk.Label(self.grid_frame, text="", width=14).grid(row=0, column=0)
        for j, d in enumerate(decisions, start=1):
            ttk.Label(self.grid_frame, text=d, font=("TkDefaultFont", 8, "bold")).grid(
                row=0, column=j, padx=2
            )
        for i, v in enumerate(variables, start=1):
            ttk.Label(self.grid_frame, text=v, width=14, anchor="w").grid(
                row=i, column=0, sticky="w"
            )
            for j, d in enumerate(decisions, start=1):
                key = (v, d)
                if key not in self._grid_vars:
                    self._grid_vars[key] = tk.StringVar()
                self._entry(self.grid_frame, self._grid_vars[key], width=9).grid(
                    row=i, column=j, padx=2, pady=1
                )
        self._refresh_choices()

    # --------------------------------------------------------- 3. metrics
    def _build_metrics_section(self):
        f = self._section("3. Metrics (per-sample, over variables / columns / y)")
        self.metric_container = ttk.Frame(f)
        self.metric_container.pack(fill=tk.X, padx=4)
        ttk.Button(f, text="+ Add metric", command=self._add_metric_row).pack(
            anchor="w", padx=4, pady=4
        )

    def _add_metric_row(self, name="", expr=""):
        row = ttk.Frame(self.metric_container)
        row.pack(fill=tk.X, pady=1)
        name_var = tk.StringVar(value=name)
        expr_var = tk.StringVar(value=expr)
        self._entry(row, name_var, width=14).pack(side=tk.LEFT)
        ttk.Label(row, text="=").pack(side=tk.LEFT, padx=2)
        self._entry(row, expr_var, width=22).pack(side=tk.LEFT)
        ttk.Button(
            row, text="✕", width=3, command=lambda: self._del_metric_row(row)
        ).pack(side=tk.LEFT, padx=2)
        self._metric_rows.append((row, name_var, expr_var))

    def _del_metric_row(self, row):
        self._metric_rows = [r for r in self._metric_rows if r[0] is not row]
        row.destroy()
        self._refresh_choices()

    # --------------------------------------------- 4. KPIs & constraints
    def _build_kpi_section(self):
        f = self._section("4. KPIs & Constraints")
        ttk.Label(f, text="KPIs (ratio = numerator / denominator):", foreground="gray").pack(
            anchor="w", padx=4
        )
        self.kpi_container = ttk.Frame(f)
        self.kpi_container.pack(fill=tk.X, padx=4)
        ttk.Button(f, text="+ Add KPI", command=self._add_kpi_row).pack(
            anchor="w", padx=4, pady=4
        )
        ttk.Separator(f, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=2)
        ttk.Label(
            f, text="Constraints (bound a KPI for one decision's population):", foreground="gray"
        ).pack(anchor="w", padx=4)
        self.con_container = ttk.Frame(f)
        self.con_container.pack(fill=tk.X, padx=4)
        ttk.Button(f, text="+ Add constraint", command=self._add_constraint_row).pack(
            anchor="w", padx=4, pady=4
        )

    def _add_kpi_row(self, name="", kind="ratio", a="", b=""):
        row = ttk.Frame(self.kpi_container)
        row.pack(fill=tk.X, pady=1)
        name_var = tk.StringVar(value=name)
        kind_var = tk.StringVar(value=kind)
        a_var = tk.StringVar(value=a)
        b_var = tk.StringVar(value=b)
        self._entry(row, name_var, width=12).pack(side=tk.LEFT)
        kind_combo = ttk.Combobox(
            row, textvariable=kind_var, values=["ratio", "sum"], width=5, state="readonly"
        )
        kind_combo.pack(side=tk.LEFT, padx=2)
        self._no_combo_wheel(kind_combo)
        a_combo = self._metric_combo(row, a_var, width=10)
        a_combo.pack(side=tk.LEFT, padx=1)
        slash = ttk.Label(row, text="/")
        slash.pack(side=tk.LEFT)
        b_combo = self._metric_combo(row, b_var, width=10)
        b_combo.pack(side=tk.LEFT, padx=1)
        ttk.Button(
            row, text="✕", width=3, command=lambda: self._del_kpi_row(row)
        ).pack(side=tk.LEFT, padx=2)
        self._kpi_rows.append((row, name_var, kind_var, a_var, b_var))
        self._refresh_choices()

    def _del_kpi_row(self, row):
        self._kpi_rows = [r for r in self._kpi_rows if r[0] is not row]
        row.destroy()
        self._refresh_choices()

    def _add_constraint_row(self, kpi="", decision="", mn="", mx=""):
        row = ttk.Frame(self.con_container)
        row.pack(fill=tk.X, pady=1)
        kpi_var = tk.StringVar(value=kpi)
        dec_var = tk.StringVar(value=decision)
        min_var = tk.StringVar(value=mn)
        max_var = tk.StringVar(value=mx)
        kpi_combo = self._kpi_combo(row, kpi_var, width=10)
        kpi_combo.pack(side=tk.LEFT, padx=1)
        ttk.Label(row, text="on").pack(side=tk.LEFT)
        dec_combo = self._dec_combo(row, dec_var, width=9)
        dec_combo.pack(side=tk.LEFT, padx=1)
        ttk.Label(row, text="min").pack(side=tk.LEFT)
        self._entry(row, min_var, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Label(row, text="max").pack(side=tk.LEFT)
        self._entry(row, max_var, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Button(
            row, text="✕", width=3, command=lambda: self._del_constraint_row(row)
        ).pack(side=tk.LEFT, padx=2)
        self._con_rows.append((row, kpi_var, dec_var, min_var, max_var))
        self._refresh_choices()

    def _del_constraint_row(self, row):
        self._con_rows = [r for r in self._con_rows if r[0] is not row]
        row.destroy()

    # ------------------------------------------------------------ 5. goal
    def _build_goal_section(self):
        f = self._section("5. Goal")
        row = ttk.Frame(f)
        row.pack(fill=tk.X, padx=4, pady=2)
        self._goal_sense = tk.StringVar(value="max")
        ttk.Radiobutton(row, text="Maximise", variable=self._goal_sense, value="max").pack(
            side=tk.LEFT
        )
        ttk.Radiobutton(row, text="Minimise", variable=self._goal_sense, value="min").pack(
            side=tk.LEFT, padx=4
        )
        ttk.Label(row, text="metric:").pack(side=tk.LEFT, padx=(8, 2))
        self._goal_metric = tk.StringVar()
        combo = self._metric_combo(row, self._goal_metric, width=16)
        combo.pack(side=tk.LEFT)

    # --------------------------------------------------------- 6. actions
    def _build_actions_section(self):
        f = self._section("6. Actions")
        ttk.Button(f, text="↻ Refresh choices", command=self._refresh_choices).pack(
            fill=tk.X, padx=4, pady=2
        )
        ttk.Button(f, text="Solve & export to Excel…", command=self._do_solve).pack(
            fill=tk.X, padx=4, pady=2
        )
        ttk.Button(f, text="Save problem (JSON)…", command=self._do_save_json).pack(
            fill=tk.X, padx=4, pady=2
        )
        ttk.Button(f, text="Load problem (JSON)…", command=self._do_load_json).pack(
            fill=tk.X, padx=4, pady=2
        )

    # ------------------------------------------------------------- helpers
    def _entry(self, parent, textvar, width):
        ent = ttk.Entry(parent, textvariable=textvar, width=width)
        ent.bind("<FocusIn>", self._on_entry_focus)
        return ent

    def _on_entry_focus(self, event):
        self._focused_entry = event.widget

    def _insert_selected_ref(self):
        sel = self.ref_list.curselection()
        if not sel or self._focused_entry is None:
            return
        text = self.ref_list.get(sel[0])
        try:
            self._focused_entry.insert("insert", text)
        except tk.TclError:
            pass

    def _metric_combo(self, parent, textvar, width):
        combo = ttk.Combobox(parent, textvariable=textvar, width=width, state="normal")
        self._no_combo_wheel(combo)
        self._metric_combos.append(combo)
        return combo

    def _kpi_combo(self, parent, textvar, width):
        combo = ttk.Combobox(parent, textvariable=textvar, width=width, state="readonly")
        self._no_combo_wheel(combo)
        self._kpi_combos.append(combo)
        return combo

    def _dec_combo(self, parent, textvar, width):
        combo = ttk.Combobox(parent, textvariable=textvar, width=width, state="readonly")
        self._no_combo_wheel(combo)
        self._dec_combos.append(combo)
        return combo

    def _refresh_choices(self):
        metrics = self._metric_names()
        kpis = self._kpi_names()
        decisions = self._decision_names()
        for combo in self._metric_combos:
            if combo.winfo_exists():
                combo["values"] = metrics
        for combo in self._kpi_combos:
            if combo.winfo_exists():
                combo["values"] = kpis
        for combo in self._dec_combos:
            if combo.winfo_exists():
                combo["values"] = [TOTAL, *decisions]
        # drop GC'd combos
        self._metric_combos = [c for c in self._metric_combos if c.winfo_exists()]
        self._kpi_combos = [c for c in self._kpi_combos if c.winfo_exists()]
        self._dec_combos = [c for c in self._dec_combos if c.winfo_exists()]

    # ---- current names
    def _decision_names(self):
        seen, out = set(), []
        for _, v in self._dec_rows:
            name = v.get().strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def _variable_names(self):
        seen, out = set(), []
        for _, v in self._var_rows:
            name = v.get().strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def _metric_names(self):
        seen, out = set(), []
        for _, n, _ in self._metric_rows:
            name = n.get().strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def _kpi_names(self):
        seen, out = set(), []
        for _, n, _, _, _ in self._kpi_rows:
            name = n.get().strip()
            if name and name not in seen:
                seen.add(name)
                out.append(name)
        return out

    # ---- build spec from widgets
    def _build_spec(self) -> OptimizerSpec:
        decisions = self._decision_names()
        variables = []
        for _, v in self._var_rows:
            name = v.get().strip()
            if not name:
                continue
            per_decision = {
                d: self._grid_vars.get((name, d), tk.StringVar()).get().strip()
                for d in decisions
            }
            variables.append(VariableDef(name=name, per_decision=per_decision))
        metrics = [
            MetricDef(name=n.get().strip(), expr=e.get().strip())
            for _, n, e in self._metric_rows
            if n.get().strip()
        ]
        kpis = []
        for _, n, kind, a, b in self._kpi_rows:
            name = n.get().strip()
            if not name:
                continue
            if kind.get() == "ratio":
                kpis.append(KpiDef(name=name, kind="ratio", num=a.get().strip(), den=b.get().strip()))
            else:
                kpis.append(
                    KpiDef(name=name, kind="linear", weights={a.get().strip(): 1.0})
                )
        constraints = []
        for _, kpi, dec, mn, mx in self._con_rows:
            if not kpi.get().strip():
                continue
            constraints.append(
                ConstraintDef(
                    kpi=kpi.get().strip(),
                    decision=dec.get().strip(),
                    min=_parse_float(mn.get()),
                    max=_parse_float(mx.get()),
                )
            )
        goal = None
        if self._goal_metric.get().strip():
            goal = GoalDef(metric=self._goal_metric.get().strip(), sense=self._goal_sense.get())
        return OptimizerSpec(
            decisions=decisions,
            variables=variables,
            metrics=metrics,
            kpis=kpis,
            constraints=constraints,
            goal=goal,
        )

    # ------------------------------------------------------------- actions
    def _do_solve(self):
        try:
            spec = self._build_spec()
            result = self.opt.solve(spec)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Solve failed", str(exc))
            self._write_log(f"ERROR: {exc}")
            return
        self._write_log(f"Solver status: {result.status}")
        if result.status != "Optimal":
            messagebox.showwarning(
                "Solve",
                f"Solver status: {result.status}.\n"
                "No assignment was produced (the constraints may be infeasible).",
            )
            return
        self._write_log(
            f"Objective ({spec.goal.sense} {spec.goal.metric}) = {result.objective:.4g}"
        )
        for (name, d), val in result.kpi_values.items():
            if val is not None:
                self._write_log(f"  KPI {name} [{d}] = {val:.4g}")
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")]
        )
        if not path:
            return
        try:
            self.opt.export_excel(spec, result, path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Export failed", str(exc))
            return
        self._write_log(f"Wrote report to {path}")
        messagebox.showinfo("Solve", f"Optimal assignment written to\n{path}")

    def _do_save_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")]
        )
        if not path:
            return
        spec = self._build_spec()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(spec), fh, indent=2)
        self._write_log(f"Saved problem to {path}")

    def _do_load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._populate_from_dict(data)
        self._write_log(f"Loaded problem from {path}")

    def _populate_from_dict(self, data):
        # clear every row list
        for rows in (self._dec_rows, self._var_rows, self._metric_rows, self._kpi_rows, self._con_rows):
            for r in list(rows):
                r[0].destroy()
            rows.clear()
        self._grid_vars.clear()

        for d in data.get("decisions", []):
            self._add_decision_row(name=d)
        for v in data.get("variables", []):
            self._add_variable_row(name=v["name"])
            for d, expr in v.get("per_decision", {}).items():
                self._grid_vars[(v["name"], d)] = tk.StringVar(value=expr)
        for m in data.get("metrics", []):
            self._add_metric_row(name=m["name"], expr=m["expr"])
        for k in data.get("kpis", []):
            if k.get("kind") == "ratio":
                self._add_kpi_row(name=k["name"], kind="ratio", a=k.get("num", ""), b=k.get("den", ""))
            else:
                weights = k.get("weights", {})
                metric = next(iter(weights), "")
                self._add_kpi_row(name=k["name"], kind="sum", a=metric)
        for c in data.get("constraints", []):
            self._add_constraint_row(
                kpi=c["kpi"],
                decision=c["decision"],
                mn="" if c.get("min") is None else str(c["min"]),
                mx="" if c.get("max") is None else str(c["max"]),
            )
        goal = data.get("goal")
        if goal:
            self._goal_metric.set(goal.get("metric", ""))
            self._goal_sense.set(goal.get("sense", "max"))
        self._rebuild_grid()

    def _write_log(self, msg):
        self.log.configure(state="normal")
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    # --------------------------------------------------------- wheel helpers
    def _no_combo_wheel(self, combo):
        def handler(event):
            self._menu_canvas.yview_scroll(int(-event.delta / 120), "units")
            return "break"

        combo.bind("<MouseWheel>", handler)

    def _bind_menu_wheel(self, event):
        self._menu_canvas.bind_all("<MouseWheel>", self._on_menu_wheel)

    def _unbind_menu_wheel(self, event):
        self._menu_canvas.unbind_all("<MouseWheel>")

    def _on_menu_wheel(self, event):
        self._menu_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"


def _parse_float(text):
    text = str(text).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None
