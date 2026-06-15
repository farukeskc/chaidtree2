"""tkinter front-end for ManualChaidTree.

A single desktop window: the tree preview canvas on the left (a Plotly figure
rendered to PNG via kaleido and painted here) and a scrollable side menu on the
right with Split / Combine / Prune / History / Actions sections.
"""
from __future__ import annotations

import io
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .binning import categorical_conditions_from_groups, conditions_from_cutpoints
from .node import OPERATORS
from .render import build_figure, figure_to_png

MONO = ("Consolas", 9)


class TreeBuilderUI:
    def __init__(self, tree):
        self.tree = tree
        self._photo = None  # keep a reference so Tk doesn't GC the image
        self._base_img = None  # full-res rendered PIL image (pre-zoom)
        self._zoom = 1.0
        self._candidates: dict = {}  # variable -> candidate dict for selected leaf
        self._cutpoint_rows: list = []  # numeric: list of (frame, value_var, op_var)
        self._cat_rows: list = []  # categorical: list of (frame, category, group_var)
        self._current_var = None
        self._current_specials: list = []

        self.root = tk.Tk()
        self.root.title("ManualChaidTree builder")
        self.root.geometry("1280x780")
        self._build_layout()
        self._refresh_controls()
        self.redraw()

    # ------------------------------------------------------------- run/layout
    def run(self):
        self.root.mainloop()

    def _build_layout(self):
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # --- left: zoom toolbar + scrollable image canvas
        left = ttk.Frame(paned)
        toolbar = ttk.Frame(left)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=2)
        ttk.Button(toolbar, text="−", width=3, command=self.zoom_out).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="+", width=3, command=self.zoom_in).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Reset", command=self.zoom_reset).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Fit width", command=self.zoom_fit).pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(toolbar, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=8)
        ttk.Label(toolbar, text="(Ctrl + mouse-wheel to zoom)", foreground="gray").pack(
            side=tk.LEFT
        )

        cv_frame = ttk.Frame(left)
        cv_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(cv_frame, bg="white")
        hbar = ttk.Scrollbar(cv_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vbar = ttk.Scrollbar(cv_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # scroll with wheel; zoom with Ctrl + wheel; drag with left button to pan
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_wheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_pan_end)
        paned.add(left, weight=1)  # left canvas absorbs extra space on resize

        # --- right: scrollable controls (fixed-width sidebar)
        sidebar_w = 380
        right_outer = ttk.Frame(paned, width=sidebar_w)
        right_outer.pack_propagate(False)
        self._menu_canvas = tk.Canvas(
            right_outer, width=sidebar_w, highlightthickness=0
        )
        rsb = ttk.Scrollbar(
            right_outer, orient=tk.VERTICAL, command=self._menu_canvas.yview
        )
        self.menu = ttk.Frame(self._menu_canvas)
        self._menu_window = self._menu_canvas.create_window(
            (0, 0), window=self.menu, anchor="nw"
        )
        # keep the scrollregion in sync with the content height...
        self.menu.bind(
            "<Configure>",
            lambda e: self._menu_canvas.configure(
                scrollregion=self._menu_canvas.bbox("all")
            ),
        )
        # ...and stretch the inner frame to the canvas width so sections fill it.
        self._menu_canvas.bind(
            "<Configure>",
            lambda e: self._menu_canvas.itemconfigure(self._menu_window, width=e.width),
        )
        self._menu_canvas.configure(yscrollcommand=rsb.set)
        rsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._menu_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # mouse-wheel scrolls the sidebar while the pointer is over it
        self._menu_canvas.bind("<Enter>", self._bind_menu_wheel)
        self._menu_canvas.bind("<Leave>", self._unbind_menu_wheel)
        paned.add(right_outer, weight=0)  # sidebar keeps a stable width

        self._build_split_section()
        self._build_combine_section()
        self._build_prune_section()
        self._build_history_section()
        self._build_actions_section()

    def _section(self, title) -> ttk.Frame:
        lf = ttk.LabelFrame(self.menu, text=title)
        lf.pack(fill=tk.X, padx=6, pady=4)
        return lf

    # ------------------------------------------------------------ split panel
    def _build_split_section(self):
        f = self._section("1. Split a leaf node")
        ttk.Label(f, text="Leaf node:").pack(anchor="w", padx=4)
        self.leaf_combo = ttk.Combobox(f, state="readonly")
        self.leaf_combo.pack(fill=tk.X, padx=4)
        self.leaf_combo.bind("<<ComboboxSelected>>", lambda e: self._load_candidates())
        self._no_combo_wheel(self.leaf_combo)

        ttk.Label(f, text="Variables (by χ²):").pack(anchor="w", padx=4, pady=(6, 0))
        self.var_list = tk.Listbox(f, height=8, font=MONO, exportselection=False)
        self.var_list.pack(fill=tk.X, padx=4)
        self.var_list.bind("<<ListboxSelect>>", lambda e: self._select_variable())
        self._own_wheel(self.var_list)

        self.cut_label = ttk.Label(f, text="Cut-offs:", font=("TkDefaultFont", 9, "bold"))
        self.cut_label.pack(anchor="w", padx=4, pady=(6, 0))
        self.specials_label = ttk.Label(f, text="", foreground="#a0522d", wraplength=340)
        self.specials_label.pack(anchor="w", padx=4)
        self.cut_frame = ttk.Frame(f)
        self.cut_frame.pack(fill=tk.X, padx=4)
        self.cut_note = ttk.Label(f, text="", foreground="gray", wraplength=340)
        self.cut_note.pack(anchor="w", padx=4)

        btns = ttk.Frame(f)
        btns.pack(fill=tk.X, padx=4, pady=4)
        self._add_cut_btn = ttk.Button(
            btns, text="+ Add cut-off", command=self._add_cutpoint
        )
        self._add_cut_btn.pack(side=tk.LEFT)
        ttk.Button(btns, text="Split", command=self._do_split).pack(side=tk.RIGHT)

    def _add_cutpoint(self, value="", op="<"):
        row = ttk.Frame(self.cut_frame)
        val_var = tk.StringVar(value=str(value))
        op_var = tk.StringVar(value=op)
        ttk.Entry(row, textvariable=val_var, width=10).pack(side=tk.LEFT)
        op_combo = ttk.Combobox(
            row, textvariable=op_var, values=OPERATORS, width=4, state="readonly"
        )
        op_combo.pack(side=tk.LEFT, padx=3)
        self._no_combo_wheel(op_combo)
        ttk.Button(
            row, text="▲", width=2, command=lambda: self._move_cutpoint(row, -1)
        ).pack(side=tk.LEFT)
        ttk.Button(
            row, text="▼", width=2, command=lambda: self._move_cutpoint(row, +1)
        ).pack(side=tk.LEFT)
        ttk.Button(
            row, text="✕", width=2, command=lambda: self._del_cutpoint(row)
        ).pack(side=tk.LEFT)
        self._cutpoint_rows.append((row, val_var, op_var))
        self._repack_cutpoints()

    def _del_cutpoint(self, row):
        self._cutpoint_rows = [r for r in self._cutpoint_rows if r[0] is not row]
        row.destroy()

    def _move_cutpoint(self, row, delta):
        rows = self._cutpoint_rows
        idx = next((i for i, r in enumerate(rows) if r[0] is row), None)
        if idx is None:
            return
        j = idx + delta
        if 0 <= j < len(rows):
            rows[idx], rows[j] = rows[j], rows[idx]
            self._repack_cutpoints()

    def _repack_cutpoints(self):
        """Re-pack rows top-to-bottom in their current list order."""
        for r in self._cutpoint_rows:
            r[0].pack_forget()
        for r in self._cutpoint_rows:
            r[0].pack(fill=tk.X, pady=1)

    def _clear_cutpoints(self):
        for row, _, _ in self._cutpoint_rows:
            row.destroy()
        self._cutpoint_rows = []

    # -- categorical group editor --------------------------------------------
    def _add_cat_row(self, category, group_id):
        row = ttk.Frame(self.cut_frame)
        row.pack(fill=tk.X, pady=1)
        ttk.Label(row, text=str(category)[:20], width=20, anchor="w").pack(side=tk.LEFT)
        ttk.Label(row, text="group:").pack(side=tk.LEFT)
        g_var = tk.StringVar(value=str(group_id))
        ttk.Entry(row, textvariable=g_var, width=4).pack(side=tk.LEFT, padx=2)
        self._cat_rows.append((row, category, g_var))

    def _clear_cat_rows(self):
        for row, _, _ in self._cat_rows:
            row.destroy()
        self._cat_rows = []

    def _clear_editor(self):
        self._clear_cutpoints()
        self._clear_cat_rows()

    def _load_candidates(self):
        nid = self._selected_leaf_id()
        if nid is None:
            return
        self.var_list.delete(0, tk.END)
        self._candidates = {}
        try:
            results = self.tree.candidate_splits(nid)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Candidate splits failed", str(exc))
            return
        for r in results:
            self._candidates[r["variable"]] = r
            chi2 = r["chi2"]
            p = r["pvalue"]
            chi2_s = "    n/a" if chi2 is None or np.isnan(chi2) else f"{chi2:7.1f}"
            p_s = "   n/a" if p is None or np.isnan(p) else f"{p:7.1g}"
            flag = "" if r["splittable"] else "  (p>=.05)"
            self.var_list.insert(
                tk.END, f"{str(r['variable'])[:14]:<14} χ²={chi2_s} p={p_s}{flag}"
            )
            if not r["splittable"]:
                self.var_list.itemconfig(tk.END, foreground="gray")
        self._clear_editor()
        self.specials_label.config(text="")
        self.cut_note.config(text="")
        self._current_var = None

    def _select_variable(self):
        sel = self.var_list.curselection()
        if not sel:
            return
        var = list(self._candidates.keys())[sel[0]]
        cand = self._candidates[var]
        self._current_var = var
        self._clear_editor()

        conditions = cand["conditions"]
        specials = [c for c in conditions if c.kind in ("eq", "isnull")]
        self._current_specials = specials
        if specials:
            self.specials_label.config(
                text="Forced bins: " + " | ".join(c.describe() for c in specials)
            )
        else:
            self.specials_label.config(text="")

        if cand["dtype"] == "numerical":
            self._add_cut_btn.state(["!disabled"])
            thresholds = sorted(
                {
                    b
                    for c in conditions
                    if c.kind == "range"
                    for b in (c.low, c.high)
                    if b is not None
                }
            )
            for t in thresholds:
                self._add_cutpoint(value=_fmt_num(t), op="<")
            self.cut_note.config(
                text="Edit / add / delete / reorder (▲▼) cut-offs. They must be "
                "strictly monotonic (smallest→largest or largest→smallest) to split."
            )
        else:
            # editable categorical groups: one row per category with a group #
            self._add_cut_btn.state(["disabled"])
            groups = [c for c in conditions if c.kind == "category"]
            for gi, gcond in enumerate(groups):
                for cat in gcond.categories or ():
                    self._add_cat_row(cat, gi)
            self.cut_note.config(
                text="Categorical: edit each category's group #. Categories with "
                "the same number are merged into one bin; change numbers to regroup."
            )

    def _do_split(self):
        nid = self._selected_leaf_id()
        if nid is None or self._current_var is None:
            messagebox.showwarning("Split", "Select a leaf node and a variable first.")
            return
        var = self._current_var
        cand = self._candidates[var]
        try:
            if cand["dtype"] == "numerical":
                cutpoints = []
                for _, val_var, op_var in self._cutpoint_rows:
                    txt = val_var.get().strip()
                    if txt == "":
                        continue
                    cutpoints.append({"op": op_var.get(), "value": float(txt)})
                vals = [c["value"] for c in cutpoints]
                ascending = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
                # descending needs >= 2 points; otherwise it's vacuously true and
                # would wrongly reverse a single-cut-off split.
                descending = len(vals) >= 2 and all(
                    vals[i] > vals[i + 1] for i in range(len(vals) - 1)
                )
                if not (ascending or descending):
                    messagebox.showwarning(
                        "Split",
                        "Cut-off points must be strictly monotonic — sorted either "
                        "smallest→largest or largest→smallest. Use the ▲▼ buttons "
                        "to reorder them.",
                    )
                    return
                conditions = conditions_from_cutpoints(
                    var, cutpoints, self._current_specials, descending=descending
                )
            else:
                items = []
                for _, cat, g_var in self._cat_rows:
                    gid = g_var.get().strip()
                    if gid == "":
                        messagebox.showwarning(
                            "Split", "Assign a group number to every category."
                        )
                        return
                    items.append((cat, gid))
                conditions = categorical_conditions_from_groups(
                    var, items, self._current_specials
                )
            res = self.tree.split(nid, var, conditions)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Split failed", str(exc))
            return
        messagebox.showinfo(
            "Split",
            f"Split node {nid} on '{var}'.\n"
            f"χ²={_fmt(res['chi2'])}  p={_fmtp(res['pvalue'])}\n"
            f"{len(res['children'])} child nodes created.",
        )
        self._refresh_controls()
        self.redraw()

    # ---------------------------------------------------------- combine panel
    def _build_combine_section(self):
        f = self._section("2. Combine child nodes")
        ttk.Label(f, text="Parent node:").pack(anchor="w", padx=4)
        self.combine_parent = ttk.Combobox(f, state="readonly")
        self.combine_parent.pack(fill=tk.X, padx=4)
        self._no_combo_wheel(self.combine_parent)
        self.combine_parent.bind(
            "<<ComboboxSelected>>", lambda e: self._load_combine_children()
        )
        ttk.Label(f, text="Children (multi-select):").pack(anchor="w", padx=4)
        self.combine_list = tk.Listbox(
            f, height=6, selectmode=tk.EXTENDED, font=MONO, exportselection=False
        )
        self.combine_list.pack(fill=tk.X, padx=4)
        self._own_wheel(self.combine_list)
        ttk.Button(f, text="Combine selected", command=self._do_combine).pack(
            anchor="e", padx=4, pady=4
        )

    def _load_combine_children(self):
        self.combine_list.delete(0, tk.END)
        pid = _parse_id(self.combine_parent.get())
        if pid is None:
            return
        self._combine_child_ids = []
        for cid in self.tree.nodes[pid].children:
            child = self.tree.nodes[cid]
            tag = "leaf" if child.is_leaf else "internal"
            self.combine_list.insert(
                tk.END,
                f"#{cid} n={child.n_samples} bad={child.target_rate:.1%} [{tag}]  "
                f"{child.condition.describe()[:24]}",
            )
            self._combine_child_ids.append(cid)
            if not child.is_leaf:
                self.combine_list.itemconfig(tk.END, foreground="gray")

    def _do_combine(self):
        pid = _parse_id(self.combine_parent.get())
        sel = self.combine_list.curselection()
        if pid is None or len(sel) < 2:
            messagebox.showwarning("Combine", "Select a parent and >= 2 child nodes.")
            return
        child_ids = [self._combine_child_ids[i] for i in sel]
        try:
            res = self.tree.combine(pid, child_ids)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Combine failed", str(exc))
            return
        messagebox.showinfo(
            "Combine",
            f"Merged {len(child_ids)} children of node {pid}.\n"
            f"New χ²={_fmt(res['chi2'])}  p={_fmtp(res['pvalue'])}",
        )
        self._refresh_controls()
        # Keep the same parent selected and repopulate its (now fewer) children,
        # so the multi-select list no longer shows the merged-away nodes.
        if pid in self.tree.nodes:
            for val in self.combine_parent["values"]:
                if _parse_id(val) == pid:
                    self.combine_parent.set(val)
                    break
            self._load_combine_children()
        self.redraw()

    # ------------------------------------------------------------ prune panel
    def _build_prune_section(self):
        f = self._section("3. Prune a node")
        ttk.Label(f, text="Node (subtree will be removed):").pack(anchor="w", padx=4)
        self.prune_combo = ttk.Combobox(f, state="readonly")
        self.prune_combo.pack(fill=tk.X, padx=4)
        self._no_combo_wheel(self.prune_combo)
        ttk.Button(f, text="Prune node", command=self._do_prune).pack(
            anchor="e", padx=4, pady=4
        )

    def _do_prune(self):
        nid = _parse_id(self.prune_combo.get())
        if nid is None:
            messagebox.showwarning("Prune", "Select a node to prune.")
            return
        if not messagebox.askyesno(
            "Prune", f"Remove all nodes below node {nid}?"
        ):
            return
        try:
            res = self.tree.prune(nid)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Prune failed", str(exc))
            return
        messagebox.showinfo("Prune", f"Removed {res['removed']} nodes.")
        self._refresh_controls()
        # Clear the now-stale prune selection.
        self.prune_combo.set("")
        # Drop any combine selection that pointed at a node removed by the prune.
        cpid = _parse_id(self.combine_parent.get())
        if cpid is None or cpid not in self.tree.nodes or self.tree.nodes[cpid].is_leaf:
            self.combine_parent.set("")
            self.combine_list.delete(0, tk.END)
            self._combine_child_ids = []
        self.redraw()

    # ---------------------------------------------------------- history panel
    def _build_history_section(self):
        f = self._section("4. Operation history")
        self.history_list = tk.Listbox(f, height=7, font=MONO)
        self.history_list.pack(fill=tk.X, padx=4)
        self._own_wheel(self.history_list)
        ttk.Button(f, text="Undo last operation", command=self._do_undo).pack(
            anchor="e", padx=4, pady=4
        )

    def _do_undo(self):
        if not self.tree.undo():
            messagebox.showinfo("Undo", "Nothing to undo.")
            return
        self._refresh_controls()
        self.redraw()

    # ----------------------------------------------------------- action panel
    def _build_actions_section(self):
        f = self._section("5. Actions")
        ttk.Button(f, text="Compute Gini (training)", command=self._do_gini).pack(
            fill=tk.X, padx=4, pady=2
        )
        ttk.Button(f, text="Save tree (pickle)…", command=self._do_save).pack(
            fill=tk.X, padx=4, pady=2
        )
        ttk.Button(f, text="Export rules to Excel…", command=self._do_export).pack(
            fill=tk.X, padx=4, pady=2
        )
        ttk.Button(f, text="Save tree image (PNG)…", command=self._do_save_png).pack(
            fill=tk.X, padx=4, pady=2
        )

    def _do_gini(self):
        try:
            g = self.tree.gini()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Gini", str(exc))
            return
        messagebox.showinfo("Gini", f"Training Gini = {g:.4f}")

    def _do_save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl", filetypes=[("Pickle", "*.pkl")]
        )
        if not path:
            return
        self.tree.save(path)
        messagebox.showinfo("Save", f"Saved tree to\n{path}")

    def _do_export(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")]
        )
        if not path:
            return
        self.tree.export_rules_excel(path)
        messagebox.showinfo("Export", f"Exported rules to\n{path}")

    def _do_save_png(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG image", "*.png")]
        )
        if not path:
            return
        try:
            self.tree.save_png(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Save PNG", str(exc))
            return
        messagebox.showinfo("Save PNG", f"Saved tree image to\n{path}")

    # --------------------------------------------------------------- refresh
    def _selected_leaf_id(self):
        return _parse_id(self.leaf_combo.get())

    def _refresh_controls(self):
        leaves = sorted(self.tree.leaves(), key=lambda n: n.node_id)
        self.leaf_combo["values"] = [
            f"#{n.node_id}  n={n.n_samples}  bad={n.target_rate:.1%}" for n in leaves
        ]
        internal = sorted(self.tree.internal_nodes(), key=lambda n: n.node_id)
        internal_vals = [
            f"#{n.node_id}  split={n.split_variable}  ({len(n.children)} kids)"
            for n in internal
        ]
        self.combine_parent["values"] = internal_vals
        self.prune_combo["values"] = internal_vals

        self.history_list.delete(0, tk.END)
        for i, h in enumerate(self.tree.history, 1):
            self.history_list.insert(tk.END, f"{i}. {h}")

        # clear stale split selections
        self.var_list.delete(0, tk.END)
        self._clear_editor()
        self.specials_label.config(text="")
        self.cut_note.config(text="")
        self._current_var = None

    # ----------------------------------------------------------------- zoom
    def zoom_in(self):
        self._set_zoom(self._zoom * 1.25)

    def zoom_out(self):
        self._set_zoom(self._zoom / 1.25)

    def zoom_reset(self):
        self._set_zoom(1.0)

    def zoom_fit(self):
        """Scale so the tree image fits the current canvas width."""
        if self._base_img is None:
            return
        cw = self.canvas.winfo_width()
        if cw > 1:
            self._set_zoom(cw / self._base_img.width)

    def _set_zoom(self, zoom: float):
        self._zoom = max(0.1, min(8.0, zoom))
        self._render_image()

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_shift_wheel(self, event):
        self.canvas.xview_scroll(int(-event.delta / 120), "units")

    def _on_ctrl_wheel(self, event):
        self.zoom_in() if event.delta > 0 else self.zoom_out()
        return "break"

    def _own_wheel(self, listbox):
        """Let a Listbox consume the wheel itself so the sidebar doesn't scroll too."""

        def handler(event):
            listbox.yview_scroll(int(-event.delta / 120), "units")
            return "break"  # stop the event before the sidebar's bind_all handler

        listbox.bind("<MouseWheel>", handler)

    def _no_combo_wheel(self, combo):
        """Stop a collapsed Combobox from changing value on wheel; scroll sidebar."""

        def handler(event):
            self._menu_canvas.yview_scroll(int(-event.delta / 120), "units")
            return "break"  # suppress the default value-change class binding

        combo.bind("<MouseWheel>", handler)

    def _bind_menu_wheel(self, event):
        self._menu_canvas.bind_all("<MouseWheel>", self._on_menu_wheel)

    def _unbind_menu_wheel(self, event):
        self._menu_canvas.unbind_all("<MouseWheel>")

    def _on_menu_wheel(self, event):
        self._menu_canvas.yview_scroll(int(-event.delta / 120), "units")
        return "break"

    def _on_pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _on_pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_pan_end(self, event):
        self.canvas.config(cursor="")

    def _render_image(self):
        """(Re)paint the cached base image at the current zoom factor."""
        if self._base_img is None:
            return
        from PIL import Image, ImageTk

        w = max(1, int(self._base_img.width * self._zoom))
        h = max(1, int(self._base_img.height * self._zoom))
        resized = self._base_img.resize((w, h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, w, h))
        self.zoom_label.config(text=f"{self._zoom * 100:.0f}%")

    # ---------------------------------------------------------------- redraw
    def redraw(self):
        """Re-render the tree to a base image, then paint it at current zoom."""
        try:
            from PIL import Image

            fig = build_figure(self.tree)
            png = figure_to_png(fig)
            self._base_img = Image.open(io.BytesIO(png))
            self._render_image()
        except Exception as exc:  # noqa: BLE001
            self._base_img = None
            self.canvas.delete("all")
            self.canvas.create_text(
                20,
                20,
                anchor="nw",
                fill="red",
                text=f"Could not render tree image:\n{exc}\n\n"
                "Ensure 'kaleido' and 'pillow' are installed.",
            )


def _parse_id(text):
    if not text:
        return None
    try:
        return int(text.split()[0].lstrip("#"))
    except (ValueError, IndexError):
        return None


def _fmt_num(x):
    if float(x).is_integer():
        return str(int(x))
    return f"{x:.6g}"


def _fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.2f}"


def _fmtp(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.3g}"
