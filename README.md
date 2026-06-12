# ManualChaidTree

An interactive, **manually-grown** CHAID-style decision tree for binary (0/1)
targets — the classic credit-scoring workflow where a modeler hand-builds a
scorecard tree instead of letting an algorithm grow it automatically.

`train()` opens a tkinter window with a tree-preview canvas and a side menu for
splitting, combining, pruning, and undoing — with variables ranked by Chi² and
optbinning-suggested (but editable) cut-offs.

## Install

```bash
pip install -r requirements.txt
```

Dependencies: `optbinning`, `scipy`, `numpy`, `pandas`, `plotly`, `kaleido`,
`pillow`, `openpyxl` (tkinter ships with Python). `kaleido` renders the Plotly
tree figure to a PNG that is painted onto the tkinter canvas.

## Dataset for the demo

Download **Default of Credit Card Clients** from Kaggle
(<https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset>)
and place `UCI_Credit_Card.csv` in the project root. Then run
`notebooks/manual_chaid_demo.ipynb`.

## Quick start

```python
import pandas as pd
from manual_chaid import ManualChaidTree

df = pd.read_csv("UCI_Credit_Card.csv").drop(columns=["ID"])
y = df.pop("default.payment.next.month")

tree = ManualChaidTree(df, y)
tree.train(
    pre_separate_zero=["PAY_0", "PAY_2"],        # value 0 gets its own node
    pre_separate_values={"PAY_0": -1},           # value -1 gets its own node
    min_samples=500,                             # floor on node size
    max_children=4,                              # max optbinning bins per split
)
# ... build the tree in the UI, then Save / Export from the Actions panel.

print("Gini:", tree.gini())
tree.save("tree.pkl")

# Reload WITH data -> editable; WITHOUT data -> view / score only.
t2 = ManualChaidTree.load("tree.pkl", X=df, y=y)
t3 = ManualChaidTree.load("tree.pkl")            # no data
print(t3.predict_proba(df.head()))
```

## How it works

* **Rules, not rows** — each node stores only the predicate that carves it from
  its parent plus cached stats, so pickles are small and data-free.
* **Special handling** — `pre_separate_zero`, `pre_separate_values`, and missing
  values each get a forced bin and are excluded from optbinning, but are still
  included in the Chi²/p calculation for the split.
* **Editable cut-offs** — numeric thresholds (with `<=, <, >, >=, =` operators)
  can be added/edited/deleted; Chi²/p are **recomputed from the current cut-offs**
  before the split is committed. Categorical splits follow optbinning's grouping.
* **Undo** — split / combine / prune are snapshot-backed; one Undo button reverts
  the last operation and removes it from the history (the undo is not recorded).

## Module map

| File | Responsibility |
|------|----------------|
| `manual_chaid/node.py`    | `ChaidNode` + `Condition` rule model |
| `manual_chaid/stats.py`   | Chi-square test, AUC / Gini |
| `manual_chaid/binning.py` | optbinning wrapper, cut-off ↔ condition translation |
| `manual_chaid/tree.py`    | `ManualChaidTree` core logic (headless) |
| `manual_chaid/render.py`  | Plotly figure + kaleido PNG |
| `manual_chaid/ui.py`      | tkinter builder window |
