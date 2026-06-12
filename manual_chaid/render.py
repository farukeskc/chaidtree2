"""Plotly rendering of the tree, plus kaleido PNG export for the tkinter canvas."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def _layout(tree) -> dict:
    """Tidy-tree positions: leaves laid out left->right, parents centred on kids."""
    pos: dict[int, tuple] = {}
    counter = [0]

    def rec(nid):
        node = tree.nodes[nid]
        if node.is_leaf:
            x = float(counter[0])
            counter[0] += 1
        else:
            xs = [rec(c) for c in node.children]
            x = float(np.mean(xs))
        pos[nid] = (x, -float(node.depth))
        return x

    rec(0)
    return pos


def _leaf_color(rate: float) -> str:
    """White (low bad ratio) -> red (high bad ratio)."""
    r = 255
    g = int(255 - 215 * max(0.0, min(1.0, rate)))
    b = int(255 - 215 * max(0.0, min(1.0, rate)))
    return f"rgb({r},{g},{b})"


def _node_text(tree, node) -> str:
    lines = [f"<b>Node {node.node_id}</b>"]
    if node.condition is not None:  # only the cut-off value (parent shows the feature)
        lines.append(f"<i>{node.condition.describe_value()}</i>")
    lines += [
        f"n={node.n_samples}",
        f"bad={node.n_target} ({node.target_rate:.1%})",
    ]
    if not node.is_leaf:
        chi2 = node.split_chi2
        p = node.split_pvalue
        chi2_s = "n/a" if chi2 is None or np.isnan(chi2) else f"{chi2:.1f}"
        p_s = "n/a" if p is None or np.isnan(p) else f"{p:.2g}"
        lines.append(f"<b>{node.split_variable}</b>")
        lines.append(f"χ²={chi2_s}  p={p_s}")
    return "<br>".join(lines)


def build_figure(tree) -> go.Figure:
    pos = _layout(tree)
    fig = go.Figure()

    # edges (the cut-off value now lives inside each child node box)
    for node in tree.nodes.values():
        x0, y0 = pos[node.node_id]
        for cid in node.children:
            x1, y1 = pos[cid]
            fig.add_shape(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line=dict(color="#888", width=1),
                layer="below",
            )

    # nodes are drawn as bordered annotations so the box auto-sizes to the
    # text and can never overflow; the opaque fill masks the edge underneath.
    for node in tree.nodes.values():
        x, y = pos[node.node_id]
        if node.is_leaf:
            fill = _leaf_color(node.target_rate)
            line_color = "#c0392b"
        else:
            fill = "#dbe9f6"
            line_color = "#3a6ea5"
        fig.add_annotation(
            x=x,
            y=y,
            text=_node_text(tree, node),
            showarrow=False,
            font=dict(size=10, color="#111"),
            align="center",
            bgcolor=fill,
            bordercolor=line_color,
            borderwidth=1.5,
            borderpad=6,
        )

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    fig.update_xaxes(
        visible=False, range=[min(xs) - 1, max(xs) + 1]
    )
    fig.update_yaxes(
        visible=False, range=[min(ys) - 1, max(ys) + 1]
    )
    n_leaves = max(1, len(tree.leaves()))
    width = max(640, int(n_leaves * 170))
    height = max(420, int((tree.depth() + 1) * 165))
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(text="Manual CHAID Tree", x=0.5, font=dict(size=14)),
        showlegend=False,
    )
    return fig


def figure_to_png(fig: go.Figure, scale: float = 2.0) -> bytes:
    """Render the Plotly figure to PNG bytes (requires the kaleido package)."""
    return fig.to_image(format="png", scale=scale)
