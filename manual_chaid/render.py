"""Plotly rendering of the tree, plus kaleido PNG export for the tkinter canvas."""
from __future__ import annotations

import re

import numpy as np
import plotly.graph_objects as go

# Figure geometry — single source of truth, reused for pixel hit-testing.
# x positions are in figure pixels (1 x-unit ~= 1px); y stays in -depth units.
MARGIN_L = 10
MARGIN_R = 10
MARGIN_T = 30
MARGIN_B = 10
PX_PER_DEPTH = 165
MIN_W = 640
MIN_H = 420
# Box width estimation (font size 10) and minimum gap between adjacent boxes.
CHAR_PX = 6.5          # approx. width of one character at font size 10
BOX_PAD = 30.0         # 2*borderpad + 2*borderwidth + safety margin
GAP_PX = 26.0          # minimum horizontal space between adjacent boxes
# Max OR terms shown inside a node box before truncating to "(+N more)".
MAX_OR_TERMS = 2

_TAG_RE = re.compile(r"</?[a-z]+>")


def _estimate_box_width(html_text: str) -> float:
    """Estimate an annotation box's rendered pixel width from its HTML text."""
    longest = 0
    for line in html_text.split("<br>"):
        plain = _TAG_RE.sub("", line)
        longest = max(longest, len(plain))
    return longest * CHAR_PX + BOX_PAD


def _layout(tree):
    """Width-aware tidy layout.

    Returns (pos, box_w, total_w) where pos maps node_id -> (x_px, -depth),
    box_w maps node_id -> estimated box width in px, and total_w is the overall
    layout width in px. Sub-trees are spaced so boxes never overlap: adjacent
    leaves sit box_w/2 + GAP_PX + box_w/2 apart, and a parent box always fits
    inside its sub-tree band.
    """
    box_w: dict[int, float] = {
        nid: _estimate_box_width(_node_text(tree, node))
        for nid, node in tree.nodes.items()
    }
    sub_w: dict[int, float] = {}

    def measure(nid) -> float:
        node = tree.nodes[nid]
        if node.is_leaf:
            sub_w[nid] = box_w[nid]
        else:
            kids = node.children
            children_total = sum(measure(c) for c in kids) + GAP_PX * (len(kids) - 1)
            sub_w[nid] = max(children_total, box_w[nid] + GAP_PX)
        return sub_w[nid]

    measure(0)

    pos: dict[int, tuple] = {}

    def place(nid, left: float):
        node = tree.nodes[nid]
        w = sub_w[nid]
        if node.is_leaf:
            x = left + w / 2.0
        else:
            kids = node.children
            children_total = (
                sum(sub_w[c] for c in kids) + GAP_PX * (len(kids) - 1)
            )
            cursor = left + (w - children_total) / 2.0  # centre the children block
            xs = []
            for c in kids:
                place(c, cursor)
                xs.append(pos[c][0])
                cursor += sub_w[c] + GAP_PX
            x = float(np.mean(xs))
        pos[nid] = (x, -float(node.depth))

    place(0, 0.0)
    return pos, box_w, sub_w[0]


def _fig_dims(tree, total_w: float):
    """Figure (width, height) in px for a given layout width."""
    width = int(max(MIN_W, total_w + MARGIN_L + MARGIN_R))
    height = max(MIN_H, int((tree.depth() + 1) * PX_PER_DEPTH))
    return width, height


def _leaf_color(rate: float) -> str:
    """White (low bad ratio) -> red (high bad ratio)."""
    r = 255
    g = int(255 - 215 * max(0.0, min(1.0, rate)))
    b = int(255 - 215 * max(0.0, min(1.0, rate)))
    return f"rgb({r},{g},{b})"


def _or_display(condition):
    """(label, full_text) for a node's cut-off value.

    For OR conditions with more than MAX_OR_TERMS members, the label is
    truncated to "<term> OR <term> (+N more)" and full_text holds the complete
    description (shown on hover); otherwise full_text is None.
    """
    if condition.kind == "or":
        members = condition.members or ()
        if len(members) > MAX_OR_TERMS:
            shown = " OR ".join(m.describe_value() for m in members[:MAX_OR_TERMS])
            label = f"{shown} (+{len(members) - MAX_OR_TERMS} more)"
            return label, condition.describe_value()
    return condition.describe_value(), None


def _node_text(tree, node) -> str:
    lines = [f"<b>Node {node.node_id}</b>"]
    if node.condition is not None:  # only the cut-off value (parent shows the feature)
        label, _ = _or_display(node.condition)
        lines.append(f"<i>{label}</i>")
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
    pos, _box_w, total_w = _layout(tree)
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

    ys = [p[1] for p in pos.values()]
    fig.update_xaxes(visible=False, range=[0, total_w])
    fig.update_yaxes(
        visible=False, range=[min(ys) - 1, max(ys) + 1]
    )
    width, height = _fig_dims(tree, total_w)
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=MARGIN_L, r=MARGIN_R, t=MARGIN_T, b=MARGIN_B),
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(text="Manual CHAID Tree", x=0.5, font=dict(size=14)),
        showlegend=False,
    )
    return fig


def figure_to_png(fig: go.Figure, scale: float = 2.0) -> bytes:
    """Render the Plotly figure to PNG bytes (requires the kaleido package)."""
    return fig.to_image(format="png", scale=scale)


def node_pixel_centers(tree, scale: float = 2.0) -> dict:
    """Base-image pixel centre of each node, for canvas hover hit-testing.

    Returns {node_id: (px, py, full_text)} where full_text is the untruncated
    condition description when the node's OR label was truncated, else None.
    The geometry mirrors build_figure exactly so the pixels line up with the PNG
    produced by figure_to_png at the same scale.
    """
    pos, _box_w, total_w = _layout(tree)
    ys = [p[1] for p in pos.values()]
    min_y, max_y = min(ys) - 1, max(ys) + 1
    range_y = max_y - min_y

    width, height = _fig_dims(tree, total_w)
    plot_w = width - MARGIN_L - MARGIN_R
    plot_h = height - MARGIN_T - MARGIN_B

    centers: dict = {}
    for nid, (x, y) in pos.items():
        fx = MARGIN_L + (x - 0) / total_w * plot_w
        fy = MARGIN_T + (max_y - y) / range_y * plot_h
        node = tree.nodes[nid]
        full_text = None
        if node.condition is not None:
            _, full_text = _or_display(node.condition)
        centers[nid] = (fx * scale, fy * scale, full_text)
    return centers
