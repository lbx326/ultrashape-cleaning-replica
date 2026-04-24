"""Dash viewer for stress_v3 benchmark results.

Shows each of the 50 meshes with its 4-view render, accept/reject verdict,
rejection reasons, VLM evaluation, and before/after mesh stats.

Layout: 2x2 grid per page (4 meshes at a time) with prev/next pagination
and an accepted-only / rejected-only filter.

Run (remote):
    /moganshan/afs_a/lbx/env/dage/bin/python scripts/stress_viewer.py

Access (local):
    ssh -L 8051:localhost:8051 -p 10125 lbx@180.184.148.169
    # browse http://localhost:8051
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html

# ──────────────────────────── Config ─────────────────────────────
DEFAULT_ROOT = Path(
    "/moganshan/afs_a/lbx/utils/mesh_clean/outputs/stress_v3"
)
PER_PAGE = 4  # 2x2 grid


def load_entries(root: Path):
    csv = pd.read_csv(root / "summary.csv")
    entries = []
    for _, row in csv.iterrows():
        sha = row["sha256"]
        entry = {"sha": sha, "row": row.to_dict()}
        grid_path = root / "_renders" / f"{sha}.grid.png"
        if grid_path.exists():
            entry["grid_b64"] = base64.b64encode(grid_path.read_bytes()).decode()
        vlm_path = root / "_renders" / f"{sha}.vlm.json"
        if vlm_path.exists():
            try:
                entry["vlm"] = json.loads(vlm_path.read_text())
            except Exception:
                entry["vlm"] = {}
        rep_path = root / sha / "report.json"
        if rep_path.exists():
            try:
                entry["report"] = json.loads(rep_path.read_text())
            except Exception:
                entry["report"] = {}
        entries.append(entry)
    return entries


# ──────────────────────────── Card render ─────────────────────────────


def _fmt_mesh_stats(summary: dict | None) -> str:
    if not summary:
        return "—"
    parts = []
    if summary.get("num_faces") is not None:
        parts.append(f"{summary['num_faces']:,} faces")
    if summary.get("num_vertices") is not None:
        parts.append(f"{summary['num_vertices']:,} v")
    wt = summary.get("is_watertight")
    if wt is not None:
        parts.append("watertight" if wt else "not-watertight")
    vol = summary.get("volume")
    if vol is not None:
        parts.append(f"vol={vol:.2e}")
    area = summary.get("area")
    if area is not None:
        parts.append(f"area={area:.2e}")
    return " | ".join(parts)


def render_card(entry: dict) -> html.Div:
    row = entry["row"]
    sha = entry["sha"]
    accepted = bool(row.get("accepted"))
    reasons_raw = row.get("reasons") or ""
    reasons = [r for r in str(reasons_raw).split("|") if r]

    report = entry.get("report", {})
    in_summ = report.get("input_summary", {})
    out_summ = report.get("output_summary", {})

    vlm = entry.get("vlm", {})

    input_path = row.get("input", "")
    input_name = Path(input_path).name if input_path else ""

    border_color = "#1c6b2a" if accepted else "#a6232c"
    badge_bg = "#d8f0d8" if accepted else "#ffe0e0"
    badge_text = "✓ ACCEPTED" if accepted else "✗ REJECTED"

    # VLM flags — only show True ones prominently
    vlm_flags = []
    for key, label in [
        ("is_primitive", "primitive"),
        ("is_ground_plane", "ground"),
        ("is_noisy_scan", "noisy"),
        ("is_fragmented", "fragmented"),
    ]:
        val = vlm.get(key)
        if val is True:
            vlm_flags.append(
                html.Span(f"  {label}  ", style={
                    "background": "#ffe0e0", "color": "#a6232c",
                    "padding": "1px 6px", "margin": "0 2px",
                    "borderRadius": "3px", "fontSize": "10px",
                })
            )

    aesthetic = vlm.get("aesthetic_quality", "—")
    vlm_class = vlm.get("classification", "—")
    vlm_reason = vlm.get("reasoning", "")

    s1 = row.get("s1_watertight")
    s1_chamfer = row.get("s1_chamfer")
    s4_ray = row.get("s4_ray_agreement")
    s4_vae = row.get("s4_vae_chamfer")

    img_src = (
        f"data:image/png;base64,{entry['grid_b64']}"
        if "grid_b64" in entry
        else None
    )

    return html.Div([
        html.Div([
            html.Div([
                html.Span(f"#{sha[:10]}", style={
                    "fontFamily": "monospace", "fontSize": "11px", "color": "#555",
                }),
                html.Span(f"  {input_name}", style={
                    "fontSize": "11px", "color": "#555", "marginLeft": "6px",
                }),
            ]),
            html.Div(badge_text, style={
                "fontWeight": "bold",
                "background": badge_bg,
                "color": border_color,
                "padding": "2px 8px",
                "borderRadius": "4px",
                "fontSize": "13px",
            }),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "marginBottom": "6px",
        }),

        html.Img(src=img_src, style={
            "width": "100%", "display": "block", "background": "#f5f5f5",
        }) if img_src else html.Div("(no render)", style={
            "padding": "40px", "background": "#f5f5f5",
            "textAlign": "center", "color": "#888",
        }),

        html.Div([
            html.Span(f"VLM aes={aesthetic} ", style={"fontSize": "11px"}),
            html.Span(f"class={vlm_class} ", style={"fontSize": "11px"}),
            *vlm_flags,
        ], style={"marginTop": "6px"}),

        html.Details([
            html.Summary(f"VLM reasoning ({len(vlm_reason)} chars)",
                         style={"fontSize": "11px", "cursor": "pointer"}),
            html.Div(vlm_reason, style={
                "fontSize": "11px", "color": "#333", "padding": "4px 8px",
                "background": "#fafafa", "maxHeight": "120px",
                "overflowY": "auto", "whiteSpace": "pre-wrap",
            }),
        ], style={"marginTop": "4px"}) if vlm_reason else None,

        html.Div([
            html.Div(f"Reasons:", style={"fontSize": "11px", "color": "#a6232c",
                                          "fontWeight": "bold"}) if reasons else None,
            *[html.Div(f"  • {r}", style={
                "fontSize": "11px", "color": "#a6232c", "fontFamily": "monospace",
            }) for r in reasons],
        ], style={"marginTop": "6px"}) if reasons else html.Div(),

        html.Details([
            html.Summary("stage metrics", style={"fontSize": "11px", "cursor": "pointer"}),
            html.Pre(
                json.dumps({
                    "input": _fmt_mesh_stats(in_summ),
                    "output": _fmt_mesh_stats(out_summ),
                    "total_s": row.get("total_s"),
                    "s1_watertight": s1,
                    "s1_chamfer": s1_chamfer,
                    "s4_ray_agreement": s4_ray,
                    "s4_vae_chamfer": s4_vae,
                }, indent=2, default=str),
                style={"fontSize": "10px", "background": "#fafafa",
                       "padding": "6px", "margin": "4px 0",
                       "maxHeight": "160px", "overflowY": "auto"},
            ),
        ], style={"marginTop": "4px"}),

    ], style={
        "border": f"3px solid {border_color}",
        "borderRadius": "8px",
        "padding": "10px",
        "background": "#fff",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    })


# ──────────────────────────── App ─────────────────────────────


def build_app(root: Path):
    entries = load_entries(root)
    n_total = len(entries)
    n_accept = sum(1 for e in entries if bool(e["row"].get("accepted")))
    n_reject = n_total - n_accept
    print(f"[stress_viewer] loaded {n_total} entries ({n_accept} accepted, {n_reject} rejected)")

    app = Dash(__name__, title="Stress v3 Review")

    app.layout = html.Div([
        html.Div([
            html.H3("UltraShape Cleaning — Stress v3 (50 HSSD meshes)",
                    style={"margin": "0"}),
            html.Div([
                html.Span(f"  ✓ {n_accept} accepted  ",
                          style={"background": "#d8f0d8", "color": "#1c6b2a",
                                 "padding": "2px 8px", "borderRadius": "4px",
                                 "marginRight": "6px"}),
                html.Span(f"  ✗ {n_reject} rejected  ",
                          style={"background": "#ffe0e0", "color": "#a6232c",
                                 "padding": "2px 8px", "borderRadius": "4px"}),
            ], style={"fontSize": "13px"}),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "padding": "8px 16px",
            "borderBottom": "1px solid #ddd", "background": "#fafafa",
        }),

        html.Div([
            dcc.RadioItems(
                id="filter",
                options=[
                    {"label": f" all ({n_total}) ", "value": "all"},
                    {"label": f" accepted ({n_accept}) ", "value": "accepted"},
                    {"label": f" rejected ({n_reject}) ", "value": "rejected"},
                ],
                value="all",
                inline=True,
                inputStyle={"marginRight": "4px", "marginLeft": "10px"},
                style={"fontSize": "13px"},
            ),
            html.Button("← Prev", id="prev", n_clicks=0,
                        style={"marginLeft": "20px", "padding": "4px 10px"}),
            html.Span(id="page-label", style={"margin": "0 12px", "fontFamily": "monospace"}),
            html.Button("Next →", id="next", n_clicks=0,
                        style={"padding": "4px 10px"}),
        ], style={"padding": "8px 16px", "borderBottom": "1px solid #eee",
                  "display": "flex", "alignItems": "center"}),

        html.Div(id="grid", style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "14px",
            "padding": "14px",
            "background": "#f0f0f0",
            "minHeight": "calc(100vh - 120px)",
        }),

        dcc.Store(id="state", data={"page": 0, "filter": "all"}),
    ])

    @app.callback(
        [Output("grid", "children"),
         Output("page-label", "children"),
         Output("state", "data")],
        [Input("prev", "n_clicks"), Input("next", "n_clicks"),
         Input("filter", "value")],
        State("state", "data"),
    )
    def render_page(prev, nxt, filt, state):
        page = state.get("page", 0)
        prev_filt = state.get("filter", "all")

        if filt != prev_filt:
            page = 0

        if filt == "accepted":
            subset = [e for e in entries if bool(e["row"].get("accepted"))]
        elif filt == "rejected":
            subset = [e for e in entries if not bool(e["row"].get("accepted"))]
        else:
            subset = entries

        total_pages = max(1, (len(subset) + PER_PAGE - 1) // PER_PAGE)

        trig = ctx.triggered_id
        if trig == "prev":
            page = max(0, page - 1)
        elif trig == "next":
            page = min(total_pages - 1, page + 1)
        page = min(page, total_pages - 1)

        start = page * PER_PAGE
        cards = [render_card(e) for e in subset[start:start + PER_PAGE]]
        while len(cards) < PER_PAGE:
            cards.append(html.Div(style={"border": "2px dashed #ddd",
                                         "borderRadius": "8px",
                                         "background": "#f8f8f8"}))

        return cards, f"Page {page + 1} / {total_pages}  ({len(subset)} items)", {
            "page": page, "filter": filt
        }

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8051)
    args = ap.parse_args()
    app = build_app(Path(args.root))
    print(f"[serving] http://{args.host}:{args.port}")
    print("[local]   ssh -L 8051:localhost:8051 -p 10125 lbx@180.184.148.169")
    app.run(host=args.host, port=args.port, debug=False)
