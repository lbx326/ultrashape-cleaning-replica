"""Aggregate benchmark JSON reports into Markdown + percentile stats.

Usage:
    python scripts/summarize_benchmark.py \\
        --reports-dir outputs/benchmark/reports \\
        --out docs/benchmark.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _get(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return default
    return cur if cur is not None else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    reports = sorted(args.reports_dir.glob("*.json"))
    if not reports:
        print(f"No reports in {args.reports_dir}")
        return 1

    rows = []
    for r in reports:
        try:
            data = json.loads(r.read_text(encoding="utf-8"))
            rows.append(data)
        except Exception as e:
            print(f"  !!! skipping {r.name}: {e}")

    lines = []
    lines.append("# Benchmark on HSSD samples")
    lines.append("")
    lines.append(f"Aggregated from {len(rows)} mesh reports in "
                 f"`{args.reports_dir}`.")
    lines.append("")

    # --- Headline table ---
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Metric | Mean | Median | Min | Max |")
    lines.append("|--------|-----:|-------:|----:|----:|")

    metrics = {
        "Total wall time (s)": [r.get("seconds_total") for r in rows],
        "Stage 1 wall time (s)": [_get(r, "stage_timings", "stage1_watertighten") for r in rows],
        "Stage 2 wall time (s)": [_get(r, "stage_timings", "stage2_vlm_filter") for r in rows],
        "Stage 3 wall time (s)": [_get(r, "stage_timings", "stage3_canonicalize") for r in rows],
        "Stage 4 wall time (s)": [_get(r, "stage_timings", "stage4_filter_geometry") for r in rows],
        "Stage 1 chamfer vs input": [_get(r, "stage_reports", "stage1_watertighten", "chamfer_to_input") for r in rows],
        "Stage 4 ray sign agreement": [_get(r, "stage_reports", "stage4_filter_geometry", "metrics", "ray_sign_agreement") for r in rows],
        "Stage 4 VAE chamfer": [_get(r, "stage_reports", "stage4_filter_geometry", "metrics", "vae", "chamfer") for r in rows],
    }
    for name, vals in metrics.items():
        clean = [v for v in vals if isinstance(v, (int, float))]
        if not clean:
            continue
        lines.append(f"| {name} | {statistics.mean(clean):.3f} | "
                     f"{statistics.median(clean):.3f} | "
                     f"{min(clean):.3f} | {max(clean):.3f} |")

    lines.append("")
    lines.append("## Binary rates")
    lines.append("")
    wt = [_get(r, "stage_reports", "stage1_watertighten", "is_watertight") for r in rows]
    winding = [_get(r, "stage_reports", "stage1_watertighten", "is_winding_consistent") for r in rows]
    accepted = [r.get("accepted") for r in rows]
    s2_accepted = [_get(r, "stage_reports", "stage2_vlm_filter", "accepted") for r in rows]
    def _frac(vals, val=True):
        hits = sum(1 for v in vals if v is val)
        total = sum(1 for v in vals if v is not None)
        return hits, total, (100 * hits / total if total else 0.0)
    h, t, f = _frac(wt)
    lines.append(f"- Watertight after Stage 1: {h}/{t} ({f:.1f}%)")
    h, t, f = _frac(winding)
    lines.append(f"- Winding consistent after Stage 1: {h}/{t} ({f:.1f}%)")
    h, t, f = _frac(s2_accepted)
    lines.append(f"- Stage 2 VLM accept: {h}/{t} ({f:.1f}%)")
    h, t, f = _frac(accepted)
    lines.append(f"- Overall pipeline accept: {h}/{t} ({f:.1f}%)")
    lines.append("")

    # --- Per-mesh table ---
    lines.append("## Per-mesh breakdown")
    lines.append("")
    lines.append("| # | sha256[:10] | V_in | F_in | V_out | F_out | "
                 "WT | wind | chamf | ray_agree | VAE_chamf | class | qual | accept | reasons |")
    lines.append("|---|-------------|-----:|-----:|------:|------:|"
                 "----|------|------:|----------:|----------:|-------|-----:|:------|:--------|")
    for i, r in enumerate(rows):
        sha = r.get("sha256", "?")[:10]
        inp = r.get("input_summary", {})
        outp = r.get("output_summary", {}) or {}
        s1 = _get(r, "stage_reports", "stage1_watertighten") or {}
        s2 = _get(r, "stage_reports", "stage2_vlm_filter") or {}
        s4 = _get(r, "stage_reports", "stage4_filter_geometry", "metrics") or {}
        vae = _get(s4, "vae") or {}
        reasons = ";".join(r.get("rejection_reasons") or []) or "-"
        lines.append(
            f"| {i+1} | {sha} | {inp.get('num_vertices', '?')} | "
            f"{inp.get('num_faces', '?')} | {outp.get('num_vertices', '?')} | "
            f"{outp.get('num_faces', '?')} | "
            f"{'Y' if s1.get('is_watertight') else 'N'} | "
            f"{'Y' if s1.get('is_winding_consistent') else 'N'} | "
            f"{s1.get('chamfer_to_input', 0):.4f} | "
            f"{s4.get('ray_sign_agreement', 0):.3f} | "
            f"{vae.get('chamfer', '?')} | "
            f"{s2.get('object_class', '?')} | "
            f"{s2.get('aesthetic_quality', '?')} | "
            f"{'Y' if r.get('accepted') else 'N'} | "
            f"{reasons} |"
        )
    lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
