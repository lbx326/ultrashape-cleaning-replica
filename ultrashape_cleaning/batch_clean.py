"""batch_clean.py -- Batch processor for the 4-stage cleaning pipeline.

Processes a directory of meshes, runs all 4 stages on each, and writes
per-mesh cleaned PLY + JSON report plus a summary CSV.

Strategy
--------
- Stages 1 and 3 and 4 each use GPU but are relatively fast (~100s total at
  1024^3 on A100-80GB). We run them sequentially per mesh.
- Stage 2 (VLM) dominates wall-clock time after the first mesh because of
  model load (~1 min) but only ~10s/mesh once loaded. We keep the VLM
  resident across meshes.
- Stage 4's VAE is also kept resident.
- Meshes are processed in order; optionally in a CSV manifest with
  metadata columns passed through.

CLI
---
    python -m ultrashape_cleaning.batch_clean \\
        --input-dir /path/to/hssd/raw \\
        --output-dir /path/to/clean \\
        --limit 20 \\
        --resolution 1024 \\
        --canonicalize geom \\
        [--manifest hssd_subset.csv]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
import time
from pathlib import Path
from typing import Optional

from .clean_mesh import CleanResult, PipelineConfig, clean_mesh_pipeline


def _collect_meshes(input_dir: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pat in patterns:
        out.extend(input_dir.rglob(pat))
    return sorted(out)


def batch_clean(
    input_paths: list[Path],
    output_dir: Path,
    report_dir: Optional[Path] = None,
    render_dir: Optional[Path] = None,
    cfg: Optional[PipelineConfig] = None,
    summary_csv: Optional[Path] = None,
    verbose: bool = True,
) -> list[CleanResult]:
    cfg = cfg or PipelineConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(report_dir) if report_dir else output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    render_dir = Path(render_dir) if render_dir else output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    results: list[CleanResult] = []
    csv_rows: list[dict] = []

    for i, input_path in enumerate(input_paths):
        out_ply = output_dir / (input_path.stem + ".ply")
        report_json = report_dir / (input_path.stem + ".json")
        t0 = time.time()
        try:
            res = clean_mesh_pipeline(
                input_path=input_path,
                output_path=out_ply,
                cfg=cfg,
                render_dir=render_dir,
                verbose=verbose,
            )
            report_json.write_text(res.to_json(), encoding="utf-8")
            results.append(res)
            if verbose:
                print(f"[batch] [{i+1}/{len(input_paths)}] "
                      f"{input_path.name} {res.seconds_total:.1f}s "
                      f"accepted={res.accepted}")
            csv_rows.append({
                "input": str(input_path),
                "output": str(out_ply),
                "accepted": int(res.accepted),
                "reasons": "|".join(res.rejection_reasons),
                "total_s": round(res.seconds_total, 2),
                "s1_watertight": res.stage_reports.get(
                    "stage1_watertighten", {}).get("is_watertight"),
                "s1_chamfer": res.stage_reports.get(
                    "stage1_watertighten", {}).get("chamfer_to_input"),
                "s2_quality": res.stage_reports.get(
                    "stage2_vlm_filter", {}).get("aesthetic_quality"),
                "s2_class": res.stage_reports.get(
                    "stage2_vlm_filter", {}).get("object_class"),
                "s4_ray_agreement": (res.stage_reports.get(
                    "stage4_filter_geometry", {}).get("metrics", {})
                    .get("ray_sign_agreement")),
                "s4_vae_chamfer": (res.stage_reports.get(
                    "stage4_filter_geometry", {}).get("metrics", {})
                    .get("vae", {}) or {}).get("chamfer"),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            if verbose:
                print(f"[batch] ERROR on {input_path.name}: {e}")
            csv_rows.append({
                "input": str(input_path),
                "output": "",
                "accepted": 0,
                "reasons": f"exception:{type(e).__name__}:{str(e)[:200]}",
                "total_s": round(time.time() - t0, 2),
                "s1_watertight": None,
                "s1_chamfer": None,
                "s2_quality": None,
                "s2_class": None,
                "s4_ray_agreement": None,
                "s4_vae_chamfer": None,
            })

    if summary_csv:
        summary_csv = Path(summary_csv)
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        if csv_rows:
            keys = list(csv_rows[0].keys())
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(csv_rows)
            if verbose:
                print(f"[batch] wrote summary {summary_csv}")

    return results


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--render-dir", type=Path, default=None)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle", action="store_true",
                   help="Randomly pick --limit meshes instead of first N.")
    p.add_argument("--patterns", nargs="+",
                   default=["*.glb", "*.obj", "*.ply"])
    # Pipeline config
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--dense-resolution", type=int, default=None)
    p.add_argument("--canonicalize", default="geom",
                   choices=["geom", "hybrid", "vlm", "identity"])
    p.add_argument("--skip-vae", action="store_true")
    p.add_argument("--skip-vlm", action="store_true")
    p.add_argument("--vlm-cache-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    mesh_files = _collect_meshes(args.input_dir, args.patterns)
    if args.shuffle:
        random.Random(args.seed).shuffle(mesh_files)
    if args.limit:
        mesh_files = mesh_files[:args.limit]
    if not mesh_files:
        print(f"No meshes found under {args.input_dir}")
        return 1

    cfg = PipelineConfig(
        resolution=args.resolution,
        dense_resolution=args.dense_resolution or min(args.resolution, 1024),
        canonicalize_method=args.canonicalize,
        skip_vae=args.skip_vae,
        skip_vlm=args.skip_vlm,
        vlm_cache_dir=str(args.vlm_cache_dir) if args.vlm_cache_dir else None,
        device=args.device,
    )

    batch_clean(
        input_paths=mesh_files,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        render_dir=args.render_dir,
        summary_csv=args.summary_csv,
        cfg=cfg,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
