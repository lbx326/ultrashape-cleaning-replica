"""Run mesh_clean filter on 4 TRELLIS datasets in parallel across 8 GPUs.

For each dataset, reads metadata.csv to enumerate meshes, resolves the
absolute path to the GLB, then in 8 parallel workers (one per GPU):

  1. Render 4-view grid (Phong + 2x SSAA, ultrashape env via cubvh)
  2. Send grids to a per-worker Qwen3-VL daemon (buildingseg env) in
     batches of `--batch-size` for amortised inference
  3. Run Stage 4 geometry filter on the RAW input mesh (paper §2.1
     applies the filter to the unwatertightened input — its output
     decides whether to KEEP this mesh, not what to save next)
  4. Persist per-mesh: grid PNG, VLM JSON, report JSON

After all workers finish, an aggregator writes per-dataset
``mesh_clean_v1/summary.csv`` + ``retention.txt``.

Stage 1 (watertighten) and Stage 3 (canonicalize) are SKIPPED — they
don't influence the accept/reject decision and are the expensive ones.

Usage (remote):
    /moganshan/afs_a/lbx/env/ultrashape/bin/python \
        scripts/batch_filter_datasets.py [--num-gpus 8] [--batch-size 4]
        [--limit-per-dataset N] [--dataset hssd]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

REPO = Path("/moganshan/afs_a/lbx/utils/mesh_clean")
BASE = Path("/moganshan/afs_a/rsync/chuan/TRELLIS/datasets")
DATASETS = ["hssd", "ABO", "renders_ruiming", "renders_123"]
OUT_SUBDIR = "mesh_clean_v1"

VLM_PYTHON = "/moganshan/afs_a/lbx/env/buildingseg/bin/python"
VLM_MODEL = "/moganshan/afs_a/anmt/action/Qwen3-VL/Qwen3-VL-8B-Instruct/"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
def resolve_path(ds: str, sha: str, local_path: str | None) -> Path | None:
    """Try several path conventions per dataset structure."""
    candidates: list[Path] = []
    if local_path:
        candidates.append(BASE / ds / local_path)
    candidates.extend([
        BASE / ds / "raw" / sha,
        BASE / ds / "raw" / "3dmodels" / sha,
    ])
    if sha:
        candidates.append(BASE / ds / "raw" / "3dmodels" / "original" / sha[0] / sha)
    for p in candidates:
        if p.exists():
            return p
    return None


def collect_meshes(only: str | None = None,
                   limit_per_dataset: int = 0) -> list[dict]:
    out: list[dict] = []
    for ds in DATASETS:
        if only and ds != only:
            continue
        meta = BASE / ds / "metadata.csv"
        if not meta.exists():
            print(f"[skip] {ds}: no metadata.csv", flush=True)
            continue
        rows: list[dict] = []
        with open(meta, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                sha = r.get("sha256")
                if not sha:
                    continue
                lp = r.get("local_path")
                p = resolve_path(ds, sha, lp)
                if p is None:
                    continue
                rows.append({"dataset": ds, "sha": sha, "path": str(p)})
        if limit_per_dataset > 0:
            rows = rows[:limit_per_dataset]
        out.extend(rows)
        print(f"[meta] {ds}: {len(rows)} resolvable meshes", flush=True)
    return out


def out_paths(ds: str, sha: str) -> dict:
    base = BASE / ds / OUT_SUBDIR
    sha_stem = sha
    for ext in (".glb", ".obj", ".ply", ".gltf", ".fbx"):
        if sha_stem.lower().endswith(ext):
            sha_stem = sha_stem[: -len(ext)]
            break
    return {
        "base": base,
        "sha_stem": sha_stem,
        "renders": base / "_renders",
        "reports": base / "_reports",
        "vlm_cache": base / ".vlm_cache",
        "summary_csv": base / "summary.csv",
        "retention": base / "retention.txt",
        "grid_png": base / "_renders" / f"{sha_stem}.grid.png",
        "vlm_json": base / "_renders" / f"{sha_stem}.vlm.json",
        "report_json": base / "_reports" / f"{sha_stem}.json",
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def worker_main(gpu_id: int, work_chunk: list[dict],
                batch_size: int, do_vae: bool,
                prompt_lang: str = "en",
                wt_resolution: int = 0) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.path.insert(0, str(REPO))

    import trimesh
    from PIL import Image
    from ultrashape_cleaning.batch_clean import VLMDaemonClient
    from ultrashape_cleaning.renderer import (
        make_2x2_grid, render_four_views,
    )
    from ultrashape_cleaning.filter_geometry import (
        filter_geometry, FilterConfig, UltraShapeVAE,
    )

    print(f"[w{gpu_id}] starting on {len(work_chunk)} meshes (batch={batch_size}, vae={do_vae})",
          flush=True)

    # Pre-create per-dataset output dirs
    used_datasets = sorted({it["dataset"] for it in work_chunk})
    for ds in used_datasets:
        for sub in ("_renders", "_reports", ".vlm_cache"):
            (BASE / ds / OUT_SUBDIR / sub).mkdir(parents=True, exist_ok=True)

    # VLM daemon (sidecar in buildingseg env)
    t = time.time()
    print(f"[w{gpu_id}] spawning VLM daemon...", flush=True)
    daemon = VLMDaemonClient(
        python_exe=VLM_PYTHON,
        model_path=VLM_MODEL,
        device="cuda",
        cwd=str(REPO),
        cuda_visible_devices=str(gpu_id),
        ready_timeout=600.0,
    )
    print(f"[w{gpu_id}] VLM daemon ready in {time.time() - t:.1f}s", flush=True)

    # Lazily-loaded UltraShapeVAE (Stage 4 chamfer signal)
    cfg = FilterConfig(device="cuda", skip_vae=not do_vae)
    vae = None
    if do_vae:
        try:
            vae = UltraShapeVAE.load(
                config_path=cfg.vae_config_path,
                ckpt_path=cfg.vae_ckpt_path,
                device=cfg.device,
            )
            print(f"[w{gpu_id}] VAE loaded", flush=True)
        except Exception as e:
            print(f"[w{gpu_id}] VAE load failed ({e}); falling back to skip_vae",
                  flush=True)
            cfg.skip_vae = True

    # Pending batch buffer
    pending: list[dict] = []

    def flush_batch(buf: list[dict]) -> None:
        if not buf:
            return
        items = []
        for x in buf:
            items.append({
                "grid": str(x["op"]["grid_png"]),
                "mesh_id": x["op"]["sha_stem"],
                "out_json": str(x["op"]["vlm_json"]),
                "cache_dir": str(x["op"]["vlm_cache"]),
            })
        try:
            vlm_results = daemon.infer_batch(items, prompt_lang=prompt_lang)
        except Exception as e:
            for x in buf:
                _emit_error(x, "vlm_batch_fail", str(e))
            return
        # Per-item Stage 4 filter on raw mesh
        for x, vr in zip(buf, vlm_results):
            try:
                report = filter_geometry(x["mesh"], cfg=cfg, vae=vae,
                                         verbose=False)
                merged = {
                    "sha256": x["item"]["sha"],
                    "dataset": x["item"]["dataset"],
                    "input_path": x["item"]["path"],
                    "stage2_vlm": vr,
                    "stage4_filter": {
                        "is_valid": report.is_valid,
                        "reasons": report.reasons,
                        "metrics": report.metrics,
                        "seconds": report.seconds,
                    },
                    "accepted": (
                        bool(vr.get("accepted")) and bool(report.is_valid)
                    ),
                    "rejection_reasons": (
                        [f"stage2:{r}" for r in (vr.get("rejection_reasons") or [])]
                        + [f"stage4:{r}" for r in report.reasons]
                    ),
                    "seconds_total": vr.get("seconds", 0.0) + report.seconds,
                }
                x["op"]["report_json"].write_text(
                    json.dumps(merged, ensure_ascii=False, indent=2,
                               default=str),
                    encoding="utf-8",
                )
            except Exception as e:
                _emit_error(x, "stage4_fail", str(e))

    def _emit_error(x: dict, where: str, msg: str) -> None:
        merged = {
            "sha256": x["item"]["sha"],
            "dataset": x["item"]["dataset"],
            "input_path": x["item"]["path"],
            "accepted": False,
            "rejection_reasons": [f"error:{where}:{msg[:200]}"],
            "error": where,
            "error_msg": msg,
        }
        x["op"]["report_json"].write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    n_done = 0
    n_skipped = 0
    n_errors = 0
    t_start = time.time()

    for item in work_chunk:
        op = out_paths(item["dataset"], item["sha"])
        # Skip if already done (idempotent)
        if op["report_json"].exists():
            n_skipped += 1
            continue

        # Load mesh
        try:
            m = trimesh.load(item["path"], force="mesh", process=False)
            if isinstance(m, trimesh.Scene):
                geos = [g for g in m.geometry.values()
                        if isinstance(g, trimesh.Trimesh)]
                if not geos:
                    raise RuntimeError("scene has no Trimesh geometry")
                m = trimesh.util.concatenate(geos)
            if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
                raise RuntimeError("invalid mesh after load")
        except Exception as e:
            _emit_error({"item": item, "op": op}, "load", str(e))
            n_errors += 1
            continue

        # Render
        try:
            views = render_four_views(m, resolution=512, supersample=2)
            grid = make_2x2_grid(views)
            Image.fromarray(grid).save(str(op["grid_png"]))
        except Exception as e:
            _emit_error({"item": item, "op": op}, "render", str(e))
            n_errors += 1
            continue

        pending.append({"item": item, "mesh": m, "op": op})
        if len(pending) >= batch_size:
            flush_batch(pending)
            n_done += len(pending)
            pending = []
            elapsed = time.time() - t_start
            rate = (n_done / elapsed) if elapsed > 0 else 0
            print(f"[w{gpu_id}] {n_done} done | {n_skipped} skip | "
                  f"{n_errors} err | rate={rate:.2f} mesh/s | "
                  f"elapsed={elapsed/60:.1f}min", flush=True)

    if pending:
        flush_batch(pending)
        n_done += len(pending)

    daemon.close()
    print(f"[w{gpu_id}] DONE: {n_done} done | {n_skipped} skipped | "
          f"{n_errors} errors | total={(time.time()-t_start)/60:.1f}min",
          flush=True)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def aggregate(only: str | None = None) -> None:
    for ds in DATASETS:
        if only and ds != only:
            continue
        base = BASE / ds / OUT_SUBDIR
        rep_dir = base / "_reports"
        if not rep_dir.exists():
            continue
        rows = []
        for rp in sorted(rep_dir.glob("*.json")):
            try:
                rows.append(json.loads(rp.read_text(encoding="utf-8")))
            except Exception:
                pass
        if not rows:
            continue
        # Write summary.csv
        cols = [
            "sha256", "accepted", "rejection_reasons",
            "seconds_total",
            "vlm_quality", "vlm_class", "vlm_primitive", "vlm_ground",
            "vlm_noisy", "vlm_frag",
            "interior_ratio", "vae_chamfer", "num_components", "is_watertight",
            "input_path",
        ]
        out_rows = []
        n_accept = 0
        for r in rows:
            v = r.get("stage2_vlm") or {}
            f4 = (r.get("stage4_filter") or {}).get("metrics") or {}
            out_rows.append({
                "sha256": r.get("sha256"),
                "accepted": r.get("accepted"),
                "rejection_reasons": "|".join(r.get("rejection_reasons") or []),
                "seconds_total": r.get("seconds_total"),
                "vlm_quality": v.get("aesthetic_quality"),
                "vlm_class": v.get("object_class"),
                "vlm_primitive": v.get("is_primitive"),
                "vlm_ground": v.get("is_ground_plane"),
                "vlm_noisy": v.get("is_noisy_scan"),
                "vlm_frag": v.get("is_fragmented"),
                "interior_ratio": f4.get("fraction_inside_gt"),
                "vae_chamfer": (f4.get("vae") or {}).get("chamfer"),
                "num_components": f4.get("num_components"),
                "is_watertight": f4.get("is_watertight"),
                "input_path": r.get("input_path"),
            })
            if r.get("accepted"):
                n_accept += 1
        with open(base / "summary.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for row in out_rows:
                w.writerow(row)
        n = len(out_rows)
        retention = (n_accept / n) if n else 0.0
        (base / "retention.txt").write_text(
            f"dataset: {ds}\n"
            f"total: {n}\n"
            f"accepted: {n_accept}\n"
            f"retention_rate: {retention:.4f} ({retention*100:.2f}%)\n",
            encoding="utf-8",
        )
        print(f"[agg] {ds}: {n_accept}/{n} = {retention*100:.2f}%", flush=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit-per-dataset", type=int, default=0,
                    help="Process only first N meshes per dataset (0=all)")
    ap.add_argument("--dataset", default=None,
                    help="Restrict to one dataset (hssd / ABO / renders_ruiming / renders_123)")
    ap.add_argument("--no-vae", action="store_true",
                    help="Skip Stage 4 VAE chamfer (faster, less faithful)")
    ap.add_argument("--prompt-lang", default="en")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip processing; just rebuild summary.csv from existing reports")
    args = ap.parse_args()

    if args.aggregate_only:
        aggregate(only=args.dataset)
        return 0

    meshes = collect_meshes(only=args.dataset,
                            limit_per_dataset=args.limit_per_dataset)
    print(f"total meshes to process: {len(meshes)}", flush=True)
    if not meshes:
        return 0

    # Even split across GPUs
    chunks: list[list[dict]] = [[] for _ in range(args.num_gpus)]
    for i, m in enumerate(meshes):
        chunks[i % args.num_gpus].append(m)

    print(f"chunks: " + ", ".join(str(len(c)) for c in chunks), flush=True)

    procs: list[mp.Process] = []
    ctx = mp.get_context("spawn")
    for gid in range(args.num_gpus):
        p = ctx.Process(
            target=worker_main,
            args=(gid, chunks[gid], args.batch_size, not args.no_vae,
                  args.prompt_lang),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("[main] all workers done; aggregating...", flush=True)
    aggregate(only=args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
