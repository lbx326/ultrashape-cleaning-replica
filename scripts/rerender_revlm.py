"""Re-render all stress_v3 meshes with the new Phong+SSAA renderer and
re-run Qwen3-VL on the new grids. Keeps stage 1/3/4 results from
stress_v3 untouched; produces a new summary.csv under ``outputs/stress_v3b/``
with old vs. new VLM verdicts side by side.

Intended for spot-checking whether the 46% ``noisy_scan`` rejection rate
was caused by aliased flat-shaded renders, not the meshes themselves.

Runs under ``ultrashape`` env (needs trimesh + cubvh + renderer).
Spawns a ``buildingseg``-env VLM daemon internally via VLMDaemonClient.

Usage (remote):
    /moganshan/afs_a/lbx/env/ultrashape/bin/python scripts/rerender_revlm.py \\
        [--limit 50] [--resolution 512] [--supersample 2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path("/moganshan/afs_a/lbx/utils/mesh_clean")
SRC_ROOT = REPO / "outputs" / "stress_v3"
DST_ROOT = REPO / "outputs" / "stress_v3b"

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N meshes (0=all)")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--cuda-devices", default="7")
    ap.add_argument("--python-exe", default="/moganshan/afs_a/lbx/env/buildingseg/bin/python")
    ap.add_argument("--model-path",
                    default="/moganshan/afs_a/anmt/action/Qwen3-VL/Qwen3-VL-8B-Instruct/")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    import pandas as pd
    import trimesh
    from PIL import Image

    from ultrashape_cleaning.batch_clean import VLMDaemonClient
    from ultrashape_cleaning.renderer import make_2x2_grid, render_four_views

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    (DST_ROOT / "_renders").mkdir(exist_ok=True)
    cache_dir = DST_ROOT / ".vlm_cache"
    cache_dir.mkdir(exist_ok=True)

    df = pd.read_csv(SRC_ROOT / "summary.csv")
    if args.limit > 0:
        df = df.head(args.limit)
    print(f"[rerender] {len(df)} meshes to process at res={args.resolution} "
          f"ss={args.supersample}", flush=True)

    t_daemon = time.time()
    print("[rerender] spawning VLM daemon ...", flush=True)
    daemon = VLMDaemonClient(
        python_exe=args.python_exe,
        model_path=args.model_path,
        device="cuda",
        cwd=str(REPO),
        cuda_visible_devices=args.cuda_devices,
        ready_timeout=600,
    )
    print(f"[rerender] daemon ready ({time.time() - t_daemon:.1f}s): "
          f"model={daemon.model_name}", flush=True)

    rows = []
    summary_path = DST_ROOT / "summary.csv"
    try:
        for idx, row in df.iterrows():
            sha = row["sha256"]
            mesh_path = Path(row["input"])

            t0 = time.time()
            m = trimesh.load(str(mesh_path), force="mesh", process=False)
            if isinstance(m, trimesh.Scene):
                m = trimesh.util.concatenate(
                    [g for g in m.geometry.values()
                     if isinstance(g, trimesh.Trimesh)]
                )
            t_load = time.time() - t0

            t0 = time.time()
            try:
                views = render_four_views(m, resolution=args.resolution,
                                          supersample=args.supersample)
                grid = make_2x2_grid(views)
                grid_path = DST_ROOT / "_renders" / f"{sha}.grid.png"
                Image.fromarray(grid).save(str(grid_path))
            except Exception as e:
                print(f"[{idx + 1}] render FAILED sha={sha[:10]}: {e}", flush=True)
                rows.append({"sha256": sha, "error": f"render:{e}"})
                continue
            t_render = time.time() - t0

            t0 = time.time()
            try:
                resp = daemon.infer(
                    grid_png=str(grid_path),
                    mesh_id=sha,
                    out_json=str(DST_ROOT / "_renders" / f"{sha}.vlm.json"),
                    cache_dir=str(cache_dir),
                    prompt_lang="en",
                )
            except Exception as e:
                print(f"[{idx + 1}] VLM FAILED sha={sha[:10]}: {e}", flush=True)
                rows.append({"sha256": sha, "error": f"vlm:{e}"})
                continue
            t_vlm = time.time() - t0

            # Daemon stdout protocol wraps the VLMResult under "result":
            #     {"ok": true, "result": {...}}  (see vlm_filter._cli_serve)
            # Unwrap first; bail if the server signalled ok=False.
            if not resp.get("ok", False):
                err = resp.get("error", "unknown")
                print(f"[{idx + 1}] VLM reported not-ok sha={sha[:10]}: {err}",
                      flush=True)
                rows.append({"sha256": sha, "error": f"vlm_not_ok:{err}"})
                continue
            result = resp.get("result") or {}

            reasons_new = "|".join(result.get("rejection_reasons") or [])
            accepted_new = bool(result.get("accepted"))
            old_noisy = bool(row.get("s2_noisy"))
            new_noisy = bool(result.get("is_noisy_scan"))
            flip_sign = ""
            if old_noisy and not new_noisy:
                flip_sign = "  FLIP: noisy->clean"
            elif not old_noisy and new_noisy:
                flip_sign = "  FLIP: clean->noisy"

            print(f"[{idx + 1}/{len(df)}] sha={sha[:10]} "
                  f"load={t_load:.1f}s ren={t_render:.1f}s vlm={t_vlm:.1f}s "
                  f"old_noisy={old_noisy} new_noisy={new_noisy}{flip_sign}",
                  flush=True)

            rows.append({
                "sha256": sha,
                "old_reasons": row.get("reasons", ""),
                "new_reasons": reasons_new,
                "old_accepted_full": bool(row.get("accepted", False)),
                "new_s2_accepted": accepted_new,
                "old_s2_quality": row.get("s2_quality"),
                "new_s2_quality": result.get("aesthetic_quality"),
                "old_s2_class": row.get("s2_class"),
                "new_s2_class": result.get("object_class"),
                "old_s2_primitive": bool(row.get("s2_primitive", False)),
                "new_s2_primitive": bool(result.get("is_primitive", False)),
                "old_s2_ground": bool(row.get("s2_ground", False)),
                "new_s2_ground": bool(result.get("is_ground_plane", False)),
                "old_s2_noisy": old_noisy,
                "new_s2_noisy": new_noisy,
                "old_s2_frag": bool(row.get("s2_frag", False)),
                "new_s2_frag": bool(result.get("is_fragmented", False)),
                "t_load": t_load,
                "t_render": t_render,
                "t_vlm": t_vlm,
            })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(df):
                pd.DataFrame(rows).to_csv(summary_path, index=False)
    finally:
        daemon.close()
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"[rerender] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
