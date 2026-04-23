"""clean_mesh.py -- End-to-end orchestrator for the 4-stage data cleaning.

Takes one mesh path, runs all 4 stages in order, and emits a cleaned mesh
plus a JSON report.

Pipeline
--------
    Input mesh.glb
        |
        v
    [Stage 1] watertighten   -> clean.ply (watertight manifold)
        |
        v
    [Stage 3] canonicalize   -> canonical.ply (+Y up, +Z front)
        |
        v
    [Stage 4] filter_geometry-> report {valid, reasons, metrics}
        |
        v
    [Stage 2] vlm_filter     -> report {quality, class, reasons}
        |
        v
    Final cleaned mesh + JSON report

Stage 2 is executed LAST on the canonicalized mesh because that gives the
VLM the clearest view of the object. Stage 4 runs before Stage 2 so the
pipeline can short-circuit on geometry failures before booting the VLM.

CLI
---
    python -m ultrashape_cleaning.clean_mesh \\
        --input messy.glb --output clean.ply --full

Flags:
    --resolution 1024          (watertighten resolution)
    --canonicalize geom|hybrid|vlm|identity
    --skip-vae                 (skip Stage 4's VAE fragmentation check)
    --skip-vlm                 (skip Stage 2)
    --strict                   (exit nonzero if the mesh is rejected)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Optional

from ._meshio import PathLike, load_mesh, save_mesh, sha256_file, summarize


# ---------------------------------------------------------------------------
# Config / Result containers
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class PipelineConfig:
    # Stage 1
    resolution: int = 1024
    dense_resolution: int = 1024
    close_iters: int = 2
    thicken_voxels: int = 0
    auto_thicken: bool = True
    # Stage 3
    canonicalize_method: str = "geom"  # geom, hybrid, vlm, identity
    # Stage 2
    vlm_prompt_lang: str = "en"
    vlm_resolution: int = 512
    vlm_min_quality: int = 2
    # Stage 4
    skip_vae: bool = False
    frag_chamfer_threshold: float = 0.15
    # Misc
    skip_vlm: bool = False
    skip_canonicalize: bool = False
    skip_stage4: bool = False
    device: str = "cuda"
    vlm_model_path: str = "/moganshan/afs_a/anmt/action/Qwen3-VL/Qwen3-VL-8B-Instruct/"
    vae_ckpt_path: str = ("/moganshan/afs_a/lbx/hf/hub/models--infinith--UltraShape/"
                          "snapshots/5aeb21a7185d39f042d02b2695802f125a6f5159/"
                          "ultrashape_v1.pt")
    vae_config_path: str = "/moganshan/afs_a/lbx/workspace/UltraShape-1.0/configs/infer_dit_refine.yaml"
    vlm_cache_dir: Optional[str] = None
    # When the VLM needs a different env (e.g. transformers 4.57 for
    # Qwen3-VL) than the rest of the pipeline, we shell out to a sidecar
    # python interpreter for Stage 2.
    vlm_python_exe: Optional[str] = "/moganshan/afs_a/lbx/env/buildingseg/bin/python"
    vlm_sidecar_visible_devices: Optional[str] = None  # "7" etc


@dataclasses.dataclass
class CleanResult:
    sha256: str
    input_path: str
    output_path: Optional[str]
    accepted: bool
    rejection_reasons: list
    stage_timings: dict
    stage_reports: dict
    input_summary: dict
    output_summary: Optional[dict]
    seconds_total: float

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True,
                          ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Shared clients cache (for batch_clean reuse)
# ---------------------------------------------------------------------------
class _ClientCache:
    def __init__(self):
        self.vae = None
        self.vlm = None


_CACHE = _ClientCache()


# ---------------------------------------------------------------------------
# Pipeline function
# ---------------------------------------------------------------------------
def clean_mesh_pipeline(
    input_path: PathLike,
    output_path: Optional[PathLike] = None,
    cfg: Optional[PipelineConfig] = None,
    vae=None,
    vlm_client=None,
    render_dir: Optional[PathLike] = None,
    verbose: bool = True,
) -> CleanResult:
    """Run the full 4-stage pipeline on one mesh."""
    from .watertighten import WatertightenConfig, watertighten_mesh
    from .canonicalize import canonicalize_mesh
    from .filter_geometry import FilterConfig, UltraShapeVAE, filter_geometry
    from .vlm_filter import (DEFAULT_PROMPT_EN, DEFAULT_PROMPT_ZH,
                             Qwen3VLClient, render_grid_png, run_vlm_filter)

    cfg = cfg or PipelineConfig()
    t0 = time.time()
    timings: dict = {}
    reports: dict = {}
    reasons: list[str] = []

    input_path = Path(input_path)
    sha = sha256_file(input_path)
    if verbose:
        print(f"[pipeline] {input_path.name} sha={sha[:10]}")

    mesh_in = load_mesh(input_path)
    input_summary = summarize(mesh_in)

    # ==== Stage 1: Watertighten ====
    t = time.time()
    stage1_cfg = WatertightenConfig(
        resolution=cfg.resolution,
        dense_resolution=cfg.dense_resolution,
        close_iters=cfg.close_iters,
        thicken_voxels=cfg.thicken_voxels,
        auto_thicken=cfg.auto_thicken,
        device=cfg.device,
    )
    mesh_wt, sdf, rep1 = watertighten_mesh(mesh_in, stage1_cfg, verbose=verbose)
    timings["stage1_watertighten"] = time.time() - t
    reports["stage1_watertighten"] = dataclasses.asdict(rep1)
    if not rep1.is_watertight:
        reasons.append("stage1_not_watertight")

    # ==== Stage 3: Canonicalize ====
    if cfg.skip_canonicalize:
        mesh_canon = mesh_wt
        reports["stage3_canonicalize"] = {"method": "skipped"}
    else:
        t = time.time()
        # If method wants VLM, make sure we have a client.
        method = cfg.canonicalize_method
        need_vlm_client = method in {"vlm", "hybrid"} and not cfg.skip_vlm
        if need_vlm_client and vlm_client is None:
            if _CACHE.vlm is None:
                _CACHE.vlm = Qwen3VLClient.from_local(
                    path=cfg.vlm_model_path, device=cfg.device,
                )
            vlm_client = _CACHE.vlm
        temp_png = None
        if need_vlm_client:
            render_dir = Path(render_dir) if render_dir else Path("/tmp/mesh_clean")
            render_dir.mkdir(parents=True, exist_ok=True)
            temp_png = render_dir / f"{sha}.6view.png"
        mesh_canon, R, rep3 = canonicalize_mesh(
            mesh_wt,
            method=(method if need_vlm_client or method in {"geom", "identity"}
                    else "geom"),
            vlm_client=vlm_client,
            temp_png=temp_png,
            resolution=384,
            device=cfg.device,
        )
        timings["stage3_canonicalize"] = time.time() - t
        reports["stage3_canonicalize"] = dataclasses.asdict(rep3)
        if rep3.needs_manual_review:
            reasons.append("stage3_canonicalize_uncertain")

    # ==== Stage 4: Geometry filter ====
    if cfg.skip_stage4:
        reports["stage4_filter_geometry"] = {"skipped": True}
    else:
        t = time.time()
        if vae is None and not cfg.skip_vae:
            if _CACHE.vae is None:
                if verbose:
                    print("[pipeline] loading UltraShape VAE (first call) ...")
                _CACHE.vae = UltraShapeVAE.load(
                    config_path=cfg.vae_config_path,
                    ckpt_path=cfg.vae_ckpt_path,
                    device=cfg.device,
                )
            vae = _CACHE.vae
        fcfg = FilterConfig(
            skip_vae=cfg.skip_vae,
            frag_vae_chamfer_threshold=cfg.frag_chamfer_threshold,
            device=cfg.device,
            vae_config_path=cfg.vae_config_path,
            vae_ckpt_path=cfg.vae_ckpt_path,
        )
        # Stage 1 already gave us an SDF and a known flood-fill sign. Use it
        # as the ground-truth-inside function so Stage 4 doesn't redo the
        # multi-bounce ray cast.
        def _gt_from_sdf(pts_np):
            import numpy as _np
            # Canonicalization rotates verts; also re-fit. The sdf is in the
            # watertighten's [0,1]^3 unit cube for the ORIGINAL mesh. We
            # need to transform test points back. Since canonicalization
            # operates AFTER watertighten, and Stage 4 runs on the
            # canonical mesh, sdf is not directly reusable. Skip the
            # optimization and fall back to ray-cast GT.
            return None
        # Use a cheaper ray-cast: single ray per point (parity trick works
        # on watertight mesh).
        rep4 = filter_geometry(mesh_canon, fcfg, vae=vae, verbose=verbose)
        timings["stage4_filter_geometry"] = time.time() - t
        reports["stage4_filter_geometry"] = dataclasses.asdict(rep4)
        if not rep4.is_valid:
            reasons.extend(f"stage4:{r}" for r in rep4.reasons)

    # ==== Stage 2: VLM filter ====
    if cfg.skip_vlm:
        reports["stage2_vlm_filter"] = {"skipped": True}
    else:
        t = time.time()
        # We always need to render the grid in-process (we have cubvh here).
        render_dir = Path(render_dir) if render_dir else Path("/tmp/mesh_clean")
        render_dir.mkdir(parents=True, exist_ok=True)
        grid_png = render_dir / f"{sha}.grid.png"
        from .renderer import render_four_views, make_2x2_grid
        from PIL import Image
        views = render_four_views(
            mesh_canon, resolution=cfg.vlm_resolution, device=cfg.device,
        )
        grid = make_2x2_grid(views)
        Image.fromarray(grid).save(str(grid_png))

        # Run inference:
        if cfg.vlm_python_exe:
            # Sidecar subprocess in the other env.
            import subprocess, os
            vlm_json = render_dir / f"{sha}.vlm.json"
            env = os.environ.copy()
            if cfg.vlm_sidecar_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = cfg.vlm_sidecar_visible_devices
            cache_args = []
            if cfg.vlm_cache_dir:
                cache_args = ["--cache-dir", cfg.vlm_cache_dir]
            cmd = [
                cfg.vlm_python_exe, "-m", "ultrashape_cleaning.vlm_filter",
                "infer", "--grid", str(grid_png),
                "--mesh-id", sha, "--out-json", str(vlm_json),
                "--model-path", cfg.vlm_model_path,
                "--prompt-lang", cfg.vlm_prompt_lang,
            ] + cache_args
            if verbose:
                print(f"[pipeline] sidecar VLM: {' '.join(cmd[:3])}...")
            proc = subprocess.run(
                cmd, capture_output=True, text=True, env=env,
                cwd="/moganshan/afs_a/lbx/utils/mesh_clean",
            )
            if proc.returncode != 0:
                reports["stage2_vlm_filter"] = {
                    "error": "sidecar_failed",
                    "stderr": proc.stderr[-500:],
                    "stdout": proc.stdout[-500:],
                }
            else:
                try:
                    vlm_data = json.loads(vlm_json.read_text(encoding="utf-8"))
                    reports["stage2_vlm_filter"] = vlm_data
                    if not vlm_data.get("accepted", True):
                        reasons.extend(
                            f"stage2:{r}" for r in
                            vlm_data.get("rejection_reasons", [])
                        )
                except Exception as e:
                    reports["stage2_vlm_filter"] = {
                        "error": f"parse_failed:{e}",
                        "stdout": proc.stdout[-500:],
                    }
        else:
            if vlm_client is None:
                if _CACHE.vlm is None:
                    if verbose:
                        print("[pipeline] loading Qwen3-VL (first call) ...")
                    _CACHE.vlm = Qwen3VLClient.from_local(
                        path=cfg.vlm_model_path, device=cfg.device,
                    )
                vlm_client = _CACHE.vlm
            prompt = (DEFAULT_PROMPT_ZH if cfg.vlm_prompt_lang == "zh"
                      else DEFAULT_PROMPT_EN)
            accept_cfg = {"min_quality": cfg.vlm_min_quality}
            rep2 = run_vlm_filter(
                grid_png=grid_png, mesh_sha256=sha, client=vlm_client,
                prompt=prompt, prompt_lang=cfg.vlm_prompt_lang,
                cache_dir=cfg.vlm_cache_dir, accept_cfg=accept_cfg,
            )
            reports["stage2_vlm_filter"] = dataclasses.asdict(rep2)
            if not rep2.accepted:
                reasons.extend(f"stage2:{r}" for r in rep2.rejection_reasons)
        timings["stage2_vlm_filter"] = time.time() - t

    # ==== Write output ====
    output_summary = None
    out_path_str: Optional[str] = None
    accepted = (len(reasons) == 0)
    if output_path is not None and (accepted or True):
        # We ALWAYS write the cleaned mesh, even if filters rejected the
        # input — users can inspect why it was flagged. Accepted is just
        # a bit in the report.
        output_path = Path(output_path)
        save_mesh(mesh_canon, output_path)
        out_path_str = str(output_path)
        output_summary = summarize(mesh_canon)

    seconds_total = time.time() - t0
    result = CleanResult(
        sha256=sha,
        input_path=str(input_path),
        output_path=out_path_str,
        accepted=accepted,
        rejection_reasons=reasons,
        stage_timings=timings,
        stage_reports=reports,
        input_summary=input_summary,
        output_summary=output_summary,
        seconds_total=seconds_total,
    )
    if verbose:
        print(f"[pipeline] DONE {input_path.name} accepted={accepted} "
              f"reasons={reasons} total={seconds_total:.1f}s")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--save-report", type=Path, default=None)
    p.add_argument("--render-dir", type=Path, default=None)
    p.add_argument("--full", action="store_true",
                   help="Run all 4 stages with defaults.")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--dense-resolution", type=int, default=None)
    p.add_argument("--canonicalize", default="geom",
                   choices=["geom", "hybrid", "vlm", "identity"])
    p.add_argument("--vlm-prompt-lang", choices=["en", "zh"], default="en")
    p.add_argument("--vlm-cache-dir", type=Path, default=None)
    p.add_argument("--vlm-min-quality", type=int, default=2)
    p.add_argument("--skip-vae", action="store_true")
    p.add_argument("--skip-vlm", action="store_true")
    p.add_argument("--skip-canonicalize", action="store_true")
    p.add_argument("--skip-stage4", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="Exit nonzero if the mesh is rejected.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = PipelineConfig(
        resolution=args.resolution,
        dense_resolution=args.dense_resolution or min(args.resolution, 1024),
        canonicalize_method=args.canonicalize,
        vlm_prompt_lang=args.vlm_prompt_lang,
        vlm_min_quality=args.vlm_min_quality,
        vlm_cache_dir=str(args.vlm_cache_dir) if args.vlm_cache_dir else None,
        skip_vae=args.skip_vae,
        skip_vlm=args.skip_vlm,
        skip_canonicalize=args.skip_canonicalize,
        skip_stage4=args.skip_stage4,
        device=args.device,
    )

    result = clean_mesh_pipeline(
        input_path=args.input,
        output_path=args.output,
        cfg=cfg,
        render_dir=args.render_dir,
        verbose=not args.quiet,
    )
    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(result.to_json(), encoding="utf-8")
    else:
        if args.quiet:
            print(result.to_json())

    if args.strict and not result.accepted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
