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

from . import _config
from .clean_mesh import CleanResult, PipelineConfig, clean_mesh_pipeline


class VLMDaemonReadyError(RuntimeError):
    """Raised when the Stage 2 sidecar fails to send a ``{"ready": true}``
    line on startup (e.g. torch/transformers ImportError in the sidecar
    env). Callers can fall back to the per-mesh subprocess path."""


class VLMDaemonClient:
    """Persistent subprocess that keeps Qwen3-VL loaded between meshes.

    Spawns ``python -m ultrashape_cleaning.vlm_filter serve`` and speaks
    the JSONL protocol defined in ``vlm_filter._cli_serve``. Reuse
    a single instance across a batch to amortise the ~60 s model load.

    Parameters
    ----------
    python_exe
        Path to the sidecar interpreter (env: ``VLM_PYTHON_EXE``).
    model_path
        Qwen3-VL model directory or HuggingFace id (env:
        ``QWEN3VL_MODEL_PATH``).
    device
        Passed through to the model.
    cwd
        Working directory for the subprocess. Must contain the
        ``ultrashape_cleaning`` package on ``sys.path`` (defaults to the
        repo root via ``VLM_SIDECAR_CWD``).
    cuda_visible_devices
        Propagated to the sidecar's environment if set.
    ready_timeout
        Seconds to wait for the ``{"ready": true}`` handshake before
        raising ``VLMDaemonReadyError``.
    """

    def __init__(self, python_exe: str, model_path: Optional[str] = None,
                 device: str = "cuda",
                 cwd: Optional[str] = None,
                 cuda_visible_devices: Optional[str] = None,
                 ready_timeout: float = 600.0):
        import subprocess, os, json as _json, threading
        self._json = _json
        model_path = model_path or _config.get_qwen3vl_model_path()
        cwd = cwd or _config.get_vlm_sidecar_cwd()
        env = os.environ.copy()
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.proc = subprocess.Popen(
            [python_exe, "-m", "ultrashape_cleaning.vlm_filter", "serve",
             "--model-path", model_path, "--device", device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, cwd=cwd, bufsize=1,
        )
        # Wait for the ready-JSON handshake.  ``vlm_filter._cli_serve`` prints
        # non-JSON progress lines (e.g. "[Qwen3VL] loading via ...") BEFORE the
        # ready line, so we must skip them.  Every readline is gated behind a
        # background thread so a crashed subprocess can't wedge us forever.
        import time as _time
        deadline = _time.time() + ready_timeout
        info = None
        last_line = ""
        while _time.time() < deadline:
            line_container: list = []

            def _read_line():
                try:
                    line_container.append(self.proc.stdout.readline())
                except Exception:
                    line_container.append("")

            t = threading.Thread(target=_read_line, daemon=True)
            t.start()
            remaining = max(deadline - _time.time(), 1.0)
            t.join(timeout=remaining)
            if t.is_alive():
                # Overall timeout reached mid-read.
                break
            if not line_container:
                continue
            line = (line_container[0] or "").rstrip("\r\n")
            if not line:
                # EOF -- subprocess exited.
                break
            last_line = line
            stripped = line.strip()
            if not (stripped.startswith("{") or stripped.startswith("[")):
                # Not JSON; this is a status / log line. Keep looping.
                continue
            try:
                info = self._json.loads(stripped)
                break
            except Exception:
                # Looks JSON-ish but isn't — treat as noise and continue.
                continue

        if info is None:
            try:
                stderr_tail = self.proc.stderr.read()[-500:]
            except Exception:
                stderr_tail = ""
            self._kill()
            raise VLMDaemonReadyError(
                "VLM sidecar did not report a valid ready JSON within "
                f"{ready_timeout}s (last stdout line: {last_line!r}). "
                f"stderr tail: {stderr_tail}"
            )
        self.ready = bool(info.get("ready", False))
        self.model_name = info.get("model")
        if not self.ready:
            self._kill()
            raise VLMDaemonReadyError(f"VLM sidecar ready=False: {info}")

    def infer(self, grid_png: str, mesh_id: str, out_json: Optional[str] = None,
              cache_dir: Optional[str] = None, prompt_lang: str = "en",
              timeout: float = 300.0) -> dict:
        import threading
        req = {"grid": grid_png, "mesh_id": mesh_id, "out_json": out_json,
               "cache_dir": cache_dir, "prompt_lang": prompt_lang}
        self.proc.stdin.write(self._json.dumps(req) + "\n")
        self.proc.stdin.flush()
        # Guard against a crashed daemon: reading without a timeout will
        # block forever on a dead pipe.
        line_container: list = []

        def _read_line():
            try:
                line_container.append(self.proc.stdout.readline())
            except Exception:
                line_container.append("")

        t = threading.Thread(target=_read_line, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive() or not line_container:
            raise TimeoutError(
                f"VLM sidecar did not respond within {timeout}s for "
                f"mesh_id={mesh_id}"
            )
        line = line_container[0]
        if not line:
            raise RuntimeError(
                f"VLM sidecar closed stdout while processing {mesh_id}"
            )
        return self._json.loads(line)

    def _kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass

    def close(self):
        try:
            self.proc.stdin.write(self._json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except Exception:
            self._kill()

    def __enter__(self) -> "VLMDaemonClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


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
    use_vlm_daemon: bool = True,
) -> list[CleanResult]:
    """Run :func:`clean_mesh_pipeline` on every mesh in ``input_paths``.

    When ``use_vlm_daemon`` is true and ``cfg.vlm_python_exe`` is set, a
    single :class:`VLMDaemonClient` subprocess is spawned for the entire
    batch, amortising the ~60 s Qwen3-VL model-load cost across every
    mesh. Falls back to per-mesh sidecar subprocesses if the daemon
    cannot be started.
    """
    cfg = cfg or PipelineConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path(report_dir) if report_dir else output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    render_dir = Path(render_dir) if render_dir else output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    # Spin up a persistent VLM daemon if one is available. When the env
    # var ``VLM_PYTHON_EXE`` (or ``cfg.vlm_python_exe``) is set we assume
    # the caller wants Stage 2 to run in a different interpreter; with
    # the daemon we do it once per batch instead of once per mesh.
    vlm_daemon: Optional[VLMDaemonClient] = None
    if (use_vlm_daemon and cfg.vlm_python_exe
            and not cfg.skip_vlm):
        try:
            if verbose:
                print(f"[batch] starting VLM daemon "
                      f"({cfg.vlm_python_exe})")
            vlm_daemon = VLMDaemonClient(
                python_exe=cfg.vlm_python_exe,
                model_path=cfg.vlm_model_path,
                device=cfg.device,
                cwd=cfg.vlm_sidecar_cwd,
                cuda_visible_devices=cfg.vlm_sidecar_visible_devices,
            )
            if verbose:
                print(f"[batch] VLM daemon ready ({vlm_daemon.model_name})")
        except VLMDaemonReadyError as e:
            if verbose:
                print(f"[batch] VLM daemon unavailable: {e}. "
                      f"Falling back to per-mesh subprocess.")
            vlm_daemon = None

    results: list[CleanResult] = []
    csv_rows: list[dict] = []

    try:
        for i, input_path in enumerate(input_paths):
            out_ply = output_dir / (input_path.stem + ".ply")
            report_json = report_dir / (input_path.stem + ".json")
            t0 = time.time()
            try:
                res = clean_mesh_pipeline(
                    input_path=input_path,
                    output_path=out_ply,
                    cfg=cfg,
                    vlm_daemon=vlm_daemon,
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
    finally:
        if vlm_daemon is not None:
            vlm_daemon.close()

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
