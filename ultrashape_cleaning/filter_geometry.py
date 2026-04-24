"""filter_geometry.py -- Stage 4 of the UltraShape cleaning pipeline.

Performs three independent quality checks on a (watertight, canonicalized)
mesh:

    A. Interior/exterior ratio via GPU ray casting. The paper's test is:
       cast N rays from a random external point INTO the mesh; the count
       of triangles hit should be even for a sound watertight manifold
       (ray enters + exits). We generalize: for random interior query
       points, the sign of cubvh.signed_distance should be negative; for
       random exterior query points, positive. Disagreement with the
       flood-fill-derived ground truth indicates wrong winding or holes.

    B. Fragmentation: connected component count + UltraShape VAE encode→
       decode→MC round-trip chamfer. A mesh far from the VAE's learned
       manifold (high chamfer) is considered out-of-distribution; we
       threshold at the 95th percentile measured on a clean calibration set.

    C. Primitiveness: vertex count + face-normal axis-alignment fraction.
       Unit cubes / spheres produced by poorly-prompted generation should
       be rejected.

Outputs a FilterReport with:
    is_valid: bool
    reasons: list[str]
    metrics: dict (all intermediate numbers for calibration)
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh

from . import _config
from ._meshio import PathLike, load_mesh


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class FilterConfig:
    ray_samples: int = 20_000
    # Paper §2.1 rejects meshes whose "interior-to-exterior point ratio" is
    # too low — i.e. thin shells where random points in the bbox almost
    # never land inside. Threshold at 0.03 = 3% of bbox samples inside.
    thin_shell_interior_ratio_min: float = 0.03
    # Diagnostic only; no rejection. Low agreement suggests non-manifold /
    # inverted winding but is not in the paper as a filter step.
    ray_sign_agreement_threshold: float = 0.0  # 0 = disabled (diagnostic only)
    max_components: int = 8
    frag_vae_chamfer_threshold: Optional[float] = 0.15  # empirical on HSSD;
    # see docs/stage4_filter.md for calibration details. Clean sofa ≈ 0.064,
    # fragmented ≈ 0.40, primitive ≈ 0.40. Above 0.15 is OOD.
    primitive_vertex_threshold: int = 500
    primitive_axis_alignment_threshold: float = 0.85
    # Kept for backwards compatibility with older configs; no longer used
    # as a rejection criterion. The paper uses interior-to-exterior point
    # ratios (see ``thin_shell_interior_ratio_min`` above) which is
    # topology-aware, not just a geometric vol/area.
    volume_area_ratio_min: Optional[float] = None
    device: str = "cuda"
    # VAE config (lazy-loaded). Resolved from env vars
    # ULTRASHAPE_VAE_CONFIG / ULTRASHAPE_VAE_CKPT — see ``_config.py``.
    vae_config_path: Optional[str] = dataclasses.field(
        default_factory=_config.get_vae_config_path,
    )
    vae_ckpt_path: Optional[str] = dataclasses.field(
        default_factory=_config.get_vae_ckpt_path,
    )
    # Optional: path to the UltraShape repo root for sys.path injection.
    # Only needed when the ultrashape package is not already importable.
    ultrashape_repo_root: Optional[str] = dataclasses.field(
        default_factory=_config.get_ultrashape_repo_root,
    )
    # Skip flags for batch throughput.
    skip_vae: bool = False


@dataclasses.dataclass
class FilterReport:
    is_valid: bool
    reasons: list
    metrics: dict
    seconds: float

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Check A: ray-cast interior/exterior ratio
# ---------------------------------------------------------------------------
def ray_sign_agreement(
    mesh: trimesh.Trimesh,
    ground_truth_inside,  # callable: (pts_np) -> bool array
    n_samples: int = 20_000,
    device: str = "cuda",
) -> dict:
    """Count fraction of agreement between cubvh.signed_distance sign and GT.

    Ground truth comes from the watertighten stage's flood fill (or, if
    only the mesh is available, from a ray-casting "even-crossings" test).
    """
    import cubvh
    verts = torch.as_tensor(np.asarray(mesh.vertices), device=device,
                            dtype=torch.float32)
    faces = torch.as_tensor(np.asarray(mesh.faces), device=device,
                            dtype=torch.int32)
    bvh = cubvh.cuBVH(verts, faces)
    bbox = np.asarray(mesh.bounds)
    extent = bbox[1] - bbox[0]

    rng = np.random.default_rng(0)
    # Sample points uniformly in a slightly-inflated bbox.
    pts_np = rng.uniform(
        bbox[0] - 0.05 * extent,
        bbox[1] + 0.05 * extent,
        size=(n_samples, 3),
    ).astype(np.float32)
    gt_inside = ground_truth_inside(pts_np)

    pts_t = torch.as_tensor(pts_np, device=device)
    out = bvh.signed_distance(pts_t, return_uvw=False)
    sdf = out[0] if isinstance(out, tuple) else out
    cubvh_inside = (sdf.detach().cpu().numpy() < 0)

    agreement = float((cubvh_inside == gt_inside).mean())
    return {
        "ray_sign_agreement": agreement,
        "ray_samples": n_samples,
        "fraction_inside_gt": float(gt_inside.mean()),
        "fraction_inside_cubvh": float(cubvh_inside.mean()),
    }


def make_even_crossings_gt(mesh: trimesh.Trimesh, device: str = "cuda",
                           max_bounces: int = 12, ray_axis: int = 0,
                           eps_factor: float = 1e-5):
    """Return a function that tells which points are inside a watertight
    mesh via parity of ray-mesh intersections (Jordan curve theorem).

    This is used as a ground-truth "inside" function in Stage 4, to compare
    against cubvh's signed_distance sign.

    Implementation
    --------------
    We fire one +X ray per test point and re-trace from just past each hit
    to count all hits. The loop is GPU-vectorized: at each iteration we
    keep only the "alive" rays still searching for the next hit. Typical
    watertight meshes have <= 6 crossings; we cap at ``max_bounces`` (12).

    Parameters
    ----------
    eps_factor : float
        Fraction of the bbox diagonal to advance past each hit before the
        next trace. Too small causes the next trace to re-hit the same
        triangle (infinite loop); too large skips thin features.
    """
    import cubvh
    verts = torch.as_tensor(np.asarray(mesh.vertices), device=device,
                            dtype=torch.float32)
    faces = torch.as_tensor(np.asarray(mesh.faces), device=device,
                            dtype=torch.int32)
    bvh = cubvh.cuBVH(verts, faces)
    bbox_diag = float(np.linalg.norm(
        np.asarray(mesh.bounds[1]) - np.asarray(mesh.bounds[0])))
    eps = bbox_diag * eps_factor

    def _gt(pts_np: np.ndarray) -> np.ndarray:
        pts = torch.as_tensor(pts_np, device=device, dtype=torch.float32)
        dirs = torch.zeros_like(pts)
        dirs[:, ray_axis] = 1.0
        n = pts.shape[0]
        hits = torch.zeros(n, dtype=torch.int32, device=device)
        origins = pts.clone()
        alive_idx = torch.arange(n, device=device)
        for _ in range(max_bounces):
            if alive_idx.numel() == 0:
                break
            o = origins[alive_idx]
            d = dirs[alive_idx]
            out = bvh.ray_trace(o, d)
            if isinstance(out, tuple):
                positions = out[0]
                face_id = out[1]
            else:
                face_id = out
                positions = None
            face_id = face_id.view(-1)
            hit_mask = face_id >= 0
            # Add a hit to alive rays that hit.
            hits.index_add_(0, alive_idx[hit_mask],
                            torch.ones(int(hit_mask.sum()),
                                       dtype=torch.int32, device=device))
            # Advance origins past hits for next iter.
            if positions is not None:
                positions = positions.view(-1, 3)
                new_o = positions[hit_mask] + eps * d[hit_mask]
                origins[alive_idx[hit_mask]] = new_o
            # Prune the still-alive set to only those that hit this round.
            alive_idx = alive_idx[hit_mask]
        return ((hits % 2) == 1).cpu().numpy()

    return _gt


# ---------------------------------------------------------------------------
# Check B: UltraShape VAE-based fragmentation detector
# ---------------------------------------------------------------------------
import contextlib


@contextlib.contextmanager
def _ultrashape_on_sys_path(repo_root: Optional[str]):
    """Context manager that temporarily inserts the UltraShape repo on
    ``sys.path`` while imports resolve, then removes it.

    Avoids permanent process-global mutation that could shadow unrelated
    packages named ``ultrashape.*``. Yields True iff it inserted a path.
    """
    import sys
    if not repo_root:
        yield False
        return
    if not Path(repo_root).exists():
        raise FileNotFoundError(
            f"ULTRASHAPE_REPO_ROOT points at {repo_root} which does not "
            f"exist. Clone https://github.com/Tencent/Hunyuan3D-2.1 (or the "
            f"UltraShape repo) there, or unset the env var if the "
            f"``ultrashape`` package is already installed."
        )
    inserted = False
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
        inserted = True
    try:
        yield inserted
    finally:
        if inserted:
            try:
                sys.path.remove(repo_root)
            except ValueError:
                pass


class UltraShapeVAE:
    """Lazy-loaded ShapeVAE wrapper.

    Builds the VAE from the UltraShape repo, loads its weights from the
    combined checkpoint, and provides an ``encode_decode_chamfer`` method
    for fragmentation detection.

    Usage:

        vae = UltraShapeVAE.load()
        cham = vae.encode_decode_chamfer(mesh)
        # > threshold => fragmented / out-of-distribution
    """

    def __init__(self, vae, device: str):
        self.vae = vae
        self.device = device

    @classmethod
    def load(
        cls,
        config_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        device: str = "cuda",
        ultrashape_repo_root: Optional[str] = None,
    ) -> "UltraShapeVAE":
        """Build the VAE from config + weights.

        Paths default to env vars ``ULTRASHAPE_VAE_CONFIG``,
        ``ULTRASHAPE_VAE_CKPT``, and ``ULTRASHAPE_REPO_ROOT``. If the
        UltraShape package is already importable you may omit the repo
        root; otherwise set ``ULTRASHAPE_REPO_ROOT`` or pass the path.
        """
        config_path = config_path or _config.get_vae_config_path()
        ckpt_path = ckpt_path or _config.get_vae_ckpt_path()
        ultrashape_repo_root = (ultrashape_repo_root
                                or _config.get_ultrashape_repo_root())
        if config_path is None or ckpt_path is None:
            raise RuntimeError(
                "UltraShape VAE weights not configured. Set the "
                "ULTRASHAPE_VAE_CONFIG and ULTRASHAPE_VAE_CKPT environment "
                "variables (or pass ``config_path`` / ``ckpt_path`` to "
                "UltraShapeVAE.load) pointing at the UltraShape 1.0 config "
                "YAML and ``ultrashape_v1.pt`` checkpoint. To run the "
                "pipeline without the VAE, pass ``--skip-vae``."
            )
        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"UltraShape VAE config not found at {config_path}. "
                f"Set ULTRASHAPE_VAE_CONFIG to the correct path."
            )
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(
                f"UltraShape VAE checkpoint not found at {ckpt_path}. "
                f"Set ULTRASHAPE_VAE_CKPT to the correct path."
            )

        from omegaconf import OmegaConf
        with _ultrashape_on_sys_path(ultrashape_repo_root):
            from ultrashape.utils.misc import instantiate_from_config

            cfg = OmegaConf.load(config_path)
            vae = instantiate_from_config(cfg.model.params.vae_config)
            weights = torch.load(ckpt_path, map_location="cpu",
                                 weights_only=False)
            vae.load_state_dict(weights["vae"], strict=True)
            vae.eval().to(device)
        return cls(vae, device)

    @torch.inference_mode()
    def encode_decode_chamfer(
        self,
        mesh: trimesh.Trimesh,
        num_surface_points: int = 409600,
        query_grid: int = 128,
        n_samples_chamfer: int = 20_000,
    ) -> dict:
        """Encode mesh surface → latents → decode on query grid → MC → chamfer.

        Returns dict with chamfer + timing + recon face count.

        Surface input format for the VAE is ``(1, N, 7)`` where:
          - columns 0..2 are XYZ in [-1, 1]^3
          - columns 3..5 are vertex normals
          - column 6 is a "sharp edge" flag (0.0 for uniform surface samples)
        The VAE then internally splits these 7 features into ``point_feats=4``
        per point: 3 normals + 1 sharp-edge flag; XYZ is treated separately.
        """
        with _ultrashape_on_sys_path(_config.get_ultrashape_repo_root()):
            from ultrashape.surface_loaders import normalize_mesh, sample_pointcloud
        from skimage import measure as skmeasure

        t0 = time.time()

        # 1. Normalize and sample surface.
        m_norm = mesh.copy()
        m_norm = normalize_mesh(m_norm, scale=0.98)
        points, normals = sample_pointcloud(m_norm, num=num_surface_points)
        # Append a zero sharp-edge flag column. VAE's encoder will then see
        # a (N, 7) surface where feats=(normals + sharp_flag) has 4 channels
        # per point, matching point_feats=4 in the config.
        sharp_flag = torch.zeros(points.shape[0], 1, dtype=torch.float32)
        surface = torch.cat([points, normals, sharp_flag], dim=-1)  # (N, 7)
        surface = surface.unsqueeze(0).to(self.device)

        # 2. Encode.
        latents = self.vae.encode(surface, sample_posterior=False)
        t_enc = time.time() - t0

        # 3. Decode to latent transformer outputs.
        t1 = time.time()
        lat_out = self.vae.decode(latents)
        t_dec = time.time() - t1

        # 4. Query at a dense grid.
        t2 = time.time()
        s = 1.0 / query_grid
        ax = torch.linspace(-1.0 + s/2, 1.0 - s/2, query_grid,
                            device=self.device)
        xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")
        queries = torch.stack([xx, yy, zz], dim=-1).view(1, -1, 3)
        chunk = 262144
        logits_all = torch.zeros(queries.shape[1], device=self.device,
                                 dtype=torch.float32)
        for start in range(0, queries.shape[1], chunk):
            end = min(start + chunk, queries.shape[1])
            q_chunk = queries[:, start:end]
            logit = self.vae.query(lat_out, q_chunk).view(-1).float()
            logits_all[start:end] = logit
        logits_vol = logits_all.view(query_grid, query_grid, query_grid)
        t_query = time.time() - t2

        # 5. Marching cubes at 0 (occupancy logits).
        t3 = time.time()
        try:
            verts_np, faces_np, _, _ = skmeasure.marching_cubes(
                logits_vol.detach().cpu().numpy(), level=0.0,
            )
            # Map from index space back to the query-grid's VOXEL-CENTER
            # world coordinates. The query grid lives at
            # ``-1 + s/2 .. 1 - s/2`` with ``s = 2/query_grid`` — voxel
            # centers, not grid corners. Using ``(v / (query_grid - 1)) * 2
            # - 1`` (corner-to-corner) biased the recon by ~1/query_grid
            # (~0.008 at grid=128) which inflated the chamfer calibration.
            s = 2.0 / float(query_grid)
            verts_np = -1.0 + s * 0.5 + verts_np.astype(np.float32) * s
            recon = trimesh.Trimesh(vertices=verts_np,
                                    faces=faces_np.astype(np.int64),
                                    process=True)
        except Exception as e:
            return {
                "chamfer": float("inf"),
                "error": str(e),
                "time_encode": t_enc,
                "time_decode": t_dec,
                "time_query": t_query,
                "time_mc": time.time() - t3,
            }
        t_mc = time.time() - t3

        # 6. Chamfer distance.
        t4 = time.time()
        from ._meshio import chamfer_distance
        cham = chamfer_distance(m_norm, recon, n_samples=n_samples_chamfer)
        t_cham = time.time() - t4
        return {
            "chamfer": float(cham),
            "recon_vertices": int(len(recon.vertices)),
            "recon_faces": int(len(recon.faces)),
            "time_encode": t_enc,
            "time_decode": t_dec,
            "time_query": t_query,
            "time_mc": t_mc,
            "time_chamfer": t_cham,
        }


# ---------------------------------------------------------------------------
# Check C: primitive-like detection
# ---------------------------------------------------------------------------
def primitive_score(mesh: trimesh.Trimesh, cone_deg: float = 10.0) -> dict:
    """Fraction of face area whose normal is axis-aligned (within cone_deg).

    A unit cube → 100% axis-aligned; sphere → ~0%; chair → 20-60%.
    """
    nrm = mesh.face_normals
    area = mesh.area_faces
    total = float(area.sum()) + 1e-12
    cos_tol = np.cos(np.deg2rad(cone_deg))
    axes = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=np.float64)
    cos = nrm @ axes.T
    axis_aligned = (cos > cos_tol).any(axis=1)
    return {
        "num_vertices": int(len(mesh.vertices)),
        "num_faces": int(len(mesh.faces)),
        "axis_aligned_area_fraction": float(area[axis_aligned].sum() / total),
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def filter_geometry(
    mesh: trimesh.Trimesh,
    cfg: Optional[FilterConfig] = None,
    vae: Optional[UltraShapeVAE] = None,
    ground_truth_inside=None,
    verbose: bool = False,
) -> FilterReport:
    cfg = cfg or FilterConfig()
    t0 = time.time()
    reasons: list[str] = []
    metrics: dict = {}

    # ---- Component count
    parts = mesh.split(only_watertight=False)
    metrics["num_components"] = int(len(parts))
    if len(parts) > cfg.max_components:
        reasons.append(f"too_many_components={len(parts)}")

    # ---- Watertight + winding
    metrics["is_watertight"] = bool(mesh.is_watertight)
    metrics["is_winding_consistent"] = bool(mesh.is_winding_consistent)

    # ---- Bookkeeping metric only — not a rejection criterion in the paper.
    if mesh.is_watertight:
        v = float(abs(mesh.volume))
        a = float(mesh.area) + 1e-12
        metrics["volume_area_ratio"] = v / a
    else:
        metrics["volume_area_ratio"] = None
    # Backwards-compat: if the caller explicitly set a threshold, honour it.
    if (cfg.volume_area_ratio_min is not None
        and metrics["volume_area_ratio"] is not None
        and metrics["volume_area_ratio"] < cfg.volume_area_ratio_min):
        reasons.append(
            f"thin_shell_vol_area={metrics['volume_area_ratio']:.4f}"
        )

    # ---- Paper §2.1 thin-shell rejection via interior-to-exterior point ratio,
    # piggy-backing on the same ray-parity GT used for Check A below.
    try:
        if ground_truth_inside is None:
            ground_truth_inside = make_even_crossings_gt(mesh, device=cfg.device)
        ray_metrics = ray_sign_agreement(mesh, ground_truth_inside,
                                         n_samples=cfg.ray_samples,
                                         device=cfg.device)
        metrics.update(ray_metrics)
        interior_ratio = ray_metrics.get("fraction_inside_gt")
        if (interior_ratio is not None
            and interior_ratio < cfg.thin_shell_interior_ratio_min):
            reasons.append(
                f"thin_shell interior_ratio={interior_ratio:.4f}"
                f" < {cfg.thin_shell_interior_ratio_min}"
            )
        # Ray sign agreement kept as diagnostic; only rejects when an
        # explicit positive threshold is configured (paper has no such step).
        if (cfg.ray_sign_agreement_threshold > 0.0
            and ray_metrics["ray_sign_agreement"]
            < cfg.ray_sign_agreement_threshold):
            reasons.append(
                f"ray_sign_disagreement={ray_metrics['ray_sign_agreement']:.3f}"
            )
    except Exception as e:
        metrics["ray_sign_error"] = str(e)

    # ---- Check B: VAE fragmentation
    if not cfg.skip_vae:
        try:
            if vae is None:
                vae = UltraShapeVAE.load(
                    config_path=cfg.vae_config_path,
                    ckpt_path=cfg.vae_ckpt_path,
                    device=cfg.device,
                )
            vae_metrics = vae.encode_decode_chamfer(mesh)
            metrics["vae"] = vae_metrics
            if (cfg.frag_vae_chamfer_threshold is not None
                and vae_metrics.get("chamfer", float("inf"))
                > cfg.frag_vae_chamfer_threshold):
                reasons.append(
                    f"vae_chamfer={vae_metrics['chamfer']:.4f}"
                    f" > {cfg.frag_vae_chamfer_threshold}"
                )
        except Exception as e:
            metrics["vae_error"] = str(e)

    # ---- Check C: primitive
    try:
        prim = primitive_score(mesh)
        metrics.update(prim)
        if (prim["num_vertices"] < cfg.primitive_vertex_threshold
            and prim["axis_aligned_area_fraction"]
            > cfg.primitive_axis_alignment_threshold):
            reasons.append("primitive_like")
    except Exception as e:
        metrics["primitive_error"] = str(e)

    is_valid = len(reasons) == 0
    return FilterReport(
        is_valid=is_valid,
        reasons=reasons,
        metrics=metrics,
        seconds=time.time() - t0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--save-report", type=Path, default=None)
    p.add_argument("--skip-vae", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)

    mesh = load_mesh(args.input)
    cfg = FilterConfig(skip_vae=args.skip_vae, device=args.device)
    rep = filter_geometry(mesh, cfg=cfg)
    print(rep.to_json())
    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(rep.to_json())
    return 0 if rep.is_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
