"""canonicalize.py -- Stage 3 of the UltraShape cleaning pipeline.

Normalize a mesh's orientation into the dataset-canonical frame. Paper uses
a trained network; we implement three methods and a hybrid:

    --method geom   : pure geometric heuristic (RANSAC ground plane + PCA
                      + symmetry check). Deterministic, CPU-only.
    --method vlm    : render six canonical views, ask Qwen3-VL to name the
                      up axis and forward axis. Requires the model loaded.
    --method hybrid : run both; if VLM and geom agree (up axis within 30°
                      of each other), use VLM's answer; otherwise fall back
                      to geom and flag the mesh for review.
    --method identity: no-op (keeps input orientation).

Output: a rotated mesh and a 3x3 rotation matrix R such that
    verts_canonical = (R @ verts_original.T).T + t
with t set so the rotated mesh is centered at the bbox centroid of the
input.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import trimesh

from . import _config
from ._meshio import PathLike, load_mesh, save_mesh


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class CanonicalizeReport:
    method: str
    up_axis_world: tuple[float, float, float]
    forward_axis_world: tuple[float, float, float]
    rotation_matrix: list  # 3x3 nested list
    vlm_up_label: Optional[str]
    vlm_front_label: Optional[str]
    vlm_agrees_with_geom: Optional[bool]
    needs_manual_review: bool
    seconds: float
    geom_inlier_fraction: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# RANSAC ground plane + PCA heuristic
# ---------------------------------------------------------------------------
def _ransac_ground_plane(
    pts: np.ndarray, n_iter: int = 400, dist_thresh: Optional[float] = None,
    candidate_down_axes: Optional[list[np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, float]:
    """Find the support plane: plane with most inliers pointing "down".

    We seed with the three bottom slabs (Y-, Z-, X- of the bbox) to bias
    toward a physical ground plane. Returns (normal, inlier_fraction).
    """
    rng = rng or np.random.default_rng(42)
    if len(pts) < 3:
        return np.array([0.0, 1.0, 0.0]), 0.0

    bbox_min = pts.min(0)
    bbox_max = pts.max(0)
    extent = bbox_max - bbox_min
    if dist_thresh is None:
        dist_thresh = 0.01 * float(np.linalg.norm(extent))

    # Candidate seed planes: take the lower 10% slab along each axis.
    if candidate_down_axes is None:
        candidate_down_axes = [
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        ]

    best_normal = np.array([0.0, 1.0, 0.0])
    best_inliers = 0.0
    for down in candidate_down_axes:
        axis = np.argmax(np.abs(down))
        sign = np.sign(down[axis])
        if sign < 0:
            # "bottom slab" = lowest 10% along this axis
            thr = bbox_min[axis] + 0.1 * extent[axis]
            mask = pts[:, axis] < thr
        else:
            thr = bbox_max[axis] - 0.1 * extent[axis]
            mask = pts[:, axis] > thr
        slab = pts[mask]
        if len(slab) < 100:
            continue

        # RANSAC on the slab.
        best_slab_normal = None
        best_slab_inl = 0
        for _ in range(n_iter):
            sel = rng.choice(len(slab), size=3, replace=False)
            a, b, c = slab[sel]
            n = np.cross(b - a, c - a)
            nl = np.linalg.norm(n)
            if nl < 1e-10:
                continue
            n = n / nl
            # Distance of all points to the plane.
            d = np.abs((pts - a) @ n)
            inl = int((d < dist_thresh).sum())
            if inl > best_slab_inl:
                best_slab_inl = inl
                best_slab_normal = n

        if best_slab_normal is None:
            continue
        # Orient the normal to point FROM the slab TO the rest of the body.
        # "up" direction is always from ground-slab centroid toward body
        # centroid; that is the unambiguous definition.
        centroid_all = pts.mean(0)
        slab_centroid = slab.mean(0)
        body_dir = centroid_all - slab_centroid
        body_dist = float(np.linalg.norm(body_dir))
        if body_dist < 1e-10:
            continue
        body_unit = body_dir / body_dist
        # Force normal into the same half-space as body_unit.
        if np.dot(best_slab_normal, body_unit) < 0:
            best_slab_normal = -best_slab_normal

        # Score weight: raw inliers × "slab is meaningfully below body" factor.
        # When the slab is at the TOP of a model (picked by mistake), body_dist
        # is small because the body centroid is just below the slab; when the
        # slab is at the BOTTOM, body_dist is close to half the extent in that
        # axis, which is the highest value.
        extent_ax = extent[axis]
        below_factor = min(1.0, body_dist / (0.3 * extent_ax))
        score = (best_slab_inl / len(pts)) * below_factor
        if score > best_inliers:
            best_inliers = score
            best_normal = best_slab_normal

    return best_normal, best_inliers


def _pca_forward_axis(
    pts: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """Longest horizontal direction (perpendicular to up)."""
    # Project to plane perpendicular to up.
    up = up / np.linalg.norm(up).clip(1e-10)
    proj = pts - (pts @ up)[:, None] * up[None, :]
    centered = proj - proj.mean(0)
    # PCA: largest eigenvector of covariance.
    c = centered.T @ centered
    _, v = np.linalg.eigh(c)
    # eigh returns ascending; last column is the largest.
    forward_candidate = v[:, -1]
    # Remove any residual component along up.
    forward_candidate -= up * np.dot(forward_candidate, up)
    forward_candidate /= np.linalg.norm(forward_candidate).clip(1e-10)
    # Disambiguate sign: for furniture/architecture, the "front" side
    # typically has more protruding/complex geometry than the back.
    # Heuristic: compute the third moment (skewness) of projection along
    # the candidate axis; flip so the positive tail is longer.
    proj1 = (proj - proj.mean(0)) @ forward_candidate  # (N,)
    third_moment = float((proj1 ** 3).mean())
    if third_moment < 0:
        forward_candidate = -forward_candidate
    return forward_candidate


def _build_rotation(up: np.ndarray, forward: np.ndarray) -> np.ndarray:
    """Return R such that R @ up_world = +Y and R @ forward_world = +Z.

    UltraShape convention (inherited from Hunyuan3D): up=+Y, front=+Z.
    The returned matrix is always a proper rotation (``det(R) = +1``);
    if the input ``(up, forward)`` frame is left-handed we flip ``right``
    so the output never contains a reflection (which would mirror the
    object instead of rotating it).
    """
    up = up / np.linalg.norm(up).clip(1e-10)
    # Orthonormalize forward w.r.t. up.
    forward = forward - up * np.dot(forward, up)
    forward = forward / np.linalg.norm(forward).clip(1e-10)
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right).clip(1e-10)

    # Columns of C are the object's axes in world coords.
    C = np.stack([right, up, forward], axis=1)  # (3,3)
    # R = C^T maps world -> canonical.
    R = C.T
    # Guard against reflection: if the (right, up, forward) frame is
    # left-handed (det = -1), flip ``right`` so R becomes a proper rotation.
    if np.linalg.det(R) < 0:
        right = -right
        C = np.stack([right, up, forward], axis=1)
        R = C.T
    return R


def _horizontal_face_up_axis(
    mesh: trimesh.Trimesh,
    cone_deg: float = 20.0,
    y_up_margin: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Pick up axis from face-normal histogram + HSSD/GLB +Y-up prior.

    Strategy:
    1. Test the three world axes (X, Y, Z). For each, sum face area whose
       outward normal is within cone_deg of ±axis. Pick the axis with the
       highest fraction.
    2. If +Y is within y_up_margin of the best score, prefer +Y (honor the
       GLB/HSSD convention that meshes ship Y-up).
    3. If no world axis dominates (score < 0.3), fall through to OBB axes.
    4. Sign convention: the "up" end is the one containing the BBOX CENTER.
       Specifically, if the vertex centroid is on the side of the bbox
       center opposite to the face-normal's positive half-space, we flip.
       More pragmatically: we flip so that the majority of geometry is BELOW
       the up axis (this matches the physical intuition of gravity).
    """
    import trimesh as tm
    faces_v = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]  # (F,3,3)
    e1 = faces_v[:, 1] - faces_v[:, 0]
    e2 = faces_v[:, 2] - faces_v[:, 0]
    nrm = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    total_area = float(area.sum()) + 1e-12
    nrm_u = nrm / (2 * area[:, None] + 1e-12)

    world_candidates = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    cos_tol = np.cos(np.deg2rad(cone_deg))

    def score_axis(a):
        a_unit = a / (np.linalg.norm(a) + 1e-12)
        cos_ = nrm_u @ a_unit
        mask = np.abs(cos_) > cos_tol
        return float(area[mask].sum()) / total_area, a_unit

    scores = [score_axis(a)[0] for a in world_candidates]
    best_idx = int(np.argmax(scores))
    best_score_world = scores[best_idx]
    y_score = scores[1]

    # Honour +Y convention if close to best.
    if y_score >= best_score_world - y_up_margin:
        chosen = world_candidates[1]
        chosen_score = y_score
    else:
        chosen = world_candidates[best_idx]
        chosen_score = best_score_world

    # If even the best world axis has low confidence, try OBB axes.
    if chosen_score < 0.3:
        try:
            _, transform = tm.bounds.oriented_bounds(mesh)
            R_obb = np.asarray(transform[:3, :3])
            obb_axes = [R_obb.T[:, i] for i in range(3)]
            for a in obb_axes:
                s, a_u = score_axis(a)
                if s > chosen_score:
                    chosen = a_u
                    chosen_score = s
        except Exception:
            pass

    # Sign disambiguation:
    # 1. If mesh has a floor contact (many vertices near Y=0 or the minimum
    #    along the chosen axis), that side is the bottom (ground). So we
    #    flip so chosen points FROM that side (bottom) TO the other side.
    # 2. Secondary tiebreaker: bigger flat-face-cluster BELOW the bbox
    #    center indicates the ground plane.
    v = np.asarray(mesh.vertices, dtype=np.float64)
    proj = v @ chosen
    pmin, pmax = float(proj.min()), float(proj.max())
    extent_along = pmax - pmin
    if extent_along < 1e-9:
        return chosen, float(chosen_score)

    # Heuristic 1: if the POSITIVE end is much closer to 0 than the negative
    # end, the mesh was shipped +axis-up (original bbox bottom sat at 0).
    # (Works for HSSD/Objaverse/GLB standard.)
    # floor = side with min |proj|
    if abs(pmin) > 1e-4 * extent_along or abs(pmax) > 1e-4 * extent_along:
        # Floor detection: the side closer to the WORLD origin along the axis
        # is usually the bottom of the model.
        if abs(pmin) < abs(pmax):
            # Negative end touches origin; +chosen points up (good).
            pass
        else:
            # Positive end touches origin; -chosen is up.
            chosen = -chosen
        return chosen, float(chosen_score)

    # Heuristic 2: Use area-weighted horizontal-face centroid.
    a_unit = chosen
    cos_ = nrm_u @ a_unit
    face_centroids = faces_v.mean(axis=1)
    proj_faces = face_centroids @ a_unit
    mask = np.abs(cos_) > cos_tol
    if mask.sum() >= 3:
        weights = area[mask]
        horiz_centroid = float(
            (proj_faces[mask] * weights).sum() / (weights.sum() + 1e-12))
        bbox_center_all = 0.5 * (pmin + pmax)
        if horiz_centroid > bbox_center_all:
            chosen = -chosen
    return chosen, float(chosen_score)


def _oriented_bbox_up_axis(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, float]:
    """Alias for horizontal-face up-axis detector (kept for back-compat)."""
    return _horizontal_face_up_axis(mesh)


def canonicalize_geom(
    mesh: trimesh.Trimesh,
    n_iter: int = 400,
) -> tuple[trimesh.Trimesh, np.ndarray, CanonicalizeReport]:
    """Geometric-only pose canonicalization (OBB + asymmetry)."""
    t = time.time()
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    up, asym_score = _oriented_bbox_up_axis(mesh)
    forward = _pca_forward_axis(pts, up)
    R = _build_rotation(up, forward)
    verts_can = (R @ pts.T).T
    bc = 0.5 * (verts_can.min(0) + verts_can.max(0))
    verts_can -= bc

    m_out = trimesh.Trimesh(vertices=verts_can, faces=mesh.faces.copy(),
                            process=False)
    report = CanonicalizeReport(
        method="geom",
        up_axis_world=tuple(float(x) for x in up),
        forward_axis_world=tuple(float(x) for x in forward),
        rotation_matrix=R.tolist(),
        vlm_up_label=None,
        vlm_front_label=None,
        vlm_agrees_with_geom=None,
        needs_manual_review=bool(asym_score < 0.05),
        seconds=time.time() - t,
        geom_inlier_fraction=float(asym_score),
    )
    return m_out, R, report


# ---------------------------------------------------------------------------
# VLM method: ask Qwen3-VL to identify up and forward
# ---------------------------------------------------------------------------
VLM_POSE_PROMPT_EN = """\
You are looking at six canonical views of a 3D object arranged in a 2x3\
 grid. Top row: (+X) (+Y) (+Z). Bottom row: (-X) (-Y) (-Z). Each cell\
 shows the object as seen when looking toward that axis from the camera\
 placed along the opposite axis.

Answer which axis points UP in the object's natural orientation and which\
 axis points FORWARD (front face). Return a STRICT JSON object with these\
 fields and NO extra text:

  - "up_axis": one of "+X", "-X", "+Y", "-Y", "+Z", "-Z"
  - "front_axis": one of "+X", "-X", "+Y", "-Y", "+Z", "-Z" (must be\
 orthogonal to up_axis)
  - "object_class": a short label (e.g. "chair", "lamp", "unidentifiable")
  - "confidence": 1-5 (1=guessing, 5=very confident)
  - "reasoning": one sentence.

Return only the JSON object."""

AXIS_MAP = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}


def _six_view_grid(mesh: trimesh.Trimesh, resolution: int = 384,
                   device: str = "cuda") -> np.ndarray:
    """Render 6 canonical views (+X/-X/+Y/-Y/+Z/-Z) into a 2x3 grid.

    We name each tile with the axis label overlaid so the VLM has a
    well-defined reference.
    """
    from .renderer import canonical_cameras, render_views, make_2x2_grid
    bbox = mesh.bounds
    center = 0.5 * (bbox[0] + bbox[1])
    radius = float(np.linalg.norm(bbox[1] - bbox[0])) * 0.5
    cams = canonical_cameras(center, radius=radius)
    # Map to axis-labelled views.
    # front=+Z, back=-Z, right=+X, left=-X, top=+Y, bottom=-Y.
    view_map = {
        "+X": cams["right"],
        "-X": cams["left"],
        "+Y": cams["top"],
        "-Y": cams["bottom"],
        "+Z": cams["front"],
        "-Z": cams["back"],
    }
    imgs = render_views(mesh, view_map, resolution=resolution, device=device)
    # 2x3: top row +X +Y +Z, bottom row -X -Y -Z.
    h, w, c = imgs["+X"].shape
    grid = np.ones((h * 2, w * 3, c), dtype=np.uint8) * 255
    order_top = ["+X", "+Y", "+Z"]
    order_bot = ["-X", "-Y", "-Z"]
    for i, k in enumerate(order_top):
        grid[0:h, i*w:(i+1)*w] = imgs[k]
    for i, k in enumerate(order_bot):
        grid[h:2*h, i*w:(i+1)*w] = imgs[k]
    # Add axis labels. Use PIL's default bitmap font; we used to hard-code
    # a DejaVu path that's absent on macOS/Windows clones.
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(grid)
        d = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        for i, k in enumerate(order_top):
            x, y = i*w + 8, 8
            d.rectangle([x-2, y-2, x+40, y+24], fill=(0, 0, 0))
            d.text((x, y), k, fill=(255, 255, 255), font=font)
        for i, k in enumerate(order_bot):
            x, y = i*w + 8, h + 8
            d.rectangle([x-2, y-2, x+40, y+24], fill=(0, 0, 0))
            d.text((x, y), k, fill=(255, 255, 255), font=font)
        grid = np.asarray(im)
    except Exception:
        pass
    return grid


def canonicalize_vlm(
    mesh: trimesh.Trimesh,
    vlm_client,  # Qwen3VLClient
    temp_png: PathLike,
    resolution: int = 384,
    device: str = "cuda",
) -> tuple[Optional[np.ndarray], dict]:
    """Ask the VLM for up/front axes. Returns (R or None, metadata)."""
    from PIL import Image
    grid = _six_view_grid(mesh, resolution=resolution, device=device)
    Image.fromarray(grid).save(str(temp_png))
    raw = vlm_client.generate(str(temp_png), VLM_POSE_PROMPT_EN,
                              max_new_tokens=256)
    from .vlm_filter import parse_vlm_response
    # Reuse parser; fields are different but the JSON extraction is shared.
    import json, re
    # Try parse strict.
    try:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None, {"raw_response": raw, "parse_error": True}
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            return None, {"raw_response": raw, "parse_error": True}

    up_label = str(parsed.get("up_axis", "")).strip().upper()
    fw_label = str(parsed.get("front_axis", "")).strip().upper()
    if up_label not in AXIS_MAP or fw_label not in AXIS_MAP:
        return None, {
            "raw_response": raw, "up_label": up_label, "front_label": fw_label,
            "parse_error": True,
        }
    up = AXIS_MAP[up_label]
    forward = AXIS_MAP[fw_label]
    if abs(np.dot(up, forward)) > 0.01:
        return None, {
            "raw_response": raw, "up_label": up_label, "front_label": fw_label,
            "parse_error": False, "not_orthogonal": True,
        }
    R = _build_rotation(up, forward)
    return R, {
        "raw_response": raw,
        "up_label": up_label,
        "front_label": fw_label,
        "object_class": parsed.get("object_class"),
        "confidence": parsed.get("confidence"),
        "parse_error": False,
    }


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------
def canonicalize_hybrid(
    mesh: trimesh.Trimesh,
    vlm_client,
    temp_png: PathLike,
    resolution: int = 384,
    device: str = "cuda",
    agreement_cosine_thresh: float = 0.8,
) -> tuple[trimesh.Trimesh, np.ndarray, CanonicalizeReport]:
    """Run VLM + geom; use VLM if they agree (up axes within ~37°)."""
    t = time.time()
    # Geom baseline
    _, R_geom, rep_geom = canonicalize_geom(mesh)
    # Recover up/forward in WORLD frame from rep_geom
    up_geom = np.array(rep_geom.up_axis_world)
    front_geom = np.array(rep_geom.forward_axis_world)

    # VLM
    R_vlm, meta = canonicalize_vlm(mesh, vlm_client, temp_png,
                                   resolution=resolution, device=device)
    if R_vlm is not None:
        # Recover world axes from R_vlm: R maps world -> canonical;
        # the WORLD up axis is the 2nd column of R^T.
        up_vlm = R_vlm.T[:, 1]
        front_vlm = R_vlm.T[:, 2]
        agree = float(np.abs(np.dot(up_geom, up_vlm))) >= agreement_cosine_thresh
    else:
        agree = False
        up_vlm = None
        front_vlm = None

    if R_vlm is not None and agree:
        R = R_vlm
        up_used = up_vlm
        forward_used = front_vlm
    else:
        R = R_geom
        up_used = up_geom
        forward_used = front_geom

    pts = np.asarray(mesh.vertices, dtype=np.float64)
    verts_can = (R @ pts.T).T
    bc = 0.5 * (verts_can.min(0) + verts_can.max(0))
    verts_can -= bc
    m_out = trimesh.Trimesh(vertices=verts_can, faces=mesh.faces.copy(),
                            process=False)

    report = CanonicalizeReport(
        method="hybrid",
        up_axis_world=tuple(float(x) for x in up_used),
        forward_axis_world=tuple(float(x) for x in forward_used),
        rotation_matrix=R.tolist(),
        vlm_up_label=meta.get("up_label"),
        vlm_front_label=meta.get("front_label"),
        vlm_agrees_with_geom=agree if R_vlm is not None else None,
        needs_manual_review=(R_vlm is None or not agree),
        seconds=time.time() - t,
        geom_inlier_fraction=rep_geom.geom_inlier_fraction,
    )
    return m_out, R, report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def canonicalize_mesh(
    mesh: trimesh.Trimesh,
    method: Literal["geom", "vlm", "hybrid", "identity"] = "geom",
    vlm_client=None,
    temp_png: Optional[PathLike] = None,
    resolution: int = 384,
    device: str = "cuda",
) -> tuple[trimesh.Trimesh, np.ndarray, CanonicalizeReport]:
    if method == "identity":
        R = np.eye(3)
        return (
            mesh.copy(),
            R,
            CanonicalizeReport(
                method="identity",
                up_axis_world=(0.0, 1.0, 0.0),
                forward_axis_world=(0.0, 0.0, 1.0),
                rotation_matrix=R.tolist(),
                vlm_up_label=None,
                vlm_front_label=None,
                vlm_agrees_with_geom=None,
                needs_manual_review=False,
                seconds=0.0,
            ),
        )
    if method == "geom":
        return canonicalize_geom(mesh)
    if method == "vlm":
        if vlm_client is None or temp_png is None:
            raise ValueError("vlm method requires vlm_client and temp_png")
        R, meta = canonicalize_vlm(mesh, vlm_client, temp_png,
                                   resolution=resolution, device=device)
        if R is None:
            raise RuntimeError(f"VLM could not parse pose: {meta}")
        pts = np.asarray(mesh.vertices, dtype=np.float64)
        v_can = (R @ pts.T).T
        bc = 0.5 * (v_can.min(0) + v_can.max(0))
        v_can -= bc
        m_out = trimesh.Trimesh(vertices=v_can, faces=mesh.faces.copy(),
                                process=False)
        up_used = R.T[:, 1]
        fw_used = R.T[:, 2]
        rep = CanonicalizeReport(
            method="vlm",
            up_axis_world=tuple(float(x) for x in up_used),
            forward_axis_world=tuple(float(x) for x in fw_used),
            rotation_matrix=R.tolist(),
            vlm_up_label=meta.get("up_label"),
            vlm_front_label=meta.get("front_label"),
            vlm_agrees_with_geom=None,
            needs_manual_review=bool(meta.get("parse_error") or meta.get("not_orthogonal")),
            seconds=0.0,  # filled by caller if needed
        )
        return m_out, R, rep
    if method == "hybrid":
        if vlm_client is None or temp_png is None:
            # Fall back silently to geom-only when the VLM is unavailable.
            return canonicalize_geom(mesh)
        return canonicalize_hybrid(mesh, vlm_client, temp_png,
                                   resolution=resolution, device=device)
    raise ValueError(f"Unknown method {method!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["geom", "vlm", "hybrid", "identity"],
                   default="geom")
    p.add_argument("--save-report", type=Path, default=None)
    p.add_argument("--temp-png", type=Path, default=None,
                   help="Where to write 6-view grid for VLM input")
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-path",
                   default=_config.get_qwen3vl_model_path(),
                   help="Qwen3-VL model dir or HF id (env: QWEN3VL_MODEL_PATH)")
    args = p.parse_args(argv)

    mesh = load_mesh(args.input)
    vlm_client = None
    if args.method in {"vlm", "hybrid"}:
        from .vlm_filter import Qwen3VLClient
        vlm_client = Qwen3VLClient.from_local(path=args.model_path, device=args.device)

    temp_png = args.temp_png or args.output.with_suffix(".6view.png")
    out_mesh, R, report = canonicalize_mesh(
        mesh, method=args.method, vlm_client=vlm_client, temp_png=temp_png,
        resolution=args.resolution, device=args.device,
    )
    save_mesh(out_mesh, args.output)
    if args.save_report:
        Path(args.save_report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_report).write_text(report.to_json())
    print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
