"""renderer.py -- GPU mesh renderer for VLM inputs and canonicalization previews.

Because pyrender is not available in any of the project environments, we
implement a minimal GPU rasterizer via cubvh's ray tracer. For each pixel we
cast a ray from the virtual camera, hit the mesh, and shade the hit face by
combining ambient + Lambertian diffuse with a key directional light and a
fill light. Output is an RGB numpy uint8 array.

This is not photorealistic and there are no shadows/textures, but the renders
are clean, deterministic, and absolutely good enough for a VLM quality
filter (shape, silhouette, topology are all preserved).

Usage:

    from ultrashape_cleaning.renderer import render_four_views, make_2x2_grid
    rgbs = render_four_views(mesh, resolution=512, device='cuda')
    grid = make_2x2_grid(rgbs, layout=['front','right','back','left'])
"""
from __future__ import annotations

import dataclasses
from typing import Iterable, Optional

import numpy as np
import torch
import trimesh

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Camera:
    """Pinhole camera; looks from `eye` toward `target` with `up` vector."""
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 35.0

    def view_rays(self, resolution: int, device: str = "cuda"
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (H*W, 3) ray origins and directions for a square image."""
        H = W = int(resolution)
        eye = torch.as_tensor(self.eye, device=device, dtype=torch.float32)
        target = torch.as_tensor(self.target, device=device, dtype=torch.float32)
        up = torch.as_tensor(self.up, device=device, dtype=torch.float32)
        forward = target - eye
        forward = forward / forward.norm().clamp_min(1e-8)
        right = torch.linalg.cross(forward, up)
        right = right / right.norm().clamp_min(1e-8)
        true_up = torch.linalg.cross(right, forward)
        # Pixel grid in [-1, 1]
        sy, sx = torch.meshgrid(
            torch.linspace(1, -1, H, device=device),  # y flip so pixel 0 is top
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        tan = float(np.tan(np.radians(self.fov_deg) / 2.0))
        # Direction per pixel: forward + x*right*tan + y*up*tan
        dirs = (forward[None, None, :]
                + sx[..., None] * right[None, None, :] * tan
                + sy[..., None] * true_up[None, None, :] * tan)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dirs = dirs.reshape(-1, 3).contiguous()
        origins = eye[None, :].expand(H * W, -1).contiguous()
        return origins, dirs


def canonical_cameras(center: np.ndarray, radius: float, fov_deg: float = 35.0
                     ) -> dict[str, Camera]:
    """Six canonical views around a center at distance ~2.5*radius.

    Uses +Y as up (standard model convention). Returns dict:
        front  (+Z), back   (-Z), right (+X),  left  (-X),
        top    (+Y), bottom (-Y)
    """
    d = max(float(radius), 1e-6) * 2.5
    c = tuple(float(x) for x in center)
    up = (0.0, 1.0, 0.0)
    cams = {
        "front":  Camera(eye=(c[0],     c[1],     c[2] + d), target=c, up=up, fov_deg=fov_deg),
        "back":   Camera(eye=(c[0],     c[1],     c[2] - d), target=c, up=up, fov_deg=fov_deg),
        "right":  Camera(eye=(c[0] + d, c[1],     c[2]    ), target=c, up=up, fov_deg=fov_deg),
        "left":   Camera(eye=(c[0] - d, c[1],     c[2]    ), target=c, up=up, fov_deg=fov_deg),
        "top":    Camera(eye=(c[0],     c[1] + d, c[2]    ), target=c, up=(0.0, 0.0, -1.0), fov_deg=fov_deg),
        "bottom": Camera(eye=(c[0],     c[1] - d, c[2]    ), target=c, up=(0.0, 0.0,  1.0), fov_deg=fov_deg),
    }
    return cams


# ---------------------------------------------------------------------------
# Shading
# ---------------------------------------------------------------------------
def _shade(
    hit_positions: torch.Tensor,       # (N, 3)
    hit_face_normals: torch.Tensor,     # (N, 3)
    view_dirs: torch.Tensor,            # (N, 3) (unit, from eye toward point)
    hit_mask: torch.Tensor,             # (N,) bool
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    key_dir: tuple[float, float, float] = (-0.4, -0.6, -0.7),
    key_color: tuple[float, float, float] = (1.0, 0.98, 0.94),
    fill_dir: tuple[float, float, float] = (0.6, 0.2, 0.3),
    fill_color: tuple[float, float, float] = (0.5, 0.55, 0.65),
    ambient: tuple[float, float, float] = (0.22, 0.22, 0.22),
    base_color: tuple[float, float, float] = (0.72, 0.72, 0.72),
) -> torch.Tensor:
    """Return (N, 3) linear RGB color in [0, 1] for each ray."""
    dev = hit_positions.device
    n = hit_positions.shape[0]
    out = torch.empty((n, 3), device=dev, dtype=torch.float32)
    out[:, 0] = background[0]
    out[:, 1] = background[1]
    out[:, 2] = background[2]
    if not hit_mask.any():
        return out

    nrm = hit_face_normals[hit_mask]
    # Ensure normals face the camera: flip if dot(nrm, view_dir) > 0
    vd = view_dirs[hit_mask]
    flip = (nrm * vd).sum(-1, keepdim=True) > 0
    nrm = torch.where(flip, -nrm, nrm)

    key_d = torch.as_tensor(key_dir, device=dev, dtype=torch.float32)
    key_d = -key_d / key_d.norm().clamp_min(1e-8)  # light direction: TO the light
    fill_d = torch.as_tensor(fill_dir, device=dev, dtype=torch.float32)
    fill_d = -fill_d / fill_d.norm().clamp_min(1e-8)

    key_c = torch.as_tensor(key_color, device=dev, dtype=torch.float32)
    fill_c = torch.as_tensor(fill_color, device=dev, dtype=torch.float32)
    amb = torch.as_tensor(ambient, device=dev, dtype=torch.float32)
    base = torch.as_tensor(base_color, device=dev, dtype=torch.float32)

    kd = (nrm @ key_d).clamp(min=0.0)
    fd = (nrm @ fill_d).clamp(min=0.0)
    shade = amb[None, :] + kd[:, None] * key_c[None, :] + 0.5 * fd[:, None] * fill_c[None, :]
    out[hit_mask] = (base[None, :] * shade).clamp(0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Core render
# ---------------------------------------------------------------------------
def render_view(
    mesh: trimesh.Trimesh,
    camera: Camera,
    resolution: int = 512,
    device: str = "cuda",
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_cached_bvh: Optional[object] = None,
) -> np.ndarray:
    """Render one view to (H, W, 3) uint8 numpy array."""
    import cubvh
    verts = torch.as_tensor(np.asarray(mesh.vertices), device=device, dtype=torch.float32)
    faces = torch.as_tensor(np.asarray(mesh.faces), device=device, dtype=torch.int32)
    if use_cached_bvh is None:
        bvh = cubvh.cuBVH(verts, faces)
    else:
        bvh = use_cached_bvh
    origins, dirs = camera.view_rays(resolution, device=device)
    # cubvh.ray_trace: inputs (N,3) rays. Returns (positions, face_id, depths).
    # The signature from cubvh/api.py: signed_distance is similar; ray_trace
    # returns a tuple (positions, face_id, depth) in some versions — the
    # code above returned the tuple, so we handle both.
    out = bvh.ray_trace(origins, dirs)
    if isinstance(out, tuple):
        if len(out) == 3:
            positions, face_id, depth = out
        elif len(out) == 2:
            positions, face_id = out
        else:
            raise RuntimeError(f"Unknown cubvh.ray_trace return arity {len(out)}")
    else:
        # Some versions return just face_id.
        face_id = out
        positions = origins + dirs  # unused
    face_id = face_id.view(-1).long()
    hit_mask = face_id >= 0

    # Compute per-face normals on GPU.
    vf = verts[faces.long()]  # (F, 3, 3)
    e1 = vf[:, 1] - vf[:, 0]
    e2 = vf[:, 2] - vf[:, 0]
    face_normals = torch.linalg.cross(e1, e2)
    face_normals = face_normals / face_normals.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    hit_normals = torch.zeros((face_id.numel(), 3), device=device, dtype=torch.float32)
    hit_normals[hit_mask] = face_normals[face_id[hit_mask]]

    colors = _shade(
        hit_positions=positions.view(-1, 3),
        hit_face_normals=hit_normals,
        view_dirs=dirs,
        hit_mask=hit_mask,
        background=background,
    )
    img = (colors.view(resolution, resolution, 3) * 255.0).clamp(0, 255)
    return img.detach().cpu().numpy().astype(np.uint8)


def render_views(
    mesh: trimesh.Trimesh,
    cameras: dict[str, Camera],
    resolution: int = 512,
    device: str = "cuda",
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, np.ndarray]:
    """Render multiple views efficiently by reusing the BVH."""
    import cubvh
    verts = torch.as_tensor(np.asarray(mesh.vertices), device=device, dtype=torch.float32)
    faces = torch.as_tensor(np.asarray(mesh.faces), device=device, dtype=torch.int32)
    bvh = cubvh.cuBVH(verts, faces)
    return {
        name: render_view(mesh, cam, resolution=resolution, device=device,
                          background=background, use_cached_bvh=bvh)
        for name, cam in cameras.items()
    }


def render_four_views(
    mesh: trimesh.Trimesh,
    resolution: int = 512,
    device: str = "cuda",
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, np.ndarray]:
    """Render front/right/back/left for a mesh centered at its bbox center."""
    bbox = mesh.bounds
    center = 0.5 * (bbox[0] + bbox[1])
    radius = float(np.linalg.norm(bbox[1] - bbox[0])) * 0.5
    cams = canonical_cameras(center, radius=radius)
    wanted = ["front", "right", "back", "left"]
    cams = {k: cams[k] for k in wanted}
    return render_views(mesh, cams, resolution=resolution, device=device,
                        background=background)


# ---------------------------------------------------------------------------
# Grid composition
# ---------------------------------------------------------------------------
def make_2x2_grid(
    views: dict[str, np.ndarray], layout: Optional[Iterable[str]] = None,
    label: bool = True
) -> np.ndarray:
    """Tile a dict of per-view images into a 2x2 grid image."""
    layout = list(layout or ["front", "right", "back", "left"])
    if len(layout) != 4:
        raise ValueError("2x2 grid requires exactly 4 views")
    h, w, c = views[layout[0]].shape
    grid = np.zeros((h * 2, w * 2, c), dtype=np.uint8)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for name, (r, c_) in zip(layout, positions):
        grid[r*h:(r+1)*h, c_*w:(c_+1)*w] = views[name]
    if label:
        # Simple labeling: draw filled rectangles in the upper left corner
        # and use PIL for text overlay.
        try:
            from PIL import Image, ImageDraw, ImageFont
            im = Image.fromarray(grid)
            draw = ImageDraw.Draw(im)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    size=max(10, h // 25),
                )
            except Exception:
                font = ImageFont.load_default()
            for name, (r, c_) in zip(layout, positions):
                x, y = c_ * w + 6, r * h + 6
                draw.rectangle([x - 2, y - 2, x + 90, y + 22], fill=(0, 0, 0))
                draw.text((x, y), name, fill=(255, 255, 255), font=font)
            grid = np.asarray(im)
        except Exception:
            pass
    return grid
