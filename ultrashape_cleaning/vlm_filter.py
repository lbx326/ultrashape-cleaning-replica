"""vlm_filter.py -- Stage 2 of the UltraShape cleaning pipeline.

Filters meshes by loading **Qwen3-VL-8B-Instruct** and asking it to score a
2x2 grid of canonical renders. The VLM rejects:

    - primitives (cubes/spheres, no semantic content)
    - ground planes (very flat, rug-like)
    - noisy scans (speckled reconstructions)
    - severe fragmentation visible from the outside

The VLM also outputs an aesthetic quality score 1-5 that can be used to
rank outputs or for percentile-based calibration.

Key design choices
------------------
1. We run rendering in the ``ultrashape`` env (needs cubvh) and inference
   in the ``buildingseg`` env (has transformers 4.57 with Qwen3-VL). The
   two-env split is plumbed via a PNG cache on disk — we write the 2x2
   grid in Stage 1's env, then the Stage 2 CLI loads it for inference.
2. Results are cached by sha256 of the INPUT mesh file; a mesh's cleaned
   variant need not re-trigger VLM calls.
3. Structured output: we prompt for a strict JSON object and validate the
   schema with Python.

Prompt design
-------------
We target the Qwen3-VL strengths: strong Chinese + English, good at
structured JSON, reasonable spatial reasoning. The prompt is in English
because the object ontology (furniture/architecture/vehicles/props) is
more stable in English for HSSD. A Chinese variant is provided too.

CLI
---

    # Rendering (needs ultrashape env)
    python -m ultrashape_cleaning.vlm_filter render \\
        --input mesh.glb --out-png grid.png --resolution 512

    # Inference (needs buildingseg env with Qwen3-VL)
    python -m ultrashape_cleaning.vlm_filter infer \\
        --grid grid.png --mesh-id <sha256> \\
        --out-json vlm.json --cache-dir .vlm_cache/

    # One-shot (tries inference in the current env; falls back gracefully)
    python -m ultrashape_cleaning.vlm_filter --input mesh.glb --out-json vlm.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ._meshio import PathLike, sha256_file


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
DEFAULT_PROMPT_EN = """\
You are inspecting four orthogonal views of a 3D object: front (top-left),\
 right (top-right), back (bottom-left), left (bottom-right) of a 2x2 grid.

Please assess the object against the following criteria and return a STRICT\
 JSON object (no prose, no markdown fences) with these fields:

  - "aesthetic_quality": integer 1-5 (1=unusable, 2=poor, 3=acceptable,\
 4=good, 5=excellent). Judge plausibility and visual appeal as a typical\
 household furniture / prop / appliance / architectural element.
  - "is_primitive": true if the shape is an unrefined primitive (plain\
 cube / sphere / cylinder with no semantic features), else false.
  - "is_ground_plane": true if the object is essentially a flat slab\
 (e.g. a rug, tile, thin mat) covering mostly one plane, else false.
  - "is_noisy_scan": true if the geometry looks like a noisy photogrammetric\
 or depth scan with rough/speckled surfaces or missing patches, else false.
  - "is_fragmented": true if the object visibly has disconnected pieces,\
 floating fragments, or large unexplained holes, else false.
  - "object_class": short English label (e.g. "chair", "sofa", "vase",\
 "unidentifiable"). Use "unidentifiable" if you cannot name a common object.
  - "reasoning": 1-2 sentences explaining the scores.

Return ONLY the JSON object. Do not wrap in ```json or any other decoration.\
"""

DEFAULT_PROMPT_ZH = """\
请检查一个 3D 物体的四个正交视图（左上=前视图，右上=右视图，左下=后视图，\
右下=左视图）。

请按以下标准评估，并严格返回 JSON 对象（不要 markdown 或注释）：

  - "aesthetic_quality": 1-5 整数（1=无法使用，5=非常优秀），作为普通家具\
/装饰/家电/建筑元素的质量评分。
  - "is_primitive": 若对象是几何原形（立方体/球体等），无语义特征，则 true。
  - "is_ground_plane": 若对象实际上是一块扁平板（如地毯、瓷砖、垫子），则 true。
  - "is_noisy_scan": 若几何看起来像噪声扫描（表面粗糙、有斑点、缺失部分），则 true。
  - "is_fragmented": 若有明显的不连通块、漂浮碎片或大的未知孔洞，则 true。
  - "object_class": 简短英文标签（如 "chair", "sofa", "vase", "unidentifiable"）。
  - "reasoning": 1-2 句英文解释。

仅返回 JSON。"""


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class VLMResult:
    mesh_sha256: str
    aesthetic_quality: Optional[int]
    is_primitive: Optional[bool]
    is_ground_plane: Optional[bool]
    is_noisy_scan: Optional[bool]
    is_fragmented: Optional[bool]
    object_class: Optional[str]
    reasoning: str
    accepted: bool
    rejection_reasons: list
    raw_response: str
    model: str
    prompt_lang: str
    seconds: float

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True,
                          ensure_ascii=False)


def _accept_decision(parsed: dict, min_quality: int = 2,
                     reject_primitive: bool = True,
                     reject_ground: bool = True,
                     reject_noisy: bool = True,
                     reject_fragmented: bool = True) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    q = parsed.get("aesthetic_quality")
    if isinstance(q, int) and q < min_quality:
        reasons.append(f"aesthetic_quality={q} < {min_quality}")
    if reject_primitive and parsed.get("is_primitive") is True:
        reasons.append("primitive")
    if reject_ground and parsed.get("is_ground_plane") is True:
        reasons.append("ground_plane")
    if reject_noisy and parsed.get("is_noisy_scan") is True:
        reasons.append("noisy_scan")
    if reject_fragmented and parsed.get("is_fragmented") is True:
        reasons.append("fragmented")
    return (len(reasons) == 0), reasons


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
_JSON_OBJECT = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def parse_vlm_response(text: str) -> dict:
    """Extract the first valid JSON object from VLM output, coercing fields."""
    text = text.strip()
    # Strip common prefixes/fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Try: whole text first, then greedy regex.
    candidates = [text]
    for m in _JSON_OBJECT.finditer(text):
        candidates.append(m.group(0))

    parsed: Optional[dict] = None
    for cand in candidates:
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                break
        except Exception:
            continue

    if parsed is None:
        return {
            "aesthetic_quality": None,
            "is_primitive": None,
            "is_ground_plane": None,
            "is_noisy_scan": None,
            "is_fragmented": None,
            "object_class": None,
            "reasoning": text[:500],
            "_parse_error": True,
        }

    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            v = v.strip().lower()
            if v in {"true", "yes", "1"}:
                return True
            if v in {"false", "no", "0"}:
                return False
        return None

    def _to_int(v):
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        if isinstance(v, str):
            m = re.search(r"-?\d+", v)
            if m:
                return int(m.group(0))
        if isinstance(v, float):
            return int(v)
        return None

    return {
        "aesthetic_quality": _to_int(parsed.get("aesthetic_quality")),
        "is_primitive": _to_bool(parsed.get("is_primitive")),
        "is_ground_plane": _to_bool(parsed.get("is_ground_plane")),
        "is_noisy_scan": _to_bool(parsed.get("is_noisy_scan")),
        "is_fragmented": _to_bool(parsed.get("is_fragmented")),
        "object_class": parsed.get("object_class"),
        "reasoning": str(parsed.get("reasoning", ""))[:1000],
        "_parse_error": False,
    }


# ---------------------------------------------------------------------------
# Rendering helper (delegates to renderer.py)
# ---------------------------------------------------------------------------
def render_grid_png(mesh_path: PathLike, grid_png: PathLike,
                   resolution: int = 512, device: str = "cuda") -> str:
    """Render the 2x2 grid for a mesh and write to grid_png. Returns sha256.

    Only works in an env with cubvh + torch + trimesh (i.e. ultrashape).
    """
    from PIL import Image
    from ._meshio import load_mesh
    from .renderer import render_four_views, make_2x2_grid

    grid_png = Path(grid_png)
    grid_png.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(mesh_path)
    views = render_four_views(mesh, resolution=resolution, device=device)
    grid = make_2x2_grid(views, layout=["front", "right", "back", "left"])
    Image.fromarray(grid).save(str(grid_png))
    return sha256_file(mesh_path)


# ---------------------------------------------------------------------------
# Qwen3-VL model holder (lazy singleton)
# ---------------------------------------------------------------------------
class Qwen3VLClient:
    """Lazy-loaded Qwen3-VL-8B-Instruct with batched inference.

    Tries, in order, transformers.Qwen3VLForConditionalGeneration (native in
    4.57+), transformers.AutoModelForImageTextToText with trust_remote_code,
    and finally falls back to an explicit AutoModel + config-driven
    architecture import.
    """
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_impl: Optional[str] = None,
    ):
        import torch
        self._torch = torch
        self.model_path = model_path
        self.device = device

        from transformers import AutoProcessor
        # Prefer the dedicated class; fall back to AutoModelForImageTextToText.
        model_loaded = False
        model = None
        try:
            from transformers import Qwen3VLForConditionalGeneration as Q3VL
            print(f"[Qwen3VL] loading via Qwen3VLForConditionalGeneration ...")
            model = Q3VL.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, dtype),
                device_map="auto" if device == "cuda" else None,
                attn_implementation=attn_impl,
            )
            model_loaded = True
        except (ImportError, AttributeError) as e:
            print(f"[Qwen3VL] Qwen3VLForConditionalGeneration unavailable ({e}); "
                  f"trying AutoModelForImageTextToText")

        if not model_loaded:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=getattr(torch, dtype),
                device_map="auto" if device == "cuda" else None,
                attn_implementation=attn_impl,
            )

        self.model = model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model_name = os.path.basename(str(model_path).rstrip("/"))

    @classmethod
    def from_local(cls, path: str = "/moganshan/afs_a/anmt/action/Qwen3-VL/"
                                     "Qwen3-VL-8B-Instruct/",
                   **kwargs) -> "Qwen3VLClient":
        return cls(path, **kwargs)

    def generate(
        self,
        image_path: PathLike,
        prompt: str,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.1,
    ) -> str:
        """Run inference on one image + prompt, return decoded assistant text.

        We use the Qwen3-VL chat template with a single user turn that embeds
        one image. Matches the sample in infer_sample.py.
        """
        import torch
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": prompt},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens,
                          do_sample=do_sample,
                          repetition_penalty=1.2)
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            gen_ids = self.model.generate(**inputs, **gen_kwargs)
        gen_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids
            in zip(inputs.input_ids, gen_ids)
        ]
        out = self.processor.batch_decode(
            gen_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return out


# ---------------------------------------------------------------------------
# High-level Stage 2 driver
# ---------------------------------------------------------------------------
def run_vlm_filter(
    grid_png: PathLike,
    mesh_sha256: str,
    client: Qwen3VLClient,
    prompt: str = DEFAULT_PROMPT_EN,
    prompt_lang: str = "en",
    cache_dir: Optional[PathLike] = None,
    max_new_tokens: int = 512,
    accept_cfg: Optional[dict] = None,
) -> VLMResult:
    """Run Qwen3-VL on a rendered grid image and produce a VLMResult."""
    accept_cfg = accept_cfg or {}
    cache_dir = Path(cache_dir) if cache_dir else None
    cache_file: Optional[Path] = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{mesh_sha256}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                return VLMResult(**cached)
            except Exception:
                pass

    t = time.time()
    raw = client.generate(grid_png, prompt, max_new_tokens=max_new_tokens)
    parsed = parse_vlm_response(raw)
    accepted, reasons = _accept_decision(parsed, **accept_cfg)
    result = VLMResult(
        mesh_sha256=mesh_sha256,
        aesthetic_quality=parsed.get("aesthetic_quality"),
        is_primitive=parsed.get("is_primitive"),
        is_ground_plane=parsed.get("is_ground_plane"),
        is_noisy_scan=parsed.get("is_noisy_scan"),
        is_fragmented=parsed.get("is_fragmented"),
        object_class=parsed.get("object_class"),
        reasoning=parsed.get("reasoning", ""),
        accepted=accepted,
        rejection_reasons=reasons,
        raw_response=raw,
        model=client.model_name,
        prompt_lang=prompt_lang,
        seconds=time.time() - t,
    )
    if cache_file:
        cache_file.write_text(result.to_json(), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli_render(args: argparse.Namespace) -> int:
    sha = render_grid_png(args.input, args.out_png,
                          resolution=args.resolution, device=args.device)
    print(f"sha256: {sha}")
    print(f"wrote: {args.out_png}")
    return 0


def _cli_infer(args: argparse.Namespace) -> int:
    client = Qwen3VLClient.from_local(
        path=args.model_path, device=args.device,
        dtype=args.dtype, attn_impl=args.attn_impl,
    )
    prompt = DEFAULT_PROMPT_ZH if args.prompt_lang == "zh" else DEFAULT_PROMPT_EN
    result = run_vlm_filter(
        grid_png=args.grid,
        mesh_sha256=args.mesh_id,
        client=client,
        prompt=prompt,
        prompt_lang=args.prompt_lang,
        cache_dir=args.cache_dir,
    )
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(result.to_json(), encoding="utf-8")
    print(result.to_json())
    return 0


def _cli_serve(args: argparse.Namespace) -> int:
    """Persistent daemon mode. Reads JSONL from stdin, writes JSONL to stdout.

    Protocol
    --------
    Input (one JSON object per line):
        {"grid": "...", "mesh_id": "...", "out_json": "...",
         "cache_dir": "...", "prompt_lang": "en"}

    Output (one JSON object per line):
        {"grid": "...", "mesh_id": "...", "ok": true,
         "result": <VLMResult as dict>}
        # or
        {"grid": "...", "mesh_id": "...", "ok": false, "error": "..."}

    Termination: a line containing {"cmd": "quit"} or EOF.
    """
    import sys as _sys
    client = Qwen3VLClient.from_local(
        path=args.model_path, device=args.device,
        dtype=args.dtype, attn_impl=args.attn_impl,
    )
    print(json.dumps({"ready": True, "model": client.model_name}),
          flush=True, file=_sys.stdout)
    for line in _sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            _sys.stdout.write(json.dumps(
                {"ok": False, "error": f"bad_json:{e}"}) + "\n")
            _sys.stdout.flush()
            continue
        if req.get("cmd") == "quit":
            break
        try:
            prompt = (DEFAULT_PROMPT_ZH
                      if req.get("prompt_lang") == "zh"
                      else DEFAULT_PROMPT_EN)
            result = run_vlm_filter(
                grid_png=req["grid"],
                mesh_sha256=req["mesh_id"],
                client=client,
                prompt=prompt,
                prompt_lang=req.get("prompt_lang", "en"),
                cache_dir=req.get("cache_dir"),
            )
            out_path = req.get("out_json")
            if out_path:
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(out_path).write_text(result.to_json(), encoding="utf-8")
            _sys.stdout.write(json.dumps({
                "grid": req.get("grid"),
                "mesh_id": req.get("mesh_id"),
                "ok": True,
                "result": dataclasses.asdict(result),
            }, ensure_ascii=False) + "\n")
            _sys.stdout.flush()
        except Exception as e:
            import traceback
            _sys.stdout.write(json.dumps({
                "grid": req.get("grid"),
                "mesh_id": req.get("mesh_id"),
                "ok": False,
                "error": str(e),
                "traceback": traceback.format_exc()[-500:],
            }) + "\n")
            _sys.stdout.flush()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    # render
    pr = sub.add_parser("render", help="Render 2x2 grid PNG for VLM input")
    pr.add_argument("--input", type=Path, required=True)
    pr.add_argument("--out-png", type=Path, required=True)
    pr.add_argument("--resolution", type=int, default=512)
    pr.add_argument("--device", default="cuda")
    pr.set_defaults(func=_cli_render)

    # infer
    pi = sub.add_parser("infer", help="Run Qwen3-VL on an existing grid PNG")
    pi.add_argument("--grid", type=Path, required=True)
    pi.add_argument("--mesh-id", required=True,
                    help="sha256 of the source mesh for caching")
    pi.add_argument("--out-json", type=Path, required=True)
    pi.add_argument("--cache-dir", type=Path, default=None)
    pi.add_argument("--model-path",
                    default="/moganshan/afs_a/anmt/action/Qwen3-VL/"
                            "Qwen3-VL-8B-Instruct/")
    pi.add_argument("--prompt-lang", choices=["en", "zh"], default="en")
    pi.add_argument("--device", default="cuda")
    pi.add_argument("--dtype", default="bfloat16")
    pi.add_argument("--attn-impl", default=None,
                    help="e.g. flash_attention_2 or sdpa")
    pi.set_defaults(func=_cli_infer)

    # serve (daemon)
    ps = sub.add_parser("serve", help="Persistent JSONL daemon mode")
    ps.add_argument("--model-path",
                    default="/moganshan/afs_a/anmt/action/Qwen3-VL/"
                            "Qwen3-VL-8B-Instruct/")
    ps.add_argument("--device", default="cuda")
    ps.add_argument("--dtype", default="bfloat16")
    ps.add_argument("--attn-impl", default=None)
    ps.set_defaults(func=_cli_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
