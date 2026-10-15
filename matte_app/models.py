from __future__ import annotations

import json
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .cancellation import Cancellation


@dataclass(frozen=True)
class PointHint:
    x: float
    y: float
    positive: bool = True


@dataclass(frozen=True)
class BoxHint:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True)
class RotoHint:
    text: str = ""
    max_size: int = 512
    chunk_frames: int = 4
    seed_path: str = ""
    seed_source: str = ""
    seed_frame: int = 1
    mask_folder: str = ""
    mask_source: str = ""
    mask_frame: int = 1
    trimap_path: str = ""
    trimap_frame: int = 1
    trimap_width: int = 24


Guidance = PointHint | BoxHint | RotoHint


class MatteModel(Protocol):
    name: str

    def predict(self, image: Image.Image, guidance: list[Guidance], cancellation: Cancellation | None = None) -> Image.Image:
        """Return an 8-bit grayscale alpha image at image resolution."""


def refine_with_guidance(alpha: Image.Image, guidance: list[Guidance]) -> Image.Image:
    """Turn point hints into component selection; use box hints as soft spatial constraints."""
    if not guidance:
        return alpha
    data = np.asarray(alpha, dtype=np.float32) / 255.0
    height, width = data.shape
    yy, xx = np.mgrid[:height, :width]
    point_hints = [hint for hint in guidance if isinstance(hint, PointHint) and hint.positive]

    if point_hints:
        # Work at a bounded resolution: connected-component selection is fast even
        # for very large exports, and preserves the full-resolution soft alpha below.
        longest_edge = 640
        scale = min(1.0, longest_edge / max(width, height))
        small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small_alpha = alpha.resize(small_size, Image.Resampling.BILINEAR)
        foreground = np.asarray(small_alpha) >= 48
        selected = np.zeros_like(foreground, dtype=bool)
        small_height, small_width = foreground.shape

        for hint in point_hints:
            target_x = min(small_width - 1, max(0, round(hint.x * (small_width - 1))))
            target_y = min(small_height - 1, max(0, round(hint.y * (small_height - 1))))
            if not foreground[target_y, target_x]:
                candidates_y, candidates_x = np.nonzero(foreground)
                if not len(candidates_x):
                    continue
                nearest = np.argmin((candidates_x - target_x) ** 2 + (candidates_y - target_y) ** 2)
                target_x, target_y = int(candidates_x[nearest]), int(candidates_y[nearest])
            if selected[target_y, target_x]:
                continue
            queue: deque[tuple[int, int]] = deque([(target_y, target_x)])
            selected[target_y, target_x] = True
            while queue:
                y, x = queue.popleft()
                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= next_y < small_height and 0 <= next_x < small_width and foreground[next_y, next_x] and not selected[next_y, next_x]:
                        selected[next_y, next_x] = True
                        queue.append((next_y, next_x))
        keep = Image.fromarray(np.uint8(selected * 255), "L").resize((width, height), Image.Resampling.BILINEAR)
        data *= np.asarray(keep, dtype=np.float32) / 255.0

    for hint in guidance:
        if isinstance(hint, PointHint):
            cx, cy = hint.x * width, hint.y * height
            radius = max(24, min(width, height) * 0.07)
            distance = np.hypot(xx - cx, yy - cy)
            influence = np.clip(1.0 - distance / radius, 0.0, 1.0)
            if not hint.positive:
                data *= 1.0 - influence
        else:
            left, right = sorted((hint.left * width, hint.right * width))
            top, bottom = sorted((hint.top * height, hint.bottom * height))
            feather = max(12, min(width, height) * 0.025)
            outside = np.maximum.reduce((left - xx, xx - right, top - yy, yy - bottom))
            keep = np.clip(1.0 - np.maximum(outside, 0.0) / feather, 0.0, 1.0)
            data *= keep
    return Image.fromarray(np.uint8(np.clip(data * 255, 0, 255)), "L")


class BiRefNetModel:
    name = "BiRefNet HR Matting"
    checkpoint = "ZhengPeng7/BiRefNet_HR-matting"

    def __init__(self) -> None:
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageSegmentation
        except ImportError as error:
            raise RuntimeError(
                "BiRefNet dependencies are missing. Start with: uv run --extra birefnet python main.py"
            ) from error
        self._torch = torch
        self._model = AutoModelForImageSegmentation.from_pretrained(
            self.checkpoint, trust_remote_code=True
        )
        self._model.eval()
        if torch.cuda.is_available():
            # This HR checkpoint is stored in fp16; stay in fp16 on CUDA for VRAM and speed.
            self._model.cuda().half()
        else:
            self._model.float()

    def predict(self, image: Image.Image, guidance: list[Guidance], cancellation: Cancellation | None = None) -> Image.Image:
        cancellation = cancellation or Cancellation()
        cancellation.check()
        self._load()
        cancellation.check()
        assert self._torch is not None and self._model is not None
        torch = self._torch
        original_size = image.size
        # The HR checkpoint's patch decoder expects a 2048 x 2048 feature grid.
        # Letterbox rather than distort the artwork, then crop its prediction back out.
        canvas_edge = 2048
        scale = canvas_edge / max(original_size)
        working_size = (
            max(1, round(original_size[0] * scale)),
            max(1, round(original_size[1] * scale)),
        )
        resized = image.convert("RGB").resize(working_size, Image.Resampling.LANCZOS)
        rgb = Image.new("RGB", (canvas_edge, canvas_edge))
        offset = ((canvas_edge - working_size[0]) // 2, (canvas_edge - working_size[1]) // 2)
        rgb.paste(resized, offset)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        if torch.cuda.is_available():
            tensor = tensor.cuda().half()
        cancellation.check()
        with torch.inference_mode():
            prediction = self._model(tensor)[-1].sigmoid()[0, 0]
        if torch.cuda.is_available():
            # Do not release the worker or enable a new job with GPU work still queued.
            torch.cuda.synchronize()
        cancellation.check()
        matte = Image.fromarray((prediction.float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "L")
        matte = matte.crop((offset[0], offset[1], offset[0] + working_size[0], offset[1] + working_size[1]))
        matte = matte.resize(original_size, Image.Resampling.LANCZOS)
        cancellation.check()
        result = refine_with_guidance(matte, guidance)
        cancellation.check()
        return result


class SDMatteCommandModel:
    """Adapter for the app's isolated LiteSDMatte runtime."""

    name = "SDMatte (external command)"

    def predict(self, image: Image.Image, guidance: list[Guidance], cancellation: Cancellation | None = None) -> Image.Image:
        cancellation = cancellation or Cancellation()
        cancellation.check()
        app_root = Path(__file__).resolve().parents[1]
        python = app_root / ".sdmatte-venv" / "Scripts" / "python.exe"
        runner = app_root / "matte_app" / "sdmatte_runner.py"
        if not python.exists() or not runner.exists():
            raise RuntimeError("LiteSDMatte runtime is not installed.")
        if not guidance:
            raise RuntimeError("LiteSDMatte needs a point or box prompt.")
        with tempfile.TemporaryDirectory(prefix="matte-app-") as folder:
            folder_path = Path(folder)
            input_path = folder_path / "input.png"
            output_path = folder_path / "matte.png"
            prompt_path = folder_path / "guidance.json"
            image.save(input_path,'PNG',compress_level=1)
            prompt_path.write_text(json.dumps([hint.__dict__ | {"type": type(hint).__name__} for hint in guidance]))
            cancellation.run_process(
                [str(python), str(runner), "--image", str(input_path), "--output", str(output_path), "--prompt", str(prompt_path)],
            )
            cancellation.check()
            if not output_path.exists():
                raise RuntimeError("LiteSDMatte finished without creating its requested output file.")
            return Image.open(output_path).convert("L").resize(image.size, Image.Resampling.LANCZOS)


_MODEL_CACHE: dict[str, MatteModel] = {}


def make_model(name: str) -> MatteModel:
    """Return a cached provider so changing guidance does not reload checkpoint weights."""
    if name.startswith(('BiRefNet','LiteSDMatte')):
        from .model_session import stop_session
        stop_session()
    if name not in _MODEL_CACHE:
        if name.startswith("BiRefNet"):
            _MODEL_CACHE[name] = BiRefNetModel()
        elif name.startswith("LiteSDMatte"):
            _MODEL_CACHE[name] = SDMatteCommandModel()
        else:
            from .roto import ROTO_MODELS, RotoCommandModel
            if name not in ROTO_MODELS:
                raise ValueError(f"Unknown model: {name}")
            _MODEL_CACHE[name] = RotoCommandModel(name)
    return _MODEL_CACHE[name]
