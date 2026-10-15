"""One-shot, isolated LiteSDMatte inference for the desktop application."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "SDMatte"
MODEL_DIR = ROOT / "models" / "LiteSDMatte"


def letterbox(image: Image.Image, edge: int = 1024) -> tuple[Image.Image, tuple[int, int], tuple[int, int]]:
    scale = edge / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    result = Image.new("RGB", (edge, edge))
    offset = ((edge - size[0]) // 2, (edge - size[1]) // 2)
    result.paste(image.convert("RGB").resize(size, Image.Resampling.LANCZOS), offset)
    return result, size, offset


def make_guidance(guidance: list[dict], size: tuple[int, int]) -> tuple[str, np.ndarray, np.ndarray]:
    width, height = size
    boxes = [item for item in guidance if item.get("type") == "BoxHint"]
    points = [item for item in guidance if item.get("type") == "PointHint" and item.get("positive", True)]
    if boxes:
        box = boxes[-1]
        left, right = sorted((box["left"], box["right"]))
        top, bottom = sorted((box["top"], box["bottom"]))
        mask = np.zeros((height, width), dtype=np.float32)
        mask[round(top * height) : max(round(bottom * height), round(top * height) + 1), round(left * width) : max(round(right * width), round(left * width) + 1)] = 1.0
        return "bbox_mask", mask, np.array([left, top, right, bottom], dtype=np.float32)
    if points:
        mask = Image.new("L", (width, height))
        coords: list[float] = []
        for point in points[:10]:
            x, y = round(point["x"] * (width - 1)), round(point["y"] * (height - 1))
            dot = Image.new("L", (width, height))
            dot.putpixel((x, y), 255)
            mask = Image.fromarray(np.maximum(np.asarray(mask), np.asarray(dot.filter(ImageFilter.GaussianBlur(radius=18)))))
            coords.extend((point["x"], point["y"]))
        coords.extend([0.0] * (20 - len(coords)))
        return "point_mask", np.asarray(mask, dtype=np.float32) / 255.0, np.array(coords, dtype=np.float32)
    raise ValueError("LiteSDMatte needs a point or box prompt.")


def apply_negative_points(alpha: Image.Image, guidance: list[dict]) -> Image.Image:
    """Negative clicks are local cleanup constraints after LiteSDMatte inference."""
    negatives = [item for item in guidance if item.get("type") == "PointHint" and not item.get("positive", True)]
    if not negatives:
        return alpha
    data = np.asarray(alpha, dtype=np.float32) / 255.0
    height, width = data.shape
    yy, xx = np.mgrid[:height, :width]
    radius = max(24, min(width, height) * 0.07)
    for point in negatives:
        distance = np.hypot(xx - point["x"] * width, yy - point["y"] * height)
        data *= 1.0 - np.clip(1.0 - distance / radius, 0.0, 1.0)
    return Image.fromarray(np.uint8(np.clip(data * 255, 0, 255)), "L")


def load_model(aux_input: str):
    import torch

    sys.path.insert(0, str(REPO))
    from modeling.LiteSDMatte import LiteSDMatte

    if not (MODEL_DIR / "LiteSDMatte.pth").exists():
        raise RuntimeError("LiteSDMatte weights are missing.")
    model = LiteSDMatte(
        pretrained_model_name_or_path=str(MODEL_DIR), load_weight=False, aux_input=aux_input,
        use_aux_input=True, use_coor_input=True, use_attention_mask=True,
        use_encoder_hidden_states=True, residual_connection=False,
        use_attention_mask_list=[True, True, True], use_encoder_hidden_states_list=[False, True, False],
    )
    checkpoint = torch.load(MODEL_DIR / "LiteSDMatte.pth", map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or missing:
        raise RuntimeError(f"LiteSDMatte checkpoint mismatch. Missing: {missing[:3]}; unexpected: {unexpected[:3]}")
    return model.cuda().half().eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    import torch

    guidance = json.loads(Path(args.prompt).read_text(encoding="utf-8"))
    source = Image.open(args.image).convert("RGB")
    working, working_size, offset = letterbox(source)
    aux_type, mask, coords = make_guidance(guidance, working.size)
    rgb = np.asarray(working, dtype=np.float32) / 127.5 - 1.0
    data = {
        "image": torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).half(),
        aux_type: torch.from_numpy(mask * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).half(),
        "is_trans": torch.zeros(1, dtype=torch.long), "caption": [""],
    }
    data["bbox_coords" if aux_type == "bbox_mask" else "point_coords"] = torch.from_numpy(coords).unsqueeze(0)
    model = load_model(aux_type)
    with torch.inference_mode():
        prediction = model(data)[0, 0].float().cpu().numpy()
    alpha = Image.fromarray(np.uint8(np.clip(prediction * 255, 0, 255)), "L")
    alpha = alpha.crop((offset[0], offset[1], offset[0] + working_size[0], offset[1] + working_size[1]))
    alpha = alpha.resize(source.size, Image.Resampling.LANCZOS)
    apply_negative_points(alpha, guidance).save(args.output, "PNG", compress_level=1)


if __name__ == "__main__":
    main()
