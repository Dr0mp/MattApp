"""Model download CLI for Matte App roto, tracking, and matting pipelines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def download_model(key: str, models_dir: Path, token: str | None = None) -> None:
    key = key.lower().strip()
    print(f"--> Starting download for [{key}]...")
    kwargs = {"token": token} if token else {}

    if key == "sam2":
        snapshot_download(
            "facebook/sam2.1-hiera-small",
            local_dir=models_dir / "SAM2",
            allow_patterns=["*.json", "model.safetensors"],
            **kwargs,
        )
    elif key == "matanyone":
        snapshot_download(
            "PeiqingYang/MatAnyone",
            local_dir=models_dir / "MatAnyone",
            allow_patterns=["config.json", "model.safetensors"],
            **kwargs,
        )
    elif key == "matanyone2":
        snapshot_download(
            "PeiqingYang/MatAnyone2",
            local_dir=models_dir / "MatAnyone2",
            allow_patterns=["config.json", "model.safetensors"],
            **kwargs,
        )
    elif key == "sam2matting":
        snapshot_download(
            "FudanCVL/SAM2Matting",
            local_dir=models_dir / "SAM2Matting",
            allow_patterns=[
                "checkpoints/SAM2Matting-SAM2.1Tiny.pt",
                "checkpoints/SAM2Matting-SAM2.1Base+.pt",
            ],
            **kwargs,
        )
    elif key in ("sam31", "sam3.1"):
        snapshot_download(
            "facebook/sam3.1",
            local_dir=models_dir / "SAM3.1",
            allow_patterns=["sam3.1_multiplex.pt"],
            **kwargs,
        )
    elif key == "sam3":
        snapshot_download(
            "facebook/sam3",
            local_dir=models_dir / "SAM3",
            allow_patterns=["*.json", "*.txt", "model.safetensors"],
            **kwargs,
        )
    elif key == "vitmatte":
        snapshot_download(
            "hustvl/vitmatte-small-composition-1k",
            local_dir=models_dir / "ViTMatte",
            allow_patterns=["*.json", "model.safetensors", "pytorch_model.bin"],
            **kwargs,
        )
    elif key == "videomama":
        snapshot_download(
            "SammyLim/VideoMaMa",
            local_dir=models_dir / "VideoMaMa",
            allow_patterns=["unet/*"],
            **kwargs,
        )
        snapshot_download(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            local_dir=models_dir / "SVD",
            allow_patterns=[
                "model_index.json",
                "feature_extractor/*",
                "image_encoder/config.json",
                "image_encoder/model.fp16.safetensors",
                "vae/config.json",
                "vae/diffusion_pytorch_model.fp16.safetensors",
                "scheduler/*",
            ],
            **kwargs,
        )
    elif key == "litesdmatte":
        snapshot_download(
            "LongfeiHuang/LiteSDMatte",
            local_dir=models_dir / "LiteSDMatte",
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model key: {key}")

    print(f"--> [OK] Finished download for [{key}].")


def main():
    parser = argparse.ArgumentParser(description="Download model weights for Matte App.")
    parser.add_argument("--models", type=str, help="Comma-separated list of model keys (e.g. sam2,matanyone,sam3)")
    parser.add_argument("--all", action="store_true", help="Download all supported models")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token")
    parser.add_argument("--sam2", action="store_true")
    parser.add_argument("--matanyone", action="store_true")
    parser.add_argument("--sam3", action="store_true")
    parser.add_argument("--videomama", action="store_true")
    parser.add_argument("--vitmatte", action="store_true")
    parser.add_argument("--sam2matting", action="store_true")
    parser.add_argument("--sam31", action="store_true")
    parser.add_argument("--matanyone2", action="store_true")
    parser.add_argument("--litesdmatte", action="store_true")
    args = parser.parse_args()

    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    to_download: list[str] = []

    if args.models:
        for item in args.models.split(","):
            item = item.strip().lower()
            if item and item not in to_download:
                to_download.append(item)
    elif args.all:
        to_download = [
            "sam2",
            "matanyone",
            "matanyone2",
            "sam2matting",
            "sam31",
            "sam3",
            "vitmatte",
            "videomama",
            "litesdmatte",
        ]
    else:
        has_specific_flag = (
            args.sam2
            or args.matanyone
            or args.sam3
            or args.videomama
            or args.vitmatte
            or args.sam2matting
            or args.sam31
            or args.matanyone2
            or args.litesdmatte
        )
        if not has_specific_flag:
            to_download = ["sam2", "matanyone"]
        else:
            to_download = ["sam2", "matanyone"]
            if args.sam2matting and "sam2matting" not in to_download:
                to_download.append("sam2matting")
            if args.matanyone2 and "matanyone2" not in to_download:
                to_download.append("matanyone2")
            if args.sam31 and "sam31" not in to_download:
                to_download.append("sam31")
            if args.sam3 and "sam3" not in to_download:
                to_download.append("sam3")
            if args.vitmatte and "vitmatte" not in to_download:
                to_download.append("vitmatte")
            if args.videomama and "videomama" not in to_download:
                to_download.append("videomama")
            if args.litesdmatte and "litesdmatte" not in to_download:
                to_download.append("litesdmatte")

    print(f"Matte App Model Downloader: target models = {to_download}")
    for idx, key in enumerate(to_download, 1):
        print(f"[{idx}/{len(to_download)}] Processing {key}...")
        download_model(key, models_dir, token=args.token)

    print("All requested downloads completed successfully.")


if __name__ == "__main__":
    main()
