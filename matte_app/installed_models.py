"""Model and pipeline installation discovery helpers for Matte App."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Pipeline definition: (Display Name, Required Component Keys)
# Matches the shell's original default order when models are installed.
PIPELINE_CATALOG: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SAM 2 + MatAnyone", ("sam2", "matanyone")),
    ("SAM 2 + MatAnyone 2", ("sam2", "matanyone2")),
    ("SAM 2 + ViTMatte", ("sam2", "vitmatte")),
    ("SAM 2 + VideoMaMa", ("sam2", "videomama")),
    ("SAM 3 + MatAnyone", ("sam3", "matanyone")),
    ("SAM 3 + MatAnyone 2", ("sam3", "matanyone2")),
    ("SAM 3 + ViTMatte", ("sam3", "vitmatte")),
    ("SAM 3 + VideoMaMa", ("sam3", "videomama")),
    ("BiRefNet HR — automatic matte", ("birefnet",)),
    ("SAM 3.1 — binary masks", ("sam31",)),
    ("SAM 3.1 + MatAnyone", ("sam31", "matanyone")),
    ("SAM 3.1 + MatAnyone 2", ("sam31", "matanyone2")),
    ("SAM 3.1 + ViTMatte", ("sam31", "vitmatte")),
    ("SAM 3.1 + VideoMaMa", ("sam31", "videomama")),
    ("SAM2Matting Tiny — soft matte", ("sam2matting",)),
    ("SAM2Matting Base+ — soft matte", ("sam2matting_base",)),
    ("LiteSDMatte — guided matte", ("litesdmatte",)),
)

SELECTOR_CATALOG: tuple[tuple[str, str], ...] = (
    ("SAM 2", "sam2"),
    ("SAM 3", "sam3"),
    ("BiRefNet HR", "birefnet"),
    ("SAM 3.1", "sam31"),
    ("SAM2Matting Tiny", "sam2matting"),
    ("SAM2Matting Base+", "sam2matting_base"),
    ("LiteSDMatte", "litesdmatte"),
)

REFINER_CATALOG: tuple[tuple[str, str | None], ...] = (
    ("MatAnyone", "matanyone"),
    ("MatAnyone 2", "matanyone2"),
    ("ViTMatte", "vitmatte"),
    ("VideoMaMa", "videomama"),
    ("None — keep base matte", None),
)


def get_installed_model_keys(root: Path | None = None) -> set[str]:
    """Detect which models and checkpoints are installed locally on disk."""
    if root is None:
        root = Path(__file__).resolve().parents[1]
    root = Path(root)

    installed: set[str] = {"birefnet"}  # BiRefNet dependencies are included with the base environment

    # Check each model on disk
    if (root / "models/SAM2/model.safetensors").is_file():
        installed.add("sam2")

    if (root / "models/SAM3/model.safetensors").is_file():
        installed.add("sam3")

    if (root / "models/SAM3.1/sam3.1_multiplex.pt").is_file() and (root / "third_party/sam3").is_dir():
        installed.add("sam31")

    if (root / "models/SAM2Matting/checkpoints/SAM2Matting-SAM2.1Tiny.pt").is_file() and (root / "third_party/SAM2Matting/sam2").is_dir():
        installed.add("sam2matting")

    if (root / "models/SAM2Matting/checkpoints/SAM2Matting-SAM2.1Base+.pt").is_file() and (root / "third_party/SAM2Matting/sam2").is_dir():
        installed.add("sam2matting_base")
        installed.add("sam2matting")

    if (root / "models/MatAnyone").is_dir() and any((root / "models/MatAnyone").iterdir()):
        installed.add("matanyone")

    if (root / "models/MatAnyone2/model.safetensors").is_file() and (root / "third_party/MatAnyone2/matanyone2/__init__.py").is_file():
        installed.add("matanyone2")

    if (root / "models/ViTMatte/model.safetensors").is_file() or (root / "models/ViTMatte/pytorch_model.bin").is_file():
        installed.add("vitmatte")

    if (root / "models/VideoMaMa/unet").is_dir() and (root / "models/SVD/vae").is_dir():
        installed.add("videomama")

    if (root / "models/LiteSDMatte/LiteSDMatte.pth").is_file():
        installed.add("litesdmatte")

    # Compatibility for mock / unit test environments where readiness is patched
    try:
        readiness = None
        app_win = sys.modules.get("matte_app.app_window")
        if app_win and hasattr(app_win, "readiness"):
            readiness = app_win.readiness
        if readiness is None:
            from .roto import readiness

        r_sam2 = readiness("SAM 2 — Tracking", root)
        if r_sam2.startswith("Ready"):
            installed.add("sam2")
        r_sam3 = readiness("SAM 3 — Tracking", root)
        if r_sam3.startswith("Ready"):
            installed.add("sam3")
        r_sam31 = readiness("SAM 3.1 — Tracking", root)
        if r_sam31.startswith("Ready"):
            installed.add("sam31")
        r_sam2m = readiness("SAM2Matting Tiny", root)
        if r_sam2m.startswith("Ready"):
            installed.add("sam2matting")
            installed.add("sam2matting_base")
        r_ma = readiness("MatAnyone — Saved masks", root)
        if r_ma.startswith("Ready"):
            installed.add("matanyone")
        r_ma2 = readiness("MatAnyone 2 — Saved masks", root)
        if r_ma2.startswith("Ready"):
            installed.add("matanyone2")
        r_vit = readiness("ViTMatte — Saved masks", root)
        if r_vit.startswith("Ready"):
            installed.add("vitmatte")
        r_vm = readiness("VideoMaMa — Saved masks", root)
        if r_vm.startswith("Ready"):
            installed.add("videomama")
    except Exception:
        pass

    return installed


def get_available_pipelines(root: Path | None = None) -> list[str]:
    """Return only pipelines whose required components are installed."""
    installed = get_installed_model_keys(root)
    valid = [name for name, reqs in PIPELINE_CATALOG if all(k in installed for k in reqs)]
    return valid if valid else ["BiRefNet HR — automatic matte"]


def get_available_selectors(root: Path | None = None) -> list[str]:
    """Return only selection models that are installed."""
    installed = get_installed_model_keys(root)
    valid = [name for name, k in SELECTOR_CATALOG if k in installed]
    return valid if valid else ["BiRefNet HR"]


def get_available_refiners(root: Path | None = None) -> list[str]:
    """Return only refiners that are installed."""
    installed = get_installed_model_keys(root)
    valid = [name for name, k in REFINER_CATALOG if k is None or k in installed]
    return valid if valid else ["None — keep base matte"]
