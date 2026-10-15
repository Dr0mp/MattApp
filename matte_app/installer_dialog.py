"""Model & Dependency Installer dialog for Matte App.

Enables in-app installation of AI models (SAM 2, MatAnyone, MatAnyone 2,
SAM2Matting, SAM 3, SAM 3.1, ViTMatte, VideoMaMa, LiteSDMatte),
Hugging Face token management for gated checkpoints, and runtime environment setup.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None
    get_token = lambda: None
    login = None


@dataclass(frozen=True)
class ModelDef:
    key: str
    name: str
    category: str
    size_str: str
    description: str
    is_gated: bool
    requires_venv: str  # "roto" | "sdmatte" | "none"
    relative_check_files: tuple[str, ...]
    third_party_repo: tuple[str, str, str] | None = None  # (name, git_url, git_commit)


ALL_MODELS: tuple[ModelDef, ...] = (
    ModelDef(
        key="sam2",
        name="SAM 2 (Segment Anything 2.1 Small)",
        category="Tracking",
        size_str="~176 MB",
        description="Interactive point and bounding-box tracking across video frames.",
        is_gated=False,
        requires_venv="roto",
        relative_check_files=("models/SAM2/model.safetensors",),
    ),
    ModelDef(
        key="matanyone",
        name="MatAnyone",
        category="Video Matting",
        size_str="~135 MB",
        description="High-consistency video matting guided by a starting-frame mask.",
        is_gated=False,
        requires_venv="roto",
        relative_check_files=("models/MatAnyone/model.safetensors",),
        third_party_repo=(
            "MatAnyone",
            "https://github.com/pq-yang/MatAnyone.git",
            "e5ddc534c1fff9bb9e54cf476095d29071b7cb4f",
        ),
    ),
    ModelDef(
        key="matanyone2",
        name="MatAnyone 2",
        category="Human Video Matting",
        size_str="~135 MB",
        description="Advanced human video matting with fine details; non-commercial S-Lab license.",
        is_gated=False,
        requires_venv="roto",
        relative_check_files=(
            "models/MatAnyone2/model.safetensors",
            "third_party/MatAnyone2/matanyone2/__init__.py",
        ),
        third_party_repo=(
            "MatAnyone2",
            "https://github.com/pq-yang/MatAnyone2.git",
            "0079197acd6d16a741f71558809c06c586c579e0",
        ),
    ),
    ModelDef(
        key="sam2matting",
        name="SAM2Matting (Tiny + Base+)",
        category="Direct Soft Matting",
        size_str="~571 MB",
        description="Direct soft-alpha matting pipeline from prompt clicks and boxes without hard thresholding.",
        is_gated=False,
        requires_venv="roto",
        relative_check_files=(
            "models/SAM2Matting/checkpoints/SAM2Matting-SAM2.1Tiny.pt",
            "third_party/SAM2Matting/sam2",
        ),
        third_party_repo=(
            "SAM2Matting",
            "https://github.com/FudanCVL/SAM2Matting.git",
            "73dd721d77b56749248aefe5e8824d7f61b9d13c",
        ),
    ),
    ModelDef(
        key="sam31",
        name="SAM 3.1 (Multiplex)",
        category="Tracking",
        size_str="~3.3 GB",
        description="Multiplex tracking supporting simultaneous points, boxes, and text concept prompts.",
        is_gated=False,
        requires_venv="roto",
        relative_check_files=(
            "models/SAM3.1/sam3.1_multiplex.pt",
            "third_party/sam3/sam3",
        ),
        third_party_repo=(
            "sam3",
            "https://github.com/facebookresearch/sam3.git",
            "0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc",
        ),
    ),
    ModelDef(
        key="sam3",
        name="SAM 3 (Gated Meta Checkpoint)",
        category="Tracking",
        size_str="~3.3 GB",
        description="Official SAM 3 checkpoint from Meta. Requires Hugging Face login with approved license terms.",
        is_gated=True,
        requires_venv="roto",
        relative_check_files=("models/SAM3/model.safetensors",),
    ),
    ModelDef(
        key="vitmatte",
        name="ViTMatte (Small Composition-1k)",
        category="Edge Refiner",
        size_str="~197 MB",
        description="Vision Transformer trimap edge refiner for sub-pixel boundary matting.",
        is_gated=False,
        requires_venv="roto",
        relative_check_files=("models/ViTMatte/model.safetensors",),
    ),
    ModelDef(
        key="videomama",
        name="VideoMaMa & Stable Video Diffusion",
        category="Temporal Diffusion",
        size_str="~7.2 GB",
        description="Temporal diffusion video matting across overlapping clip sections.",
        is_gated=False,
        requires_venv="roto",
        relative_check_files=(
            "models/VideoMaMa/unet",
            "models/SVD/model_index.json",
        ),
        third_party_repo=(
            "VideoMaMa",
            "https://github.com/cvlab-kaist/VideoMaMa.git",
            "d5cce3e0ffe3b6429c147e658bb28bcfb576374c",
        ),
    ),
    ModelDef(
        key="litesdmatte",
        name="LiteSDMatte",
        category="Guided Diffusion",
        size_str="~8.1 GB",
        description="Guided Stable Diffusion matting with interactive click and box constraints.",
        is_gated=False,
        requires_venv="sdmatte",
        relative_check_files=("models/LiteSDMatte/LiteSDMatte.pth",),
        third_party_repo=(
            "SDMatte",
            "https://github.com/vivoCameraResearch/SDMatte.git",
            "c3ea6949f270d253539a2bda2ad290e127934de9",
        ),
    ),
)


def is_model_installed(root: Path, model: ModelDef) -> bool:
    """Check whether all required files/directories for a model exist locally."""
    for rel_path in model.relative_check_files:
        path = root / rel_path
        if model.key == "vitmatte":
            alt = root / "models/ViTMatte/pytorch_model.bin"
            if not path.exists() and not alt.exists():
                return False
        else:
            if not path.exists():
                return False
    if model.third_party_repo:
        repo_name = model.third_party_repo[0]
        if not (root / "third_party" / repo_name).exists():
            return False
    return True


def check_hf_login() -> tuple[bool, str, str | None]:
    """Inspect current Hugging Face authentication status (in-memory only)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        return False, "No token set (gated models like SAM 3 require a token)", None
    if HfApi is None:
        return True, "Token set for current session", token
    try:
        info = HfApi().whoami(token=token)
        username = info.get("name") or info.get("fullname") or "user"
        return True, f"Authenticated for this session as: @{username}", token
    except Exception as err:
        return False, f"Token validation warning: {err}", token


def verify_hf_token(token: str) -> tuple[bool, str]:
    """Verify given token with Hugging Face in-memory without saving to disk."""
    cleaned = token.strip()
    if not cleaned:
        return False, "Token cannot be empty."
    if HfApi is None:
        os.environ["HF_TOKEN"] = cleaned
        return True, "Token verified for this session."
    try:
        info = HfApi().whoami(token=cleaned)
        username = info.get("name") or info.get("fullname") or "user"
        os.environ["HF_TOKEN"] = cleaned
        return True, f"Verified as @{username} (in-memory only, not saved to disk)."
    except Exception as err:
        return False, f"Authentication failed: {err}"


# Alias for compatibility with external references and tests
verify_and_save_hf_token = verify_hf_token


def detect_environments(root: Path) -> dict[str, bool]:
    """Detect presence of virtualenvs and required system command-line utilities."""
    has_uv = (shutil.which("uv") is not None) or ((root / "bin/uv.exe").is_file())
    return {
        "roto_venv": (root / ".roto-venv/Scripts/python.exe").is_file(),
        "sdmatte_venv": (root / ".sdmatte-venv/Scripts/python.exe").is_file(),
        "git": shutil.which("git") is not None,
        "uv": has_uv,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


class InstallerWorker(QThread):
    """Background worker for non-blocking downloads, git clones, and environment setup."""

    log_line = Signal(str)
    status_text = Signal(str)
    progress = Signal(int, int)  # current, total
    model_installed = Signal(str, bool)  # model_key, success
    finished = Signal(bool, str)  # all_success, summary_message

    def __init__(
        self,
        root: Path,
        models_to_download: list[str],
        setup_roto: bool,
        setup_sdmatte: bool,
        token: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.root = root
        self.models_to_download = list(models_to_download)
        self.setup_roto = setup_roto
        self.setup_sdmatte = setup_sdmatte
        self.token = token
        self._cancelled = False
        self._current_process: subprocess.Popen | None = None

    def cancel(self) -> None:
        """Signal the worker to abort and terminate active child processes."""
        self._cancelled = True
        if self._current_process is not None:
            try:
                self._current_process.terminate()
            except OSError:
                pass

    def _get_env(self) -> dict[str, str]:
        env = os.environ.copy()
        bin_dir = str(self.root / "bin")
        if (self.root / "bin").is_dir():
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        if self.token:
            env["HF_TOKEN"] = self.token
        return env

    def _run_subprocess(self, cmd: list[str], cwd: Path | None = None) -> bool:
        if self._cancelled:
            return False
        cmd_str = " ".join(cmd)
        self.log_line.emit(f"$ {cmd_str}")
        try:
            self._current_process = subprocess.Popen(
                cmd,
                cwd=str(cwd or self.root),
                env=self._get_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            if self._current_process.stdout:
                for line in iter(self._current_process.stdout.readline, ""):
                    if self._cancelled:
                        self._current_process.terminate()
                        break
                    self.log_line.emit(line.rstrip())
            self._current_process.wait()
            success = self._current_process.returncode == 0
            self._current_process = None
            return success
        except Exception as err:
            self.log_line.emit(f"Process execution error: {err}")
            self._current_process = None
            return False

    def run(self) -> None:
        total_steps = (1 if self.setup_roto else 0) + (1 if self.setup_sdmatte else 0) + len(self.models_to_download)
        current_step = 0

        self.log_line.emit("==================================================")
        self.log_line.emit("Starting Matte App Installation Task")
        self.log_line.emit(f"Project root: {self.root}")
        self.log_line.emit(f"Selected models: {self.models_to_download}")
        self.log_line.emit(f"Setup Roto: {self.setup_roto} | Setup SDMatte: {self.setup_sdmatte}")
        self.log_line.emit("==================================================")

        # 1. Roto environment setup
        if self.setup_roto:
            if self._cancelled:
                self.finished.emit(False, "Installation cancelled.")
                return
            current_step += 1
            self.progress.emit(current_step, total_steps)
            self.status_text.emit("Setting up Roto environment (.roto-venv)...")
            self.log_line.emit("\n--- [Step: Setup .roto-venv] ---")
            ps1 = self.root / "scripts/setup_roto.ps1"
            if not ps1.is_file():
                ps1 = self.root / "setup_roto.ps1"
            if ps1.is_file():
                flags = []
                if "matanyone2" in self.models_to_download:
                    flags.append("-MatAnyone2")
                if "sam2matting" in self.models_to_download:
                    flags.append("-Sam2Matting")
                if "sam31" in self.models_to_download:
                    flags.append("-Sam31")
                if "sam3" in self.models_to_download:
                    flags.append("-Sam3")
                if "vitmatte" in self.models_to_download:
                    flags.append("-ViTMatte")
                if "videomama" in self.models_to_download:
                    flags.append("-VideoMaMa")
                ok = self._run_subprocess(
                    ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1), *flags]
                )
                if not ok and not self._cancelled:
                    self.log_line.emit("Warning: setup_roto.ps1 exited with errors.")
            else:
                self.log_line.emit("Warning: setup_roto.ps1 not found.")

        # 2. SDMatte environment setup
        if self.setup_sdmatte:
            if self._cancelled:
                self.finished.emit(False, "Installation cancelled.")
                return
            current_step += 1
            self.progress.emit(current_step, total_steps)
            self.status_text.emit("Setting up LiteSDMatte environment (.sdmatte-venv)...")
            self.log_line.emit("\n--- [Step: Setup .sdmatte-venv] ---")
            ps1 = self.root / "scripts/setup_litesdmatte.ps1"
            if not ps1.is_file():
                ps1 = self.root / "setup_litesdmatte.ps1"
            if ps1.is_file():
                self._run_subprocess(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)])
            else:
                self.log_line.emit("Warning: setup_litesdmatte.ps1 not found.")

        # 3. Model Downloads & Third-party Clones
        model_map = {m.key: m for m in ALL_MODELS}
        python_exe = str(self.root / ".roto-venv/Scripts/python.exe")
        if not Path(python_exe).is_file():
            python_exe = sys.executable

        for key in self.models_to_download:
            if self._cancelled:
                self.finished.emit(False, "Installation cancelled.")
                return
            current_step += 1
            self.progress.emit(current_step, total_steps)
            m_def = model_map.get(key)
            name = m_def.name if m_def else key
            self.status_text.emit(f"Installing {name}...")
            self.log_line.emit(f"\n--- [Model: {name} ({key})] ---")

            # Clone third_party git repo if required and missing
            if m_def and m_def.third_party_repo:
                repo_name, git_url, git_commit = m_def.third_party_repo
                repo_path = self.root / "third_party" / repo_name
                if not repo_path.exists():
                    self.log_line.emit(f"Cloning {repo_name} from {git_url}...")
                    self.root.joinpath("third_party").mkdir(parents=True, exist_ok=True)
                    if self._run_subprocess(["git", "clone", git_url, str(repo_path)]):
                        self._run_subprocess(["git", "-C", str(repo_path), "checkout", git_commit])
                    else:
                        self.log_line.emit(f"Warning: git clone failed for {repo_name}.")

            # Download weights via download_roto.py
            dl_script = self.root / "matte_app/download_roto.py"
            cmd = [python_exe, "-u", str(dl_script), "--models", key]
            if self.token:
                cmd.extend(["--token", self.token])
            success = self._run_subprocess(cmd)

            # Check if model is now ready
            ready = is_model_installed(self.root, m_def) if m_def else success
            self.model_installed.emit(key, ready)
            if ready:
                self.log_line.emit(f"✓ Model [{key}] successfully installed and verified.")
            else:
                self.log_line.emit(f"✗ Model [{key}] verification failed or files are incomplete.")

        if self._cancelled:
            self.finished.emit(False, "Installation cancelled by user.")
        else:
            self.progress.emit(total_steps, total_steps)
            self.status_text.emit("All tasks completed.")
            self.finished.emit(True, "Installation tasks completed.")


class ModelCardWidget(QFrame):
    """Interactive card representing an AI model with checkboxes, badges, and status."""

    def __init__(self, model: ModelDef, installed: bool, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = model
        self.setObjectName("modelCard")
        self.setStyleSheet(
            """
            QFrame#modelCard {
                background: #fffdf7;
                border: 1px solid #c1c1b6;
                border-radius: 8px;
                padding: 4px;
            }
            QFrame#modelCard:hover {
                background: #faf6ea;
                border-color: #262828;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Header row: Checkbox, Name, Badges, Status
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        self.checkbox = QCheckBox(model.name)
        self.checkbox.setStyleSheet("font-weight: 700; font-size: 13px; color: #262828;")
        top_row.addWidget(self.checkbox)

        # Category badge
        cat_badge = QLabel(model.category)
        cat_badge.setStyleSheet(
            "background: #e4ddfc; color: #262828; border: 1px solid #262828; border-radius: 4px; padding: 2px 6px; font-size: 10px; font-weight: 600;"
        )
        top_row.addWidget(cat_badge)

        # Size badge
        size_badge = QLabel(model.size_str)
        size_badge.setStyleSheet(
            "background: #efede3; color: #5a5d55; border: 1px solid #b7b9ae; border-radius: 4px; padding: 2px 6px; font-size: 10px;"
        )
        top_row.addWidget(size_badge)

        # Gated warning tag
        if model.is_gated:
            gated_badge = QLabel("HF TOKEN REQUIRED")
            gated_badge.setStyleSheet(
                "background: #f6d85c; color: #262828; border: 1px solid #262828; border-radius: 4px; padding: 2px 6px; font-size: 10px; font-weight: 700;"
            )
            top_row.addWidget(gated_badge)

        top_row.addStretch()

        # Status badge
        self.status_label = QLabel()
        self.set_installed(installed)
        top_row.addWidget(self.status_label)

        layout.addLayout(top_row)

        # Description label
        desc_label = QLabel(model.description)
        desc_label.setStyleSheet("color: #676b61; font-size: 11px; margin-left: 20px;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

    def set_installed(self, installed: bool) -> None:
        """Update visual badge indicating installation status."""
        if installed:
            self.status_label.setText("● Installed")
            self.status_label.setStyleSheet(
                "background: #dcf5e7; color: #196c38; border: 1px solid #196c38; border-radius: 5px; padding: 2px 8px; font-size: 11px; font-weight: 700;"
            )
        else:
            self.status_label.setText("○ Not Downloaded")
            self.status_label.setStyleSheet(
                "background: #fdf5d6; color: #7d6006; border: 1px solid #b79524; border-radius: 5px; padding: 2px 8px; font-size: 11px; font-weight: 600;"
            )

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def set_checked(self, val: bool) -> None:
        self.checkbox.setChecked(val)


class ModelInstallerDialog(QDialog):
    """Full-featured model, dependency, and Hugging Face installer window."""

    models_updated = Signal()

    def __init__(self, parent: QWidget | None = None, focus_model: str | None = None) -> None:
        super().__init__(parent)
        self.root = Path(__file__).resolve().parents[1]
        self.worker: InstallerWorker | None = None
        self.model_cards: dict[str, ModelCardWidget] = {}

        self.setWindowTitle("Matte App — Model & Runtime Installer")
        self.resize(880, 720)
        self.setMinimumSize(780, 580)
        self._setup_style()
        self._init_ui()
        self.refresh_statuses()

        if focus_model:
            self.select_focus_model(focus_model)

    def _setup_style(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background: #efecdf;
                color: #262828;
                font-family: "Segoe UI", sans-serif;
                font-size: 12px;
            }
            QFrame#panelSection {
                background: #fffdf7;
                border: 1px solid #262828;
                border-radius: 10px;
                padding: 10px;
            }
            QLabel#sectionHeader {
                font-size: 13px;
                font-weight: 800;
                color: #262828;
                letter-spacing: 0.5px;
            }
            QLabel#dialogTitle {
                font-size: 18px;
                font-weight: 800;
                color: #262828;
            }
            QLabel#dialogSubtitle {
                color: #676b61;
                font-size: 12px;
            }
            QPushButton {
                background: #fffdf7;
                border: 1px solid #262828;
                border-bottom: 3px solid #262828;
                border-radius: 7px;
                padding: 6px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f6f0d9;
            }
            QPushButton:pressed {
                border-bottom: 1px solid #262828;
                background: #e7dfbf;
            }
            QPushButton#primaryBtn {
                background: #f6d85c;
                font-weight: 700;
            }
            QPushButton#primaryBtn:disabled {
                background: #f2e9ba;
                color: #96917c;
                border-color: #c4c5bb;
            }
            QPushButton#stopBtn {
                background: #f5c6c5;
            }
            QPushButton#stopBtn:disabled {
                background: #f2eeea;
                color: #a69c94;
                border-color: #c4c5bb;
            }
            QLineEdit {
                background: #fffdf7;
                border: 1px solid #b7b9ae;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 2px solid #262828;
            }
            QPlainTextEdit {
                background: #17191d;
                color: #e0e4df;
                border: 1px solid #262828;
                border-radius: 6px;
                padding: 6px;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
            QProgressBar {
                border: 1px solid #a5a89b;
                border-radius: 4px;
                background: #efede3;
                text-align: center;
                height: 18px;
            }
            QProgressBar::chunk {
                background: #bda9ef;
                border-radius: 3px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
            """
        )

    def _init_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        # 1. Dialog Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(2)
        title = QLabel("Model & Runtime Installer")
        title.setObjectName("dialogTitle")
        header_layout.addWidget(title)
        subtitle = QLabel(
            "Download AI models, verify Hugging Face authentication, and configure runtimes for video roto & matting."
        )
        subtitle.setObjectName("dialogSubtitle")
        header_layout.addWidget(subtitle)
        outer.addLayout(header_layout)

        # 2. Hugging Face Authentication Section
        hf_panel = QFrame()
        hf_panel.setObjectName("panelSection")
        hf_layout = QVBoxLayout(hf_panel)
        hf_layout.setContentsMargins(12, 10, 12, 10)
        hf_layout.setSpacing(6)

        hf_top = QHBoxLayout()
        hf_title = QLabel("🔑 Hugging Face Authentication")
        hf_title.setObjectName("sectionHeader")
        hf_top.addWidget(hf_title)
        hf_top.addStretch()

        self.hf_status_pill = QLabel()
        self.hf_status_pill.setStyleSheet(
            "border: 1px solid #b7b9ae; border-radius: 10px; padding: 3px 10px; font-size: 11px;"
        )
        hf_top.addWidget(self.hf_status_pill)
        hf_layout.addLayout(hf_top)

        hf_desc = QLabel(
            "Required for gated models like <b>SAM 3</b> (<code>facebook/sam3</code>). "
            "Public models (SAM 2, MatAnyone, ViTMatte, etc.) do not require login."
        )
        hf_desc.setStyleSheet("color: #676b61; font-size: 11px;")
        hf_desc.setWordWrap(True)
        hf_layout.addWidget(hf_desc)

        # Token row: Input, Show/Hide, Verify, Clear, Get Token
        token_row = QHBoxLayout()
        token_row.setSpacing(6)

        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("Paste Hugging Face User Access Token (starts with hf_...)")
        self.token_input.setEchoMode(QLineEdit.Password)
        token_row.addWidget(self.token_input, 1)

        self.toggle_eye_btn = QPushButton("Show")
        self.toggle_eye_btn.setFixedWidth(56)
        self.toggle_eye_btn.clicked.connect(self._toggle_token_visibility)
        token_row.addWidget(self.toggle_eye_btn)

        self.verify_save_btn = QPushButton("Verify Token")
        self.verify_save_btn.setToolTip("Verifies your token in-memory for this session. It is never saved to disk.")
        self.verify_save_btn.clicked.connect(self._save_token_clicked)
        token_row.addWidget(self.verify_save_btn)

        self.clear_token_btn = QPushButton("Clear")
        self.clear_token_btn.setToolTip("Clear the in-memory token")
        self.clear_token_btn.clicked.connect(self._clear_token_clicked)
        token_row.addWidget(self.clear_token_btn)

        self.get_token_btn = QPushButton("Get Token Online ↗")
        self.get_token_btn.setToolTip("Opens https://huggingface.co/settings/tokens in your browser")
        self.get_token_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl("https://huggingface.co/settings/tokens"))
        )
        token_row.addWidget(self.get_token_btn)

        hf_layout.addLayout(token_row)

        privacy_note = QLabel("🔒 Privacy guarantee: Your token is held in-memory only for downloading gated models and is NEVER saved to disk or repository.")
        privacy_note.setObjectName("sectionSubtitle")
        privacy_note.setStyleSheet("font-size: 11px; color: #5a5f5f; font-style: italic;")
        hf_layout.addWidget(privacy_note)
        outer.addWidget(hf_panel)

        # 3. Environments & System Tools Section
        env_panel = QFrame()
        env_panel.setObjectName("panelSection")
        env_layout = QVBoxLayout(env_panel)
        env_layout.setContentsMargins(12, 10, 12, 10)
        env_layout.setSpacing(6)

        env_top = QHBoxLayout()
        env_title = QLabel("⚙️ Python Runtimes & Dependencies")
        env_title.setObjectName("sectionHeader")
        env_top.addWidget(env_title)
        env_top.addStretch()

        self.tools_label = QLabel()
        self.tools_label.setStyleSheet("color: #676b61; font-size: 11px;")
        env_top.addWidget(self.tools_label)
        env_layout.addLayout(env_top)

        # Runtime checkboxes / status row
        runtimes_row = QHBoxLayout()
        runtimes_row.setSpacing(16)

        self.roto_venv_box = QCheckBox("Set up / rebuild Roto runtime (.roto-venv)")
        self.roto_status_lbl = QLabel()
        runtimes_row.addWidget(self.roto_venv_box)
        runtimes_row.addWidget(self.roto_status_lbl)

        runtimes_row.addSpacing(20)

        self.sdmatte_venv_box = QCheckBox("Set up LiteSDMatte runtime (.sdmatte-venv)")
        self.sdmatte_status_lbl = QLabel()
        runtimes_row.addWidget(self.sdmatte_venv_box)
        runtimes_row.addWidget(self.sdmatte_status_lbl)

        runtimes_row.addStretch()
        env_layout.addLayout(runtimes_row)
        outer.addWidget(env_panel)

        # 4. Model Catalog Section
        catalog_panel = QFrame()
        catalog_panel.setObjectName("panelSection")
        cat_layout = QVBoxLayout(catalog_panel)
        cat_layout.setContentsMargins(12, 10, 12, 10)
        cat_layout.setSpacing(8)

        cat_top = QHBoxLayout()
        cat_title = QLabel("📦 AI Models Catalog")
        cat_title.setObjectName("sectionHeader")
        cat_top.addWidget(cat_title)
        cat_top.addStretch()

        # Preset selection buttons
        btn_missing = QPushButton("Select Missing")
        btn_missing.clicked.connect(self.select_missing_models)
        cat_top.addWidget(btn_missing)

        btn_core = QPushButton("Core Essentials")
        btn_core.setToolTip("Select SAM 2, MatAnyone, and ViTMatte (Fastest ~500 MB core)")
        btn_core.clicked.connect(self.select_core_models)
        cat_top.addWidget(btn_core)

        btn_all = QPushButton("Select All")
        btn_all.clicked.connect(lambda: self._set_all_models_checked(True))
        cat_top.addWidget(btn_all)

        btn_clear = QPushButton("Clear Selection")
        btn_clear.clicked.connect(lambda: self._set_all_models_checked(False))
        cat_top.addWidget(btn_clear)

        cat_layout.addLayout(cat_top)

        # Scroll Area for Model Cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 4, 0)
        scroll_layout.setSpacing(6)

        for model in ALL_MODELS:
            card = ModelCardWidget(model, installed=is_model_installed(self.root, model))
            self.model_cards[model.key] = card
            scroll_layout.addWidget(card)

        scroll.setWidget(scroll_content)
        cat_layout.addWidget(scroll, 1)
        outer.addWidget(catalog_panel, 2)

        # 5. Live Console & Progress Bar
        progress_panel = QFrame()
        progress_panel.setObjectName("panelSection")
        prog_layout = QVBoxLayout(progress_panel)
        prog_layout.setContentsMargins(10, 8, 10, 8)
        prog_layout.setSpacing(6)

        prog_top = QHBoxLayout()
        self.status_msg = QLabel("Ready.")
        self.status_msg.setStyleSheet("font-weight: 600; color: #262828;")
        prog_top.addWidget(self.status_msg)
        prog_top.addStretch()

        clear_log_btn = QPushButton("Clear Console")
        clear_log_btn.clicked.connect(self._clear_console)
        prog_top.addWidget(clear_log_btn)
        prog_layout.addLayout(prog_top)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.progress_bar)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(140)
        prog_layout.addWidget(self.console)

        outer.addWidget(progress_panel)

        # 6. Bottom Action Footer
        footer = QHBoxLayout()
        footer.setSpacing(8)

        self.install_btn = QPushButton("Start Installation")
        self.install_btn.setObjectName("primaryBtn")
        self.install_btn.clicked.connect(self.start_installation)
        footer.addWidget(self.install_btn)

        self.cancel_btn = QPushButton("Stop / Cancel")
        self.cancel_btn.setObjectName("stopBtn")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_installation)
        footer.addWidget(self.cancel_btn)

        footer.addStretch()

        refresh_btn = QPushButton("Refresh Status")
        refresh_btn.clicked.connect(self.refresh_statuses)
        footer.addWidget(refresh_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        footer.addWidget(close_btn)

        outer.addLayout(footer)

    def _toggle_token_visibility(self) -> None:
        if self.token_input.echoMode() == QLineEdit.Password:
            self.token_input.setEchoMode(QLineEdit.Normal)
            self.toggle_eye_btn.setText("Hide")
        else:
            self.token_input.setEchoMode(QLineEdit.Password)
            self.toggle_eye_btn.setText("Show")

    def _clear_token_clicked(self) -> None:
        self.token_input.clear()
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]
        self.refresh_hf_status()

    def _save_token_clicked(self) -> None:
        token = self.token_input.text().strip()
        if not token:
            QMessageBox.warning(self, "Token Required", "Please paste your Hugging Face User Access Token first.")
            return
        self.verify_save_btn.setEnabled(False)
        self.verify_save_btn.setText("Verifying...")
        QApplication.processEvents()
        ok, msg = verify_hf_token(token)
        self.verify_save_btn.setEnabled(True)
        self.verify_save_btn.setText("Verify Token")
        if ok:
            QMessageBox.information(self, "Authentication Verified", msg)
            self.refresh_hf_status()
        else:
            QMessageBox.warning(self, "Authentication Failed", msg)

    def refresh_hf_status(self) -> None:
        ok, label, token = check_hf_login()
        if ok:
            self.hf_status_pill.setText(f"● {label}")
            self.hf_status_pill.setStyleSheet(
                "background: #dcf5e7; color: #196c38; border: 1px solid #196c38; border-radius: 10px; padding: 3px 10px; font-weight: 700; font-size: 11px;"
            )
        else:
            self.hf_status_pill.setText(f"○ {label}")
            self.hf_status_pill.setStyleSheet(
                "background: #fdf5d6; color: #7d6006; border: 1px solid #b79524; border-radius: 10px; padding: 3px 10px; font-weight: 600; font-size: 11px;"
            )

    def refresh_statuses(self) -> None:
        """Scan local filesystem and update all badges."""
        self.refresh_hf_status()

        # Update environment checks
        envs = detect_environments(self.root)
        if envs["roto_venv"]:
            self.roto_status_lbl.setText("● Ready (Python 3.12 CUDA)")
            self.roto_status_lbl.setStyleSheet("color: #196c38; font-weight: 600;")
        else:
            self.roto_status_lbl.setText("○ Missing (.roto-venv)")
            self.roto_status_lbl.setStyleSheet("color: #7d6006; font-weight: 600;")
            self.roto_venv_box.setChecked(True)

        if envs["sdmatte_venv"]:
            self.sdmatte_status_lbl.setText("● Ready (Python 3.10)")
            self.sdmatte_status_lbl.setStyleSheet("color: #196c38; font-weight: 600;")
        else:
            self.sdmatte_status_lbl.setText("○ Missing (.sdmatte-venv)")
            self.sdmatte_status_lbl.setStyleSheet("color: #7d6006; font-weight: 600;")

        tools_str = f"Tools: Git {'✓' if envs['git'] else '✗'} | uv {'✓' if envs['uv'] else '✗'} | FFmpeg {'✓' if envs['ffmpeg'] else '✗'}"
        self.tools_label.setText(tools_str)

        # Update model card badges
        for model in ALL_MODELS:
            card = self.model_cards.get(model.key)
            if card:
                card.set_installed(is_model_installed(self.root, model))

    def select_focus_model(self, model_key: str) -> None:
        """Focus and check a specific model (e.g. SAM 3)."""
        card = self.model_cards.get(model_key)
        if card:
            card.set_checked(True)
            if model_key == "sam3":
                ok, _, _ = check_hf_login()
                if not ok:
                    self.status_msg.setText("Notice: SAM 3 requires Hugging Face authentication token above.")

    def select_missing_models(self) -> None:
        """Select all models that are currently not installed."""
        for model in ALL_MODELS:
            card = self.model_cards.get(model.key)
            if card:
                card.set_checked(not is_model_installed(self.root, model))

    def select_core_models(self) -> None:
        """Select core lightweight models: SAM 2, MatAnyone, and ViTMatte."""
        self._set_all_models_checked(False)
        for key in ("sam2", "matanyone", "vitmatte"):
            card = self.model_cards.get(key)
            if card:
                card.set_checked(True)

    def _set_all_models_checked(self, val: bool) -> None:
        for card in self.model_cards.values():
            card.set_checked(val)

    def _clear_console(self) -> None:
        self.console.clear()

    def start_installation(self) -> None:
        selected_models = [k for k, card in self.model_cards.items() if card.is_checked()]
        setup_roto = self.roto_venv_box.isChecked()
        setup_sdmatte = self.sdmatte_venv_box.isChecked()

        if not selected_models and not setup_roto and not setup_sdmatte:
            QMessageBox.information(
                self,
                "Nothing Selected",
                "Please select at least one model or runtime environment to install.",
            )
            return

        # Check SAM 3 gating warning
        if "sam3" in selected_models:
            ok, _, token = check_hf_login()
            entered = self.token_input.text().strip()
            if not ok and not entered:
                reply = QMessageBox.question(
                    self,
                    "Hugging Face Token Recommended",
                    "SAM 3 is a gated model requiring Hugging Face token authentication.\n"
                    "You have not provided a token above. The download may fail if your Hugging Face account has not accepted the license terms.\n\n"
                    "Do you want to proceed anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    return

        token_to_use = self.token_input.text().strip() or os.environ.get("HF_TOKEN")

        self.install_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_msg.setText("Starting installation...")

        self.worker = InstallerWorker(
            root=self.root,
            models_to_download=selected_models,
            setup_roto=setup_roto,
            setup_sdmatte=setup_sdmatte,
            token=token_to_use,
            parent=self,
        )
        self.worker.log_line.connect(self._on_log_line)
        self.worker.status_text.connect(self._on_status_text)
        self.worker.progress.connect(self._on_progress)
        self.worker.model_installed.connect(self._on_model_installed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def cancel_installation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.cancel_btn.setEnabled(False)
            self.status_msg.setText("Cancelling installation...")
            self.worker.cancel()

    def _on_log_line(self, line: str) -> None:
        self.console.appendPlainText(line)

    def _on_status_text(self, text: str) -> None:
        self.status_msg.setText(text)

    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            percent = int((current / total) * 100)
            self.progress_bar.setValue(percent)

    def _on_model_installed(self, model_key: str, ready: bool) -> None:
        card = self.model_cards.get(model_key)
        if card:
            card.set_installed(ready)

    def _on_worker_finished(self, success: bool, message: str) -> None:
        self.install_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_msg.setText(message)
        self.refresh_statuses()
        self.models_updated.emit()
        if hasattr(self.parent(), "refresh_installed_pipelines"):
            try:
                self.parent().refresh_installed_pipelines()
            except Exception:
                pass
        if success:
            QMessageBox.information(self, "Installation Complete", message)
        else:
            QMessageBox.warning(self, "Installation Incomplete", message)
        self.worker = None

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Installation In Progress",
                "An installation task is currently running. Do you want to cancel it and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.worker.cancel()
                self.worker.wait(2000)
                event.accept()
            else:
                event.ignore()
                return
        event.accept()
