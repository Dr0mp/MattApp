from __future__ import annotations

import json
import uuid
import queue
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Qt, Signal, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QSizePolicy,
    QButtonGroup,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QSplitter,
    QSpinBox,
    QSlider,
    QScrollArea,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .cancellation import Cancellation, JobCancelled
from .canvas import ImageCanvas
from .flow_layout import FlowLayout
from .models import BoxHint, Guidance, PointHint, RotoHint, make_model
from .roto import ROTO_MODELS, REFINERS, capabilities, readiness
from .workflow import WorkflowMixin
from .guidance_state import GuidanceMixin
from .guidance_store import source_key, encode
from .video import VideoInfo, load_frame, probe_video, process_video, read_metadata, save_frame, write_metadata
from .matte_editor import MatteEditor

def toolbar(layout: QHBoxLayout) -> QScrollArea:
    content = QWidget()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(12)
    layout.setSizeConstraint(QLayout.SetMinimumSize)
    content.setLayout(layout)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll.setMinimumWidth(0)
    scroll.setFixedHeight(70)
    scroll.setWidget(content)
    scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; border: none; }")
    return scroll



VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def checkerboard(size: tuple[int, int]) -> Image.Image:
    # A repeating tile avoids full-size int64 NumPy coordinate arrays.
    from PIL import ImageDraw
    cell=24;tile=Image.new('RGB',(cell*2,cell*2),(35,38,44))
    draw=ImageDraw.Draw(tile)
    draw.rectangle((cell,0,cell*2-1,cell-1),fill=(55,59,67))
    draw.rectangle((0,cell,cell-1,cell*2-1),fill=(55,59,67))
    width,height=size
    # Doubling fills the image in logarithmically many PIL copies.
    result=Image.new('RGB',size);result.paste(tile,(0,0))
    filled_w=min(width,cell*2);filled_h=min(height,cell*2)
    while filled_w<width:
        amount=min(filled_w,width-filled_w);result.paste(result.crop((0,0,amount,filled_h)),(filled_w,0));filled_w+=amount
    while filled_h<height:
        amount=min(filled_h,height-filled_h);result.paste(result.crop((0,0,width,amount)),(0,filled_h));filled_h+=amount
    return result


class CancellableWorker(QThread):
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.cancellation = Cancellation()
        # Keep only the newest committed frame, never a backlog of large images.
        self.live_frames = queue.Queue(maxsize=1)

    def publish_frame(self, preview) -> None:
        try:
            self.live_frames.get_nowait()
        except queue.Empty:
            pass
        self.live_frames.put_nowait(preview)

    def cancel(self) -> None:
        self.cancellation.cancel()

    def run(self) -> None:
        try:
            self.cancellation.check()
            self.execute()
        except JobCancelled:
            self.cancelled.emit()
        except Exception:
            if self.cancellation.requested:
                self.cancelled.emit()
            else:
                self.failed.emit(traceback.format_exc())


class InferenceWorker(CancellableWorker):
    completed = Signal(object)

    def __init__(self, model_name: str, image: Image.Image, guidance: list[Guidance]) -> None:
        super().__init__()
        self.model_name, self.image, self.guidance = model_name, image.copy(), guidance

    def execute(self) -> None:
        alpha = make_model(self.model_name).predict(self.image, self.guidance, self.cancellation)
        self.cancellation.check()
        self.completed.emit(alpha)


@dataclass
class MediaPreview:
    path: str
    original: Image.Image
    alpha: Image.Image | None
    frame: int
    info: VideoInfo | None
    output_dir: str | None
    foreground: Image.Image | None = None


class PreviewWorker(CancellableWorker):
    completed = Signal(object)

    def __init__(self, path: str, frame: int = 1, output_dir: str | None = None, info: VideoInfo | None = None, cutout_dir: str | None = None) -> None:
        super().__init__()
        self.path, self.frame, self.output_dir, self.info = path, frame, output_dir, info
        self.cutout_dir=cutout_dir

    def execute(self) -> None:
        folder = Path(self.output_dir) if self.output_dir else None
        saved = folder / f"frame_{self.frame:06d}.png" if folder else None
        alpha = None
        if Path(self.path).suffix.lower() in VIDEO_SUFFIXES:
            metadata = read_metadata(folder)
            info = self.info
            if info is None and metadata.get("frame_count"):
                info = VideoInfo(int(metadata["frame_count"]), float(metadata.get("fps", 0)))
                if not info.fps and Path(self.path).exists():
                    info = probe_video(self.path, self.cancellation, info.frame_count)
            info = info or probe_video(self.path, self.cancellation)
            if not 1 <= self.frame <= info.frame_count:
                raise ValueError("Frame number is outside this video.")
            if saved and saved.exists():
                with Image.open(saved) as image:
                    original = image.convert("RGBA")
                alpha = original.getchannel("A")
                original.putalpha(255)
            else:
                original = load_frame(self.path, self.frame, self.cancellation)
        else:
            info = None
            with Image.open(self.path) as image:
                original = image.convert("RGBA")
            if saved and saved.exists():
                with Image.open(saved) as image:
                    alpha = image.convert("RGBA").getchannel("A")
                if alpha.size != original.size:
                    raise RuntimeError("Saved matte dimensions do not match the source image.")
        self.cancellation.check()
        foreground=None
        if self.cutout_dir:
            meta=read_metadata(Path(self.cutout_dir));corrected=meta.get('foreground_sources',{}).get(str(self.frame),meta.get('foreground_estimated',False))
            path=Path(self.cutout_dir)/f'frame_{self.frame:06d}.png'
            if corrected and path.is_file():
                with Image.open(path) as saved_cutout:
                    if saved_cutout.size==original.size:foreground=saved_cutout.convert('RGBA')
        self.cancellation.check()
        self.completed.emit(MediaPreview(self.path, original, alpha, self.frame, info, self.output_dir,foreground))


class VideoInferenceWorker(CancellableWorker):
    progress = Signal(int, int)
    completed = Signal(str, int)

    def __init__(self, video_path: str, output_dir: str, model_name: str, guidance: list[Guidance], start_frame: int = 1, info: VideoInfo | None = None, end_frame: int | None = None) -> None:
        super().__init__()
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.model_name, self.guidance, self.start_frame = model_name, guidance, start_frame
        self.info = info
        self.end_frame = end_frame

    def _frame_ready(self, frame, total, image):
        info = VideoInfo(total, self.info.fps if self.info else 0)
        self.publish_frame(MediaPreview(self.video_path, image, None, frame, info, str(self.output_dir)))

    def execute(self) -> None:
        total = process_video(self.video_path, self.output_dir, make_model(self.model_name), self.model_name,
                              self.guidance, self.cancellation, self.start_frame, self.progress.emit, self._frame_ready, self.end_frame)
        self.cancellation.check()
        self.completed.emit(str(self.output_dir), total)


class QueueWorker(CancellableWorker):
    item_output = Signal(str, str)
    progress = Signal(int, int, str)
    completed = Signal(str, int)

    def __init__(self, paths: list[str], output_root: str, model_name: str, guidance: list[Guidance], guidance_by_path: dict | None = None) -> None:
        super().__init__()
        self.paths, self.output_root = paths, Path(output_root)
        self.outputs: dict[str, str] = {}
        self.model_name, self.guidance = model_name, guidance
        self.guidance_by_path = None if guidance_by_path is None else {source_key(path): list(hints) for path,hints in guidance_by_path.items()}
        self._item_guidance = list(guidance)

    @staticmethod
    def _available_path(path: Path, folder: bool = False) -> Path:
        candidate, suffix = path, 2
        while candidate.exists():
            candidate = path.with_name(f"{path.stem}_{suffix}" + ("" if folder else path.suffix))
            suffix += 1
        return candidate

    def _process_video(self, source_path: Path, destination: Path, model) -> int:
        info = None
        def ready(frame, total, image):
            nonlocal info
            self.publish_frame(MediaPreview(str(source_path), image, None, frame, info or VideoInfo(total, 0), str(destination)))
            if info is None:
                info = probe_video(str(source_path), self.cancellation, total)
                self.publish_frame(MediaPreview(str(source_path), image, None, frame, info, str(destination)))
        return process_video(str(source_path), destination, model, self.model_name, self._item_guidance, self.cancellation,
                             frame_ready=ready)

    def execute(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cancellation.check()
        model = make_model(self.model_name)
        for index, raw_path in enumerate(self.paths, 1):
            self.cancellation.check()
            self._item_guidance = list(self.guidance if self.guidance_by_path is None else self.guidance_by_path.get(source_key(raw_path), []))
            source_path = Path(raw_path)
            # `stem` retains the user's filename (including spaces) but avoids
            # carrying a video/image extension into the output folder name.
            destination = self._available_path(self.output_root / source_path.stem, folder=True)
            self.outputs[raw_path] = str(destination)
            self.item_output.emit(raw_path, str(destination))
            if source_path.suffix.lower() in VIDEO_SUFFIXES:
                self._process_video(source_path, destination, model)
            else:
                destination.mkdir(parents=True)
                source = Image.open(source_path).convert("RGBA")
                source.putalpha(model.predict(source, self._item_guidance, self.cancellation))
                self.cancellation.check()
                save_frame(source, destination / "frame_000001.png")
                write_metadata(destination, {"source_image": raw_path, "frame_count": 1, "model": self.model_name,
                    "guidance": encode(self._item_guidance), "status": "complete"})
                self.publish_frame(MediaPreview(raw_path, source, None, 1, None, str(destination)))
            self.cancellation.check()
            self.progress.emit(index, len(self.paths), source_path.name)
        self.cancellation.check()
        (self.output_root / "queue.json").write_text(json.dumps({
            "item_count": len(self.paths), "model": self.model_name,
            "guidance": [hint.__dict__ | {"type": type(hint).__name__} for hint in self.guidance],
            "guidance_by_source": {path: encode(self.guidance if self.guidance_by_path is None else self.guidance_by_path.get(source_key(path), [])) for path in self.paths},
            "sources": self.paths,
            "outputs": self.outputs,
        }, indent=2), encoding="utf-8")
        self.cancellation.check()
        self.completed.emit(str(self.output_root), len(self.paths))


class PlaybackWorker(CancellableWorker):
    """Decode saved RGBA frames off the UI thread with bounded buffering."""
    def __init__(self, folder: str, start: int, end: int, fallback_folder: str | None = None) -> None:
        super().__init__()
        self.folder, self.start_frame, self.end_frame = Path(folder), start, end
        self.fallback_folder=Path(fallback_folder) if fallback_folder else None
        self.frames = queue.Queue(maxsize=3)
        self.exhausted = False

    def execute(self) -> None:
        background = None
        for frame in range(self.start_frame, self.end_frame + 1):
            self.cancellation.check()
            path = self.folder / f"frame_{frame:06d}.png"
            if not path.exists() and self.fallback_folder:path=self.fallback_folder/path.name
            if not path.exists():
                raise RuntimeError(f"No saved matte for frame {frame}. Generate or resume matting to complete the video.")
            with Image.open(path) as source:
                image = source.convert("RGBA")
                image.thumbnail((1600, 1600), Image.Resampling.BILINEAR)
            if background is None or background.size != image.size:
                background = checkerboard(image.size).convert("RGBA")
            composite = background.copy()
            composite.alpha_composite(image)
            while True:
                self.cancellation.check()
                try:
                    self.frames.put((frame, composite), timeout=.05)
                    break
                except queue.Full:
                    pass
        self.exhausted = True
        while not self.frames.empty():
            self.cancellation.check()
            time.sleep(.01)


class RotoSetupWorker(CancellableWorker):
    def execute(self):
        root = Path(__file__).resolve().parents[1]
        code = "from huggingface_hub import snapshot_download; snapshot_download('facebook/sam3', local_dir=" + repr(str(root / 'models/SAM3')) + ", allow_patterns=['*.json','*.txt','model.safetensors'])"
        self.cancellation.run_process([str(root / '.roto-venv/Scripts/python.exe'), '-c', code], capture_output=True)
        self.done = True


class MatteWindow(GuidanceMixin, WorkflowMixin, QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Matte App")
        self.resize(1360, 900)
        self.original: Image.Image | None = None
        self.alpha: Image.Image | None = None
        self.seed_path = ""
        self.seed_source = ""
        self.seed_frame = 1
        self.points: list[PointHint] = []
        self.boxes: list[BoxHint] = []
        self.video_path: str | None = None
        self.video_test_active = False
        self.current_media_path: str | None = None
        self.current_frame = 1
        self._requested_frame: int | None = None
        self.video_info: VideoInfo | None = None
        self.video_output_dir: str | None = None
        self.preview_worker: PreviewWorker | None = None
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._playback_tick)
        self._playback_displayed = False
        self._live_preview_timer = QTimer(self)
        self._live_preview_timer.setInterval(100)
        self._live_preview_timer.timeout.connect(self._processed_frame_tick)
        self._live_preview_started = False
        self._restore_guidance = False
        self._reload_after_worker = False
        self._pending_frame_action: str | None = None
        self._preview_loaded = False
        self.results_index_path = Path(__file__).resolve().parents[1] / "results_index.json"
        try:
            self.queue_outputs = json.loads(self.results_index_path.read_text(encoding="utf-8"))
            if not isinstance(self.queue_outputs, dict):
                self.queue_outputs = {}
        except (OSError, ValueError):
            self.queue_outputs: dict[str, str] = {}
        self._navigation_timer = QTimer(self)
        self._navigation_timer.setSingleShot(True)
        self._navigation_timer.setInterval(200)
        self._navigation_timer.timeout.connect(self.navigate_to_frame)
        self.worker: InferenceWorker | None = None
        self.video_worker: VideoInferenceWorker | None = None
        self.queue_worker: QueueWorker | None = None
        self._active_worker: CancellableWorker | None = None
        self._cancel_requested = False
        self._close_pending = False
        self.matte_editor = MatteEditor()
        self._preview_source: Image.Image | None = None
        self._original_preview: Image.Image | None = None
        self._paint_refresh = QTimer(self)
        self._paint_refresh.setSingleShot(True)
        self._paint_refresh.setInterval(33)
        self._paint_refresh.timeout.connect(self._refresh_canvas)
        self._build_ui()
        self._refresh_canvas()
        self._update_navigation()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)

        controls = QFrame()
        controls.setObjectName("controls")
        controls.setMinimumWidth(300)
        sidebar = QVBoxLayout(controls)
        sidebar.setContentsMargins(18, 20, 18, 20)
        sidebar.setSpacing(14)
        title = QLabel("MATTE APP")
        title.setObjectName("title")
        sidebar.addWidget(title)
        caption = QLabel("Local image and video matting")
        caption.setObjectName("caption")
        sidebar.addWidget(caption)
        sidebar.addSpacing(16)

        self.model_picker = QComboBox()
        self.model_picker.addItems(["BiRefNet HR Matting", "LiteSDMatte (guided)", *ROTO_MODELS])
        self.model_picker.currentTextChanged.connect(lambda _: self._update_roto_controls())
        self.roto_status = QLabel("")
        self.roto_status.setObjectName("hint")
        self.roto_status.setWordWrap(True)
        self.get_sam3_button = QPushButton("Get SAM 3 weights")
        self.get_sam3_button.clicked.connect(self.get_sam3_weights)
        self.generate_button = QPushButton("Generate matte")
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self.generate)
        self.test_frame_button = QPushButton("Test selected frame")
        self.test_frame_button.clicked.connect(self.test_current_frame)

        self.stop_button = QPushButton("Stop / Cancel")
        self.stop_button.setObjectName("stop")
        self.stop_button.setToolTip("Cancel the current job and stop all remaining queued work")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.cancel_jobs)

        sidebar.addSpacing(8)
        sidebar.addWidget(QLabel("Clips to process"))
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(100)
        self.queue_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.queue_list.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.queue_list.setTextElideMode(Qt.ElideNone)
        self.queue_list.setToolTip("Drag the divider beside this panel to widen file paths")
        self.queue_list.currentItemChanged.connect(lambda current, previous: current and self.review_queue_item(current))
        self.queue_list.setSpacing(5)
        sidebar.addWidget(self.queue_list)
        queue_buttons = FlowLayout()
        self.add_queue_button = add_queue = QPushButton("Add files")
        add_queue.clicked.connect(self.add_queue_files)
        self.remove_queue_button = remove_queue = QPushButton("Remove")
        remove_queue.clicked.connect(self.remove_queue_item)
        queue_buttons.addWidget(add_queue)
        queue_buttons.addWidget(remove_queue)
        sidebar.addLayout(queue_buttons)
        queue_extra = FlowLayout()
        self.clear_queue_button = QPushButton("Clear queue")
        self.clear_queue_button.setToolTip("Remove all queue entries; saved results stay on disk")
        self.clear_queue_button.clicked.connect(self.clear_queue)
        self.load_results_button = QPushButton("Load results…")
        self.load_results_button.setToolTip("Open an exported frames folder or an earlier queue output folder")
        self.load_results_button.clicked.connect(self.load_results)
        queue_extra.addWidget(self.clear_queue_button)
        queue_extra.addWidget(self.load_results_button)
        sidebar.addLayout(queue_extra)
        self.queue_run_button = QPushButton("Run queue")
        self.queue_run_button.setObjectName("primary")
        self.queue_run_button.clicked.connect(self.run_queue)
        sidebar.addWidget(self.queue_run_button)

        self.point_button = QToolButton(text="Point +")
        self.point_button.setToolTip("Keep foreground near this point")
        self.point_button.setCheckable(True)
        self.negative_button = QToolButton(text="Point −")
        self.negative_button.setToolTip("Remove foreground near this point")
        self.negative_button.setCheckable(True)
        self.box_button = QToolButton(text="Box")
        self.box_button.setCheckable(True)
        group = QButtonGroup(self)
        group.addButton(self.point_button)
        group.addButton(self.negative_button)
        group.addButton(self.box_button)
        self.point_button.setChecked(True)
        self.point_button.toggled.connect(lambda checked: checked and self._update_edit_controls())
        self.negative_button.toggled.connect(lambda checked: checked and self._update_edit_controls())
        self.box_button.toggled.connect(lambda checked: checked and self._update_edit_controls())
        guidance_layout = QHBoxLayout()
        guidance_layout.setSpacing(12)
        guidance_layout.addWidget(QLabel("Guidance"))
        guidance_layout.addWidget(self.point_button)
        guidance_layout.addWidget(self.negative_button)
        guidance_layout.addWidget(self.box_button)
        self.clear_button = clear = QPushButton("Clear guidance")
        clear.setText("Clear all")
        clear.clicked.connect(self.clear_guidance)
        guidance_layout.addWidget(clear)
        hint = QLabel("Point + keeps foreground; Point − removes it locally. Or drag a box to select a target. Then regenerate. Guidance saves automatically per file; video queues use frame 1.")
        hint.setWordWrap(True)
        hint.setObjectName("hint")
        sidebar.addStretch()
        self.status = QLabel("Add files, then select an item in the queue to load it.")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(10)
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setMinimumWidth(320)
        sidebar_scroll.setWidget(controls)
        self.splitter.addWidget(sidebar_scroll)
        layout.addWidget(self.splitter)

        preview = QWidget()
        main = QVBoxLayout(preview)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(14)
        switcher = FlowLayout()
        switcher.setSpacing(12)
        self.view_tabs = QTabBar()
        self.view_tabs.setExpanding(False)
        self.view_tabs.setDrawBase(False)
        for label in ("Original", "Matte", "Cutout"):
            self.view_tabs.addTab(label)
        self.view_tabs.currentChanged.connect(self._refresh_canvas)
        switcher.addWidget(self.view_tabs)
        switcher.addStretch()
        self.export_matte_button = QPushButton("Export matte")
        self.export_matte_button.clicked.connect(self.export_matte)
        self.export_button = QPushButton("Export PNG")
        self.export_button.setObjectName("primary")
        self.export_button.clicked.connect(self.export_cutout)
        switcher.addWidget(self.export_matte_button)
        switcher.addWidget(self.export_button)
        self.fit_button = QPushButton("Fit")
        self.fit_button.clicked.connect(lambda: self.canvas.reset_zoom())
        self.zoom_out_button = QPushButton("−")
        self.zoom_in_button = QPushButton("+")
        self.zoom_out_button.clicked.connect(lambda: self.canvas.set_zoom(self.canvas._zoom / 1.25))
        self.zoom_in_button.clicked.connect(lambda: self.canvas.set_zoom(self.canvas._zoom * 1.25))
        self.zoom_percent = QSpinBox()
        self.zoom_percent.setRange(100, 1600)
        self.zoom_percent.setSingleStep(25)
        self.zoom_percent.setSuffix("%")
        self.zoom_percent.setMinimumWidth(120)
        self.zoom_percent.setKeyboardTracking(False)
        self.zoom_percent.setToolTip("Zoom relative to Fit. Mouse wheel zooms at the pointer; middle-button drag pans.")
        self.zoom_percent.valueChanged.connect(lambda value: self.canvas.set_zoom(value / 100))
        for control in (self.fit_button, self.zoom_out_button, self.zoom_percent, self.zoom_in_button):
            switcher.addWidget(control)
        main.addLayout(switcher)
        self.guidance_panel = QWidget()
        guidance_panel_layout = QVBoxLayout(self.guidance_panel)
        self.guidance_panel.setObjectName("toolPanel")
        guidance_panel_layout.setContentsMargins(16, 14, 16, 14)
        guidance_panel_layout.setSpacing(10)
        elements = guidance_layout
        self.guidance_picker = QComboBox()
        self.guidance_picker.setMinimumWidth(180)
        self.guidance_picker.setToolTip("Select a point or box to remove, or right-click it in Original")
        self.remove_guidance_button = QPushButton("Remove selected")
        self.remove_guidance_button.clicked.connect(self.remove_selected_guidance)
        elements.addWidget(self.guidance_picker, 1)
        elements.addWidget(self.remove_guidance_button)
        self.roto_panel = QWidget()
        self.roto_panel.setObjectName("rotoOptions")
        roto_layout = QHBoxLayout(self.roto_panel)
        roto_layout.setContentsMargins(0, 0, 0, 0)
        roto_layout.setSpacing(10)
        self.roto_text = QLineEdit()
        self.roto_text_label = QLabel("SAM 3 object description (optional)")
        self.roto_text_label.setWordWrap(True)
        guidance_panel_layout.addWidget(self.roto_text_label)
        self.roto_text.editingFinished.connect(self.save_file_guidance)
        self.roto_text.setPlaceholderText("e.g. person in red jacket")
        self.roto_text.setToolTip("SAM 3 can find objects by text. Leave blank to select with points or boxes. A starting mask takes priority over text.")
        guidance_panel_layout.addWidget(self.roto_text)
        settings = roto_layout
        settings.addWidget(QLabel("Working size"))
        self.roto_size = QSpinBox()
        self.roto_size.setRange(256, 1536)
        self.roto_size.setSingleStep(128)
        self.roto_size.setValue(512)
        self.roto_size.setSuffix(" px")
        self.roto_size.setMinimumWidth(130)
        self.roto_size.setToolTip("Matting working resolution; exports keep the original dimensions. Start at 512 px for VideoMaMa on a 12 GB GPU.")
        settings.addWidget(self.roto_size)
        self.roto_chunk_label = QLabel("VideoMaMa section")
        settings.addWidget(self.roto_chunk_label)
        self.roto_chunk = QSpinBox()
        self.roto_chunk.setRange(2, 16)
        self.roto_chunk.setValue(4)
        self.roto_chunk.setSuffix(" frames")
        self.roto_chunk.setMinimumWidth(140)
        settings.addWidget(self.roto_chunk)
        seeds = roto_layout
        self.use_matte_seed_button = QPushButton("Use displayed matte as start mask")
        self.use_matte_seed_button.clicked.connect(self.use_matte_seed)
        self.load_seed_button = QPushButton("Load start mask…")
        self.load_seed_button.clicked.connect(self.load_roto_seed)
        self.clear_seed_button = QPushButton("Clear start mask")
        self.clear_seed_button.clicked.connect(self.clear_roto_seed)
        for button in (self.use_matte_seed_button, self.load_seed_button, self.clear_seed_button): seeds.addWidget(button)
        self.seed_label = QLabel("No starting mask — select the subject with points, a box, or SAM 3 text.")
        self.seed_label.setObjectName("hint")
        self.seed_label.setWordWrap(True)
        guidance_layout.addStretch()
        self.guidance_toolbar = toolbar(guidance_layout)
        guidance_panel_layout.insertWidget(0, self.guidance_toolbar)
        guidance_panel_layout.addWidget(hint)
        guidance_panel_layout.addWidget(self.seed_label)
        main.addWidget(self.guidance_panel)
        self.brush_group = QButtonGroup(self)
        brush_tools = FlowLayout()
        brush_tools.setSpacing(12)
        brush_tools.addWidget(QLabel("Paint this frame"))
        self.white_brush_button = QToolButton(text="White / Keep")
        self.black_brush_button = QToolButton(text="Black / Remove")
        for button, mode, tooltip in (
            (self.white_brush_button, "paint_white", "Paint opaque alpha (white) to restore the subject"),
            (self.black_brush_button, "paint_black", "Paint transparent alpha (black) to remove the background"),
        ):
            button.setCheckable(True)
            button.setToolTip(tooltip)
            self.brush_group.addButton(button)
            button.toggled.connect(lambda checked, brush_mode=mode: checked and self._select_brush(brush_mode))
            brush_tools.addWidget(button)
        brush_tools.addWidget(QLabel("Size"))
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 2000)
        self.brush_size.setValue(40)
        self.brush_size.setSuffix(" px")
        self.brush_size.setMinimumWidth(110)
        self.brush_size.setToolTip("Brush diameter in original-image pixels")
        brush_tools.addWidget(self.brush_size)
        self.undo_brush_button = QPushButton("Undo")
        self.redo_brush_button = QPushButton("Redo")
        self.undo_brush_button.setToolTip("Undo the last brush stroke (Ctrl+Z)")
        self.redo_brush_button.setToolTip("Redo a brush stroke (Ctrl+Shift+Z or Ctrl+Y)")
        self.undo_brush_button.clicked.connect(self.undo_brush)
        self.redo_brush_button.clicked.connect(self.redo_brush)
        brush_tools.addWidget(self.undo_brush_button)
        brush_tools.addWidget(self.redo_brush_button)
        brush_tools.addStretch()
        brush_hint = QLabel("White keeps pixels; black removes pixels. Paint on Matte, then inspect Cutout or Export PNG. Regenerating replaces edits.")
        brush_hint.setWordWrap(True)
        brush_hint.setObjectName("hint")
        self.paint_panel = QWidget()
        paint_panel_layout = QVBoxLayout(self.paint_panel)
        self.paint_panel.setObjectName("toolPanel")
        paint_panel_layout.setContentsMargins(16, 14, 16, 14)
        paint_panel_layout.setSpacing(10)
        paint_panel_layout.addLayout(brush_tools)
        paint_panel_layout.addWidget(brush_hint)
        self.white_brush_button.setChecked(True)
        main.addWidget(self.paint_panel)
        self.video_panel = QFrame()
        self.video_panel.setObjectName("videoPanel")
        video_layout = QVBoxLayout(self.video_panel)
        video_layout.setContentsMargins(14, 14, 14, 14)
        video_layout.setSpacing(12)
        navigation = QHBoxLayout()
        navigation.setSpacing(12)
        self.previous_frame_button = QPushButton("Previous")
        self.next_frame_button = QPushButton("Next")
        self.previous_frame_button.clicked.connect(lambda: self.request_frame(self.frame_spin.value() - 1))
        self.next_frame_button.clicked.connect(lambda: self.request_frame(self.frame_spin.value() + 1))
        self.play_button = QPushButton("Play matted video")
        self.play_button.setToolTip("Play saved frames from the selected frame to the end; at the end, restart from frame 1")
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_all_button = QPushButton("Play from beginning")
        self.play_all_button.clicked.connect(lambda: self.start_playback(1))
        playback_controls = navigation
        playback_controls.setSpacing(12)
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(self.play_all_button)
        navigation.addWidget(self.previous_frame_button)
        navigation.addWidget(self.next_frame_button)
        navigation.addWidget(QLabel("Frame"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setMinimumWidth(95)
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.valueChanged.connect(self.request_frame)
        navigation.addWidget(self.frame_spin)
        self.frame_total = QLabel("/ 1")
        navigation.addWidget(self.frame_total)
        self.frame_time = QLabel("")
        navigation.addWidget(self.frame_time)
        navigation.addStretch()
        self.resume_button = QPushButton("Resume from this frame")
        self.resume_button.setObjectName("primary")
        self.resume_button.clicked.connect(self.resume_video)
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.valueChanged.connect(self.request_frame)
        self.frame_slider.sliderPressed.connect(self._navigation_timer.stop)
        self.frame_slider.sliderReleased.connect(self.commit_slider_frame)
        video_layout.addWidget(self.frame_slider)
        self.video_toolbar = toolbar(navigation)
        video_layout.addWidget(self.video_toolbar)
        self.media_actions_panel = QWidget()
        media_actions_layout = QVBoxLayout(self.media_actions_panel)
        self.media_actions_panel.setObjectName("toolPanel")
        media_actions_layout.setContentsMargins(16, 14, 16, 14)
        media_actions_layout.setSpacing(10)
        saved_actions = FlowLayout()
        saved_actions.setSpacing(12)
        self.load_matte_folder_button = QPushButton("Load matte folder…")
        self.load_matte_folder_button.setToolTip("Attach saved matting to this video, including after reopening the app")
        self.load_matte_folder_button.clicked.connect(self.load_video_matte_folder)
        saved_actions.addWidget(self.load_matte_folder_button)
        self.matte_folder_label = QLabel("No saved matte folder linked")
        self.matte_folder_label.setWordWrap(True)
        self.matte_folder_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.matte_folder_label.setObjectName("hint")
        media_actions_layout.addWidget(self.matte_folder_label)
        saved_actions.addWidget(self.test_frame_button)
        saved_actions.addWidget(self.generate_button)
        saved_actions.addWidget(self.resume_button)
        media_actions_layout.addLayout(saved_actions)
        self.resume_hint = resume_hint = QLabel("Resume replaces the selected frame and later results. Earlier frames are kept.")
        resume_hint.setObjectName("hint")
        resume_hint.setWordWrap(True)
        media_actions_layout.addWidget(resume_hint)
        self.video_panel.hide()
        self.undo_brush_action = QAction("Undo brush stroke", self)
        self.undo_brush_action.setShortcut(QKeySequence.Undo)
        self.undo_brush_action.triggered.connect(self.undo_brush)
        self.addAction(self.undo_brush_action)
        self.redo_brush_action = QAction("Redo brush stroke", self)
        self.redo_brush_action.setShortcuts([QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")])
        self.redo_brush_action.triggered.connect(self.redo_brush)
        self.addAction(self.redo_brush_action)
        self.canvas = ImageCanvas()
        self.canvas.zoom_changed.connect(self._sync_zoom)
        self.canvas.point_added.connect(self.add_point)
        self.canvas.box_added.connect(self.add_box)
        self.canvas.guidance_removed.connect(self.remove_guidance)
        self.brush_size.valueChanged.connect(self.canvas.set_brush_diameter)
        self.canvas.stroke_started.connect(self.matte_editor.begin_stroke)
        self.canvas.brush_segment.connect(self.paint_matte)
        self.canvas.stroke_finished.connect(self.finish_matte_stroke)
        main.addWidget(self.canvas, 1)
        main.addWidget(self.video_panel)
        self.build_workflow(main, media_actions_layout, sidebar)
        saved_actions.addWidget(self.save_edits_button)
        process_status = QHBoxLayout()
        process_status.addWidget(self.status,1)
        process_status.addWidget(self.stop_button)
        media_actions_layout.addLayout(process_status)
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setFrameShape(QFrame.NoFrame)
        self.preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.preview_scroll.setMinimumWidth(330)
        self.preview_scroll.setWidget(preview)
        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0,0,0,0)
        workspace_layout.addWidget(self.preview_scroll,1)
        workspace_layout.addWidget(self.media_actions_panel)
        self.splitter.addWidget(workspace)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([320, 1020])
        self.splitter.handle(1).setToolTip("Drag to resize the Queue/path panel")
        self.setStyleSheet("""
            QWidget { background: #10141c; color: #e9edf5; font-size: 13px; font-family: Segoe UI; }
            QLabel { background: transparent; border: none; }
            QWidget#refineControls, QWidget#rangeControls { background: transparent; border: none; }
            #rotoOptions { background: transparent; }
            #controls, #videoPanel, #toolPanel { background: #191f2b; border: 1px solid #303b4e; border-radius: 12px; }
            #title { font-size: 21px; font-weight: 700; letter-spacing: 1px; color: #f2f6ff; }
            #caption, #hint { color: #a4b2c8; } #hint { font-size: 12px; }
            #status { color: #bdcce2; padding: 10px; background: #222c3d; border-radius: 8px; }
            QPushButton, QToolButton, QComboBox, QLineEdit { background: #263246; border: 1px solid #3a4a62; border-radius: 8px; padding: 10px 14px; min-height: 22px; }
            QPushButton:hover, QToolButton:hover, QComboBox:hover { background: #32415a; border-color: #7499c7; }
            QPushButton:pressed, QToolButton:pressed { background: #1e6183; }
            QPushButton:focus, QToolButton:focus, QComboBox:focus, QSpinBox:focus, QLineEdit:focus { border: 1px solid #6ac9ff; }
            QPushButton#primary, QToolButton:checked { background: #247ca3; border-color: #4ab2d8; color: white; }
            QPushButton#primary:hover, QToolButton:checked:hover { background: #3093bc; }
            QPushButton#stop { background: #8f3f52; border-color: #c2667c; }
            QPushButton#stop:hover { background: #a84d64; }
            QPushButton:disabled, QToolButton:disabled, QComboBox:disabled, QSpinBox:disabled, QLineEdit:disabled { color: #8b98ad; background: #222938; border-color: #333e51; }
            QComboBox { padding-right: 38px; }
            QComboBox::drop-down { width: 28px; border: none; border-left: 1px solid #3a4a62; }
            QComboBox::down-arrow { image: url(__ASSETS__/down.svg); width: 12px; height: 8px; }
            QComboBox QAbstractItemView { background: #202b3d; color: #eef3fb; border: 1px solid #526582; selection-background-color: #247ca3; padding: 6px; outline: none; }
            QComboBox QAbstractItemView::item { min-height: 32px; padding: 6px 10px; }
            QSpinBox { background: #202b3d; padding: 9px 32px 9px 12px; min-height: 24px; border: 1px solid #3a4a62; border-radius: 8px; }
            QSpinBox::up-button, QSpinBox::down-button { width: 25px; border-left: 1px solid #3a4a62; background: #2c3a50; }
            QSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; border-top-right-radius: 8px; }
            QSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; border-bottom-right-radius: 8px; }
            QSpinBox::up-arrow { image: url(__ASSETS__/up.svg); width: 10px; height: 7px; }
            QSpinBox::down-arrow { image: url(__ASSETS__/down.svg); width: 10px; height: 7px; }
            QSplitter::handle { background: #303b4e; border-radius: 4px; }
            QSplitter::handle:hover { background: #4ab2d8; }
            QSlider::groove:horizontal { height: 6px; background: #3b4b64; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #4ab2d8; border-radius: 3px; }
            QSlider::handle:horizontal { width: 16px; margin: -6px 0; background: #6ac9ff; border-radius: 8px; }
            QListWidget { background: #121925; border: 1px solid #35445b; border-radius: 8px; padding: 8px; }
            QListWidget::item { padding: 8px 10px; border-radius: 5px; }
            QListWidget::item:selected { background: #254f71; color: white; }
            QListWidget::item:hover { background: #28364b; }
            QTabBar::tab { background: #202b3d; color: #b8c5d9; padding: 12px 20px; margin-right: 6px; border: 1px solid #35445b; border-radius: 8px; }
            QTabBar::tab:selected { background: #247ca3; color: white; border-color: #4ab2d8; }
            QTabBar::tab:hover { background: #32415a; }
            QScrollBar:vertical { background: #151d29; width: 10px; margin: 2px; }
            QScrollBar:horizontal { background: #151d29; height: 10px; margin: 2px; }
            QScrollBar::handle { background: #53667f; border-radius: 4px; min-height: 24px; min-width: 24px; }
            QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
            QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
            QToolTip { color: #edf4ff; background: #29384d; border: 1px solid #627c9f; padding: 7px; }
            QMessageBox QLabel { min-width: 260px; }
        """.replace("__ASSETS__", (Path(__file__).parent / "assets").as_posix()))
        # Qt tool buttons can otherwise report sizes smaller than styled text.
        for button in self.findChildren(QPushButton) + self.findChildren(QToolButton):
            button.setMinimumWidth(button.fontMetrics().horizontalAdvance(button.text()) + 32)
            button.setMinimumHeight(44)
        self.model_picker.setMinimumWidth(max(self.model_picker.fontMetrics().horizontalAdvance(self.model_picker.itemText(i)) for i in range(self.model_picker.count())) + 56)
        self.guidance_picker.setMinimumWidth(280)
        self.brush_size.setMinimumWidth(self.brush_size.fontMetrics().horizontalAdvance("2000 px") + 58)
        self.view_tabs.setUsesScrollButtons(False)
        self.view_tabs.setElideMode(Qt.ElideNone)

    def _sync_zoom(self, zoom):
        self.zoom_percent.blockSignals(True)
        self.zoom_percent.setValue(round(zoom * 100))
        self.zoom_percent.blockSignals(False)
        if self.original is not None and self.canvas._image is not None:
            self._refresh_canvas()

    def _update_roto_controls(self):
        if not hasattr(self, "roto_panel"):
            return
        name = self.model_picker.currentText()
        roto = name in ROTO_MODELS
        self.roto_panel.setVisible(roto)
        if hasattr(self,"advanced_panel"):
            self.advanced_panel.setVisible(roto and self.advanced_toggle.isChecked())
            self.update_workflow()
        self.seed_label.setVisible(roto and name not in REFINERS)
        self.roto_text.setVisible(roto and name.startswith("SAM 3"))
        self.roto_text_label.setVisible(roto and name.startswith("SAM 3"))
        self.roto_status.setVisible(roto)
        root = Path(__file__).resolve().parents[1]
        self.get_sam3_button.setVisible(roto and name.startswith("SAM 3") and not (root / "models/SAM3/model.safetensors").exists())
        self.get_sam3_button.setEnabled(self._active_worker is None)
        if roto:
            tracker, matte = capabilities(name)
            self.roto_status.setText(readiness(name, root))
            self.roto_text.setVisible(tracker == "sam3")
            self.roto_text_label.setVisible(tracker == "sam3")
            self.roto_size.setEnabled(matte != "tracking" and self._active_worker is None)
            self.roto_chunk.setVisible(matte == "videomama")
            self.roto_chunk_label.setVisible(matte == "videomama")
        editable = self._active_worker is None and self.original is not None
        self.roto_text.setEnabled(editable)
        self.roto_chunk.setEnabled(editable)
        self.use_matte_seed_button.setEnabled(editable and self.alpha is not None)
        self.load_seed_button.setEnabled(editable)
        self.clear_seed_button.setEnabled(editable and bool(self.seed_path))
        self.seed_label.setText(f"Starting mask: frame {self.seed_frame} of {Path(self.seed_source).name}. Resume from that frame to apply it." if self.seed_path else "No starting mask — select the subject with points, a box, or SAM 3 text.")
        self.seed_label.setToolTip(self.seed_path)

    def get_sam3_weights(self):
        return self.open_installer(focus_model='sam3')

    def _save_roto_seed(self, alpha):
        if self.original is None or self._active_worker is not None:
            return
        if alpha.size != self.original.size:
            QMessageBox.information(self, "Mask size mismatch", "The starting mask must match this image or video frame’s dimensions.")
            return
        folder = Path(__file__).resolve().parents[1] / ".roto-seeds"
        folder.mkdir(exist_ok=True)
        path = folder / f"{uuid.uuid4().hex}.png"
        alpha.convert("L").save(path, "PNG", compress_level=1)
        self.seed_path, self.seed_source, self.seed_frame = str(path), self.current_media_path or self.video_path or "", self.current_frame
        self.save_file_guidance()
        self._update_roto_controls()
        self.status.setText(f"Starting mask saved for frame {self.current_frame}. Use Resume from this frame for later-frame corrections.")

    def use_matte_seed(self):
        if self.alpha is not None:
            self._save_roto_seed(self.alpha)

    def load_roto_seed(self):
        if self.original is None or self._active_worker is not None:
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Choose starting mask", "", "Masks (*.png *.tif *.tiff)")
        if filename:
            try:
                with Image.open(filename) as image:
                    alpha = image.getchannel("A") if "A" in image.getbands() else image.convert("L")
                self._save_roto_seed(alpha)
            except Exception as error:
                QMessageBox.information(self, "Could not load mask", str(error))

    def clear_roto_seed(self):
        self.seed_path = self.seed_source = ""
        self.save_file_guidance()
        self._update_roto_controls()

    def job_guidance(self):
        hints = [*self.points, *self.boxes]
        if self.model_picker.currentText() in ROTO_MODELS:
            hints.append(RotoHint(self.roto_text.text().strip(), self.roto_size.value(), self.roto_chunk.value(),
                self.seed_path, self.seed_source, self.seed_frame,
                self._mask_input_folder if self.model_picker.currentText() in REFINERS else "",
                self.current_media_path or "", self.current_frame,
                self._trimap_path if self._trimap_source == self.current_media_path else "", self._trimap_frame, self.trimap_width.value()))
        return hints

    def _validate_roto(self, start_frame=None, queue=False):
        name = self.model_picker.currentText()
        if name not in ROTO_MODELS:
            return True
        message = readiness(name, Path(__file__).resolve().parents[1])
        if not message.startswith("Ready"):
            QMessageBox.information(self, "Model setup needed", message)
            return False
        if name in REFINERS:
            if queue:
                QMessageBox.information(self,"Select one clip","Refine saved masks for the selected clip. Automatic workflows still support the queue.")
                return False
            return self.prepare_refinement(self.current_frame if start_frame is None else start_frame)
        if self.video_path and start_frame is not None and start_frame != self.current_frame and (self.points or self.boxes):
            QMessageBox.information(self, "Guidance is on a different frame", "Choose frame 1 before a full export, or use Resume from this frame to apply the displayed guidance.")
            return False
        if self.seed_path:
            if queue:
                QMessageBox.information(self, "Starting mask is specific to one frame", "Clear the starting mask before running a mixed queue. Use points, boxes, or SAM 3 text for queued files.")
                return False
            if start_frame is not None and (self.seed_frame != start_frame or self.seed_source != (self.current_media_path or self.video_path or "")):
                QMessageBox.information(self, "Starting frame mismatch", f"The starting mask belongs to frame {self.seed_frame}. Resume from that frame, or clear the starting mask.")
                return False
        elif not (self.boxes or any(p.positive for p in self.points) or (name.startswith("SAM 3") and self.roto_text.text().strip())):
            QMessageBox.information(self, "Select the subject", "Add a Point + or Box, use SAM 3 text, or load a starting mask.")
            return False
        return True

    def _set_busy(self, busy: bool) -> None:
        for control in (
            self.model_picker,
            self.generate_button, self.test_frame_button, self.queue_run_button,
            self.add_queue_button, self.remove_queue_button,
            self.clear_queue_button, self.load_results_button,
            self.clear_button, self.point_button, self.negative_button, self.box_button,
            self.export_button, self.export_matte_button, self.canvas,
        ):
            control.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.queue_list.setEnabled(not busy or not isinstance(self._active_worker, PreviewWorker))
        self._update_edit_controls()
        self._update_roto_controls()
        self._update_navigation()

    def _start_worker(self, worker: CancellableWorker) -> None:
        self._live_preview_started = False
        self._navigation_timer.stop()
        self.view_tabs.setEnabled(not isinstance(worker, PlaybackWorker))
        if not isinstance(worker, PreviewWorker):
            self._requested_frame = None
        self.canvas.finish_stroke()
        self._active_worker = worker
        self._cancel_requested = False
        self.stop_button.setText("Stop / Cancel")
        self._set_busy(True)
        worker.finished.connect(self._on_worker_finished)
        worker.start()
        if isinstance(worker, (VideoInferenceWorker, QueueWorker)):
            self._live_preview_timer.start()

    def _processed_frame_tick(self, update_status=True) -> None:
        worker = self._active_worker
        if not isinstance(worker, (VideoInferenceWorker, QueueWorker)):
            return
        try:
            preview = worker.live_frames.get_nowait()
        except queue.Empty:
            return
        changed_media = self.current_media_path != preview.path
        if changed_media:
            self.restore_file_guidance(preview.path,preview.frame,True)
        if changed_media:
            self.canvas.reset_zoom()
        self.current_media_path = preview.path
        self.video_path = preview.path if preview.info else None
        self.video_info = preview.info
        self.video_output_dir = preview.output_dir
        self.current_frame = preview.frame
        self._requested_frame = None
        self.original = preview.original.convert("RGBA")
        self.alpha = self.original.getchannel("A")
        self.original.putalpha(255)
        self.video_test_active = False
        self.clear_image_save_state()
        self._reset_matte_edits()
        if not self._live_preview_started or changed_media:
            self.view_tabs.setCurrentIndex(2)
        self._live_preview_started = True
        if isinstance(worker, QueueWorker):
            self.queue_list.blockSignals(True)
            for row in range(self.queue_list.count()):
                if self.queue_list.item(row).text() == preview.path:
                    self.queue_list.setCurrentRow(row)
                    break
            self.queue_list.blockSignals(False)
            total = preview.info.frame_count if preview.info else 1
            if update_status:
                self.status.setText(f"Queue: {Path(preview.path).name} — frame {preview.frame} / {total}")
        self._update_navigation()
        self._refresh_canvas()

    def cancel_jobs(self) -> None:
        if self._active_worker is None:
            return
        self._cancel_requested = True
        self._active_worker.cancel()
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stopping…")
        self.status.setText("Stopping… waiting for the current operation to finish safely. Remaining queued jobs will not run.")

    def _on_worker_finished(self) -> None:
        self._processed_frame_tick(update_status=False)
        self._live_preview_timer.stop()
        if isinstance(self.sender(), PlaybackWorker):
            self._playback_timer.stop()
            self._active_worker = None
            self._cancel_requested = False
            self.view_tabs.setEnabled(True)
            self.stop_button.setText("Stop / Cancel")
            self._set_busy(False)
            if self._close_pending:
                self.close()
            elif self._playback_displayed:
                self._load_media(self.video_path, self.video_output_dir, self.current_frame, self.video_info)
            return
        if isinstance(self.sender(), PreviewWorker) and not self._preview_loaded:
            self._requested_frame = None
        self._active_worker = None
        self._set_busy(False)
        self.stop_button.setText("Stop / Cancel")
        if isinstance(self.sender(), RotoSetupWorker) and getattr(self.sender(), "done", False):
            self.status.setText("SAM 3 weights are ready. Select the subject and test a frame.")
        if self._cancel_requested:
            self.status.setText("Cancelled — remaining queued jobs stopped. Any completed exports were kept. Press Run queue to restart manually.")
        if self._close_pending:
            self.close()
        elif self._reload_after_worker:
            self._reload_after_worker = False
            if self.current_media_path:
                self._load_media(self.current_media_path, self.video_output_dir, self.current_frame, self.video_info)
        elif isinstance(self.sender(), PreviewWorker) and self._pending_frame_action:
            action, self._pending_frame_action = self._pending_frame_action, None
            if self._preview_loaded and not self._cancel_requested:
                QTimer.singleShot(0, self.test_video_frame if action == "test" else self.resume_video)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._active_worker is not None:
            self._close_pending = True
            self.cancel_jobs()
            event.ignore()
        else:
            if not self.save_file_guidance(): event.ignore(); return
            if self.guard_frame_edits(): event.accept()
            else: event.ignore()

    def current_view(self) -> str:
        return ("original", "matte", "cutout")[self.view_tabs.currentIndex()]

    def _update_navigation(self) -> None:
        if not hasattr(self, "video_panel"):
            return
        video = bool(self.video_path and self.video_info)
        selected_frame = self._requested_frame if self._requested_frame is not None else self.current_frame
        self.video_panel.setVisible(video)
        self.load_matte_folder_button.setVisible(video)
        self.matte_folder_label.setVisible(video)
        self.resume_button.setVisible(video)
        self.resume_hint.setVisible(video)
        self.generate_button.setEnabled(self.original is not None and self._active_worker is None)
        playing = isinstance(self._active_worker, PlaybackWorker)
        self.play_button.setText("Pause" if playing else "Play matted video")
        self.play_button.setEnabled(playing or (video and bool(self.video_output_dir) and self._active_worker is None))
        self.play_all_button.setEnabled(video and bool(self.video_output_dir) and self._active_worker is None)
        available = video and self._active_worker is None
        self.frame_spin.setEnabled(available)
        self.frame_slider.setEnabled(available)
        self.resume_button.setEnabled(available)
        self.load_matte_folder_button.setEnabled(available)
        if self.video_output_dir:
            name = Path(self.video_output_dir).name
            self.matte_folder_label.setText(f"Saved: {name[:40]}{'…' if len(name) > 40 else ''}")
            self.matte_folder_label.setToolTip(self.video_output_dir)
        else:
            self.matte_folder_label.setText("No saved matte folder linked")
            self.matte_folder_label.setToolTip("")
        self.previous_frame_button.setEnabled(available and selected_frame > 1)
        self.next_frame_button.setEnabled(available and selected_frame < self.video_info.frame_count)
        self.test_frame_button.setEnabled(available)
        if video:
            for control in (self.frame_spin, self.frame_slider):
                control.blockSignals(True)
                control.setRange(1, self.video_info.frame_count)
                if control is self.frame_spin:
                    control.setMinimumWidth(max(110,control.fontMetrics().horizontalAdvance(str(self.video_info.frame_count)) + 68))
                control.setValue(selected_frame)
                control.blockSignals(False)
            self.frame_total.setText(f"/ {self.video_info.frame_count}")
            if self.video_info.fps:
                seconds = (selected_frame - 1) / self.video_info.fps
                self.frame_time.setText(f"~{int(seconds // 60):02d}:{seconds % 60:05.2f}")
            else:
                self.frame_time.setText("")
        self.update_workflow()

    def toggle_playback(self) -> None:
        if isinstance(self._active_worker, PlaybackWorker):
            self._active_worker.cancel()
            self._playback_timer.stop()
            self.play_button.setEnabled(False)
            return
        start = self._requested_frame or self.current_frame
        if self.video_info and start >= self.video_info.frame_count:
            start = 1
        self.start_playback(start)

    def start_playback(self, frame: int = 1) -> None:
        if self._active_worker is not None or not self.video_info or not self.video_output_dir:
            return
        self._playback_displayed = False
        self.view_tabs.setCurrentIndex(2)
        available = sorted(int(p.stem[6:]) for p in Path(self.video_output_dir).glob('frame_*.png') if p.stem[6:].isdigit())
        if available and frame == 1: frame = available[0]
        last = min(self.video_info.frame_count,max(available)) if available else self.video_info.frame_count
        worker = PlaybackWorker(self.video_output_dir, frame, max(frame,last))
        worker.failed.connect(self._playback_failed)
        self._start_worker(worker)
        self._playback_timer.start(max(1, round(1000 / (self.video_info.fps or 24))))
        self.status.setText("Playing saved matted frames. Pause to inspect or edit a frame.")

    def _playback_failed(self, details: str) -> None:
        self.status.setText("Playback stopped — saved matte unavailable.")
        QMessageBox.information(self, "Matted video playback", details.splitlines()[-1])

    def _playback_tick(self) -> None:
        worker = self._active_worker
        if not isinstance(worker, PlaybackWorker):
            self._playback_timer.stop()
            return
        try:
            frame, image = worker.frames.get_nowait()
        except queue.Empty:
            return
        self.canvas.set_image(image, self.original.size)
        self.current_frame = frame
        self._requested_frame = None
        self._playback_displayed = True
        self._update_navigation()

    def request_frame(self, frame: int) -> None:
        if self._active_worker is not None or not self.video_info:
            return
        frame = max(1, min(frame, self.video_info.frame_count))
        self._navigation_timer.stop()
        self._requested_frame = frame if frame != self.current_frame else None
        for control in (self.frame_spin, self.frame_slider):
            control.blockSignals(True)
            control.setValue(frame)
            control.blockSignals(False)
        self._update_navigation()
        if self._requested_frame is not None and not self.frame_slider.isSliderDown():
            self._navigation_timer.start()

    def commit_slider_frame(self) -> None:
        self._navigation_timer.stop()
        if self._requested_frame is not None:
            self.navigate_to_frame()

    def navigate_to_frame(self) -> None:
        if self.video_path and self._active_worker is None:
            self._load_media(self.video_path, self.video_output_dir, self.frame_spin.value(), self.video_info)

    def _load_media(self, path: str, output_dir: str | None = None, frame: int = 1,
                    info: VideoInfo | None = None, restore_guidance: bool = False) -> None:
        if self._active_worker is not None:
            return
        if not self.save_file_guidance(): return
        if not self.guard_frame_edits():
            self.refresh_results()
            return
        self.canvas.finish_stroke()
        self._restore_guidance = restore_guidance
        self._requested_frame = frame
        self._preview_loaded = False
        self.status.setText(f"Loading {Path(path).name} — frame {frame}…")
        self.preview_worker = PreviewWorker(path, frame, output_dir, info)
        self.preview_worker.completed.connect(self.on_preview_ready)
        self.preview_worker.failed.connect(self.on_matte_failed)
        self._start_worker(self.preview_worker)

    def on_preview_ready(self, preview: MediaPreview) -> None:
        if self._cancel_requested:
            return
        self._preview_loaded = True
        if self.current_media_path != preview.path:
            self.canvas.reset_zoom()
        changed_source = self.current_media_path != preview.path
        self.current_media_path = preview.path
        if self._mask_input_folder:
            entries = self._library().get(str(Path(preview.path).resolve()),[])
            if not any(Path(e['folder']).resolve()==Path(self._mask_input_folder).resolve() for e in entries):
                self._mask_input_folder = ''
        self.video_path = preview.path if preview.info else None
        self.video_info = preview.info
        self.current_frame = preview.frame
        self._requested_frame = None
        self.video_output_dir = preview.output_dir
        self.original, self.alpha = preview.original, preview.alpha
        self.clear_image_save_state()
        self._matte_dirty = False
        if preview.output_dir:
            self.register_result(preview.path,preview.output_dir)
        self.refresh_results()
        self.video_test_active = False
        self._reset_matte_edits()
        has_saved_guidance = self.restore_file_guidance(preview.path,preview.frame,changed_source)
        if self._restore_guidance and not has_saved_guidance:
            self._restore_saved_guidance(preview.output_dir, preview.frame)
            self.save_file_guidance()
        kind = "Saved matte loaded" if preview.alpha is not None else "Source loaded — no saved matte for this frame"
        self.status.setText(f"{kind}: {Path(preview.path).name}, frame {preview.frame}.")
        self.view_tabs.setCurrentIndex(2 if preview.alpha is not None else 0)
        if getattr(self,'_pending_view_after_load',None) is not None:
            self.view_tabs.setCurrentIndex(self._pending_view_after_load)
            self._pending_view_after_load = None
        self._update_navigation()
        self._refresh_canvas()

        self.restore_project_ui()

    def _restore_saved_guidance(self, output_dir: str | None, frame: int) -> None:
        metadata = read_metadata(Path(output_dir)) if output_dir else {}
        # Later runs take precedence only over the frames they actually completed.
        settings = metadata
        for run in metadata.get("runs", []):
            if run.get("start_frame", 1) <= frame <= run.get("last_frame", 0):
                settings = run
        self.points.clear()
        self.boxes.clear()
        self.seed_path = self.seed_source = ""
        try:
            for hint in settings.get("guidance", []):
                if hint.get("type") == "RotoHint":
                    self.roto_text.setText(hint.get("text", ""))
                    self.roto_size.setValue(hint.get("max_size",512))
                    self.roto_chunk.setValue(hint.get("chunk_frames",4))
                    self.seed_path = hint.get("seed_path", "") if Path(hint.get("seed_path", "")).is_file() else ""
                    self.seed_source, self.seed_frame = hint.get("seed_source", ""), hint.get("seed_frame",1)
                    self._mask_input_folder = hint.get('mask_folder','')
                    self.trimap_width.setValue(hint.get('trimap_width',24))
                    self._trimap_path = hint.get('trimap_path','') if Path(hint.get('trimap_path','')).is_file() else ''
                    self._trimap_source,self._trimap_frame = metadata.get('source_video') or metadata.get('source_image'),hint.get('trimap_frame',1)
                    if settings.get("model") in ROTO_MODELS:
                        self.model_picker.setCurrentText(settings["model"])
                    continue
                if hint.get("type") == "PointHint":
                    self.points.append(PointHint(float(hint["x"]), float(hint["y"]), bool(hint.get("positive", True))))
                elif hint.get("type") == "BoxHint":
                    self.boxes.append(BoxHint(*(float(hint[key]) for key in ("left", "top", "right", "bottom"))))
        except (TypeError, KeyError, ValueError):
            self.points.clear()
            self.boxes.clear()
        model = settings.get("model")
        if model in [self.model_picker.itemText(i) for i in range(self.model_picker.count())]:
            self.model_picker.setCurrentText(model)

    def output_for(self, path: str) -> str | None:
        return self.queue_outputs.get(str(Path(path).resolve()))

    def remember_output(self, path: str, output_dir: str) -> None:
        self.queue_outputs[str(Path(path).resolve())] = output_dir
        self.register_result(path,output_dir)
        try:
            pending = self.results_index_path.with_suffix(".json.tmp")
            pending.write_text(json.dumps(self.queue_outputs, indent=2), encoding="utf-8")
            pending.replace(self.results_index_path)
        except OSError as error:
            self.status.setToolTip(f"Result links are available this session, but could not be saved: {error}")

    def review_queue_item(self, item: QListWidgetItem) -> None:
        if self._active_worker is not None:
            return
        path = item.text()
        self._load_media(path, self.output_for(path), restore_guidance=True)

    def clear_queue(self) -> None:
        if self._active_worker is not None:
            return
        self.queue_list.clear()
        self.status.setText("Queue cleared. Saved results are still on disk.")

    def load_video_matte_folder(self) -> None:
        if self._active_worker is not None or not self.video_path or not self.video_info:
            return
        selected = QFileDialog.getExistingDirectory(
            self, "Choose saved matte frames for this video", self.video_output_dir or str(Path(self.video_path).parent),
        )
        if selected:
            try:
                self.attach_video_matte_folder(selected)
            except (OSError, ValueError, RuntimeError) as error:
                QMessageBox.warning(self, "Could not load matte folder", str(error))

    def attach_video_matte_folder(self, selected: str) -> None:
        if self._active_worker is not None or not self.video_path or not self.video_info:
            return
        folder = Path(selected)
        metadata = read_metadata(folder)
        bound = metadata.get("source_video") or metadata.get("source_image")
        if bound and Path(bound).resolve() != Path(self.video_path).resolve():
            raise RuntimeError("This matte folder belongs to a different source. Choose the folder exported for the open video.")
        saved_frames = sorted(int(path.stem[6:]) for path in folder.glob("frame_*.png") if path.stem[6:].isdigit())
        if not saved_frames:
            raise RuntimeError("No saved frame PNGs were found. Choose the video's frames folder containing frame_000001.png (or later numbered frames).")
        if saved_frames[0] < 1 or saved_frames[-1] > self.video_info.frame_count:
            raise RuntimeError("The saved frame numbers are outside the open video's frame range.")
        frame = self.current_frame if self.current_frame in saved_frames else saved_frames[0]
        with Image.open(folder / f"frame_{frame:06d}.png") as image:
            if image.size != self.original.size or "A" not in image.getbands():
                raise RuntimeError("These saved frames must be transparent PNGs matching the video's original dimensions.")
        self.remember_output(self.video_path, str(folder))
        self._load_media(self.video_path, str(folder), frame, self.video_info, restore_guidance=True)

    def load_results(self) -> None:
        if self._active_worker is not None:
            return
        selected = QFileDialog.getExistingDirectory(self, "Choose saved frames folder or queue output folder")
        if not selected:
            return
        folder = Path(selected)
        try:
            metadata = read_metadata(folder)
            source = metadata.get("source_video") or metadata.get("source_image")
            if source:
                self._add_result_item(source, selected)
                return
            manifest = folder / "queue.json"
            if manifest.exists():
                queue = json.loads(manifest.read_text(encoding="utf-8"))
                outputs = queue.get("outputs", {})
                sources = queue.get("sources", [])
                imported = 0
                for path in sources:
                    destination = outputs.get(path)
                    if destination is None:
                        # Legacy exports do not record destinations; use only an
                        # unambiguous, exact filename folder rather than guessing.
                        stem = Path(path).stem
                        if sum(Path(other).stem == stem for other in sources) == 1 and (folder / stem / "frame_000001.png").exists():
                            destination = str(folder / stem)
                    if destination and Path(destination).is_dir():
                        self._add_result_item(path, destination, load=False)
                        imported += 1
                if not imported:
                    raise RuntimeError("No matching saved results were found. Select a per-item frames folder with its source selected in the queue.")
                self.status.setText(f"Loaded result links for {imported} queue item(s). Click an item to review it.")
                return
            item = self.queue_list.currentItem()
            if self.video_path and any(folder.glob("frame_*.png")):
                self.attach_video_matte_folder(selected)
                return
            if item and any(folder.glob("frame_*.png")):
                self._add_result_item(item.text(), selected)
                return
            raise RuntimeError("Select the source item in the queue first, then choose its saved frames folder.")
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.warning(self, "Could not load results", str(error))

    def _add_result_item(self, source: str, output_dir: str, load: bool = True) -> None:
        if not any(self.queue_list.item(i).text() == source for i in range(self.queue_list.count())):
            item = QListWidgetItem(source)
            item.setToolTip(source)
            self.queue_list.addItem(item)
        self.remember_output(source, output_dir)
        if load:
            self._load_media(source, output_dir, restore_guidance=True)

    def _refresh_canvas(self) -> None:
        # View controls emit while the window is being assembled.
        if not hasattr(self, "canvas"):
            return
        self._update_edit_controls()
        if self.original is None:
            self.canvas.set_image(None)
            return
        view = self.current_view()
        # Keep live brush previews responsive for large artwork; edits and exports
        # always use the untouched original dimensions.
        scale = min(1.0, 1600 * self.canvas._zoom / max(self.original.size))
        preview_size = (max(1, round(self.original.width * scale)), max(1, round(self.original.height * scale)))
        if self._preview_source is not self.original or self._original_preview.size != preview_size:
            self._preview_source = self.original
            self._original_preview = self.original if scale == 1 else self.original.resize(preview_size, Image.Resampling.BILINEAR)
        original_preview = self._original_preview
        alpha_preview = self.alpha if self.alpha is None or scale == 1 else self.alpha.resize(preview_size, Image.Resampling.BILINEAR)
        if view == "matte" and self.alpha:
            self.canvas.set_image(alpha_preview.convert("RGBA"), self.original.size)
        elif view == "cutout" and self.alpha:
            result = checkerboard(preview_size).convert("RGBA")
            foreground = original_preview.convert("RGBA")
            foreground.putalpha(alpha_preview)
            result.alpha_composite(foreground)
            self.canvas.set_image(result, self.original.size)
        else:
            self.canvas.set_image(original_preview, self.original.size)
        self.canvas.set_guidance(self.points, self.boxes)

    def _update_edit_controls(self) -> None:
        for control in (self.fit_button, self.zoom_out_button, self.zoom_percent, self.zoom_in_button):
            control.setEnabled(self.original is not None)
        view = self.current_view()
        self.guidance_panel.setVisible(view == "original" and self.model_picker.currentText() not in REFINERS)
        self.paint_panel.setVisible(view == "matte")
        selected = self.guidance_picker.currentData()
        self.guidance_picker.clear()
        for i, point in enumerate(self.points):
            self.guidance_picker.addItem(f"Point {'+' if point.positive else '−'} {i + 1} ({point.x:.2f}, {point.y:.2f})", ("point", i))
        for i, box in enumerate(self.boxes):
            self.guidance_picker.addItem(f"Box {i + 1} ({box.left:.2f}, {box.top:.2f})", ("box", i))
        idx = self.guidance_picker.findData(selected)
        if idx >= 0:
            self.guidance_picker.setCurrentIndex(idx)
        guidance_enabled = view == "original" and self.model_picker.currentText() not in REFINERS and self.original is not None and self._active_worker is None
        for control in (self.point_button, self.negative_button, self.box_button, self.clear_button):
            control.setEnabled(guidance_enabled)
        self.guidance_picker.setEnabled(guidance_enabled and bool(self.points or self.boxes))
        self.remove_guidance_button.setEnabled(guidance_enabled and bool(self.points or self.boxes))
        editable = view == "matte" and self.alpha is not None and self.original is not None and self._active_worker is None
        self.white_brush_button.setEnabled(editable)
        self.black_brush_button.setEnabled(editable)
        self.brush_size.setEnabled(editable)
        can_undo = editable and bool(self.matte_editor.undo_strokes)
        can_redo = editable and bool(self.matte_editor.redo_strokes)
        self.undo_brush_button.setEnabled(can_undo)
        self.redo_brush_button.setEnabled(can_redo)
        self.undo_brush_action.setEnabled(can_undo)
        self.redo_brush_action.setEnabled(can_redo)
        self.canvas.set_painting_enabled(editable)
        self.update_workflow()
        if guidance_enabled:
            mode = "box" if self.box_button.isChecked() else "negative" if self.negative_button.isChecked() else "positive"
        elif editable:
            mode = "paint_black" if self.black_brush_button.isChecked() else "paint_white"
        else:
            mode = "view"
        self.canvas.set_mode(mode)
        self._update_roto_controls()

    def _reset_matte_edits(self) -> None:
        self._matte_dirty = False
        self.canvas.finish_stroke()
        self._paint_refresh.stop()
        self.matte_editor.reset(self.alpha)
        self.point_button.setChecked(True)
        self.white_brush_button.setChecked(True)
        self._update_edit_controls()

    def _select_brush(self, mode: str) -> None:
        if self.alpha is None or self._active_worker is not None or self.current_view() != "matte":
            return
        self.canvas.set_mode(mode)
        self._update_roto_controls()
        self.status.setText("Paint white to keep pixels or black to remove them. Export PNG uses your edited matte.")
        self._refresh_canvas()

    def paint_matte(self, x1: float, y1: float, x2: float, y2: float, value: int) -> None:
        if self.alpha is None or self._active_worker is not None or self.current_view() != "matte":
            return
        self.matte_editor.paint((x1, y1), (x2, y2), self.brush_size.value(), value)
        if not self._paint_refresh.isActive():
            self._paint_refresh.start()

    def finish_matte_stroke(self) -> None:
        self._matte_dirty = True
        self.matte_editor.end_stroke()
        self._paint_refresh.stop()
        self._refresh_canvas()

    def undo_brush(self) -> None:
        if self._active_worker is not None:
            return
        self.canvas.finish_stroke()
        if self.matte_editor.undo_strokes: self._matte_dirty = True
        self.matte_editor.undo()
        self._refresh_canvas()

    def redo_brush(self) -> None:
        if self._active_worker is not None:
            return
        self.canvas.finish_stroke()
        if self.matte_editor.redo_strokes: self._matte_dirty = True
        self.matte_editor.redo()
        self._refresh_canvas()

    def open_image(self) -> None:
        if self._active_worker is not None:
            return
        self._navigation_timer.stop()
        self._requested_frame = None
        self._update_navigation()
        filename, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.webp *.tif *.tiff)")
        if not filename:
            return
        try:
            self.original = Image.open(filename).convert("RGBA")
            self.video_path = None
            self.current_media_path = filename
            self.video_info = None
            self.current_frame = 1
            self.video_output_dir = None
            self.video_test_active = False
            self.alpha = None
            self._reset_matte_edits()
            self.clear_guidance()
            self.status.setText(f"Loaded {Path(filename).name} — {self.original.width} × {self.original.height}")
            self._refresh_canvas()
            self._update_navigation()
        except Exception as error:
            QMessageBox.critical(self, "Could not open image", str(error))

    def open_video(self) -> None:
        if self._active_worker is not None:
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Videos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v)")
        if filename:
            self._load_media(filename, self.output_for(filename), restore_guidance=True)

    def add_point(self, x: float, y: float, positive: bool) -> None:
        if self.current_view() != "original" or self._active_worker is not None:
            return
        self.points.append(PointHint(x, y, positive))
        self.save_file_guidance()
        kind = "foreground" if positive else "negative"
        self.status.setText(f"Added {kind} point ({len(self.points)} total). Regenerate when ready.")
        self._refresh_canvas()

    def add_box(self, left: float, top: float, right: float, bottom: float) -> None:
        if self.current_view() != "original" or self._active_worker is not None:
            return
        self.boxes.append(BoxHint(left, top, right, bottom))
        self.save_file_guidance()
        self.status.setText(f"Added box ({len(self.boxes)} total). Regenerate when ready.")
        self._refresh_canvas()

    def remove_selected_guidance(self) -> None:
        item = self.guidance_picker.currentData()
        if item:
            self.remove_guidance(*item)

    def remove_guidance(self, kind: str, index: int) -> None:
        if self.current_view() != "original" or self._active_worker is not None:
            return
        items = self.points if kind == "point" else self.boxes
        if 0 <= index < len(items):
            del items[index]
            self.save_file_guidance()
            self.status.setText("Guidance element removed. Regenerate or test the selected frame when ready.")
            self._refresh_canvas()

    def clear_guidance(self) -> None:
        if self._active_worker is not None: return
        self.points.clear()
        self.boxes.clear()
        self.save_file_guidance(force=True)
        self._refresh_canvas()

    def add_queue_files(self) -> None:
        if self._active_worker is not None:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add images or videos to queue", "",
            "Media (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.mp4 *.mov *.mkv *.avi *.webm *.m4v)",
        )
        existing = {self.queue_list.item(index).text() for index in range(self.queue_list.count())}
        for filename in files:
            if filename not in existing:
                item = QListWidgetItem(filename)
                item.setToolTip(filename)
                self.queue_list.addItem(item)
                existing.add(filename)
        self.update_guidance_tooltips()
        self.status.setText(f"Queue has {self.queue_list.count()} item(s).")

    def remove_queue_item(self) -> None:
        if self._active_worker is not None:
            return
        row = self.queue_list.currentRow()
        if row >= 0:
            self.queue_list.takeItem(row)
        self.status.setText(f"Queue has {self.queue_list.count()} item(s).")

    def run_queue(self) -> None:
        if self._active_worker is not None:
            return
        if not self.queue_list.count():
            QMessageBox.information(self, "Queue is empty", "Add image or video files to the queue first.")
            return
        if self.queue_worker and self.queue_worker.isRunning():
            return
        paths = [self.queue_list.item(index).text() for index in range(self.queue_list.count())]
        guidance_by_path = self.queue_guidance_snapshot(paths)
        if guidance_by_path is None: return
        output_root = QFileDialog.getExistingDirectory(self, "Choose output folder for queue")
        if not output_root:
            return
        paths = [self.queue_list.item(index).text() for index in range(self.queue_list.count())]
        self.queue_run_button.setEnabled(False)
        self.status.setText("Starting queue…")
        self.queue_worker = QueueWorker(paths, output_root, self.model_picker.currentText(), [], guidance_by_path)
        self.queue_worker.item_output.connect(self.remember_output)
        self.queue_worker.progress.connect(self.on_queue_progress)
        self.queue_worker.completed.connect(self.on_queue_ready)
        self.queue_worker.failed.connect(self.on_matte_failed)
        self._start_worker(self.queue_worker)

    def on_queue_progress(self, current: int, total: int, name: str) -> None:
        if self._cancel_requested:
            return
        self.queue_list.setCurrentRow(current - 1)
        self.status.setText(f"Queue: {current} / {total} — {name}")

    def on_queue_ready(self, output_root: str, item_count: int) -> None:
        if self._cancel_requested:
            return
        self._processed_frame_tick(update_status=False)
        self.status.setText(f"Queue finished — {item_count} item(s) saved. Click a queue item to review its matte. Output: {output_root}")

    def generate(self) -> None:
        if self._active_worker is not None:
            return
        if self.original is None:
            QMessageBox.information(self, "Open an image", "Choose an illustration before generating a matte.")
            return
        if self.worker and self.worker.isRunning():
            return
        if self.video_worker and self.video_worker.isRunning():
            return
        if self.queue_worker and self.queue_worker.isRunning():
            return
        if self.video_path:
            self.generate_video()
            return
        if not self._validate_roto(self.current_frame):
            return
        self.generate_button.setEnabled(False)
        self.status.setText("Loading local model and generating matte…")
        self.worker = InferenceWorker(self.model_picker.currentText(), self.original, self.job_guidance())
        self.worker.completed.connect(self.on_matte_ready)
        self.worker.failed.connect(self.on_matte_failed)
        self._start_worker(self.worker)

    def test_current_frame(self) -> None:
        if self.video_path:
            self.test_video_frame()
            return
        if self._active_worker is not None or self.original is None or not self._validate_roto(self.current_frame): return
        self.worker = InferenceWorker(self.model_picker.currentText(),self.original,self.job_guidance())
        self.worker.completed.connect(self.on_video_test_ready)
        self.worker.failed.connect(self.on_matte_failed)
        self._start_worker(self.worker)

    def test_video_frame(self) -> None:
        if self._active_worker is not None:
            return
        if self.video_info and self.frame_spin.value() != self.current_frame:
            self._pending_frame_action = "test"
            self.navigate_to_frame()
            return
        if self.original is None or not self.video_path:
            QMessageBox.information(self, "Open a video", "Open a video and select a frame to test.")
            return
        if self.worker and self.worker.isRunning():
            return
        if self.video_worker and self.video_worker.isRunning():
            return
        if self.queue_worker and self.queue_worker.isRunning():
            return
        if self.model_picker.currentText().startswith("LiteSDMatte") and not (self.points or self.boxes):
            QMessageBox.information(self, "Guidance needed", "LiteSDMatte needs a Point + or Box before testing.")
            return
        if not self._validate_roto(self.current_frame):
            return
        self.test_frame_button.setEnabled(False)
        self.status.setText(f"Testing video frame {self.current_frame} — nothing will be saved…")
        self.worker = InferenceWorker(self.model_picker.currentText(), self.original, self.job_guidance())
        self.worker.completed.connect(self.on_video_test_ready)
        self.worker.failed.connect(self.on_matte_failed)
        self._start_worker(self.worker)

    def generate_video(self) -> None:
        if self._active_worker is not None:
            return
        if self.original is None or not self.video_path:
            return
        if self.model_picker.currentText() in REFINERS:
            start = self.range_start.value() if self.range_picker.currentIndex()==2 else self.current_frame if self.range_picker.currentIndex()==1 else 1
            self._start_video_export(self.new_refinement_folder(self._mask_input_folder),start)
            return
        default_dir = str(Path(self.video_path).with_suffix("")) + "_frames"
        output_dir = QFileDialog.getExistingDirectory(self, "Choose folder for transparent PNG frames", default_dir)
        if not output_dir:
            return
        start = self.range_start.value() if self.range_picker.currentIndex()==2 else self.current_frame if self.range_picker.currentIndex()==1 else 1
        self._start_video_export(output_dir, start)

    def resume_video(self) -> None:
        if self._active_worker is not None or not self.video_path or not self.video_info:
            return
        self._navigation_timer.stop()
        # An uncommitted slider change should first load the requested source frame,
        # so guidance cannot accidentally be applied to a different displayed frame.
        if self.frame_spin.value() != self.current_frame:
            self._pending_frame_action = "resume"
            self.navigate_to_frame()
            return
        output_dir = self.new_refinement_folder(self._mask_input_folder) if self.model_picker.currentText() in REFINERS else self.video_output_dir
        if not output_dir:
            output_dir = QFileDialog.getExistingDirectory(self, "Choose output folder for resumed video frames")
        if output_dir:
            self._start_video_export(output_dir, self.current_frame)

    def _start_video_export(self, output_dir: str, start_frame: int) -> None:
        end_frame = self.range_end.value() if self.range_picker.currentIndex()==2 else None
        if end_frame is not None and end_frame < start_frame:
            QMessageBox.information(self,"Invalid range","The last frame must be at or after the first frame.")
            return
        count = (end_frame or self.video_info.frame_count)-start_frame+1
        if self.model_picker.currentText() in REFINERS and not self.prepare_refinement(start_frame,count): return
        if not self._validate_roto(start_frame):
            return
        if self.model_picker.currentText().startswith("LiteSDMatte") and not (self.points or self.boxes):
            QMessageBox.information(self, "Guidance needed", "LiteSDMatte needs a Point + or Box before exporting.")
            return
        self.video_output_dir = output_dir
        self.remember_output(self.video_path, output_dir)
        self.video_test_active = False
        self.status.setText(f"Generating transparent PNG frames from frame {start_frame}…")
        self.video_worker = VideoInferenceWorker(self.video_path, output_dir, self.model_picker.currentText(), self.job_guidance(), start_frame, self.video_info, end_frame)
        self.video_worker.progress.connect(self.on_video_progress)
        self.video_worker.completed.connect(self.on_video_ready)
        self.video_worker.failed.connect(self.on_matte_failed)
        self._start_worker(self.video_worker)

    def on_video_progress(self, current: int, total: int) -> None:
        if self._cancel_requested:
            return
        self.status.setText(f"{'Refining edges' if self.model_picker.currentText() in REFINERS else 'Creating masks' if self.model_picker.currentText().endswith('— Tracking') else 'Generating matte'}: {current} / {total}")

    def on_video_ready(self, output_dir: str, frame_count: int) -> None:
        if self._cancel_requested:
            return
        self._processed_frame_tick(update_status=False)
        self.video_output_dir = output_dir
        if self.video_info:
            self.video_info = VideoInfo(frame_count, self.video_info.fps)
        self.register_result(self.video_path,output_dir)
        self.refresh_results()
        self._reload_after_worker = True
        self.status.setText(f"Video export finished in {output_dir}. Use the frame controls to review the results.")

    def on_video_test_ready(self, alpha: Image.Image) -> None:
        if self._cancel_requested:
            return
        # Replacing this value deletes the previous test matte; it is never written to disk.
        self.alpha = alpha
        self._reset_matte_edits()
        self.video_test_active = True
        if not self.video_path:
            self.prepare_image_save()
        self.status.setText(f"Temporary test for frame {self.current_frame} ready — no file was saved. Export PNG to save this frame, or resume video matting with this guidance." if self.video_path else "Temporary image matte ready. Regenerate or paint as needed, then Save to image to keep it.")
        self.view_tabs.setCurrentIndex(2)
        self._refresh_canvas()

    def on_matte_ready(self, alpha: Image.Image) -> None:
        if self._cancel_requested:
            return
        self.alpha = alpha
        self._reset_matte_edits()
        if not self.video_path:
            self.prepare_image_save()
        self.status.setText("Matte ready — regenerate or paint as needed, then Save to image to update its saved result." if not self.video_path else "Result ready — review the edges, save any paint edits, then refine or export.")
        self.view_tabs.setCurrentIndex(2)
        self._refresh_canvas()

    def on_matte_failed(self, details: str) -> None:
        if self._cancel_requested:
            return
        self.status.setText("Matte generation failed.")
        QMessageBox.critical(self, "Matte generation failed", details)

    def _require_alpha(self) -> bool:
        if self.alpha is not None and self.original is not None:
            return True
        QMessageBox.information(self, "No matte yet", "Generate a matte before exporting.")
        return False

    def export_cutout(self) -> None:
        if not self._require_alpha():
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export transparent PNG", "cutout.png", "PNG image (*.png)")
        if filename:
            output = self.original.convert("RGBA")
            output.putalpha(self.alpha)
            output.save(filename, "PNG", compress_level=1)
            self.status.setText(f"Exported original-resolution transparent PNG: {Path(filename).name}")

    def export_matte(self) -> None:
        if not self._require_alpha():
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Export matte", "matte.png", "PNG image (*.png)")
        if filename:
            self.alpha.save(filename, "PNG", compress_level=1)
            self.status.setText(f"Exported 8-bit alpha matte: {Path(filename).name}")
