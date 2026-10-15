from __future__ import annotations

import io
import json
import shutil
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

from PIL import Image

from .cancellation import Cancellation
from .models import Guidance


@dataclass(frozen=True)
class VideoInfo:
    frame_count: int
    fps: float


def read_metadata(folder: Path | None) -> dict:
    if folder:
        try:
            return json.loads((folder / "frames.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def write_metadata(folder: Path, data: dict) -> None:
    pending = folder / "frames.json.tmp"
    pending.write_text(json.dumps(data, indent=2), encoding="utf-8")
    pending.replace(folder / "frames.json")


def save_frame(image: Image.Image, path: Path) -> None:
    pending = path.with_suffix(".png.tmp")
    try:
        image.save(pending, "PNG", compress_level=1)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def probe_video(path: str, cancellation: Cancellation, known_count: int | None = None) -> VideoInfo:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe was not found. Install the complete FFmpeg package.")
    command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames,nb_read_frames,avg_frame_rate", "-of", "json", path]
    data = json.loads(cancellation.run_process(command, capture_output=True))
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream was found.")
    stream = streams[0]
    if not known_count and not str(stream.get("nb_frames", "")).isdigit():
        command.insert(1, "-count_frames")
        stream = json.loads(cancellation.run_process(command, capture_output=True))["streams"][0]
    count = known_count or int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    if count < 1:
        raise RuntimeError("Could not count the video frames.")
    try:
        fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    return VideoInfo(count, fps)


def load_frame(path: str, frame: int, cancellation: Cancellation) -> Image.Image:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found.")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path]
    if frame > 1:
        command += ["-vf", f"select=eq(n\\,{frame - 1})"]
    command += ["-vsync", "0", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    content = cancellation.run_process(command, capture_output=True)
    if not content:
        raise RuntimeError(f"Frame {frame} could not be decoded.")
    with Image.open(io.BytesIO(content)) as image:
        return image.convert("RGBA")


def process_video(
    path: str, folder: Path, model, model_name: str, guidance: list[Guidance],
    cancellation: Cancellation, start_frame: int = 1,
    progress: Callable[[int, int], None] | None = None,
    frame_ready: Callable[[int, int, Image.Image], None] | None = None,
    end_frame: int | None = None,
) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found. Install FFmpeg and restart Matte App.")
    if start_frame < 1:
        raise ValueError("Start frame must be at least 1.")
    if end_frame is not None and end_frame < start_frame:
        raise ValueError('The last frame must be at or after the first frame.')
    folder.mkdir(parents=True, exist_ok=True)
    options = next((hint for hint in guidance if getattr(hint,'mask_folder','')),None)
    if options and Path(options.mask_folder).resolve() == folder.resolve():
        raise ValueError('Refinement must use a different output folder; input masks are preserved.')
    metadata = read_metadata(folder)
    bound_source = metadata.get("source_video") or metadata.get("source_image")
    if bound_source and Path(bound_source).resolve() != Path(path).resolve():
        raise RuntimeError("This output folder belongs to a different video. Choose a new folder.")
    if metadata and "runs" not in metadata:
        metadata["runs"] = [{"start_frame": 1, "last_frame": metadata.get("frame_count", 0),
                             "model": metadata.get("model"), "guidance": metadata.get("guidance", []), "status": "complete"}]
    runs = metadata.setdefault("runs", [])
    run = {"start_frame": start_frame, "last_frame": start_frame - 1, "model": model_name,
           "guidance": [hint.__dict__ | {"type": type(hint).__name__} for hint in guidance], "status": "running"}
    runs.append(run)
    metadata.update({"source_video": path, "model": model_name, "guidance": run["guidance"], "status": "running"})
    if options:
        metadata.update({'source_masks': options.mask_folder, 'stage': 'refined'})
    try:
        with tempfile.TemporaryDirectory(prefix="matte-app-video-") as raw_dir:
            pattern = str(Path(raw_dir) / "source_%06d.png")
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", path]
            if end_frame is not None:
                command += ['-vf',f'select=between(n\\,{start_frame-1}\\,{end_frame-1})']
            elif start_frame > 1:
                command += ["-vf", f"select=gte(n\\,{start_frame - 1})"]
            command += ["-vsync", "0", "-start_number", str(start_frame), pattern]
            cancellation.run_process(command)
            frames = sorted(Path(raw_dir).glob("source_*.png"))
            if not frames:
                raise RuntimeError(f"No frames decoded at or after frame {start_frame}.")
            total = probe_video(path,cancellation).frame_count if end_frame is not None else start_frame - 1 + len(frames)
            metadata["frame_count"] = total
            write_metadata(folder, metadata)
            def consume(offset, alpha):
                index = start_frame + offset
                with Image.open(frames[offset]) as raw:
                    source = raw.convert("RGBA")
                if alpha.size != source.size:
                    raise ValueError("Matte dimensions do not match the source frame.")
                source.putalpha(alpha)
                save_frame(source, folder / f"frame_{index:06d}.png")
                run["last_frame"] = index
                write_metadata(folder, metadata)
                if frame_ready:
                    frame_ready(index, total, source)
                if progress:
                    progress(index, total)
            if hasattr(model, "predict_sequence"):
                model.predict_sequence(frames, guidance, cancellation, start_frame, consume, path)
            else:
                for offset, frame_path in enumerate(frames):
                    cancellation.check()
                    with Image.open(frame_path) as raw:
                        source = raw.convert("RGBA")
                    alpha = model.predict(source, guidance, cancellation)
                    cancellation.check()
                    consume(offset, alpha)
            cancellation.check()
            run["status"] = "complete"
            # An export started partway through a new video is intentionally partial.
            metadata["status"] = "complete" if all((folder / f"frame_{i:06d}.png").exists() for i in range(1, total + 1)) else "partial"
            return total
    finally:
        if run["status"] == "running":
            run["status"] = "cancelled" if cancellation.requested else "failed"
            metadata["status"] = "partial"
        write_metadata(folder, metadata)
