from __future__ import annotations

import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image

from .cancellation import Cancellation
from .alpha_collector import AlphaCollector

REFINERS = ["ViTMatte — Saved masks", "MatAnyone — Saved masks", "MatAnyone 2 — Saved masks", "VideoMaMa — Saved masks"]
ENABLE_WARM_SESSIONS=False
ROTO_MODELS = ["SAM 2 — Tracking", "SAM 3 — Tracking", "SAM 2 + MatAnyone", "SAM 3 + MatAnyone", "SAM 2 + VideoMaMa", "SAM 3 + VideoMaMa", "SAM 2 + ViTMatte", "SAM 3 + ViTMatte", *REFINERS]
ROTO_MODELS += ['SAM2Matting Tiny','SAM2Matting Base+','SAM 3.1 — Tracking']


def capabilities(name: str) -> tuple[str, str]:
    if name not in ROTO_MODELS:
        raise ValueError(f"Unknown roto model: {name}")
    if name.startswith('SAM2Matting'):return ('sam2matting_base' if 'Base+' in name else 'sam2matting_tiny','tracking')
    if name.startswith('SAM 3.1'):return ('sam31','tracking')
    if name in REFINERS:
        if name.startswith('MatAnyone 2'):return ('none','matanyone2')
        return ('none', 'vitmatte' if name.startswith('ViTMatte') else 'matanyone' if name.startswith('MatAnyone') else 'videomama')
    return ("sam3" if name.startswith("SAM 3") else "sam2", "vitmatte" if "ViTMatte" in name else "matanyone" if "MatAnyone" in name else "videomama" if "VideoMaMa" in name else "tracking")


def readiness(name: str, root: Path) -> str:
    tracker, matte = capabilities(name)
    if not (root / '.roto-venv/Scripts/python.exe').exists():
        return "Roto setup needed: run setup_roto.ps1."
    if tracker.startswith('sam2matting'):
        weight='SAM2Matting-SAM2.1Base+.pt' if tracker.endswith('base') else 'SAM2Matting-SAM2.1Tiny.pt'
        return 'Ready locally. Direct soft-alpha image/video matting; non-commercial research license.' if (root/'models/SAM2Matting/checkpoints'/weight).is_file() and (root/'third_party/SAM2Matting/sam2').is_dir() else 'SAM2Matting setup is needed.'
    if tracker=='sam31':return 'Ready locally. SAM 3.1 multiplex binary tracking with points, boxes, or text; mask seeds unsupported.' if (root/'models/SAM3.1/sam3.1_multiplex.pt').is_file() and (root/'third_party/sam3/sam3').is_dir() else 'SAM 3.1 code/checkpoint setup is needed.'
    if tracker == 'sam3' and not (root / 'models/SAM3/model.safetensors').exists():
        return "SAM 3 needs approved checkpoint access. Sign in locally, then choose Get SAM 3 weights."
    if matte=='matanyone2':
        return 'Ready locally. MatAnyone 2 human video matting; non-commercial S-Lab license.' if all((root/p).is_file() for p in ('models/MatAnyone2/config.json','models/MatAnyone2/model.safetensors','third_party/MatAnyone2/matanyone2/__init__.py')) else 'MatAnyone 2 source/checkpoint setup is needed.'
    required = [] if tracker == 'none' else ['SAM3' if tracker == 'sam3' else 'SAM2']
    if matte == 'vitmatte': required.append('ViTMatte')
    if matte == 'matanyone': required.append('MatAnyone')
    if matte == 'videomama': required.extend(['VideoMaMa/unet', 'SVD/vae'])
    if any(not (root / 'models' / entry).is_dir() for entry in required):
        return "Checkpoint setup is incomplete. Run setup_roto.ps1 with the selected model."
    return "Ready locally. " + ({'tracking': 'Tracked binary masks for objects and hard edges.', 'matanyone': 'Uses the starting mask and follows the subject; designed primarily for people.', 'videomama': 'Uses masks throughout the clip; slower, processed in overlapping sections.', 'vitmatte': 'Refines each frame from a mask/trimap; no temporal tracking.'}[matte])


class RotoCommandModel:
    def __init__(self, name: str):
        self.name = name
        self.root = Path(__file__).resolve().parents[1]

    def _run(self, frames: list[Path], guidance, token: Cancellation, start_frame: int, consume, source: str = '') -> None:
        from .models import RotoHint, _MODEL_CACHE
        tracker, matte = capabilities(self.name)
        # Return GPU memory held by a previously selected automatic model.
        _MODEL_CACHE.clear()
        gc.collect()
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_available():
            sys.modules['torch'].cuda.empty_cache()
        python = self.root / '.roto-venv/Scripts/python.exe'
        if not python.exists():
            raise RuntimeError(readiness(self.name, self.root))
        options = next((h for h in guidance if isinstance(h, RotoHint)), RotoHint())
        mask_paths = []
        if tracker == 'none':
            from .saved_masks import mask_files
            if options.mask_folder:
                mask_paths = mask_files(options.mask_folder, source or options.mask_source, start_frame, len(frames), matte in ('matanyone','matanyone2'))
            elif options.seed_path and (len(frames) == 1 or matte in ('matanyone','matanyone2')):
                mask_paths = [Path(options.seed_path)]
            else:
                raise ValueError('Choose saved masks before refining. Video refinement needs masks for the requested frames.')
            for i, path in enumerate(mask_paths):
                from .saved_masks import read_mask
                with Image.open(frames[i]) as original:
                    if read_mask(path).size != original.size:
                        raise ValueError(f'Input mask dimensions do not match frame {start_frame+i}.')
        if options.seed_path and not (tracker == 'none' and options.mask_folder) and source and (Path(options.seed_source).resolve() != Path(source).resolve() or options.seed_frame != start_frame):
            raise ValueError("The starting mask belongs to a different source/frame. Use Resume from that frame or clear the starting mask.")
        with tempfile.TemporaryDirectory(prefix='matte-roto-', ignore_cleanup_errors=True) as temporary:
            work = Path(temporary)
            job = {'root': str(self.root), 'tracker': tracker, 'matte': matte, 'frames': [str(p) for p in frames], 'output': str(work),
                   'guidance': [h.__dict__ | {'type': type(h).__name__} for h in guidance], 'start_frame': start_frame,
                   'mask_paths': [str(p) for p in mask_paths]}
            request = work / 'job.json'
            request.write_text(json.dumps(job), encoding='utf-8')
            collect = AlphaCollector(work, consume, token)
            try:
                from .model_session import SESSION,stop_session
                if matte=='tracking' and ENABLE_WARM_SESSIONS:SESSION.run(python,self.root/'matte_app/roto_runner.py',request,token,collect)
                else:
                    stop_session()
                    token.run_process([str(python), str(self.root / 'matte_app/roto_runner.py'), '--job', str(request)], capture_output=True, on_poll=collect)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(error.stderr or "Roto inference failed") from error
            finally:
                if sys.exc_info()[0] is None and not token.requested:
                    collect.finish()
                else:
                    # Retain readable outputs without replacing cancellation or
                    # the original inference/export failure with a flush error.
                    try:
                        collect()
                    except Exception:
                        pass
            token.check()
            if collect.delivered != len(frames):
                raise RuntimeError(f"Roto returned {collect.delivered} mattes for {len(frames)} frames.")

    def predict_sequence(self, frames, guidance, cancellation, start_frame, consume, source=''):
        self._run(list(frames), guidance, cancellation, start_frame, consume, source)

    def predict(self, image, guidance, cancellation=None):
        token = cancellation or Cancellation()
        token.check()
        result = []
        with tempfile.TemporaryDirectory(prefix='matte-roto-image-') as temporary:
            path = Path(temporary) / 'input.png'
            image.save(path, 'PNG', compress_level=1)
            from .models import RotoHint
            options = next((h for h in guidance if isinstance(h, RotoHint)), RotoHint())
            frame = options.mask_frame if self.name in REFINERS else options.seed_frame if options.seed_path else 1
            self._run([path], guidance, token, frame, lambda i, alpha: result.append(alpha))
        return result[0].resize(image.size, Image.Resampling.LANCZOS)
