"""Discover source-bound results in a shared root or one item's folder."""
from pathlib import Path
import json
import os
from PIL import Image
from .result_store import key
from .video import read_metadata


def discover_results(folder,queued=(),selected=None,known=None):
    root=Path(folder).resolve();queued=list(dict.fromkeys(str(Path(p).resolve()) for p in queued))
    candidates=[];errors=[]
    manifest=root/'queue.json'
    if manifest.is_file():
        data=json.loads(manifest.read_text(encoding='utf-8'))
        queued=list(dict.fromkeys(queued+data.get('sources',[])))
        for source,destination in data.get('outputs',{}).items():
            path=Path(destination)
            if not path.is_absolute():path=root/path
            if path.is_dir():candidates.append((source,path))
    for source,destination in (known or {}).items():
        path=Path(destination).resolve()
        if path==root or root in path.parents:candidates.append((source,path))
    for directory,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in ('.drafts','.cache')]
        path=Path(directory)
        frame=next((name for name in files if name.startswith('frame_') and name.endswith('.png') and name[6:-4].isdigit()),None)
        if not frame:continue
        try:
            meta=read_metadata(path);source=meta.get('source_image') or meta.get('source_video')
            if not source:
                matches=[p for p in queued if path.name.casefold() in (Path(p).name.casefold(),Path(p).stem.casefold())]
                if path==root and not matches and selected:matches=[selected]
                if len(matches)!=1:
                    errors.append(f'{path.name}: no unambiguous source match');continue
                source=matches[0]
            destination=path.parent if path.name in ('matte','refined','cutout') and meta.get('stage')==path.name else path
            candidates.append((source,destination))
        except (OSError,ValueError,TypeError) as error:errors.append(f'{path.name}: {error}')
    links={};ambiguous=set()
    for source,destination in candidates:
        try:
            source=Path(source).resolve();destination=Path(destination).resolve()
            if not source.is_file():raise ValueError('original source is missing')
            folders=[destination/stage for stage in ('matte','refined','cutout') if (destination/stage/'frames.json').is_file()] or [destination]
            valid=False
            for stage_folder in folders:
                meta=read_metadata(stage_folder);bound=meta.get('source_image') or meta.get('source_video')
                if bound and key(bound)!=key(source):raise ValueError('result belongs to another source')
                frame=next(stage_folder.glob('frame_*.png'),None)
                if frame is None:continue
                with Image.open(frame) as rgba:
                    if 'A' not in rgba.getbands():raise ValueError('result has no alpha channel')
                    if source.suffix.lower() not in ('.mp4','.mov','.mkv','.avi','.webm','.m4v'):
                        with Image.open(source) as original:
                            if rgba.size!=original.size:raise ValueError('mask dimensions do not match source')
                valid=True
            if valid:
                if str(source) in ambiguous:continue
                previous=links.get(str(source))
                if previous and previous!=str(destination):
                    links.pop(str(source));ambiguous.add(str(source));raise ValueError('multiple saved folders match this source; select one item folder')
                links[str(source)]=str(destination)
        except (OSError,ValueError,TypeError) as error:errors.append(f'{Path(source).name}: {error}')
    return links,list(dict.fromkeys(errors))
