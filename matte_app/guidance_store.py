"""Persistent, file-specific guidance, separate from generated matte outputs."""
import json
import os
from pathlib import Path
from dataclasses import fields
from .models import PointHint, BoxHint, RotoHint

def source_key(path):
    return os.path.normcase(str(Path(path).resolve()))

def encode(hints):
    return [hint.__dict__ | {'type': type(hint).__name__} for hint in hints]

def decode(data):
    classes = {'PointHint':PointHint,'BoxHint':BoxHint,'RotoHint':RotoHint}
    result = []
    for item in data:
        cls = classes.get(item.get('type'))
        if cls:
            allowed = {field.name for field in fields(cls)}
            result.append(cls(**{key:value for key,value in item.items() if key in allowed}))
    return result

_CACHE: dict[str, tuple[int, dict]] = {}

class GuidanceStore:
    def __init__(self,path):
        self.path = Path(path)

    def read(self):
        resolved = source_key(self.path)
        try:
            mtime = self.path.stat().st_mtime_ns
        except OSError:
            _CACHE.pop(resolved, None)
            return {}
        cached = _CACHE.get(resolved)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        try:
            records = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(records, dict):
                records = {}
        except (OSError, ValueError):
            records = {}
        _CACHE[resolved] = (mtime, records)
        return records

    def get(self,source,frame=1):
        record = self.read().get(source_key(source),{}).get(str(frame))
        return None if record is None else decode(record)

    def save(self,source,frame,hints):
        key = source_key(source)
        frame_str = str(frame)
        encoded = encode(hints)
        records = self.read()
        current = records.get(key, {}).get(frame_str)
        if current == encoded:
            return
        records.setdefault(key,{})[frame_str] = encoded
        root_guidance = (Path(__file__).resolve().parents[1] / 'guidance_by_file.json').resolve()
        if self.path.resolve() == root_guidance and ('appdata' in key.lower() and 'temp' in key.lower()):
            return
        self.write(records)

    def write(self,records):
        resolved = source_key(self.path)
        pending = self.path.with_suffix('.json.tmp')
        pending.write_text(json.dumps(records,indent=2),encoding='utf-8')
        pending.replace(self.path)
        try:
            mtime = self.path.stat().st_mtime_ns
        except OSError:
            mtime = 0
        _CACHE[resolved] = (mtime, records)

    def merge(self,records):
        existing = self.read()
        changed = False
        for source,frames in records.items():
            target = existing.setdefault(source_key(source),{})
            for frame_str, hints in frames.items():
                if target.get(frame_str) != hints:
                    target[frame_str] = hints
                    changed = True
        if changed:
            self.write(existing)

