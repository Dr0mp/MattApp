from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageDraw


@dataclass
class Stroke:
    bounds: tuple[int, int, int, int]
    before: dict[tuple[int,int,int,int],Image.Image]
    after: dict[tuple[int,int,int,int],Image.Image]

    @property
    def bytes(self) -> int:
        return sum(image.width*image.height*2 for image in self.before.values())


class MatteEditor:
    """Paint the original-resolution alpha; retain bounded, per-stroke undo patches."""

    def __init__(self) -> None:
        self.alpha: Image.Image | None = None
        self.undo_strokes: list[Stroke] = []
        self.redo_strokes: list[Stroke] = []
        self._before: dict | None = None
        self._base=None;self._current_tiles={};self.tile_size=256
        self._bounds: tuple[int, int, int, int] | None = None

    def reset(self, alpha: Image.Image | None) -> None:
        self.alpha = alpha
        self._base=alpha.copy() if alpha is not None else None;self._current_tiles={}
        self.undo_strokes.clear()
        self.redo_strokes.clear()
        self._before = None
        self._bounds = None

    def begin_stroke(self) -> None:
        if self.alpha is not None:
            self._before = {}
            self._bounds = None

    def paint(self, start: tuple[float, float], end: tuple[float, float], diameter: int, value: int) -> tuple[int,int,int,int] | None:
        if self.alpha is None or self._before is None:
            return
        width, height = self.alpha.size
        x1, y1 = min(width - 1, max(0, start[0] * width)), min(height - 1, max(0, start[1] * height))
        x2, y2 = min(width - 1, max(0, end[0] * width)), min(height - 1, max(0, end[1] * height))
        diameter = max(1, diameter)
        radius = (diameter - 1) / 2
        bounds = (
            max(0, math.floor(min(x1, x2) - radius - 1)),
            max(0, math.floor(min(y1, y2) - radius - 1)),
            min(width, math.ceil(max(x1, x2) + radius + 2)),
            min(height, math.ceil(max(y1, y2) + radius + 2)),
        )
        changed_bounds=bounds
        # Capture touched tiles before modification. Subdivide long input jumps
        # so a diagonal does not snapshot its entire enclosing rectangle.
        steps=max(1,math.ceil(max(abs(x2-x1),abs(y2-y1))/self.tile_size))
        for n in range(steps):
            ax=x1+(x2-x1)*n/steps;ay=y1+(y2-y1)*n/steps
            bx=x1+(x2-x1)*(n+1)/steps;by=y1+(y2-y1)*(n+1)/steps
            for ty in range(max(0,math.floor(min(ay,by)-radius-1))//self.tile_size,(min(height-1,math.ceil(max(ay,by)+radius+1)))//self.tile_size+1):
                for tx in range(max(0,math.floor(min(ax,bx)-radius-1))//self.tile_size,(min(width-1,math.ceil(max(ax,bx)+radius+1)))//self.tile_size+1):
                    tile=(tx*self.tile_size,ty*self.tile_size,min(width,(tx+1)*self.tile_size),min(height,(ty+1)*self.tile_size))
                    if tile not in self._before:self._before[tile]=self.alpha.crop(tile)
        if self._bounds:
            old = self._bounds
            bounds = (min(old[0], bounds[0]), min(old[1], bounds[1]), max(old[2], bounds[2]), max(old[3], bounds[3]))
        self._bounds = bounds
        draw = ImageDraw.Draw(self.alpha)
        draw.line((x1, y1, x2, y2), fill=value, width=diameter)
        for x, y in ((x1, y1), (x2, y2)):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=value)
        return changed_bounds

    def end_stroke(self) -> None:
        if self.alpha is not None and self._before is not None and self._bounds:
            before={};after={}
            for tile,old in self._before.items():
                current=self.alpha.crop(tile)
                if old.tobytes()!=current.tobytes():before[tile]=old;after[tile]=current
            if before:
                self._current_tiles.update(after)
                self.undo_strokes.append(Stroke(self._bounds,before,after))
                self.redo_strokes.clear()
                # Keep at least the latest stroke, even for a very large image.
                retained_bytes = sum(stroke.bytes for stroke in self.undo_strokes)
                while len(self.undo_strokes) > 1 and (len(self.undo_strokes) > 30 or retained_bytes > 64 * 1024 * 1024):
                    retained_bytes -= self.undo_strokes.pop(0).bytes
        self._before = None
        self._bounds = None

    def undo(self) -> None:
        self.end_stroke()
        if self.alpha is not None and self.undo_strokes:
            stroke = self.undo_strokes.pop()
            for tile,image in stroke.before.items():self.alpha.paste(image,tile)
            self._current_tiles.update(stroke.before)
            self.redo_strokes.append(stroke)

    def redo(self) -> None:
        if self.alpha is not None and self.redo_strokes:
            stroke = self.redo_strokes.pop()
            for tile,image in stroke.after.items():self.alpha.paste(image,tile)
            self._current_tiles.update(stroke.after)
            self.undo_strokes.append(stroke)

    def immutable_snapshot(self):
        """Worker reconstructs this immutable baseline + COW tiles off the UI thread."""
        return self._base,dict(self._current_tiles)
