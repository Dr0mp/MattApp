from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QRect, Qt, Signal, QEvent
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget
from PIL import Image

from .models import BoxHint, PointHint
from .brush_telemetry import BrushTelemetry


class ImageCanvas(QWidget):
    point_added = Signal(float, float, bool)
    guidance_removed = Signal(str, int)
    box_added = Signal(float, float, float, float)
    stroke_started = Signal()
    brush_segment = Signal(float, float, float, float, int)
    stroke_finished = Signal()
    zoom_changed = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(300, 240)
        self.setMouseTracking(True)
        self._image: Image.Image | None = None
        self._qimage: QImage | None = None
        self._source_size: tuple[int, int] | None = None
        self._mode = "positive"
        self._points: list[PointHint] = []
        self._boxes: list[BoxHint] = []
        self._drag_start: QPointF | None = None
        self._drag_end: QPointF | None = None
        self._brush_diameter = 40
        self._painting_enabled = False
        self._brush_last: tuple[float, float] | None = None
        self._brush_down = False
        self._hover: QPointF | None = None
        self._zoom = 1.0
        self._pan = QPointF()
        self._pan_last: QPointF | None = None
        self.telemetry=BrushTelemetry()
        self.setToolTip("Mouse wheel: zoom · Middle-button drag: pan · Fit: show the whole image")

    def set_image(self, image: Image.Image | None, source_size: tuple[int, int] | None = None) -> None:
        size = source_size or (image.size if image else None)
        changed = size != self._source_size
        self._image = image
        if image is None:self._qimage=None
        else:
            rgba=image.convert('RGBA')
            self._qimage=QImage(rgba.tobytes('raw','RGBA'),rgba.width,rgba.height,QImage.Format_RGBA8888).copy()
        self._source_size = size
        if changed:
            self.reset_zoom()
        self.update()

    def reset_zoom(self) -> None:
        self.finish_stroke()
        self._drag_start = self._drag_end = None
        self._zoom = 1.0
        self._pan = QPointF()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def update_image_region(self,bounds,patch):
        if self._image is None or self._qimage is None:return
        self._image.paste(patch,bounds)
        rgba=patch.convert('RGBA')
        qpatch=QImage(rgba.tobytes('raw','RGBA'),rgba.width,rgba.height,QImage.Format_RGBA8888)
        painter=QPainter(self._qimage);painter.setCompositionMode(QPainter.CompositionMode_Source);painter.drawImage(bounds[0],bounds[1],qpatch);painter.end()
        rect=self._target_rect();width,height=self._image.size
        dirty=QRectF(rect.left()+bounds[0]*rect.width()/width,rect.top()+bounds[1]*rect.height()/height,(bounds[2]-bounds[0])*rect.width()/width,(bounds[3]-bounds[1])*rect.height()/height)
        self.update(dirty.toAlignedRect().adjusted(-2,-2,2,2))
        self.telemetry.patch()

    def set_zoom(self, zoom: float, anchor: QPointF | None = None) -> None:
        if self._image is None:
            return
        self.finish_stroke()
        self._drag_start = self._drag_end = None
        anchor = anchor if anchor is not None else QPointF(self.width() / 2, self.height() / 2)
        before = self._target_rect()
        x = (anchor.x() - before.left()) / before.width()
        y = (anchor.y() - before.top()) / before.height()
        self._zoom = max(1.0, min(16.0, zoom))
        after = self._target_rect()
        self._pan += anchor - QPointF(after.left() + x * after.width(), after.top() + y * after.height())
        self._clamp_pan()
        self.zoom_changed.emit(self._zoom)
        self.update()

    def _clamp_pan(self) -> None:
        rect = self._target_rect()
        limit_x = max(0.0, (rect.width() - self.width()) / 2)
        limit_y = max(0.0, (rect.height() - self.height()) / 2)
        self._pan = QPointF(max(-limit_x, min(limit_x, self._pan.x())),
                            max(-limit_y, min(limit_y, self._pan.y())))

    def wheelEvent(self, event) -> None:
        if self._image is not None and event.angleDelta().y():
            self.set_zoom(self._zoom * 1.25 ** (event.angleDelta().y() / 120), event.position())
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event) -> None:
        self._clamp_pan()
        super().resizeEvent(event)

    def set_guidance(self, points: list[PointHint], boxes: list[BoxHint]) -> None:
        self._points, self._boxes = points, boxes
        self.update()

    def set_mode(self, mode: str) -> None:
        if mode == self._mode:
            return
        self.finish_stroke()
        self._mode = mode
        self.setCursor(Qt.ArrowCursor if mode == "view" else Qt.CrossCursor)
        self.update()

    def set_brush_diameter(self, diameter: int) -> None:
        self._brush_diameter = diameter
        self.update()

    def set_painting_enabled(self, enabled: bool) -> None:
        if not enabled:
            self.finish_stroke()
        self._painting_enabled = enabled
        self.update()

    def finish_stroke(self) -> None:
        if self._brush_down:
            self._brush_down = False
            self._brush_last = None
            self.stroke_finished.emit()

    def _target_rect(self) -> QRectF:
        if self._image is None:
            return QRectF()
        width, height = self._source_size or self._image.size
        scale = min(self.width() / width, self.height() / height) * self._zoom
        rendered_w, rendered_h = width * scale, height * scale
        return QRectF((self.width() - rendered_w) / 2 + self._pan.x(),
                      (self.height() - rendered_h) / 2 + self._pan.y(), rendered_w, rendered_h)

    def _to_normalized(self, pos: QPointF) -> tuple[float, float] | None:
        rect = self._target_rect()
        if not rect.contains(pos):
            return None
        return ((pos.x() - rect.left()) / rect.width(), (pos.y() - rect.top()) / rect.height())

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        with self.telemetry.span('qt_paint_ms'):self._paint_canvas(event)
        self.telemetry.painted(event.rect().width()*event.rect().height())

    def _paint_canvas(self,event):
        painter = QPainter(self)
        painter.setClipRegion(event.region())
        painter.fillRect(event.rect(), QColor("#17191d"))
        if self._image is None:
            painter.setPen(QColor("#9aa4b2"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Add files to the queue, then select an item")
            return
        rect = self._target_rect()
        visible=rect.intersected(QRectF(event.rect()))
        if not visible.isEmpty():
            source=QRectF((visible.left()-rect.left())*self._qimage.width()/rect.width(),(visible.top()-rect.top())*self._qimage.height()/rect.height(),visible.width()*self._qimage.width()/rect.width(),visible.height()*self._qimage.height()/rect.height())
            painter.drawImage(visible,self._qimage,source)
        painter.setPen(QPen(QColor("#69d5ff"), 2))
        for point in (self._points if self._mode in {"positive", "negative", "box"} else []):
            x, y = rect.left() + point.x * rect.width(), rect.top() + point.y * rect.height()
            painter.setPen(QPen(QColor("#69d5ff") if point.positive else QColor("#ff647c"), 2))
            painter.drawEllipse(QPointF(x, y), 6, 6)
            painter.drawLine(QPointF(x - 10, y), QPointF(x + 10, y))
            if point.positive:
                painter.drawLine(QPointF(x, y - 10), QPointF(x, y + 10))
        painter.setPen(QPen(QColor("#f5c451"), 2, Qt.DashLine))
        for box in (self._boxes if self._mode in {"positive", "negative", "box"} else []):
            painter.drawRect(QRectF(
                QPointF(rect.left() + box.left * rect.width(), rect.top() + box.top * rect.height()),
                QPointF(rect.left() + box.right * rect.width(), rect.top() + box.bottom * rect.height()),
            ).normalized())
        if self._drag_start and self._drag_end:
            painter.drawRect(QRectF(self._drag_start, self._drag_end).normalized())
        if self._painting_enabled and self._mode.startswith("paint_") and self._hover and rect.contains(self._hover):
            radius = self._brush_diameter * rect.width() / (self._source_size or self._image.size)[0] / 2
            painter.setPen(QPen(QColor("black"), 3))
            painter.drawEllipse(self._hover, radius, radius)
            painter.setPen(QPen(QColor("white"), 1))
            painter.drawEllipse(self._hover, radius, radius)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MiddleButton and self._image is not None:
            self.finish_stroke()
            self._drag_start = self._drag_end = None
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.RightButton and self._mode in {"positive", "negative", "box"}:
            rect = self._target_rect()
            pos = event.position()
            candidates = []
            for i, point in enumerate(self._points):
                x, y = rect.left() + point.x * rect.width(), rect.top() + point.y * rect.height()
                distance = ((pos.x() - x) ** 2 + (pos.y() - y) ** 2) ** .5
                if distance <= 16:
                    candidates.append((distance, "point", i))
            for i, box in enumerate(self._boxes):
                bounds = QRectF(QPointF(rect.left() + box.left * rect.width(), rect.top() + box.top * rect.height()),
                                QPointF(rect.left() + box.right * rect.width(), rect.top() + box.bottom * rect.height())).normalized()
                if bounds.adjusted(-12, -12, 12, 12).contains(pos):
                    distance = min(abs(pos.x()-bounds.left()), abs(pos.x()-bounds.right()), abs(pos.y()-bounds.top()), abs(pos.y()-bounds.bottom()))
                    if distance <= 12:
                        candidates.append((distance, "box", i))
            if candidates:
                _, kind, index = min(candidates)
                self.guidance_removed.emit(kind, index)
            return
        if event.button() == Qt.LeftButton and self._mode.startswith("paint_"):
            value = self._to_normalized(event.position())
            if self._painting_enabled and value:
                self.finish_stroke()
                self.telemetry.input()
                self._brush_down = True
                self._brush_last = value
                self.stroke_started.emit()
                self.brush_segment.emit(*value, *value, 255 if self._mode == "paint_white" else 0)
        elif event.button() == Qt.LeftButton and self._mode == "box":
            self._drag_start = event.position()
            self._drag_end = event.position()
            self.update()
        elif event.button() == Qt.LeftButton and self._mode in {"positive", "negative"}:
            value = self._to_normalized(event.position())
            if value:
                self.point_added.emit(*value, self._mode == "positive")

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        old_hover=self.brush_hover_rect()
        if self._pan_last is not None:
            self._pan += event.position() - self._pan_last
            self._pan_last = event.position()
            self._clamp_pan()
            self.update()
            return
        self._hover = event.position()
        if self._brush_down and not event.buttons() & Qt.LeftButton:
            self.finish_stroke()
        if self._brush_down:
            value = self._to_normalized(event.position())
            if value:
                self.telemetry.input()
                last = self._brush_last or value
                self.brush_segment.emit(*last, *value, 255 if self._mode == "paint_white" else 0)
            # Break the stroke on leaving the artwork to avoid painting across it on re-entry.
            self._brush_last = value
        if self._drag_start:
            self._drag_end = event.position()
        if self._drag_start:self.update()
        else:self.update(old_hover.united(self.brush_hover_rect()))

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MiddleButton:
            self._pan_last = None
            self.setCursor(Qt.ArrowCursor if self._mode == "view" else Qt.CrossCursor)
            return
        if event.button() == Qt.LeftButton and self._brush_down:
            value = self._to_normalized(event.position())
            if value:
                last = self._brush_last or value
                self.brush_segment.emit(*last, *value, 255 if self._mode == "paint_white" else 0)
            self.finish_stroke()
        if event.button() == Qt.LeftButton and self._drag_start and self._drag_end:
            start, end = self._to_normalized(self._drag_start), self._to_normalized(self._drag_end)
            if start and end and abs(start[0] - end[0]) > 0.01 and abs(start[1] - end[1]) > 0.01:
                self.box_added.emit(*start, *end)
            self._drag_start = self._drag_end = None
            self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        old_hover=self.brush_hover_rect()
        self._hover = None
        self._brush_last = None
        self.update(old_hover)
        super().leaveEvent(event)

    def brush_hover_rect(self):
        if self._hover is None or self._image is None or not self._painting_enabled or not self._mode.startswith('paint_'):return QRect()
        rect=self._target_rect();radius=self._brush_diameter*rect.width()/(self._source_size or self._image.size)[0]/2+4
        return QRectF(self._hover.x()-radius,self._hover.y()-radius,2*radius,2*radius).toAlignedRect()

    def event(self,event):
        if event.type() in (QEvent.UngrabMouse,QEvent.WindowDeactivate,QEvent.Hide):
            self.finish_stroke()
        return super().event(event)

    def focusOutEvent(self,event):
        self.finish_stroke()
        super().focusOutEvent(event)
