import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
sys.path.insert(0, os.environ.get('MATTE_APP_TEST_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[1])))
import unittest
from pathlib import Path
from PIL import Image
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QFontDatabase, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from matte_app.window import MatteWindow
from matte_app.video import VideoInfo

app = QApplication.instance() or QApplication([])
QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')

class ZoomTests(unittest.TestCase):
    def setUp(self):
        self.w = MatteWindow()
        self.w.model_picker.setCurrentText("BiRefNet HR Matting")
        self.w.original = Image.new('RGB', (800, 600), '#447799')
        self.w.on_matte_ready(Image.new('L', (800, 600), 128))
        self.w.resize(1800, 1000)
        self.w.show()
        app.processEvents()

    def tearDown(self):
        self.w.close()

    def test_zoom_anchor_fit_and_same_size_refresh(self):
        c = self.w.canvas
        anchor = QPointF(c.width() * .6, c.height() * .55)
        before = c._to_normalized(anchor)
        c.set_zoom(3, anchor)
        after = c._to_normalized(anchor)
        self.assertAlmostEqual(before[0], after[0])
        self.assertAlmostEqual(before[1], after[1])
        self.w._refresh_canvas()
        self.assertEqual(c._zoom, 3)
        self.assertEqual(self.w.zoom_percent.value(), 300)
        self.w.fit_button.click()
        self.assertEqual(c._zoom, 1)
        self.assertEqual(c._pan, QPointF())

    def test_brush_and_guidance_after_pan(self):
        w, c = self.w, self.w.canvas
        w.view_tabs.setCurrentIndex(1)
        w.brush_size.setValue(8)
        c.set_zoom(3)
        edits = len(w.matte_editor.undo_strokes)
        center = QPoint(c.width() // 2, c.height() // 2)
        QTest.mousePress(c, Qt.MiddleButton, pos=center)
        moved = QPointF(center + QPoint(70, 40))
        event = QMouseEvent(QMouseEvent.MouseMove, moved, moved, Qt.NoButton, Qt.MiddleButton, Qt.NoModifier)
        app.sendEvent(c, event)
        QTest.mouseRelease(c, Qt.MiddleButton, pos=moved.toPoint())
        self.assertNotEqual(c._pan, QPointF())
        self.assertEqual(len(w.matte_editor.undo_strokes), edits)
        norm = c._to_normalized(QPointF(center))
        QTest.mouseClick(c, Qt.LeftButton, pos=center)
        self.assertEqual(w.alpha.getpixel((round(norm[0]*799), round(norm[1]*599))), 255)
        w.undo_brush()
        self.assertEqual(w.alpha.getpixel((round(norm[0]*799), round(norm[1]*599))), 128)
        w.view_tabs.setCurrentIndex(0)
        app.processEvents()
        norm = c._to_normalized(QPointF(center))
        QTest.mouseClick(c, Qt.LeftButton, pos=center)
        self.assertAlmostEqual(w.points[-1].x, norm[0])
        self.assertAlmostEqual(w.points[-1].y, norm[1])

    def test_wheel_and_zoom_limits(self):
        c = self.w.canvas
        pos = QPointF(c.width()/2, c.height()/2)
        event = QWheelEvent(pos, pos, QPoint(), QPoint(0,120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        app.sendEvent(c,event)
        self.assertEqual(c._zoom, 1.25)
        c.set_zoom(100)
        self.assertEqual(c._zoom, 16)
        c.set_zoom(.1)
        self.assertEqual(c._zoom, 1)

    def test_zoom_reveals_source_detail_and_new_size_resets(self):
        w, c = self.w, self.w.canvas
        w.original = Image.new('RGB', (3200, 2400), '#447799')
        w.on_matte_ready(Image.new('L', (3200, 2400), 128))
        self.assertEqual(c._image.size, (1600, 1200))
        w.zoom_percent.setValue(200)
        self.assertEqual(c._image.size, (3200, 2400))
        w.view_tabs.setCurrentIndex(1)
        self.assertEqual(c._zoom, 2)
        self.assertEqual(c._image.size, (3200, 2400))
        w.original = Image.new('RGB', (240, 120))
        w.on_matte_ready(Image.new('L', (240, 120), 128))
        self.assertEqual(c._zoom, 1)
        self.assertEqual(w.zoom_percent.value(), 100)

    def test_box_after_zoom_uses_source_coordinates(self):
        w, c = self.w, self.w.canvas
        w.view_tabs.setCurrentIndex(0)
        app.processEvents()
        w.box_button.click()
        c.set_zoom(2)
        start = QPoint(c.width()//2 - 30, c.height()//2 - 30)
        end = QPoint(c.width()//2 + 30, c.height()//2 + 30)
        first, last = c._to_normalized(QPointF(start)), c._to_normalized(QPointF(end))
        QTest.mousePress(c, Qt.LeftButton, pos=start)
        event = QMouseEvent(QMouseEvent.MouseMove, QPointF(end), QPointF(end), Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        app.sendEvent(c,event)
        QTest.mouseRelease(c, Qt.LeftButton, pos=end)
        box = w.boxes[-1]
        for actual, expected in zip((box.left,box.top,box.right,box.bottom), (*first,*last)):
            self.assertAlmostEqual(actual, expected)

    def test_single_rows_and_slider_above_narrow_scrolling(self):
        w = self.w
        w.video_path = 'test.mp4'
        w.video_info = VideoInfo(107, 24)
        w.model_picker.setCurrentText('SAM 2 + VideoMaMa')
        w.view_tabs.setCurrentIndex(0)
        w._update_navigation()
        app.processEvents()
        groups = [(w.play_button,w.play_all_button,w.previous_frame_button,w.next_frame_button,w.frame_spin),
                  (w.point_button,w.negative_button,w.box_button,w.clear_button,w.guidance_picker,w.remove_guidance_button)]
        for group in groups:
            centers = [b.mapTo(w,QPoint(0,b.height()//2)).y() for b in group]
            self.assertLessEqual(max(centers)-min(centers), 2)
        self.assertLess(w.frame_slider.mapTo(w,QPoint()).y(), w.play_button.mapTo(w,QPoint()).y())
        if os.environ.get('MATTE_ZOOM_SCREENSHOTS'):
            w.grab().save(str(Path(os.environ['MATTE_ZOOM_SCREENSHOTS'])/'toolbar-wide.png'))
        w.resize(1000, 800)
        app.processEvents()
        self.assertGreater(w.video_toolbar.horizontalScrollBar().maximum(), 0)
        self.assertGreater(w.guidance_toolbar.horizontalScrollBar().maximum(), 0)
        self.assertEqual(w.video_toolbar.verticalScrollBar().maximum(),0)
        w.guidance_toolbar.horizontalScrollBar().setValue(w.guidance_toolbar.horizontalScrollBar().maximum())
        app.processEvents()
        if os.environ.get('MATTE_ZOOM_SCREENSHOTS'):
            w.grab().save(str(Path(os.environ['MATTE_ZOOM_SCREENSHOTS'])/'toolbar-narrow.png'))

if __name__ == '__main__':
    unittest.main()
