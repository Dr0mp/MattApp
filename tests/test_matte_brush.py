import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
if os.environ.get('MATTE_APP_TEST_ROOT'):
    sys.path.insert(0, os.environ['MATTE_APP_TEST_ROOT'])
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent, QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog

from matte_app.matte_editor import MatteEditor
from matte_app.window import MatteWindow

app = QApplication.instance() or QApplication([])
QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
app.setFont(QFont('Segoe UI', 9))


class BrushTests(unittest.TestCase):
    def setUp(self):
        self.window = MatteWindow()
        self.window.model_picker.setCurrentText("BiRefNet HR Matting")
        self.window.original = Image.new('RGBA', (240, 120), (30, 80, 160, 255))
        self.window.on_matte_ready(Image.new('L', (240, 120), 128))
        self.window.view_tabs.setCurrentIndex(1)
        self.window.show()
        app.processEvents()

    def tearDown(self):
        self.window.close()

    def position(self, x, y):
        rect = self.window.canvas._target_rect()
        return QPointF(rect.left() + rect.width() * x, rect.top() + rect.height() * y)

    def stroke(self, start, end):
        canvas = self.window.canvas
        a, b = self.position(*start), self.position(*end)
        QTest.mousePress(canvas, Qt.LeftButton, pos=a.toPoint())
        # A single fast move must fill the whole segment, not leave spaced dots.
        event = QMouseEvent(QMouseEvent.MouseMove, b, canvas.mapToGlobal(b.toPoint()), Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(canvas, event)
        QTest.mouseRelease(canvas, Qt.LeftButton, pos=b.toPoint())
        app.processEvents()

    def test_white_black_brush_export_and_undo(self):
        w = self.window
        original_bytes = w.original.tobytes()
        w.brush_size.setValue(10)
        w.white_brush_button.click()
        self.stroke((.2, .4), (.8, .4))
        for x in range(49, 190):
            self.assertEqual(w.alpha.getpixel((x, 48)), 255)
        self.assertEqual(w.alpha.getpixel((120, 80)), 128)
        self.assertEqual(len(w.points), 0)
        w.black_brush_button.click()
        self.stroke((.5, .3), (.5, .7))
        self.assertEqual(w.alpha.getpixel((120, 48)), 0)
        self.assertEqual(len(w.matte_editor.undo_strokes), 2)
        self.assertTrue(w.undo_brush_button.isEnabled())
        with tempfile.TemporaryDirectory() as folder:
            cutout, matte = Path(folder) / 'cutout.png', Path(folder) / 'matte.png'
            with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(cutout), '')):
                w.export_cutout()
            with patch.object(QFileDialog, 'getSaveFileName', return_value=(str(matte), '')):
                w.export_matte()
            with Image.open(cutout) as exported:
                self.assertEqual(exported.size, (240, 120))
                self.assertEqual(exported.getpixel((120, 48)), (30, 80, 160, 0))
                self.assertEqual(exported.getpixel((60, 48))[3], 255)
            with Image.open(matte) as exported:
                self.assertEqual(exported.tobytes(), w.alpha.tobytes())
        w.undo_brush_button.click()
        self.assertEqual(w.alpha.getpixel((120, 48)), 255)
        w.redo_brush_button.click()
        self.assertEqual(w.alpha.getpixel((120, 48)), 0)
        self.assertEqual(w.original.tobytes(), original_bytes)

    def test_tabs_limit_painting_to_matte(self):
        w = self.window
        w.view_tabs.setCurrentIndex(0)
        self.assertTrue(w.guidance_panel.isVisible())
        self.assertFalse(w.paint_panel.isVisible())
        before = w.alpha.tobytes()
        self.stroke((.3, .5), (.8, .5))
        self.assertEqual(w.alpha.tobytes(), before)
        w.view_tabs.setCurrentIndex(2)
        self.assertFalse(w.guidance_panel.isVisible())
        self.assertFalse(w.paint_panel.isVisible())
        points = len(w.points)
        self.stroke((.3, .5), (.8, .5))
        self.assertEqual(w.alpha.tobytes(), before)
        self.assertEqual(len(w.points), points)
        w.view_tabs.setCurrentIndex(1)
        self.assertTrue(w.paint_panel.isVisible())
        self.assertFalse(w.guidance_panel.isVisible())
        self.stroke((.3, .5), (.8, .5))
        self.assertEqual(w.alpha.getpixel((120, 60)), 255)

    def test_letterbox_and_edge_clipping(self):
        w = self.window
        w.black_brush_button.click()
        w.brush_size.setValue(80)
        QTest.mouseClick(w.canvas, Qt.LeftButton, pos=QPointF(5, 5).toPoint())
        self.assertFalse(w.matte_editor.undo_strokes)
        self.stroke((.002, .5), (.05, .5))
        self.assertEqual(w.alpha.getpixel((0, 60)), 0)
        self.assertEqual(w.alpha.getpixel((239, 60)), 128)

    def test_regeneration_clears_undo_and_tools_reset(self):
        w = self.window
        w.black_brush_button.click()
        self.stroke((.3, .5), (.5, .5))
        self.assertTrue(w.undo_brush_button.isEnabled())
        w.on_matte_ready(Image.new('L', w.original.size, 80))
        self.assertFalse(w.undo_brush_button.isEnabled())
        self.assertFalse(w.redo_brush_button.isEnabled())
        self.assertTrue(w.point_button.isChecked())
        self.assertEqual(w.alpha.getpixel((100, 60)), 80)
        self.assertIs(w.matte_editor.alpha, w.alpha)

    def test_busy_disables_editing(self):
        w = self.window
        w._active_worker = object()
        w._set_busy(True)
        self.assertFalse(w.black_brush_button.isEnabled())
        self.assertFalse(w.canvas._painting_enabled)
        before = w.alpha.tobytes()
        w.paint_matte(.2, .5, .8, .5, 0)
        self.assertEqual(w.alpha.tobytes(), before)
        w._active_worker = None
        w._set_busy(False)
        self.assertTrue(w.black_brush_button.isEnabled())

    def test_large_preview_preserves_source_brush_coordinates(self):
        w = self.window
        w.original = Image.new('RGBA', (3200, 1600), (20, 50, 90, 255))
        w.on_matte_ready(Image.new('L', w.original.size, 128))
        w.view_tabs.setCurrentIndex(1)
        w.brush_size.setValue(20)
        w.black_brush_button.click()
        self.stroke((.25, .5), (.75, .5))
        self.assertEqual(w.alpha.size, (3200, 1600))
        self.assertEqual(w.canvas._image.size, (1600, 800))
        self.assertEqual(w.alpha.getpixel((1600, 800)), 0)
        self.assertEqual(w.alpha.getpixel((1600, 840)), 128)

    def test_redo_branch_and_single_pixel_brush(self):
        editor = MatteEditor()
        alpha = Image.new('L', (20, 10), 128)
        editor.reset(alpha)
        editor.begin_stroke()
        editor.paint((.5, .5), (.5, .5), 1, 255)
        editor.end_stroke()
        self.assertEqual(alpha.getpixel((10, 5)), 255)
        editor.undo()
        self.assertEqual(alpha.getpixel((10, 5)), 128)
        editor.begin_stroke()
        editor.paint((.25, .5), (.25, .5), 1, 0)
        editor.end_stroke()
        self.assertFalse(editor.redo_strokes)
        editor.undo()
        self.assertEqual(alpha.tobytes(), Image.new('L', alpha.size, 128).tobytes())

    def test_brush_preview(self):
        self.window.black_brush_button.click()
        self.stroke((.2, .5), (.8, .5))
        self.window.grab().save(str(Path(__file__).parent / 'brush-preview.png'))

    def test_keyboard_undo_redo(self):
        w = self.window
        w.activateWindow()
        w.black_brush_button.click()
        self.stroke((.25, .5), (.75, .5))
        w.canvas.setFocus()
        QTest.keyClick(w.canvas, Qt.Key_Z, Qt.ControlModifier)
        app.processEvents()
        self.assertEqual(w.alpha.getpixel((120, 60)), 128)
        QTest.keyClick(w.canvas, Qt.Key_Y, Qt.ControlModifier)
        app.processEvents()
        self.assertEqual(w.alpha.getpixel((120, 60)), 0)

    def test_opening_new_image_clears_history_and_disables_brush(self):
        w = self.window
        w.black_brush_button.click()
        self.stroke((.25, .5), (.75, .5))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'new.png'
            Image.new('RGB', (30, 20), 'green').save(path)
            with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(path), '')):
                w.open_image()
        self.assertIsNone(w.alpha)
        self.assertIsNone(w.matte_editor.alpha)
        self.assertFalse(w.matte_editor.undo_strokes)
        self.assertFalse(w.black_brush_button.isEnabled())
        self.assertFalse(w.canvas._painting_enabled)

    def test_ui_uses_image_tabs_and_has_no_open_buttons(self):
        from PySide6.QtWidgets import QPushButton
        w = self.window
        texts = {button.text() for button in w.findChildren(QPushButton)}
        self.assertNotIn('Open image', texts)
        self.assertNotIn('Open video', texts)
        self.assertEqual([w.view_tabs.tabText(i) for i in range(w.view_tabs.count())], ['Original', 'Matte', 'Cutout'])
        self.assertIs(w.generate_button.parentWidget(), w.resume_button.parentWidget())
        self.assertFalse(w.resume_button.isVisible())
        self.assertTrue(w.generate_button.isVisible())

    def test_tab_switch_preserves_edits_and_guidance(self):
        w = self.window
        w.black_brush_button.click()
        self.stroke((.25, .5), (.75, .5))
        painted = w.alpha.tobytes()
        w.view_tabs.setCurrentIndex(0)
        QTest.mouseClick(w.canvas, Qt.LeftButton, pos=self.position(.2, .3).toPoint())
        self.assertEqual(len(w.points), 1)
        points = list(w.points)
        w.view_tabs.setCurrentIndex(2)
        self.assertEqual(w.canvas._mode, 'view')
        self.assertFalse(w.undo_brush_action.isEnabled())
        self.stroke((.2, .3), (.8, .3))
        self.assertEqual(w.alpha.tobytes(), painted)
        self.assertEqual(w.points, points)
        w.view_tabs.setCurrentIndex(1)
        self.assertTrue(w.undo_brush_action.isEnabled())
        self.assertEqual(w.canvas._mode, 'paint_black')
        w.undo_brush_button.click()
        self.assertEqual(w.alpha.getextrema(), (128, 128))
        self.assertEqual(w.points, points)
        for index, name in enumerate(('original', 'matte', 'cutout')):
            w.view_tabs.setCurrentIndex(index)
            app.processEvents()
            w.grab().save(str(Path(__file__).parent / f'tab-{name}-preview.png'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
