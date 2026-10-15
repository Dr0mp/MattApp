import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import sys
if os.environ.get('MATTE_APP_TEST_ROOT'):
    sys.path.insert(0, os.environ['MATTE_APP_TEST_ROOT'])
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event, Timer
from unittest.mock import patch

from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from matte_app.cancellation import Cancellation, JobCancelled
from matte_app.window import MatteWindow, InferenceWorker, QueueWorker, VideoInferenceWorker

app = QApplication.instance() or QApplication([])
QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
app.setFont(QFont('Segoe UI', 9))


class FakeModel:
    def __init__(self, action=None):
        self.calls = 0
        self.action = action

    def predict(self, image, guidance, cancellation):
        self.calls += 1
        if self.action:
            self.action(cancellation)
        return Image.new('L', image.size, 128)


class UIFixesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = []
        for name in ('first image.png', 'second image.png'):
            path = self.root / name
            Image.new('RGB', (16, 12), 'red').save(path)
            self.sources.append(str(path))

    def tearDown(self):
        self.temp.cleanup()

    def run_worker(self, worker):
        events = []
        worker.completed.connect(lambda *args: events.append('completed'))
        worker.cancelled.connect(lambda: events.append('cancelled'))
        worker.failed.connect(lambda error: events.append(error))
        worker.run()
        return events

    def test_image_cancel_discards_result(self):
        worker = InferenceWorker('fake', Image.new('RGB', (16, 12)), [])
        with patch('matte_app.window.make_model', return_value=FakeModel(lambda token: token.cancel())):
            self.assertEqual(self.run_worker(worker), ['cancelled'])

    def test_cancel_before_start_does_not_load_model(self):
        worker = QueueWorker(self.sources, str(self.root / 'out'), 'fake', [])
        worker.cancel()
        with patch('matte_app.window.make_model') as factory:
            self.assertEqual(self.run_worker(worker), ['cancelled'])
            factory.assert_not_called()

    def test_queue_cancel_during_inference_stops_next_item(self):
        out = self.root / 'out'
        worker = QueueWorker(self.sources, str(out), 'fake', [])
        model = FakeModel(lambda token: token.cancel())
        with patch('matte_app.window.make_model', return_value=model):
            self.assertEqual(self.run_worker(worker), ['cancelled'])
        self.assertEqual(model.calls, 1)
        self.assertFalse((out / 'second image').exists())
        self.assertFalse(list(out.rglob('*.png')))
        self.assertFalse((out / 'queue.json').exists())

    def test_queue_cancel_between_items_preserves_completed_export(self):
        out = self.root / 'out'
        worker = QueueWorker(self.sources, str(out), 'fake', [])
        worker.progress.connect(lambda *args: worker.cancel())
        model = FakeModel()
        with patch('matte_app.window.make_model', return_value=model):
            self.assertEqual(self.run_worker(worker), ['cancelled'])
        self.assertEqual(model.calls, 1)
        self.assertTrue((out / 'first image/frame_000001.png').exists())
        self.assertFalse((out / 'second image').exists())

    def test_queue_success_and_manual_restart_preserve_outputs(self):
        out = self.root / 'out'
        with patch('matte_app.window.make_model', return_value=FakeModel()):
            for _ in range(2):
                self.assertEqual(self.run_worker(QueueWorker(self.sources, str(out), 'fake', [])), ['completed'])
        self.assertTrue((out / 'queue.json').exists())
        for name in ('first image', 'first image_2', 'second image', 'second image_2'):
            with Image.open(out / name / 'frame_000001.png') as image:
                self.assertEqual(image.size, (16, 12))
                self.assertEqual(image.getpixel((0, 0))[3], 128)

    def test_failure_remains_failure(self):
        worker = InferenceWorker('fake', Image.new('RGB', (16, 12)), [])
        with patch('matte_app.window.make_model', side_effect=RuntimeError('test failure')):
            self.assertIn('test failure', self.run_worker(worker)[0])

    def test_subprocess_cancel_reaps_child(self):
        marker = self.root / 'late.txt'
        token = Cancellation()
        timer = Timer(0.25, token.cancel)
        timer.start()
        start = time.monotonic()
        try:
            with self.assertRaises(JobCancelled):
                token.run_process([sys.executable, '-c', "import time, pathlib; time.sleep(3); pathlib.Path(__import__('sys').argv[1]).write_text('late')", str(marker)])
        finally:
            timer.join()
        self.assertLess(time.monotonic() - start, 2)
        self.assertFalse(marker.exists())

    def test_video_and_queued_video_cancel_between_frames(self):
        def decode(token, command):
            for index in range(1, 4):
                Image.new('RGB', (16, 12), 'red').save(command[-1].replace('%06d', f'{index:06d}'))

        for queued in (False, True):
            out = self.root / f'video-{queued}'
            worker = QueueWorker(['clip.mp4', self.sources[1]], str(out), 'fake', []) if queued else VideoInferenceWorker('clip.mp4', str(out), 'fake', [])
            model = FakeModel(lambda token: token.cancel())
            with patch('matte_app.video.shutil.which', return_value='ffmpeg'), patch.object(Cancellation, 'run_process', decode), patch('matte_app.window.make_model', return_value=model):
                self.assertEqual(self.run_worker(worker), ['cancelled'])
            self.assertEqual(model.calls, 1)
            self.assertFalse(list(out.rglob('*.png')))
            self.assertFalse((out / 'second image').exists())

    def test_real_ffmpeg_video_export(self):
        import json
        import shutil
        import subprocess
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            self.skipTest('FFmpeg unavailable')
        video = self.root / 'test.mp4'
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=red:s=32x24:r=3:d=1', '-pix_fmt', 'yuv420p', str(video)], check=True)
        out = self.root / 'frames'
        with patch('matte_app.window.make_model', return_value=FakeModel()):
            self.assertEqual(self.run_worker(VideoInferenceWorker(str(video), str(out), 'fake', [])), ['completed'])
        self.assertEqual(len(list(out.glob('frame_*.png'))), 3)
        self.assertEqual(json.loads((out / 'frames.json').read_text())['frame_count'], 3)

    def test_splitter_paths_and_scrolling(self):
        window = MatteWindow()
        window.model_picker.setCurrentText("BiRefNet HR Matting")
        window.show()
        app.processEvents()
        long_path = 'D:/' + 'very long folder/' * 20 + 'artwork.png'
        with patch.object(QFileDialog, 'getOpenFileNames', return_value=([long_path], '')):
            window.add_queue_files()
        app.processEvents()
        before = window.splitter.sizes()[0]
        handle = window.splitter.handle(1)
        QTest.mousePress(handle, Qt.LeftButton, pos=handle.rect().center())
        QTest.mouseMove(handle, handle.rect().center() + QPoint(170, 0), delay=30)
        QTest.mouseRelease(handle, Qt.LeftButton, pos=handle.rect().center() + QPoint(170, 0))
        app.processEvents()
        self.assertGreater(window.splitter.sizes()[0], before + 50)
        self.assertTrue(window.queue_list.item(0).toolTip().startswith(long_path))
        self.assertGreater(window.queue_list.horizontalScrollBar().maximum(), 0)
        window.grab().save(str(Path(__file__).parent / 'ui-preview.png'))
        window.close()

    def test_stop_lifecycle_late_signals_and_restart(self):
        window = MatteWindow()
        window.model_picker.setCurrentText("BiRefNet HR Matting")
        window.original = Image.new('RGB', (16, 12))
        started, release = Event(), Event()

        def slow_predict(token):
            started.set()
            release.wait(3)

        with patch('matte_app.window.make_model', return_value=FakeModel(slow_predict)):
            window.generate()
            self.assertTrue(started.wait(2))
            self.assertTrue(window.stop_button.isEnabled())
            self.assertFalse(window.queue_run_button.isEnabled())
            self.assertTrue(window.queue_list.isEnabled())
            worker = window.worker
            window.stop_button.click()
            window.run_queue()
            window.generate()
            self.assertIs(window.worker, worker)
            window.on_matte_ready(Image.new('L', (16, 12)))
            self.assertIsNone(window.alpha)
            release.set()
            self.assertTrue(worker.wait(3000))
            app.processEvents()
        self.assertIsNone(window._active_worker)
        self.assertTrue(window.queue_run_button.isEnabled())
        self.assertFalse(window.stop_button.isEnabled())
        self.assertIn('Cancelled', window.status.text())
        with patch('matte_app.window.make_model', return_value=FakeModel()):
            window.generate()
            self.assertTrue(window.worker.wait(3000))
            app.processEvents()
        self.assertIsNotNone(window.alpha)
        window.close()

    def test_close_waits_for_safe_cancellation(self):
        window = MatteWindow()
        window.model_picker.setCurrentText("BiRefNet HR Matting")
        window.original = Image.new('RGB', (16, 12))
        window.show()
        started, release = Event(), Event()
        def slow_predict(token):
            started.set()
            release.wait(3)
        with patch('matte_app.window.make_model', return_value=FakeModel(slow_predict)):
            window.generate()
            self.assertTrue(started.wait(2))
            self.assertFalse(window.close())
            self.assertTrue(window._cancel_requested)
            release.set()
            self.assertTrue(window.worker.wait(3000))
            app.processEvents()
        self.assertFalse(window.isVisible())


if __name__ == '__main__':
    unittest.main(verbosity=2)
