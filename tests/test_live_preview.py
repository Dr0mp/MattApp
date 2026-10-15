import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shutil
import subprocess
import tempfile
import time
import unittest
from threading import Event
from unittest.mock import patch
from PIL import Image
from PySide6.QtWidgets import QApplication, QListWidgetItem
from matte_app.window import MatteWindow, VideoInferenceWorker, QueueWorker
from matte_app.video import VideoInfo

app = QApplication.instance() or QApplication([])

class PacedModel:
    def __init__(self, pause=True):
        self.calls = 0
        self.pause = pause
        self.release = Event()

    def predict(self, image, guidance, token):
        self.calls += 1
        if self.pause and self.calls == 3:
            if not self.release.wait(8):
                raise RuntimeError('Test did not release model')
        return Image.new('L', image.size, self.calls * 30)

class LivePreviewTests(unittest.TestCase):
    def setUp(self):
        if not shutil.which('ffmpeg'):
            self.skipTest('FFmpeg required')
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root/'clip.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i',
                        'color=red:size=64x48:rate=3:duration=2','-pix_fmt','yuv420p',str(self.video)],check=True)
        self.w = MatteWindow()
        self.w.model_picker.setCurrentText("BiRefNet HR Matting")
        self.w.results_index_path = self.root/'index.json'
        self.w.original = Image.new('RGBA',(64,48),'red')
        self.w.current_media_path = self.w.video_path = str(self.video)
        self.w.video_info = VideoInfo(6,3)
        self.w._refresh_canvas()
        self.model = PacedModel()
        self.model_patch = patch('matte_app.window.make_model', return_value=self.model)
        self.model_patch.start()

    def tearDown(self):
        self.model.release.set()
        if self.w._active_worker:
            self.w.cancel_jobs()
            self.wait(lambda: self.w._active_worker is None)
        self.w.close()
        self.model_patch.stop()
        self.temp.cleanup()

    def wait(self, predicate, timeout=8):
        end = time.monotonic()+timeout
        while time.monotonic()<end:
            app.processEvents()
            if predicate():
                return
            time.sleep(.01)
        self.fail('Timed out waiting for live preview')

    def start_video(self):
        folder = self.root/'matte'
        self.w.video_output_dir = str(folder)
        worker = VideoInferenceWorker(str(self.video),str(folder),'fake',[],info=VideoInfo(6,3))
        self.w.video_worker = worker
        worker.progress.connect(self.w.on_video_progress)
        worker.completed.connect(self.w.on_video_ready)
        self.w._start_worker(worker)
        return folder, worker

    def assert_live_second(self):
        self.wait(lambda: self.w.current_frame == 2)
        self.assertIsNotNone(self.w._active_worker)
        self.assertEqual(self.w.current_view(),'cutout')
        self.assertEqual(self.w.alpha.getpixel((20,20)),60)
        self.assertEqual(self.w.original.getpixel((20,20))[3],255)
        self.assertEqual(self.w.frame_spin.value(),2)
        self.assertFalse(self.w.white_brush_button.isEnabled())
        self.w.view_tabs.setCurrentIndex(1)
        self.assertEqual(self.w.canvas._image.getpixel((20,20))[:3],(60,60,60))

    def test_running_video_preview_and_final_frame_remain(self):
        folder, worker = self.start_video()
        self.assert_live_second()
        self.model.release.set()
        self.wait(lambda: self.w._active_worker is None)
        self.assertEqual(self.w.current_frame,6)
        self.assertEqual(self.w.alpha.getpixel((20,20)),180)
        self.assertEqual(self.w.frame_slider.value(),6)
        self.assertTrue((folder/'frame_000006.png').exists())
        self.assertFalse(self.w._live_preview_timer.isActive())

    def test_cancel_keeps_last_committed_frame(self):
        folder, worker = self.start_video()
        self.assert_live_second()
        self.w.cancel_jobs()
        self.model.release.set()
        self.wait(lambda: self.w._active_worker is None)
        self.assertEqual(self.w.current_frame,2)
        self.assertEqual(self.w.alpha.getpixel((20,20)),60)
        self.assertFalse((folder/'frame_000003.png').exists())
        self.assertTrue(self.w.white_brush_button.isEnabled())

    def test_queue_follows_video_then_image(self):
        image_path = self.root/'still.png'
        Image.new('RGB',(80,60),'blue').save(image_path)
        paths = [str(self.video), str(image_path)]
        for path in paths:
            self.w.queue_list.addItem(QListWidgetItem(path))
        worker = QueueWorker(paths,str(self.root/'queue'),'fake',[])
        self.w.queue_worker = worker
        self.w._start_worker(worker)
        self.assert_live_second()
        self.assertEqual(self.w.video_info.fps,3)
        self.model.release.set()
        self.wait(lambda: self.w._active_worker is None)
        self.assertEqual(self.w.current_media_path,str(image_path))
        self.assertIsNone(self.w.video_path)
        self.assertEqual(self.w.alpha.size,(80,60))
        self.assertEqual(self.w.alpha.getpixel((20,20)),210)
        self.assertEqual(self.w.queue_list.currentRow(),1)
        self.assertTrue((Path(self.w.video_output_dir)/'frame_000001.png').exists())

    def test_burst_frames_are_bounded_and_latest_is_kept(self):
        self.model.pause = False
        folder = self.root/'burst'
        worker = VideoInferenceWorker(str(self.video),str(folder),'fake',[])
        worker.execute()
        self.assertEqual(worker.live_frames.qsize(),1)
        preview = worker.live_frames.get_nowait()
        self.assertEqual(preview.frame,6)
        self.assertEqual(preview.original.getpixel((20,20))[3],180)
        self.assertTrue((folder/'frame_000006.png').exists())

if __name__ == '__main__':
    unittest.main()
