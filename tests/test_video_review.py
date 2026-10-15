import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
sys.path.insert(0, os.environ.get('MATTE_APP_TEST_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[1])))
import json
import shutil
import subprocess
import tempfile
import time
from threading import Event
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PySide6.QtWidgets import QApplication, QFileDialog, QListWidgetItem, QMessageBox
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest

from matte_app.cancellation import Cancellation, JobCancelled
from matte_app.models import PointHint, BoxHint
from matte_app.video import VideoInfo, load_frame, probe_video, process_video, read_metadata
from matte_app.window import MatteWindow, PreviewWorker, QueueWorker, PlaybackWorker

app = QApplication.instance() or QApplication([])


class Model:
    def __init__(self, alpha=60, cancel_on=0):
        self.alpha, self.cancel_on = alpha, cancel_on
        self.images, self.guidance = [], []

    def predict(self, image, guidance, token):
        self.images.append(image.copy())
        self.guidance.append(list(guidance))
        if len(self.images) == self.cancel_on:
            token.cancel()
        return Image.new('L', image.size, self.alpha)


class VideoReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
            raise unittest.SkipTest('FFmpeg and FFprobe required')
        cls.fixture = tempfile.TemporaryDirectory()
        cls.video = Path(cls.fixture.name) / 'six frames.mp4'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=3:duration=2', '-pix_fmt', 'yuv420p', str(cls.video)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / 'results'
        self.window = MatteWindow()
        self.window.model_picker.setCurrentText("BiRefNet HR Matting")
        self.window.results_index_path = self.root / 'index.json'
        self.window.queue_outputs = {}
        self.dialog_patch = patch.object(QMessageBox, 'critical')
        self.errors = self.dialog_patch.start()

    def tearDown(self):
        self.window.close()
        self.wait_idle()
        self.dialog_patch.stop()
        self.temp.cleanup()

    def wait_idle(self, timeout=8):
        limit = time.monotonic() + timeout
        quiet = 0
        while time.monotonic() < limit:
            app.processEvents()
            busy = self.window._active_worker is not None or self.window._navigation_timer.isActive() or self.window._pending_frame_action is not None
            quiet = 0 if busy else quiet + 1
            if quiet >= 3:
                return
            time.sleep(.01)
        self.fail('Worker did not finish')

    def export(self, alpha=60, guidance=None):
        return process_video(str(self.video), self.folder, Model(alpha), 'BiRefNet HR Matting', guidance or [], Cancellation())

    def test_real_probe_and_exact_frame_navigation(self):
        info = probe_video(str(self.video), Cancellation())
        self.assertEqual(info, VideoInfo(6, 3.0))
        self.window._load_media(str(self.video))
        self.wait_idle()
        self.assertEqual(self.window.video_info.frame_count, 6)
        self.window.request_frame(5)
        self.wait_idle()
        self.assertEqual(self.window.current_frame, 5)
        expected = load_frame(str(self.video), 5, Cancellation())
        self.assertEqual(self.window.original.tobytes(), expected.tobytes())
        self.assertIsNone(self.window.alpha)
        self.assertFalse(self.errors.called)

    def test_resume_preserves_earlier_frames_and_updates_later(self):
        self.export(60, [PointHint(.2, .3)])
        earlier = [(self.folder / f'frame_{i:06d}.png').read_bytes() for i in (1, 2)]
        process_video(str(self.video), self.folder, Model(220), 'BiRefNet HR Matting', [PointHint(.8, .7)], Cancellation(), 3)
        self.assertEqual(earlier, [(self.folder / f'frame_{i:06d}.png').read_bytes() for i in (1, 2)])
        for i in range(3, 7):
            with Image.open(self.folder / f'frame_{i:06d}.png') as image:
                self.assertEqual(image.getchannel('A').getextrema(), (220, 220))
        metadata = read_metadata(self.folder)
        self.assertEqual(metadata['status'], 'complete')
        self.assertEqual(metadata['runs'][-1]['start_frame'], 3)
        self.window._restore_saved_guidance(str(self.folder), 1)
        self.assertEqual(self.window.points, [PointHint(.2, .3)])
        self.window._restore_saved_guidance(str(self.folder), 4)
        self.assertEqual(self.window.points, [PointHint(.8, .7)])

    def test_cancelled_resume_keeps_prior_and_unprocessed_results(self):
        self.export()
        before = {i: (self.folder / f'frame_{i:06d}.png').read_bytes() for i in range(1, 7)}
        with self.assertRaises(JobCancelled):
            process_video(str(self.video), self.folder, Model(180, cancel_on=2), 'BiRefNet HR Matting', [], Cancellation(), 3)
        for i in (1, 2, 4, 5, 6):
            self.assertEqual((self.folder / f'frame_{i:06d}.png').read_bytes(), before[i])
        self.assertNotEqual((self.folder / 'frame_000003.png').read_bytes(), before[3])
        metadata = read_metadata(self.folder)
        self.assertEqual(metadata['runs'][-1]['status'], 'cancelled')
        self.assertEqual(metadata['runs'][-1]['last_frame'], 3)
        self.assertFalse(list(self.folder.glob('*.tmp')))

    def test_saved_preview_uses_exported_alpha_and_original_rgb(self):
        self.export(85)
        expected = load_frame(str(self.video), 4, Cancellation())
        events = []
        worker = PreviewWorker(str(self.video), 4, str(self.folder))
        worker.completed.connect(events.append)
        with patch('matte_app.window.load_frame', side_effect=AssertionError('Saved frame should not decode again')):
            worker.run()
        self.assertEqual(len(events), 1)
        preview = events[0]
        self.assertEqual(preview.alpha.getextrema(), (85, 85))
        self.assertEqual(preview.original.tobytes(), expected.tobytes())

    def test_queue_links_selection_and_clear(self):
        source = self.root / 'image.png'
        Image.new('RGB', (64, 48), 'red').save(source)
        for path in (str(source), str(self.video)):
            item = QListWidgetItem(path)
            item.setToolTip(path)
            self.window.queue_list.addItem(item)
        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(self.root / 'queue')), patch('matte_app.window.make_model', return_value=Model(77)):
            self.window.run_queue()
            self.wait_idle()
        self.assertTrue(self.window.output_for(str(source)))
        self.assertTrue(self.window.output_for(str(self.video)))
        index = json.loads(self.window.results_index_path.read_text())
        self.assertEqual(len(index), 2)
        self.window.review_queue_item(self.window.queue_list.item(0))
        self.wait_idle()
        self.assertEqual(self.window.alpha.getextrema(), (77, 77))
        self.assertIsNone(self.window.video_path)
        self.window.review_queue_item(self.window.queue_list.item(1))
        self.wait_idle()
        self.assertEqual(self.window.alpha.getextrema(), (77, 77))
        self.assertEqual(self.window.video_info.frame_count, 6)
        self.window.clear_queue_button.click()
        self.assertEqual(self.window.queue_list.count(), 0)
        self.assertTrue((self.root / 'queue/image/frame_000001.png').exists())
        self.assertFalse(self.errors.called)

    def test_selected_frame_testing_and_resume_ui(self):
        self.export(60)
        w = self.window
        w._load_media(str(self.video), str(self.folder))
        self.wait_idle()
        model = Model(210)
        w.points[:] = [PointHint(.75, .25)]
        w.frame_spin.setValue(4)
        with patch('matte_app.window.make_model', return_value=model):
            w.test_frame_button.click()
            self.wait_idle()
        self.assertEqual(w.current_frame, 4)
        self.assertEqual(w.alpha.getextrema(), (210, 210))
        self.assertEqual(model.images[0].tobytes(), load_frame(str(self.video), 4, Cancellation()).tobytes())
        self.assertEqual(model.guidance[0], [PointHint(.75, .25)])
        with Image.open(self.folder / 'frame_000004.png') as image:
            self.assertEqual(image.getchannel('A').getextrema(), (60, 60))
        earlier = (self.folder / 'frame_000003.png').read_bytes()
        with patch('matte_app.window.make_model', return_value=Model(210)):
            w.resume_button.click()
            self.wait_idle()
        self.assertEqual((self.folder / 'frame_000003.png').read_bytes(), earlier)
        self.assertEqual(w.alpha.getextrema(), (210, 210))
        # Processing now follows completed frames and finishes on the last result.
        self.assertEqual(w.current_frame, 6)
        self.assertFalse(self.errors.called)

    def test_partial_new_export_has_no_stale_matte_on_missing_frame(self):
        process_video(str(self.video), self.folder, Model(88), 'BiRefNet HR Matting', [], Cancellation(), 4)
        self.assertEqual(read_metadata(self.folder)['status'], 'partial')
        self.window._load_media(str(self.video), str(self.folder), 4)
        self.wait_idle()
        self.assertIsNotNone(self.window.alpha)
        self.window.request_frame(2)
        self.wait_idle()
        self.assertIsNone(self.window.alpha)

    def test_legacy_queue_and_video_results_can_be_loaded(self):
        self.export(110)
        metadata = read_metadata(self.folder)
        metadata.pop('runs')
        (self.folder / 'frames.json').write_text(json.dumps(metadata))
        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(self.folder)):
            self.window.load_results()
            self.wait_idle()
        self.assertEqual(self.window.alpha.getextrema(), (110, 110))
        source = self.root / 'legacy.png'
        Image.new('RGB', (64, 48), 'blue').save(source)
        legacy_root = self.root / 'legacy-queue'
        legacy_folder = legacy_root / 'legacy'
        legacy_folder.mkdir(parents=True)
        Image.new('RGBA', (64, 48), (0, 0, 255, 90)).save(legacy_folder / 'frame_000001.png')
        (legacy_root / 'queue.json').write_text(json.dumps({'sources': [str(source)]}))
        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(legacy_root)):
            self.window.load_results()
        self.assertEqual(self.window.output_for(str(source)), str(legacy_folder))

    def test_output_folder_cannot_be_used_for_wrong_video(self):
        self.export()
        before = (self.folder / 'frame_000001.png').read_bytes()
        with self.assertRaises(RuntimeError):
            process_video(str(self.root / 'another.mp4'), self.folder, Model(), 'fake', [], Cancellation(), 2)
        self.assertEqual((self.folder / 'frame_000001.png').read_bytes(), before)

    def test_navigation_layout_and_queue_controls(self):
        self.export()
        self.window._load_media(str(self.video), str(self.folder))
        self.wait_idle()
        self.window.show()
        app.processEvents()
        self.assertTrue(self.window.video_panel.isVisible())
        self.assertTrue(self.window.clear_queue_button.isEnabled())
        self.assertFalse(self.window.previous_frame_button.isEnabled())
        self.assertTrue(self.window.next_frame_button.isEnabled())
        self.window.grab().save(str(Path(__file__).parent / 'video-review-preview.png'))

    def test_reopened_video_can_attach_folder_and_continue_without_queue(self):
        self.export(97, [PointHint(.2, .3)])
        self.window.close()
        self.window = MatteWindow()
        self.window.model_picker.setCurrentText("BiRefNet HR Matting")
        self.window.results_index_path = self.root / 'reopened-index.json'
        self.window.queue_outputs = {}
        with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(self.video), '')):
            self.window.open_video()
            self.wait_idle()
        self.assertEqual(self.window.queue_list.count(), 0)
        self.assertIsNone(self.window.video_output_dir)
        self.assertTrue(self.window.load_matte_folder_button.isEnabled())
        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(self.folder)):
            self.window.load_matte_folder_button.click()
            self.wait_idle()
        self.assertEqual(self.window.video_output_dir, str(self.folder))
        self.assertEqual(self.window.alpha.getextrema(), (97, 97))
        self.assertEqual(self.window.points, [PointHint(.2, .3)])
        self.assertEqual(self.window.output_for(str(self.video)), str(self.folder))
        self.assertEqual(json.loads(self.window.results_index_path.read_text())[str(self.video.resolve())], str(self.folder))
        self.window.request_frame(5)
        self.wait_idle()
        before = (self.folder / 'frame_000004.png').read_bytes()
        self.window.points[:] = [PointHint(.7, .6)]
        model = Model(200)
        with patch('matte_app.window.make_model', return_value=model):
            self.window.resume_button.click()
            self.wait_idle()
        self.assertEqual((self.folder / 'frame_000004.png').read_bytes(), before)
        self.assertEqual(self.window.alpha.getextrema(), (200, 200))
        self.assertEqual(model.guidance[0], [PointHint(.7, .6)])

    def test_legacy_matte_folder_can_attach_to_open_video_without_queue(self):
        self.export(130)
        (self.folder / 'frames.json').unlink()
        self.window._load_media(str(self.video))
        self.wait_idle()
        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(self.folder)):
            self.window.load_results()
            self.wait_idle()
        self.assertEqual(self.window.queue_list.count(), 0)
        self.assertEqual(self.window.video_output_dir, str(self.folder))
        self.assertEqual(self.window.alpha.getextrema(), (130, 130))

    def test_attach_rejects_wrong_video_and_empty_folder(self):
        self.export()
        self.window._load_media(str(self.video))
        self.wait_idle()
        metadata = read_metadata(self.folder)
        metadata['source_video'] = str(self.root / 'wrong-video.mp4')
        (self.folder / 'frames.json').write_text(json.dumps(metadata))
        with self.assertRaises(RuntimeError):
            self.window.attach_video_matte_folder(str(self.folder))
        self.assertIsNone(self.window.video_output_dir)
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaises(RuntimeError):
            self.window.attach_video_matte_folder(str(empty))
        self.assertIsNone(self.window.output_for(str(self.video)))

    def test_attach_partial_folder_loads_first_available_saved_frame(self):
        process_video(str(self.video), self.folder, Model(155), 'BiRefNet HR Matting', [], Cancellation(), 4)
        self.window._load_media(str(self.video))
        self.wait_idle()
        self.window.attach_video_matte_folder(str(self.folder))
        self.wait_idle()
        self.assertEqual(self.window.current_frame, 4)
        self.assertEqual(self.window.alpha.getextrema(), (155, 155))

    def test_drag_waits_for_release_and_does_not_snap_back_while_loading(self):
        self.export(95)
        w = self.window
        w._load_media(str(self.video), str(self.folder))
        self.wait_idle()
        started, release = Event(), Event()
        execute = PreviewWorker.execute
        def slow(worker):
            started.set()
            release.wait(3)
            execute(worker)
        with patch.object(PreviewWorker, 'execute', slow):
            w.frame_slider.setSliderDown(True)
            w.frame_slider.setValue(3)
            QTest.qWait(300)
            self.assertIsNone(w._active_worker)
            self.assertFalse(w._navigation_timer.isActive())
            w.frame_slider.setValue(5)
            QTest.qWait(300)
            self.assertIsNone(w._active_worker)
            w.frame_slider.setSliderDown(False)
            try:
                self.assertTrue(started.wait(2))
                self.assertEqual(w.frame_slider.value(), 5)
                self.assertEqual(w.frame_spin.value(), 5)
                self.assertEqual(w.current_frame, 1)
                self.assertIn('frame 5', w.status.text())
            finally:
                release.set()
            self.wait_idle()
        self.assertEqual(w.current_frame, 5)
        self.assertEqual(w.frame_slider.value(), 5)
        self.assertIsNone(w._requested_frame)

    def test_rapid_keyboard_seeks_only_load_final_frame(self):
        w = self.window
        w._load_media(str(self.video))
        self.wait_idle()
        with patch.object(PreviewWorker, 'execute', autospec=True, side_effect=PreviewWorker.execute) as execute:
            w.request_frame(2)
            w.request_frame(4)
            w.request_frame(6)
            self.wait_idle()
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(w.current_frame, 6)
        self.assertEqual(w.frame_spin.value(), 6)

    def test_same_frame_does_not_reload_or_discard_matte(self):
        w = self.window
        w._load_media(str(self.video))
        self.wait_idle()
        w.alpha = Image.new('L', w.original.size, 177)
        w._reset_matte_edits()
        w.request_frame(1)
        QTest.qWait(300)
        self.assertIsNone(w._active_worker)
        self.assertEqual(w.alpha.getextrema(), (177, 177))

    def test_cancelled_frame_load_returns_controls_to_displayed_frame(self):
        w = self.window
        w._load_media(str(self.video))
        self.wait_idle()
        started, release = Event(), Event()
        execute = PreviewWorker.execute
        def slow(worker):
            started.set()
            release.wait(3)
            execute(worker)
        with patch.object(PreviewWorker, 'execute', slow):
            w.request_frame(5)
            w.navigate_to_frame()
            try:
                self.assertTrue(started.wait(2))
                self.assertEqual(w.frame_slider.value(), 5)
                w.cancel_jobs()
            finally:
                release.set()
            self.wait_idle()
        self.assertEqual(w.current_frame, 1)
        self.assertEqual(w.frame_slider.value(), 1)
        self.assertIsNone(w._requested_frame)

    def test_remove_individual_guidance_and_right_click(self):
        w = self.window
        w.original = Image.new('RGBA', (100, 100), 'white')
        w.show()
        w._refresh_canvas()
        w.add_point(.2, .3, True)
        w.add_point(.7, .8, False)
        w.add_box(.1, .1, .9, .9)
        w.guidance_picker.setCurrentIndex(1)
        w.remove_selected_guidance()
        self.assertEqual(w.points, [PointHint(.2, .3, True)])
        self.assertEqual(len(w.boxes), 1)
        app.processEvents()
        rect = w.canvas._target_rect()
        QTest.mouseClick(w.canvas, Qt.RightButton, pos=QPoint(round(rect.left()+.2*rect.width()), round(rect.top()+.3*rect.height())))
        self.assertEqual(w.points, [])
        self.assertEqual(len(w.boxes), 1)
        QTest.mouseClick(w.canvas, Qt.RightButton, pos=QPoint(round(rect.left()+.1*rect.width()), round(rect.top()+.5*rect.height())))
        self.assertEqual(w.boxes, [])
        w.add_point(.5, .5, True)
        w.view_tabs.setCurrentIndex(2)
        w.remove_guidance('point', 0)
        self.assertEqual(len(w.points), 1)

    def test_video_controls_below_preview_hidden_for_images(self):
        self.export()
        w = self.window
        w._load_media(str(self.video), str(self.folder))
        self.wait_idle()
        w.show()
        app.processEvents()
        self.assertGreater(w.video_panel.y(), w.canvas.y() + w.canvas.height()-1)
        self.assertTrue(w.play_button.isVisible())
        w.grab().save(str(Path(__file__).parent / 'playback-layout-preview.png'))
        image = self.root / 'image.png'
        Image.new('RGBA', (64,48), 'red').save(image)
        w._load_media(str(image))
        self.wait_idle()
        self.assertFalse(w.video_panel.isVisible())
        self.assertFalse(w.play_button.isVisible())
        self.assertFalse(w.resume_button.isVisible())
        self.assertFalse(w.load_matte_folder_button.isVisible())
        self.assertTrue(w.generate_button.isVisible())

    def test_play_complete_matted_video_then_inspect_last_frame(self):
        self.export(77)
        w = self.window
        w._load_media(str(self.video), str(self.folder))
        self.wait_idle()
        seen = []
        set_image = w.canvas.set_image
        def record(image, size=None):
            if isinstance(w._active_worker, PlaybackWorker):
                seen.append(image.copy())
            set_image(image, size)
        with patch.object(w.canvas, 'set_image', record):
            w.start_playback(1)
            self.assertEqual(w.play_button.text(), 'Pause')
            self.assertFalse(w.frame_slider.isEnabled())
            self.wait_idle()
        self.assertEqual(len(seen), 6)
        self.assertEqual(w.current_frame, 6)
        self.assertEqual(w.alpha.getextrema(), (77,77))
        self.assertEqual(w.original.tobytes(), load_frame(str(self.video), 6, Cancellation()).tobytes())
        self.assertFalse(w._playback_timer.isActive())
        self.assertTrue(w.view_tabs.isEnabled())
        self.assertFalse(self.errors.called)

    def test_pause_playback_and_resume_from_displayed_frame(self):
        self.export(88)
        w = self.window
        w._load_media(str(self.video), str(self.folder))
        self.wait_idle()
        w.start_playback(2)
        deadline = time.monotonic()+3
        while not w._playback_displayed and time.monotonic()<deadline:
            app.processEvents()
            time.sleep(.01)
        self.assertTrue(w._playback_displayed)
        frame = w.current_frame
        w.toggle_playback()
        self.wait_idle()
        self.assertEqual(w.current_frame, frame)
        self.assertEqual(w.alpha.getextrema(), (88,88))
        self.assertTrue(w.frame_slider.isEnabled())
        w.toggle_playback()
        self.wait_idle()
        self.assertEqual(w.current_frame, 6)

    def test_missing_saved_frame_stops_playback_without_source_fallback(self):
        self.export()
        (self.folder / 'frame_000003.png').unlink()
        w = self.window
        w._load_media(str(self.video), str(self.folder))
        self.wait_idle()
        with patch.object(QMessageBox, 'information') as message:
            w.start_playback(1)
            self.wait_idle()
            self.assertTrue(message.called)
            self.assertIn('No saved matte for frame 3', message.call_args.args[2])
        self.assertFalse(w._playback_timer.isActive())
        self.assertTrue(w.play_button.isEnabled())

    def test_playback_buffer_is_bounded_and_cancel_cleans_worker(self):
        self.export()
        worker = PlaybackWorker(str(self.folder), 1, 6)
        worker.start()
        deadline = time.monotonic()+3
        while worker.frames.qsize()<3 and time.monotonic()<deadline:
            app.processEvents()
            time.sleep(.01)
        self.assertEqual(worker.frames.qsize(), 3)
        self.assertTrue(worker.isRunning())
        worker.cancel()
        self.assertTrue(worker.wait(2000))



if __name__ == '__main__':
    unittest.main(verbosity=2)
