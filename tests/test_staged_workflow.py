import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import sys,json,tempfile,time,subprocess,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageDraw
from PySide6.QtWidgets import QApplication,QFileDialog,QListWidgetItem,QMessageBox
from matte_app.models import RotoHint,PointHint,make_model
from matte_app.roto import REFINERS,capabilities
from matte_app.saved_masks import make_trimap,mask_files,validate_trimap
from matte_app.cancellation import Cancellation
from matte_app.video import process_video,write_metadata,read_metadata
from matte_app.window import MatteWindow

app=QApplication.instance() or QApplication([])

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/'image.png'
        Image.new('RGB',(128,96),'red').save(self.source)
        self.masks=self.root/'masks'
        self.masks.mkdir()
        alpha=Image.new('L',(128,96),0)
        ImageDraw.Draw(alpha).rectangle((25,20,100,75),fill=255)
        image=Image.open(self.source).convert('RGBA')
        image.putalpha(alpha)
        image.save(self.masks/'frame_000001.png')
        write_metadata(self.masks,{'source_image':str(self.source),'frame_count':1,'model':'SAM 2 — Tracking','status':'complete'})
        self.w=MatteWindow()
        self.w.results_index_path=self.root/'links.json'
        self.w.queue_outputs={}
        self.w.queue_list.addItem(QListWidgetItem(str(self.source)))
        self.wait_idle()
        self.w.remember_output(str(self.source),str(self.masks))
        self.w._load_media(str(self.source),str(self.masks))
        self.wait_idle()

    def tearDown(self):
        self.w._matte_dirty=False
        if self.w._active_worker:
            self.w.cancel_jobs()
            self.wait_idle()
        self.w.close()
        self.temp.cleanup()

    def wait_idle(self):
        end=time.monotonic()+8
        while time.monotonic()<end:
            app.processEvents()
            if self.w._active_worker is None:return
            time.sleep(.01)
        self.fail('Worker timed out')

    def test_stage_hierarchy_menu_and_context_controls(self):
        w=self.w
        self.assertEqual(w.stage_picker.currentIndex(),0)
        self.assertEqual(w.model_picker.currentText(),'SAM 2 — Tracking')
        self.assertEqual([a.text() for a in w.menuBar().actions()],['File','Edit','View','Help'])
        w.show()
        w.view_tabs.setCurrentIndex(0)
        self.assertTrue(w.guidance_panel.isVisible())
        w.stage_picker.setCurrentIndex(1)
        self.assertEqual(w.model_picker.currentText(),REFINERS[0])
        self.assertEqual(w.generate_button.text(),'Refine edges')
        self.assertEqual(w._mask_input_folder,str(self.masks))
        self.assertFalse(w.guidance_panel.isVisible())
        self.assertFalse(w.advanced_panel.isVisible())
        w.advanced_toggle.setChecked(True)
        self.assertTrue(w.trimap_controls.isVisible())
        self.assertFalse(w.video_panel.isVisible())
        w.view_tabs.setCurrentIndex(1)
        self.assertTrue(w.paint_panel.isVisible())
        self.assertTrue(w.test_frame_button.isEnabled())
        self.assertFalse(w.queue_run_button.isEnabled())

    def test_trimap_three_regions_and_custom_validation(self):
        mask=Image.new('L',(100,100),0)
        ImageDraw.Draw(mask).rectangle((25,25,75,75),fill=255)
        trimap=make_trimap(mask,5)
        self.assertEqual(set(np.unique(np.asarray(trimap))),{0,128,255})
        self.assertEqual(trimap.getpixel((50,50)),255)
        self.assertEqual(trimap.getpixel((25,50)),128)
        self.assertEqual(trimap.getpixel((0,0)),0)
        self.assertEqual(validate_trimap(trimap,mask.size).size,mask.size)
        with self.assertRaises(ValueError):validate_trimap(Image.new('L',mask.size,127),mask.size)
        with self.assertRaises(ValueError):validate_trimap(trimap,(20,20))

    def test_refiners_reuse_saved_masks_without_tracker(self):
        w=self.w
        w.stage_picker.setCurrentIndex(1)
        for name in REFINERS:
            w.model_picker.setCurrentText(name)
            hint=w.job_guidance()[-1]
            self.assertEqual(hint.mask_folder,str(self.masks))
            self.assertEqual(capabilities(name)[0],'none')
            captured=[]
            def run_process(token,command,**kwargs):
                job=json.loads(Path(command[-1]).read_text(encoding='utf-8'))
                captured.append(job)
                Image.new('L',(128,96),150).save(Path(job['output'])/'alpha_000000.png')
                kwargs['on_poll']()
            before=(self.masks/'frame_000001.png').read_bytes()
            orig_exists = Path.exists
            with patch.object(Cancellation,'run_process',run_process), patch.object(Path, 'exists', autospec=True, side_effect=lambda p: True if '.roto-venv' in str(p) else orig_exists(p)):
                alpha=make_model(name).predict(w.original,[hint],Cancellation())
            self.assertEqual(alpha.getextrema(),(150,150))
            self.assertEqual(captured[0]['tracker'],'none')
            self.assertEqual(captured[0]['mask_paths'],[str(self.masks/'frame_000001.png')])
            self.assertEqual(before,(self.masks/'frame_000001.png').read_bytes())

    def test_mask_source_and_missing_frame_validation(self):
        self.assertEqual(len(mask_files(self.masks,str(self.source),1,1)),1)
        with self.assertRaises(ValueError):mask_files(self.masks,str(self.root/'other.png'),1,1)
        with self.assertRaises(ValueError):mask_files(self.masks,str(self.source),1,2)
        self.assertEqual(len(mask_files(self.masks,str(self.source),1,8,True)),1)

    def test_refinement_creates_separate_result_and_can_review_masks(self):
        w=self.w
        w.stage_picker.setCurrentIndex(1)
        before=(self.masks/'frame_000001.png').read_bytes()
        w.on_matte_ready(Image.new('L',w.original.size,150))
        w.save_frame_edits()
        refined=w.video_output_dir
        self.assertNotEqual(refined,str(self.masks))
        self.assertEqual(read_metadata(Path(refined))['source_masks'],str(self.masks))
        self.assertEqual(before,(self.masks/'frame_000001.png').read_bytes())
        entries=w._library()[str(self.source.resolve())]
        self.assertEqual(len(entries),2)
        index=w.result_picker.findData(str(self.masks))
        w.result_picker.setCurrentIndex(index)
        self.wait_idle()
        self.assertEqual(w.video_output_dir,str(self.masks))
        self.assertEqual(w.alpha.getextrema(),(0,255))

    def test_frame_edits_save_and_mark_refinement_stale(self):
        w=self.w
        w.stage_picker.setCurrentIndex(1)
        w.on_matte_ready(Image.new('L',w.original.size,150))
        w.save_frame_edits()
        refined=w.video_output_dir
        w._load_media(str(self.source),str(self.masks))
        self.wait_idle()
        w.view_tabs.setCurrentIndex(1)
        w.matte_editor.begin_stroke()
        w.paint_matte(.5,.5,.5,.5,0)
        w.finish_matte_stroke()
        self.assertTrue(w._matte_dirty)
        w.save_frame_edits()
        self.assertFalse(w._matte_dirty)
        with Image.open(self.masks/'frame_000001.png') as image:self.assertEqual(image.getchannel('A').getpixel((64,48)),0)
        entry=next(e for e in w._library()[str(self.source.resolve())] if e['folder']==refined)
        self.assertTrue(entry['stale'])
        w.undo_brush()
        self.assertTrue(w._matte_dirty)
        with patch.object(QMessageBox,'warning',return_value=QMessageBox.Cancel):self.assertFalse(w.guard_frame_edits())

    def test_project_reopens_masks_stage_guidance_zoom_and_view(self):
        w=self.w
        w.stage_picker.setCurrentIndex(1)
        w.points=[PointHint(.2,.4)]
        w.trimap_width.setValue(13)
        w.view_tabs.setCurrentIndex(1)
        w.canvas.set_zoom(2)
        project=self.root/'saved.matte.json'
        with patch.object(QFileDialog,'getSaveFileName',return_value=(str(project),'')):w.save_project()
        w.model_picker.setCurrentText('BiRefNet HR Matting')
        w.points=[]
        w.canvas.reset_zoom()
        with patch.object(QFileDialog,'getOpenFileName',return_value=(str(project),'')):w.open_project()
        self.wait_idle()
        self.assertEqual(w.model_picker.currentText(),REFINERS[0])
        self.assertEqual(w._mask_input_folder,str(self.masks))
        self.assertEqual(w.points,[PointHint(.2,.4)])
        self.assertEqual(w.trimap_width.value(),13)
        self.assertEqual(w.current_view(),'matte')
        self.assertEqual(w.canvas._zoom,2)

    def test_selected_range_exports_exact_frames_and_preserves_input(self):
        if not shutil.which('ffmpeg'):self.skipTest('FFmpeg required')
        video=self.root/'range.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i','color=red:size=64x48:rate=3:duration=2','-pix_fmt','yuv420p',str(video)],check=True)
        class Model:
            def predict(self,image,guidance,token):return Image.new('L',image.size,100)
        folder=self.root/'range'
        total=process_video(str(video),folder,Model(),'fake',[],Cancellation(),start_frame=3,end_frame=4)
        self.assertEqual(total,6)
        self.assertEqual([p.name for p in sorted(folder.glob('frame_*.png'))],['frame_000003.png','frame_000004.png'])
        self.assertEqual(read_metadata(folder)['status'],'partial')
        with self.assertRaises(ValueError):process_video(str(video),self.masks,Model(),'fake',[RotoHint(mask_folder=str(self.masks))],Cancellation())

    def test_partial_result_playback_uses_processed_range(self):
        if not shutil.which('ffmpeg'):self.skipTest('FFmpeg required')
        from matte_app.video import VideoInfo
        folder=self.root/'partial'
        folder.mkdir()
        for frame in (3,4):Image.new('RGBA',(128,96),(255,0,0,120)).save(folder/f'frame_{frame:06d}.png')
        self.w.video_path=str(self.root/'video.mp4')
        self.w.video_info=VideoInfo(6,3)
        self.w.video_output_dir=str(folder)
        from matte_app.window import PlaybackWorker
        with patch.object(PlaybackWorker,'start'):
            self.w.start_playback(1)
            worker=self.w._active_worker
            self.assertEqual((worker.start_frame,worker.end_frame),(3,4))
            self.w._active_worker=None
            self.w._playback_timer.stop()

if __name__=='__main__':unittest.main()
