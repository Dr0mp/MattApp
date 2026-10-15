import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import tempfile,time,unittest,subprocess,shutil,sys,threading
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from PySide6.QtWidgets import QApplication,QMessageBox
from PySide6.QtCore import Qt,QPointF
from PySide6.QtTest import QTest
from PySide6.QtGui import QMouseEvent
from matte_app.app_window import StudioWindow
from matte_app.window import MatteWindow,PlaybackWorker
from matte_app.canvas import ImageCanvas
from matte_app.video import VideoInfo
from matte_app.result_store import ResultStore,key
from matte_app.pipeline_jobs import PipelineWorker,Job
from matte_app.models import PointHint,RotoHint
from matte_app.cancellation import Cancellation,JobCancelled

APP=QApplication.instance() or QApplication([])

def wait_until(predicate,timeout=12):
    end=time.monotonic()+timeout
    while not predicate() and time.monotonic()<end:APP.processEvents();time.sleep(.01)
    APP.processEvents()
    if not predicate():raise AssertionError('Timed out waiting for work')

class FakeModel:
    def __init__(self,name,calls,delay=0):self.name=name;self.calls=calls;self.delay=delay
    def predict(self,image,hints,token):
        self.calls.append((self.name,list(hints),image.size))
        if self.delay:token.wait(self.delay)
        token.check()
        if 'Saved masks' in self.name:
            option=next(h for h in hints if isinstance(h,RotoHint));assert (Path(option.mask_folder)/f'frame_{option.mask_frame:06d}.png').exists()
        return Image.new('L',image.size,128 if 'Saved masks' in self.name else 255)

class StudioTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);folder=self.root/'folder'/'images';folder.mkdir(parents=True)
        self.a=folder/'same.png';self.b=folder/'other.png'
        Image.new('RGB',(160,100),'red').save(self.a);Image.new('RGB',(160,100),'blue').save(self.b)
        self.original=self.a.read_bytes();self.calls=[];self.errors=[]
        self.warning=patch('matte_app.app_window.QMessageBox.warning',side_effect=lambda *args:self.errors.append(args[-1]) or QMessageBox.Ok);self.warning.start()
        self.model=patch('matte_app.pipeline_jobs.make_model',side_effect=lambda name:FakeModel(name,self.calls));self.model.start()
        self.ready=patch('matte_app.app_window.readiness',return_value='Ready locally.');self.ready.start()
        self.w=StudioWindow(self.root/'state');self.w.show();self.w.queue_list.addItems([str(self.a),str(self.b)])
        self.w._load_media(str(self.a));wait_until(lambda:self.w._active_worker is None)
    def tearDown(self):
        self.w.close();wait_until(lambda:not self.w._active_worker and not self.w._job_worker and not self.w._catalog_worker and not self.w._draft_worker);self.w.close();APP.processEvents()
        self.ready.stop();self.model.stop();self.warning.stop();self.temp.cleanup()
    def guide(self):self.w.add_point(.5,.5,True)
    def prepare_brush(self):
        self.w.store.save(self.a,'matte',1,self.w.original,Image.new('L',self.w.original.size,128))
        self.w.select_shell_tab(1);self.w.choose_tool('paint_black');self.w.brush_size.setValue(8);APP.processEvents()
    def test_matanyone2_auto_and_manual_are_distinct_from_version_one(self):
        self.w.pipeline_picker.setCurrentText('SAM 2 + MatAnyone 2')
        self.assertEqual(self.w.backend_refiner(),'MatAnyone 2 — Saved masks')
        self.w.workflow_tabs.setCurrentIndex(1);self.w.manual_refiner.setCurrentText('MatAnyone')
        self.assertEqual(self.w.backend_refiner(),'MatAnyone — Saved masks')
        self.w.manual_refiner.setCurrentText('MatAnyone 2');self.w.select_shell_tab(2)
        self.assertEqual(self.w.viewport_model.currentText(),'MatAnyone 2')
        from matte_app.roto import capabilities
        self.assertEqual(capabilities('MatAnyone 2 — Saved masks'),('none','matanyone2'))

    def test_estimated_cutout_export_preserves_alpha_and_original_and_reopens(self):
        self.prepare_brush();self.w.select_shell_tab(2);self.w.foreground_estimation.setChecked(True)
        expected=self.w.alpha.tobytes()
        with patch('matte_app.pipeline_jobs.ForegroundEstimator.estimate',side_effect=lambda estimator_image,alpha,token:Image.new('RGB',estimator_image.size,'blue')) as estimate:
            self.w.cutout_save_button.click();wait_until(lambda:not self.w._job_worker)
            self.assertEqual(estimate.call_count,1)
        with Image.open(self.w.store.frame(self.a,'cutout')) as result:
            self.assertEqual(result.getpixel((0,0)),(0,0,255,128));self.assertEqual(result.getchannel('A').tobytes(),expected)
        self.assertEqual(self.w.original.getpixel((0,0))[:3],(255,0,0));self.assertEqual(self.a.read_bytes(),self.original)
        self.assertEqual(self.w.cutout_source().getpixel((0,0))[:3],(0,0,255))
        self.assertTrue(self.w.store.metadata(self.a,'cutout')['foreground_sources']['1'])
        self.w._load_media(str(self.b));wait_until(lambda:not self.w._active_worker)
        self.w._load_media(str(self.a));wait_until(lambda:not self.w._active_worker)
        self.assertEqual(self.w.cutout_source().getpixel((0,0))[:3],(0,0,255))
        self.w.select_shell_tab(0);self.assertEqual(self.w.canvas._image.getpixel((0,0))[:3],(255,0,0))
        self.w.select_shell_tab(1);self.assertEqual(self.w.alpha.tobytes(),expected)
        with patch('matte_app.pipeline_jobs.ForegroundEstimator.estimate') as estimate:
            self.w.save_frame_edits();estimate.assert_not_called()

    def test_apply_masks_uses_foreground_estimation_for_each_queued_image(self):
        self.prepare_brush();self.w.store.save(self.b,'matte',1,Image.new('RGB',(160,100),'blue'),Image.new('L',(160,100),128))
        self.w.foreground_estimation.setChecked(True)
        with patch('matte_app.pipeline_jobs.ForegroundEstimator.estimate',side_effect=lambda image,alpha,token:Image.new('RGB',image.size,'green')) as estimate:
            self.w.apply_masks();wait_until(lambda:not self.w._job_worker)
            self.assertEqual(estimate.call_count,2)
        for source in (self.a,self.b):
            with Image.open(self.w.store.frame(source,'cutout')) as result:self.assertEqual(result.getpixel((0,0)),(0,128,0,128))

    def test_production_never_builds_legacy_ui_and_has_one_canvas(self):
        with patch.object(MatteWindow,'_build_ui',side_effect=AssertionError('Legacy UI built')):
            other=StudioWindow(self.root/'isolated-state')
            self.assertEqual(len(other.findChildren(ImageCanvas)),1)
            self.assertEqual(other.brush_size.receivers('2valueChanged(int)'),1)
            other.close();APP.processEvents()

    def test_real_file_click_requests_one_load(self):
        self.w.queue_list.blockSignals(True);self.w.queue_list.setCurrentRow(0);self.w.queue_list.blockSignals(False)
        with patch.object(self.w,'_load_media') as load:
            QTest.mouseClick(self.w.queue_list.viewport(),Qt.LeftButton,pos=self.w.queue_list.visualItemRect(self.w.queue_list.item(1)).center())
            self.assertEqual(load.call_count,1)

    def test_playback_menu_and_direct_saves_reject_stale_frame(self):
        self.prepare_brush();self.w.select_shell_tab(2)
        worker=PlaybackWorker(str(self.root),1,2);self.w._active_worker=worker
        self.w.video_info=VideoInfo(2,24);self.w.current_frame=2;self.w.sync_shell()
        try:
            with patch.object(self.w.store,'save') as save,patch.object(self.w,'stash_draft') as stash:
                for name in ('Export cutout…','Export matte…','Save frame edits'):
                    self.assertFalse(self.w.file_actions[name].isEnabled());self.w.file_actions[name].trigger()
                self.w.export_cutout();self.w.export_matte();self.w.save_frame_edits()
                save.assert_not_called();stash.assert_not_called()
        finally:
            self.w._active_worker=None;self.w.current_frame=1;self.w.video_info=None;worker.deleteLater()
        self.w._loaded_frame_identity=(key(self.a),2)
        with patch.object(self.w.store,'save') as save:self.w.export_cutout();save.assert_not_called()
        self.w._loaded_frame_identity=(key(self.a),1)

    def test_wheel_zoom_coalesces_image_rebuild(self):
        self.prepare_brush();self.w._zoom_refresh.stop()
        with patch.object(self.w,'_refresh_canvas',wraps=self.w._refresh_canvas) as render:
            for value in (1.1,1.2,1.3,1.4):self.w.canvas.set_zoom(value)
            self.assertEqual(render.call_count,0)
            wait_until(lambda:not self.w._zoom_refresh.isActive())
            self.assertEqual(render.call_count,1)

    def test_catalog_only_requests_changed_source_and_prunes_removed_files(self):
        wait_until(lambda:not self.w._catalog_worker and not self.w._catalog_timer.isActive())
        with patch('matte_app.app_window.CatalogWorker',wraps=__import__('matte_app.app_window',fromlist=['CatalogWorker']).CatalogWorker) as worker:
            self.w.stage_changed(str(self.a),'matte','complete')
            wait_until(lambda:not self.w._catalog_worker and not self.w._catalog_timer.isActive())
            self.assertEqual(set(worker.call_args.args[0]),{str(self.a)})
        self.assertIn(str(self.b),self.w.thumbnail_delegate.cache)
        self.w.queue_list.takeItem(1)
        self.assertNotIn(str(self.b),self.w.thumbnail_delegate.cache);self.assertNotIn(key(self.b),self.w._catalog)

    def test_empty_queue_selection_and_clear_are_safe(self):
        with patch.object(self.w,'_load_media') as load:
            self.w.review_queue_item(None)
            self.w.queue_list.setCurrentRow(0)
            load.reset_mock();self.w.queue_list.clear();self.w.review_queue_item(None)
            load.assert_not_called()

    def test_views_share_live_alpha_and_undo_without_stashing(self):
        self.prepare_brush();self.w.begin_edit_stroke();self.w.paint_matte(.5,.5,.5,.5,0);self.w.finish_matte_stroke()
        editor=self.w.matte_editor;alpha=self.w.alpha
        with patch.object(self.w,'stash_draft',wraps=self.w.stash_draft) as stash:
            self.w.select_shell_tab(2);self.assertIs(self.w.matte_editor,editor);self.assertIs(self.w.alpha,alpha)
            self.w.undo_brush();self.assertEqual(self.w.alpha.getpixel((80,50)),128)
            self.w.select_shell_tab(1);self.assertIs(self.w.matte_editor,editor);self.w.redo_brush()
            self.assertEqual(self.w.alpha.getpixel((80,50)),0);stash.assert_not_called()

    def test_original_frame_pipeline_auto_and_manual_uses_fresh_intermediate(self):
        self.guide();self.w.select_shell_tab(0)
        self.assertTrue(self.w.pipeline_frame_button.isVisible());self.assertFalse(self.w.pipeline_clip_button.isVisible())
        for mode in (0,1):
            with self.subTest(mode=mode):
                self.w.select_shell_tab(0);self.w.workflow_tabs.setCurrentIndex(mode)
                self.w.manual_refiner.setCurrentText('ViTMatte');self.calls.clear()
                self.w.pipeline_frame_button.click();wait_until(lambda:not self.w._job_worker)
                self.assertEqual(len(self.calls),2);self.assertEqual(self.w.alpha.getextrema(),(128,128))
                self.w.select_shell_tab(1);self.assertEqual(self.w.current_stage(),'refined')
                self.assertFalse(self.w.store.frame(self.a,'matte').exists())
        self.assertFalse(self.errors,self.errors)

    def test_brush_missing_release_and_real_shortcut_undo_redo(self):
        self.prepare_brush();canvas=self.w.canvas;rect=canvas._target_rect()
        start=QPointF(rect.left()+rect.width()*.3,rect.top()+rect.height()*.5)
        end=QPointF(rect.left()+rect.width()*.8,rect.top()+rect.height()*.5)
        QTest.mousePress(canvas,Qt.LeftButton,pos=start.toPoint())
        event=QMouseEvent(QMouseEvent.MouseMove,end,canvas.mapToGlobal(end.toPoint()),Qt.NoButton,Qt.NoButton,Qt.NoModifier)
        QApplication.sendEvent(canvas,event)
        self.assertFalse(canvas._brush_down);self.assertEqual(self.w.alpha.getpixel((128,50)),128)
        self.assertEqual(len(self.w.matte_editor.undo_strokes),1);self.assertTrue(self.w.undo_brush_action.isEnabled())
        QTest.mouseRelease(canvas,Qt.LeftButton,pos=end.toPoint());canvas.setFocus();APP.processEvents()
        QTest.keyClick(canvas,Qt.Key_Z,Qt.ControlModifier);APP.processEvents()
        self.assertEqual(self.w.alpha.getpixel((48,50)),128);self.assertTrue(self.w.redo_brush_action.isEnabled())
        QTest.keyClick(canvas,Qt.Key_Y,Qt.ControlModifier);APP.processEvents();self.assertEqual(self.w.alpha.getpixel((48,50)),0)
    def test_brush_refresh_skips_ui_and_disk_and_history_survives_save(self):
        self.prepare_brush();canvas=self.w.canvas;pos=canvas._target_rect().center().toPoint()
        with patch.object(self.w,'sync_shell',wraps=self.w.sync_shell) as sync,patch.object(self.w,'stash_draft',wraps=self.w.stash_draft) as draft:
            QTest.mousePress(canvas,Qt.LeftButton,pos=pos);self.w.refresh_paint_preview();QTest.mouseRelease(canvas,Qt.LeftButton,pos=pos)
            self.assertEqual(sync.call_count,0);self.assertEqual(draft.call_count,0)
        self.assertEqual(self.w.alpha.getpixel((80,50)),0)
        self.w.save_frame_edits();self.w.select_shell_tab(0);self.w.select_shell_tab(1)
        self.assertTrue(self.w.undo_brush_action.isEnabled());self.w.undo_brush_action.trigger();self.assertEqual(self.w.alpha.getpixel((80,50)),128)
    def test_combined_refined_cutout_opacity_paint_and_export(self):
        self.w.store.save(self.a,'refined',1,self.w.original,Image.new('L',self.w.original.size,0));self.w.select_shell_tab(2)
        self.assertFalse(self.w.shell_tabs.isTabVisible(3));self.assertEqual(self.w.shell_tabs.tabText(2),'Cutout')
        clear=self.w.canvas._image.getpixel((80,50))[:3];self.assertNotEqual(clear,(255,0,0))
        self.w.original_opacity.setValue(100);wait_until(lambda:not self.w._opacity_refresh.isActive());self.assertEqual(self.w.canvas._image.getpixel((80,50))[:3],(255,0,0))
        self.w.original_opacity.setValue(0);wait_until(lambda:not self.w._opacity_refresh.isActive())
        self.w.choose_tool('paint_white');canvas=self.w.canvas;pos=canvas._target_rect().center().toPoint();QTest.mouseClick(canvas,Qt.LeftButton,pos=pos)
        self.assertEqual(self.w.alpha.getpixel((80,50)),255);self.assertEqual(canvas._image.getpixel((80,50))[:3],(255,0,0))
        self.w.undo_brush();self.assertEqual(canvas._image.getpixel((80,50))[:3],clear)
        self.w.redo_brush();self.w.cutout_save_button.click();self.assertEqual(self.w.store.alpha(self.a,'cutout').getpixel((80,50)),255)
        self.w.alpha_view.setChecked(True);self.assertEqual(canvas._image.getpixel((80,50))[:3],(255,255,255))
    def test_refinement_uses_current_alpha_including_edits(self):
        self.prepare_brush()
        self.w.store.save(self.a,'refined',1,self.w.original,Image.new('L',self.w.original.size,64))
        self.w.store.choose_alpha(self.a,'refined',1);self.w.select_shell_tab(2)
        self.w.begin_edit_stroke();self.w.paint_matte(.5,.5,.5,.5,255);self.w.finish_matte_stroke()
        captured=[]
        class Refiner:
            def predict(model,image,hints,token):
                hint=next(h for h in hints if isinstance(h,RotoHint))
                with Image.open(Path(hint.mask_folder)/'frame_000001.png') as mask:captured.append((mask.getchannel('A').getpixel((80,50)),mask.getchannel('A').getpixel((0,0))))
                return Image.new('L',image.size,90)
        with patch('matte_app.pipeline_jobs.make_model',return_value=Refiner()):
            self.w.generate();wait_until(lambda:not self.w._job_worker)
        self.assertEqual(captured,[(255,64)]);self.assertEqual(self.w.alpha.getextrema(),(90,90))
        self.w.select_shell_tab(1);self.assertEqual(self.w.alpha.getextrema(),(90,90));self.assertFalse(self.w.matte_editor.undo_strokes)

    def test_idle_draft_saves_off_ui_thread(self):
        self.prepare_brush();self.w.begin_edit_stroke();self.w.paint_matte(.5,.5,.5,.5,0);self.w.finish_matte_stroke()
        self.w.save_idle_draft();self.assertIsNotNone(self.w._draft_worker)
        wait_until(lambda:self.w._draft_worker is None)
        with Image.open(self.w.draft_path(self.a,'matte',1)) as image:self.assertEqual(image.getpixel((80,50)),0)
    def test_combined_view_can_use_base_matte_before_refinement(self):
        self.prepare_brush();self.w.select_shell_tab(2);self.assertEqual(self.w.current_stage(),'matte');self.assertIsNotNone(self.w.alpha)
        self.w.generate();wait_until(lambda:not self.w._job_worker)
        self.assertEqual(self.w.current_stage(),'refined');self.assertEqual(self.w.alpha.getpixel((80,50)),128)
    def test_generate_save_refine_edit_and_autoreload(self):
        self.guide();self.w.select_shell_tab(1);self.w.generate();wait_until(lambda:self.w._job_worker is None)
        self.assertEqual(self.w.alpha.getpixel((80,50)),255)
        self.assertFalse(self.w.store.frame(self.a,'matte').exists(),'Generate must not overwrite saved outputs before Save')
        self.w.save_frame_edits();self.assertTrue(self.w.store.frame(self.a,'matte').exists())
        self.w.select_shell_tab(2);self.w.generate();wait_until(lambda:self.w._job_worker is None)
        self.assertEqual(self.w.alpha.getpixel((80,50)),128);self.w.save_frame_edits()
        self.w.select_shell_tab(0);self.assertEqual(self.w.canvas._image.getpixel((80,50))[:3],(255,0,0))
        self.w.select_shell_tab(2);self.w.alpha_view.setChecked(True);self.assertEqual(self.w.canvas._image.getpixel((80,50))[:3],(128,128,128));self.w.alpha_view.setChecked(False)
        self.w.select_shell_tab(3);self.assertNotEqual(self.w.canvas._image.getpixel((80,50))[:3],(255,0,0))
        self.w.select_shell_tab(2)
        self.w.choose_tool('paint_black');self.w.canvas.stroke_started.emit();self.w.paint_matte(.5,.5,.5,.5,0);self.w.finish_matte_stroke()
        self.assertEqual(self.w.alpha.getpixel((80,50)),0)
        self.w.select_shell_tab(1);self.assertEqual(self.w.alpha.getpixel((80,50)),0);self.assertEqual(self.w.store.alpha(self.a,'matte').getpixel((80,50)),255)
        self.w.select_shell_tab(2);self.assertEqual(self.w.alpha.getpixel((80,50)),0)
        self.w.undo_brush();self.assertEqual(self.w.alpha.getpixel((80,50)),128);self.w.redo_brush();self.assertEqual(self.w.alpha.getpixel((80,50)),0)
        self.w.save_frame_edits();self.w.select_shell_tab(3);self.w.export_cutout()
        self.assertEqual(self.w.store.alpha(self.a,'cutout').getpixel((80,50)),0)
        self.w._load_media(str(self.b));wait_until(lambda:self.w._active_worker is None)
        self.w._load_media(str(self.a));wait_until(lambda:self.w._active_worker is None)
        self.w.select_shell_tab(2);self.assertEqual(self.w.alpha.getpixel((80,50)),0)
        self.assertEqual(self.a.read_bytes(),self.original);self.assertFalse(self.errors,self.errors)
    def test_queue_guidance_autopipeline_and_apply_without_inference(self):
        self.guide();self.w._load_media(str(self.b));wait_until(lambda:self.w._active_worker is None);self.w.add_box(.1,.1,.9,.9)
        self.w.run_queue();wait_until(lambda:self.w._job_worker is None)
        for source in (self.a,self.b):
            for stage in ('matte','refined','cutout'):self.assertTrue(self.w.store.frame(source,stage).exists())
        self.assertEqual(len(self.calls),4)
        self.assertTrue(any(isinstance(h,PointHint) for h in self.calls[0][1]))
        self.assertFalse(any(isinstance(h,PointHint) for h in self.calls[2][1]))
        self.w.apply_masks();wait_until(lambda:self.w._job_worker is None);self.assertEqual(len(self.calls),4)
        self.assertEqual(self.w.queue_progress.value(),100);self.assertFalse(self.errors,self.errors)
    def test_new_native_models_workflow_selection(self):
        self.guide();self.w.queue_list.takeItem(1);self.w.pipeline_picker.setCurrentText('SAM2Matting Tiny — soft matte')
        self.assertEqual(self.w.backend_selector(),'SAM2Matting Tiny');self.assertIsNone(self.w.backend_refiner())
        self.w.run_queue();wait_until(lambda:not self.w._job_worker)
        self.assertEqual(self.calls[0][0],'SAM2Matting Tiny');self.assertTrue(self.w.store.frame(self.a,'cutout').exists())
        self.w.workflow_tabs.setCurrentIndex(1);self.w.manual_selector.setCurrentText('SAM 3.1');self.assertEqual(self.w.backend_selector(),'SAM 3.1 — Tracking')
    def test_redo_replaces_scribbles_and_history_for_each_tracker(self):
        self.guide();self.w.workflow_tabs.setCurrentIndex(1)
        for name in ('SAM 2','SAM 3','SAM 3.1','SAM2Matting Tiny'):
            with self.subTest(model=name):
                self.w.store.save(self.a,'matte',1,self.w.original,Image.new('L',self.w.original.size,0));self.w.store.choose_alpha(self.a,'matte',1)
                self.w.select_shell_tab(1);self.w.choose_tool('paint_black' if self.w.alpha.getpixel((80,50))==255 else 'paint_white');pos=self.w.canvas._target_rect().center().toPoint();QTest.mouseClick(self.w.canvas,Qt.LeftButton,pos=pos)
                self.w.save_idle_draft();self.w.manual_selector.setCurrentText(name);self.w.generate_button.click();wait_until(lambda:not self.w._job_worker and not self.w._draft_worker)
                self.assertEqual(self.w.alpha.getextrema(),(255,255));self.assertFalse(self.w.matte_editor.undo_strokes)
                with Image.open(self.w.draft_path(self.a,'matte',1)) as draft:self.assertEqual(draft.getextrema(),(255,255))
                self.w.select_shell_tab(0);self.w.select_shell_tab(1);self.assertEqual(self.w.alpha.getextrema(),(255,255))
        self.assertFalse(self.errors,self.errors)
    def test_matte_displays_final_refined_alpha_only(self):
        self.w.store.save(self.a,'matte',1,self.w.original,Image.new('L',self.w.original.size,255));self.w.store.save(self.a,'refined',1,self.w.original,Image.new('L',self.w.original.size,128))
        self.w.select_shell_tab(1);self.assertEqual(self.w.current_stage(),'refined');self.assertEqual(self.w.canvas._image.getpixel((80,50)),(128,128,128,255))
        self.w.original_opacity.setValue(100);self.w.alpha_view.setChecked(False);self.w._refresh_canvas();self.assertEqual(self.w.canvas._image.getpixel((80,50)),(128,128,128,255))
    def test_failed_redo_preserves_painted_draft_and_undo(self):
        self.prepare_brush();self.guide();QTest.mouseClick(self.w.canvas,Qt.LeftButton,pos=self.w.canvas._target_rect().center().toPoint());before=self.w.alpha.tobytes()
        with patch('matte_app.pipeline_jobs.make_model',side_effect=RuntimeError('Inference failed')):
            self.w.generate();wait_until(lambda:not self.w._job_worker)
        self.assertEqual(self.w.alpha.tobytes(),before);self.assertTrue(self.w.matte_editor.undo_strokes)
    def test_cancel_and_browse_during_job(self):
        self.guide();self.w._load_media(str(self.b));wait_until(lambda:self.w._active_worker is None);self.guide();self.w._load_media(str(self.a));wait_until(lambda:self.w._active_worker is None)
        with patch('matte_app.pipeline_jobs.make_model',side_effect=lambda name:FakeModel(name,self.calls,.6)):
            self.w.run_queue();wait_until(lambda:bool(self.calls));self.assertFalse(self.w.workflow_tabs.isEnabled());self.assertTrue(self.w.queue_list.isEnabled())
            self.w.review_queue_item(self.w.queue_list.item(1));wait_until(lambda:self.w.current_media_path==str(self.b) and not self.w._active_worker)
            self.assertFalse(self.w._follow);self.w.run_queue();wait_until(lambda:not self.w._job_worker)
        self.assertEqual(len(self.calls),1);self.assertFalse(self.w.store.frame(self.b,'matte').exists());self.assertEqual(self.w.queue_run_button.text(),'Run queue')
    def test_missing_guidance_and_rapid_navigation(self):
        self.w.run_queue();self.assertFalse(self.w._job_worker);self.assertIn('missing',self.w._queue_states[key(self.a)])
        self.w._load_media(str(self.a));self.w._load_media(str(self.b));self.w._load_media(str(self.a));wait_until(lambda:not self.w._active_worker)
        self.assertEqual(self.w.current_media_path,str(self.a))
    def test_load_shared_results_root_without_selected_queue_item(self):
        external=self.root/'exports'/'matted'
        for source,value in ((self.a,91),(self.b,177)):
            folder=external/source.stem;folder.mkdir(parents=True)
            image=Image.open(source).convert('RGBA');image.putalpha(Image.new('L',image.size,value));image.save(folder/'frame_000001.png')
        self.w.queue_list.setCurrentRow(-1)
        with patch('matte_app.app_window.QFileDialog.getExistingDirectory',return_value=str(external)):self.w.load_results()
        wait_until(lambda:not self.w._active_worker);self.w.select_shell_tab(1)
        self.assertEqual(self.w.alpha.getpixel((80,50)),91);self.assertFalse(self.errors,self.errors)
        reopened=ResultStore(legacy=self.w.queue_outputs);self.assertEqual(reopened.alpha(self.b,'matte').getpixel((0,0)),177)
    def test_modifier_points_and_selection_backwards(self):
        self.w.choose_tool('box');rect=self.w.canvas._target_rect();pos=rect.center().toPoint()
        QTest.mouseClick(self.w.canvas,Qt.LeftButton,Qt.ShiftModifier,pos);QTest.mouseClick(self.w.canvas,Qt.LeftButton,Qt.AltModifier,pos)
        self.assertEqual([p.positive for p in self.w.points],[True,False]);self.assertEqual(self.w.guidance_picker.currentData(),('point',1))
        self.w.remove_selected_guidance();self.assertEqual(self.w.guidance_picker.currentData(),('point',0))

    def test_project_preserves_workflow_channels_and_guidance(self):
        self.guide();self.w.foreground_estimation.setChecked(True);self.w.workflow_tabs.setCurrentIndex(1);self.w.manual_refiner.setCurrentText('ViTMatte');self.w.select_shell_tab(2)
        project=self.root/'project.matte.json'
        with patch('matte_app.app_window.QFileDialog.getSaveFileName',return_value=(str(project),'')):self.w.save_project()
        self.w.manual_refiner.setCurrentText('MatAnyone');self.w.clear_guidance()
        with patch('matte_app.app_window.QFileDialog.getOpenFileName',return_value=(str(project),'')):self.w.open_project()
        wait_until(lambda:self.w._active_worker is None)
        self.assertEqual(self.w.channel(),2);self.assertTrue(self.w.foreground_estimation.isChecked());self.assertEqual(self.w.manual_refiner.currentText(),'ViTMatte');self.assertEqual(len(self.w.points),1)

    def test_video_generate_seek_resume_and_playback(self):
        if not shutil.which('ffmpeg'):self.skipTest('FFmpeg missing')
        video=self.root/'folder'/'videos'/'clip.mp4';video.parent.mkdir()
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i','testsrc2=size=96x64:rate=4:duration=1','-pix_fmt','yuv420p',str(video)],check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        self.w.queue_list.clear();self.w.queue_list.addItem(str(video));self.w._load_media(str(video));wait_until(lambda:not self.w._active_worker);self.guide()
        self.w.select_shell_tab(0);self.assertTrue(self.w.pipeline_clip_button.isVisible());self.w.foreground_estimation.setChecked(True)
        with patch('matte_app.pipeline_jobs.ForegroundEstimator.estimate',side_effect=lambda image,alpha,token:Image.new('RGB',image.size,'blue')):
            self.w.pipeline_clip_button.click();wait_until(lambda:not self.w._job_worker)
        for number in range(1,5):
            with Image.open(self.w.store.frame(video,'cutout',number)) as saved:self.assertEqual(saved.getpixel((0,0))[:3],(0,0,255))
        self.assertEqual(self.w.store.frames(video,'matte'),[1,2,3,4]);self.assertEqual(self.w.store.frames(video,'refined'),[1,2,3,4])
        before=self.w.store.frame(video,'matte',1).read_bytes()
        self.w.request_frame(3);self.w.commit_slider_frame();wait_until(lambda:not self.w._active_worker);self.assertEqual(self.w.current_frame,3)
        self.w.select_shell_tab(0);self.w.pipeline_frame_button.click();wait_until(lambda:not self.w._job_worker)
        self.assertEqual(self.w.current_frame,3);self.assertTrue(self.w.draft_path(video,'refined',3).exists());self.assertEqual(self.w.store.frame(video,'matte',1).read_bytes(),before)
        self.w.select_shell_tab(1);self.w.resume_video();wait_until(lambda:not self.w._job_worker)
        self.assertEqual(self.w.store.frame(video,'matte',1).read_bytes(),before)
        self.w._load_media(str(video),frame=1);wait_until(lambda:not self.w._active_worker);self.w.select_shell_tab(3);self.w.start_playback(1)
        self.assertEqual(self.w._active_worker.folder,self.w.store.folder(video,'cutout'))
        wait_until(lambda:not self.w._active_worker,8);self.assertEqual(self.w.current_frame,4);self.assertFalse(self.errors,self.errors)

    def test_close_cancels_running_queue(self):
        self.guide()
        with patch('matte_app.pipeline_jobs.make_model',side_effect=lambda name:FakeModel(name,self.calls,1)):
            self.w.generate(); # Original intentionally has no action, invoke the method for lifecycle.
            wait_until(lambda:bool(self.calls));self.w.close_window_button.click();wait_until(lambda:not self.w._job_worker and not self.w._active_worker and not self.w._catalog_worker)
        self.assertFalse(self.w.isVisible())

class StoreTests(unittest.TestCase):
    def test_checkerboard_uses_small_tile_and_preserves_pattern(self):
        from matte_app.window import checkerboard
        with patch('matte_app.window.np.indices',side_effect=AssertionError('Full integer grid allocated')):
            image=checkerboard((97,73))
        for x,y in ((0,0),(23,23),(24,0),(0,24),(24,24),(96,72)):
            self.assertEqual(image.getpixel((x,y)),(55,59,67) if (x//24+y//24)%2 else (35,38,44))

    def test_single_frame_refinement_snapshots_only_that_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'clip.mp4';store=ResultStore();raw=Image.new('RGBA',(20,12),'red');info=VideoInfo(8,24)
            for n in range(1,9):store.save(source,'matte',n,raw,Image.new('L',raw.size,128),info=info)
            calls=[];failures=[]
            class Refiner:
                def predict(inner,image,hints,token):
                    option=next(h for h in hints if isinstance(h,RotoHint));self.assertEqual([p.name for p in Path(option.mask_folder).glob('frame_*.png')],['frame_000005.png'])
                    return Image.new('L',image.size,200)
            worker=PipelineWorker([Job(str(source),(('refined','ViTMatte — Saved masks'),),(),5,5,True)],store);worker.failed.connect(failures.append)
            with patch('matte_app.pipeline_jobs.probe_video',return_value=info),patch('matte_app.pipeline_jobs.load_frame',side_effect=lambda path,n,token:calls.append(n) or raw.copy()),patch('matte_app.pipeline_jobs.make_model',return_value=Refiner()):
                worker.start();self.assertTrue(worker.wait(5000));APP.processEvents()
            self.assertEqual(calls,[5,5]);self.assertFalse(failures,failures);worker.deleteLater();APP.processEvents()

    def test_implicit_redo_seed_uses_saved_alpha_not_brush_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'image.png';original=Image.new('RGB',(20,12),'red');original.save(source);store=ResultStore()
            store.save(source,'refined',1,original,Image.new('L',original.size,128))
            draft=store.media_folder(source)/'.drafts/matte/frame_000001.png';draft.parent.mkdir(parents=True);Image.new('L',original.size,0).save(draft)
            class SeedModel:
                def predict(inner,image,hints,token):
                    option=next(h for h in hints if isinstance(h,RotoHint))
                    with Image.open(option.seed_path) as seed:self.assertEqual(seed.getextrema(),(128,128))
                    return Image.new('L',image.size,255)
            failures=[];worker=PipelineWorker([Job(str(source),(('matte','SAM 2 — Tracking'),),(),temporary=True)],store);worker.failed.connect(failures.append)
            with patch('matte_app.pipeline_jobs.make_model',return_value=SeedModel()):
                worker.start();self.assertTrue(worker.wait(5000));APP.processEvents()
            self.assertFalse(failures,failures);worker.deleteLater();APP.processEvents()
    def test_tiled_undo_and_immutable_autosave_snapshot(self):
        from matte_app.matte_editor import MatteEditor
        editor=MatteEditor();alpha=Image.new('L',(4096,3072),128);editor.reset(alpha);editor.begin_stroke()
        self.assertEqual(editor._before,{})
        editor.paint((.1,.1),(.9,.9),12,0);editor.end_stroke()
        self.assertLess(editor.undo_strokes[-1].bytes,alpha.width*alpha.height)
        base,tiles=editor.immutable_snapshot();snapshot=base.copy()
        for tile,image in tiles.items():snapshot.paste(image,tile)
        editor.begin_stroke();editor.paint((.5,.5),(.5,.5),20,255);editor.end_stroke()
        self.assertEqual(snapshot.getpixel((2048,1536)),0);self.assertEqual(alpha.getpixel((2048,1536)),255)
        editor.undo();self.assertEqual(alpha.tobytes(),snapshot.tobytes());editor.undo();self.assertEqual(alpha.getextrema(),(128,128));editor.redo();self.assertEqual(alpha.getpixel((2048,1536)),0)
    def test_discover_native_results_root_and_stages(self):
        from matte_app.result_import import discover_results
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'folder'/'images';folder.mkdir(parents=True);source=folder/'subject.png'
            image=Image.new('RGB',(20,12),'red');image.save(source);store=ResultStore()
            for stage,value in (('matte',255),('refined',128),('cutout',128)):store.save(source,stage,1,image,Image.new('L',image.size,value))
            links,errors=discover_results(root/'matted');self.assertFalse(errors);self.assertEqual(links,{str(source):str(store.media_folder(source))})
            imported=ResultStore(legacy=links);self.assertEqual(imported.alpha(source,'refined').getpixel((0,0)),128)
    def test_discovery_rejects_ambiguous_and_wrong_size_legacy_masks(self):
        from matte_app.result_import import discover_results
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sources=[]
            for ext in ('png','jpg'):
                path=root/f'same.{ext}';Image.new('RGB',(20,12),'red').save(path);sources.append(str(path))
            folder=root/'matted'/'same';folder.mkdir(parents=True);Image.new('RGBA',(20,12),(0,0,0,128)).save(folder/'frame_000001.png')
            links,errors=discover_results(root/'matted',sources);self.assertFalse(links);self.assertTrue(errors)
            Image.new('RGBA',(2,2),(0,0,0,128)).save(folder/'frame_000001.png')
            links,errors=discover_results(root/'matted',sources[:1]);self.assertFalse(links);self.assertTrue(errors)
    def test_cutout_mask_choice_survives_reopening_per_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'image.png';image=Image.new('RGB',(20,12),'red');image.save(source)
            store=ResultStore()
            for frame in (1,2):
                for stage,value in (('matte',255),('refined',128)):store.save(source,stage,frame,image,Image.new('L',image.size,value))
            store.choose_alpha(source,'matte',1);store.choose_alpha(source,'refined',2)
            reopened=ResultStore();self.assertEqual(reopened.preferred_alpha(source,1),'matte');self.assertEqual(reopened.preferred_alpha(source,2),'refined')

    def test_root_collisions_binding_and_downstream_staleness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'folder'/'images';folder.mkdir(parents=True);source=folder/'subject.png';other=folder/'subject.jpg'
            image=Image.new('RGB',(20,12),'red');image.save(source);image.save(other)
            store=ResultStore();self.assertEqual(store.project_root(source),root)
            self.assertNotEqual(store.media_folder(source),store.media_folder(other))
            alpha=Image.new('L',image.size,255);store.save(source,'matte',1,image,alpha);store.save(source,'refined',1,image,Image.new('L',image.size,128));store.save(source,'cutout',1,image,alpha)
            store.save(source,'matte',1,image,Image.new('L',image.size,0));self.assertTrue(store.metadata(source,'refined')['stale']);self.assertEqual(store.alpha(source,'refined').getpixel((0,0)),128)
            store.legacy[key(other)]=store.folder(source,'matte')
            with self.assertRaises(ValueError):store.alpha(other,'matte')

    @unittest.skipUnless(os.name=='nt','Windows job ownership')
    def test_cancel_reaps_descendant_processes(self):
        import ctypes
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pid=root/'child.txt';parent=root/'parent.py'
            parent.write_text('import subprocess,sys,time,pathlib\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])\npathlib.Path('+repr(str(pid))+').write_text(str(p.pid))\ntime.sleep(60)\n')
            token=Cancellation()
            def cancel():
                end=time.monotonic()+5
                while not pid.exists() and time.monotonic()<end:time.sleep(.01)
                token.cancel()
            thread=threading.Thread(target=cancel);thread.start()
            with self.assertRaises(JobCancelled):token.run_process([sys.executable,str(parent)],capture_output=True)
            thread.join();self.assertTrue(pid.exists())
            kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.restype=ctypes.c_void_p;kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong];kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
            handle=kernel.OpenProcess(0x100000,False,int(pid.read_text()))
            if handle:self.assertEqual(kernel.WaitForSingleObject(handle,3000),0);kernel.CloseHandle(handle)

if __name__=='__main__':unittest.main()

