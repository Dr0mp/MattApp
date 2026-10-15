import os,sys,tempfile,time
from pathlib import Path
import unittest
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from PySide6.QtWidgets import QApplication,QFileDialog,QMessageBox
from matte_app.window import MatteWindow
from matte_app.video import write_metadata,read_metadata

app=QApplication.instance() or QApplication([])

class SaveImageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/'illustration.png'
        Image.new('RGB',(96,64),'orange').save(self.source)
        self.saved=self.root/'saved'
        self.saved.mkdir()
        Image.new('RGBA',(96,64),(255,165,0,80)).save(self.saved/'frame_000001.png')
        write_metadata(self.saved,{'source_image':str(self.source),'frame_count':1,'model':'SAM 2 — Tracking','status':'complete'})
        self.w=MatteWindow()
        self.w.results_index_path=self.root/'index.json'
        self.w.queue_outputs={}
        self.w.model_picker.setCurrentText('BiRefNet HR Matting')
        self.w._load_media(str(self.source),str(self.saved))
        self.wait_idle()

    def tearDown(self):
        self.w._matte_dirty=False
        self.w.close()
        self.temp.cleanup()

    def wait_idle(self):
        end=time.monotonic()+6
        while time.monotonic()<end:
            app.processEvents()
            if self.w._active_worker is None:return
            time.sleep(.01)
        self.fail('Worker timed out')

    def test_regenerate_and_paint_only_commit_when_save_is_clicked(self):
        w=self.w
        before=(self.saved/'frame_000001.png').read_bytes()
        original=self.source.read_bytes()
        w.on_matte_ready(Image.new('L',(96,64),150))
        w.on_matte_ready(Image.new('L',(96,64),190))
        self.assertEqual((self.saved/'frame_000001.png').read_bytes(),before)
        self.assertEqual(w.save_edits_button.text(),'Save to image')
        self.assertTrue(w.save_edits_button.isEnabled())
        w.view_tabs.setCurrentIndex(1)
        w.matte_editor.begin_stroke()
        w.paint_matte(.5,.5,.5,.5,0)
        w.finish_matte_stroke()
        w.save_edits_button.click()
        with Image.open(self.saved/'frame_000001.png') as image:
            self.assertEqual(image.getchannel('A').getpixel((48,32)),0)
            self.assertEqual(image.getchannel('A').getpixel((0,0)),190)
        self.assertEqual(original,self.source.read_bytes())
        self.assertEqual(w.video_output_dir,str(self.saved))
        self.assertFalse(w._matte_dirty)
        self.assertEqual(read_metadata(self.saved)['model'],'BiRefNet HR Matting')
        w._load_media(str(self.source),str(self.saved))
        self.wait_idle()
        self.assertEqual(w.alpha.getpixel((48,32)),0)

    def test_sam_regeneration_updates_existing_result_without_auto_new_folder(self):
        w=self.w
        w.model_picker.setCurrentText('SAM 2 — Tracking')
        before=(self.saved/'frame_000001.png').read_bytes()
        w.on_matte_ready(Image.new('L',(96,64),220))
        self.assertEqual(w.video_output_dir,str(self.saved))
        self.assertEqual((self.saved/'frame_000001.png').read_bytes(),before)
        self.assertEqual(len(w._library()[str(self.source.resolve())]),1)
        w.save_edits_button.click()
        self.assertEqual(len(w._library()[str(self.source.resolve())]),1)
        with Image.open(self.saved/'frame_000001.png') as image:self.assertEqual(image.getchannel('A').getextrema(),(220,220))

    def test_first_save_chooses_folder_once_and_later_saves_update_it(self):
        w=self.w
        w.video_output_dir=None
        w.on_matte_ready(Image.new('L',(96,64),120))
        folder=self.root/'new-image-result'
        with patch.object(QFileDialog,'getExistingDirectory',return_value=str(folder)) as choose:
            w.save_edits_button.click()
            w.on_matte_ready(Image.new('L',(96,64),170))
            w.save_edits_button.click()
        self.assertEqual(choose.call_count,1)
        with Image.open(folder/'frame_000001.png') as image:self.assertEqual(image.getchannel('A').getextrema(),(170,170))

    def test_refinement_saves_separately_then_regenerates_its_own_result(self):
        w=self.w
        w.stage_picker.setCurrentIndex(1)
        before=(self.saved/'frame_000001.png').read_bytes()
        w.on_matte_ready(Image.new('L',(96,64),120))
        target=w._image_save_target
        self.assertFalse(Path(target).exists())
        w.save_edits_button.click()
        self.assertNotEqual(w.video_output_dir,str(self.saved))
        self.assertEqual((self.saved/'frame_000001.png').read_bytes(),before)
        w.on_matte_ready(Image.new('L',(96,64),200))
        self.assertEqual(w._image_save_target,target)
        w.save_edits_button.click()
        with Image.open(Path(target)/'frame_000001.png') as image:self.assertEqual(image.getchannel('A').getextrema(),(200,200))

    def test_wrong_source_folder_is_rejected_before_any_overwrite(self):
        w=self.w
        write_metadata(self.saved,{'source_image':str(self.root/'other.png'),'model':'SAM 2 — Tracking'})
        before=(self.saved/'frame_000001.png').read_bytes()
        w.on_matte_ready(Image.new('L',(96,64),200))
        with patch.object(QMessageBox,'information') as message:w.save_edits_button.click()
        self.assertTrue(message.called)
        self.assertEqual((self.saved/'frame_000001.png').read_bytes(),before)
        self.assertTrue(w._matte_dirty)

    def test_unsaved_generated_matte_is_protected_on_navigation(self):
        w=self.w
        w.on_matte_ready(Image.new('L',(96,64),190))
        with patch.object(QMessageBox,'warning',return_value=QMessageBox.Cancel):
            w._load_media(str(self.source),str(self.saved))
        self.assertEqual(w.alpha.getextrema(),(190,190))
        self.assertIsNone(w._active_worker)

if __name__=='__main__':unittest.main()
