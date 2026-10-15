import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
from pathlib import Path
sys.path.insert(0, os.environ.get('MATTE_APP_TEST_ROOT',str(Path(__file__).resolve().parents[1])))
import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from PySide6.QtWidgets import QApplication, QMessageBox
from matte_app.models import PointHint, BoxHint, RotoHint, make_model, _MODEL_CACHE
from matte_app.roto import ROTO_MODELS, RotoCommandModel, capabilities
from matte_app.cancellation import Cancellation, JobCancelled
from matte_app.video import process_video, read_metadata, VideoInfo
from matte_app.window import MatteWindow

app = QApplication.instance() or QApplication([])

class Sequence:
    def __init__(self,cancel=False): self.calls=[]; self.cancel=cancel
    def predict(self,*args): raise AssertionError('Temporal model used as independent frame predictions')
    def predict_sequence(self,frames,guidance,token,start,consume,source):
        self.calls.append((start,len(frames),source,list(guidance)))
        for i,path in enumerate(frames):
            with Image.open(path) as image: alpha=Image.new('L', image.size, 130+i)
            consume(i,alpha)
            if self.cancel: token.cancel()
            token.check()

class RotoTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.w=MatteWindow()
        self.w.model_picker.setCurrentText("BiRefNet HR Matting")
        self.w.original=Image.new('RGBA',(64,48),'white')
        self.w.alpha=Image.new('L',(64,48),150)
        self.w.current_media_path=str(self.root/'source.mp4')
        self.w.video_path=self.w.current_media_path
        self.w.video_info=VideoInfo(4,2)
        self.w._update_navigation()
        self.w._refresh_canvas()

    def tearDown(self):
        self.w.close()
        self.temp.cleanup()
        _MODEL_CACHE.clear()

    def video(self):
        if not shutil.which('ffmpeg'): self.skipTest('FFmpeg needed')
        path=self.root/'source.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i','testsrc2=size=64x48:rate=2:duration=2','-pix_fmt','yuv420p',str(path)],check=True)
        return path

    def test_new_models_have_explicit_providers(self):
        for name in ROTO_MODELS:
            self.assertIsInstance(make_model(name),RotoCommandModel)
        self.assertEqual(capabilities('SAM 3 + MatAnyone'),('sam3','matanyone'))
        self.assertEqual(capabilities('SAM 2 + VideoMaMa'),('sam2','videomama'))
        with self.assertRaises(ValueError): make_model('misspelled model')

    def test_model_specific_controls_and_guidance(self):
        w=self.w
        w.show()
        w.advanced_toggle.setChecked(True)
        w.model_picker.setCurrentText('SAM 2 + MatAnyone')
        self.assertFalse(w.roto_text.isVisible())
        self.assertFalse(w.roto_chunk.isVisible())
        w.model_picker.setCurrentText('SAM 3 + VideoMaMa')
        self.assertTrue(w.roto_text.isVisible())
        self.assertTrue(w.roto_chunk.isVisible())
        w.roto_text.setText('person in red')
        self.assertEqual(w.job_guidance()[-1].text,'person in red')
        w.view_tabs.setCurrentIndex(1)
        self.assertTrue(w.roto_panel.isVisible())
        w.model_picker.setCurrentIndex(0)
        self.assertNotIn(RotoHint, [type(h) for h in w.job_guidance()])

    def test_seed_frame_binding_and_queue_guard(self):
        w=self.w
        w.model_picker.setCurrentText('SAM 2 + MatAnyone')
        w.current_frame=3
        with patch('matte_app.window.readiness',return_value='Ready locally.'),patch.object(QMessageBox,'information') as message:
            w._save_roto_seed(w.alpha)
            self.assertTrue(Path(w.seed_path).exists())
            self.assertEqual(w.seed_frame,3)
            self.assertTrue(w._validate_roto(3))
            self.assertFalse(w._validate_roto(1))
            self.assertFalse(w._validate_roto(queue=True))
            w.clear_roto_seed()
            self.assertFalse(w._validate_roto(3))
            w.add_point(.5,.5,True)
            self.assertTrue(w._validate_roto(3))
        Path(w.seed_path).unlink(missing_ok=True) if w.seed_path else None

    def test_save_restore_roto_options(self):
        w=self.w
        w.model_picker.setCurrentText('SAM 3 + VideoMaMa')
        w.roto_text.setText('person')
        w.roto_size.setValue(768)
        w.roto_chunk.setValue(6)
        w.points=[PointHint(.3,.4)]
        hints=w.job_guidance()
        metadata={'model':w.model_picker.currentText(),'guidance':[h.__dict__|{'type':type(h).__name__} for h in hints]}
        (self.root/'frames.json').write_text(json.dumps(metadata),encoding='utf-8')
        w.roto_text.clear()
        w._restore_saved_guidance(str(self.root),1)
        self.assertEqual(w.roto_text.text(),'person')
        self.assertEqual(w.roto_size.value(),768)
        self.assertEqual(w.roto_chunk.value(),6)
        self.assertEqual(w.points,[PointHint(.3,.4)])

    def test_temporal_resume_offsets_preserve_earlier_frames(self):
        source=self.video(); folder=self.root/'out'
        process_video(str(source),folder,Sequence(),'SAM 2 — Tracking',[],Cancellation())
        before=(folder/'frame_000001.png').read_bytes()
        model=Sequence()
        process_video(str(source),folder,model,'SAM 2 + MatAnyone',[RotoHint()],Cancellation(),3)
        self.assertEqual(model.calls[0][:2],(3,2))
        self.assertEqual((folder/'frame_000001.png').read_bytes(),before)
        with Image.open(folder/'frame_000004.png') as image: self.assertEqual(image.getchannel('A').getextrema(),(131,131))
        self.assertEqual(read_metadata(folder)['runs'][-1]['last_frame'],4)

    def test_temporal_cancel_keeps_completed_and_unprocessed_results(self):
        source=self.video(); folder=self.root/'out'
        process_video(str(source),folder,Sequence(),'SAM 2 — Tracking',[],Cancellation())
        later=(folder/'frame_000004.png').read_bytes()
        with self.assertRaises(JobCancelled): process_video(str(source),folder,Sequence(cancel=True),'SAM 2 + MatAnyone',[],Cancellation(),2)
        self.assertEqual((folder/'frame_000004.png').read_bytes(),later)
        run=read_metadata(folder)['runs'][-1]
        self.assertEqual(run['status'],'cancelled')
        self.assertEqual(run['last_frame'],2)

    def test_adapter_streams_completed_files_before_cancel(self):
        runtime=self.root/'.roto-venv/Scripts'
        runtime.mkdir(parents=True)
        (runtime/'python.exe').touch()
        scripts=self.root/'matte_app'; scripts.mkdir()
        (scripts/'roto_runner.py').write_text("import json,sys,time\nfrom PIL import Image\nfrom pathlib import Path\nj=json.loads(Path(sys.argv[2]).read_text())\nImage.new('L',(64,48),177).save(Path(j['output'])/'alpha_000000.png')\ntime.sleep(20)\n",encoding='utf-8')
        paths=[]
        for i in range(2):
            p=self.root/f'{i}.png'; self.w.original.save(p); paths.append(p)
        class Token(Cancellation):
            def run_process(self,command,**kwargs): return super().run_process([sys.executable,*command[1:]],**kwargs)
        token=Token(); received=[]
        model=RotoCommandModel('SAM 2 — Tracking'); model.root=self.root
        def consume(i,alpha): received.append((i,alpha.getextrema())); token.cancel()
        with self.assertRaises(JobCancelled): model.predict_sequence(paths,[],token,1,consume)
        self.assertEqual(received,[(0,(177,177))])

if __name__=='__main__': unittest.main()
