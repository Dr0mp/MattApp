import os,sys,json,time,tempfile
from pathlib import Path
from unittest.mock import patch
import unittest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from PySide6.QtWidgets import QApplication,QFileDialog,QMessageBox
from matte_app.window import MatteWindow,QueueWorker
from matte_app.models import PointHint,BoxHint,RotoHint
from matte_app.guidance_store import GuidanceStore,source_key
from matte_app.video import read_metadata,write_metadata

app=QApplication.instance() or QApplication([])

class PerFileGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.a=self.root/'a'/'same name.png'
        self.b=self.root/'b'/'same name.png'
        self.c=self.root/'unguided.png'
        for path,color in ((self.a,'red'),(self.b,'blue'),(self.c,'green')):
            path.parent.mkdir(exist_ok=True)
            Image.new('RGB',(32,24),color).save(path)
        self.ready_patch = patch('matte_app.guidance_state.readiness', return_value='Ready')
        self.ready_patch.start()
        self.w=MatteWindow()
        self.w.results_index_path=self.root/'links.json'
        self.w.queue_outputs={}
        self.w.queue_list.addItems([str(self.a),str(self.b)])
        self.wait_idle()

    def tearDown(self):
        self.ready_patch.stop()
        if self.w._active_worker:self.w.cancel_jobs();self.wait_idle()
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

    def select(self,path):
        self.w._load_media(str(path),self.w.output_for(str(path)))
        self.wait_idle()
        self.w.view_tabs.setCurrentIndex(0)

    def prepare(self):
        self.select(self.a)
        self.w.add_point(.2,.3,True)
        self.w.add_box(.1,.1,.4,.8)
        self.select(self.b)
        self.assertEqual(self.w.points,[])
        self.w.add_point(.8,.6,True)
        self.w.add_point(.1,.2,False)

    def test_switch_restore_and_restart_before_processing(self):
        self.prepare()
        self.select(self.a)
        self.assertEqual(self.w.points,[PointHint(.2,.3)])
        self.assertEqual(self.w.boxes,[BoxHint(.1,.1,.4,.8)])
        self.w.close()
        self.w=MatteWindow()
        self.w.results_index_path=self.root/'links.json'
        self.w.queue_outputs={}
        self.w.queue_list.addItems([str(self.a),str(self.b)])
        self.wait_idle()
        self.select(self.b)
        self.assertEqual(self.w.points,[PointHint(.8,.6),PointHint(.1,.2,False)])
        self.assertEqual(self.w.boxes,[])

    def test_cleared_guidance_does_not_return_from_old_export(self):
        self.prepare()
        self.select(self.a)
        self.w.clear_guidance()
        folder=self.root/'old'
        folder.mkdir()
        image=Image.new('RGBA',(32,24),(255,0,0,100))
        image.save(folder/'frame_000001.png')
        write_metadata(folder,{'source_image':str(self.a),'model':'SAM 2 — Tracking','frame_count':1,'guidance':[{'type':'PointHint','x':.9,'y':.9,'positive':True}]})
        self.select(self.b)
        self.w._load_media(str(self.a),str(folder),restore_guidance=True)
        self.wait_idle()
        self.assertEqual(self.w.points,[])
        self.assertEqual(self.w.boxes,[])

    def test_remove_element_is_persisted_for_only_that_image(self):
        self.prepare()
        self.w.remove_guidance('point',1)
        self.select(self.a)
        self.select(self.b)
        self.assertEqual(self.w.points,[PointHint(.8,.6)])
        self.assertEqual(self.w.guidance_store().get(str(self.a))[0],PointHint(.2,.3))

    def test_gui_queue_applies_each_files_guidance_and_records_it(self):
        self.prepare()
        calls=[]
        class Model:
            def predict(self,image,guidance,token):
                calls.append((image.getpixel((0,0))[:3],list(guidance)))
                point=next(h for h in guidance if isinstance(h,PointHint) and h.positive)
                return Image.new('L',image.size,round(point.x*255))
        with patch.object(QFileDialog,'getExistingDirectory',return_value=str(self.root/'batch')),patch('matte_app.window.make_model',return_value=Model()):
            self.w.run_queue()
            self.wait_idle()
        self.assertEqual(calls[0][1][0],PointHint(.2,.3))
        self.assertEqual(calls[1][1][0],PointHint(.8,.6))
        self.assertIn(BoxHint(.1,.1,.4,.8),calls[0][1])
        self.assertNotIn(BoxHint(.1,.1,.4,.8),calls[1][1])
        first=self.root/'batch'/'same name'
        second=self.root/'batch'/'same name_2'
        self.assertEqual(read_metadata(first)['guidance'][0]['x'],.2)
        self.assertEqual(read_metadata(second)['guidance'][0]['x'],.8)
        with Image.open(first/'frame_000001.png') as image:self.assertEqual(image.getchannel('A').getextrema(),(51,51))
        with Image.open(second/'frame_000001.png') as image:self.assertEqual(image.getchannel('A').getextrema(),(204,204))
        self.assertEqual(self.w.points,[PointHint(.8,.6),PointHint(.1,.2,False)])

    def test_queue_missing_guidance_preflight_lists_only_missing_files(self):
        self.prepare()
        self.w.queue_list.addItem(str(self.c))
        with patch.object(QMessageBox,'information') as message,patch.object(QFileDialog,'getExistingDirectory') as folder:
            self.w.run_queue()
        self.assertFalse(folder.called)
        self.assertIn('unguided.png',message.call_args.args[2])
        self.assertNotIn('same name.png',message.call_args.args[2])

    def test_no_guidance_inherits_from_another_file_for_automatic_model(self):
        self.prepare()
        self.w.model_picker.setCurrentText('BiRefNet HR Matting')
        snapshot=self.w.queue_guidance_snapshot([str(self.a),str(self.c)])
        self.assertEqual(snapshot[source_key(self.c)],[])
        self.assertEqual(snapshot[source_key(self.a)][0],PointHint(.2,.3))

    def test_worker_takes_immutable_snapshot_and_video_uses_own_guidance(self):
        a,b=str(self.root/'a.mp4'),str(self.root/'b.mp4')
        mapping={a:[PointHint(.2,.2)],b:[BoxHint(.4,.4,.9,.9)]}
        worker=QueueWorker([a,b],str(self.root/'videos'),'fake',[],mapping)
        mapping[a].clear()
        calls=[]
        def process(path,folder,model,name,guidance,token,**kwargs):calls.append((path,list(guidance)));return 1
        with patch('matte_app.window.make_model',return_value=object()),patch('matte_app.window.process_video',side_effect=process):worker.execute()
        self.assertEqual(calls,[(a,[PointHint(.2,.2)]),(b,[BoxHint(.4,.4,.9,.9)])])

    def test_project_contains_all_guidance_and_restores_it(self):
        self.prepare()
        project=self.root/'project.matte.json'
        with patch.object(QFileDialog,'getSaveFileName',return_value=(str(project),'')):self.w.save_project()
        document=json.loads(project.read_text())
        self.assertEqual(len(document['guidance_by_file']),2)
        self.w.guidance_store().write({})
        self.w.points=[]
        self.w.boxes=[]
        with patch.object(QFileDialog,'getOpenFileName',return_value=(str(project),'')):self.w.open_project()
        self.wait_idle()
        self.select(self.a)
        self.assertEqual(self.w.points,[PointHint(.2,.3)])
        self.select(self.b)
        self.assertEqual(self.w.points,[PointHint(.8,.6),PointHint(.1,.2,False)])

    def test_sam3_text_is_file_specific(self):
        self.select(self.a)
        self.w.roto_text.setText('person in red')
        self.w.save_file_guidance()
        self.select(self.b)
        self.w.roto_text.setText('blue bag')
        self.select(self.a)
        self.assertEqual(self.w.roto_text.text(),'person in red')
        self.select(self.b)
        self.assertEqual(self.w.roto_text.text(),'blue bag')

    def test_each_file_can_have_its_own_starting_mask(self):
        for path in (self.a,self.b):
            self.select(path)
            mask=self.root/(path.parent.name+'_mask.png')
            Image.new('L',(32,24),255).save(mask)
            self.w.seed_path,self.w.seed_source,self.w.seed_frame=str(mask),str(path),1
            self.w.save_file_guidance()
        snapshot=self.w.queue_guidance_snapshot([str(self.a),str(self.b)])
        self.assertIsNotNone(snapshot)
        for path in (self.a,self.b):
            option=next(h for h in snapshot[source_key(path)] if isinstance(h,RotoHint))
            self.assertEqual(option.seed_source,str(path))
            self.assertEqual(Path(option.seed_path).name,path.parent.name+'_mask.png')

    def test_litesdmatte_still_requires_points_or_box_not_sam_seed(self):
        self.select(self.a)
        mask=self.root/'seed.png'
        Image.new('L',(32,24),255).save(mask)
        self.w.seed_path,self.w.seed_source,self.w.seed_frame=str(mask),str(self.a),1
        self.w.save_file_guidance()
        self.w.model_picker.setCurrentText('LiteSDMatte (guided)')
        with patch.object(QMessageBox,'information') as message:
            self.assertIsNone(self.w.queue_guidance_snapshot([str(self.a)]))
        self.assertIn('same name.png',message.call_args.args[2])

if __name__=='__main__':unittest.main()
