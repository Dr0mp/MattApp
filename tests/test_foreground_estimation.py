import unittest,time,threading
import numpy as np
from PIL import Image
from matte_app.foreground import ForegroundEstimator
from matte_app.cancellation import Cancellation,JobCancelled

class ForegroundTests(unittest.TestCase):
    def test_binary_alpha_bypasses_worker(self):
        original=Image.new('RGB',(24,20),'red');alpha=Image.new('L',original.size,255)
        with ForegroundEstimator() as estimator:
            result=estimator.estimate(original,alpha)
            self.assertIsNone(estimator.session.process);self.assertEqual(result.tobytes(),original.tobytes())
    def test_real_estimation_removes_edge_colour_without_touching_alpha(self):
        width,height=96,48;x=np.arange(width);a=np.clip((24-np.abs(x-48))/12,0,1);alpha=Image.fromarray(np.tile(np.uint8(np.rint(a*255)),(height,1)))
        mask=np.asarray(alpha,dtype=float)/255;rgb=np.stack([mask,1-mask,np.zeros_like(mask)],axis=-1)
        original=Image.fromarray(np.uint8(np.rint(rgb*255)));before=alpha.tobytes();edges=(mask>.2)&(mask<.8)
        with ForegroundEstimator() as estimator:
            result=estimator.estimate(original,alpha,Cancellation());self.assertIsNotNone(estimator.session.process)
        self.assertIsNone(estimator.session.process);self.assertEqual(alpha.tobytes(),before)
        values=np.asarray(result);self.assertLess(values[...,1][edges].mean(),np.asarray(original)[...,1][edges].mean()-10)
        self.assertTrue(np.array_equal(values[mask==1],np.asarray(original)[mask==1]))
    def test_cancellation_stops_owned_cpu_process(self):
        original=Image.new('RGB',(512,512),'red');alpha=Image.new('L',original.size,128);token=Cancellation();timer=threading.Timer(.4,token.cancel)
        with ForegroundEstimator() as estimator:
            timer.start()
            try:
                with self.assertRaises(JobCancelled):estimator.estimate(original,alpha,token)
            finally:timer.cancel();timer.join()
            self.assertIsNone(estimator.session.process)

if __name__=='__main__':unittest.main()
