import os,sys,tempfile,time,json,subprocess
from pathlib import Path
from threading import Timer
from unittest.mock import patch
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,UnidentifiedImageError
from matte_app.alpha_collector import AlphaCollector
from matte_app.cancellation import Cancellation,JobCancelled
from matte_app.roto import RotoCommandModel

class AlphaLockTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        for i in range(2): Image.new('L',(16,12),110+i).save(self.root/f'alpha_{i:06d}.png')
        self.received=[]
        self.token=Cancellation()

    def tearDown(self):self.temp.cleanup()

    def collector(self,**kwargs):
        return AlphaCollector(self.root,lambda i,a:self.received.append((i,a.getextrema())),self.token,**kwargs)

    def test_transient_lock_retries_same_frame_without_reordering_or_duplicates(self):
        collector=self.collector()
        original=Image.open
        failures=2
        def opening(path,*args,**kwargs):
            nonlocal failures
            if Path(path).name=='alpha_000000.png' and failures:
                failures-=1
                raise PermissionError('Temporary sharing violation')
            return original(path,*args,**kwargs)
        with patch('matte_app.alpha_collector.Image.open',side_effect=opening):
            collector()
            collector()
            self.assertEqual(self.received,[])
            collector()
            collector()
        self.assertEqual(self.received,[(0,(110,110)),(1,(111,111))])
        self.assertEqual(collector.delivered,2)

    def test_final_flush_retries_after_writer_exits(self):
        collector=self.collector(retry_interval=.001)
        original=Image.open
        calls=0
        def opening(path,*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls<=2:raise PermissionError('Locked at process exit')
            return original(path,*args,**kwargs)
        with patch('matte_app.alpha_collector.Image.open',side_effect=opening):collector.finish()
        self.assertEqual(collector.delivered,2)

    def test_persistent_lock_has_bounded_timeout(self):
        collector=self.collector(timeout=.02,retry_interval=.001)
        began=time.monotonic()
        with patch('matte_app.alpha_collector.Image.open',side_effect=PermissionError('Persistent denial')):
            with self.assertRaisesRegex(RuntimeError,'kept the matte file locked'):collector.finish()
        self.assertLess(time.monotonic()-began,.5)
        self.assertEqual(collector.delivered,0)

    def test_cancel_interrupts_retry_wait(self):
        collector=self.collector(retry_interval=2)
        timer=Timer(.05,self.token.cancel)
        began=time.monotonic()
        timer.start()
        try:
            with patch('matte_app.alpha_collector.Image.open',side_effect=PermissionError('Locked')):collector.finish()
        finally:timer.join()
        self.assertLess(time.monotonic()-began,.5)
        self.assertTrue(self.token.requested)

    def test_corrupt_png_is_reported_instead_of_retried(self):
        (self.root/'alpha_000000.png').write_bytes(b'not a PNG')
        with self.assertRaises(UnidentifiedImageError):self.collector().finish()

    def test_callback_failure_is_not_repeated(self):
        calls=[]
        def consume(i,alpha):
            calls.append(i)
            raise ValueError('Export failed')
        collector=AlphaCollector(self.root,consume,self.token)
        with self.assertRaisesRegex(ValueError,'Export failed'):collector()
        collector.finish()
        self.assertEqual(calls,[0])

    @unittest.skipUnless(os.name=='nt','Windows sharing lock')
    def test_real_windows_exclusive_file_lock(self):
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.CreateFileW.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        kernel.CreateFileW.restype=wintypes.HANDLE
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        kernel.CloseHandle.restype=wintypes.BOOL
        path=self.root/'alpha_000000.png'
        handle=kernel.CreateFileW(str(path),0x80000000,0,None,3,0,None)
        self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
        closed=False
        try:
            with self.assertRaises(PermissionError):Image.open(path)
            collector=self.collector()
            collector()
            self.assertEqual(collector.delivered,0)
            kernel.CloseHandle(handle)
            closed=True
            collector.finish()
            self.assertEqual(collector.delivered,2)
        finally:
            if not closed:kernel.CloseHandle(handle)

    def adapter(self):
        runtime=self.root/'.roto-venv/Scripts'
        runtime.mkdir(parents=True)
        (runtime/'python.exe').touch()
        frame=self.root/'source.png'
        Image.new('RGB',(16,12),'red').save(frame)
        model=RotoCommandModel('SAM 2 — Tracking')
        model.root=self.root
        return model,[frame]

    def test_adapter_preserves_original_process_error_during_flush(self):
        model,frames=self.adapter()
        def failing(command,**kwargs):
            job=json.loads(Path(command[-1]).read_text())
            Image.new('L',(16,12),177).save(Path(job['output'])/'alpha_000000.png')
            raise subprocess.CalledProcessError(1,command,stderr='Original inference failure')
        with patch.object(self.token,'run_process',side_effect=failing),patch('matte_app.alpha_collector.Image.open',side_effect=PermissionError('Locked')):
            with self.assertRaisesRegex(RuntimeError,'Original inference failure'):model.predict_sequence(frames,[],self.token,1,lambda *a:None)

    def test_adapter_preserves_cancel_when_final_frame_is_locked(self):
        model,frames=self.adapter()
        def cancelled(command,**kwargs):
            job=json.loads(Path(command[-1]).read_text())
            Image.new('L',(16,12),177).save(Path(job['output'])/'alpha_000000.png')
            self.token.cancel()
            raise JobCancelled()
        with patch.object(self.token,'run_process',side_effect=cancelled),patch('matte_app.alpha_collector.Image.open',side_effect=PermissionError('Locked')):
            with self.assertRaises(JobCancelled):model.predict_sequence(frames,[],self.token,1,lambda *a:None)

if __name__=='__main__':unittest.main()
