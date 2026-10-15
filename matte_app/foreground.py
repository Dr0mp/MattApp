"""Owned CPU foreground-colour estimation. A batch reuses one child and reaps it."""
from pathlib import Path
import json,tempfile,atexit,weakref
from PIL import Image
from .cancellation import Cancellation
from .model_session import Session

_ACTIVE_ESTIMATORS = weakref.WeakSet()

def cleanup_foreground():
    for estimator in list(_ACTIVE_ESTIMATORS):
        try:estimator.session.stop()
        except Exception:pass
    _ACTIVE_ESTIMATORS.clear()

atexit.register(cleanup_foreground)

class ForegroundEstimator:
    def __init__(self):
        self.session=Session()
        _ACTIVE_ESTIMATORS.add(self)
    def __enter__(self):return self
    def __exit__(self,*args):self.session.stop()
    def estimate(self,original,alpha,token=None):
        token=token or Cancellation();token.check()
        if original.size!=alpha.size:raise ValueError('Foreground alpha/source dimensions differ')
        import numpy as np
        values=np.asarray(alpha.convert('L'))
        if not np.any((values>0)&(values<255)):return original.convert('RGB')
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='matte-foreground-') as temporary:
            work=Path(temporary);original.convert('RGB').save(work/'image.png','PNG',compress_level=1);alpha.convert('L').save(work/'alpha.png','PNG',compress_level=1)
            request=work/'job.json';request.write_text(json.dumps({'image':str(work/'image.png'),'alpha':str(work/'alpha.png'),'output':str(work/'foreground.png')}),encoding='utf-8')
            self.session.run(root/'.venv/Scripts/python.exe',root/'matte_app/foreground_runner.py',request,token,lambda:None)
            token.check()
            with Image.open(work/'foreground.png') as image:return image.convert('RGB')
