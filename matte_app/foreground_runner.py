"""CPU-only PyMatting worker; communicates through atomic files, not GUI callbacks."""
import os
os.environ.setdefault('NUMBA_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,json,sys,traceback
from pathlib import Path
import numpy as np
from PIL import Image

def run_job(job):
    from pymatting import estimate_foreground_ml
    with Image.open(job['image']) as raw:image=np.asarray(raw.convert('RGB')).copy()
    with Image.open(job['alpha']) as raw:alpha=np.asarray(raw.convert('L')).copy()
    if image.shape[:2]!=alpha.shape:raise ValueError('Foreground alpha/source dimensions differ')
    foreground=estimate_foreground_ml(image.astype(np.float64)/255,alpha.astype(np.float64)/255)
    foreground=np.uint8(np.rint(np.clip(foreground,0,1)*255))
    foreground[alpha==255]=image[alpha==255]
    target=Path(job['output']);pending=target.with_suffix('.tmp');Image.fromarray(foreground).save(pending,'PNG',compress_level=1);pending.replace(target)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--server',action='store_true');parser.add_argument('--job');args=parser.parse_args()
    if args.server:
        for line in sys.stdin:
            path=Path(line.strip())
            try:run_job(json.loads(path.read_text(encoding='utf-8')));status={'ok':True}
            except Exception:status={'error':traceback.format_exc()}
            pending=path.parent/'server-status.tmp';pending.write_text(json.dumps(status),encoding='utf-8');pending.replace(path.parent/'server-status.json')
    else:run_job(json.loads(Path(args.job).read_text(encoding='utf-8')))
if __name__=='__main__':main()
