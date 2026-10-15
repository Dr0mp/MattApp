from pathlib import Path
import sys,traceback,faulthandler,ctypes
from datetime import datetime
root=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
logs=root/'logs';logs.mkdir(exist_ok=True);path=logs/'matte-app.log'
if path.exists() and path.stat().st_size>1_000_000:path.replace(path.with_suffix('.previous.log'))
with path.open('a',encoding='utf-8',buffering=1) as log:
    previous=(sys.stdout,sys.stderr);sys.stdout=sys.stderr=log;faulthandler.enable(file=log,all_threads=True)
    print(f'\n[{datetime.now().isoformat(timespec="seconds")}] Starting Matte App')
    try:
        from main import main
        code=main();print(f'Matte App exited: {code}')
    except Exception:
        traceback.print_exc();log.flush();ctypes.windll.user32.MessageBoxW(None,f'Matte App could not start. Details:\n{path}','Matte App',0x10);code=1
    finally:faulthandler.disable();sys.stdout,sys.stderr=previous
raise SystemExit(code)
