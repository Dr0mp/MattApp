from __future__ import annotations
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFontDatabase,QFont
from matte_app.app_window import StudioWindow
from matte_app.console_stream import ConsoleStream
from pathlib import Path
from PySide6.QtCore import QLockFile

def main():
    app=QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName('Matte App');app.setStyle('Fusion')
    lock_path=Path(__file__).parent/'logs'/'app.lock';lock_path.parent.mkdir(exist_ok=True)
    lock=QLockFile(str(lock_path));lock.setStaleLockTime(0)
    if not lock.tryLock(0):
        import ctypes
        hwnd=ctypes.windll.user32.FindWindowW(None,'Matte App')
        if hwnd:
            if ctypes.windll.user32.IsIconic(hwnd):ctypes.windll.user32.ShowWindow(hwnd,9)
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        return 0
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf');app.setFont(QFont('Segoe UI',9))
    window=StudioWindow();stream=ConsoleStream();stream.message.connect(window.append_console)
    previous=(sys.stdout,sys.stderr,sys.excepthook)
    sys.stdout=sys.stderr=stream
    import traceback
    def report(kind,error,tb):
        stream.write(''.join(traceback.format_exception(kind,error,tb)));window.file_tabs.setCurrentIndex(1)
    sys.excepthook=report
    window.show();print('Matte App ready. Results automatically load/save under the media project’s matted folder.')
    try:return app.exec()
    finally:sys.stdout,sys.stderr,sys.excepthook=previous;lock.unlock()

if __name__=='__main__':raise SystemExit(main())
