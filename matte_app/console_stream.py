from PySide6.QtCore import QObject,Signal

class ConsoleStream(QObject):
    message=Signal(str)
    encoding='utf-8'
    def write(self,text):
        if text:self.message.emit(str(text))
        return len(text)
    def flush(self):pass
    def isatty(self):return False
