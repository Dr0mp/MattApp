"""Shared styling, icons, panels and file thumbnails for the production UI."""
from pathlib import Path
from PySide6.QtCore import Qt,QPointF,QRectF,QSize,QEvent,QRect,QPersistentModelIndex
from PySide6.QtGui import QColor,QPainter,QPen,QPixmap,QIcon,QPainterPath,QImageReader
from PySide6.QtWidgets import QWidget,QStyledItemDelegate,QToolTip

def glyph(name):
    pix=QPixmap(24,24);pix.fill(Qt.transparent)
    p=QPainter(pix);p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor('#242627'),1.8,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin))
    if name in ('plus','minus'):
        p.drawEllipse(QRectF(3,3,18,18));p.drawLine(7,12,17,12)
        if name=='plus':p.drawLine(12,7,12,17)
    elif name=='box':p.drawRoundedRect(QRectF(3,4,18,16),2,2)
    elif name=='trash':
        p.drawLine(5,6,19,6);p.drawLine(9,3,15,3);p.drawRoundedRect(QRectF(7,6,10,15),1,1)
        p.drawLine(10,10,10,17);p.drawLine(14,10,14,17)
    elif name=='brush':
        p.drawLine(11,14,19,4);p.drawLine(14,16,21,6)
        path=QPainterPath(QPointF(11,14));path.cubicTo(5,12,9,20,3,21);path.cubicTo(11,22,15,21,14,16);p.drawPath(path)
    elif name=='erase':
        path=QPainterPath(QPointF(3,15));path.lineTo(13,4);path.lineTo(21,11);path.lineTo(13,20);path.lineTo(8,20);path.closeSubpath();p.drawPath(path);p.drawLine(9,9,17,16)
    elif name in ('undo','redo'):
        direction=1 if name=='undo' else -1
        p.drawArc(QRectF(5,8,15,12),0,180*16)
        x=5 if direction==1 else 20;p.drawLine(x,8,x,14);p.drawLine(x,8,x+5*direction,8)
    elif name=='grid':
        for x in (4,14):
            for y in (4,14):p.drawRoundedRect(QRectF(x,y,6,6),1,1)
    elif name=='list':
        for y in (5,12,19):p.drawLine(4,y,20,y)
    p.end();return QIcon(pix)


class Panel(QWidget):
    def paintEvent(self,event):
        p=QPainter(self);p.setRenderHint(QPainter.Antialiasing)
        r=QRectF(1,1,self.width()-6,self.height()-6)
        p.setPen(Qt.NoPen);p.setBrush(QColor('#262828'));p.drawRoundedRect(r.translated(4,4),19,19)
        p.setBrush(QColor('#fffdf7'));p.setPen(QPen(QColor('#262828'),2));p.drawRoundedRect(r,19,19)


class ThumbnailFiles(QStyledItemDelegate):
    def __init__(self,owner):
        super().__init__(owner.queue_list);self.owner=owner;self.cache={};self._scaled_cache={};self.trash=glyph('trash')
        self._trash_pressed=None

    def sizeHint(self,option,index):return QSize(180,166)

    def trash_rect(self,rect):return QRect(rect.right()-31,rect.top()+7,24,24)

    def thumbnail(self,path):
        if path not in self.cache:
            reader=QImageReader(path)
            size=reader.size()
            if size.isValid():reader.setScaledSize(size.scaled(144,87,Qt.KeepAspectRatio))
            pix=QPixmap.fromImage(reader.read())
            if len(self.cache)>=128:self.cache.pop(next(iter(self.cache)))
            self.cache[path]=pix
        return self.cache[path]

    @staticmethod
    def item_path(index):
        if not index.isValid():return None
        path=index.data(Qt.DisplayRole)
        return path if isinstance(path,str) and path.strip() else None

    def paint(self,painter,option,index):
        path=self.item_path(index)
        if path is None:return
        selected=index==self.owner.queue_list.currentIndex()
        state=getattr(self.owner,'_queue_states',{}).get(str(Path(path).resolve()).casefold(),{})
        missing=bool(state.get('missing'))
        r=option.rect.adjusted(3,3,-3,-3)
        painter.save();painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor('#262828' if selected else '#c1c1b6'),2 if selected else 1))
        painter.setBrush(QColor('#f5c6c5' if missing else '#e4ddfc' if selected else '#fffdf7'));painter.drawRoundedRect(r,9,9)
        image_rect=QRect(r.left()+7,r.top()+7,r.width()-14,87)
        painter.setPen(Qt.NoPen);painter.setBrush(QColor('#282b2e'));painter.drawRoundedRect(image_rect,5,5)
        pix=self.thumbnail(path)
        if not pix.isNull():
            target_size=(image_rect.width(),image_rect.height())
            cache_key=(path,target_size,pix.cacheKey())
            scaled_pix=self._scaled_cache.get(cache_key)
            if scaled_pix is None:
                scaled_pix=pix.scaled(image_rect.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
                if len(self._scaled_cache)>=128:self._scaled_cache.pop(next(iter(self._scaled_cache)))
                self._scaled_cache[cache_key]=scaled_pix
            painter.drawPixmap(image_rect.center().x()-scaled_pix.width()//2,image_rect.center().y()-scaled_pix.height()//2,scaled_pix)
        else:
            painter.setPen(QColor('#e0e4df'));painter.drawText(image_rect,Qt.AlignCenter,'VIDEO' if Path(path).suffix.lower() in ('.mp4','.mov','.avi','.mkv','.webm') else 'FILE')
        painter.setPen(QColor('#262828'))
        name=painter.fontMetrics().elidedText(Path(path).name,Qt.ElideRight,r.width()-14)
        painter.drawText(QRect(r.left()+7,image_rect.bottom()+10,r.width()-14,painter.fontMetrics().height()+4),Qt.AlignLeft|Qt.AlignVCenter,name)
        states=self.owner._shell_stage_states.get(path,(False,False,False))
        for n,done in enumerate(states):
            painter.setPen(QPen(QColor('#262828'),1));painter.setBrush(QColor('#f6d85c' if done in ('partial','stale') else '#46bf72' if done else '#262828'))
            painter.drawEllipse(QPointF(r.left()+14+n*17,r.bottom()-13),4,4)
        if state.get('processing') or missing or state.get('draft'):
            label='Processing' if state.get('processing') else 'Needs guidance / mask' if missing else 'Draft'
            painter.setPen(Qt.NoPen);painter.setBrush(QColor('#f6d85c' if not missing else '#ef9896'));badge=QRect(r.left()+7,r.top()+7,min(r.width()-45,126),19);painter.drawRoundedRect(badge,4,4);painter.setPen(QColor('#262828'));painter.drawText(badge,Qt.AlignCenter,label)
        if selected:
            trash=self.trash_rect(option.rect)
            painter.setPen(QPen(QColor('#262828'),1));painter.setBrush(QColor('#fffdf7'));painter.drawRoundedRect(trash,5,5)
            self.trash.paint(painter,trash.adjusted(3,3,-3,-3))
        painter.restore()

    def editorEvent(self,event,model,option,index):
        if self.item_path(index) is None:
            self._trash_pressed=None;return False
        if event.type()==QEvent.MouseButtonPress:
            self._trash_pressed=QPersistentModelIndex(index) if event.button()==Qt.LeftButton and index==self.owner.queue_list.currentIndex() and self.trash_rect(option.rect).contains(event.position().toPoint()) else None
        elif event.type()==QEvent.MouseButtonRelease:
            remove=self._trash_pressed is not None and self._trash_pressed.isValid() and self._trash_pressed==index and event.button()==Qt.LeftButton and self.trash_rect(option.rect).contains(event.position().toPoint())
            self._trash_pressed=None
            if remove:
                self.owner.remove_queue_item();return True
        return False

    def helpEvent(self,event,view,option,index):
        path=self.item_path(index)
        if path is None:
            QToolTip.hideText();return False
        text=path+'\nSteps: Matte · Refined · Cutout\nBlack: no saved result linked · Green: result available'
        state=getattr(self.owner,'_queue_states',{}).get(str(Path(path).resolve()).casefold(),{})
        if state.get('missing'):text+='\n'+state['missing']
        if state.get('processing'):text+='\nProcessing '+state.get('stage','')
        if index==self.owner.queue_list.currentIndex() and self.trash_rect(option.rect).contains(event.pos()):text='Remove this file from the queue; keep the original file.'
        QToolTip.showText(event.globalPos(),text,view);return True


STYLE='''
QWidget { color:#262828; font-family:"Segoe UI"; font-size:12px; background:transparent; }
QWidget#shellRoot,QMenuBar { background:#efecdf; }
QLabel { border:none; background:transparent; }
QLabel#wordmark { font-size:25px; font-weight:800; }
QLabel#previewBadge { background:#f6d85c; border:1px solid #262828; border-radius:6px; padding:5px 9px; font-size:10px; font-weight:700; }
QLabel#sectionLabel { font-size:11px; font-weight:800; letter-spacing:2px; }
QLabel#hint { color:#77786e; font-size:11px; }
QLabel#status { color:#676b61; font-size:11px; }
QPushButton,QToolButton { background:#fffdf7; border:1px solid #262828; border-bottom:3px solid #262828; border-radius:8px; padding:6px 9px; }
QPushButton:hover,QToolButton:hover { background:#f6f0d9; }
QPushButton:pressed,QToolButton:pressed { border-bottom:1px solid #262828; background:#e7dfbf; }
QPushButton:disabled,QToolButton:disabled { color:#989b92; border-color:#c4c5bb; background:#f2f1e9; }
QToolButton:checked { background:#f6d85c; border-color:#262828; }
QPushButton#primary { background:#f6d85c; font-weight:700; }
QPushButton#primary:disabled { background:#f2e9ba; color:#96917c; border-color:#c4c5bb; }
QPushButton#queueTile { background:#f6d85c; font-size:14px; font-weight:700; }
QPushButton#queueTile:disabled { background:#f2e9ba; color:#96917c; border-color:#c4c5bb; }
QPushButton#applyTile { background:#e4ddfc; font-size:14px; font-weight:700; }
QPushButton#applyTile:disabled { background:#eeebf5; color:#94909e; border-color:#c4c5bb; }
QPushButton#stop { background:#f5c6c5; }
QPushButton#stop:disabled { background:#f2eeea; color:#a69c94; border-color:#c4c5bb; }
QComboBox,QSpinBox,QLineEdit { background:#fffdf7; border:1px solid #b7b9ae; border-radius:7px; padding:7px 30px 7px 9px; min-height:18px; }
QComboBox::drop-down { width:25px; border:none; }
QComboBox::down-arrow { image:url(__ASSETS__/down.svg); width:10px; height:7px; }
QComboBox QAbstractItemView { background:#fffdf7; selection-background-color:#f6d85c; selection-color:#262828; padding:6px; }
QSpinBox::up-button,QSpinBox::down-button { width:24px; border:none; }
QSpinBox::up-arrow { image:url(__ASSETS__/up.svg); width:10px; height:7px; }
QSpinBox::down-arrow { image:url(__ASSETS__/down.svg); width:10px; height:7px; }
QTabBar::tab { background:#e8e4d8; border:2px solid #262828; border-bottom:0; border-top-left-radius:10px; border-top-right-radius:10px; padding:10px 18px; margin-right:5px; font-weight:600; }
QTabBar::tab:selected { background:#fffdf7; }
QFrame#optionsBar { background:#fffdf7; border:1px solid #b7b9ae; border-radius:10px; }
QFrame#canvasFrame { background:#17191d; border:2px solid #262828; border-top-left-radius:0; border-top-right-radius:12px; border-bottom-left-radius:12px; border-bottom-right-radius:12px; }
QTabBar::tab:selected { background:#17191d; color:#fffdf7; }
QFrame#consolePanel { background:#fffdf7; border:1px solid #262828; border-radius:10px; }
QFrame#workflowSection { background:#f7f5ed; border:1px solid #c1c1b6; border-radius:10px; }
QPlainTextEdit { background:#17191d; color:#e0e4df; border:none; border-radius:6px; padding:6px; font-family:Consolas; font-size:11px; }
QLabel#placeholderText { color:#a4aaa5; font-size:14px; }
QFrame#toolRail { background:#f2efe4; border:1px solid #c1c1b6; border-radius:12px; }
QFrame#zoomOverlay { background:#fffdf7; border:1px solid #262828; border-radius:9px; }
QFrame#videoPanel { border:none; background:#f2efe4; border-radius:10px; }
QListWidget { background:#f7f5ed; border:1px solid #c1c1b6; border-radius:10px; padding:6px; }
QListWidget::item { padding:13px 8px; border-radius:7px; }
QListWidget::item:selected { background:#e4ddfc; color:#262828; }
QListWidget::item:hover { background:#eee8d8; }
QSplitter::handle { background:transparent; }
QScrollArea { border:none; }
QScrollBar:vertical { width:8px; background:transparent; }
QScrollBar:horizontal { height:8px; background:transparent; }
QScrollBar::handle { background:#b9baa9; border-radius:4px; min-width:24px; min-height:24px; }
QScrollBar::add-line,QScrollBar::sub-line { width:0; height:0; }
QProgressBar { border:1px solid #a5a89b; border-radius:4px; background:#efede3; }
QProgressBar::chunk { background:#bda9ef; border-radius:3px; }
QSlider::groove:horizontal { height:5px; background:#cccdbf; border-radius:2px; }
QSlider::handle:horizontal { width:13px; margin:-4px 0; border-radius:6px; background:#262828; }
QMenu { background:#fffdf7; border:1px solid #262828; padding:6px; }
QMenu::item:selected { background:#f6d85c; }
QToolTip { background:#fffdf7; color:#262828; border:1px solid #262828; padding:6px; }
'''
