import os,unittest
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from types import SimpleNamespace
from unittest.mock import Mock,patch
from PySide6.QtCore import Qt,QModelIndex,QPersistentModelIndex,QPoint,QRect,QEvent,QPointF
from PySide6.QtGui import QStandardItemModel,QStandardItem,QHelpEvent,QImage,QPainter,QMouseEvent
from PySide6.QtWidgets import QApplication,QListWidget,QStyleOptionViewItem
from ui_components import ThumbnailFiles
APP=QApplication.instance() or QApplication([])

class ThumbnailDelegateTests(unittest.TestCase):
    def setUp(self):
        self.queue=QListWidget();self.queue.addItems(['first.png','second.png']);self.queue.setCurrentRow(0)
        self.owner=SimpleNamespace(queue_list=self.queue,_queue_states={},_shell_stage_states={},remove_queue_item=Mock())
        self.delegate=ThumbnailFiles(self.owner);self.option=QStyleOptionViewItem();self.option.rect=QRect(0,0,180,166)
        self.help=QHelpEvent(QEvent.ToolTip,QPoint(10,110),QPoint(10,110))
    def tearDown(self):self.queue.close();self.queue.deleteLater();APP.processEvents()
    def test_empty_or_missing_path_hides_tooltip(self):
        model=QStandardItemModel();model.appendRow(QStandardItem())
        with patch('ui_components.QToolTip.showText') as show,patch('ui_components.QToolTip.hideText') as hide:
            for value in (None,'',42):
                model.setData(model.index(0,0),value,Qt.DisplayRole)
                self.assertFalse(self.delegate.helpEvent(self.help,self.queue.viewport(),self.option,model.index(0,0)))
            self.assertFalse(self.delegate.helpEvent(self.help,self.queue.viewport(),self.option,QModelIndex()))
            show.assert_not_called();self.assertEqual(hide.call_count,4)
    def test_hover_after_queue_clear_is_safe(self):
        index=QPersistentModelIndex(self.queue.model().index(0,0));self.queue.clear()
        self.assertFalse(index.isValid())
        self.assertFalse(self.delegate.helpEvent(self.help,self.queue.viewport(),self.option,index))
    def test_valid_path_tooltip_is_preserved(self):
        with patch('ui_components.QToolTip.showText') as show:
            self.assertTrue(self.delegate.helpEvent(self.help,self.queue.viewport(),self.option,self.queue.model().index(0,0)))
            self.assertIn('first.png',show.call_args.args[1]);self.assertIn('Steps:',show.call_args.args[1])
    def test_invalid_paint_and_click_do_not_read_path_or_remove(self):
        image=QImage(180,166,QImage.Format_RGBA8888);image.fill(Qt.transparent);painter=QPainter(image)
        try:self.delegate.paint(painter,self.option,QModelIndex())
        finally:painter.end()
        self.assertFalse(self.delegate.editorEvent(QEvent(QEvent.MouseButtonRelease),self.queue.model(),self.option,QModelIndex()))
        self.owner.remove_queue_item.assert_not_called()
    def test_trash_press_does_not_transfer_to_replacement_row(self):
        position=QPointF(self.delegate.trash_rect(self.option.rect).center())
        press=QMouseEvent(QEvent.MouseButtonPress,position,position,Qt.LeftButton,Qt.LeftButton,Qt.NoModifier)
        release=QMouseEvent(QEvent.MouseButtonRelease,position,position,Qt.LeftButton,Qt.NoButton,Qt.NoModifier)
        self.delegate.editorEvent(press,self.queue.model(),self.option,self.queue.model().index(0,0))
        self.queue.takeItem(0)
        self.assertFalse(self.delegate.editorEvent(release,self.queue.model(),self.option,self.queue.model().index(0,0)))
        self.owner.remove_queue_item.assert_not_called()

    def test_trash_click_still_removes_selected_valid_item(self):
        position=QPointF(self.delegate.trash_rect(self.option.rect).center());index=self.queue.model().index(0,0)
        press=QMouseEvent(QEvent.MouseButtonPress,position,position,Qt.LeftButton,Qt.LeftButton,Qt.NoModifier)
        release=QMouseEvent(QEvent.MouseButtonRelease,position,position,Qt.LeftButton,Qt.NoButton,Qt.NoModifier)
        self.delegate.editorEvent(press,self.queue.model(),self.option,index)
        self.assertTrue(self.delegate.editorEvent(release,self.queue.model(),self.option,index))
        self.owner.remove_queue_item.assert_called_once()

if __name__=='__main__':unittest.main()

