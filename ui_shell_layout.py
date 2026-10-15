"""Revised review shell: attached workflow, viewport tools, files/console dock."""
from PySide6.QtCore import Qt,QPointF,QRectF,QSize,Signal,QEvent
from PySide6.QtGui import QPainter,QPen,QColor,QIcon,QPixmap,QPainterPath,QAction,QShortcut,QKeySequence
from PySide6.QtWidgets import (QWidget,QFrame,QVBoxLayout,QHBoxLayout,QLabel,QToolButton,
    QPushButton,QComboBox,QTabBar,QSplitter,QStackedWidget,QPlainTextEdit,QProgressBar,
    QListView,QScrollArea,QSizePolicy,QListWidget,QButtonGroup,QApplication,QMessageBox)
from matte_app.canvas import ImageCanvas
from ui_components import Panel,ThumbnailFiles,STYLE,glyph
from matte_app.window import MatteWindow,PreviewWorker,PlaybackWorker
import tempfile
from pathlib import Path


class ReviewCanvas(ImageCanvas):
    guidance_selected=Signal(str,int)
    def __init__(self):
        super().__init__();self.shell_tool='box';self.padding=200;self.guidance_select_mode=False;self.selected_guidance=None

    def _target_rect(self):
        if self._image is None:return QRectF()
        width,height=self._source_size or self._image.size
        scale=min(self.width()/(width+2*self.padding),self.height()/(height+2*self.padding))*self._zoom
        return QRectF((self.width()-width*scale)/2+self._pan.x(),(self.height()-height*scale)/2+self._pan.y(),width*scale,height*scale)

    def _clamp_pan(self):
        if self._image is None:return
        rect=self._target_rect();scale=rect.width()/(self._source_size or self._image.size)[0]
        x=max(0,(rect.width()-self.width())/2)+self.padding*scale
        y=max(0,(rect.height()-self.height())/2)+self.padding*scale
        self._pan=QPointF(max(-x,min(x,self._pan.x())),max(-y,min(y,self._pan.y())))

    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton and self.guidance_select_mode and self.shell_tool in ('box','positive','negative'):
            rect=self._target_rect();pos=event.position();hits=[]
            if self.shell_tool=='box':
                for n,box in enumerate(self._boxes):
                    bounds=QRectF(QPointF(rect.left()+box.left*rect.width(),rect.top()+box.top*rect.height()),QPointF(rect.left()+box.right*rect.width(),rect.top()+box.bottom*rect.height())).normalized()
                    if bounds.adjusted(-8,-8,8,8).contains(pos):hits.append((bounds.width()*bounds.height(),'box',n))
            else:
                for n,point in enumerate(self._points):
                    distance=((pos.x()-rect.left()-point.x*rect.width())**2+(pos.y()-rect.top()-point.y*rect.height())**2)**.5
                    if distance<=16:hits.append((distance,'point',n))
            if hits:
                _,kind,n=min(hits);self.guidance_selected.emit(kind,n)
            return
        if event.button()==Qt.LeftButton and self.shell_tool=='hand' and self._image is not None:
            self.finish_stroke();self._pan_last=event.position();self.setCursor(Qt.ClosedHandCursor);return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.LeftButton and self.shell_tool=='hand':
            self._pan_last=None;self.setCursor(Qt.OpenHandCursor);return
        super().mouseReleaseEvent(event)

    def paintEvent(self,event):
        super().paintEvent(event)
        if not self.selected_guidance or self.shell_tool not in ('box','positive','negative') or self._image is None:return
        kind,n=self.selected_guidance;rect=self._target_rect();p=QPainter(self);p.setRenderHint(QPainter.Antialiasing);p.setPen(QPen(QColor('#f6d85c'),3));p.setBrush(Qt.NoBrush)
        if kind=='point' and 0<=n<len(self._points):
            point=self._points[n];p.drawEllipse(QPointF(rect.left()+point.x*rect.width(),rect.top()+point.y*rect.height()),13,13)
        elif kind=='box' and 0<=n<len(self._boxes):
            box=self._boxes[n];bounds=QRectF(QPointF(rect.left()+box.left*rect.width(),rect.top()+box.top*rect.height()),QPointF(rect.left()+box.right*rect.width(),rect.top()+box.bottom*rect.height())).normalized();p.drawRect(bounds)
            p.setBrush(QColor('#f6d85c'))
            for corner in (bounds.topLeft(),bounds.topRight(),bounds.bottomLeft(),bounds.bottomRight()):p.drawRect(QRectF(corner.x()-3,corner.y()-3,6,6))
        p.end()


def navigation_icon(name,selected=False):
    pix=QPixmap(24,24);pix.fill(Qt.transparent);p=QPainter(pix);p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor('#262828'),1.7,Qt.SolidLine,Qt.RoundCap,Qt.RoundJoin))
    if name=='zoom':p.drawEllipse(QRectF(3,3,12,12));p.drawLine(14,14,21,21);p.drawLine(6,9,12,9);p.drawLine(9,6,9,12)
    elif name=='minimize':p.drawLine(5,17,19,17)
    elif name=='maximize':p.drawRect(QRectF(5,5,14,14));p.drawLine(5,8,19,8)
    elif name=='restore':p.drawRect(QRectF(4,8,12,12));p.drawLine(8,8,8,4);p.drawLine(8,4,20,4);p.drawLine(20,4,20,16);p.drawLine(20,16,16,16)
    elif name=='close':p.drawLine(6,6,18,18);p.drawLine(18,6,6,18)
    elif name=='save':
        path=QPainterPath(QPointF(3,3));path.lineTo(18,3);path.lineTo(21,6);path.lineTo(21,21);path.lineTo(3,21);path.closeSubpath();p.drawPath(path);p.drawRect(QRectF(6,3,9,6));p.drawRect(QRectF(6,13,12,8));p.drawLine(12,4,12,7)
    elif name in ('box','positive','negative'):
        if name=='box':
            for x,y,dx,dy in ((3,3,1,1),(20,3,-1,1),(3,20,1,-1),(20,20,-1,-1)):p.drawLine(x,y,x+5*dx,y);p.drawLine(x,y,x,y+5*dy)
        else:
            p.drawEllipse(QRectF(3,3,12,12));p.setBrush(QColor('#46bf72' if name=='positive' else '#ef6d6d'));p.drawEllipse(QRectF(7,7,4,4));p.setBrush(Qt.NoBrush)
            if not selected:
                p.drawLine(14,19,22,19)
                if name=='positive':p.drawLine(18,15,18,23)
        if selected:
            path=QPainterPath(QPointF(12,11));path.lineTo(13,22);path.lineTo(16,18);path.lineTo(21,17);path.closeSubpath();p.setBrush(QColor('#fffdf7'));p.drawPath(path)
    else:
        path=QPainterPath(QPointF(7,20));path.lineTo(3,12);path.cubicTo(2,9,5,9,7,13);path.lineTo(7,5);path.cubicTo(7,2,10,2,10,5);path.lineTo(10,11);path.lineTo(10,3);path.cubicTo(10,1,13,1,13,3);path.lineTo(13,11);path.lineTo(13,5);path.cubicTo(13,3,16,3,16,5);path.lineTo(16,12);path.lineTo(16,8);path.cubicTo(16,6,19,6,19,8);path.lineTo(19,16);path.lineTo(16,21);path.closeSubpath();p.drawPath(path)
    p.end();return QIcon(pix)


class ToolViewport(QFrame):
    def __init__(self):
        super().__init__();self.setObjectName('canvasFrame');self.rail=None;self.popup=None

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.rail:self.rail.move(12,12);self.rail.raise_()
        if self.popup:self.popup.move(68,12);self.popup.raise_()


class WindowHeader(QFrame):
    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton and self.window().windowHandle():
            self.window().windowHandle().startSystemMove();event.accept();return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self,event):
        if event.button()==Qt.LeftButton:self.window().toggle_maximize();event.accept();return
        super().mouseDoubleClickEvent(event)


class ShellPreviewWindowV3(MatteWindow):
    def __init__(self):
        self._shell_ready=False
        super().__init__()
        self._preview_profile=tempfile.TemporaryDirectory(prefix='matte-shell-review-')
        self.results_index_path=Path(self._preview_profile.name)/'results_index.json'
        self.queue_outputs={}
        self._shell_stage_states={}
        self._refined_preview=False
        self._workflow_review_stage=0
        self.setWindowTitle('Matte App — UI shell review')
        self.resize(1440,980)
        self.build_shell()
        self._shell_ready=True
        self.sync_shell()
    def select_shell_tab(self,index):
        self._refined_preview=index==2
        if index!=2:self.view_tabs.setCurrentIndex({0:0,1:1,3:2}[index])
        self.sync_shell()
    def append_console(self,text):
        from PySide6.QtGui import QTextCursor
        cursor=self.console_output.textCursor();cursor.movePosition(QTextCursor.End);cursor.insertText(text)
        self.console_output.setTextCursor(cursor);self.console_output.ensureCursorVisible()
    def _refresh_canvas(self):
        super()._refresh_canvas();self.sync_shell()
    def _start_worker(self,worker):
        if not isinstance(worker,(PreviewWorker,PlaybackWorker)):
            self.preview_notice()
            return
        super()._start_worker(worker)
    def _update_navigation(self):
        super()._update_navigation()
        self.sync_shell()
    def on_preview_ready(self,preview):
        self._refined_preview=False
        super().on_preview_ready(preview)
        self._shell_stage_states[str(self.current_media_path)]=(self.alpha is not None,False,False)
        self.queue_list.viewport().update()
        self.sync_shell()
    def save_file_guidance(self,*args,**kwargs):return True
    def guard_frame_edits(self):return True
    def preview_notice(self,*args):QMessageBox.information(self,'Interface preview','This stage is for reviewing the shell. Saving, processing and new functions connect after approval; source files and saved results are read-only here.')
    save_frame_edits=preview_notice
    export_matte=preview_notice
    export_cutout=preview_notice
    save_project=preview_notice
    generate=preview_notice
    run_queue=preview_notice
    resume_video=preview_notice
    test_current_frame=preview_notice
    get_sam3_weights=preview_notice

    def open_installer(self, focus_model=None):
        from matte_app.installer_dialog import ModelInstallerDialog
        if not hasattr(self, '_installer_dialog') or self._installer_dialog is None:
            self._installer_dialog = ModelInstallerDialog(self, focus_model=focus_model)
            if hasattr(self._installer_dialog, 'models_updated'):
                self._installer_dialog.models_updated.connect(self.refresh_installed_pipelines)
            self._installer_dialog.finished.connect(lambda *_: self.refresh_installed_pipelines())
        else:
            if focus_model:
                self._installer_dialog.select_focus_model(focus_model)
        self._installer_dialog.show()
        self._installer_dialog.raise_()
        self._installer_dialog.activateWindow()
        return self._installer_dialog

    def refresh_installed_pipelines(self):
        root = getattr(self, 'root', None)
        if root is None:
            root = Path(__file__).resolve().parent
        from matte_app.installed_models import get_available_pipelines, get_available_selectors, get_available_refiners

        valid_pipes = get_available_pipelines(root)
        current_pipe = self.pipeline_picker.currentText()
        self.pipeline_picker.blockSignals(True)
        self.pipeline_picker.clear()
        self.pipeline_picker.addItems(valid_pipes)
        if current_pipe in valid_pipes:
            self.pipeline_picker.setCurrentText(current_pipe)
        elif valid_pipes:
            self.pipeline_picker.setCurrentIndex(0)
        self.pipeline_picker.blockSignals(False)

        valid_selectors = get_available_selectors(root)
        current_sel = self.manual_selector.currentText()
        self.manual_selector.blockSignals(True)
        self.manual_selector.clear()
        self.manual_selector.addItems(valid_selectors)
        if current_sel in valid_selectors:
            self.manual_selector.setCurrentText(current_sel)
        elif valid_selectors:
            self.manual_selector.setCurrentIndex(0)
        self.manual_selector.blockSignals(False)

        valid_refiners = get_available_refiners(root)
        current_ref = self.manual_refiner.currentText()
        self.manual_refiner.blockSignals(True)
        self.manual_refiner.clear()
        self.manual_refiner.addItems(valid_refiners)
        if current_ref in valid_refiners:
            self.manual_refiner.setCurrentText(current_ref)
        elif valid_refiners:
            self.manual_refiner.setCurrentIndex(0)
        self.manual_refiner.blockSignals(False)

        if getattr(self, '_shell_ready', False):
            self.sync_shell()
            self.update_pipeline()

    def build_shell(self):
        old=self.takeCentralWidget();root=QWidget();root.setObjectName('shellRoot');self.setCentralWidget(root);old.setParent(root);old.hide();self._legacy_ui=old
        self._active_tool='box';self._last_channel=0;self._guidance_states={'box':False,'point':False}
        old_canvas=None
        if not getattr(self,'_studio_controls_ready',False):
            old_canvas=self.canvas;old_canvas.hide();self.brush_size.valueChanged.disconnect(old_canvas.set_brush_diameter);self.canvas=getattr(self,'canvas_class',ReviewCanvas)()
        for signal,slot in ((self.canvas.zoom_changed,self._sync_zoom),(self.canvas.point_added,self.add_point),(self.canvas.box_added,self.add_box),(self.canvas.guidance_removed,self.remove_guidance),(self.canvas.stroke_started,self.matte_editor.begin_stroke),(self.canvas.brush_segment,self.paint_matte),(self.canvas.stroke_finished,self.finish_matte_stroke)):signal.connect(slot)
        self.brush_size.valueChanged.connect(self.canvas.set_brush_diameter);self.fit_button.clicked.disconnect();self.fit_button.clicked.connect(self.canvas.reset_zoom)
        self.canvas.guidance_selected.connect(self.select_guidance_object);self.guidance_picker.currentIndexChanged.connect(self.sync_guidance_selection)
        for action in self.menuBar().findChildren(QAction):
            if action.text()=='Fit preview':action.triggered.disconnect();action.triggered.connect(self.canvas.reset_zoom)
        if old_canvas:old_canvas.setParent(None);old_canvas.deleteLater()
        outer=QVBoxLayout(root);outer.setContentsMargins(12,8,12,12);outer.setSpacing(9)
        self.setWindowFlag(Qt.FramelessWindowHint,True);self.setAttribute(Qt.WA_TranslucentBackground,True);self.setMouseTracking(True);root.setMouseTracking(True)
        self.menuBar().hide();self.window_header=WindowHeader();self.window_header.setObjectName('windowHeader');self.window_header.setFixedHeight(34)
        self.window_header.setMouseTracking(True)
        header=QHBoxLayout(self.window_header);header.setContentsMargins(0,0,0,0);header.setSpacing(5);self.app_menu_buttons=[]
        for action in self.menuBar().actions():
            if action.menu():
                button=QToolButton();button.setObjectName('appMenu');button.setText(action.text().replace('&',''));button.setMenu(action.menu());button.setPopupMode(QToolButton.InstantPopup);button.setAccessibleName(button.text()+' menu');header.addWidget(button);self.app_menu_buttons.append(button)
        self.installer_header_button=QToolButton();self.installer_header_button.setObjectName('appMenu');self.installer_header_button.setText('Install Models');self.installer_header_button.setToolTip('Download AI models, verify Hugging Face login, and set up runtimes');self.installer_header_button.clicked.connect(self.open_installer);header.addWidget(self.installer_header_button)
        header.addStretch()
        self.minimize_button=QToolButton();self.maximize_button=QToolButton();self.close_window_button=QToolButton()
        for button,icon,tip,callback in ((self.minimize_button,'minimize','Minimize',self.showMinimized),(self.maximize_button,'maximize','Maximize',self.toggle_maximize),(self.close_window_button,'close','Close',self.close)):
            button.setObjectName('windowClose' if icon=='close' else 'windowControl');button.setIcon(navigation_icon(icon));button.setToolTip(tip);button.setAccessibleName(tip);button.clicked.connect(callback);header.addWidget(button)
        outer.addWidget(self.window_header)
        QApplication.instance().installEventFilter(self)
        connected=Panel();connected_layout=QVBoxLayout(connected);connected_layout.setContentsMargins(8,8,12,12)
        self.splitter=QSplitter(Qt.Horizontal);self.splitter.setChildrenCollapsible(False);self.splitter.setHandleWidth(6);connected_layout.addWidget(self.splitter);outer.addWidget(connected,1)
        editor=QWidget();editor.setMinimumWidth(590);body=QVBoxLayout(editor);body.setContentsMargins(8,4,8,0);body.setSpacing(0)
        self.toolbar=QFrame();self.toolbar.setObjectName('optionsBar');self.toolbar.setFixedHeight(42)
        toolbar=QHBoxLayout(self.toolbar);toolbar.setContentsMargins(5,4,5,4);toolbar.setSpacing(4)
        self.viewport_model=QComboBox();self.viewport_model.setFixedWidth(130);toolbar.addWidget(self.viewport_model);toolbar.addStretch()
        top=toolbar
        self.redo_matte=self.generate_button;self.generate_button.setObjectName('primary')
        self.result_action=QPushButton('Save matte');self.result_action.clicked.connect(self.preview_notice)
        self.result_action.setIcon(navigation_icon('save'));self.result_action.setIconSize(QSize(16,16))
        for name,action in self.file_actions.items():
            if name.startswith(('Save','Export')):action.setIcon(navigation_icon('save'))
        self.export_matte_button.hide();self.export_button.hide();self.save_edits_button.hide()
        self.toolbar_actions=(self.remove_guidance_button,self.clear_button,self.test_frame_button,self.generate_button,self.resume_button,self.result_action)
        for control in self.toolbar_actions:control.setMinimumWidth(0);toolbar.addWidget(control)
        self.edit_state.hide();self.edit_help=QLabel('?');self.edit_help.setObjectName('helpIcon');self.edit_help.setFixedSize(20,20);toolbar.addWidget(self.edit_help)
        self.shell_tabs=QTabBar();self.shell_tabs.setExpanding(False);self.shell_tabs.setDrawBase(False)
        for name in ('Original','Matte','Refined','Cutout'):self.shell_tabs.addTab(name)
        body.addWidget(self.shell_tabs)
        self.canvas_frame=ToolViewport();view=QVBoxLayout(self.canvas_frame);view.setContentsMargins(5,5,5,5)
        self.viewer_stack=QStackedWidget();self.viewer_stack.addWidget(self.canvas)
        placeholder=QWidget();ph=QVBoxLayout(placeholder);note=QLabel('REFINED\n\nSeparate editable results connect after shell approval.');note.setObjectName('placeholderText');note.setAlignment(Qt.AlignCenter);ph.addWidget(note);self.viewer_stack.addWidget(placeholder);view.addWidget(self.viewer_stack);body.addWidget(self.canvas_frame,1);body.addSpacing(6);body.addWidget(self.toolbar);body.addWidget(self.video_panel)
        rail=QFrame(self.canvas_frame);rail.setObjectName('toolRail');rail.setFixedWidth(48);tools=QVBoxLayout(rail);tools.setContentsMargins(4,6,4,6);tools.setSpacing(5)
        self.tool_buttons={};self.shell_guide_tools=(self.box_button,self.point_button,self.negative_button);self.shell_paint_tools=(self.white_brush_button,self.black_brush_button)
        self.hand_button=QToolButton();self.zoom_tool_button=QToolButton()
        for button,mode,icon in ((self.hand_button,'hand','hand'),(self.zoom_tool_button,'zoom','zoom'),(self.box_button,'box','box'),(self.point_button,'positive','plus'),(self.negative_button,'negative','minus'),(self.white_brush_button,'paint_white','brush'),(self.black_brush_button,'paint_black','erase')):
            button.setText('');button.setIcon(navigation_icon(mode) if mode in ('hand','zoom','box','positive','negative') else glyph(icon));button.setToolTip({'hand':'Hand — drag to pan','zoom':'Zoom controls','box':'Box — place or select','positive':'Foreground point — place or select','negative':'Background point — place or select','paint_white':'Paint white — keep','paint_black':'Paint black — remove'}[mode]);button.setCheckable(True);button.setFixedSize(40,36);button.clicked.connect(lambda checked=False,tool=mode:self.choose_tool(tool));tools.addWidget(button);self.tool_buttons[mode]=button
        for group in self.findChildren(QButtonGroup):
            for button in self.tool_buttons.values():group.removeButton(button)
        self.viewport_tool_group=QButtonGroup(self);self.viewport_tool_group.setExclusive(True)
        for button in self.tool_buttons.values():self.viewport_tool_group.addButton(button)
        rail.adjustSize();self.canvas_frame.rail=rail
        self.tool_popup=QFrame(self.canvas_frame);self.tool_popup.setObjectName('zoomOverlay');self.tool_popup.setFixedWidth(238);pop=QVBoxLayout(self.tool_popup);pop.setContentsMargins(10,9,10,9);pop.setSpacing(7)
        header=QHBoxLayout();self.popup_title=QLabel();header.addWidget(self.popup_title,1);hide=QToolButton(text='×');hide.setFixedSize(24,24);hide.clicked.connect(self.tool_popup.hide);header.addWidget(hide);pop.addLayout(header)
        self.popup_stack=QStackedWidget();pop.addWidget(self.popup_stack)
        guidance=QWidget();gl=QVBoxLayout(guidance);gl.setContentsMargins(0,0,0,0);gl.setSpacing(7)
        states=QHBoxLayout();states.setSpacing(6);self.place_guidance_button=QToolButton(text='Place');self.select_guidance_button=QToolButton(text='Select');self.guidance_mode_group=QButtonGroup(self);self.guidance_mode_group.setExclusive(True)
        for button,selected in ((self.place_guidance_button,False),(self.select_guidance_button,True)):
            button.setCheckable(True);button.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed);states.addWidget(button);self.guidance_mode_group.addButton(button);button.clicked.connect(lambda checked=False,state=selected:self.set_guidance_state(state))
        gl.addLayout(states);self.guidance_objects=QListWidget();self.guidance_objects.setFixedHeight(116);gl.addWidget(self.guidance_objects);self.guidance_objects.currentRowChanged.connect(lambda row:self.guidance_picker.setCurrentIndex(row) if row>=0 else None)
        guidance_actions=QHBoxLayout();guidance_actions.addWidget(self.remove_guidance_button);guidance_actions.addWidget(self.clear_button);gl.addLayout(guidance_actions);self.popup_stack.addWidget(guidance)
        zoom=QWidget();zl=QHBoxLayout(zoom);zl.setContentsMargins(0,2,0,2);zl.setSpacing(6)
        for button,width in ((self.fit_button,42),(self.zoom_out_button,30),(self.zoom_percent,88),(self.zoom_in_button,30)):
            button.setFixedSize(width,32);zl.addWidget(button)
        self.zoom_percent.setStyleSheet('QSpinBox { padding:2px 24px 2px 6px; min-height:0; }');self.popup_stack.addWidget(zoom)
        brush=QWidget();bl=QVBoxLayout(brush);bl.setContentsMargins(0,0,0,0);diameter=QHBoxLayout();diameter.addWidget(QLabel('Diameter'));size_help=QLabel('?');size_help.setObjectName('helpIcon');size_help.setToolTip('[ and ] adjust brush diameter when the canvas has focus.');diameter.addWidget(size_help);diameter.addStretch();bl.addLayout(diameter);bl.addWidget(self.brush_size);undo=QHBoxLayout();undo.addWidget(self.undo_brush_button);undo.addWidget(self.redo_brush_button);bl.addLayout(undo);self.popup_stack.addWidget(brush)
        self.tool_popup.hide();self.canvas_frame.popup=self.tool_popup
        self.canvas.setFocusPolicy(Qt.StrongFocus);self.canvas_shortcuts=[]
        for key,callback in (('H',lambda:self.choose_tool('hand')),('Z',lambda:self.choose_tool('zoom')),('[',lambda:self.adjust_brush(-1)),(']',lambda:self.adjust_brush(1))):
            shortcut=QShortcut(QKeySequence(key),self.canvas);shortcut.setContext(Qt.WidgetWithChildrenShortcut);shortcut.activated.connect(callback);self.canvas_shortcuts.append(shortcut)
        delete=QShortcut(QKeySequence(Qt.Key_Delete),self.canvas);delete.setContext(Qt.WidgetWithChildrenShortcut);delete.activated.connect(lambda:self.remove_selected_guidance() if self._active_tool in ('box','positive','negative') else None);self.canvas_shortcuts.append(delete)
        self.splitter.addWidget(editor)
        workflow=QWidget();workflow.setObjectName('attachedWorkflow');workflow.setMinimumWidth(310);work=QVBoxLayout(workflow);work.setContentsMargins(12,4,4,0);work.setSpacing(10)
        self.workflow_tabs=QTabBar();self.workflow_tabs.addTab('Auto');self.workflow_tabs.addTab('Manual');self.workflow_tabs.setExpanding(True);work.addWidget(self.workflow_tabs)
        self.workflow_pages=QStackedWidget();work.addWidget(self.workflow_pages,1);self.workflow_tabs.currentChanged.connect(self.workflow_pages.setCurrentIndex)
        auto=QWidget();al=QVBoxLayout(auto);al.setContentsMargins(0,0,0,0);al.setSpacing(10)
        pipeline_heading=QHBoxLayout();title=QLabel('PIPELINE');title.setObjectName('sectionLabel');pipeline_heading.addWidget(title);pipeline_heading.addStretch();self.pipeline_summary=QLabel();pipeline_heading.addWidget(self.pipeline_summary);al.addLayout(pipeline_heading)
        self.pipeline_picker=QComboBox();al.addWidget(self.pipeline_picker)
        defaults_heading=QHBoxLayout();defaults_heading.addWidget(QLabel('Default settings'));defaults_heading.addStretch();self.auto_defaults_help=QLabel('Choose a complete pipeline here. Test or generate from the viewport; Run queue applies it to the queue.');self.auto_defaults_help.setObjectName('hint');defaults_heading.addWidget(self.auto_defaults_help);al.addLayout(defaults_heading)
        self.defaults_summary=QLabel();self.defaults_summary.setObjectName('workflowDefaults');self.defaults_summary.setWordWrap(True);al.addWidget(self.defaults_summary)
        al.addStretch();self.workflow_pages.addWidget(auto)
        manual_scroll=QScrollArea();manual_scroll.setWidgetResizable(True);manual_scroll.setFrameShape(QFrame.NoFrame);manual_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        manual=QWidget();ml=QVBoxLayout(manual);ml.setContentsMargins(0,0,3,0);ml.setSpacing(8)
        def heading(layout,label,tip):
            row=QHBoxLayout();row.addWidget(QLabel(label));row.addStretch();help_icon=QLabel(tip);self.as_help(help_icon);row.addWidget(help_icon);layout.addLayout(row);return help_icon
        def field(label,items,tip):
            heading(ml,label,tip);combo=QComboBox();combo.addItems(items);ml.addWidget(combo);return combo
        self.manual_selector=field('Selection model',['BiRefNet HR'],'SAM models use your points or boxes; BiRefNet produces an automatic foreground matte.')
        self.manual_refiner=field('Refiner',['None — keep base matte'],'Choose a separate edge-refinement stage, or keep the base matte.')
        self.manual_masks=field('Input masks',['Current file’s saved matte','Choose saved masks…'],'MatAnyone starts from one mask. ViTMatte and VideoMaMa require masks for their input frames.')
        self.manual_extra=QFrame();self.manual_extra.setObjectName('workflowSection');extra=QVBoxLayout(self.manual_extra);extra.setContentsMargins(10,10,10,10);extra.setSpacing(8)
        heading(extra,'Processing settings','Only settings relevant to the chosen selection model and refiner appear here.')
        self.resolution_settings=QWidget();res=QVBoxLayout(self.resolution_settings);res.setContentsMargins(0,0,0,0);heading(res,'Working resolution','Refinement resolution. Source and exported image dimensions remain unchanged.');self.roto_size.setMinimumWidth(0);res.addWidget(self.roto_size);extra.addWidget(self.resolution_settings)
        self.video_settings=QWidget();video=QVBoxLayout(self.video_settings);video.setContentsMargins(0,0,0,0);heading(video,'VideoMaMa section','Number of frames processed together in overlapping sections.');self.roto_chunk.setMinimumWidth(0);video.addWidget(self.roto_chunk);extra.addWidget(self.video_settings)
        self.vit_settings=QWidget();vit=QVBoxLayout(self.vit_settings);vit.setContentsMargins(0,0,0,0);heading(vit,'ViTMatte edge band','Width of the uncertain region around the input mask edge, in source pixels.');vit.addWidget(self.trimap_width);extra.addWidget(self.vit_settings)
        self.sam3_settings=QWidget();sam=QVBoxLayout(self.sam3_settings);sam.setContentsMargins(0,0,0,0);heading(sam,'Object description','Optional SAM 3 text prompt. A starting mask takes priority.');sam.addWidget(self.roto_text);extra.addWidget(self.sam3_settings)
        self.native_settings=QLabel('Model-native resolution');extra.addWidget(self.native_settings)
        ml.addWidget(self.manual_extra)
        ml.addWidget(self.range_panel);ml.addStretch();manual_scroll.setWidget(manual);self.workflow_pages.addWidget(manual_scroll)
        self.history_toggle=QToolButton(self._legacy_ui);self.history_toggle.hide()
        self.history_area=QWidget(self._legacy_ui);self.history_area.hide()
        self.splitter.addWidget(workflow);self.splitter.setSizes([1030,340]);self.splitter.setStretchFactor(0,1)
        self.pipeline_picker.currentTextChanged.connect(self.update_pipeline);self.update_pipeline()
        self.viewport_model.currentTextChanged.connect(self.choose_viewport_model)
        self.manual_selector.currentTextChanged.connect(lambda _:self.sync_shell());self.manual_refiner.currentTextChanged.connect(lambda _:self.sync_shell());self.workflow_tabs.currentChanged.connect(lambda _:self.sync_shell())
        files=Panel();fl=QVBoxLayout(files);fl.setContentsMargins(14,10,18,14);fl.setSpacing(7);outer.addWidget(files)
        fh=QHBoxLayout();self.file_tabs=QTabBar();self.file_tabs.addTab('Files');self.file_tabs.addTab('Console');self.file_tabs.setExpanding(False);fh.addWidget(self.file_tabs);self.file_count=QLabel();fh.addWidget(self.file_count);fh.addStretch();legend=QLabel('● Matte   ● Refined   ● Cutout');legend.setObjectName('hint');fh.addWidget(legend)
        for button in (self.add_queue_button,self.clear_queue_button,self.load_results_button):button.setMinimumWidth(0);fh.addWidget(button)
        self.add_queue_button.setText('');self.add_queue_button.setIcon(glyph('plus'));self.add_queue_button.setAccessibleName('Add files');self.add_queue_button.setToolTip('Add files');self.add_queue_button.setFixedWidth(34)
        fl.addLayout(fh);strip=QHBoxLayout();strip.setSpacing(10);self.file_stack=QStackedWidget();self.file_stack.setFixedHeight(196)
        self.queue_list.setViewMode(QListView.IconMode);self.queue_list.setFlow(QListView.LeftToRight);self.queue_list.setWrapping(False);self.queue_list.setMovement(QListView.Static);self.queue_list.setResizeMode(QListView.Adjust);self.queue_list.setSpacing(3);self.queue_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff);self.queue_list.setMinimumWidth(250);self.thumbnail_delegate=ThumbnailFiles(self);self.queue_list.setItemDelegate(self.thumbnail_delegate);self.queue_list.currentRowChanged.connect(lambda _:self.queue_list.viewport().update());self.file_stack.addWidget(self.queue_list);strip.addWidget(self.file_stack,1)
        self.remove_shortcut=QShortcut(QKeySequence(Qt.Key_Delete),self.queue_list);self.remove_shortcut.setContext(Qt.WidgetWithChildrenShortcut);self.remove_shortcut.activated.connect(self.remove_queue_item)
        self.console_panel=QFrame();cl=QVBoxLayout(self.console_panel);cl.setContentsMargins(0,0,0,0);self.clear_console_button=QToolButton(text='Clear console');fh.addWidget(self.clear_console_button);self.console_output=QPlainTextEdit();self.console_output.setReadOnly(True);self.console_output.setMaximumBlockCount(1000);cl.addWidget(self.console_output);self.clear_console_button.clicked.connect(self.console_output.clear);self.file_stack.addWidget(self.console_panel);self.file_tabs.currentChanged.connect(self.file_stack.setCurrentIndex)
        self.files_help=legend;self.file_tabs.currentChanged.connect(self.sync_file_context)
        self.console_toggle=QToolButton();self.console_toggle.setCheckable(True);self.console_toggle.hide();self.console_toggle.toggled.connect(lambda checked:self.file_tabs.setCurrentIndex(1 if checked else 0));self.file_tabs.currentChanged.connect(lambda index:self.console_toggle.setChecked(index==1))
        self.remove_queue_button.hide();self.apply_masks_preview=QPushButton('Apply masks');self.apply_masks_preview.setObjectName('applyTile');self.apply_masks_preview.setEnabled(False);self.queue_run_button.setObjectName('queueTile')
        for button in (self.queue_run_button,self.apply_masks_preview):strip.addWidget(button)
        fl.addLayout(strip)
        self.queue_progress=QProgressBar();self.file_progress=QProgressBar()
        for name,bar in (('Full queue progress',self.queue_progress),('Current file progress',self.file_progress)):
            bar.setRange(0,100);bar.setValue(0);bar.setTextVisible(False);bar.setFixedHeight(5);bar.setAccessibleName(name);bar.setToolTip(name);fl.addWidget(bar)
        self.status.hide();self.stop_button.hide()
        for button in self.findChildren(QPushButton)+self.findChildren(QToolButton):
            if button not in self.tool_buttons.values():button.setMinimumHeight(30);button.setMaximumHeight(36)
        hide.setFixedSize(24,24)
        for button in (self.queue_run_button,self.apply_masks_preview):button.setFixedSize(120,120)
        for button in (self.minimize_button,self.maximize_button,self.close_window_button):button.setFixedSize(36,32)
        for button,width in ((self.fit_button,42),(self.zoom_out_button,30),(self.zoom_percent,88),(self.zoom_in_button,30)):button.setFixedSize(width,32)
        for label in self.findChildren(QLabel):
            if label.objectName()=='hint' and label not in (self.status,self.edit_state):self.as_help(label)
        self.as_help(self.pipeline_summary)
        self.setStyleSheet(STYLE.replace('__ASSETS__',(Path(__file__).parent/'ui_shell_assets').as_posix())+EXTRA_STYLE)
        self.shell_tabs.currentChanged.connect(self.select_shell_tab);self.view_tabs.currentChanged.connect(lambda _:self.sync_shell());self.queue_list.model().rowsInserted.connect(lambda *args:self.sync_shell());self.queue_list.model().rowsRemoved.connect(lambda *args:self.sync_shell())
        self.sync_file_context()

    def update_pipeline(self,*args):
        name=self.pipeline_picker.currentText();description='Original → Matte → Refined → Cutout' if '+' in name else 'Original → Matte → Cutout'
        self.pipeline_summary.setText(description)
        if self._shell_ready:self.as_help(self.pipeline_summary)
        text='Model-native size: 2048 × 2048' if name.startswith('BiRefNet') else 'Working size: 512 px'
        if 'VideoMaMa' in name:text+='\nSections: 4 frames'
        if 'ViTMatte' in name:text+='\nUnknown edge band: 24 px'
        self.defaults_summary.setText(text)
        self.roto_size.setValue(512);self.roto_chunk.setValue(4);self.trimap_width.setValue(24)
        self.sync_shell()

    def toggle_maximize(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()
        self.maximize_button.setIcon(navigation_icon('restore' if self.isMaximized() else 'maximize'))
        self.maximize_button.setToolTip('Restore' if self.isMaximized() else 'Maximize')

    def resize_edges(self,pos):
        if self.isMaximized() or self.isFullScreen():return Qt.Edges()
        edges=Qt.Edges()
        if pos.x()<6:edges|=Qt.LeftEdge
        elif pos.x()>=self.width()-6:edges|=Qt.RightEdge
        if pos.y()<6:edges|=Qt.TopEdge
        elif pos.y()>=self.height()-6:edges|=Qt.BottomEdge
        return edges

    def eventFilter(self,watched,event):
        if event.type() in (QEvent.MouseMove,QEvent.MouseButtonPress) and isinstance(watched,QWidget) and watched.window() is self and hasattr(self,'window_header'):
            pos=self.mapFromGlobal(event.globalPosition().toPoint());edges=self.resize_edges(pos)
            if event.type()==QEvent.MouseMove:
                if edges in (Qt.TopEdge|Qt.LeftEdge,Qt.BottomEdge|Qt.RightEdge):self.setCursor(Qt.SizeFDiagCursor)
                elif edges in (Qt.TopEdge|Qt.RightEdge,Qt.BottomEdge|Qt.LeftEdge):self.setCursor(Qt.SizeBDiagCursor)
                elif edges & (Qt.LeftEdge|Qt.RightEdge):self.setCursor(Qt.SizeHorCursor)
                elif edges & (Qt.TopEdge|Qt.BottomEdge):self.setCursor(Qt.SizeVerCursor)
                elif self.testAttribute(Qt.WA_SetCursor):self.unsetCursor()
            if event.type()==QEvent.MouseButtonPress and event.button()==Qt.LeftButton and edges and self.windowHandle():
                if self.windowHandle().startSystemResize(edges):event.accept();return True
        return super().eventFilter(watched,event)

    def closeEvent(self,event):
        super().closeEvent(event)
        if event.isAccepted() and QApplication.instance():QApplication.instance().removeEventFilter(self)

    def choose_viewport_model(self,name):
        if not self._shell_ready or not name:return
        if name in ('MatAnyone','MatAnyone 2','ViTMatte','VideoMaMa'):self.manual_refiner.setCurrentText(name)
        else:self.manual_selector.setCurrentText(name)
        self.workflow_tabs.setCurrentIndex(1);self.sync_shell()

    def choose_tool(self,tool):
        self._active_tool=tool;self.canvas.shell_tool=tool
        self.canvas.set_mode('view' if tool in ('hand','zoom') else tool)
        self.canvas.setCursor(Qt.OpenHandCursor if tool=='hand' else Qt.ArrowCursor if tool=='zoom' else Qt.CrossCursor)
        for name,button in self.tool_buttons.items():button.setChecked(name==tool)
        if tool=='hand':self.tool_popup.hide()
        else:
            page=1 if tool=='zoom' else 2 if tool.startswith('paint_') else 0
            self.popup_stack.setCurrentIndex(page);self.popup_title.setText(('Guidance objects','Zoom','Paint white' if tool=='paint_white' else 'Erase / paint black')[page]);self.sync_guidance_state()
            widget=self.popup_stack.currentWidget();widget.layout().activate();self.popup_stack.setFixedHeight(widget.sizeHint().height());self.tool_popup.adjustSize();self.tool_popup.show();self.tool_popup.raise_()

    def set_guidance_state(self,selected):
        key='box' if self._active_tool=='box' else 'point';self._guidance_states[key]=selected;self.sync_guidance_state()

    def sync_guidance_state(self):
        selected=self._guidance_states['box' if self._active_tool=='box' else 'point'];self.canvas.guidance_select_mode=selected
        self.place_guidance_button.setChecked(not selected);self.select_guidance_button.setChecked(selected)
        for tool in ('box','positive','negative'):self.tool_buttons[tool].setIcon(navigation_icon(tool,self._guidance_states['box' if tool=='box' else 'point']))
        self.sync_guidance_selection()

    def select_guidance_object(self,kind,n):
        row=next((i for i in range(self.guidance_picker.count()) if self.guidance_picker.itemData(i)==(kind,n)),-1)
        if row>=0:self.guidance_picker.setCurrentIndex(row);self.sync_guidance_selection()

    def sync_guidance_selection(self,*args):
        if not self._shell_ready:return
        selected=self.guidance_picker.currentData();self.canvas.selected_guidance=selected
        self.guidance_objects.blockSignals(True);self.guidance_objects.setCurrentRow(self.guidance_picker.currentIndex());self.guidance_objects.blockSignals(False);self.canvas.update()

    def adjust_brush(self,direction):
        if self._active_tool.startswith('paint_'):
            value=self.brush_size.value();self.brush_size.setValue(value+direction*(5 if value<50 else 10 if value<200 else 25))

    def as_help(self,label):
        label.setToolTip(label.text());label.setAccessibleName('Help: '+label.text());label.setText('?');label.setObjectName('helpIcon');label.setWordWrap(False);label.setFixedSize(20,20)

    def set_queue_button_state(self,running=False):
        self.queue_run_button.setText('Stop / Cancel' if running else 'Run queue')
        self.queue_run_button.setProperty('running',running)
        self.queue_run_button.setToolTip('Stop the active queue and cancel remaining queued jobs.' if running else 'Run the queue with the chosen workflow.')
        self.queue_run_button.style().unpolish(self.queue_run_button);self.queue_run_button.style().polish(self.queue_run_button)

    def sync_file_context(self,*args):
        files=self.file_tabs.currentIndex()==0
        for widget in (self.file_count,self.files_help,self.add_queue_button,self.clear_queue_button,self.load_results_button,self.queue_run_button,self.apply_masks_preview):widget.setVisible(files)
        self.clear_console_button.setVisible(not files)

    def sync_shell(self):
        if not self._shell_ready:return
        index=2 if self._refined_preview else {0:0,1:1,2:3}[self.view_tabs.currentIndex()]
        self.shell_tabs.blockSignals(True);self.shell_tabs.setCurrentIndex(index);self.shell_tabs.blockSignals(False);self.viewer_stack.setCurrentIndex(1 if index==2 else 0)
        self.remove_guidance_button.setVisible(index==0);self.clear_button.setVisible(index==0)
        self.export_matte_button.hide();self.export_button.hide();self.save_edits_button.hide()
        self.result_action.setVisible(index!=0);self.result_action.setText('Save refined' if index==2 else 'Export PNG' if index==3 else 'Save matte')
        self.result_action.setEnabled(index in (1,3) and self.alpha is not None)
        for button in self.shell_guide_tools:button.setVisible(index==0)
        for button in self.shell_paint_tools:button.setVisible(index in (1,2));button.setEnabled(index==1 and self.alpha is not None)
        self.canvas_frame.rail.adjustSize()
        if index!=self._last_channel:
            self._last_channel=index;self.tool_popup.hide()
            if self._active_tool not in ('hand','zoom'):self._active_tool='box' if index==0 else 'paint_white' if index in (1,2) else 'hand'
        self.canvas.shell_tool=self._active_tool;self.canvas.set_mode('view' if self._active_tool in ('hand','zoom') else self._active_tool)
        for tool,button in self.tool_buttons.items():button.setChecked(tool==self._active_tool)
        choices=[self.manual_refiner.itemText(i) for i in range(self.manual_refiner.count()) if not self.manual_refiner.itemText(i).startswith('None')] if index==2 else [self.manual_selector.itemText(i) for i in range(self.manual_selector.count())]
        if self.workflow_tabs.currentIndex()==0:
            pipeline=self.pipeline_picker.currentText()
            chosen=pipeline.split(' + ')[1] if index==2 and ' + ' in pipeline else 'MatAnyone' if index==2 else pipeline.split(' + ')[0].split(' — ')[0]
        else:chosen=self.manual_refiner.currentText() if index==2 else self.manual_selector.currentText()
        self.viewport_model.blockSignals(True)
        if [self.viewport_model.itemText(i) for i in range(self.viewport_model.count())]!=choices:
            self.viewport_model.clear();self.viewport_model.addItems(choices)
        if chosen in choices:self.viewport_model.setCurrentText(chosen)
        self.viewport_model.blockSignals(False)
        self.viewport_model.setVisible(index in (1,2) and bool(choices));self.edit_help.setVisible(index!=0)
        has_result=self._shell_stage_states.get(str(self.current_media_path),(False,False,False))[1] if index==2 else self.alpha is not None
        self.generate_button.setText(('Redo ' if has_result else 'Generate ')+('refine' if index==2 else 'matte'));self.test_frame_button.setText('Test refinement' if index==2 else 'Test frame' if self.video_path else 'Test image')
        self.generate_button.setVisible(index in (1,2));self.test_frame_button.setVisible(index in (1,2));self.resume_button.setVisible(index in (1,2) and bool(self.video_path))
        if not getattr(self,'_production_ready',False):
            for button in (self.generate_button,self.test_frame_button,self.resume_button,self.queue_run_button,self.get_sam3_button,self.redo_matte,self.stop_button):button.setEnabled(False)
            self.set_queue_button_state(False)
        for button in self.toolbar_actions:button.setMinimumWidth(0)
        self.edit_help.setToolTip(self.edit_state.text() or 'No saved result selected.');self.edit_help.setAccessibleName('Result status')
        refiner=self.manual_refiner.currentText();selector=self.manual_selector.currentText();refining=not refiner.startswith('None')
        self.resolution_settings.setVisible(refining);self.roto_size.setEnabled(refining)
        self.video_settings.setVisible(refiner=='VideoMaMa');self.vit_settings.setVisible(refiner=='ViTMatte');self.sam3_settings.setVisible(selector=='SAM 3')
        self.roto_text.setVisible(selector=='SAM 3')
        self.native_settings.setVisible(not refining);self.native_settings.setText('Resolution: 2048 × 2048' if selector=='BiRefNet HR' else 'Resolution: model-native')
        self.manual_masks.setEnabled(refining)
        self.sync_file_context()
        self.file_count.setText(f'{self.queue_list.count()} files')
        objects=[self.guidance_picker.itemText(i) for i in range(self.guidance_picker.count())]
        if [self.guidance_objects.item(i).text() for i in range(self.guidance_objects.count())]!=objects:
            self.guidance_objects.blockSignals(True);self.guidance_objects.clear();self.guidance_objects.addItems(objects);self.guidance_objects.setCurrentRow(self.guidance_picker.currentIndex());self.guidance_objects.blockSignals(False)
        self.sync_guidance_state()
        message=self.status.text()
        if message and message!=getattr(self,'_last_console_status',None):self._last_console_status=message;self.append_console(message+'\n')


EXTRA_STYLE='''
QMainWindow { background:transparent; }
QWidget#shellRoot { border-radius:12px; }
QToolButton#appMenu { padding:4px 12px; }
QToolButton#appMenu::menu-indicator { image:none; width:0; }
QToolButton#windowControl { padding:4px; background:#f7f5ed; }
QToolButton#windowClose { padding:4px; background:#f5c6c5; }
QToolButton#windowClose:hover { background:#ef9896; }
QWidget#attachedWorkflow { border-left:1px solid #b7b9ae; }
QFrame#optionsBar { border:none; background:#f7f5ed; border-radius:8px; }
QLabel#workflowDefaults { border:1px solid #c1c1b6; border-radius:9px; padding:12px; background:#f7f5ed; }
QTabBar::tab { padding:8px 13px; }
QFrame#toolRail { background:#fffdf7; }
QLabel#helpIcon { border:1px solid #b7b9ae; border-radius:9px; color:#77786e; font-weight:600; qproperty-alignment:AlignCenter; }
QFrame#optionsBar QPushButton { font-size:11px; padding:4px 6px; }
QProgressBar { border-radius:2px; }
QProgressBar::chunk { border-radius:2px; }
QPushButton#queueTile[running="true"] { background:#f5c6c5; }
'''

