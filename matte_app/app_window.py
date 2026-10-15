"""Approved studio UI connected to independent processing, preview and result storage."""
from pathlib import Path
from dataclasses import replace
from collections import OrderedDict
import json
import queue
import uuid
import math
from PIL import Image, ImageChops
from PySide6.QtCore import Qt,QTimer,Signal
from PySide6.QtGui import QPixmap,QImage,QAction
from PySide6.QtWidgets import QFileDialog,QMessageBox,QApplication,QSlider,QLabel,QPushButton,QToolButton,QHBoxLayout,QWidget,QCheckBox
from ui_shell_layout import ShellPreviewWindowV3,ReviewCanvas
from ui_components import ThumbnailFiles
from .window import MatteWindow,PreviewWorker,PlaybackWorker,CancellableWorker,VIDEO_SUFFIXES
from .workflow import WorkflowMixin
from .guidance_state import GuidanceMixin
from .guidance_store import encode,decode
from .models import PointHint,BoxHint,RotoHint
from .matte_editor import MatteEditor
from .roto import ROTO_MODELS,readiness
from .video import load_frame
from .preview_renderer import PreviewRenderer
from .result_store import ResultStore,key
from .pipeline_jobs import Job,PipelineWorker

SELECTORS={'SAM 2':'SAM 2 — Tracking','SAM 3':'SAM 3 — Tracking','BiRefNet HR':'BiRefNet HR Matting','LiteSDMatte':'LiteSDMatte (guided)'}
SELECTORS.update({'SAM2Matting Tiny':'SAM2Matting Tiny','SAM2Matting Base+':'SAM2Matting Base+','SAM 3.1':'SAM 3.1 — Tracking'})
REFINE={'MatAnyone 2':'MatAnyone 2 — Saved masks','MatAnyone':'MatAnyone — Saved masks','ViTMatte':'ViTMatte — Saved masks','VideoMaMa':'VideoMaMa — Saved masks'}


class AppCanvas(ReviewCanvas):
    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton and self.shell_tool in ('box','positive','negative') and not self.guidance_select_mode and event.modifiers() & (Qt.AltModifier|Qt.ShiftModifier):
            point=self._to_normalized(event.position())
            if point:self.point_added.emit(*point,not bool(event.modifiers() & Qt.AltModifier))
            return
        super().mousePressEvent(event)


class CatalogWorker(CancellableWorker):
    found=Signal(str,object,object)
    def __init__(self,paths,store,need_thumbnails=None):
        super().__init__();self.paths=tuple(paths);self.store=store
        self.need_thumbnails=set(paths) if need_thumbnails is None else set(need_thumbnails)
    def execute(self):
        for path in self.paths:
            self.cancellation.check()
            try:
                state=self.store.scan(path);image=None
                if path in self.need_thumbnails:
                    if Path(path).suffix.lower() in VIDEO_SUFFIXES:
                        folder=next((self.store.existing_folder(path,s) for s in ('matte','refined') if state[s]['available']),None)
                        if folder:
                            frames=self.store.frames(path,'matte' if folder==self.store.existing_folder(path,'matte') else 'refined')
                            with Image.open(folder/f'frame_{frames[0]:06d}.png') as raw:image=raw.convert('RGB')
                        else:image=load_frame(path,1,self.cancellation).convert('RGB')
                    else:
                        with Image.open(path) as raw:image=raw.convert('RGB')
                    image.thumbnail((160,90))
                self.found.emit(path,state,image)
            except Exception as error:self.found.emit(path,{'error':str(error)},None)



class StudioFiles(ThumbnailFiles):
    def thumbnail(self,path):return self.cache.get(path,QPixmap())


class DraftWorker(CancellableWorker):
    def __init__(self,store,path,source,stage,frame,snapshot):
        super().__init__();self.store=store;self.path=path;self.source=source;self.stage=stage;self.frame=frame;self.snapshot=snapshot
    def execute(self):
        pending=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp');self.path.parent.mkdir(parents=True,exist_ok=True)
        try:
            base,tiles=self.snapshot;alpha=base.copy()
            for tile,image in tiles.items():alpha.paste(image,tile)
            alpha.save(pending,'PNG',compress_level=1)
            with self.store.lock:
                self.cancellation.check();self.store.replace_file(pending,self.path);self.store.choose_alpha(self.source,self.stage,self.frame)
        finally:pending.unlink(missing_ok=True)


class StudioWindow(ShellPreviewWindowV3):
    canvas_class=AppCanvas

    def _build_ui(self):
        from .studio_controls import build_controls
        build_controls(self)

    def _update_navigation(self):
        if not self._production_ready:return
        MatteWindow._update_navigation(self);self.sync_shell()

    def __init__(self,state_dir=None):
        self._shell_ready=False;self._production_ready=False
        self._job_worker=None;self._draft_worker=None;self._catalog_worker=None;self._catalog_pending=set();self._preview_pending=None
        self._requested_preview=None;self._load_serial=0;self._processing_source='';self._processing_stage=''
        self._draft_models={};self._editors=OrderedDict();self._cutout_stage={};self._queue_states={};self._catalog={};self._shell_stage_states={}
        self._refined_preview=False;self._alpha_stage='matte';self._closing=False;self._follow=True
        self._queued_generate=None;self._queued_pipeline=None
        self._estimated_foreground=None
        self._applying_live=False
        self._opening_project=False
        MatteWindow.__init__(self)
        if state_dir:
            self.results_index_path=Path(state_dir)/'results_index.json';self.results_index_path.parent.mkdir(parents=True,exist_ok=True)
            try:self.queue_outputs=json.loads(self.results_index_path.read_text())
            except (OSError,ValueError):self.queue_outputs={}
        self.store=ResultStore(legacy=self.queue_outputs)
        self.resize(1440,980);self.build_shell();self.setWindowTitle('Matte App')
        self.shell_tabs.setTabText(2,'Cutout');self.shell_tabs.setTabVisible(3,False)
        self.refine_view_controls=QWidget();controls=QHBoxLayout(self.refine_view_controls);controls.setContentsMargins(0,0,0,0)
        self.alpha_view=QToolButton();self.alpha_view.setText('α');self.alpha_view.setCheckable(True);self.alpha_view.setToolTip('Show the refined alpha instead of the cutout');controls.addWidget(self.alpha_view)
        controls.addWidget(QLabel('Original'));self.original_opacity=QSlider(Qt.Horizontal);self.original_opacity.setRange(0,100);self.original_opacity.setFixedWidth(100);self.original_opacity.setToolTip('Opacity of the original image behind the edited cutout');controls.addWidget(self.original_opacity)
        self.original_opacity_label=QLabel('0%');self.original_opacity_label.setFixedWidth(34);controls.addWidget(self.original_opacity_label)
        self.foreground_estimation=QCheckBox('Estimate foreground');self.foreground_estimation.setToolTip('Recover foreground colours at soft alpha edges when exporting or applying masks. Alpha stays unchanged. A fresh estimate appears in Cutout after export; painting requires a new estimate.')
        self.foreground_estimation.setStyleSheet('QCheckBox::indicator { width:14px; height:14px; border:1px solid #262828; border-radius:3px; background:#fffdf7; } QCheckBox::indicator:checked { background:#f6d85c; border:2px solid #262828; }')
        self.foreground_estimation.toggled.connect(lambda _:self._refresh_canvas())
        controls.addWidget(self.foreground_estimation)
        self.toolbar.layout().insertWidget(1,self.refine_view_controls)
        self.cutout_save_button=QPushButton('Export PNG');self.cutout_save_button.setIcon(self.result_action.icon());self.cutout_save_button.clicked.connect(self.export_cutout);self.toolbar.layout().insertWidget(self.toolbar.layout().count()-1,self.cutout_save_button)
        self.pipeline_clip_button=QPushButton('Run pipeline for this clip')
        self.pipeline_frame_button=QPushButton('Run pipeline for this frame')
        self.pipeline_clip_button.clicked.connect(lambda:self.run_selected_pipeline(True))
        self.pipeline_frame_button.clicked.connect(lambda:self.run_selected_pipeline(False))
        self.toolbar.layout().insertWidget(0,self.pipeline_clip_button)
        self.toolbar.layout().insertWidget(1,self.pipeline_frame_button)
        self.preview_renderer=PreviewRenderer();self._loaded_frame_identity=None
        self._zoom_refresh=QTimer(self);self._zoom_refresh.setSingleShot(True);self._zoom_refresh.setInterval(80);self._zoom_refresh.timeout.connect(lambda:self._refresh_canvas(False))
        self._opacity_refresh=QTimer(self);self._opacity_refresh.setSingleShot(True);self._opacity_refresh.setInterval(25);self._opacity_refresh.timeout.connect(lambda:self._refresh_canvas(False))
        self._navigation_timer.setInterval(75)
        self.original_opacity.valueChanged.connect(lambda value:(self.original_opacity_label.setText(f'{value}%'),self._opacity_refresh.start()))
        self.alpha_view.toggled.connect(lambda _:self._refresh_canvas())
        self.setStyleSheet(self.styleSheet()+'\nQMessageBox { background:#efecdf; } QMessageBox QLabel { color:#262828; }')
        self.thumbnail_delegate=StudioFiles(self);self.queue_list.setItemDelegate(self.thumbnail_delegate)
        self._shell_ready=self._production_ready=True
        from . import roto
        self._previous_warm=roto.ENABLE_WARM_SESSIONS;roto.ENABLE_WARM_SESSIONS=True
        self.canvas.stroke_started.disconnect();self.canvas.stroke_started.connect(self.begin_edit_stroke)
        self._paint_dirty_bounds=None
        self._paint_refresh.timeout.disconnect();self._paint_refresh.setInterval(0);self._paint_refresh.timeout.connect(self.refresh_paint_preview)
        self._draft_save_timer=QTimer(self);self._draft_save_timer.setSingleShot(True);self._draft_save_timer.setInterval(600);self._draft_save_timer.timeout.connect(self.save_idle_draft)
        from PySide6.QtGui import QShortcut,QKeySequence
        self.cancel_shortcut=QShortcut(QKeySequence('Escape'),self);self.cancel_shortcut.setContext(Qt.WidgetWithChildrenShortcut);self.cancel_shortcut.activated.connect(self.cancel_jobs)
        self.result_action.clicked.disconnect();self.result_action.clicked.connect(self.save_frame_edits)
        self.apply_masks_preview.clicked.connect(self.apply_masks)
        self.manual_masks.currentIndexChanged.connect(self.choose_mask_folder)
        self._job_tick=QTimer(self);self._job_tick.setInterval(100);self._job_tick.timeout.connect(self.job_frame_tick)
        self.follow_action=QAction('Follow processing',self);self.follow_action.setCheckable(True);self.follow_action.setChecked(True);self.follow_action.toggled.connect(self.set_follow)
        menus={b.text():b.menu() for b in self.app_menu_buttons};menus['View'].addAction(self.follow_action)
        menus['File'].addAction('Set project root…',self.choose_project_root)
        menus['File'].addAction('Install models & dependencies…',self.open_installer)
        menus['File'].addAction('Get SAM 3 weights',self.get_sam3_weights)
        if 'Help' in menus and menus['Help']:
            menus['Help'].addAction('Install models & dependencies…',self.open_installer)
        menus['View'].addAction('Brush performance report',self.show_brush_report)
        self._catalog_timer=QTimer(self);self._catalog_timer.setSingleShot(True);self._catalog_timer.setInterval(0);self._catalog_timer.timeout.connect(self.start_catalog)
        self.queue_list.model().rowsInserted.connect(self.queue_membership_changed)
        self.queue_list.model().rowsRemoved.connect(self.queue_membership_changed)
        self.refresh_installed_pipelines()
        self.sync_shell();self._refresh_canvas()

    def current_stage(self):return self._alpha_stage
    def channel(self):return self.shell_tabs.currentIndex() if self._shell_ready else 0
    def set_follow(self,value):self._follow=value

    def draft_path(self,source,stage,frame):return self.store.media_folder(source)/'.drafts'/stage/f'frame_{frame:06d}.png'
    def stash_draft(self):
        if not self._production_ready or self._opening_project or isinstance(self._active_worker,PlaybackWorker) or self._loaded_frame_identity!=(key(self.current_media_path),self.current_frame) or self.channel() not in (1,2) or self.original is None or self.alpha is None or not self.current_media_path or not self._matte_dirty:return
        path=self.draft_path(self.current_media_path,self.current_stage(),self.current_frame);path.parent.mkdir(parents=True,exist_ok=True)
        if self._draft_worker and self._draft_worker.path==path:self._draft_worker.cancel()
        pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            self.alpha.save(pending,'PNG',compress_level=1)
            with self.store.lock:self.store.replace_file(pending,path);self.store.choose_alpha(self.current_media_path,self.current_stage(),self.current_frame)
        finally:pending.unlink(missing_ok=True)
        self._editors[(key(self.current_media_path),self.current_frame,self.current_stage())]=self.matte_editor
        while len(self._editors)>4:self._editors.popitem(last=False)

    def stage_alpha(self,stage):
        if not self.current_media_path or self.original is None:return None
        draft=self.draft_path(self.current_media_path,stage,self.current_frame)
        if draft.is_file():
            with Image.open(draft) as image:
                if image.size!=self.original.size:raise ValueError('Draft dimensions differ from source')
                return image.convert('L')
        cached_preview=getattr(self,'_preview_alpha',None)
        if cached_preview is not None and getattr(self,'_preview_alpha_stage',None)==stage:
            if cached_preview.size!=self.original.size:raise ValueError('Saved mask dimensions do not match source')
            return cached_preview
        return self.store.alpha(self.current_media_path,stage,self.current_frame,self.original.size)

    def select_shell_tab(self,index):
        if not self._production_ready:return ShellPreviewWindowV3.select_shell_tab(self,index)
        if index==3:index=2
        if isinstance(self._active_worker,PlaybackWorker):
            self._pending_channel=index;self._active_worker.cancel();self._playback_timer.stop();return
        if self._job_worker and not self._applying_live:self.follow_action.setChecked(False)
        self.canvas.finish_stroke()
        preferred=self.store.preferred_alpha(self.current_media_path,self.current_frame) if self.current_media_path else 'matte'
        editor_key=(key(self.current_media_path),self.current_frame,preferred) if self.current_media_path else None
        # Switching presentation keeps the exact live mask and its undo history.
        shared=(not self._applying_live and not isinstance(self._active_worker,PreviewWorker) and self.channel() in (1,2) and index in (1,2) and preferred==self.current_stage()
                and self.alpha is not None and self.alpha is self.matte_editor.alpha
                and getattr(self,'_loaded_editor_key',None)==editor_key)
        if not shared:self.stash_draft()
        self._refined_preview=index==2
        self.shell_tabs.blockSignals(True);self.shell_tabs.setCurrentIndex(index);self.shell_tabs.blockSignals(False)
        self.view_tabs.blockSignals(True);self.view_tabs.setCurrentIndex(0 if index==0 else 1);self.view_tabs.blockSignals(False)
        self._alpha_stage=preferred;stage=preferred
        if shared:
            self._refresh_canvas();self._update_navigation();return
        try:
            self.alpha=self.stage_alpha(stage)
            if self.alpha is None and stage=='refined':
                stage='matte';self._alpha_stage=stage;self.alpha=self.stage_alpha(stage)
        except Exception as error:self.report_error(str(error));self.alpha=None
        self._matte_dirty=bool(self.current_media_path and self.draft_path(self.current_media_path,stage,self.current_frame).is_file())
        editor_key=(key(self.current_media_path),self.current_frame,stage) if self.current_media_path else None
        cached=self._editors.get(editor_key)
        if cached and self.alpha is not None and cached.alpha is not None and cached.alpha.size==self.alpha.size and (cached.alpha is self.alpha or (cached.alpha.getextrema()==self.alpha.getextrema() and not ImageChops.difference(cached.alpha,self.alpha).getbbox())):self.matte_editor=cached;self.alpha=cached.alpha
        else:self.matte_editor=MatteEditor();self.matte_editor.reset(self.alpha)
        self._loaded_editor_key=(key(self.current_media_path),self.current_frame,stage) if self.current_media_path else None
        if self.current_media_path:self.video_output_dir=str(self.store.existing_folder(self.current_media_path,stage))
        self._refresh_canvas();self._update_navigation()

    def _refresh_canvas(self,refresh_ui=True):
        if not self._production_ready:return
        if not self.original:self.canvas.set_image(None);self.sync_shell();return
        if isinstance(self._active_worker,PlaybackWorker):return
        image=self.original
        alpha=self.alpha
        index=self.channel()
        scale=min(1,1600*self.canvas._zoom/max(image.size));size=(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
        cutout=index==2 and not self.alpha_view.isChecked()
        resized=self.preview_renderer.render(self.cutout_source() if cutout else image,alpha if index else None,size,cutout,self.original_opacity.value(),image)
        self.canvas.set_image(resized,image.size);self.canvas.set_guidance(self.points,self.boxes)
        self._paint_dirty_bounds=None
        if refresh_ui:self.sync_shell()

    def _sync_zoom(self,zoom):
        self.zoom_percent.blockSignals(True);self.zoom_percent.setValue(round(zoom*100));self.zoom_percent.blockSignals(False)
        # QPainter scales the existing preview immediately; resample once after
        # wheel input settles rather than block every wheel event.
        if self._production_ready and hasattr(self,'_zoom_refresh') and not isinstance(self._active_worker,PlaybackWorker):self._zoom_refresh.start()

    def can_write_frame(self):
        return bool(self.current_media_path and self.original is not None and self.alpha is not None
                    and not isinstance(self._active_worker,(PreviewWorker,PlaybackWorker))
                    and self._requested_frame is None and self._preview_pending is None
                    and self._loaded_frame_identity==(key(self.current_media_path),self.current_frame))

    def guard_result_write(self):
        if self.can_write_frame():return True
        self.status.setText('Pause playback and wait for the selected frame to load before saving.');return False

    def sync_shell(self):
        if not self._production_ready:return ShellPreviewWindowV3.sync_shell(self)
        ShellPreviewWindowV3.sync_shell(self)
        index=self.channel();busy=self._job_worker is not None or self._active_worker is not None and not isinstance(self._active_worker,(PreviewWorker,PlaybackWorker))
        missing=index!=0 and self.alpha is None
        self.viewer_stack.setCurrentIndex(1 if missing else 0)
        if missing:
            from PySide6.QtWidgets import QLabel
            label=self.viewer_stack.widget(1).findChild(QLabel)
            if label:label.setText('No '+'alpha'+' for this frame')
        ready=self.original is not None and not isinstance(self._active_worker,(PreviewWorker,PlaybackWorker))
        self.canvas.setEnabled(self.original is not None);self.canvas.set_painting_enabled(ready and index in (1,2) and self.alpha is not None)
        for control in (self.generate_button,self.test_frame_button,self.resume_button):control.setEnabled(ready and not busy)
        if index==2 and not self.backend_refiner():
            for control in (self.generate_button,self.test_frame_button,self.resume_button):control.setEnabled(False)
        for control in (self.viewport_model,self.workflow_tabs,self.pipeline_picker,self.manual_selector,self.manual_refiner,self.manual_masks,self.roto_size,self.roto_chunk,self.trimap_width,self.roto_text):control.setEnabled(not busy)
        for control in self.shell_guide_tools:control.setEnabled(ready and index==0)
        for control in self.shell_paint_tools:control.setEnabled(ready and index in (1,2) and self.alpha is not None)
        self.brush_size.setEnabled(ready and index in (1,2) and self.alpha is not None)
        self.remove_guidance_button.setEnabled(ready and bool(self.guidance_picker.currentData()));self.clear_button.setEnabled(ready and bool(self.points or self.boxes))
        self.undo_brush_button.setEnabled(ready and bool(self.matte_editor.undo_strokes));self.redo_brush_button.setEnabled(ready and bool(self.matte_editor.redo_strokes))
        self.update_brush_history()
        self.result_action.setEnabled(self.can_write_frame())
        self.refine_view_controls.setVisible(index==2);self.cutout_save_button.setVisible(index in (1,2));self.cutout_save_button.setEnabled(self.can_write_frame())
        self.original_opacity.setEnabled(not self.alpha_view.isChecked());self.original_opacity_label.setEnabled(not self.alpha_view.isChecked())
        self.apply_masks_preview.setEnabled(not busy and self.queue_list.count()>0)
        self.set_queue_button_state(busy);self.queue_run_button.setEnabled(busy or self.queue_list.count()>0)
        if busy and (self._job_worker and self._job_worker.cancellation.requested or self._cancel_requested):self.queue_run_button.setText('Stopping…');self.queue_run_button.setEnabled(False)
        self.queue_list.setEnabled(True)
        # Add/remove membership is frozen, but existing items remain inspectable.
        for button in (self.add_queue_button,self.clear_queue_button,self.remove_queue_button,self.load_results_button):button.setEnabled(not busy)
        self.edit_state.setText('Unsaved draft' if self._matte_dirty else 'Saved result' if self.alpha is not None else 'No result for this channel/frame')
        self.edit_help.setToolTip(self.edit_state.text())
        if self.current_media_path and index in (1,2):
            saved=self._catalog.get(key(self.current_media_path),{}).get(self.current_stage(),{})
            if saved.get('model'):self.edit_help.setToolTip(self.edit_help.toolTip()+'\nSaved model: '+saved['model'])
        self.pipeline_clip_button.setVisible(index==0 and bool(self.video_path))
        self.pipeline_frame_button.setVisible(index==0)
        for button in (self.pipeline_clip_button,self.pipeline_frame_button):button.setEnabled(ready and not busy)
        self.pipeline_frame_button.setToolTip('Run the selected Auto or Manual pipeline on the displayed frame. Save the resulting alpha when ready.')
        self.pipeline_clip_button.setToolTip('Run the selected Auto or Manual pipeline on every frame of this clip.')
        if index in (1,2):self.result_action.setText('Save matte')
        self.shell_tabs.setTabToolTip(1,'Current alpha. Each matte or refinement step replaces this mask.')
        self.shell_tabs.setTabToolTip(2,'Cutout made from the same editable alpha and undo history as Matte.')
        self.follow_action.setEnabled(busy)
        self.sam3_settings.setVisible(self.manual_selector.currentText().startswith('SAM 3'));self.roto_text.setVisible(self.manual_selector.currentText().startswith('SAM 3'))
        self.cancel_shortcut.setEnabled(busy)
        for label,action in self.file_actions.items():action.setEnabled(self.can_write_frame() if label.startswith(('Export','Save frame')) else not busy)
        self.queue_list.viewport().update()

    def choose_viewport_model(self,name):
        ShellPreviewWindowV3.choose_viewport_model(self,name)

    def _update_edit_controls(self):
        if not self._production_ready:return MatteWindow._update_edit_controls(self)
        selected=self.guidance_picker.currentData();self.guidance_picker.blockSignals(True);self.guidance_picker.clear()
        for n,p in enumerate(self.points):self.guidance_picker.addItem(f'Point {"+" if p.positive else "−"} {n+1} ({p.x:.2f}, {p.y:.2f})',('point',n))
        for n,b in enumerate(self.boxes):self.guidance_picker.addItem(f'Box {n+1} ({b.left:.2f}, {b.top:.2f})',('box',n))
        row=next((n for n in range(self.guidance_picker.count()) if self.guidance_picker.itemData(n)==selected),self.guidance_picker.count()-1)
        self.guidance_picker.setCurrentIndex(row);self.guidance_picker.blockSignals(False)
        self.sync_shell()

    def _update_roto_controls(self):
        if not self._production_ready:return MatteWindow._update_roto_controls(self)
    def update_workflow(self):
        if not self._production_ready:return WorkflowMixin.update_workflow(self)
        video=bool(self.video_path);self.range_panel.setVisible(video)
        total=self.video_info.frame_count if self.video_info else 1
        selected=self.range_picker.currentIndex()==2
        self.range_to.setVisible(selected)
        for control in (self.range_start,self.range_end):
            oldmax=control.maximum();control.setMaximum(total);control.setVisible(selected)
            if control is self.range_end and oldmax==1:control.setValue(total)
            control.setEnabled(not self._job_worker and selected)
        self.sync_shell()
    def _set_busy(self,busy):
        if not self._production_ready:return MatteWindow._set_busy(self,busy)
        self.sync_shell();self._update_navigation()
    def _reset_matte_edits(self):
        if not self._production_ready:return MatteWindow._reset_matte_edits(self)
        self.matte_editor.reset(self.alpha)
    def _select_brush(self,mode):
        if self._production_ready:self.choose_tool(mode)
        else:MatteWindow._select_brush(self,mode)

    def paint_matte(self,x1,y1,x2,y2,value):
        if self.channel() not in (1,2) or self.alpha is None or isinstance(self._active_worker,PreviewWorker):return
        with self.canvas.telemetry.span('brush_draw_ms'):bounds=self.matte_editor.paint((x1,y1),(x2,y2),self.brush_size.value(),value)
        if bounds:
            old=self._paint_dirty_bounds
            self._paint_dirty_bounds=(min(old[0],bounds[0]),min(old[1],bounds[1]),max(old[2],bounds[2]),max(old[3],bounds[3])) if old else bounds
        if not self._paint_refresh.isActive():self._paint_refresh.start()
    def refresh_paint_preview(self):
        with self.canvas.telemetry.span('preview_patch_ms'):self._refresh_paint_patch()
    def _refresh_paint_patch(self):
        bounds=self._paint_dirty_bounds;self._paint_dirty_bounds=None
        if not bounds or self.alpha is None or self.canvas._image is None or self.channel() not in (1,2):return
        width,height=self.canvas._image.size;sw,sh=self.alpha.size
        left=max(0,math.floor(bounds[0]*width/sw)-2);top=max(0,math.floor(bounds[1]*height/sh)-2)
        right=min(width,math.ceil(bounds[2]*width/sw)+2);bottom=min(height,math.ceil(bounds[3]*height/sh)+2)
        patch=self.preview_renderer.patch(self.cutout_source(),self.alpha,(width,height),(left,top,right,bottom),
                                           self.channel()==2 and not self.alpha_view.isChecked(),self.original_opacity.value(),self.original)
        self.canvas.update_image_region((left,top,right,bottom),patch)
    def begin_edit_stroke(self):
        self._draft_save_timer.stop()
        if self._job_worker:self.follow_action.setChecked(False)
        with self.canvas.telemetry.span('stroke_begin_ms'):self.matte_editor.begin_stroke()
        self._editors[(key(self.current_media_path),self.current_frame,self.current_stage())]=self.matte_editor
    def finish_matte_stroke(self):
        if not self._production_ready:return MatteWindow.finish_matte_stroke(self)
        with self.canvas.telemetry.span('stroke_end_ms'):self.matte_editor.end_stroke()
        self._matte_dirty=True;self._paint_refresh.stop();self._cutout_stage[key(self.current_media_path)]=self.current_stage();self._draft_save_timer.start();self.refresh_paint_preview();self.update_brush_history()
    def show_brush_report(self):
        self.append_console(self.canvas.telemetry.report()+'\n');self.file_tabs.setCurrentIndex(1)
    def save_idle_draft(self):
        if self.channel() not in (1,2) or not self.can_write_frame():return
        if self.canvas._brush_down:self._draft_save_timer.start();return
        if self._matte_dirty and self.current_media_path:
            if self._draft_worker:self._draft_save_timer.start();return
            worker=DraftWorker(self.store,self.draft_path(self.current_media_path,self.current_stage(),self.current_frame),self.current_media_path,self.current_stage(),self.current_frame,self.matte_editor.immutable_snapshot())
            worker.setParent(self);self._draft_worker=worker;worker.failed.connect(self.report_error);worker.finished.connect(self.draft_finished);worker.start()
    def draft_finished(self):
        worker=self._draft_worker
        if worker:worker.wait();worker.deleteLater()
        self._draft_worker=None
        if self._closing:self.close()
    def update_brush_history(self):
        editable=self.channel() in (1,2) and self.alpha is not None and not isinstance(self._active_worker,(PreviewWorker,PlaybackWorker))
        undo=editable and bool(self.matte_editor.undo_strokes);redo=editable and bool(self.matte_editor.redo_strokes)
        self.undo_brush_button.setEnabled(undo);self.undo_brush_action.setEnabled(undo)
        self.redo_brush_button.setEnabled(redo);self.redo_brush_action.setEnabled(redo)
    def undo_brush(self):
        self.canvas.finish_stroke()
        if self.channel() not in (1,2) or not self.matte_editor.undo_strokes:return
        self._paint_dirty_bounds=self.matte_editor.undo_strokes[-1].bounds;self.matte_editor.undo();self._matte_dirty=True;self._draft_save_timer.start();self.refresh_paint_preview();self.update_brush_history()
    def redo_brush(self):
        self.canvas.finish_stroke()
        if self.channel() not in (1,2) or not self.matte_editor.redo_strokes:return
        self._paint_dirty_bounds=self.matte_editor.redo_strokes[-1].bounds;self.matte_editor.redo();self._matte_dirty=True;self._draft_save_timer.start();self.refresh_paint_preview();self.update_brush_history()

    def save_file_guidance(self,*args,**kwargs):
        if self._opening_project:return True
        if self._production_ready:
            self.model_picker.blockSignals(True);self.model_picker.setCurrentText(SELECTORS.get(self.manual_selector.currentText(),'SAM 2 — Tracking'));self.model_picker.blockSignals(False)
        result=GuidanceMixin.save_file_guidance(self,*args,**kwargs)
        if self._production_ready and self.current_media_path and (self.boxes or any(p.positive for p in self.points) or self.roto_text.text().strip()):self._queue_states.get(key(self.current_media_path),{}).pop('missing',None)
        return result
    def add_point(self,*args):
        if self._production_ready and self._job_worker:self.follow_action.setChecked(False)
        MatteWindow.add_point(self,*args);self._update_edit_controls();self.guidance_picker.setCurrentIndex(max(0,len(self.points)-1));self.sync_guidance_selection()
    def add_box(self,*args):
        if self._production_ready and self._job_worker:self.follow_action.setChecked(False)
        MatteWindow.add_box(self,*args);self._update_edit_controls();self.guidance_picker.setCurrentIndex(self.guidance_picker.count()-1);self.sync_guidance_selection()
    def remove_selected_guidance(self):
        row=self.guidance_picker.currentIndex();item=self.guidance_picker.currentData()
        if item:MatteWindow.remove_guidance(self,*item);self._update_edit_controls();self.guidance_picker.setCurrentIndex(max(0,row-1));self.sync_guidance_selection()
    def clear_guidance(self):MatteWindow.clear_guidance(self);self._update_edit_controls()
    def guard_frame_edits(self):self.stash_draft();return True

    def _start_worker(self,worker):worker.setParent(self);MatteWindow._start_worker(self,worker)
    def _load_media(self,path,output_dir=None,frame=1,info=None,restore_guidance=False):
        if not self._production_ready:return
        if not self.save_file_guidance():return
        self.canvas.finish_stroke();self.stash_draft();self._load_serial+=1
        request=(str(Path(path).resolve()),frame,info,self._load_serial)
        self._requested_preview=request;self._navigation_timer.stop()
        if self._active_worker is not None:
            self._preview_pending=request;self._active_worker.cancel();self._playback_timer.stop();return
        self.begin_preview(request)

    def begin_preview(self,request):
        path,frame,info,serial=request;self._restore_guidance=True;self._preview_loaded=False;self._requested_frame=frame
        stage=self.store.preferred_alpha(path,frame)
        folder=self.store.existing_folder(path,stage)
        if not (folder/f'frame_{frame:06d}.png').exists():folder=self.store.existing_folder(path,'matte')
        worker=PreviewWorker(path,frame,str(folder) if (folder/f'frame_{frame:06d}.png').exists() else None,info,cutout_dir=str(self.store.existing_folder(path,'cutout')))
        worker.serial=serial;worker.completed.connect(lambda preview,s=serial:self.accept_preview(preview,s));worker.failed.connect(self.report_error)
        self.preview_worker=worker;self.status.setText(f'Loading {Path(path).name}, frame {frame}');self._start_worker(worker)

    def accept_preview(self,preview,serial):
        if not self._requested_preview or serial!=self._requested_preview[3] or self._closing:return
        self.on_preview_ready(preview)

    def on_preview_ready(self,preview):
        changed=key(self.current_media_path)!=key(preview.path) if self.current_media_path else True
        self._preview_loaded=True;self.current_media_path=preview.path;self.current_frame=preview.frame;self._requested_frame=None
        self.video_path=preview.path if preview.info else None;self.video_info=preview.info;self.original=preview.original;self.video_output_dir=preview.output_dir
        self._loaded_frame_identity=(key(preview.path),preview.frame)
        if changed:self.canvas.reset_zoom()
        if not self.restore_file_guidance(preview.path,preview.frame,changed):
            if preview.frame==1 and preview.output_dir:MatteWindow._restore_saved_guidance(self,preview.output_dir,preview.frame);self.save_file_guidance()
            else:self.points.clear();self.boxes.clear();self.seed_path=self.seed_source='';self.roto_text.clear()
        self._opening_project=False
        index=getattr(self,'_pending_channel',self.channel())
        if hasattr(self,'_pending_channel'):del self._pending_channel
        self._matte_dirty=False
        self._preview_alpha=preview.alpha
        self._preview_alpha_stage=self.store.preferred_alpha(preview.path,preview.frame) if preview.path else 'matte'
        if preview.foreground is not None:
            self._estimated_foreground=(key(preview.path),preview.frame,id(preview.alpha),preview.foreground.convert('RGB'))
        try:
            self.select_shell_tab(index)
        finally:
            self._preview_alpha=None
            self._preview_alpha_stage=None
        self._update_edit_controls()
        self.status.setText(f'Loaded {Path(preview.path).name}, frame {preview.frame}')

    def _on_worker_finished(self):
        worker=self.sender();self._active_worker=None;self._cancel_requested=False
        worker.wait();worker.deleteLater()
        if isinstance(worker,PlaybackWorker):self._playback_timer.stop()
        pending=self._preview_pending;self._preview_pending=None
        self._set_busy(False)
        if self._closing:self.close()
        elif pending:self.begin_preview(pending)
        elif isinstance(worker,PlaybackWorker) and self.current_media_path:self._load_media(self.current_media_path,frame=self.current_frame,info=self.video_info)
        elif self._queued_pipeline is not None:
            clip=self._queued_pipeline;self._queued_pipeline=None;QTimer.singleShot(0,lambda:self.run_selected_pipeline(clip))
        elif self._queued_generate:
            args=self._queued_generate;self._queued_generate=None;QTimer.singleShot(0,lambda:self.generate_selected(*args))

    def review_queue_item(self,item):
        if item is None:return
        self._queued_generate=None;self._queued_pipeline=None
        if self._job_worker:self.follow_action.setChecked(False)
        self._load_media(item.text(),restore_guidance=True)
    def request_frame(self,frame):
        if self._job_worker:self.follow_action.setChecked(False)
        if isinstance(self._active_worker,PlaybackWorker) or not self.video_info:return
        frame=max(1,min(frame,self.video_info.frame_count))
        self._navigation_timer.stop()
        self._requested_frame=frame if frame!=self.current_frame else None
        for control in (self.frame_spin,self.frame_slider):
            control.blockSignals(True);control.setValue(frame);control.blockSignals(False)
        self._update_navigation()
        if self._requested_frame is not None and not self.frame_slider.isSliderDown():
            self._navigation_timer.start()
    def navigate_to_frame(self):
        if self.video_path:self._load_media(self.video_path,frame=self.frame_spin.value(),info=self.video_info)

    def queue_membership_changed(self,*args):
        paths={self.queue_list.item(i).text() for i in range(self.queue_list.count())};keys={key(p) for p in paths}
        current=key(self.current_media_path)
        for mapping in (self._catalog,self._queue_states,self._cutout_stage):
            for name in list(mapping):
                if name not in keys:mapping.pop(name,None)
        for mapping in (self._shell_stage_states,self.thumbnail_delegate.cache):
            for name in list(mapping):
                if name not in paths:mapping.pop(name,None)
        if hasattr(self.thumbnail_delegate,'_scaled_cache'):
            for k in list(self.thumbnail_delegate._scaled_cache):
                if k[0] not in paths:self.thumbnail_delegate._scaled_cache.pop(k,None)
        for mapping in (self._editors,self._draft_models):
            for entry in list(mapping):
                if entry[0] not in keys and entry[0]!=current:mapping.pop(entry,None)
        self._catalog_pending.intersection_update(paths)
        self.request_catalog([p for p in paths if key(p) not in self._catalog])

    def request_catalog(self,paths=None):
        if not self._production_ready or self._closing:return
        queued={self.queue_list.item(i).text() for i in range(self.queue_list.count())}
        self._catalog_pending.update(queued if paths is None else queued.intersection(map(str,paths)))
        if self._catalog_pending and not self._catalog_worker:self._catalog_timer.start()

    def start_catalog(self):
        if self._closing or self._catalog_worker or not self._catalog_pending:return
        paths=tuple(self._catalog_pending);self._catalog_pending.clear()
        need={p for p in paths if p not in self.thumbnail_delegate.cache}
        worker=CatalogWorker(paths,self.store,need_thumbnails=need);worker.setParent(self);worker.found.connect(self.accept_catalog);worker.finished.connect(self.catalog_finished);self._catalog_worker=worker;worker.start()

    def accept_catalog(self,path,state,image):
        if not any(self.queue_list.item(i).text()==path for i in range(self.queue_list.count())):return
        self._catalog[key(path)]=state
        self._shell_stage_states[path]=tuple('stale' if state.get(s,{}).get('stale') else True if state.get(s,{}).get('complete') else 'partial' if state.get(s,{}).get('available') else False for s in ('matte','refined','cutout'))
        if state.get('error'):self._queue_states.setdefault(key(path),{})['missing']=state['error']
        if image:
            rgb=image.convert('RGB');qimage=QImage(rgb.tobytes(),rgb.width,rgb.height,rgb.width*3,QImage.Format_RGB888).copy();self.thumbnail_delegate.cache[path]=QPixmap.fromImage(qimage)
        self.queue_list.viewport().update()

    def catalog_finished(self):
        worker=self._catalog_worker
        if worker:worker.wait();worker.deleteLater()
        self._catalog_worker=None
        if self._closing:self.close()
        else:
            self.sync_shell()
            if self._catalog_pending:self._catalog_timer.start()

    def refresh_results(self):
        if self._production_ready:self.request_catalog()
    def register_result(self,source,folder):
        if self._production_ready:self.store.legacy[key(source)]=Path(folder);self.request_catalog()
    def output_for(self,path):return str(self.store.existing_folder(path,'matte')) if self._production_ready else MatteWindow.output_for(self,path)

    def backend_selector(self):
        name=self.pipeline_picker.currentText().split(' + ')[0].split(' — ')[0] if self.workflow_tabs.currentIndex()==0 else self.manual_selector.currentText()
        return SELECTORS.get(name,'SAM 2 — Tracking')
    def backend_refiner(self):
        name=self.pipeline_picker.currentText().split(' + ')[1] if self.workflow_tabs.currentIndex()==0 and ' + ' in self.pipeline_picker.currentText() else self.manual_refiner.currentText() if self.workflow_tabs.currentIndex()==1 else ''
        return REFINE.get(name)
    def options(self,source,frame=1):
        hints=self.guidance_store().get(source,frame)
        if hints is None:hints=decode(self.store.metadata(source,'matte').get('guidance',[])) if frame==1 else []
        existing=next((h for h in hints if isinstance(h,RotoHint)),RotoHint())
        return tuple([h for h in hints if not isinstance(h,RotoHint)]+[replace(existing,max_size=self.roto_size.value(),chunk_frames=self.roto_chunk.value(),trimap_width=self.trimap_width.value())])

    def preflight(self,jobs):
        missing=[]
        for job in jobs:
            for stage,name in job.stages:
                if stage=='matte' and not name.startswith('BiRefNet'):
                    if name=='SAM 3.1 — Tracking' and not (any(isinstance(h,BoxHint) or isinstance(h,PointHint) and h.positive for h in job.guidance) or any(isinstance(h,RotoHint) and h.text.strip() for h in job.guidance)):
                        self._queue_states.setdefault(key(job.source),{})['missing']='Points, box, or text needed'
                        self.report_error('SAM 3.1 needs a point, box, or text prompt. Its multiplex tracker does not accept saved-mask seeds; place guidance at the resume frame.');return False
                    option=next((h for h in job.guidance if isinstance(h,RotoHint)),RotoHint())
                    valid_seed=bool(option.seed_path and Path(option.seed_path).is_file() and key(option.seed_source)==key(job.source) and option.seed_frame==job.start)
                    valid=any(isinstance(h,BoxHint) or isinstance(h,PointHint) and h.positive for h in job.guidance) or name.startswith('SAM 3') and bool(option.text.strip()) or valid_seed or name.startswith('SAM') and (job.start in self.store.frames(job.source,'matte') or job.start in self.store.frames(job.source,'refined'))
                    if not valid:missing.append(job.source);self._queue_states.setdefault(key(job.source),{})['missing']='Guidance needed'
                if stage=='refined' and not any(s=='matte' for s,n in job.stages) and not (self.store.available_frames(job.source,'matte') or self.store.available_frames(job.source,'refined')):
                    missing.append(job.source);self._queue_states.setdefault(key(job.source),{})['missing']='Mask needed'
                if name in ROTO_MODELS:
                    status=readiness(name,Path(__file__).resolve().parents[1])
                    if not status.startswith('Ready'):self.report_error(status);return False
        if missing:self.queue_list.viewport().update();self.report_error('Guidance or masks are needed for:\n'+'\n'.join(Path(p).name for p in dict.fromkeys(missing)));return False
        return True

    def launch_jobs(self,jobs):
        if self._job_worker or self._active_worker is not None:return
        if not jobs or not self.preflight(jobs):return
        jobs=[replace(job,foreground=self.foreground_estimation.isChecked()) for job in jobs]
        # The preceding stash preserves edits on disk. Do not re-stash those
        # old pixels when a successful live result arrives for this item.
        self._matte_dirty=False
        worker=PipelineWorker(jobs,self.store);worker.setParent(self);self._job_worker=worker;self._job_failed=False;self.queue_progress.setValue(0);self.file_progress.setValue(0);self.follow_action.setChecked(True)
        worker.cutout_ready.connect(self.accept_foreground)
        worker.progress.connect(self.job_progress);worker.result.connect(self.job_result);worker.stage_changed.connect(self.stage_changed);worker.failed.connect(self.report_error);worker.finished.connect(self.job_finished)
        self.sync_shell();worker.start();self._job_tick.start()
    def configured_stages(self):
        refiner=self.backend_refiner()
        return (('matte',self.backend_selector()),)+((('refined',refiner),) if refiner else ())

    def run_selected_pipeline(self,clip=False):
        if not self.current_media_path or self._job_worker:return
        if clip and not self.video_path:return
        if self._requested_frame is not None or isinstance(self._active_worker,PreviewWorker):
            self._queued_pipeline=clip
            if self._active_worker is None:self.navigate_to_frame()
            return
        if self._active_worker is not None:return
        if not self.save_file_guidance():return
        self.canvas.finish_stroke();self.stash_draft()
        start=1 if clip else self.current_frame
        stages=self.configured_stages()
        # Frame tests keep the result as an editable draft until explicitly saved.
        if clip:stages+=(('cutout',''),)
        self.launch_jobs([Job(self.current_media_path,stages,self.options(self.current_media_path,start),
                              start,None if clip else start,not clip,
                              'refined' if self.backend_refiner() else 'matte')])

    def generate(self):self.generate_selected(False)
    def test_current_frame(self):self.generate_selected(True)
    def test_video_frame(self):self.generate_selected(True)
    def generate_selected(self,test=False,start_override=None):
        if not self.current_media_path or self._job_worker:return
        if self._requested_frame is not None:
            self._queued_generate=(test,start_override)
            if self._active_worker is None:self.navigate_to_frame()
            return
        if not self.save_file_guidance():return
        stage='refined' if self.channel()==2 else 'matte';model=self.backend_refiner() if stage=='refined' else self.backend_selector()
        if not model:self.report_error('Select a refiner first.');return
        self.stash_draft()
        start=start_override or self.current_frame if test else start_override or (self.range_start.value() if self.video_path and self.range_picker.currentIndex()==2 else self.current_frame if self.video_path and self.range_picker.currentIndex()==1 else 1)
        end=start if test else self.range_end.value() if self.video_path and self.range_picker.currentIndex()==2 else None
        temporary=test or not self.video_path
        self.launch_jobs([Job(self.current_media_path,((stage,model),),self.options(self.current_media_path,start),start,end,temporary)])
    def resume_video(self):self.generate_selected(False,self.current_frame)
    def run_queue(self):
        if self._job_worker or self._active_worker is not None and not isinstance(self._active_worker,PreviewWorker):self.cancel_jobs();return
        if not self.save_file_guidance():return
        self.stash_draft()
        selector=self.backend_selector();refiner=self.backend_refiner();jobs=[]
        for n in range(self.queue_list.count()):
            source=self.queue_list.item(n).text()
            if self.workflow_tabs.currentIndex()==0:stages=(('matte',selector),)+( (('refined',refiner),) if refiner else () )+(('cutout',''),)
            elif self.channel()==2:
                if not refiner:self.report_error('Choose a refiner.');return
                stages=(('refined',refiner),('cutout',''))
            else:stages=(('matte',selector),('cutout',''))
            jobs.append(Job(source,stages,self.options(source),alpha_stage='refined' if refiner and (self.workflow_tabs.currentIndex()==0 or self.channel()==2) else 'matte'))
        self.launch_jobs(jobs)
    def apply_masks(self):
        if self._job_worker:return
        self.stash_draft();jobs=[];missing=[]
        for n in range(self.queue_list.count()):
            source=self.queue_list.item(n).text();stage=self._cutout_stage.get(key(source),'refined')
            if not self.store.available_frames(source,stage):stage='matte'
            if not self.store.available_frames(source,stage):missing.append(source)
            else:jobs.append(Job(source,(('cutout',''),),(),alpha_stage='chosen'))
        if missing:self.report_error('No mask for:\n'+'\n'.join(Path(p).name for p in missing));return
        self.launch_jobs(jobs)

    def job_progress(self,done,total,source,stage,current,frames):
        self.queue_progress.setValue(round(100*(done+(current/frames if frames else 0))/max(total,1)));self.file_progress.setValue(round(100*current/frames) if frames else 100)
        self.queue_progress.setToolTip(f'Queue: {done}/{total} files');self.file_progress.setToolTip(f'{Path(source).name}: {stage}, {current}/{frames}')
    def stage_changed(self,source,stage,status):
        self._processing_source=source;self._processing_stage=stage;state=self._queue_states.setdefault(key(source),{});state['processing']=status=='running';state['stage']=stage
        if status=='complete':self.request_catalog([source])
        self.queue_list.viewport().update();self.status.setText(f'{Path(source).name}: {stage} {status}')
        self.append_console(f'{Path(source).name}: {stage} {status}\n')
    def job_result(self,source,stage,frame,original,alpha,info,temporary):
        if not temporary:return
        path=self.draft_path(source,stage,frame);path.parent.mkdir(parents=True,exist_ok=True)
        same=key(source)==key(self.current_media_path) and frame==self.current_frame
        if same:
            self.canvas.finish_stroke();self._draft_save_timer.stop();self._paint_refresh.stop()
            if self.current_stage()!=stage:self.stash_draft()
            else:self._matte_dirty=False
        if self._draft_worker and self._draft_worker.path==path:self._draft_worker.cancel()
        self._editors.pop((key(source),frame,stage),None)
        pending=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            alpha.save(pending,'PNG',compress_level=1)
            with self.store.lock:self.store.replace_file(pending,path);self.store.choose_alpha(source,stage,frame)
        finally:pending.unlink(missing_ok=True)
        self._draft_models[(key(source),frame,stage)]=dict(self._job_worker.jobs[0].stages).get(stage,'')
        self._queue_states.setdefault(key(source),{})['draft']=stage
        if same and (self._follow or self.channel() in (1,2)):
            self.original=original.copy();self.original.putalpha(255);self.alpha=alpha.copy();self._matte_dirty=False
            self._loaded_frame_identity=(key(source),frame)
            self._applying_live=True
            try:self.select_shell_tab(self.channel() if self.channel() in (1,2) else 2 if stage=='refined' else 1)
            finally:self._applying_live=False
            self._update_edit_controls();self._update_navigation()
    def job_frame_tick(self):
        if not self._job_worker:return
        try:stage,preview,temporary=self._job_worker.live_frames.get_nowait()
        except queue.Empty:return
        if temporary:return  # The result signal installs image/test drafts once.
        if not self._follow or self._active_worker or self._closing:return
        self.canvas.finish_stroke();self.stash_draft();self.current_media_path=preview.path;self.current_frame=preview.frame;self.video_path=preview.path if preview.info else None;self.video_info=preview.info
        self.original=preview.original.copy();self.original.putalpha(255);self._matte_dirty=False
        self._loaded_frame_identity=(key(preview.path),preview.frame)
        self.restore_file_guidance(preview.path,preview.frame,True);self._applying_live=True
        try:self.select_shell_tab(self.channel() if self.channel() in (1,2) else 1)
        finally:self._applying_live=False
        self.queue_list.blockSignals(True)
        for n in range(self.queue_list.count()):
            if key(self.queue_list.item(n).text())==key(preview.path):self.queue_list.setCurrentRow(n);break
        self.queue_list.blockSignals(False);self._update_edit_controls();self._update_navigation()
    def job_finished(self):
        self.job_frame_tick();worker=self._job_worker;self._job_worker=None;self._job_tick.stop()
        worker.wait();worker.deleteLater()
        for state in self._queue_states.values():state['processing']=False
        if self.current_media_path:self._matte_dirty=self.draft_path(self.current_media_path,self.current_stage(),self.current_frame).is_file()
        self.status.setText('Cancelled. Completed results kept; remaining jobs stopped.' if worker.cancellation.requested else 'Processing failed. Completed results kept; see Console.' if self._job_failed else 'Processing finished.');self._processing_source='';self._processing_stage='';self._cancel_requested=False
        self.request_catalog();self.sync_shell()
        if self._closing:self.close()
    def cancel_jobs(self):
        self._cancel_requested=True
        self._queued_generate=None;self._queued_pipeline=None
        if self._job_worker:self._job_worker.cancel()
        if self._active_worker:self._active_worker.cancel()
        self.status.setText('Stopping safely; remaining queued files will not start.');self.sync_shell()

    def save_frame_edits(self):
        if not self.guard_result_write():return
        if self._job_worker and key(self._processing_source)==key(self.current_media_path) and self._processing_stage==self.current_stage():
            self.stash_draft();self.status.setText('Draft preserved. Save after this stage finishes.');return
        self._draft_save_timer.stop()
        if self._draft_worker:self._draft_worker.cancel()
        self._editors[(key(self.current_media_path),self.current_frame,self.current_stage())]=self.matte_editor
        stage=self.current_stage();model=self._draft_models.get((key(self.current_media_path),self.current_frame,stage),'')
        folder=self.store.save(self.current_media_path,stage,self.current_frame,self.original,self.alpha,model,self.video_info,stale=False,guidance=encode(self.options(self.current_media_path,self.current_frame)))
        self.draft_path(self.current_media_path,stage,self.current_frame).unlink(missing_ok=True);self._matte_dirty=False;self.video_output_dir=str(folder);self.request_catalog([self.current_media_path]);self.sync_shell();self.status.setText(f'Saved {stage}: {folder}')
        self._queue_states.get(key(self.current_media_path),{}).pop('draft',None)
    def export_matte(self):
        if not self.guard_result_write():return
        channel=self.channel();self.select_shell_tab(2 if channel==2 else 1);self.save_frame_edits();self.select_shell_tab(channel)
    def cutout_source(self):
        cached=self._estimated_foreground
        if self.foreground_estimation.isChecked() and cached and cached[:3]==(key(self.current_media_path),self.current_frame,id(self.alpha)):return cached[3]
        return self.original

    def accept_foreground(self,source,frame,foreground,alpha):
        if key(source)==key(self.current_media_path) and frame==self.current_frame and self.alpha is not None and (alpha is self.alpha or (alpha.size==self.alpha.size and alpha.getextrema()==self.alpha.getextrema() and not ImageChops.difference(alpha,self.alpha).getbbox())):
            self._estimated_foreground=(key(source),frame,id(self.alpha),foreground)
            if self.channel()==2:self._refresh_canvas(False)

    def export_cutout(self):
        if not self.guard_result_write():return
        if self.foreground_estimation.isChecked():
            if self._job_worker:return
            self.canvas.finish_stroke();self.stash_draft()
            self.launch_jobs([Job(self.current_media_path,(('cutout',''),),(),self.current_frame,self.current_frame,
                                  alpha_stage=self.current_stage(),foreground=True,snapshot=(self.original.copy(),self.alpha.copy()))])
            return
        stage=self.current_stage();self.store.save(self.current_media_path,'cutout',self.current_frame,self.original,self.alpha,info=self.video_info,alpha_stage=stage,stale=False,foreground_estimated=False)
        self.request_catalog([self.current_media_path]);self.status.setText('Cutout saved under '+str(self.store.folder(self.current_media_path,'cutout')))
    def start_playback(self,frame=1):
        if self._active_worker or not self.video_path or not self.video_info or not self.video_output_dir or not self.can_write_frame():return
        self.canvas.finish_stroke();self.stash_draft();self.select_shell_tab(2)
        available=self.store.frames(self.current_media_path,self.current_stage());folder=self.video_output_dir;fallback=None
        if self.foreground_estimation.isChecked():
            cutouts=self.store.frames(self.current_media_path,'cutout')
            if cutouts:fallback=folder;folder=str(self.store.existing_folder(self.current_media_path,'cutout'));available=sorted(set(available)|set(cutouts))
        if available and frame==1:frame=available[0]
        last=min(self.video_info.frame_count,max(available)) if available else self.video_info.frame_count
        worker=PlaybackWorker(folder,frame,max(frame,last),fallback);worker.failed.connect(self._playback_failed)
        self._playback_displayed=False;self._start_worker(worker)
        self._playback_timer.start(max(1,round(1000/(self.video_info.fps or 24))))

    def report_error(self,details):
        if self._job_worker:self._job_failed=True
        if self._processing_source and 'mask' in str(details).lower():self._queue_states.setdefault(key(self._processing_source),{})['missing']=str(details).splitlines()[-1]
        self.append_console(str(details)+'\n');self.status.setText(str(details).splitlines()[-1]);self.file_tabs.setCurrentIndex(1)
        QMessageBox.warning(self,'Matte App',str(details).splitlines()[-1])
    def on_matte_failed(self,details):self.report_error(details)
    def preview_notice(self,*args):pass

    def choose_mask_folder(self,index):
        if index!=1:return
        folder=QFileDialog.getExistingDirectory(self,'Attach saved matte frames')
        if folder and self.current_media_path:self.store.legacy[key(self.current_media_path)]=Path(folder);self.request_catalog();self._load_media(self.current_media_path)
        self.manual_masks.setCurrentIndex(0)
    def choose_project_root(self):
        if self._job_worker:return
        folder=QFileDialog.getExistingDirectory(self,'Project root containing the source folders')
        if folder:
            candidate=Path(folder).resolve()
            try:
                for i in range(self.queue_list.count()):Path(self.queue_list.item(i).text()).resolve().relative_to(candidate)
            except ValueError:self.report_error('Choose a project root containing all queued files.');return
            self.store.root=candidate;self.request_catalog()
    def clear_queue(self):
        if self._job_worker or self._active_worker:return
        self.canvas.finish_stroke();self.stash_draft();MatteWindow.clear_queue(self)

    def remove_queue_item(self):
        if not self._job_worker:MatteWindow.remove_queue_item(self)
    def get_sam3_weights(self):return self.open_installer(focus_model='sam3')
    def load_results(self):
        if self._job_worker:return
        folder=QFileDialog.getExistingDirectory(self,'Choose matted folder or an individual saved result')
        if not folder:return
        try:
            from .result_import import discover_results
            queued=[self.queue_list.item(i).text() for i in range(self.queue_list.count())]
            item=self.queue_list.currentItem()
            links,errors=discover_results(folder,queued,item.text() if item else self.current_media_path,self.queue_outputs)
            if not links:raise ValueError('No matching saved results found. Add the original files to Files, then choose the matted folder. Each image folder must contain numbered PNG results.'+ ('\n\n'+'\n'.join(errors[:5]) if errors else ''))
            from PySide6.QtWidgets import QListWidgetItem
            self.queue_list.blockSignals(True)
            try:
                for source,destination in links.items():
                    if source not in queued:
                        entry=QListWidgetItem(source);entry.setToolTip(source);self.queue_list.addItem(entry);queued.append(source)
                    self.queue_outputs[source]=destination;self.store.legacy[key(source)]=Path(destination)
            finally:self.queue_list.blockSignals(False)
            self.store.write_json(self.results_index_path,self.queue_outputs);self.request_catalog()
            source=self.current_media_path if self.current_media_path in links else next(iter(links))
            self._load_media(source,restore_guidance=True)
            self.status.setText(f'Loaded saved results for {len(links)} file(s).')
            if errors:self.append_console('Skipped result folders:\n'+'\n'.join(errors)+'\n')
        except (OSError,ValueError,RuntimeError) as error:QMessageBox.warning(self,'Could not load results',str(error))
    def remember_output(self,path,folder):
        self.queue_outputs[str(Path(path).resolve())]=folder;self.store.legacy[key(path)]=Path(folder);self.store.write_json(self.results_index_path,self.queue_outputs);self.request_catalog()
    def save_project(self):
        self.stash_draft();self.save_file_guidance()
        filename,_=QFileDialog.getSaveFileName(self,'Save project','','Matte project (*.matte.json)')
        if not filename:return
        data={'format':'matte-project-2','clips':[self.queue_list.item(i).text() for i in range(self.queue_list.count())],'root':str(self.store.root) if self.store.root else None,'outputs':self.queue_outputs,'current':self.current_media_path,'frame':self.current_frame,'channel':self.channel(),'workflow':self.workflow_tabs.currentIndex(),'pipeline':self.pipeline_picker.currentText(),'selector':self.manual_selector.currentText(),'refiner':self.manual_refiner.currentText(),'size':self.roto_size.value(),'chunk':self.roto_chunk.value(),'edge':self.trimap_width.value(),'cutout_sources':self._cutout_stage,'guidance_by_file':self.guidance_store().read(),'tool':self._active_tool,'foreground_estimation':self.foreground_estimation.isChecked()}
        self.store.write_json(Path(filename),data);self.status.setText('Project saved. Draft edits remain recoverable on disk.')
    def open_project(self):
        if self._job_worker:return
        filename,_=QFileDialog.getOpenFileName(self,'Open project','','Matte project (*.matte.json)')
        if not filename:return
        try:
            data=json.loads(Path(filename).read_text(encoding='utf-8'))
            if data.get('format') not in ('matte-project-1','matte-project-2'):raise ValueError('Not a Matte App project')
            self.stash_draft();self.save_file_guidance();self._opening_project=True;self.queue_list.blockSignals(True);self.queue_list.clear();self.queue_list.addItems(data.get('clips',[]));self.queue_list.blockSignals(False)
            self.queue_outputs.update(data.get('outputs',{}));self.store.legacy.update({key(p):Path(f) for p,f in self.queue_outputs.items()});self.guidance_store().merge(data.get('guidance_by_file',{}))
            self.store.root=Path(data['root']) if data.get('root') else None;self._cutout_stage=data.get('cutout_sources',{})
            self.pipeline_picker.setCurrentText(data.get('pipeline','SAM 2 + MatAnyone'));self.manual_selector.setCurrentText(data.get('selector','SAM 2'));self.manual_refiner.setCurrentText(data.get('refiner','MatAnyone'));self.roto_size.setValue(data.get('size',512));self.roto_chunk.setValue(data.get('chunk',4));self.trimap_width.setValue(data.get('edge',24));self.workflow_tabs.setCurrentIndex(data.get('workflow',0))
            self.foreground_estimation.setChecked(data.get('foreground_estimation',False))
            self._pending_channel=data.get('channel',data.get('view',0));self.choose_tool(data.get('tool','box'))
            if data.get('current'):self._load_media(data['current'],frame=data.get('frame',1))
            self.request_catalog()
        except Exception as error:self._opening_project=False;self.queue_list.blockSignals(False);self.report_error(str(error))

    def closeEvent(self,event):
        if not self._production_ready:return MatteWindow.closeEvent(self,event)
        self.canvas.finish_stroke();self.stash_draft();self.save_file_guidance()
        self._draft_save_timer.stop();self._opacity_refresh.stop();self._zoom_refresh.stop();self._catalog_timer.stop()
        if self._job_worker or self._active_worker or self._catalog_worker or self._draft_worker:
            self._closing=True;self.cancel_jobs()
            if self._catalog_worker:self._catalog_worker.cancel()
            if self._draft_worker:self._draft_worker.cancel()
            event.ignore();return
        self._job_tick.stop();self._paint_refresh.stop();self._navigation_timer.stop();self._playback_timer.stop();event.accept();QApplication.instance().removeEventFilter(self)
        from .model_session import stop_session
        stop_session()
        from .foreground import cleanup_foreground
        cleanup_foreground()
        from .models import _MODEL_CACHE
        _MODEL_CACHE.clear()
        from . import roto
        roto.ENABLE_WARM_SESSIONS=self._previous_warm
