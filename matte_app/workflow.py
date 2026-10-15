"""Task-first controls, saved result versions, and explicit refinement inputs."""
from pathlib import Path
import json
import uuid
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QToolButton, QSpinBox, QFileDialog, QMessageBox, QScrollArea, QFrame, QLayout, QSizePolicy
from .roto import REFINERS, ROTO_MODELS
from .saved_masks import mask_files, validate_trimap
from .video import read_metadata, write_metadata, save_frame

class WorkflowMixin:
    def build_workflow(self, main, actions_layout, sidebar):
        self._mask_input_folder = ''
        self._trimap_path = ''
        self._trimap_source = ''
        self._trimap_frame = 1
        self._matte_dirty = False
        self._image_save_target = None
        self._image_generation_model = ''
        self._image_generation_guidance = []
        self._image_refine_parent = ''
        self.workflow_card = QWidget()
        self.workflow_card.setObjectName('toolPanel')
        layout = QVBoxLayout(self.workflow_card)
        layout.setContentsMargins(16,12,16,12)
        self.stage_picker = QComboBox()
        self.stage_picker.addItems(['1. Create masks','2. Refine edges','Automatic'])
        self.stage_picker.setToolTip('Processing stage. Original / Matte / Cutout choose the image view separately.')
        self.result_picker = QComboBox()
        self.result_picker.setMinimumWidth(220)
        self.result_picker.setToolTip('Choose a saved result to review. Refinement preserves its input masks.')
        stage_row = QVBoxLayout()
        stage_row.addWidget(QLabel('Workflow'))
        stage_row.addWidget(self.stage_picker)
        result_row = QHBoxLayout()
        result_row.addWidget(QLabel('Result'))
        result_row.addWidget(self.result_picker,1)
        main.insertLayout(0,result_row)
        layout.addLayout(stage_row)
        self.model_label = QLabel('Selection model')
        model_row = QVBoxLayout()
        model_row.addWidget(self.model_label)
        model_row.addWidget(self.model_picker)
        layout.addLayout(model_row)
        layout.addWidget(self.roto_status)
        layout.addWidget(self.get_sam3_button)
        self.refine_controls = QWidget()
        self.refine_controls.setObjectName('refineControls')
        refine_layout = QVBoxLayout(self.refine_controls)
        refine_layout.setContentsMargins(0,0,0,0)
        input_row = QVBoxLayout()
        self.mask_input_picker = QComboBox()
        self.mask_input_picker.setMinimumWidth(220)
        self.attach_masks_button = QPushButton('Attach saved masks…')
        self.attach_masks_button.clicked.connect(self.attach_refinement_masks)
        input_row.addWidget(QLabel('Input masks'))
        input_row.addWidget(self.mask_input_picker,1)
        input_row.addWidget(self.attach_masks_button)
        refine_layout.addLayout(input_row)
        self.refine_note = QLabel('')
        self.refine_note.setWordWrap(True)
        self.refine_note.setObjectName('hint')
        refine_layout.addWidget(self.refine_note)
        layout.addWidget(self.refine_controls)
        self.advanced_toggle = QToolButton(text='Advanced settings')
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setCheckable(True)
        self.advanced_panel = QWidget()
        advanced = QVBoxLayout(self.advanced_panel)
        advanced.setContentsMargins(0,0,0,0)
        # Technical settings are no longer mixed into the guidance toolbar.
        advanced.addWidget(self.settings_scroll(self.roto_panel))
        self.trimap_controls = QWidget()
        trimap_row = QHBoxLayout(self.trimap_controls)
        trimap_row.setContentsMargins(0,0,0,0)
        self.trimap_width = QSpinBox()
        self.trimap_width.setRange(1,256)
        self.trimap_width.setValue(24)
        self.trimap_width.setSuffix(' px')
        self.trimap_width.setToolTip('Unknown band on either side of mask edges, in source-image pixels.')
        trimap_row.addWidget(QLabel('Unknown edge band'))
        trimap_row.addWidget(self.trimap_width)
        self.load_trimap_button = QPushButton('Load trimap…')
        self.load_trimap_button.clicked.connect(self.load_trimap)
        self.clear_trimap_button = QPushButton('Clear trimap')
        self.clear_trimap_button.clicked.connect(self.clear_trimap)
        trimap_row.addWidget(self.load_trimap_button)
        trimap_row.addWidget(self.clear_trimap_button)
        self.trimap_scroll = self.settings_scroll(self.trimap_controls)
        advanced.addWidget(self.trimap_scroll)
        self.trimap_note = QLabel('ViTMatte calculates gray/unknown pixels; black and white regions stay fixed. A loaded trimap applies only to its selected frame.')
        self.trimap_note.setWordWrap(True)
        self.trimap_note.setObjectName('hint')
        advanced.addWidget(self.trimap_note)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(self.advanced_panel)
        self.advanced_panel.hide()
        self.advanced_toggle.toggled.connect(self.toggle_advanced)
        sidebar.insertWidget(3,self.workflow_card)
        self.save_edits_button = QPushButton('Save frame edits')
        self.save_edits_button.clicked.connect(self.save_frame_edits)
        self.edit_state = QLabel('')
        self.edit_state.setObjectName('hint')
        actions_layout.addWidget(self.edit_state)
        range_row = QHBoxLayout()
        self.range_picker = QComboBox()
        self.range_picker.setMaximumWidth(260)
        self.range_picker.addItems(['Whole clip','From displayed frame','Selected range'])
        self.range_start = QSpinBox()
        self.range_end = QSpinBox()
        for control in (self.range_start,self.range_end):
            control.setRange(1,1)
            control.setMinimumWidth(100)
        label = QLabel('Process')
        label.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Preferred)
        range_row.addWidget(label)
        range_row.addWidget(self.range_picker)
        range_row.addWidget(self.range_start)
        self.range_to = QLabel('to')
        range_row.addWidget(self.range_to)
        range_row.addWidget(self.range_end)
        range_row.addStretch()
        self.range_panel = QWidget()
        self.range_panel.setObjectName('rangeControls')
        self.range_panel.setLayout(range_row)
        actions_layout.insertWidget(0,self.range_panel)
        self.range_picker.currentIndexChanged.connect(lambda _: self.update_workflow())
        self.stage_picker.currentIndexChanged.connect(self.choose_stage)
        self.model_picker.currentTextChanged.connect(self.sync_stage_from_model)
        self.result_picker.currentIndexChanged.connect(self.select_result)
        self.mask_input_picker.currentIndexChanged.connect(self.select_mask_input)
        self._build_menus()
        self.model_picker.setCurrentText('SAM 2 — Tracking')
        self.sync_stage_from_model()

    def settings_scroll(self, widget):
        widget.layout().setSizeConstraint(QLayout.SetMinimumSize)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(0)
        scroll.setFixedHeight(70)
        scroll.setWidget(widget)
        return scroll

    def toggle_advanced(self, checked):
        self.advanced_panel.setVisible(checked)
        self.advanced_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _build_menus(self):
        self.file_actions = {}
        file_menu = self.menuBar().addMenu('File')
        for label, callback, shortcut in [('Open project…',self.open_project,'Ctrl+O'),('Save project…',self.save_project,'Ctrl+S'),('Add media…',self.add_queue_files,'Ctrl+Shift+O'),('Attach saved results…',self.load_results,None),('Save frame edits',self.save_frame_edits,None),('Export matte…',self.export_matte,None),('Export cutout…',self.export_cutout,None)]:
            action = QAction(label,self)
            action.triggered.connect(callback)
            if shortcut: action.setShortcut(QKeySequence(shortcut))
            file_menu.addAction(action)
            self.file_actions[label] = action
        self.file_installer_action = file_menu.addAction('Install models & dependencies…', self.open_installer)
        edit = self.menuBar().addMenu('Edit')
        edit.addAction(self.undo_brush_action)
        edit.addAction(self.redo_brush_action)
        self.remove_guidance_action = edit.addAction('Remove selected guidance',self.remove_selected_guidance)
        self.clear_guidance_action = edit.addAction('Clear guidance',self.clear_guidance)
        view = self.menuBar().addMenu('View')
        view.addAction('Fit preview',self.canvas.reset_zoom)
        view.addAction('Zoom in',lambda: self.canvas.set_zoom(self.canvas._zoom*1.25))
        view.addAction('Zoom out',lambda: self.canvas.set_zoom(self.canvas._zoom/1.25))
        view.addAction('Reset panel sizes',lambda: self.splitter.setSizes([340,max(330,self.width()-380)]))
        help_menu = self.menuBar().addMenu('Help')
        help_menu.addAction('Workflow guide',lambda: QMessageBox.information(self,'Two-stage matting','Create masks with SAM, review and save corrections, then choose Refine edges.\n\nMatAnyone uses the starting mask and tracks independently. VideoMaMa uses masks across the range. ViTMatte refines individual frames from a trimap.\n\nRefinements are saved separately. Painting edits the displayed frame only; use Save frame edits to keep them.'))
        self.help_installer_action = help_menu.addAction('Install models & dependencies…', self.open_installer)

    def open_installer(self, focus_model=None):
        from .installer_dialog import ModelInstallerDialog
        if not hasattr(self, '_installer_dialog') or self._installer_dialog is None:
            self._installer_dialog = ModelInstallerDialog(self, focus_model=focus_model)
        else:
            if focus_model:
                self._installer_dialog.select_focus_model(focus_model)
        self._installer_dialog.show()
        self._installer_dialog.raise_()
        self._installer_dialog.activateWindow()
        return self._installer_dialog

    def choose_stage(self, index):
        if self._active_worker is not None: return
        if index == 1:
            self._mask_input_folder = self.video_output_dir or ''
            self.model_picker.setCurrentText(REFINERS[0])
        elif index == 0:
            self.model_picker.setCurrentText('SAM 2 — Tracking')
        else:
            self.model_picker.setCurrentText('BiRefNet HR Matting')
        self.refresh_results()
        self.update_workflow()

    def sync_stage_from_model(self, *_):
        if not hasattr(self,'stage_picker'): return
        name = self.model_picker.currentText()
        stage = 1 if name in REFINERS else 0 if name.endswith('— Tracking') else 2
        self.stage_picker.blockSignals(True)
        self.stage_picker.setCurrentIndex(stage)
        self.stage_picker.blockSignals(False)
        for i in range(self.model_picker.count()):
            item = self.model_picker.itemText(i)
            item_stage = 1 if item in REFINERS else 0 if item.endswith('— Tracking') else 2
            self.model_picker.view().setRowHidden(i,item_stage != stage)
        self.model_label.setText(['Selection model','Refinement model','Automatic workflow'][stage])
        self.update_workflow()
        self._refresh_canvas()

    def update_workflow(self):
        if not hasattr(self,'stage_picker'): return
        refining = self.model_picker.currentText() in REFINERS
        busy = self._active_worker is not None
        self.stage_picker.setEnabled(not busy)
        self.result_picker.setEnabled(not busy)
        self.mask_input_picker.setEnabled(not busy)
        self.attach_masks_button.setEnabled(not busy and self.original is not None)
        self.load_trimap_button.setEnabled(not busy and self.original is not None)
        self.clear_trimap_button.setEnabled(not busy and bool(self._trimap_path))
        self.refine_controls.setVisible(refining)
        name = self.model_picker.currentText()
        self.trimap_controls.setVisible('ViTMatte' in name)
        self.trimap_scroll.setVisible('ViTMatte' in name)
        self.trimap_note.setVisible('ViTMatte' in name)
        self.advanced_toggle.setVisible(name in ROTO_MODELS)
        self.range_panel.setVisible(bool(self.video_path))
        self.range_picker.setEnabled(not busy)
        self.range_to.setVisible(self.range_picker.currentIndex()==2)
        total = self.video_info.frame_count if self.video_info else 1
        for spin in (self.range_start,self.range_end):
            oldmax = spin.maximum()
            spin.setMaximum(total)
            if spin is self.range_end and oldmax == 1: spin.setValue(total)
            spin.setEnabled(not busy and self.range_picker.currentIndex()==2)
            spin.setVisible(self.range_picker.currentIndex()==2)
        self.generate_button.setText('Refine edges' if refining else 'Create masks' if name.endswith('— Tracking') else 'Generate matte')
        self.resume_button.setText('Refine from this frame' if refining else 'Update masks from here' if name.endswith('— Tracking') else 'Resume from this frame')
        self.test_frame_button.setText('Test refinement' if refining else 'Test selected frame')
        self.resume_hint.setText('Input masks are preserved. Each refinement creates a separate result.' if refining else 'From this frame recalculates the displayed frame and later results. Earlier frames are kept.')
        self.queue_run_button.setEnabled(not busy and not refining)
        self.save_edits_button.setText('Save frame edits' if self.video_path else 'Save to image')
        self.save_edits_button.setToolTip('Save the current matte/cutout to this image’s linked result. The source image stays unchanged.' if not self.video_path else 'Save painting changes to the displayed video frame.')
        self.save_edits_button.setMinimumWidth(self.save_edits_button.fontMetrics().horizontalAdvance(self.save_edits_button.text())+32)
        self.save_edits_button.setEnabled(not busy and self.alpha is not None and bool(self.current_media_path) and (self._matte_dirty or not self.video_path))
        self.advanced_toggle.setEnabled(not busy)
        for label,action in getattr(self,'file_actions',{}).items():
            action.setEnabled(not busy and (self.alpha is not None if label.startswith('Export') else True))
        if 'Save frame edits' in getattr(self,'file_actions',{}):
            self.file_actions['Save frame edits'].setText('Save frame edits' if self.video_path else 'Save to image')
            self.file_actions['Save frame edits'].setEnabled(self.save_edits_button.isEnabled())
        if hasattr(self,'clear_guidance_action'):
            self.clear_guidance_action.setEnabled(not busy and self.current_view()=='original' and bool(self.points or self.boxes))
            self.remove_guidance_action.setEnabled(self.clear_guidance_action.isEnabled())
        if not self.video_path:
            self.test_frame_button.setEnabled(not busy and self.original is not None)
        self.edit_state.setText(('Unsaved frame edits' if self.video_path else 'Unsaved image matte · Save to image when finished') if self._matte_dirty else 'Temporary preview' if self.video_test_active else 'Saved result' if self.video_output_dir else '')
        notes = {'ViTMatte — Saved masks':'Uses each saved mask to create a trimap. Refines frames independently; check motion for flicker.', 'MatAnyone — Saved masks':'Uses the starting-frame mask and follows the subject itself. Later saved masks are not used; designed primarily for people.', 'VideoMaMa — Saved masks':'Uses the saved masks throughout the selected range. SAM will not run again.'}
        self.refine_note.setText(notes.get(name,''))
        if hasattr(self,'model_picker'):
            for button in (self.generate_button,self.resume_button,self.test_frame_button):
                button.setMinimumWidth(button.fontMetrics().horizontalAdvance(button.text())+32)

    def _library(self):
        path = self.results_index_path.with_name('result_versions.json')
        try: return json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError): return {}

    def _save_library(self, library):
        path = self.results_index_path.with_name('result_versions.json')
        pending = path.with_suffix('.json.tmp')
        pending.write_text(json.dumps(library,indent=2),encoding='utf-8')
        pending.replace(path)

    def register_result(self, source, folder):
        library = self._library()
        entries = library.setdefault(str(Path(source).resolve()),[])
        if not any(Path(e['folder']).resolve()==Path(folder).resolve() for e in entries):
            metadata = read_metadata(Path(folder))
            model = metadata.get('model',self.model_picker.currentText())
            stage = 'masks' if model.endswith('— Tracking') else 'refined'
            entries.append({'folder':folder,'model':model,'stage':stage,'parent':metadata.get('source_masks',self._mask_input_folder if model in REFINERS else '')})
            self._save_library(library)
        else:
            metadata = read_metadata(Path(folder))
            if metadata.get('model'):
                entry=next(e for e in entries if Path(e['folder']).resolve()==Path(folder).resolve())
                entry.update({'model':metadata['model'],'stage':'masks' if metadata['model'].endswith('— Tracking') else 'refined','parent':metadata.get('source_masks','')})
                self._save_library(library)
        self.refresh_results()

    def refresh_results(self):
        if not hasattr(self,'result_picker'): return
        source = self.current_media_path or self.video_path
        entries = self._library().get(str(Path(source).resolve()),[]) if source else []
        for picker in (self.result_picker,self.mask_input_picker):
            picker.blockSignals(True)
            picker.clear()
            picker.addItem('Current preview' if picker is self.result_picker else 'Displayed frame mask', '')
            counts = {'masks':0,'refined':0}
            for entry in entries:
                if Path(entry['folder']).is_dir():
                    counts[entry['stage']] += 1
                    model = entry['model'].replace(' — Saved masks','').replace(' — Tracking','')
                    label = ('Masks' if entry['stage']=='masks' else 'Matte')+f' v{counts[entry["stage"]]} · '+model+(' · Needs update' if entry.get('stale') else '')
                    picker.addItem(label,entry['folder'])
                    picker.setItemData(picker.count()-1,entry['folder'],Qt.ToolTipRole)
            target = self.video_output_dir if picker is self.result_picker else self._mask_input_folder
            found = picker.findData(target)
            if found>=0: picker.setCurrentIndex(found)
            picker.blockSignals(False)

    def select_result(self, index):
        if self._active_worker is not None: return
        folder = self.result_picker.itemData(index)
        if folder and self.current_media_path:
            self._pending_view_after_load = self.view_tabs.currentIndex()
            self._load_media(self.current_media_path,folder,self.current_frame,self.video_info)

    def select_mask_input(self, index):
        self._mask_input_folder = self.mask_input_picker.itemData(index) or ''
        self.update_workflow()

    def attach_refinement_masks(self):
        if self._active_worker is not None or not self.current_media_path: return
        selected = QFileDialog.getExistingDirectory(self,'Choose saved masks for this clip')
        if not selected: return
        try:
            mask_files(selected,self.current_media_path,self.current_frame,1)
            self._mask_input_folder = selected
            self.register_result(self.current_media_path,selected)
            self.refresh_results()
        except Exception as error: QMessageBox.information(self,'Could not attach masks',str(error))

    def load_trimap(self):
        if self._active_worker is not None or self.original is None: return
        path,_ = QFileDialog.getOpenFileName(self,'Choose a 0 / 128 / 255 trimap','','PNG (*.png)')
        if not path: return
        try:
            with Image.open(path) as image: validate_trimap(image,self.original.size)
            self._trimap_path, self._trimap_source, self._trimap_frame = path,self.current_media_path,self.current_frame
            self.status.setText(f'Trimap loaded for frame {self.current_frame}. Other frames use generated trimaps.')
        except Exception as error: QMessageBox.information(self,'Invalid trimap',str(error))

    def clear_trimap(self):
        self._trimap_path = self._trimap_source = ''

    def prepare_refinement(self, start, count=1):
        if self.model_picker.currentText() not in REFINERS: return True
        try:
            if self._matte_dirty and self._mask_input_folder == self.video_output_dir:
                raise ValueError('Save frame edits before refining these saved masks.')
            if self._mask_input_folder:
                mask_files(self._mask_input_folder,self.current_media_path,start,count,self.model_picker.currentText().startswith('MatAnyone'))
            elif self.alpha is not None and (count==1 or self.model_picker.currentText().startswith('MatAnyone')) and start==self.current_frame:
                self._save_roto_seed(self.alpha)
            else: raise ValueError('Attach saved masks for this range first. A displayed mask can seed MatAnyone, or refine one frame.')
            return True
        except Exception as error:
            QMessageBox.information(self,'Refinement inputs',str(error))
            return False

    def new_refinement_folder(self, parent):
        root = Path(parent).parent if parent else Path(self.current_media_path).parent
        return str(root/(Path(self.current_media_path).stem+'_refined_'+uuid.uuid4().hex[:8]))

    def save_frame_edits(self):
        if self._active_worker is not None or self.alpha is None or not self.current_media_path: return
        folder = self.video_output_dir if self.video_path else self._image_save_target or self.video_output_dir
        if not folder:
            folder = QFileDialog.getExistingDirectory(self,'Choose folder for saved masks')
        if not folder: return
        image = self.original.convert('RGBA')
        image.putalpha(self.alpha)
        metadata = read_metadata(Path(folder))
        bound = metadata.get('source_video') or metadata.get('source_image')
        if bound and Path(bound).resolve()!=Path(self.current_media_path).resolve():
            QMessageBox.information(self,'Different image result','This folder belongs to another source. Choose a separate folder for this image.')
            return
        Path(folder).mkdir(parents=True,exist_ok=True)
        save_frame(image,Path(folder)/f'frame_{self.current_frame:06d}.png')
        metadata.update({'source_video' if self.video_path else 'source_image':self.current_media_path,'frame_count':self.video_info.frame_count if self.video_info else 1})
        metadata.setdefault('model',self.model_picker.currentText())
        if not self.video_path and self._image_generation_model:
            metadata.update({'model':self._image_generation_model,'guidance':self._image_generation_guidance,'status':'complete'})
            if self._image_refine_parent:metadata['source_masks']=self._image_refine_parent
            else:metadata.pop('source_masks',None)
        metadata['revision'] = metadata.get('revision',0)+1
        write_metadata(Path(folder),metadata)
        self.video_output_dir = folder
        self.remember_output(self.current_media_path,folder)
        library = self._library()
        for entry in library.get(str(Path(self.current_media_path).resolve()),[]):
            if entry.get('parent') and Path(entry['parent']).resolve()==Path(folder).resolve(): entry['stale']=True
        self._save_library(library)
        self._matte_dirty = False
        self._image_save_target = None
        self._image_generation_model = ''
        self.video_test_active = False
        self.refresh_results()
        self.update_workflow()
        self.status.setText(f'Saved edits to frame {self.current_frame}. Other frames are unchanged.' if self.video_path else f'Saved matte for {Path(self.current_media_path).name}. Its saved cutout is updated.')

    def prepare_image_save(self):
        self._matte_dirty = True
        self._image_generation_model = self.model_picker.currentText()
        self._image_generation_guidance = [h.__dict__ | {'type':type(h).__name__} for h in self.job_guidance()]
        self._image_refine_parent = self._mask_input_folder if self._image_generation_model in REFINERS else ''
        self._image_save_target = self.video_output_dir
        if self._image_refine_parent:
            metadata = read_metadata(Path(self.video_output_dir)) if self.video_output_dir else {}
            if not self.video_output_dir or Path(self.video_output_dir).resolve()==Path(self._image_refine_parent).resolve() or metadata.get('source_masks')!=self._image_refine_parent:
                self._image_save_target = self.new_refinement_folder(self._image_refine_parent)

    def clear_image_save_state(self):
        self._image_save_target = None
        self._image_generation_model = ''
        self._image_generation_guidance = []
        self._image_refine_parent = ''

    def guard_frame_edits(self):
        if not self._matte_dirty or not self.current_media_path: return True
        answer = QMessageBox.warning(self,'Unsaved edits','Save this frame’s paint edits before leaving it?' if self.video_path else 'Save this image’s current matte before leaving it?',QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,QMessageBox.Save)
        if answer == QMessageBox.Cancel: return False
        if answer == QMessageBox.Save:
            self.save_frame_edits()
            return not self._matte_dirty
        self._matte_dirty = False
        return True

    def save_project(self):
        if self._active_worker is not None: return
        if not self.save_file_guidance(): return
        path,_ = QFileDialog.getSaveFileName(self,'Save Matte project','project.matte.json','Matte project (*.matte.json)')
        if path:
            document = {'format':'matte-project-1','clips':[self.queue_list.item(i).text() for i in range(self.queue_list.count())], 'outputs':self.queue_outputs,'results':self._library(),'current':self.current_media_path,'frame':self.current_frame,'folder':self.video_output_dir,'model':self.model_picker.currentText(),'mask_input':self._mask_input_folder,
                'view':self.view_tabs.currentIndex(),'zoom':self.canvas._zoom,'range':[self.range_picker.currentIndex(),self.range_start.value(),self.range_end.value()],
                'guidance':[h.__dict__ | {'type':type(h).__name__} for h in self.job_guidance()]}
            if self._matte_dirty:
                self.save_frame_edits()
                if self._matte_dirty: return
                document['outputs'],document['results'] = self.queue_outputs,self._library()
                document['folder'] = self.video_output_dir
            document['guidance_by_file'] = self.guidance_store().read()
            Path(path).write_text(json.dumps(document,indent=2),encoding='utf-8')
            self.status.setText('Project saved. Media and result files remain in their existing folders.')

    def open_project(self):
        if self._active_worker is not None: return
        if not self.guard_frame_edits(): return
        path,_ = QFileDialog.getOpenFileName(self,'Open Matte project','','Matte project (*.matte.json)')
        if not path: return
        try:
            document = json.loads(Path(path).read_text(encoding='utf-8'))
            if document.get('format')!='matte-project-1': raise ValueError('Not a Matte App project.')
            self.queue_list.blockSignals(True)
            self.queue_list.clear()
            self.queue_list.addItems(document['clips'])
            self.queue_list.blockSignals(False)
            self.queue_outputs.update(document.get('outputs',{}))
            self.guidance_store().merge(document.get('guidance_by_file',{}))
            self.results_index_path.write_text(json.dumps(self.queue_outputs,indent=2),encoding='utf-8')
            library = self._library()
            library.update(document.get('results',{}))
            self._save_library(library)
            self._mask_input_folder = document.get('mask_input','')
            self.model_picker.setCurrentText(document.get('model','SAM 2 — Tracking'))
            self._pending_project_state = document
            source = document.get('current')
            if source: self._load_media(source,document.get('folder'),document.get('frame',1))
        except Exception as error:
            self.queue_list.blockSignals(False)
            QMessageBox.information(self,'Could not open project',str(error))

    def restore_project_ui(self):
        document = getattr(self,'_pending_project_state',None)
        if not document: return
        self._pending_project_state = None
        from .models import PointHint,BoxHint
        self.points.clear()
        self.boxes.clear()
        for hint in document.get('guidance',[]):
            if hint['type']=='PointHint': self.points.append(PointHint(hint['x'],hint['y'],hint.get('positive',True)))
            elif hint['type']=='BoxHint': self.boxes.append(BoxHint(hint['left'],hint['top'],hint['right'],hint['bottom']))
            elif hint['type']=='RotoHint':
                self.roto_text.setText(hint.get('text',''))
                self.roto_size.setValue(hint.get('max_size',512))
                self.roto_chunk.setValue(hint.get('chunk_frames',4))
                self.trimap_width.setValue(hint.get('trimap_width',24))
                self.seed_path = hint.get('seed_path','') if Path(hint.get('seed_path','')).is_file() else ''
                self.seed_source,self.seed_frame = hint.get('seed_source',''),hint.get('seed_frame',1)
                self._trimap_path = hint.get('trimap_path','') if Path(hint.get('trimap_path','')).is_file() else ''
                self._trimap_source,self._trimap_frame = document.get('current'),hint.get('trimap_frame',1)
        mode,start,end = document.get('range',[0,1,1])
        self.range_picker.setCurrentIndex(mode)
        self.range_start.setValue(start)
        self.range_end.setValue(end)
        self.view_tabs.setCurrentIndex(document.get('view',0))
        self.canvas.set_zoom(document.get('zoom',1))
        self._refresh_canvas()
