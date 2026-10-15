"""Create only controls used by the production shell, without the legacy layout."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction,QKeySequence
from PySide6.QtWidgets import QWidget,QPushButton,QToolButton,QComboBox,QLabel,QListWidget,QSpinBox,QSlider,QTabBar,QLineEdit,QFrame,QVBoxLayout,QHBoxLayout
from .roto import ROTO_MODELS

def build_controls(w):
    root=QWidget();w.setCentralWidget(root)
    def button(name,text,callback=None,tool=False):
        value=QToolButton(root) if tool else QPushButton(root);value.setText(text)
        if callback:value.clicked.connect(callback)
        setattr(w,name,value);return value
    def spin(name,low,high,value,suffix=''):
        control=QSpinBox(root);control.setRange(low,high);control.setValue(value);control.setSuffix(suffix);control.setKeyboardTracking(False);setattr(w,name,control);return control
    for name,text,callback in (
        ('generate_button','Generate matte',w.generate),('test_frame_button','Test frame',w.test_current_frame),
        ('stop_button','Stop / Cancel',w.cancel_jobs),('get_sam3_button','Get SAM 3 weights',w.get_sam3_weights),
        ('add_queue_button','Add files',w.add_queue_files),('remove_queue_button','Remove',w.remove_queue_item),
        ('clear_queue_button','Clear queue',w.clear_queue),('load_results_button','Load results…',w.load_results),
        ('queue_run_button','Run queue',w.run_queue),('clear_button','Clear all',w.clear_guidance),
        ('remove_guidance_button','Remove selected',w.remove_selected_guidance),
        ('undo_brush_button','Undo',w.undo_brush),('redo_brush_button','Redo',w.redo_brush),
        ('resume_button','Resume from this frame',w.resume_video),
        ('export_matte_button','Export matte',None),('export_button','Export PNG',None),('save_edits_button','Save matte',None)):
        button(name,text,callback)
    for name,text in (('point_button','Point +'),('negative_button','Point −'),('box_button','Box'),('white_brush_button','White'),('black_brush_button','Black')):
        button(name,text,tool=True).setCheckable(True)
    w.status=QLabel('Add files, then select an item.',root);w.edit_state=QLabel('',root)
    w.model_picker=QComboBox(root);w.model_picker.addItems(['BiRefNet HR Matting','LiteSDMatte (guided)',*ROTO_MODELS]);w.model_picker.setCurrentText('SAM 2 — Tracking')
    w.queue_list=QListWidget(root);w.queue_list.setHorizontalScrollMode(QListWidget.ScrollPerPixel)
    w.queue_list.currentItemChanged.connect(lambda current,previous:w.review_queue_item(current))
    w.view_tabs=QTabBar(root)
    for label in ('Original','Matte','Cutout'):w.view_tabs.addTab(label)
    w.view_tabs.hide();w.guidance_picker=QComboBox(root);w.guidance_picker.hide()
    w.canvas=w.canvas_class();w._studio_controls_ready=True
    button('fit_button','Fit',lambda:w.canvas.reset_zoom());button('zoom_out_button','−',lambda:w.canvas.set_zoom(w.canvas._zoom/1.25));button('zoom_in_button','+',lambda:w.canvas.set_zoom(w.canvas._zoom*1.25))
    spin('zoom_percent',100,1600,100,'%').valueChanged.connect(lambda value:w.canvas.set_zoom(value/100))
    spin('brush_size',1,2000,40,' px');spin('roto_size',256,1536,512,' px').setSingleStep(128)
    spin('roto_chunk',2,16,4,' frames');spin('trimap_width',1,256,24,' px')
    w.roto_text=QLineEdit(root);w.roto_text.setPlaceholderText('e.g. person in red jacket');w.roto_text.editingFinished.connect(w.save_file_guidance)
    w.video_panel=QFrame(root);layout=QVBoxLayout(w.video_panel);layout.setContentsMargins(5,5,5,5)
    w.frame_slider=QSlider(Qt.Horizontal);w.frame_slider.setRange(1,1);w.frame_slider.valueChanged.connect(w.request_frame);w.frame_slider.sliderPressed.connect(w._navigation_timer.stop);w.frame_slider.sliderReleased.connect(w.commit_slider_frame);layout.addWidget(w.frame_slider)
    w.video_toolbar=QWidget();row=QHBoxLayout(w.video_toolbar);row.setContentsMargins(0,0,0,0)
    for name,text,callback in (
        ('play_button','Play matted video',w.toggle_playback),('play_all_button','Play from beginning',lambda:w.start_playback(1)),
        ('previous_frame_button','Previous',lambda:w.request_frame(w.frame_spin.value()-1)),('next_frame_button','Next',lambda:w.request_frame(w.frame_spin.value()+1))):row.addWidget(button(name,text,callback))
    row.addWidget(QLabel('Frame'));row.addWidget(spin('frame_spin',1,1,1));w.frame_spin.valueChanged.connect(w.request_frame)
    w.frame_total=QLabel('/ 1');w.frame_time=QLabel('');row.addWidget(w.frame_total);row.addWidget(w.frame_time);row.addStretch();layout.addWidget(w.video_toolbar)
    # The inherited navigation helper updates these compatibility labels. They
    # are not a second interface or attached to the visible layout.
    button('load_matte_folder_button','Load matte folder',w.load_video_matte_folder)
    w.matte_folder_label=QLabel('',root);w.resume_hint=QLabel('',root)
    w.range_panel=QWidget(root);range_layout=QVBoxLayout(w.range_panel);range_layout.setContentsMargins(0,0,0,0)
    w.range_picker=QComboBox();w.range_picker.addItems(['Whole clip','From displayed frame','Selected range']);range_layout.addWidget(w.range_picker)
    range_row=QHBoxLayout();range_row.addWidget(spin('range_start',1,1,1));w.range_to=QLabel('to');range_row.addWidget(w.range_to);range_row.addWidget(spin('range_end',1,1,1));range_layout.addLayout(range_row);w.range_picker.currentIndexChanged.connect(w.update_workflow)
    w.undo_brush_action=QAction('Undo brush stroke',w);w.undo_brush_action.setShortcut(QKeySequence.Undo);w.undo_brush_action.triggered.connect(w.undo_brush);w.addAction(w.undo_brush_action)
    w.redo_brush_action=QAction('Redo brush stroke',w);w.redo_brush_action.setShortcuts([QKeySequence('Ctrl+Shift+Z'),QKeySequence('Ctrl+Y')]);w.redo_brush_action.triggered.connect(w.redo_brush);w.addAction(w.redo_brush_action)
    w._mask_input_folder=w._trimap_path=w._trimap_source='';w._trimap_frame=1;w._matte_dirty=False
    w._image_save_target=None;w._image_generation_model='';w._image_generation_guidance=[];w._image_refine_parent=''
    w._build_menus()
