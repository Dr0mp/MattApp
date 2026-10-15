from pathlib import Path
from PySide6.QtWidgets import QMessageBox
from .guidance_store import GuidanceStore, source_key
from .models import PointHint,BoxHint,RotoHint
from .roto import REFINERS,ROTO_MODELS,readiness

class GuidanceMixin:
    def guidance_store(self):
        return GuidanceStore(self.results_index_path.with_name('guidance_by_file.json'))

    def save_file_guidance(self, force=False):
        if self._active_worker is not None or getattr(self,'_pending_project_state',None) or not self.current_media_path or self.model_picker.currentText() in REFINERS:
            return True
        try:
            # Merely viewing an unguided file must not create an explicit clear
            # that would hide guidance imported later from an older result.
            if not force and not (self.points or self.boxes or self.roto_text.text().strip() or self.seed_path) and self.guidance_store().get(self.current_media_path,self.current_frame) is None:
                return True
            hints = [*self.points,*self.boxes,RotoHint(text=self.roto_text.text().strip(),max_size=self.roto_size.value(),chunk_frames=self.roto_chunk.value(),seed_path=self.seed_path,seed_source=self.seed_source,seed_frame=self.seed_frame)]
            self.guidance_store().save(self.current_media_path,self.current_frame,hints)
            self.update_guidance_tooltips()
            return True
        except (OSError,ValueError,TypeError) as error:
            QMessageBox.information(self,'Could not save guidance',f'Guidance is still visible but could not be saved: {error}')
            return False

    def restore_file_guidance(self,source,frame,changed):
        try:saved = self.guidance_store().get(source,frame)
        except (OSError,ValueError,TypeError) as error:
            QMessageBox.information(self,'Could not restore saved guidance',str(error))
            saved = None
        if saved is None and not changed:return False
        self.points.clear()
        self.boxes.clear()
        self.seed_path = self.seed_source = ''
        self.seed_frame = frame
        self.roto_text.blockSignals(True)
        self.roto_text.clear()
        for hint in saved or []:
            if isinstance(hint,PointHint):self.points.append(hint)
            elif isinstance(hint,BoxHint):self.boxes.append(hint)
            elif isinstance(hint,RotoHint):
                self.roto_text.setText(hint.text)
                self.roto_size.setValue(hint.max_size)
                self.roto_chunk.setValue(hint.chunk_frames)
                self.seed_path,self.seed_source,self.seed_frame = hint.seed_path,hint.seed_source,hint.seed_frame
        self.roto_text.blockSignals(False)
        self.update_guidance_tooltips()
        return saved is not None

    def update_guidance_tooltips(self):
        records = self.guidance_store().read()
        for row in range(self.queue_list.count()):
            item=self.queue_list.item(row)
            record=records.get(source_key(item.text()),{}).get('1')
            if record is None:description='No saved guidance'
            else:
                points=sum(h.get('type')=='PointHint' for h in record)
                boxes=sum(h.get('type')=='BoxHint' for h in record)
                description=f'Saved guidance: {points} point(s), {boxes} box(es)'
            item.setToolTip(item.text()+'\n'+description)

    def queue_guidance_snapshot(self,paths):
        if not self.save_file_guidance():return None
        name=self.model_picker.currentText()
        if name in REFINERS:
            QMessageBox.information(self,'Select one clip','Saved-mask refinement acts on the selected clip. Create masks or Automatic workflows can run the queue.')
            return None
        if name in ROTO_MODELS:
            message=readiness(name,Path(__file__).resolve().parents[1])
            if not message.startswith('Ready'):
                QMessageBox.information(self,'Model setup needed',message)
                return None
        snapshot={}
        missing=[]
        try:
            for path in paths:
                hints=self.guidance_store().get(path,1) or []
                option=next((h for h in hints if isinstance(h,RotoHint)),RotoHint(max_size=self.roto_size.value(),chunk_frames=self.roto_chunk.value()))
                valid_seed=bool(option.seed_path) and Path(option.seed_path).is_file() and option.seed_frame==1 and source_key(option.seed_source)==source_key(path)
                selection=any(isinstance(h,BoxHint) or isinstance(h,PointHint) and h.positive for h in hints)
                if name in ROTO_MODELS:
                    selection=selection or name.startswith('SAM 3') and bool(option.text.strip()) or valid_seed
                if (name in ROTO_MODELS or name.startswith('LiteSDMatte')) and not selection:missing.append(path)
                if name in ROTO_MODELS and option.seed_path and not valid_seed:missing.append(path)
                # Copy each immutable hint into a separate list for the running job.
                hints=[h for h in hints if not isinstance(h,RotoHint)]
                if name in ROTO_MODELS:hints.append(option)
                snapshot[source_key(path)]=list(hints)
        except (OSError,ValueError,TypeError) as error:
            QMessageBox.information(self,'Could not read saved guidance',str(error))
            return None
        if missing:
            names='\n'.join(Path(path).name for path in dict.fromkeys(missing))
            QMessageBox.information(self,'Guidance needed for these files','Select each file and add its own foreground point, box, or supported prompt/mask before running the queue. Video queues use frame 1 guidance.\n\n'+names)
            return None
        return snapshot
