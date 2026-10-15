"""Independent processing worker. GUI browsing never owns or replaces its job."""
from dataclasses import dataclass,replace
from pathlib import Path
import tempfile
import shutil
from PIL import Image
from PySide6.QtCore import Signal
from .window import CancellableWorker,MediaPreview,VIDEO_SUFFIXES
from .models import make_model,RotoHint,PointHint,BoxHint
from .video import process_video,probe_video,load_frame,write_metadata
from .result_store import key
from .guidance_store import encode
from .foreground import ForegroundEstimator


@dataclass(frozen=True)
class Job:
    source:str
    stages:tuple
    guidance:tuple
    start:int=1
    end:int|None=None
    temporary:bool=False
    alpha_stage:str='refined'
    foreground:bool=False
    snapshot:tuple|None=None


class PipelineWorker(CancellableWorker):
    progress=Signal(int,int,str,str,int,int)
    result=Signal(str,str,int,object,object,object,bool)
    stage_changed=Signal(str,str,str)
    done=Signal()
    cutout_ready=Signal(str,int,object,object)

    def __init__(self,jobs,store):
        super().__init__();self.jobs=tuple(jobs);self.store=store

    def execute(self):
        # Snapshot existing refinement inputs before any outputs run. Painting or
        # saving in the UI cannot silently change the inputs of an active job.
        with tempfile.TemporaryDirectory(prefix='matte-pipeline-') as work,ForegroundEstimator() as estimator:
            self.foreground_estimator=estimator
            self.work=Path(work);self.inputs={};self.cut_inputs={};self.cut_choices={};self.generated={};self.input_stages={}
            for i,job in enumerate(self.jobs):
                self.cancellation.check()
                if not job.temporary and job.snapshot is None:self.store.commit_drafts(job.source,self.cancellation)
                if job.snapshot is None and all(stage=='cutout' for stage,name in job.stages):
                    self.cut_inputs[i]={};frames=set()
                    for stage in ('matte','refined'):
                        folder=self.work/f'cut-{i}-{stage}';folder.mkdir();metadata=self.store.metadata(job.source,stage)
                        for path in self.store.existing_folder(job.source,stage).glob('frame_*.png'):
                            self.cancellation.check();shutil.copyfile(path,folder/path.name);frames.add(int(path.stem[6:]))
                        self.cut_inputs[i][stage]=(folder,metadata.get('revision',0))
                    self.cut_choices[i]={n:self.store.preferred_alpha(job.source,n) for n in frames}
                if any(stage=='refined' for stage,name in job.stages) and not any(stage=='matte' for stage,name in job.stages):
                    snapshot=self.work/f'input-{i}';snapshot.mkdir()
                    frames=set(self.store.available_frames(job.source,'matte'))|set(self.store.available_frames(job.source,'refined'))
                    seed_only=all(name.startswith('MatAnyone') for stage,name in job.stages if stage=='refined')
                    frames={n for n in frames if n>=job.start and (job.end is None or n<=job.end) and (not seed_only or n==job.start)}
                    for number in sorted(frames):
                        self.cancellation.check();chosen=self.store.preferred_alpha(job.source,number)
                        draft=self.store.media_folder(job.source)/'.drafts'/chosen/f'frame_{number:06d}.png'
                        alpha=None
                        if draft.exists():
                            with Image.open(draft) as image:alpha=image.convert('L')
                        if alpha is None:alpha=self.store.alpha(job.source,chosen,number)
                        if alpha is None:continue
                        raw=load_frame(job.source,number,self.cancellation) if Path(job.source).suffix.lower() in VIDEO_SUFFIXES else Image.open(job.source).convert('RGBA')
                        raw.putalpha(alpha);raw.save(snapshot/f'frame_{number:06d}.png','PNG',compress_level=1)
                    chosen=self.store.preferred_alpha(job.source,job.start);self.input_stages[key(job.source)]=chosen
                    meta=self.store.metadata(job.source,chosen);meta.setdefault('source_image' if Path(job.source).suffix.lower() not in VIDEO_SUFFIXES else 'source_video',job.source)
                    write_metadata(snapshot,meta);self.inputs[i]=(snapshot,meta.get('revision',0))
            for i,job in enumerate(self.jobs):
                self.cancellation.check();self.process_item(i,job)
            self.cancellation.check();self.done.emit()

    def process_item(self,index,job):
        video=Path(job.source).suffix.lower() in VIDEO_SUFFIXES
        info=probe_video(job.source,self.cancellation) if video else None
        end=job.end or (info.frame_count if info else 1)
        if not 1<=job.start<=end or info and end>info.frame_count:raise ValueError('Invalid processing range')
        count=end-job.start+1
        for step,(stage,name) in enumerate(job.stages):
            self.cancellation.check();self.stage_changed.emit(job.source,stage,'running')
            parent_revision=None
            hints=list(job.guidance)
            if stage=='matte' and name.startswith('SAM') and not any(isinstance(h,BoxHint) or isinstance(h,PointHint) and h.positive for h in hints):
                option=next((h for h in hints if isinstance(h,RotoHint)),RotoHint())
                if not option.seed_path and not option.text:
                    # Redo must not recycle brush scribbles as implicit guidance.
                    # A displayed/painted seed is used only when explicitly chosen.
                    seed=self.store.alpha(job.source,'matte',job.start)
                    if seed is None:seed=self.store.alpha(job.source,'refined',job.start)
                    if seed is not None:
                        seed_path=self.work/f'seed-{index}.png';seed.save(seed_path,'PNG',compress_level=1)
                        hints=[h for h in hints if not isinstance(h,RotoHint)]+[replace(option,seed_path=str(seed_path),seed_source=job.source,seed_frame=job.start)]
            if stage=='refined':
                if index in self.inputs:mask_folder,parent_revision=self.inputs[index]
                else:
                    mask_folder=self.generated[key(job.source)];parent_revision=self.store.metadata(job.source,'matte').get('revision',0)
                self.expected_input_revision=parent_revision
                options=next((h for h in hints if isinstance(h,RotoHint)),RotoHint())
                hints=[h for h in hints if not isinstance(h,RotoHint)]+[replace(options,mask_folder=str(mask_folder),mask_source=job.source,mask_frame=job.start,seed_path='')]
                from .saved_masks import mask_files
                mask_files(mask_folder,job.source,job.start,count,name.startswith('MatAnyone'))
            if stage=='matte' and not name.startswith('SAM'):hints=[h for h in hints if isinstance(h,(PointHint,BoxHint))]
            def consume(number,total,image):
                self.cancellation.check();alpha=image.getchannel('A');original=image.copy();original.putalpha(255)
                self.commit(job,stage,name,number,original,alpha,info,parent_revision)
                self.progress.emit(index,len(self.jobs),job.source,stage,step*count+number-job.start+1,len(job.stages)*count)
            if stage=='cutout':
                source_stage=job.alpha_stage
                if job.snapshot is not None:frames=[job.start]
                elif source_stage=='chosen':frames=sorted(self.cut_choices[index]) if index in self.cut_choices else sorted(set(self.store.frames(job.source,'matte'))|set(self.store.frames(job.source,'refined')))
                else:
                    if not self.store.frames(job.source,source_stage):source_stage='matte'
                    frames=self.store.frames(job.source,source_stage)
                frames=[n for n in frames if job.start<=n<=end]
                if not frames:raise ValueError('No saved masks for cutouts')
                for offset,number in enumerate(frames,1):
                    self.cancellation.check()
                    chosen=self.cut_choices.get(index,{}).get(number,self.store.preferred_alpha(job.source,number)) if source_stage=='chosen' else source_stage
                    if job.snapshot is not None:
                        original,alpha=job.snapshot;revision=self.store.metadata(job.source,chosen).get('revision',0);stale=False
                    else:
                        if index in self.cut_inputs:
                            folder,revision=self.cut_inputs[index][chosen];path=folder/f'frame_{number:06d}.png'
                        else:revision=self.store.metadata(job.source,chosen).get('revision',0);path=self.store.frame(job.source,chosen,number)
                        if not path.exists():raise ValueError(f'Missing {chosen} mask for frame {number}')
                        with Image.open(path) as image:rgba=image.convert('RGBA')
                        original=rgba.copy();original.putalpha(255);alpha=rgba.getchannel('A')
                        stale=revision!=self.store.metadata(job.source,chosen).get('revision',0) or (self.store.media_folder(job.source)/'.drafts'/chosen/f'frame_{number:06d}.png').exists()
                    if job.foreground:original=self.foreground_estimator.estimate(original,alpha,self.cancellation)
                    self.cancellation.check()
                    stale=stale or revision!=self.store.metadata(job.source,chosen).get('revision',0)
                    if job.snapshot is None:stale=stale or (self.store.media_folder(job.source)/'.drafts'/chosen/f'frame_{number:06d}.png').exists()
                    self.store.save(job.source,'cutout',number,original,alpha,info=info,alpha_stage=chosen,parent_revision=revision,stale=stale,foreground_estimated=job.foreground)
                    if job.foreground:self.cutout_ready.emit(job.source,number,original,alpha)
                    self.progress.emit(index,len(self.jobs),job.source,stage,step*count+offset,len(job.stages)*count)
            elif video and not job.temporary:
                stage_folder=self.work/f'stage-{index}-{step}'
                process_video(job.source,stage_folder,make_model(name),name,hints,self.cancellation,job.start,frame_ready=consume,end_frame=end)
            else:
                original=load_frame(job.source,job.start,self.cancellation) if video else Image.open(job.source).convert('RGBA')
                alpha=make_model(name).predict(original,hints,self.cancellation)
                self.cancellation.check();self.commit(job,stage,name,job.start,original,alpha,info,parent_revision)
                self.progress.emit(index,len(self.jobs),job.source,stage,(step+1)*count,len(job.stages)*count)
            self.cancellation.check();self.stage_changed.emit(job.source,stage,'complete')
        self.progress.emit(index+1,len(self.jobs),job.source,job.stages[-1][0],0,0)

    def commit(self,job,stage,name,number,original,alpha,info,parent_revision):
        if job.temporary:
            folder=self.work/('generated-'+str(self.jobs.index(job)))
            folder.mkdir(exist_ok=True)
            rgba=original.convert('RGBA');rgba.putalpha(alpha);rgba.save(folder/f'frame_{number:06d}.png','PNG',compress_level=1)
            write_metadata(folder,{'source_video' if info else 'source_image':job.source,'width':original.width,'height':original.height,'frame_count':info.frame_count if info else 1})
            self.generated[key(job.source)]=folder
        else:self.generated[key(job.source)]=self.store.folder(job.source,stage)
        input_stage=self.input_stages.get(key(job.source),'matte')
        if not job.temporary:
            stale=stage=='refined' and (self.store.metadata(job.source,input_stage).get('revision',0)!=self.expected_input_revision or any((self.store.media_folder(job.source)/'.drafts'/input_stage).glob('frame_*.png')))
            self.store.save(job.source,stage,number,original,alpha,name,info,source_masks=str(self.store.existing_folder(job.source,input_stage)) if stage=='refined' else '',parent_revision=parent_revision,stale=stale,guidance=encode(job.guidance))
            self.store.choose_alpha(job.source,stage,number)
            if stage=='refined' and input_stage=='refined':self.expected_input_revision+=1
        if job.temporary:self.result.emit(job.source,stage,number,original,alpha,info,True)
        # Live preview uses bounded buffering, never a signal backlog of images.
        image=original.convert('RGBA');image.putalpha(alpha)
        self.publish_frame((stage,MediaPreview(job.source,image,None,number,info,str(self.store.folder(job.source,stage))),job.temporary))
