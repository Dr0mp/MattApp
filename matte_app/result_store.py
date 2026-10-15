"""Source-bound, atomic base/refined/cutout storage beneath the media project root."""
from pathlib import Path
import json
import threading
import uuid
import time
from PIL import Image
from .video import read_metadata


def key(path):return str(Path(path).resolve()).casefold() if path else ''


class ResultStore:
    def __init__(self,root=None,legacy=None):
        self.root=Path(root).resolve() if root else None
        self.legacy={key(p):Path(f) for p,f in (legacy or {}).items()}
        self.lock=threading.RLock()

    def project_root(self,source):
        source=Path(source).resolve()
        if self.root:
            source.relative_to(self.root)
            return self.root
        # The agreed layout is root/folder/images/file, not a drive-root walk.
        if source.parent.name.casefold() in ('images','videos') and len(source.parents)>=3:
            return source.parents[2]
        return source.parent

    def media_folder(self,source):
        source=Path(source).resolve();root=self.project_root(source)
        relative=source.relative_to(root)
        destination=root/'matted'/relative
        destination.resolve().relative_to((root/'matted').resolve())
        return destination

    def folder(self,source,stage):
        if stage not in ('matte','refined','cutout'):raise ValueError('Unknown result stage')
        return self.media_folder(source)/stage

    def existing_folder(self,source,stage):
        folder=self.folder(source,stage)
        if (folder/'frames.json').is_file():return folder
        legacy=self.legacy.get(key(source))
        if legacy and legacy.is_dir():
            if (legacy/stage/'frames.json').is_file():return legacy/stage
            meta=read_metadata(legacy);model=meta.get('model','')
            legacy_stage=meta.get('stage') or ('refined' if 'Saved masks' in model or '+' in model else 'matte')
            if stage==legacy_stage:return legacy
        return folder

    def metadata(self,source,stage):
        folder=self.existing_folder(source,stage);meta=read_metadata(folder)
        bound=meta.get('source_image') or meta.get('source_video')
        if bound and key(bound)!=key(source):raise ValueError('Saved result belongs to another source')
        return meta

    def frame(self,source,stage,number=1):
        self.metadata(source,stage)
        return self.existing_folder(source,stage)/f'frame_{number:06d}.png'

    def alpha(self,source,stage,number=1,size=None):
        path=self.frame(source,stage,number)
        if not path.is_file():return None
        with Image.open(path) as image:
            if 'A' not in image.getbands():raise ValueError('Saved result has no alpha channel')
            if size and image.size!=size:raise ValueError('Saved mask dimensions do not match source')
            return image.getchannel('A')

    def frames(self,source,stage):
        self.metadata(source,stage)
        return sorted(int(p.stem[6:]) for p in self.existing_folder(source,stage).glob('frame_*.png') if p.stem[6:].isdigit())

    def available_frames(self,source,stage):
        drafts=self.media_folder(source)/'.drafts'/stage
        return sorted(set(self.frames(source,stage))|{int(p.stem[6:]) for p in drafts.glob('frame_*.png') if p.stem[6:].isdigit()})

    def preferred_alpha(self,source,frame=1):
        try:manifest=json.loads((self.media_folder(source)/'source.json').read_text())
        except (OSError,ValueError):manifest={}
        chosen=manifest.get('alpha_sources',{}).get(str(frame))
        if chosen in ('matte','refined'):return chosen
        saved=self.metadata(source,'cutout');chosen=saved.get('alpha_sources',{}).get(str(frame),saved.get('alpha_stage'))
        if chosen in ('matte','refined'):return chosen
        if (self.existing_folder(source,'refined')/f'frame_{frame:06d}.png').is_file() or (self.media_folder(source)/'.drafts'/'refined'/f'frame_{frame:06d}.png').is_file():
            return 'refined'
        return 'matte'

    def choose_alpha(self,source,stage,frame=1):
        if stage not in ('matte','refined'):raise ValueError('Invalid cutout alpha')
        path=self.media_folder(source)/'source.json'
        with self.lock:
            try:data=json.loads(path.read_text())
            except (OSError,ValueError):data={}
            data.setdefault('alpha_sources',{})[str(frame)]=stage;self.write_json(path,data)

    def commit_drafts(self,source,token,only=None):
        from .video import load_frame,probe_video
        video=Path(source).suffix.lower() in ('.mp4','.mov','.mkv','.avi','.webm','.m4v')
        info=probe_video(source,token) if video else None
        for stage in ((only,) if only else ('matte','refined')):
            for draft in sorted((self.media_folder(source)/'.drafts'/stage).glob('frame_*.png')):
                token.check();stamp=draft.stat().st_mtime_ns;number=int(draft.stem[6:])
                with Image.open(draft) as raw:alpha=raw.convert('L')
                if video:original=load_frame(source,number,token)
                else:
                    with Image.open(source) as raw:original=raw.convert('RGBA')
                self.save(source,stage,number,original,alpha,info=info,stale=False)
                if draft.exists() and draft.stat().st_mtime_ns==stamp:draft.unlink()

    def scan(self,source):
        result={}
        expected=None
        if Path(source).suffix.lower() not in ('.mp4','.mov','.mkv','.avi','.webm','.m4v'):
            with Image.open(source) as image:expected=image.size
        for stage in ('matte','refined','cutout'):
            try:
                meta=self.metadata(source,stage);frames=self.frames(source,stage)
                if frames:self.alpha(source,stage,frames[0],expected)
                complete=bool(frames) and meta.get('status')=='complete' and len(frames)>=meta.get('frame_count',1)
                result[stage]={'available':bool(frames),'complete':complete,'frames':len(frames),'revision':meta.get('revision',0),'stale':bool(meta.get('stale')),'model':meta.get('model','')}
            except (OSError,ValueError,TypeError):result[stage]={'available':False,'complete':False,'frames':0,'revision':0,'stale':False}
        return result

    def write_json(self,path,data):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:temp.write_text(json.dumps(data,indent=2),encoding='utf-8');self.replace_file(temp,path)
        finally:temp.unlink(missing_ok=True)

    @staticmethod
    def replace_file(pending,path):
        deadline=time.monotonic()+2
        while True:
            try:pending.replace(path);return
            except PermissionError:
                if time.monotonic()>=deadline:raise
                time.sleep(.02)

    def save(self,source,stage,frame,original,alpha,model='',info=None,**extra):
        if alpha.size!=original.size:raise ValueError('Alpha/source dimensions differ')
        folder=self.folder(source,stage)
        with self.lock:
            folder.mkdir(parents=True,exist_ok=True)
            image=original.convert('RGBA');image.putalpha(alpha)
            pending=folder/(f'.frame-{frame}-{uuid.uuid4().hex}.tmp')
            try:image.save(pending,'PNG',compress_level=1);self.replace_file(pending,folder/f'frame_{frame:06d}.png')
            finally:pending.unlink(missing_ok=True)
            if stage in ('matte','refined'):
                pending=folder/(f'.alpha-{frame}-{uuid.uuid4().hex}.tmp')
                try:alpha.convert('L').save(pending,'PNG',compress_level=1);self.replace_file(pending,folder/f'alpha_{frame:06d}.png')
                finally:pending.unlink(missing_ok=True)
            meta=read_metadata(folder)
            bound=meta.get('source_image') or meta.get('source_video')
            if bound and key(bound)!=key(source):raise ValueError('Saved result belongs to another source')
            meta.update({'source_video' if info else 'source_image':str(Path(source).resolve()),'stage':stage,'frame_count':info.frame_count if info else 1,'fps':info.fps if info else 0,'width':original.width,'height':original.height,'model':model or meta.get('model',''),'revision':meta.get('revision',0)+1,**extra})
            if stage=='cutout':meta.setdefault('foreground_sources',{})[str(frame)]=bool(extra.get('foreground_estimated',False))
            if stage=='cutout' and extra.get('alpha_stage'):meta.setdefault('alpha_sources',{})[str(frame)]=extra['alpha_stage']
            if not info and meta['frame_count']==1:
                meta['status']='complete'
            else:
                numbers={int(p.stem[6:]) for p in folder.glob('frame_*.png') if p.stem[6:].isdigit()}
                meta['status']='complete' if all(n in numbers for n in range(1,meta['frame_count']+1)) else 'partial'
            self.write_json(folder/'frames.json',meta)
            source_file=self.media_folder(source)/'source.json'
            try:manifest=json.loads(source_file.read_text())
            except (OSError,ValueError):manifest={}
            resolved_source=str(Path(source).resolve())
            rel=str(Path(source).resolve().relative_to(self.project_root(source)))
            if manifest.get('source')!=resolved_source or manifest.get('relative_source')!=rel or manifest.get('stages')!=['matte','refined','cutout']:
                manifest.update({'source':resolved_source,'relative_source':rel,'stages':['matte','refined','cutout']})
                self.write_json(source_file,manifest)
            if stage in ('matte','refined'):
                for downstream in (('refined','cutout') if stage=='matte' else ('cutout',)):
                    target=self.folder(source,downstream)/'frames.json'
                    if target.exists():
                        data=read_metadata(target.parent)
                        if not data.get('stale'):
                            data['stale']=True;self.write_json(target,data)
        return folder

