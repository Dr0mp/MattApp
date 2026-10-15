"""Native SAM2Matting and SAM3.1 adapters inside the owned roto runtime."""
from pathlib import Path
import sys,inspect
import numpy as np
import torch
from PIL import Image


class CudaFrames:
    """Keep decoded frames on CPU; move only the frame requested by native heads."""
    def __init__(self,frames):self.frames=frames
    def __len__(self):return len(self.frames)
    def __getitem__(self,index):return self.frames[index].cuda()


def run_native(job,output,options,cache,save_alpha,release):
    root=Path(job['root']);family=job['tracker'];identity=('native',family)
    if identity not in cache:
        cache.clear();release()
        if family.startswith('sam2matting'):
            sys.path.insert(0,str(root/'third_party/SAM2Matting'))
            from sam2.build_sam import build_sam2matting_video_predictor
            variant='sam2.1base+' if family.endswith('base') else 'sam2.1tiny'
            weight='SAM2Matting-SAM2.1Base+.pt' if family.endswith('base') else 'SAM2Matting-SAM2.1Tiny.pt'
            predictor=build_sam2matting_video_predictor(f'configs/sam2matting-{variant}.yaml',str(root/'models/SAM2Matting/checkpoints'/weight),device='cuda')
        else:
            sys.path.insert(0,str(root/'third_party/sam3'))
            from sam3.model_builder import build_sam3_predictor
            from sam3.model import decoder
            from torch.nn.attention import sdpa_kernel,SDPBackend
            decoder.sdpa_kernel=lambda *args,**kwargs:sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH])
            predictor=build_sam3_predictor(checkpoint_path=str(root/'models/SAM3.1/sam3.1_multiplex.pt'),version='sam3.1',bpe_path=str(root/'third_party/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'),use_fa3=False,use_rope_real=False,compile=False,async_loading_frames=False)
            predictor.bf16_context.__exit__(None,None,None)
            # User-specified roto targets must appear immediately, including
            # short test/resume ranges; detector hot-start filters defer output.
            predictor.model.hotstart_delay=0;predictor.model.masklet_confirmation_enable=False
            predictor.model.postprocess_batch_size=1
            init_state=predictor.model.init_state;allowed=inspect.signature(init_state).parameters
            predictor.model.init_state=lambda **kwargs:init_state(**{key:value for key,value in kwargs.items() if key in allowed})
        cache[identity]=predictor
    predictor=cache[identity]
    frames=output/'native-frames';frames.mkdir(exist_ok=True)
    sizes=[]
    for i,path in enumerate(job['frames']):
        with Image.open(path) as raw:
            sizes.append(raw.size)
            raw.convert('RGB').save(frames/f'{i:06d}.jpg',quality=97,subsampling=0)
    points=[h for h in job['guidance'] if h['type']=='PointHint'];boxes=[h for h in job['guidance'] if h['type']=='BoxHint']
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        if family.startswith('sam2matting'):
            state=predictor.init_state(video_path=str(frames),offload_video_to_cpu=True,offload_state_to_cpu=True)
            state['images']=CudaFrames(state['images'])
            try:
                width,height=sizes[0]
                if options.get('seed_path'):
                    with Image.open(options['seed_path']) as raw:
                        if raw.size!=sizes[0]:raise ValueError('Starting mask dimensions differ from source')
                        values=(np.asarray(raw.convert('L'))>0).astype('float32')*20-10
                    mask=torch.nn.functional.interpolate(torch.from_numpy(values)[None,None],(256,256),mode='bilinear',align_corners=False).cuda()
                    predictor.add_new_mask(state,frame_idx=0,obj_id=1,mask=mask)
                else:
                    if not boxes and not any(p['positive'] for p in points):raise ValueError('SAM2Matting needs a foreground point, box, or starting mask')
                    for n,box in enumerate(boxes or [None]):
                        kwargs={}
                        if points:kwargs.update(points=np.array([[p['x']*width,p['y']*height] for p in points],dtype='float32'),labels=np.array([int(p['positive']) for p in points],dtype='int32'))
                        if box:kwargs['box']=np.array([min(box['left'],box['right'])*width,min(box['top'],box['bottom'])*height,max(box['left'],box['right'])*width,max(box['top'],box['bottom'])*height],dtype='float32')
                        predictor.add_new_points_or_box(state,frame_idx=0,obj_id=n+1,**kwargs)
                for index,ids,masks,alpha,unknown in predictor.propagate_in_video(state):
                    while alpha.ndim>2:alpha=alpha.max(0).values
                    save_alpha(alpha,output/f'alpha_{index:06d}.png',sizes[index])
            finally:predictor.reset_state(state);del state
        else:
            if options.get('seed_path') and not (points or boxes or options.get('text')):raise ValueError('SAM 3.1 multiplex does not accept mask seeds. Add a point, box, or text prompt to resume.')
            session=predictor.handle_request({'type':'start_session','resource_path':str(frames),'offload_video_to_cpu':True,'offload_state_to_cpu':True})['session_id']
            raw_outputs={};tracker=predictor.model.tracker.model;propagate=tracker.propagate_in_video
            def capture_tracking(*args,**kwargs):
                for result in propagate(*args,**kwargs):
                    index,ids,low,high,scores=result
                    masks=high>0
                    valid=scores.reshape(-1)>=0
                    masks=masks & valid.reshape(-1,*([1]*(masks.ndim-1)))
                    while masks.ndim>2:masks=masks.any(0)
                    alpha=masks.float().cpu().numpy()
                    raw_outputs[index]=np.maximum(raw_outputs.get(index,np.zeros_like(alpha)),alpha)
                    yield result
            tracker.propagate_in_video=capture_tracking
            try:
                common={'type':'add_prompt','session_id':session,'frame_index':0}
                if options.get('text') or boxes:
                    request={**common,'text':options.get('text') or ''}
                    if boxes:
                        request['bounding_boxes']=[[min(b['left'],b['right']),min(b['top'],b['bottom']),abs(b['right']-b['left']),abs(b['bottom']-b['top'])] for b in boxes];request['bounding_box_labels']=[1]*len(boxes)
                    initial=predictor.handle_request(request)
                    if points:
                        for obj_id in initial['outputs'].get('out_obj_ids',[]):
                            initial=predictor.handle_request({**common,'obj_id':int(obj_id),'points':[[p['x'],p['y']] for p in points],'point_labels':[int(p['positive']) for p in points]})
                elif points:
                    initial=predictor.handle_request({**common,'obj_id':1,'points':[[p['x'],p['y']] for p in points],'point_labels':[int(p['positive']) for p in points]})
                else:raise ValueError('SAM 3.1 needs points, boxes, or text')
                initial_masks=np.asarray(initial['outputs']['out_binary_masks'])
                if len(sizes)==1 and initial_masks.any():
                    save_alpha(initial_masks.any(axis=0).astype('float32'),output/'alpha_000000.png',sizes[0]);return
                for result in predictor.handle_stream_request({'type':'propagate_in_video','session_id':session,'propagation_direction':'forward','start_frame_index':0,'evict_cached_frame_outputs':True,'force_tracker_propagation':True}):
                    index=result['frame_index'];masks=np.asarray(result['outputs']['out_binary_masks'])
                    # The semantic wrapper suppresses tracker-only objects on
                    # later frames. Roto follows the explicitly prompted native
                    # tracker masks, including their object-presence scores.
                    if index in raw_outputs:
                        save_alpha(raw_outputs.pop(index),output/f'alpha_{index:06d}.png',sizes[index]);continue
                    if index==0 and not masks.any():masks=initial_masks
                    if index==0 and not masks.any():raise ValueError('SAM 3.1 found no object. Adjust the points, box, or text prompt.')
                    alpha=masks.any(axis=0).astype('float32') if len(masks) else np.zeros((sizes[index][1],sizes[index][0]),dtype='float32')
                    save_alpha(alpha,output/f'alpha_{index:06d}.png',sizes[index])
            finally:
                tracker.propagate_in_video=propagate
                predictor.handle_request({'type':'close_session','session_id':session})
