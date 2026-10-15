"""Isolated local inference. Only atomic, complete alpha files cross into the app."""
from __future__ import annotations
import argparse
import gc
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch

_TRACKER_CACHE={}


def save_alpha(alpha, path, original_size):
    if not isinstance(alpha, Image.Image):
        if isinstance(alpha, torch.Tensor): alpha = alpha.detach().float().cpu().numpy()
        alpha = Image.fromarray((np.clip(np.asarray(alpha).squeeze(), 0, 1)*255).astype('uint8'), 'L')
    alpha = alpha.convert('L').resize(original_size, Image.Resampling.LANCZOS)
    pending = path.with_suffix('.tmp')
    alpha.save(pending, 'PNG', compress_level=1)
    pending.replace(path)


def release(*objects):
    gc.collect()
    torch.cuda.empty_cache()


def track(job, output, options, first_only=False):
    from transformers import Sam2VideoModel, Sam2VideoProcessor, Sam3TrackerVideoModel, Sam3TrackerVideoProcessor, Sam3VideoModel, Sam3VideoProcessor
    seed = Image.open(options['seed_path']).convert('L') if options.get('seed_path') else None
    text_mode = job['tracker'] == 'sam3' and options.get('text', '').strip() and seed is None
    model_cls, processor_cls = (Sam3VideoModel, Sam3VideoProcessor) if text_mode else (Sam3TrackerVideoModel, Sam3TrackerVideoProcessor) if job['tracker'] == 'sam3' else (Sam2VideoModel, Sam2VideoProcessor)
    checkpoint = Path(job['root']) / 'models' / ('SAM3' if job['tracker']=='sam3' else 'SAM2')
    if not (checkpoint / 'model.safetensors').exists():
        raise RuntimeError('SAM 3 checkpoint access/setup is needed.' if job['tracker']=='sam3' else 'SAM 2 checkpoint setup is needed.')
    cache_key=(str(checkpoint),model_cls.__name__)
    if cache_key not in _TRACKER_CACHE:
        _TRACKER_CACHE.clear();release()
        model = model_cls.from_pretrained(checkpoint, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
        processor = processor_cls.from_pretrained(checkpoint, local_files_only=True)
        _TRACKER_CACHE[cache_key]=(model,processor)
    model,processor=_TRACKER_CACHE[cache_key]
    session = processor.init_video_session(inference_device='cuda', processing_device='cpu', video_storage_device='cpu', dtype=torch.bfloat16)
    if text_mode: processor.add_text_prompt(session, options['text'].strip())
    points = [h for h in job['guidance'] if h['type']=='PointHint']
    boxes = [h for h in job['guidance'] if h['type']=='BoxHint']
    if not text_mode and seed is None and not (boxes or any(p['positive'] for p in points)):
        raise ValueError('Add a foreground point or box, or use a starting mask.')
    selected_ids = None
    for i, path in enumerate(job['frames'][:1] if first_only else job['frames']):
        with Image.open(path) as source: image = source.convert('RGB')
        inputs = processor(images=image, device='cuda', return_tensors='pt').to('cuda')
        if i == 0 and not text_mode:
            width, height = image.size
            if seed is not None:
                if seed.size != image.size: raise ValueError('Starting mask dimensions do not match the source.')
                processor.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=1,
                    input_masks=(np.asarray(seed) >= 128).astype(np.float32), original_size=inputs.original_sizes[0])
            else:
                for n, box in enumerate(boxes or [None]):
                    kwargs = {}
                    if box:
                        left,right=sorted((box['left']*width,box['right']*width)); top,bottom=sorted((box['top']*height,box['bottom']*height))
                        kwargs['input_boxes'] = [[[left,top,right,bottom]]]
                    if points:
                        kwargs['input_points'] = [[[[p['x']*width, p['y']*height] for p in points]]]
                        kwargs['input_labels'] = [[[int(p['positive']) for p in points]]]
                    processor.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=n+1, original_size=inputs.original_sizes[0], **kwargs)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(inference_session=session, frame=inputs.pixel_values[0])
        if text_mode:
            found = processor.postprocess_outputs(session, result, original_sizes=inputs.original_sizes)
            masks = found['masks'].bool()
            ids = found['object_ids'].tolist()
            if i == 0 and (points or boxes):
                selected_ids = set()
                for index, object_id in enumerate(ids):
                    mask = masks[index].squeeze()
                    positive = [p for p in points if p['positive']]
                    in_points = not positive or any(bool(mask[min(mask.shape[0]-1,round(p['y']*(mask.shape[0]-1))),min(mask.shape[1]-1,round(p['x']*(mask.shape[1]-1)))]) for p in positive)
                    excluded = any(bool(mask[min(mask.shape[0]-1,round(p['y']*(mask.shape[0]-1))),min(mask.shape[1]-1,round(p['x']*(mask.shape[1]-1)))]) for p in points if not p['positive'])
                    in_box = not boxes or any(mask[round(min(b['top'],b['bottom'])*mask.shape[0]):round(max(b['top'],b['bottom'])*mask.shape[0]),round(min(b['left'],b['right'])*mask.shape[1]):round(max(b['left'],b['right'])*mask.shape[1])].any().item() for b in boxes)
                    if in_points and in_box and not excluded: selected_ids.add(object_id)
                if not selected_ids: raise ValueError('SAM 3 found no object matching the text and guidance. Adjust the prompt or box.')
            chosen = [masks[n].squeeze() for n, oid in enumerate(ids) if selected_ids is None or oid in selected_ids]
            alpha = torch.stack(chosen).any(0).float() if chosen else torch.zeros((image.height,image.width))
        else:
            masks = processor.post_process_masks([result.pred_masks], original_sizes=inputs.original_sizes, binarize=False)[0]
            alpha = (masks > 0).any(0).squeeze().float()
        save_alpha(alpha, output / f'alpha_{i:06d}.png', image.size)
    del model, processor, session
    release()


def matanyone(job, output, options, seed_path):
    if job['matte']=='matanyone2':
        sys.path.insert(0,str(Path(job['root'])/'third_party/MatAnyone2'))
        from matanyone2 import MatAnyone2,InferenceCore
        from omegaconf import OmegaConf
        from safetensors.torch import load_model
        folder=Path(job['root'])/'models/MatAnyone2';config=json.loads((folder/'config.json').read_text())
        cfg=OmegaConf.create(config['cfg']);cfg.model.pretrained_resnet=False
        network=MatAnyone2(cfg,single_object=config.get('single_object',True))
        load_model(network,str(folder/'model.safetensors'),strict=True)
        processor=InferenceCore(network,device='cuda')
    else:
        sys.path.insert(0,str(Path(job['root'])/'third_party/MatAnyone'))
        from matanyone import InferenceCore
        processor=InferenceCore(str(Path(job['root'])/'models/MatAnyone'),device='cuda')
    with Image.open(job['frames'][0]) as source: original_size = source.size
    edge = options.get('max_size', 768)
    scale = min(1, edge/max(original_size))
    size = (max(1,round(original_size[0]*scale)), max(1,round(original_size[1]*scale)))
    with Image.open(seed_path) as source: mask = source.convert('L').resize(size, Image.Resampling.NEAREST)
    if not np.any(np.asarray(mask)>=128): raise ValueError('The starting mask is empty. Select the subject first.')
    mask = torch.from_numpy((np.asarray(mask)>=128).astype('float32')*255).cuda()
    for i, path in enumerate(job['frames']):
        with Image.open(path) as source: image = source.convert('RGB').resize(size, Image.Resampling.BILINEAR)
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float().cuda()/255
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            if i == 0:
                processor.step(tensor, mask, objects=[1])
                for _ in range(10): probability = processor.step(tensor, first_frame_pred=True)
            else: probability = processor.step(tensor)
            alpha = processor.output_prob_to_mask(probability)
        save_alpha(alpha, output/f'alpha_{i:06d}.png', original_size)
    del processor
    release()


def videomama(job, output, options, masks):
    sys.path.insert(0, str(Path(job['root'])/'third_party/VideoMaMa'))
    from pipeline_svd_mask import VideoInferencePipeline
    pipeline = VideoInferencePipeline(str(Path(job['root'])/'models/SVD'),str(Path(job['root'])/'models/VideoMaMa'))
    pipeline.unet.enable_forward_chunking(chunk_size=1, dim=1)
    original_size = Image.open(job['frames'][0]).size
    edge = options.get('max_size', 512)
    ratio = min(1,edge/max(original_size))
    scaled = (max(1,round(original_size[0]*ratio)),max(1,round(original_size[1]*ratio)))
    padded = (max(64,((scaled[0]+63)//64)*64),max(64,((scaled[1]+63)//64)*64))
    chunk = max(2,options.get('chunk_frames',4))
    overlap = min(2,chunk//2)
    pending = {}
    length = len(job['frames'])
    for start in range(0,length,chunk-overlap):
        end = min(length,start+chunk)
        images, guides = [], []
        for i in range(start,end):
            with Image.open(job['frames'][i]) as source: image = source.convert('RGB').resize(scaled, Image.Resampling.BILINEAR)
            with Image.open(masks/f'alpha_{i:06d}.png') as source: mask = source.convert('L').resize(scaled, Image.Resampling.NEAREST)
            images.append(ImageOps.expand(image,(0,0,padded[0]-scaled[0],padded[1]-scaled[1])))
            guides.append(ImageOps.expand(mask,(0,0,padded[0]-scaled[0],padded[1]-scaled[1])))
        predictions = pipeline.run(images, guides)
        for i, prediction in enumerate(predictions, start):
            alpha = prediction.convert('L').crop((0,0,*scaled))
            if i in pending:
                weight = (i-start+1)/(overlap+1)
                alpha = Image.blend(pending.pop(i), alpha, weight)
            if end==length or i < end-overlap:
                save_alpha(alpha,output/f'alpha_{i:06d}.png',original_size)
            else: pending[i] = alpha
        if end==length: break
    del pipeline
    release()


def vitmatte(job, output, options, masks):
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor
    from saved_masks import make_trimap, validate_trimap
    checkpoint = Path(job['root'])/'models/ViTMatte'
    model = VitMatteForImageMatting.from_pretrained(checkpoint,local_files_only=True).cuda().eval()
    processor = VitMatteImageProcessor.from_pretrained(checkpoint,local_files_only=True)
    for i,path in enumerate(job['frames']):
        with Image.open(path) as source: image = source.convert('RGB')
        with Image.open(masks/f'alpha_{i:06d}.png') as source: mask = source.convert('L')
        if options.get('trimap_path') and job['start_frame']+i == options.get('trimap_frame',1):
            with Image.open(options['trimap_path']) as source: trimap = validate_trimap(source.convert('L'),image.size)
        else: trimap = make_trimap(mask, options.get('trimap_width',24))
        scale = min(1,options.get('max_size',1024)/max(image.size))
        size = (max(1,round(image.width*scale)),max(1,round(image.height*scale)))
        small_image = image.resize(size,Image.Resampling.BILINEAR)
        small_trimap = trimap.resize(size,Image.Resampling.NEAREST)
        inputs = processor(images=small_image,trimaps=small_trimap,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            alpha = model(**inputs).alphas[0,0,:size[1],:size[0]]
        predicted = Image.fromarray((alpha.float().cpu().numpy().clip(0,1)*255).astype('uint8')).resize(image.size,Image.Resampling.BILINEAR)
        values = np.asarray(predicted).copy()
        known = np.asarray(trimap)
        values[known==0] = 0
        values[known==255] = 255
        save_alpha(Image.fromarray(values),output/f'alpha_{i:06d}.png',image.size)
    del model, processor
    release()


def run_job(job):
    if not torch.cuda.is_available(): raise RuntimeError('Roto requires an NVIDIA GPU with CUDA in this runtime.')
    torch.set_num_threads(2)
    output = Path(job['output'])
    options = next((h for h in job['guidance'] if h['type']=='RotoHint'), {})
    if job['tracker'].startswith('sam2matting') or job['tracker']=='sam31':
        from new_trackers import run_native
        run_native(job,output,options,_TRACKER_CACHE,save_alpha,release)
    elif job['matte']=='tracking': track(job,output,options)
    else:
        _TRACKER_CACHE.clear();release()
        masks = output/'masks'
        masks.mkdir(exist_ok=True)
        if job['tracker']=='none':
            from saved_masks import read_mask
            for i, path in enumerate(job['mask_paths']):
                read_mask(path).save(masks/f'alpha_{i:06d}.png')
            seed_path = masks/'alpha_000000.png'
        elif job['matte'] in ('matanyone','matanyone2') and options.get('seed_path'):
            seed_path = Path(options['seed_path'])
        else:
            track(job,masks,options,first_only=job['matte'] in ('matanyone','matanyone2'))
            seed_path = masks/'alpha_000000.png'
            _TRACKER_CACHE.clear();release()
        if job['matte']=='vitmatte': vitmatte(job,output,options,masks)
        elif job['matte'] in ('matanyone','matanyone2'): matanyone(job,output,options,seed_path)
        else: videomama(job,output,options,masks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job')
    parser.add_argument('--server',action='store_true')
    args = parser.parse_args()
    if args.server:
        import traceback
        for line in sys.stdin:
            path=Path(line.strip())
            try:run_job(json.loads(path.read_text(encoding='utf-8')));status={'ok':True}
            except Exception:status={'error':traceback.format_exc()}
            pending=path.parent/'server-status.tmp';pending.write_text(json.dumps(status));pending.replace(path.parent/'server-status.json')
        return
    job = json.loads(Path(args.job).read_text(encoding='utf-8'))
    run_job(job)


if __name__=='__main__': main()
