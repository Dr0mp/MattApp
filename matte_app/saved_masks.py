"""Read-only saved mask inputs and foreground/background/unknown trimaps."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageFilter

def read_mask(path):
    with Image.open(path) as image:
        return image.getchannel('A') if 'A' in image.getbands() else image.convert('L')

def mask_files(folder, source, start, count, first_only=False):
    folder = Path(folder)
    try:
        metadata = json.loads((folder/'frames.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        metadata = {}
    bound = metadata.get('source_video') or metadata.get('source_image')
    if bound and source and Path(bound).resolve() != Path(source).resolve():
        raise ValueError('These masks belong to a different source clip.')
    paths = [folder/f'frame_{frame:06d}.png' for frame in range(start, start+(1 if first_only else count))]
    missing = next((p for p in paths if not p.is_file()),None)
    if missing:
        raise ValueError(f'Missing input mask for frame {int(missing.stem.split("_")[-1])}. Create or attach masks for the requested range first.')
    return paths

def make_trimap(mask, width=24):
    width = max(1, min(256, int(width)))
    binary = np.asarray(mask.convert('L')) >= 128
    if binary.all() or not binary.any():
        return Image.fromarray(binary.astype(np.uint8)*255)
    try:
        from scipy.ndimage import distance_transform_edt
        foreground = distance_transform_edt(binary) > width
        background = distance_transform_edt(~binary) > width
    except ImportError:
        image = Image.fromarray(binary.astype(np.uint8)*255)
        kernel = 2*width+1
        foreground = np.asarray(image.filter(ImageFilter.MinFilter(kernel))) == 255
        background = np.asarray(image.filter(ImageFilter.MaxFilter(kernel))) == 0
    trimap = np.full((mask.height,mask.width),128,dtype=np.uint8)
    trimap[background] = 0
    trimap[foreground] = 255
    return Image.fromarray(trimap)

def validate_trimap(trimap, size):
    if trimap.size != size:
        raise ValueError('The trimap must match the source frame dimensions.')
    if not set(np.unique(np.asarray(trimap.convert('L')))).issubset({0,128,255}):
        raise ValueError('A trimap must contain only black (0), unknown gray (128), and white (255).')
    return trimap.convert('L')
