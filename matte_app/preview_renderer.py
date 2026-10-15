"""Shared CPU preview rendering for the current alpha's grayscale and cutout views."""
from PIL import Image

class PreviewRenderer:
    def __init__(self):
        self.source_key=None;self.source_original=None;self.source=None;self.background=None;self.opacity=None;self.background_original=None

    def prepare(self,original,size,opacity=0,background_original=None):
        background_original=background_original if background_original is not None else original
        identity=(id(original),size)
        if self.source_original is not original or self.source_key!=identity:
            self.source=original.resize(size,Image.Resampling.BILINEAR).convert('RGBA')
            self.source_key=identity;self.source_original=original;self.opacity=None
        if self.opacity!=opacity or self.background_original is not background_original:
            from .window import checkerboard
            self.background=checkerboard(size).convert('RGBA')
            if opacity:self.background=Image.blend(self.background,background_original.resize(size,Image.Resampling.BILINEAR).convert('RGBA'),opacity/100)
            self.opacity=opacity;self.background_original=background_original

    def compose(self,alpha,box=None):
        foreground=self.source.copy() if box is None else self.source.crop(box)
        foreground.putalpha(alpha)
        result=self.background.copy() if box is None else self.background.crop(box)
        result.alpha_composite(foreground)
        return result

    def render(self,original,alpha,size,cutout=False,opacity=0,background_original=None):
        if alpha is None:return original.resize(size,Image.Resampling.BILINEAR)
        mask=alpha.resize(size,Image.Resampling.BILINEAR)
        if not cutout:return mask.convert('RGBA')
        self.prepare(original,size,opacity,background_original)
        return self.compose(mask)

    def patch(self,original,alpha,size,box,cutout=False,opacity=0,background_original=None):
        width,height=size;sw,sh=alpha.size;left,top,right,bottom=box
        mask=alpha.transform((right-left,bottom-top),Image.Transform.EXTENT,
                             (left*sw/width,top*sh/height,right*sw/width,bottom*sh/height),Image.Resampling.BILINEAR)
        if not cutout:return mask.convert('RGBA')
        self.prepare(original,size,opacity,background_original)
        return self.compose(mask,box)
