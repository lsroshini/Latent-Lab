"""Automatic raster conversion for reconstruction previews, not MRI registration."""
import io
import cv2
import numpy as np
from PIL import Image,ImageOps,UnidentifiedImageError
from scipy.ndimage import binary_fill_holes


def prepare_image_preview(raw,size=192):
    if not raw or len(raw)>32*1024**2:raise ValueError('Choose a PNG or JPEG image smaller than 32 MB.')
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in ('PNG','JPEG'):raise ValueError('Choose a PNG or JPEG image.')
            if source.width*source.height>16_000_000:raise ValueError('Image must contain at most 16 million pixels.')
            image=ImageOps.exif_transpose(source)
            if image.mode in ('I;16','I;16L','I;16B','I','L'):
                gray=np.asarray(image,dtype=np.float32)
            else:
                rgba=image.convert('RGBA')
                background=Image.new('RGBA',rgba.size,(0,0,0,255))
                rgb=np.asarray(Image.alpha_composite(background,rgba).convert('RGB'))
                gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:
        raise ValueError('Could not read this image. Choose a valid PNG or JPEG.') from error
    if not np.isfinite(gray).all() or gray.max()<=0:raise ValueError('Image is empty or has invalid intensities.')
    lo,hi=np.percentile(gray,[1,99])
    if hi<=lo:lo,hi=0.,float(gray.max())
    values=np.clip((gray-lo)/(hi-lo),0,1).astype(np.float32)
    height,width=gray.shape;scale=min(size/height,size/width)
    target=(max(1,min(size,round(width*scale))),max(1,min(size,round(height*scale))))
    resized=cv2.resize(values,target,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    output=np.zeros((size,size),np.float32);left=(size-target[0])//2;top=(size-target[1])//2
    output[top:top+target[1],left:left+target[0]]=resized
    mask=binary_fill_holes(output>0).astype(np.uint8)
    if not mask.any():raise ValueError('No visible image content remains after conversion.')
    return output[None],mask[None],{'original_size':[width,height],'resized_size':list(target),
        'padding_left_top':[left,top],'grayscale':True,'anatomical_registration':False}
