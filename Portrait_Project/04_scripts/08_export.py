import numpy as np, io
from PIL import Image, ImageCms
d=np.load('03_intermediate/placed.npz'); plate=np.load('03_intermediate/plate.npy')/255.0
out=np.load('03_intermediate/final_float.npy'); Fj=np.load('03_intermediate/young_graded.npy')
srgb=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')); icc=srgb.tobytes()
u8=lambda x:(np.clip(x,0,1)*255+0.5).astype(np.uint8)
rgb=Image.fromarray(u8(out))
rgb.save('Portrait_Final.png',icc_profile=icc,compress_level=9,dpi=(300,300))
rgb.save('Portrait_Final.jpg',quality=95,subsampling=0,optimize=True,icc_profile=icc,dpi=(300,300))
def rgba(F,A): return Image.fromarray(np.dstack([u8(F),u8(A)]),'RGBA')
layers={'01_Fond_bois_reconstruit':Image.fromarray(u8(plate)).convert('RGBA'),
        '02_Jeune_homme_etalonne':rgba(Fj,d['aj']),
        '03_Femme':rgba(d['Fw'],d['aw']),
        '04_Homme':rgba(d['Fm'],d['am'])}
import os; os.makedirs('02_layers/png',exist_ok=True)
for k,v in layers.items(): v.save(f'02_layers/png/{k}.png',icc_profile=icc)
Image.fromarray(u8(d['aj'])).save('01_masks/final_mask_jeune_homme.png'); Image.fromarray(u8(d['aw'])).save('01_masks/final_mask_femme.png'); Image.fromarray(u8(d['am'])).save('01_masks/final_mask_homme.png')
try:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
    psd=PSDImage.new('RGB',rgb.size,depth=8)
    for k,v in layers.items():
        psd.append(PixelLayer.frompil(v,psd,k))
    psd.save('Portrait_Editable.psd'); print('PSD ok')
except Exception as e: print('PSD failed:',e)
print('exported',rgb.size)
