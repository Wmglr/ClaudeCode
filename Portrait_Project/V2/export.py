import numpy as np, os
from PIL import Image, ImageCms
d=np.load('V2/variants/B_s1064.npz'); u8=lambda x:(np.clip(x,0,1)*255+0.5).astype(np.uint8)
icc=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
rgb=Image.fromarray(u8(d['out']))
rgb.save('V2/Portrait_Final_V2.png',icc_profile=icc,compress_level=9,dpi=(300,300))
rgb.save('V2/Portrait_Final_V2.jpg',quality=96,subsampling=0,optimize=True,icc_profile=icc,dpi=(300,300))
rgba=lambda F,A:Image.fromarray(np.dstack([u8(F),u8(A)]),'RGBA')
L={'01_Fond_bois':Image.fromarray(u8(d['plate'])).convert('RGBA'),'02_Jeune_homme':rgba(d['Fj'],d['aj']),'03_Femme':rgba(d['Fw'],d['aw']),'04_Homme':rgba(d['Fm'],d['am'])}
os.makedirs('V2/layers/png',exist_ok=True)
for k,v in L.items(): v.save(f'V2/layers/png/{k}.png',icc_profile=icc)
for k,a in [('jeune_homme','aj'),('femme','aw'),('homme','am')]: Image.fromarray(u8(d[a])).save(f'V2/masks/mask_{k}.png')
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
p=PSDImage.new('RGB',rgb.size,depth=8)
for k,v in L.items(): p.append(PixelLayer.frompil(v,p,k))
p.save('V2/Portrait_Editable_V2.psd')
# comparison sheet: originals | V1 | V2
v1=Image.open('Portrait_Final.png').convert('RGB'); h=640
ims=[Image.open('00_originals/jeune_homme.jpg'),Image.open('00_originals/duo_reference.jpg'),v1,rgb]
rs=[i.resize((int(i.width*h/i.height),h),Image.LANCZOS) for i in ims]; m=Image.new('RGB',(sum(r.width for r in rs)+30,h),'black'); x=0
for r in rs: m.paste(r,(x,0)); x+=r.width+10
m.save('V2/Comparaison_originaux_V1_V2.png'); print(rgb.size)
