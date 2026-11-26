import numpy as np, cv2, sys
sys.path.insert(0,'04_scripts')
from gf import guided
from PIL import Image
img=np.array(Image.open('00_originals/duo_reference.jpg').convert('RGB')).astype(np.float32)/255
a=np.array(Image.open('01_masks/duo_alpha.png')).astype(np.float32)/255
# 1) snap the blocky edges onto true image edges
g=guided(img,a,6,2e-4)
# 2) steepen the transition (keeps ~1.5px antialiasing; fine hair keeps its partial alpha)
g=np.clip((g-0.5)*1.7+0.5,0,1)
# keep the original fine-detail alpha where it was already clean (hair): take the max in hair zones (above shoulders)
yy=np.arange(a.shape[0])[:,None]*np.ones((1,a.shape[1]))
hair=np.clip((560-yy)/120,0,1)
out=g*(1-hair)+np.maximum(a*0.6+g*0.4,g)*hair*0+ (g*0.5+a*0.5)*hair
Image.fromarray((np.clip(out,0,1)*255).astype(np.uint8)).save('01_masks/duo_alpha_v2.png')
c=img*out[...,None]+np.array([0.2,0.9,0.3])*(1-out[...,None])
Image.fromarray((c*255).astype(np.uint8)).crop((480,820,600,940)).resize((600,600),Image.NEAREST).save('03_intermediate/q_refined_edge.png')
Image.fromarray((c*255).astype(np.uint8)).crop((130,150,570,560)).save('03_intermediate/q_refined_hair.png')
