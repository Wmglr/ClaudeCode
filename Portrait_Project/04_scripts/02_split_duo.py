import numpy as np, cv2
from PIL import Image
im=np.array(Image.open('00_originals/duo_reference.jpg').convert('RGB'))
a=np.array(Image.open('01_masks/duo_alpha.png')).astype(np.float32)/255
H,W=a.shape
fg=(a>0.5).astype(np.uint8)
markers=np.zeros((H,W),np.int32)
core=cv2.erode(fg,np.ones((15,15),np.uint8))
yy,xx=np.mgrid[:H,:W]
# seeds: woman = left of x=470 (core) or cream-lower-left ; man = right of x=700 core
markers[(core==1)&(xx<430)]=1
markers[(core==1)&(xx>650)]=2
# extra woman seed: hair/blazer up to x 520 for y<480 ; extra man seed for jacket
markers[(core==1)&(xx<520)&(yy<480)]=1
markers[(core==1)&(xx>600)&(yy>520)]=2
markers[fg==0]=3
grad=cv2.cvtColor(im,cv2.COLOR_RGB2LAB)
ws=cv2.watershed(cv2.GaussianBlur(im,(0,0),1.2),markers.copy())
lab=np.where(ws==1,1,np.where(ws==2,2,0)).astype(np.uint8)*fg
np.save('01_masks/duo_label.npy',lab)
vis=im.copy(); vis[lab==1]=(vis[lab==1]*0.5+np.array([255,0,0])*0.5).astype(np.uint8); vis[lab==2]=(vis[lab==2]*0.5+np.array([0,255,0])*0.5).astype(np.uint8)
Image.fromarray(vis).save('03_intermediate/split_vis.jpg')
Image.fromarray(vis).crop((380,380,720,1431)).save('03_intermediate/split_zoom.jpg')
