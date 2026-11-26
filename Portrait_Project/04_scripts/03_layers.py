import numpy as np, cv2, json
from PIL import Image
from scipy import ndimage as ndi
from pymatting import estimate_foreground_ml
G=json.load(open('04_scripts/geometry.json'))
D=G['D']; S=G['scale']; W=1099+D; H=1431
img=np.array(Image.open('00_originals/duo_reference.jpg').convert('RGB')).astype(np.float64)/255
a=np.array(Image.open('01_masks/duo_alpha_v2.png')).astype(np.float64)/255
lab=np.load('01_masks/duo_label.npy')
# propagate labels to soft-edge pixels by nearest labelled pixel
idx=ndi.distance_transform_edt(lab==0,return_distances=False,return_indices=True)
lab2=lab[idx[0],idx[1]]
S1=cv2.GaussianBlur((lab2==1).astype(np.float32),(0,0),2.2); S2=cv2.GaussianBlur((lab2==2).astype(np.float32),(0,0),2.2)
ratio=S1/(S1+S2+1e-6)                       # smooth woman-vs-man partition (follows the true cream/navy edge)
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
contact=(cv2.dilate((lab2==1).astype(np.uint8),k)>0)&(cv2.dilate((lab2==2).astype(np.uint8),k)>0)
def layer(L,other,erode_if):
    base=(lab2==L).astype(np.float64)
    r=ratio if L==1 else 1-ratio
    soft=np.clip((r-0.5-0.12)*3.2+0.5,0,1)   # ~1px smooth ramp, pulled ~0.5px inside to avoid the other person's colour
    m=np.where(contact,soft,base)
    return np.clip(a*m,0,1)
aw=layer(1,2,True); am=layer(2,1,True)
Fw=estimate_foreground_ml(img,aw); Fm=estimate_foreground_ml(img,am)
np.save('02_layers/woman_F.npy',Fw.astype(np.float32)); np.save('02_layers/woman_A.npy',aw.astype(np.float32))
np.save('02_layers/man_F.npy',Fm.astype(np.float32)); np.save('02_layers/man_A.npy',am.astype(np.float32))
print('layers ok')
